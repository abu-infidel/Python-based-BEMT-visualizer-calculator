"""Section aerodynamics for full-size propellers: polars over the full +/-180 deg.

A BEMT solver spends essentially all of its time asking "what are Cl and Cd at
this angle of attack, Reynolds number and Mach number?".  Three things matter:

1.  The answer must exist for *every* angle of attack.  During iteration the
    solver probes angles no real propeller ever sees, and a polar that returns
    garbage outside +/-15 deg will wreck convergence.  Hence Viterna-Corrigan
    extrapolation to the full circle.
2.  It must be cheap, and evaluable inside a GPU kernel.  Hence
    :meth:`AirfoilPolar.tabulate`, which bakes a polar down to a uniform alpha
    grid that device code can interpolate with two multiplies and an add.
3.  Reynolds, thickness and Mach effects have to be right *for full-size
    propellers*: blade Reynolds numbers of 1-10 million, sections from 30%
    thick at the shank to 5% at the tip, and tips running at Mach 0.7-0.9.

The library holds sections actually used on full-size propellers.  Each is a
parametrised model whose numbers are fitted to published characteristics at a
full-scale Reynolds number of 3 million -- they are not digitised wind-tunnel
data.  Use :func:`load_polar_file` if you have real XFOIL or test output.

Two thickness effects are applied per blade station, relative to the
thickness the section's parameters were fitted at:

* profile drag scales with Hoerner's form factor ``1 + 2 t + 60 t^4``, so the
  thick shank makes more drag than the thin tip;
* maximum lift falls steeply below about 12% thickness (thin sections stall at
  the leading edge) and gently above 15% -- the shape of the classic maximum-
  lift-versus-thickness curves for NACA sections.
"""

from __future__ import annotations

import math
import re as _re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .atmosphere import GAMMA

ArrayLike = np.ndarray | float

DEG = math.pi / 180.0
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Full-size corrections (host-side twins of the kernel code in bemt/core.py)
# ---------------------------------------------------------------------------

#: Reference Reynolds number every library section is fitted at.
FULL_SCALE_RE = 3.0e6

#: Lock's offset between critical and drag-divergence Mach, (0.1/80)^(1/3).
LOCK_OFFSET = 0.1077


def prandtl_glauert(cl: ArrayLike, mach: ArrayLike, m_limit: float = 0.92) -> ArrayLike:
    """Subsonic compressibility lift amplification, 1/sqrt(1 - M^2)."""
    m = np.clip(np.abs(mach), 0.0, m_limit)
    return cl / np.sqrt(np.maximum(1.0 - m * m, 1.0e-3))


def critical_mach(cl: ArrayLike, thickness: ArrayLike, kappa: float) -> ArrayLike:
    """Korn critical Mach: drag divergence ``kappa - t/c - Cl/10`` less Lock's offset."""
    return kappa - 0.1 * np.abs(cl) - np.asarray(thickness, dtype=float) - LOCK_OFFSET


def drag_divergence(mach: ArrayLike, m_crit: ArrayLike, k: float = 20.0) -> ArrayLike:
    """Lock's fourth-power wave drag above the critical Mach number."""
    dm = np.maximum(np.minimum(np.asarray(mach, dtype=float), 1.2) - m_crit, 0.0)
    return k * dm ** 4


def reynolds_drag_scale(re: ArrayLike, re_ref: float = FULL_SCALE_RE) -> ArrayLike:
    """Turbulent skin-friction scaling Re^-0.2, bounded to the full-size range."""
    r = np.maximum(np.asarray(re, dtype=float), 1.0e4)
    return np.clip((re_ref / r) ** 0.2, 0.75, 1.6)


def reynolds_lift_ceiling(re: ArrayLike, re_ref: float = FULL_SCALE_RE) -> ArrayLike:
    """Maximum-lift scaling with Reynolds number: a few percent per decade."""
    r = np.maximum(np.asarray(re, dtype=float), 1.0e4)
    return np.clip(1.0 + 0.06 * np.log10(r / re_ref), 0.85, 1.05)


def thickness_drag_factor(t: ArrayLike) -> ArrayLike:
    """Hoerner's profile-drag form factor ``1 + 2 t + 60 t^4``."""
    tt = np.clip(np.asarray(t, dtype=float), 0.0, 0.5)
    return 1.0 + 2.0 * tt + 60.0 * tt ** 4


def thickness_lift_factor(t: ArrayLike) -> ArrayLike:
    """Relative maximum lift against thickness ratio.

    Flat between 12% and 15%; falls as ``(t/0.12)^0.85`` below (a 6% section
    reaches roughly 55% of the maximum lift of a 12% one) and by 1.5 per unit
    thickness above.  A fit to the trend in NACA section data, not a law.
    """
    tt = np.clip(np.asarray(t, dtype=float), 0.02, 0.45)
    thin = (tt / 0.12) ** 0.85
    thick = np.maximum(1.0 - 1.5 * (tt - 0.15), 0.5)
    return np.where(tt < 0.12, thin, np.where(tt <= 0.15, 1.0, thick))


# ---------------------------------------------------------------------------
# Viterna-Corrigan full-circle extrapolation
# ---------------------------------------------------------------------------

def viterna_extrapolate(alpha_stall: float, cl_stall: float, cd_stall: float,
                        aspect_ratio: float, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Viterna-Corrigan deep-stall model on ``alpha`` (radians, any range).

    Valid from the stall angle out to 90 deg; the caller mirrors it into the
    remaining quadrants.
    """
    a_s = max(abs(alpha_stall), 3.0 * DEG)
    sa, ca = math.sin(a_s), math.cos(a_s)
    ca = max(ca, 1e-3)

    cd_max = 1.11 + 0.018 * aspect_ratio if aspect_ratio <= 50.0 else 2.01
    b1 = cd_max
    a1 = b1 / 2.0
    b2 = (cd_stall - cd_max * sa * sa) / ca
    a2 = (cl_stall - cd_max * sa * ca) * sa / (ca * ca)

    a = np.asarray(alpha, dtype=float)
    sin_a = np.sin(a)
    cos_a = np.cos(a)

    # The a2 cos^2/sin term is singular at alpha=0, where the model has no
    # meaning anyway (that is the attached-flow region).  Floor |sin| at the
    # stall angle's sine so the branch stays bounded and monotone.
    floor = max(abs(sa), 0.05)
    safe_sin = np.where(np.abs(sin_a) < floor, np.copysign(floor, np.where(sin_a >= 0.0, 1.0, -1.0)), sin_a)

    cl = a1 * np.sin(2.0 * a) + a2 * cos_a * cos_a / safe_sin
    cd = b1 * sin_a * sin_a + b2 * cos_a
    cl = np.clip(cl, -1.2 * cd_max, 1.2 * cd_max)
    cd = np.clip(cd, 1e-4, 1.3 * cd_max)
    return cl, cd


# ---------------------------------------------------------------------------
# Polar models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AirfoilPolar:
    """Base class: a section's Cl/Cd behaviour over the full circle."""

    name: str = "generic"
    thickness: float = 0.12          # t/c at this station
    re_ref: float = FULL_SCALE_RE    # Reynolds number the base polar describes
    aspect_ratio: float = 8.0        # blade aspect ratio, for Viterna Cd_max
    kappa: float = 0.87              # Korn technology factor

    def base_polar(self, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Incompressible Cl, Cd at ``re_ref`` for alpha in radians."""
        raise NotImplementedError

    # -- public API ---------------------------------------------------------
    def evaluate(self, alpha: ArrayLike, reynolds: ArrayLike | None = None,
                 mach: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Cl and Cd including Reynolds and compressibility corrections."""
        a = np.asarray(alpha, dtype=float)
        cl, cd = self.base_polar(a)
        return self.apply_corrections(cl, cd, reynolds, mach)

    def apply_corrections(self, cl: np.ndarray, cd: np.ndarray,
                          reynolds: ArrayLike | None,
                          mach: ArrayLike | None) -> tuple[np.ndarray, np.ndarray]:
        """Host-side version of the solver's corrections, for plotting a polar."""
        cl = np.asarray(cl, dtype=float)
        cd = np.asarray(cd, dtype=float)
        ceiling = np.full_like(cl, self.max_lift())

        if reynolds is not None:
            cd = cd * reynolds_drag_scale(reynolds, self.re_ref)
            ceiling = ceiling * reynolds_lift_ceiling(reynolds, self.re_ref)

        if mach is not None:
            m = np.maximum(np.asarray(mach, dtype=float), 0.0)
            cl = prandtl_glauert(cl, m)
            excess = np.maximum(np.minimum(m, 0.92) - 0.35, 0.0)
            ceiling = ceiling * np.maximum(1.0 - 0.9 * excess ** 1.5, 0.35)
            cl = np.clip(cl, -ceiling, ceiling)
            cd = cd + drag_divergence(m, critical_mach(cl, self.thickness, self.kappa))
        else:
            cl = np.clip(cl, -ceiling, ceiling)

        return cl, np.maximum(cd, 1e-5)

    def tabulate(self, n_alpha: int = 721) -> "PolarTable":
        """Bake this polar onto a uniform -180..180 deg grid for fast lookup."""
        alpha = np.linspace(-math.pi, math.pi, n_alpha)
        cl, cd = self.base_polar(alpha)
        return PolarTable(
            alpha=alpha,
            cl=np.ascontiguousarray(cl, dtype=np.float64),
            cd=np.ascontiguousarray(cd, dtype=np.float64),
            thickness=self.thickness,
            re_ref=self.re_ref,
            kappa=self.kappa,
            name=self.name,
        )

    def max_lift(self) -> float:
        """Maximum |Cl| in the attached-flow range.

        Needed by the compressibility correction: Prandtl-Glauert amplifies the
        lift *slope*, but maximum lift falls with Mach rather than rising, so
        the amplified value has to be bounded by something.
        """
        a = np.linspace(-25.0 * DEG, 25.0 * DEG, 401)
        cl, _ = self.base_polar(a)
        return float(np.max(np.abs(cl)))

    def summary(self) -> dict[str, float]:
        """Headline numbers: alpha_0, Cl_alpha, Cl_max, Cd_min, best L/D."""
        a = np.linspace(-20.0 * DEG, 25.0 * DEG, 901)
        cl, cd = self.base_polar(a)
        lin = (a > -4.0 * DEG) & (a < 4.0 * DEG)
        slope = float(np.polyfit(a[lin], cl[lin], 1)[0])
        zero_lift = float(np.interp(0.0, cl[lin], a[lin]))
        ld = cl / np.maximum(cd, 1e-9)
        i_best = int(np.argmax(ld))
        return {
            "alpha_0_deg": zero_lift / DEG,
            "cl_alpha_per_rad": slope,
            "cl_max": float(cl.max()),
            "alpha_cl_max_deg": float(a[int(np.argmax(cl))] / DEG),
            "cd_min": float(cd.min()),
            "ld_max": float(ld[i_best]),
            "alpha_ld_max_deg": float(a[i_best] / DEG),
        }


@dataclass(slots=True)
class AnalyticPolar(AirfoilPolar):
    """Thin-airfoil linear range blended into a Viterna deep-stall model.

    The pre-stall parameters are the ones people actually quote for a section:
    zero-lift angle, lift-curve slope, maximum lift, minimum drag and the
    parabolic drag-polar coefficient -- all at ``nominal_thickness``.  The
    station's own ``thickness`` then scales drag and maximum lift.
    """

    alpha_0: float = 0.0             # rad, zero-lift angle
    cl_alpha: float = 2.0 * math.pi * 0.97
    cl_max: float = 1.5
    cl_min: float = -1.2
    cd_min: float = 0.0070
    cd_k: float = 0.010              # parabolic drag polar: cd = cd_min + k (cl - cl_cdmin)^2
    cl_cdmin: float = 0.3
    stall_width: float = 4.0 * DEG   # blend width around stall
    nominal_thickness: float = 0.12  # t/c the parameters above describe

    # -- thickness-adjusted parameters --------------------------------------
    def _lift_scale(self) -> float:
        return float(thickness_lift_factor(self.thickness)
                     / thickness_lift_factor(self.nominal_thickness))

    def effective_cl_max(self) -> float:
        return self.cl_max * self._lift_scale()

    def effective_cl_min(self) -> float:
        return self.cl_min * self._lift_scale()

    def effective_cd_min(self) -> float:
        return float(self.cd_min * thickness_drag_factor(self.thickness)
                     / thickness_drag_factor(self.nominal_thickness))

    def max_lift(self) -> float:
        return float(max(abs(self.effective_cl_max()), abs(self.effective_cl_min())))

    def base_polar(self, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(alpha, dtype=float)
        a_wrapped = wrap_angle(a)

        cl_max = self.effective_cl_max()
        cl_min = self.effective_cl_min()
        cd_min = self.effective_cd_min()

        # --- attached flow -------------------------------------------------
        cl_lin = self.cl_alpha * (a_wrapped - self.alpha_0)
        cd_lin = cd_min + self.cd_k * (cl_lin - self.cl_cdmin) ** 2

        a_stall_p = self.alpha_0 + cl_max / self.cl_alpha
        a_stall_n = self.alpha_0 + cl_min / self.cl_alpha

        cd_stall_p = cd_min + self.cd_k * (cl_max - self.cl_cdmin) ** 2 + 0.02
        cd_stall_n = cd_min + self.cd_k * (cl_min - self.cl_cdmin) ** 2 + 0.02

        # --- deep stall, both signs ----------------------------------------
        cl_p, cd_p = viterna_extrapolate(a_stall_p, cl_max, cd_stall_p,
                                         self.aspect_ratio, a_wrapped)
        cl_n, cd_n = viterna_extrapolate(abs(a_stall_n), abs(cl_min), cd_stall_n,
                                         self.aspect_ratio, -a_wrapped)
        cl_n, cd_n = -cl_n, cd_n

        cl_post = np.where(a_wrapped >= 0.0, cl_p, cl_n)
        cd_post = np.where(a_wrapped >= 0.0, cd_p, cd_n)

        # --- smooth blend ---------------------------------------------------
        w = _blend_weight(a_wrapped, a_stall_n, a_stall_p, self.stall_width)
        cl = w * cl_lin + (1.0 - w) * cl_post
        cd = w * cd_lin + (1.0 - w) * cd_post

        # Beyond +/-90 deg the section is a bluff body moving backwards; the
        # reversed-camber lift is weaker, and everything must return to zero
        # lift at +/-180 deg so the polar is continuous around the circle.
        deep = np.abs(a_wrapped) > math.pi / 2.0
        if np.any(deep):
            mirror = np.sign(a_wrapped) * math.pi - a_wrapped
            cl_m, cd_m = viterna_extrapolate(a_stall_p, cl_max, cd_stall_p,
                                             self.aspect_ratio, np.abs(mirror))
            taper = np.clip(np.abs(mirror) / max(a_stall_p, 1e-3), 0.0, 1.0)
            cl = np.where(deep, -0.7 * np.sign(a_wrapped) * cl_m * taper, cl)
            cd = np.where(deep, cd_m, cd)

        return cl, np.maximum(cd, 1e-4)


@dataclass(slots=True)
class TablePolar(AirfoilPolar):
    """Polar backed by measured or XFOIL-computed data, extended with Viterna."""

    alpha_data: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cl_data: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cd_data: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _full_alpha: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    _full_cl: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    _full_cd: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def __post_init__(self) -> None:
        self._build_full_circle()

    def _build_full_circle(self) -> None:
        if self.alpha_data.size < 3:
            raise ValueError("TablePolar needs at least three data points")
        order = np.argsort(self.alpha_data)
        a = np.asarray(self.alpha_data, dtype=float)[order]
        cl = np.asarray(self.cl_data, dtype=float)[order]
        cd = np.asarray(self.cd_data, dtype=float)[order]

        grid = np.linspace(-math.pi, math.pi, 721)
        i_pos = int(np.argmax(cl))
        i_neg = int(np.argmin(cl))
        cl_p, cd_p = viterna_extrapolate(a[i_pos], cl[i_pos], cd[i_pos],
                                         self.aspect_ratio, grid)
        cl_n, cd_n = viterna_extrapolate(abs(a[i_neg]), abs(cl[i_neg]), cd[i_neg],
                                         self.aspect_ratio, -grid)
        ext_cl = np.where(grid >= 0.0, cl_p, -cl_n)
        ext_cd = np.where(grid >= 0.0, cd_p, cd_n)

        inside = (grid >= a[0]) & (grid <= a[-1])
        full_cl = np.where(inside, np.interp(grid, a, cl), ext_cl)
        full_cd = np.where(inside, np.interp(grid, a, cd), ext_cd)

        # Taper to zero lift at +/-180 deg for continuity around the circle.
        edge = np.abs(grid) > 170.0 * DEG
        full_cl = np.where(edge, full_cl * (math.pi - np.abs(grid)) / (10.0 * DEG), full_cl)

        self._full_alpha = grid
        self._full_cl = full_cl
        self._full_cd = np.maximum(full_cd, 1e-4)

    def base_polar(self, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = wrap_angle(np.asarray(alpha, dtype=float))
        return (np.interp(a, self._full_alpha, self._full_cl),
                np.interp(a, self._full_alpha, self._full_cd))


@dataclass(slots=True)
class PolarTable:
    """A polar frozen onto a uniform alpha grid -- the GPU-friendly form.

    Lookup is ``i = (alpha + pi) / d_alpha`` plus a linear blend, which is four
    instructions in a CUDA kernel and needs no branching.
    """

    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    thickness: float = 0.12
    re_ref: float = 5.0e5
    kappa: float = 0.78
    name: str = "tabulated"

    @property
    def d_alpha(self) -> float:
        return float(self.alpha[1] - self.alpha[0])

    @property
    def n(self) -> int:
        return int(self.alpha.size)

    def lookup(self, alpha: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised host-side equivalent of the device interpolation."""
        a = wrap_angle(np.asarray(alpha, dtype=float))
        x = (a + math.pi) / self.d_alpha
        i0 = np.clip(np.floor(x).astype(np.int64), 0, self.n - 2)
        t = x - i0
        cl = self.cl[i0] * (1.0 - t) + self.cl[i0 + 1] * t
        cd = self.cd[i0] * (1.0 - t) + self.cd[i0 + 1] * t
        return cl, cd


def wrap_angle(alpha: np.ndarray) -> np.ndarray:
    """Wrap angles into (-pi, pi]."""
    return (np.asarray(alpha, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite smoothstep, exactly 0 at t<=0 and exactly 1 at t>=1."""
    x = np.clip(t, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _blend_weight(alpha: np.ndarray, a_neg: float, a_pos: float, width: float) -> np.ndarray:
    """Exactly 1 inside the linear range, exactly 0 past stall + ``width``.

    A tanh blend is smoother but never reaches 1, which lets a sliver of the
    deep-stall model leak into the attached-flow region.  Finite support keeps
    the linear range pristine.
    """
    w = max(width, 1e-4)
    up = 1.0 - _smoothstep((alpha - a_pos) / w)
    dn = _smoothstep((alpha - a_neg) / w + 1.0)
    return np.clip(up * dn, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Full-size propeller section library
# ---------------------------------------------------------------------------

def _section(name: str, t_nom: float, alpha0_deg: float, cl_alpha: float,
             cl_max: float, cl_min: float, cd_min: float, cd_k: float,
             cl_cdmin: float, kappa: float) -> AnalyticPolar:
    return AnalyticPolar(name=name, thickness=t_nom, nominal_thickness=t_nom,
                         re_ref=FULL_SCALE_RE, alpha_0=alpha0_deg * DEG,
                         cl_alpha=cl_alpha, cl_max=cl_max, cl_min=cl_min,
                         cd_min=cd_min, cd_k=cd_k, cl_cdmin=cl_cdmin, kappa=kappa)


#: Sections used on full-size propellers, fitted at Re = 3 million.
#: ``kappa`` is Korn's technology factor: higher means a later drag rise.
AIRFOIL_LIBRARY: dict[str, AnalyticPolar] = {
    # The classic flat-bottomed general-aviation propeller section (NACA
    # Report 640's full-scale propellers used it).  Propeller blade angles and
    # pitch are measured off the flat face, which sits about 2 deg nose-down
    # of the leading-to-trailing-edge chord, so the zero-lift angle here is
    # referenced to the face: -4.2 deg, inside the published range (-3.6 deg
    # chord-referenced to about -5.5 deg face-referenced) and the value that
    # matches the Cessna 172N cruise data in docs/VALIDATION.md.
    "clark_y": _section("Clark Y", 0.117, -4.2, 6.10, 1.55, -1.05,
                        0.0080, 0.0095, 0.40, 0.84),
    # The British counterpart, also tested full-scale in NACA Report 640.
    "raf_6": _section("R.A.F. 6", 0.100, -3.5, 6.00, 1.45, -0.95,
                      0.0085, 0.0100, 0.35, 0.83),
    # NACA 16-series: designed for high critical Mach -- the usual choice for
    # propeller tips that run near Mach 0.8.
    "naca_16": _section("NACA 16-series", 0.090, -2.6, 6.20, 1.20, -0.95,
                        0.0058, 0.0110, 0.40, 0.90),
    # NACA 64-series laminar-flow section, common on constant-speed blades.
    "naca_64": _section("NACA 64-series", 0.100, -2.8, 6.25, 1.45, -1.00,
                        0.0055, 0.0105, 0.40, 0.88),
    # ARA-D: the Aircraft Research Association's propeller family, used on
    # modern composite blades.
    "ara_d": _section("ARA-D", 0.100, -3.2, 6.20, 1.50, -0.95,
                      0.0065, 0.0095, 0.45, 0.90),
    # Symmetric reference section, mainly for checking the model.
    "naca_0012": _section("NACA 0012", 0.120, 0.0, 6.30, 1.55, -1.55,
                          0.0062, 0.0090, 0.0, 0.87),
}


def _key(name: str) -> str:
    return _re.sub(r"[^a-z0-9]", "", name.lower())


_LOOKUP = {_key(k): k for k in AIRFOIL_LIBRARY}
_LOOKUP.update({_key(v.name): k for k, v in AIRFOIL_LIBRARY.items()})


def get_airfoil(name: str) -> AnalyticPolar:
    """Look up a section by a forgiving key ('Clark Y', 'clark_y', 'CLARKY')."""
    key = _LOOKUP.get(_key(name))
    if key is None:
        raise KeyError(f"unknown airfoil {name!r}; have {sorted(AIRFOIL_LIBRARY)}")
    return replace(AIRFOIL_LIBRARY[key])


def list_airfoils() -> list[str]:
    return sorted(AIRFOIL_LIBRARY)


def blended_polar(root: AirfoilPolar, tip: AirfoilPolar, fraction: float) -> AnalyticPolar:
    """Linear blend between two analytic sections (root -> tip transition)."""
    if not isinstance(root, AnalyticPolar) or not isinstance(tip, AnalyticPolar):
        raise TypeError("blended_polar only supports AnalyticPolar inputs")
    f = float(np.clip(fraction, 0.0, 1.0))
    mix = lambda a, b: (1.0 - f) * a + f * b  # noqa: E731
    return AnalyticPolar(
        name=f"{root.name}->{tip.name} @{f:.2f}",
        thickness=mix(root.thickness, tip.thickness),
        nominal_thickness=mix(root.nominal_thickness, tip.nominal_thickness),
        re_ref=mix(root.re_ref, tip.re_ref),
        aspect_ratio=mix(root.aspect_ratio, tip.aspect_ratio),
        kappa=mix(root.kappa, tip.kappa),
        alpha_0=mix(root.alpha_0, tip.alpha_0),
        cl_alpha=mix(root.cl_alpha, tip.cl_alpha),
        cl_max=mix(root.cl_max, tip.cl_max),
        cl_min=mix(root.cl_min, tip.cl_min),
        cd_min=mix(root.cd_min, tip.cd_min),
        cd_k=mix(root.cd_k, tip.cd_k),
        cl_cdmin=mix(root.cl_cdmin, tip.cl_cdmin),
        stall_width=mix(root.stall_width, tip.stall_width),
    )


def load_polar_file(path: str | Path, *, reynolds: float, thickness: float,
                    name: str | None = None, **kw) -> TablePolar:
    """Read an XFOIL ``.pol`` / AirfoilTools CSV with alpha, Cl, Cd columns.

    ``reynolds`` and ``thickness`` are required: the solver's Reynolds and
    compressibility corrections are applied *relative to* the conditions the
    data describe, so guessing them would silently bias every result.
    """
    p = Path(path)
    rows: list[tuple[float, float, float]] = []
    for line in p.read_text(errors="ignore").splitlines():
        parts = _re.split(r"[,\s]+", line.strip())
        if len(parts) < 3:
            continue
        try:
            a, cl, cd = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if abs(a) <= 180.0 and abs(cl) < 10.0 and 0.0 <= cd < 10.0:
            rows.append((a * DEG, cl, cd))
    if len(rows) < 3:
        raise ValueError(f"no usable polar rows found in {p}")
    arr = np.array(sorted(rows))
    return TablePolar(name=name or p.stem, alpha_data=arr[:, 0],
                      cl_data=arr[:, 1], cd_data=arr[:, 2], re_ref=float(reynolds),
                      thickness=float(thickness), **kw)


def stack_tables(polars: Sequence[AirfoilPolar], n_alpha: int = 721) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bake one polar per blade station into ``(alpha, cl[n, na], cd[n, na])``.

    This is the exact layout the CUDA kernel expects: one row per radial
    station, a shared alpha axis, contiguous in C order.
    """
    tables = [p.tabulate(n_alpha) for p in polars]
    alpha = tables[0].alpha
    cl = np.ascontiguousarray(np.stack([t.cl for t in tables]))
    cd = np.ascontiguousarray(np.stack([t.cd for t in tables]))
    return alpha, cl, cd


__all__ = [
    "AirfoilPolar", "AnalyticPolar", "TablePolar", "PolarTable",
    "AIRFOIL_LIBRARY", "FULL_SCALE_RE", "LOCK_OFFSET", "get_airfoil", "list_airfoils",
    "blended_polar", "load_polar_file", "stack_tables", "viterna_extrapolate",
    "prandtl_glauert", "critical_mach", "drag_divergence", "reynolds_drag_scale",
    "reynolds_lift_ceiling", "thickness_drag_factor", "thickness_lift_factor",
    "wrap_angle", "DEG", "GAMMA",
]
