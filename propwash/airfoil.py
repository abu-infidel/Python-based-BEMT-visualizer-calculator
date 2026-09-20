"""Section aerodynamics: lift/drag polars over the full +/-180 deg range.

A BEMT solver spends essentially all of its time asking "what are Cl and Cd at
this angle of attack, Reynolds number and Mach number?".  Three things matter:

1.  The answer must exist for *every* angle of attack.  During iteration the
    solver probes angles no real propeller ever sees, and a polar that returns
    garbage outside +/-15 deg will wreck convergence.  Hence Viterna-Corrigan
    extrapolation to the full circle.
2.  It must be cheap, and evaluable inside a CUDA kernel.  Hence
    :meth:`AirfoilPolar.tabulate`, which bakes a polar down to a uniform alpha
    grid that device code can interpolate with two multiplies and an add.
3.  Reynolds and Mach corrections have to be applied *on top* of the table,
    because a propeller blade spans a factor of ~5 in Reynolds number and the
    tip can be transonic while the root is incompressible.

The bundled presets are parametrised models fitted to published section
characteristics -- they are not digitised wind-tunnel data.  Use
:func:`load_polar_file` if you have real XFOIL or AirfoilTools output.
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
# Compressibility and Reynolds corrections
# ---------------------------------------------------------------------------

def prandtl_glauert(cl: ArrayLike, mach: ArrayLike, m_limit: float = 0.92) -> ArrayLike:
    """Subsonic compressibility lift amplification, 1/sqrt(1 - M^2).

    Clipped at ``m_limit`` so the correction stays finite through the transonic
    region, where the linearised theory has no business being anyway.
    """
    m = np.clip(np.abs(mach), 0.0, m_limit)
    return cl / np.sqrt(np.maximum(1.0 - m * m, 1.0e-3))


def critical_mach(cl: ArrayLike, thickness: float, m_crit0: float = 0.78) -> ArrayLike:
    """Korn-style critical Mach estimate falling off with lift and thickness."""
    return m_crit0 - 0.1 * np.abs(cl) - 1.0 * thickness


def drag_divergence(mach: ArrayLike, m_crit: ArrayLike, k: float = 20.0) -> ArrayLike:
    """Wave drag increment beyond the drag-divergence Mach number.

    The classic Lock fourth-power rise: ``dCd = k (M - M_dd)^4``.  Small below
    M_dd, brutal above it -- which is exactly why propeller tips are the part
    that limits RPM.
    """
    dm = np.maximum(np.asarray(mach, dtype=float) - m_crit, 0.0)
    return k * dm ** 4


def reynolds_drag_scale(re: ArrayLike, re_ref: float, exponent: float = 0.2,
                        re_floor: float = 2.0e4) -> ArrayLike:
    """Scale profile drag with Reynolds number.

    Turbulent skin friction goes as Re^-0.2.  Below ``re_floor`` the boundary
    layer is laminar and separation-prone, so drag is inflated further -- the
    reason small propellers are so much less efficient than big ones.
    """
    r = np.maximum(np.asarray(re, dtype=float), 1.0e3)
    scale = (re_ref / r) ** exponent
    low = np.maximum(re_floor / r, 1.0)
    # The two factors compound, and at the near-stalled root of a small
    # propeller Re can fall to a few thousand.  Cap the product: measured
    # low-Re polars show profile drag rising by roughly 3-4x, not 10x.
    return np.clip(scale * (1.0 + 0.45 * (low - 1.0)), 0.5, 4.0)


def reynolds_clmax_scale(re: ArrayLike, re_ref: float) -> ArrayLike:
    """Maximum lift degradation at low Reynolds number."""
    r = np.maximum(np.asarray(re, dtype=float), 1.0e3)
    return np.clip(1.0 - 0.14 * np.log10(re_ref / r), 0.45, 1.12)


def du_selig_stall_delay(cl_2d: ArrayLike, cd_2d: ArrayLike, alpha: ArrayLike,
                         cl_alpha: float, cl0: float, cd0: float,
                         chord_over_r: ArrayLike, tsr_local: ArrayLike,
                         a_coef: float = 1.0, b_coef: float = 1.0,
                         d_coef: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Du-Selig / Eggers rotational augmentation ("centrifugal pumping").

    Rotation throws separated boundary-layer fluid outboard, which delays stall
    near the root and can lift inboard Cl well above its 2-D value.  Ignoring it
    under-predicts static thrust noticeably.
    """
    c_r = np.clip(np.asarray(chord_over_r, dtype=float), 1e-4, 2.0)
    lam = np.sqrt(1.0 + np.asarray(tsr_local, dtype=float) ** 2)
    expo = d_coef / (lam * c_r)
    f_l = (1.0 / TWO_PI) * (
        (1.6 * c_r / 0.1267) * (a_coef - c_r ** expo) / (b_coef + c_r ** expo) - 1.0
    )
    f_d = (1.0 / TWO_PI) * (
        (1.6 * c_r / 0.1267) * (a_coef - c_r ** (expo / 2.0)) / (b_coef + c_r ** (expo / 2.0)) - 1.0
    )
    f_l = np.clip(f_l, 0.0, 1.0)
    f_d = np.clip(f_d, 0.0, 1.0)

    cl_pot = cl0 + cl_alpha * np.asarray(alpha, dtype=float)
    cl_rot = cl_2d + f_l * (cl_pot - cl_2d)
    # Eggers: the same mechanism that raises lift also recovers some of the
    # separation drag, referenced to the attached-flow value ``cd0``.
    cd_rot = cd_2d - f_d * (np.asarray(cd_2d, dtype=float) - cd0)
    return cl_rot, np.maximum(cd_rot, 1e-4)


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
    thickness: float = 0.12          # t/c, used for the Mach corrections
    re_ref: float = 5.0e5            # Reynolds number the base polar describes
    aspect_ratio: float = 12.0       # for the Viterna Cd_max estimate
    m_crit0: float = 0.78

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
        cl = np.asarray(cl, dtype=float)
        cd = np.asarray(cd, dtype=float)

        if reynolds is not None:
            cd = cd * reynolds_drag_scale(reynolds, self.re_ref)
            cl = cl * reynolds_clmax_scale(reynolds, self.re_ref)

        if mach is not None:
            cl = prandtl_glauert(cl, mach)
            m_dd = critical_mach(cl, self.thickness, self.m_crit0)
            cd = cd + drag_divergence(mach, m_dd)

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
            m_crit0=self.m_crit0,
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
    parabolic drag-polar coefficient.
    """

    alpha_0: float = 0.0             # rad, zero-lift angle
    cl_alpha: float = 2.0 * math.pi * 0.95
    cl_max: float = 1.35
    cl_min: float = -1.05
    cd_min: float = 0.0085
    cd_k: float = 0.012              # parabolic drag polar: cd = cd_min + k (cl - cl_cdmin)^2
    cl_cdmin: float = 0.15
    stall_width: float = 4.0 * DEG   # blend width around stall

    def max_lift(self) -> float:
        return float(max(abs(self.cl_max), abs(self.cl_min)))

    def base_polar(self, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(alpha, dtype=float)
        a_wrapped = wrap_angle(a)

        # --- attached flow -------------------------------------------------
        cl_lin = self.cl_alpha * (a_wrapped - self.alpha_0)
        cd_lin = self.cd_min + self.cd_k * (cl_lin - self.cl_cdmin) ** 2

        a_stall_p = self.alpha_0 + self.cl_max / self.cl_alpha
        a_stall_n = self.alpha_0 + self.cl_min / self.cl_alpha

        cd_stall_p = self.cd_min + self.cd_k * (self.cl_max - self.cl_cdmin) ** 2 + 0.02
        cd_stall_n = self.cd_min + self.cd_k * (self.cl_min - self.cl_cdmin) ** 2 + 0.02

        # --- deep stall, both signs ----------------------------------------
        cl_p, cd_p = viterna_extrapolate(a_stall_p, self.cl_max, cd_stall_p,
                                         self.aspect_ratio, a_wrapped)
        cl_n, cd_n = viterna_extrapolate(abs(a_stall_n), abs(self.cl_min), cd_stall_n,
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
            cl_m, cd_m = viterna_extrapolate(a_stall_p, self.cl_max, cd_stall_p,
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
    m_crit0: float = 0.78
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
# Preset library
# ---------------------------------------------------------------------------

def _preset(name: str, **kw) -> AnalyticPolar:
    return AnalyticPolar(name=name, **kw)


AIRFOIL_LIBRARY: dict[str, AnalyticPolar] = {
    "naca0012": _preset(
        "NACA 0012", thickness=0.12, alpha_0=0.0, cl_alpha=6.05, cl_max=1.35,
        cl_min=-1.35, cd_min=0.0080, cd_k=0.0115, cl_cdmin=0.0, re_ref=5e5),
    "naca2412": _preset(
        "NACA 2412", thickness=0.12, alpha_0=-2.1 * DEG, cl_alpha=6.10, cl_max=1.52,
        cl_min=-1.05, cd_min=0.0075, cd_k=0.0100, cl_cdmin=0.25, re_ref=5e5),
    "naca4412": _preset(
        "NACA 4412", thickness=0.12, alpha_0=-4.2 * DEG, cl_alpha=6.15, cl_max=1.62,
        cl_min=-0.90, cd_min=0.0080, cd_k=0.0095, cl_cdmin=0.45, re_ref=5e5),
    "clarky": _preset(
        "Clark Y", thickness=0.117, alpha_0=-3.6 * DEG, cl_alpha=6.05, cl_max=1.47,
        cl_min=-0.95, cd_min=0.0085, cd_k=0.0105, cl_cdmin=0.38, re_ref=5e5),
    "e63": _preset(
        "Eppler 63", thickness=0.058, alpha_0=-5.8 * DEG, cl_alpha=6.20, cl_max=1.42,
        cl_min=-0.62, cd_min=0.0150, cd_k=0.0160, cl_cdmin=0.70, re_ref=1e5,
        m_crit0=0.82),
    "mh117": _preset(
        "MH 117", thickness=0.093, alpha_0=-3.0 * DEG, cl_alpha=6.10, cl_max=1.38,
        cl_min=-0.85, cd_min=0.0095, cd_k=0.0110, cl_cdmin=0.35, re_ref=2e5),
    "arad20": _preset(
        "ARA-D 20%", thickness=0.20, alpha_0=-2.4 * DEG, cl_alpha=5.90, cl_max=1.30,
        cl_min=-0.85, cd_min=0.0120, cd_k=0.0140, cl_cdmin=0.30, re_ref=1e6,
        m_crit0=0.70),
    "naca16509": _preset(
        "NACA 16-509", thickness=0.09, alpha_0=-3.2 * DEG, cl_alpha=6.00, cl_max=1.10,
        cl_min=-0.80, cd_min=0.0055, cd_k=0.0130, cl_cdmin=0.50, re_ref=1e6,
        m_crit0=0.86),
    "flatplate": _preset(
        "Flat plate", thickness=0.02, alpha_0=0.0, cl_alpha=6.28, cl_max=0.85,
        cl_min=-0.85, cd_min=0.0150, cd_k=0.0400, cl_cdmin=0.0, re_ref=1e5),
}

DEFAULT_AIRFOIL = "clarky"


def get_airfoil(name: str) -> AnalyticPolar:
    """Look up a preset by a forgiving key ('NACA 4412' -> 'naca4412')."""
    key = _re.sub(r"[^a-z0-9]", "", name.lower())
    if key not in AIRFOIL_LIBRARY:
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
        re_ref=mix(root.re_ref, tip.re_ref),
        aspect_ratio=mix(root.aspect_ratio, tip.aspect_ratio),
        m_crit0=mix(root.m_crit0, tip.m_crit0),
        alpha_0=mix(root.alpha_0, tip.alpha_0),
        cl_alpha=mix(root.cl_alpha, tip.cl_alpha),
        cl_max=mix(root.cl_max, tip.cl_max),
        cl_min=mix(root.cl_min, tip.cl_min),
        cd_min=mix(root.cd_min, tip.cd_min),
        cd_k=mix(root.cd_k, tip.cd_k),
        cl_cdmin=mix(root.cl_cdmin, tip.cl_cdmin),
        stall_width=mix(root.stall_width, tip.stall_width),
    )


def load_polar_file(path: str | Path, name: str | None = None, **kw) -> TablePolar:
    """Read an XFOIL ``.pol`` / AirfoilTools CSV with alpha, Cl, Cd columns."""
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
                      cl_data=arr[:, 1], cd_data=arr[:, 2], **kw)


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
    "AIRFOIL_LIBRARY", "DEFAULT_AIRFOIL", "get_airfoil", "list_airfoils",
    "blended_polar", "load_polar_file", "stack_tables", "viterna_extrapolate",
    "prandtl_glauert", "critical_mach", "drag_divergence", "reynolds_drag_scale",
    "reynolds_clmax_scale", "du_selig_stall_delay", "wrap_angle", "DEG", "GAMMA",
]
