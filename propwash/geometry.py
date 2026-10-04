"""Full-size propeller blade geometry: chord, twist, thickness and sections.

A propeller blade is described here as a set of *distributions* over the
non-dimensional radius ``x = r / R``.  Each distribution is a small serialisable
spec (kind + parameters) rather than a raw array, so a blade round-trips
through JSON and the 3-D mesher and the solver read the exact same geometry.
There are no preset blades: :func:`full_size_blade` builds one from the
numbers a propeller is specified by.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .airfoil import (DEG, AnalyticPolar, blended_polar, get_airfoil,
                      list_airfoils, stack_tables)
from .units import INCH

DistKind = Literal["constant", "linear", "elliptic", "inverse", "betz", "spline", "parabolic"]


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Distribution:
    """A named one-parameter-family curve over x = r/R in [x_hub, 1]."""

    kind: DistKind = "constant"
    root: float = 0.10
    tip: float = 0.10
    power: float = 1.0
    control_x: tuple[float, ...] = ()
    control_y: tuple[float, ...] = ()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        xa = np.clip(np.asarray(x, dtype=float), 1e-6, 1.0)

        if self.kind == "constant":
            return np.full_like(xa, self.root)

        if self.kind == "linear":
            return self.root + (self.tip - self.root) * xa

        if self.kind == "parabolic":
            return self.root + (self.tip - self.root) * xa ** max(self.power, 1e-3)

        if self.kind == "elliptic":
            # Elliptic planform: c(x) = c_root * sqrt(1 - x^2), scaled so the
            # value at x = 0.75 equals ``root`` (the usual reference station).
            shape = np.sqrt(np.clip(1.0 - xa ** 2, 0.0, 1.0))
            ref = math.sqrt(max(1.0 - 0.75 ** 2, 1e-9))
            return self.root * shape / ref + self.tip

        if self.kind == "inverse":
            # Ideal-twist / constant-circulation family: value ~ 1/x.
            return self.root * 0.75 / xa + self.tip

        if self.kind == "betz":
            # Betz-Prandtl minimum-induced-loss chord shape, normalised to
            # ``root`` at x = 0.75.  ``power`` plays the role of the design
            # tip-speed ratio.
            lam = max(self.power, 0.5)
            f = lambda t: t / (lam ** 2 + t ** 2)  # noqa: E731
            return self.root * f(xa) / max(f(0.75), 1e-9) + self.tip

        if self.kind == "spline":
            if len(self.control_x) < 2:
                return np.full_like(xa, self.root)
            cx = np.asarray(self.control_x, dtype=float)
            cy = np.asarray(self.control_y, dtype=float)
            order = np.argsort(cx)
            return _pchip(cx[order], cy[order], xa)

        raise ValueError(f"unknown distribution kind {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Distribution":
        d = dict(d)
        d["control_x"] = tuple(d.get("control_x") or ())
        d["control_y"] = tuple(d.get("control_y") or ())
        return cls(**d)


def _pchip(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Shape-preserving cubic interpolation (no SciPy dependency).

    Monotone Fritsch-Carlson derivatives, so dragging a chord control point
    cannot make the planform overshoot into a negative chord.
    """
    n = x.size
    if n == 2:
        return np.interp(xi, x, y)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n)
    d[1:-1] = np.where(
        delta[:-1] * delta[1:] > 0.0,
        2.0 / (1.0 / np.where(np.abs(delta[:-1]) < 1e-12, 1e-12, delta[:-1])
               + 1.0 / np.where(np.abs(delta[1:]) < 1e-12, 1e-12, delta[1:])),
        0.0,
    )
    d[0] = delta[0]
    d[-1] = delta[-1]

    xi_c = np.clip(xi, x[0], x[-1])
    idx = np.clip(np.searchsorted(x, xi_c) - 1, 0, n - 2)
    hs = h[idx]
    t = (xi_c - x[idx]) / hs
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * y[idx] + h10 * hs * d[idx] + h01 * y[idx + 1] + h11 * hs * d[idx + 1]


# ---------------------------------------------------------------------------
# Blade
# ---------------------------------------------------------------------------

#: Lower limit of the standard activity-factor integral (r/R = 0.15).
AF_LOWER = 0.15


@dataclass(slots=True)
class BladeGeometry:
    """Full parametric description of one propeller.

    Lengths are metres and angles radians.  Every field that describes the
    propeller is required -- there is no default blade.  ``pitch_offset`` is the
    collective a constant-speed hub (or the GUI) applies on top of the built-in
    twist; it rigidly rotates every section.

    Use :func:`full_size_blade` to build one from the numbers a propeller is
    normally specified by (diameter, blades, activity factor, pitch).
    """

    n_blades: int
    radius: float                         # m
    hub_radius_frac: float                # r_hub / R
    chord: Distribution                   # c/R against r/R
    twist: Distribution                   # blade angle (rad) against r/R
    thickness: Distribution               # t/c against r/R
    root_airfoil: str
    tip_airfoil: str
    name: str = ""
    sweep: Distribution | None = None     # tangential offset / R; None = straight
    dihedral: Distribution | None = None  # axial offset / R; None = straight
    pitch_offset: float = 0.0             # rad, collective
    airfoil_blend_start: float = 0.30     # r/R where root -> tip blending begins
    chord_scale: float = 1.0
    rake: float = 0.0

    def __post_init__(self) -> None:
        if self.n_blades < 1:
            raise ValueError("a propeller needs at least one blade")
        if not self.radius > 0.0:
            raise ValueError("radius must be positive")
        if not 0.0 < self.hub_radius_frac < 0.9:
            raise ValueError("hub radius ratio must be between 0 and 0.9")

    # -- derived -----------------------------------------------------------
    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def hub_radius(self) -> float:
        return self.hub_radius_frac * self.radius

    @property
    def disk_area(self) -> float:
        return math.pi * (self.radius ** 2 - self.hub_radius ** 2)

    def x_stations(self, n: int, spacing: str = "cosine") -> tuple[np.ndarray, np.ndarray]:
        """Non-dimensional radii, clustered at the tip where gradients bite."""
        x0 = self.hub_radius_frac
        if spacing == "uniform":
            edges = np.linspace(x0, 1.0, n + 1)
        elif spacing == "cosine":
            beta = np.linspace(0.0, math.pi, n + 1)
            edges = x0 + (1.0 - x0) * 0.5 * (1.0 - np.cos(beta))
        elif spacing == "tip":
            t = np.linspace(0.0, 1.0, n + 1)
            edges = x0 + (1.0 - x0) * (1.0 - (1.0 - t) ** 1.6)
        else:
            raise ValueError(f"unknown spacing {spacing!r}")
        return 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)

    def chord_at(self, x: np.ndarray) -> np.ndarray:
        """Local chord in metres."""
        c = self.chord.evaluate(x) * self.radius * self.chord_scale
        return np.maximum(c, 1e-5)

    def twist_at(self, x: np.ndarray) -> np.ndarray:
        """Local blade angle (rad) including the collective offset."""
        return self.twist.evaluate(x) + self.pitch_offset

    def thickness_at(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.thickness.evaluate(x), 0.02, 0.45)

    def solidity_at(self, x: np.ndarray) -> np.ndarray:
        """Local solidity sigma = B c / (2 pi r)."""
        r = np.maximum(np.asarray(x, dtype=float) * self.radius, 1e-9)
        return self.n_blades * self.chord_at(x) / (2.0 * math.pi * r)

    def activity_factor(self, n: int = 400) -> float:
        """Per-blade activity factor, the standard power-absorption index.

        ``AF = (100000/16) * integral from 0.15 to 1 of (c/D) x^3 dx``, with
        chord referenced to *diameter*.  Full-size propellers run from about 80
        (narrow fixed-pitch) to 200 (wide turboprop blades).
        """
        x = np.linspace(AF_LOWER, 1.0, n)
        c_over_d = self.chord_at(x) / self.diameter
        return float(100000.0 / 16.0 * np.trapezoid(c_over_d * x ** 3, x))

    def mean_solidity(self, n: int = 200) -> float:
        x, dx = self.x_stations(n, "uniform")
        c = self.chord_at(x)
        return float(self.n_blades * np.sum(c * dx * self.radius) / (math.pi * self.radius ** 2))

    def beta75(self) -> float:
        """Blade angle at 0.75 R (rad), including collective."""
        return float(self.twist_at(np.array([0.75]))[0])

    def pitch_diameter_ratio(self, x_ref: float = 0.75) -> float:
        """Geometric pitch/diameter at ``x_ref``: the "75x57" number over D."""
        theta = float(self.twist_at(np.array([x_ref]))[0])
        return 2.0 * math.pi * x_ref * self.radius * math.tan(theta) / self.diameter

    def geometric_pitch(self, x_ref: float = 0.75) -> float:
        """Geometric pitch at ``x_ref`` in metres."""
        return self.pitch_diameter_ratio(x_ref) * self.diameter

    def sections(self, x: np.ndarray) -> list[AnalyticPolar]:
        """Per-station section, blended from root to tip, at its own thickness."""
        root = get_airfoil(self.root_airfoil)
        tip = get_airfoil(self.tip_airfoil)
        xs = np.asarray(x, dtype=float)
        thickness = self.thickness_at(xs)
        if root.name == tip.name:
            return [replace(root, thickness=float(t)) for t in thickness]
        x0 = self.airfoil_blend_start
        # Smoothstep, not a linear ramp: a linear blend is continuous but its
        # derivative jumps at x0, which shows as a corner in spanwise loading.
        u = np.clip((xs - x0) / max(1.0 - x0, 1e-6), 0.0, 1.0)
        f = u * u * (3.0 - 2.0 * u)
        return [replace(blended_polar(root, tip, float(fi)), thickness=float(ti))
                for fi, ti in zip(f, thickness)]

    def discretize(self, n: int = 60, spacing: str = "cosine") -> "BladeStations":
        x, dx = self.x_stations(n, spacing)
        zero = np.zeros_like(x)
        sweep = self.sweep.evaluate(x) * self.radius if self.sweep is not None else zero
        dihedral = (self.dihedral.evaluate(x) * self.radius if self.dihedral is not None
                    else zero) + self.rake * x ** 2
        return BladeStations(
            x=x, dx=dx, r=x * self.radius, dr=dx * self.radius,
            chord=self.chord_at(x), twist=self.twist_at(x),
            thickness=self.thickness_at(x), sweep=sweep, dihedral=dihedral,
            solidity=self.solidity_at(x), polars=self.sections(x), geometry=self,
        )

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("chord", "twist", "thickness", "sweep", "dihedral"):
            value = getattr(self, key)
            d[key] = value.to_dict() if value is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BladeGeometry":
        d = dict(d)
        for key in ("chord", "twist", "thickness", "sweep", "dihedral"):
            if isinstance(d.get(key), dict):
                d[key] = Distribution.from_dict(d[key])
        known = set(cls.__slots__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BladeGeometry":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def describe(self) -> str:
        label = f"{self.name}: " if self.name else ""
        return (f"{label}{self.n_blades}-blade, D = {self.diameter:.3f} m "
                f"({self.diameter / INCH:.1f} in), beta75 = {math.degrees(self.beta75()):.1f} deg, "
                f"P/D = {self.pitch_diameter_ratio():.2f}, AF = {self.activity_factor():.0f}")


@dataclass(slots=True)
class BladeStations:
    """Discretised blade -- the array form every solver and mesher consumes."""

    x: np.ndarray
    dx: np.ndarray
    r: np.ndarray
    dr: np.ndarray
    chord: np.ndarray
    twist: np.ndarray
    thickness: np.ndarray
    sweep: np.ndarray
    dihedral: np.ndarray
    solidity: np.ndarray
    polars: list[AnalyticPolar]
    geometry: BladeGeometry

    @property
    def n(self) -> int:
        return int(self.x.size)

    def polar_tables(self, n_alpha: int = 721) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(alpha_grid, cl[n_stations, n_alpha], cd[n_stations, n_alpha])``."""
        return stack_tables(self.polars, n_alpha)

    def section_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-station ``(thickness, re_ref, kappa, cl_max)`` for the kernels.

        ``cl_max`` is the station's thickness-adjusted maximum lift; it bounds
        the Prandtl-Glauert amplification, without which a transonic tip is
        handed a lift coefficient it could never reach.
        """
        return (
            np.array([p.thickness for p in self.polars], dtype=np.float64),
            np.array([p.re_ref for p in self.polars], dtype=np.float64),
            np.array([p.kappa for p in self.polars], dtype=np.float64),
            np.array([p.max_lift() for p in self.polars], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Building a blade from the numbers a full-size propeller is specified by
# ---------------------------------------------------------------------------

#: Planform shapes, as relative chord against r/R.  Each is scaled so the blade
#: has exactly the activity factor asked for; the shape only sets *where* the
#: chord is.  These are typical outlines, not any manufacturer's drawing.
PLANFORMS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    # Fixed-pitch metal blade: widest near mid-span, rounded tip.
    "standard": ((0.00, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95, 1.00),
                 (0.50, 0.62, 0.83, 0.98, 1.00, 0.95, 0.80, 0.55, 0.18)),
    # Wide-chord constant-speed ("paddle") blade: chord carried well outboard.
    "paddle":   ((0.00, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95, 1.00),
                 (0.45, 0.55, 0.70, 0.86, 0.96, 1.00, 0.98, 0.80, 0.25)),
    # Straight taper from root to tip.
    "tapered":  ((0.00, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95, 1.00),
                 (1.00, 1.00, 0.97, 0.92, 0.86, 0.79, 0.70, 0.60, 0.30)),
}


def list_planforms() -> list[str]:
    return list(PLANFORMS)


def helical_twist(pitch_over_diameter: float) -> Distribution:
    """Blade angle of a constant-geometric-pitch screw: ``atan(P/D / (pi x))``.

    This is what a propeller's "75x57" designation means: 57 inches of advance
    per revolution at every station.
    """
    xs = np.linspace(0.02, 1.0, 50)
    beta = np.arctan(pitch_over_diameter / (math.pi * xs))
    return Distribution("spline", control_x=tuple(float(v) for v in xs),
                        control_y=tuple(float(v) for v in beta))


def pitch_ratio_from_beta75(beta75_deg: float) -> float:
    """P/D of a helical blade whose 0.75 R blade angle is ``beta75_deg``."""
    return math.pi * 0.75 * math.tan(math.radians(beta75_deg))


def full_size_blade(*, diameter: float, n_blades: int, activity_factor: float,
                    hub_ratio: float, root_airfoil: str, tip_airfoil: str,
                    thickness_root: float, thickness_tip: float, planform: str,
                    beta75_deg: float | None = None, pitch: float | None = None,
                    name: str = "") -> BladeGeometry:
    """Build a blade from the way full-size propellers are specified.

    Exactly one of ``beta75_deg`` (blade angle at 0.75 R) or ``pitch``
    (geometric pitch in metres, the "57" in "75x57") must be given; the twist
    is helical (constant geometric pitch).  Chord follows ``planform``, scaled
    so the blade has exactly ``activity_factor``.  Thickness ratio varies
    linearly from ``thickness_root`` at the hub to ``thickness_tip`` at the tip.
    Nothing is assumed: every argument is required.
    """
    if (beta75_deg is None) == (pitch is None):
        raise ValueError("give exactly one of beta75_deg or pitch")
    if planform not in PLANFORMS:
        raise ValueError(f"unknown planform {planform!r}; have {list(PLANFORMS)}")
    for label, value in (("diameter", diameter), ("activity_factor", activity_factor),
                         ("thickness_root", thickness_root), ("thickness_tip", thickness_tip)):
        if not value > 0.0:
            raise ValueError(f"{label} must be positive")
    for section in (root_airfoil, tip_airfoil):
        get_airfoil(section)                          # raises on an unknown name

    p_over_d = (pitch / diameter) if pitch is not None else pitch_ratio_from_beta75(beta75_deg)

    xs, shape = (np.asarray(v, dtype=float) for v in PLANFORMS[planform])
    # Integrate with the same interpolant the blade will be evaluated with, or
    # the activity factor comes back a couple of percent off what was asked.
    outline = Distribution("spline", control_x=tuple(float(v) for v in xs),
                           control_y=tuple(float(v) for v in shape))
    grid = np.linspace(AF_LOWER, 1.0, 4001)
    integral = np.trapezoid(outline.evaluate(grid) * grid ** 3, grid)
    k = activity_factor / (100000.0 / 16.0 * integral)      # c/D = k * shape
    c_over_r = 2.0 * k * shape                               # c/R = 2 c/D

    x_hub = float(hub_ratio)
    return BladeGeometry(
        n_blades=int(n_blades), radius=diameter / 2.0, hub_radius_frac=x_hub,
        chord=Distribution("spline", control_x=tuple(float(v) for v in xs),
                           control_y=tuple(float(v) for v in c_over_r)),
        twist=helical_twist(p_over_d),
        thickness=Distribution("spline", control_x=(0.0, x_hub, 1.0),
                               control_y=(float(thickness_root), float(thickness_root),
                                          float(thickness_tip))),
        root_airfoil=root_airfoil, tip_airfoil=tip_airfoil, name=name,
    )


__all__ = [
    "Distribution", "BladeGeometry", "BladeStations", "PLANFORMS", "AF_LOWER",
    "list_planforms", "list_airfoils", "helical_twist", "pitch_ratio_from_beta75",
    "full_size_blade", "DEG",
]
