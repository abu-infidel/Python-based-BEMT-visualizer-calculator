"""Blade planform: chord, twist, thickness and section distributions.

A propeller blade is described here as a set of *distributions* over the
non-dimensional radius ``x = r / R``.  Each distribution is a small serialisable
spec (kind + parameters) rather than a raw array, so the GUI can expose a
handful of sliders, presets round-trip through JSON, and the 3-D mesher and the
solver read the exact same geometry.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from .airfoil import (DEG, AirfoilPolar, AnalyticPolar, blended_polar,
                      get_airfoil, stack_tables)
from .units import INCH, inch_pitch_to_twist

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

@dataclass(slots=True)
class BladeGeometry:
    """Full parametric description of one propeller.

    Lengths are metres and angles radians.  ``pitch_offset`` is the collective
    the user actually drives -- it rigidly rotates every section, exactly like a
    variable-pitch hub.
    """

    name: str = "Custom prop"
    n_blades: int = 2
    radius: float = 0.127                 # m (a 10-inch prop)
    hub_radius_frac: float = 0.15         # r_hub / R
    chord: Distribution = field(default_factory=lambda: Distribution("elliptic", root=0.105, tip=0.008))
    twist: Distribution = field(default_factory=lambda: Distribution("inverse", root=0.35, tip=0.0))
    thickness: Distribution = field(default_factory=lambda: Distribution("linear", root=0.16, tip=0.08))
    sweep: Distribution = field(default_factory=lambda: Distribution("constant", root=0.0))
    dihedral: Distribution = field(default_factory=lambda: Distribution("constant", root=0.0))
    pitch_offset: float = 0.0             # rad, collective
    root_airfoil: str = "clarky"
    tip_airfoil: str = "clarky"
    airfoil_blend_start: float = 0.35     # x where the transition begins
    chord_scale: float = 1.0              # global chord multiplier (solidity knob)
    rake: float = 0.0                     # m, axial offset of the tip

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

    def x_stations(self, n: int, spacing: str = "cosine") -> np.ndarray:
        """Non-dimensional radii, clustered at the tip where gradients bite.

        Cosine spacing puts more elements where the tip-loss factor is changing
        fastest, which is worth several percent of integrated thrust accuracy
        for a given element count.
        """
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

    def activity_factor(self, n: int = 200) -> float:
        """Per-blade activity factor, the classic power-absorption index.

        ``AF = (100000/16) * integral( (c/D) x^3 dx )``.  Note the chord is
        referenced to *diameter*, not radius -- getting that wrong doubles the
        number.  Sport model propellers land around 90-140.
        """
        x, _ = self.x_stations(n, "uniform")
        c_over_d = self.chord_at(x) / self.diameter
        return float(100000.0 / 16.0 * np.trapezoid(c_over_d * x ** 3, x))

    def mean_solidity(self, n: int = 200) -> float:
        x, dx = self.x_stations(n, "uniform")
        c = self.chord_at(x)
        return float(self.n_blades * np.sum(c * dx * self.radius) / (math.pi * self.radius ** 2))

    def pitch_diameter_ratio(self, x_ref: float = 0.75) -> float:
        """Geometric pitch/diameter at the 75% station -- the "10x5" number.

        Geometric pitch is the advance of a screw with the local blade angle:
        ``p = 2 pi r tan(theta)``.
        """
        theta = float(self.twist_at(np.array([x_ref]))[0])
        return 2.0 * math.pi * x_ref * self.radius * math.tan(theta) / self.diameter

    def sections(self, x: np.ndarray) -> list[AnalyticPolar]:
        """Per-station airfoil, blended from root section to tip section."""
        root = get_airfoil(self.root_airfoil)
        tip = get_airfoil(self.tip_airfoil)
        xs = np.asarray(x, dtype=float)
        if self.root_airfoil == self.tip_airfoil:
            return [replace(root, thickness=float(t)) for t in self.thickness_at(xs)]
        x0 = self.airfoil_blend_start
        f = np.clip((xs - x0) / max(1.0 - x0, 1e-6), 0.0, 1.0)
        out = []
        for fi, ti in zip(f, self.thickness_at(xs)):
            out.append(replace(blended_polar(root, tip, float(fi)), thickness=float(ti)))
        return out

    def discretize(self, n: int = 60, spacing: str = "cosine") -> "BladeStations":
        x, dx = self.x_stations(n, spacing)
        return BladeStations(
            x=x,
            dx=dx,
            r=x * self.radius,
            dr=dx * self.radius,
            chord=self.chord_at(x),
            twist=self.twist_at(x),
            thickness=self.thickness_at(x),
            sweep=self.sweep.evaluate(x) * self.radius,
            dihedral=self.dihedral.evaluate(x) * self.radius + self.rake * x ** 2,
            solidity=self.solidity_at(x),
            polars=self.sections(x),
            geometry=self,
        )

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("chord", "twist", "thickness", "sweep", "dihedral"):
            d[key] = self.__getattribute__(key).to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BladeGeometry":
        d = dict(d)
        for key in ("chord", "twist", "thickness", "sweep", "dihedral"):
            if key in d and isinstance(d[key], dict):
                d[key] = Distribution.from_dict(d[key])
        known = {f for f in cls.__slots__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BladeGeometry":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def describe(self) -> str:
        return (
            f"{self.name}: {self.n_blades}-blade, D={self.diameter * 1000:.0f} mm "
            f"({self.diameter / INCH:.1f} in), P/D={self.pitch_diameter_ratio():.2f}, "
            f"AF={self.activity_factor():.0f}, sigma={self.mean_solidity():.3f}"
        )


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

    def section_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-station (thickness, re_ref, m_crit0) for the in-kernel corrections."""
        return (
            np.array([p.thickness for p in self.polars], dtype=np.float64),
            np.array([p.re_ref for p in self.polars], dtype=np.float64),
            np.array([p.m_crit0 for p in self.polars], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def constant_pitch_twist(pitch_inches: float, radius: float,
                         x_ref: Sequence[float] = (0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)) -> Distribution:
    """Twist distribution of a true constant-geometric-pitch screw.

    This is what "10x5" means: 5 inches of advance per revolution at every
    station, so the blade angle falls off as atan(p / 2 pi r).
    """
    p = pitch_inches * INCH
    xs = np.asarray(x_ref, dtype=float)
    ys = np.array([inch_pitch_to_twist(p, float(xi) * radius) for xi in xs])
    return Distribution("spline", control_x=tuple(xs), control_y=tuple(ys))


PROP_PRESETS: dict[str, BladeGeometry] = {}


def _register(geom: BladeGeometry) -> BladeGeometry:
    PROP_PRESETS[geom.name] = geom
    return geom


def _apc_style(name: str, d_in: float, p_in: float, blades: int = 2) -> BladeGeometry:
    r = d_in * INCH / 2.0
    return _register(BladeGeometry(
        name=name, n_blades=blades, radius=r, hub_radius_frac=0.15,
        chord=Distribution("spline",
                           control_x=(0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0),
                           control_y=(0.114, 0.155, 0.178, 0.188, 0.186, 0.175, 0.155, 0.127, 0.080, 0.020)),
        twist=constant_pitch_twist(p_in, r),
        thickness=Distribution("spline",
                               control_x=(0.15, 0.3, 0.5, 0.75, 1.0),
                               control_y=(0.22, 0.145, 0.098, 0.072, 0.055)),
        root_airfoil="clarky", tip_airfoil="e63", airfoil_blend_start=0.4,
    ))


_apc_style("APC 10x5 (sport)", 10.0, 5.0)
_apc_style("APC 10x7 (fast)", 10.0, 7.0)
_apc_style("APC 9x4.5 (slow-fly)", 9.0, 4.5)
_apc_style("APC 13x6.5 (trainer)", 13.0, 6.5)
_apc_style("APC 6x4 (micro)", 6.0, 4.0)
_apc_style("Tri-blade 10x6", 10.0, 6.0, blades=3)

_register(BladeGeometry(
    name="UAV quad 15x5.5 (2-blade)", n_blades=2, radius=15.0 * INCH / 2.0,
    hub_radius_frac=0.12,
    chord=Distribution("spline",
                       control_x=(0.12, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0),
                       control_y=(0.098, 0.152, 0.166, 0.150, 0.118, 0.072, 0.016)),
    twist=constant_pitch_twist(5.5, 15.0 * INCH / 2.0),
    thickness=Distribution("linear", root=0.20, tip=0.06),
    root_airfoil="clarky", tip_airfoil="mh117", airfoil_blend_start=0.3,
))

_register(BladeGeometry(
    name="Ideal-twist research prop", n_blades=2, radius=0.15, hub_radius_frac=0.20,
    chord=Distribution("betz", root=0.145, tip=0.004, power=3.0),
    twist=Distribution("inverse", root=0.42, tip=-0.02),
    thickness=Distribution("linear", root=0.14, tip=0.07),
    root_airfoil="naca4412", tip_airfoil="naca16509", airfoil_blend_start=0.3,
))

_register(BladeGeometry(
    name="Elliptical testbed", n_blades=2, radius=0.127, hub_radius_frac=0.15,
    chord=Distribution("elliptic", root=0.155, tip=0.006),
    twist=Distribution("linear", root=0.45, tip=0.12),
    thickness=Distribution("linear", root=0.15, tip=0.08),
    root_airfoil="naca0012", tip_airfoil="naca0012",
))

_register(BladeGeometry(
    name="Scale warbird 4-blade", n_blades=4, radius=0.20, hub_radius_frac=0.22,
    chord=Distribution("spline",
                       control_x=(0.22, 0.4, 0.6, 0.8, 0.95, 1.0),
                       control_y=(0.105, 0.148, 0.156, 0.134, 0.076, 0.018)),
    twist=constant_pitch_twist(9.0, 0.20),
    thickness=Distribution("spline",
                           control_x=(0.22, 0.5, 0.8, 1.0),
                           control_y=(0.24, 0.12, 0.080, 0.060)),
    root_airfoil="arad20", tip_airfoil="naca16509", airfoil_blend_start=0.35,
    sweep=Distribution("parabolic", root=0.0, tip=0.05, power=2.5),
))

DEFAULT_PRESET = "APC 10x5 (sport)"


def get_preset(name: str) -> BladeGeometry:
    if name not in PROP_PRESETS:
        raise KeyError(f"unknown preset {name!r}; have {list(PROP_PRESETS)}")
    return replace(PROP_PRESETS[name])


def list_presets() -> list[str]:
    return list(PROP_PRESETS)


__all__ = [
    "Distribution", "BladeGeometry", "BladeStations", "constant_pitch_twist",
    "PROP_PRESETS", "DEFAULT_PRESET", "get_preset", "list_presets", "DEG",
]
