r"""Propeller sizing and design: how big, and what shape?

Two methods, in increasing order of what they tell you.

**1. Momentum-theory disk sizing** (:func:`momentum_sizing`).  You know the
thrust you need -- from the drag at your target cruise speed -- and you want to
know how much engine that costs for a given diameter.  An actuator disk of area
:math:`A` producing thrust :math:`T` at forward speed :math:`V` must accelerate
the air by an induced velocity :math:`w`:

.. math::
    T = 2\rho A w (V + w)
    \quad\Longrightarrow\quad
    w = -\frac{V}{2} + \sqrt{\frac{V^2}{4} + \frac{T}{2\rho A}}

and the ideal power is :math:`P = T(V + w)`, with shaft power
:math:`P/\eta_p`.  This is a *lower bound* -- no propeller of that diameter can
do better -- which makes it exactly the right tool for choosing a diameter.
Bigger disk, smaller induced velocity, less wasted power; the limits are ground
clearance and tip Mach number, and :func:`diameter_sweep` shows you where they
bite.

**2. Minimum-induced-loss blade design** (:func:`adkins_liebeck_design`).  The
momentum method sizes the disk but says nothing about the blade.  This one
returns the actual optimum chord and twist distribution, following Adkins &
Liebeck (1994), which is a corrected and convergent form of the classical
Betz-Prandtl condition for minimum induced loss.

The physical idea is Betz's: induced losses are least when the shed vortex
sheet is a rigid helicoid moving aft at constant speed.  That condition fixes
the circulation distribution, and with it the product :math:`Wc` at every
station; dividing by the local resultant velocity gives the chord, and the
flow angle plus the section's design angle of attack gives the twist.

The output is a :class:`~propwash.geometry.BladeGeometry`, so a designed
propeller goes straight back through the BEMT solver -- which is the honest
check, because the design method and the analysis method are independent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .airfoil import (DEG, AirfoilPolar, critical_mach, drag_divergence,
                      get_airfoil)
from .atmosphere import AirState
from .geometry import BladeGeometry, Distribution
from .units import HORSEPOWER, rpm_to_rad_s


# ---------------------------------------------------------------------------
# 1. Momentum-theory disk sizing
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DiskSizing:
    """What an ideal actuator disk of a given diameter would cost."""

    diameter: float
    disk_area: float
    thrust: float
    v_inf: float
    induced_velocity: float
    ideal_power: float             # W, induced only
    shaft_power: float             # W, after propulsive efficiency
    disk_loading: float            # N/m^2
    ideal_efficiency: float        # V / (V + w), the Froude limit
    assumed_efficiency: float
    tip_mach: float | None = None

    @property
    def shaft_power_hp(self) -> float:
        return self.shaft_power / HORSEPOWER

    def describe(self) -> str:
        return (f"D = {self.diameter:.2f} m ({self.diameter / 0.0254:.0f} in): "
                f"w = {self.induced_velocity:.2f} m/s, "
                f"ideal {self.ideal_power / 1000:.1f} kW, "
                f"shaft {self.shaft_power_hp:.0f} hp, "
                f"Froude limit {self.ideal_efficiency * 100:.1f}%")


def drag_from_weight(mass: float, lift_to_drag: float, g: float = 9.80665) -> float:
    """Cruise drag from weight and a target L/D -- ``D = W / (L/D)``.

    The conceptual-design shortcut: if you do not have a drag polar yet, an L/D
    target is usually the one number you *do* have.  A clean light single sits
    around 10-12.
    """
    return mass * g / max(lift_to_drag, 1e-6)


def induced_velocity(thrust: float, v_inf: float, disk_area: float,
                     rho: float = 1.225) -> float:
    r"""Solve :math:`T = 2\rho A w (V + w)` for the induced velocity.

    The positive root of the quadratic; at :math:`V = 0` it reduces to the
    familiar hover result :math:`w = \sqrt{T / 2\rho A}`.
    """
    if disk_area <= 0.0:
        return float("inf")
    return -0.5 * v_inf + math.sqrt(0.25 * v_inf ** 2
                                    + max(thrust, 0.0) / (2.0 * rho * disk_area))


def momentum_sizing(thrust: float, v_inf: float, diameter: float, air: AirState,
                    eta_p: float, k_fan: float = 1.0,
                    rpm: float | None = None) -> DiskSizing:
    """Size an actuator disk: thrust and speed in, required power out.

    ``eta_p`` is the propulsive efficiency you expect of a real propeller of
    this class (0.80-0.85 for a decent fixed-pitch cruise propeller, 0.85-0.88
    constant-speed).  ``k_fan`` is Gudmundsson's ducted-fan factor; leave it at
    1 for an open propeller.
    """
    area = math.pi * diameter ** 2 / 4.0
    w = induced_velocity(thrust, v_inf, area, air.density)
    ideal = thrust * (v_inf + w)
    shaft = ideal / max(eta_p * k_fan, 1e-6)

    tip_mach = None
    if rpm is not None:
        tip_mach = math.hypot(rpm_to_rad_s(rpm) * diameter / 2.0, v_inf) / air.sound_speed

    return DiskSizing(
        diameter=diameter, disk_area=area, thrust=thrust, v_inf=v_inf,
        induced_velocity=w, ideal_power=ideal, shaft_power=shaft,
        disk_loading=thrust / area if area > 0 else float("inf"),
        ideal_efficiency=v_inf / (v_inf + w) if (v_inf + w) > 0 else 0.0,
        assumed_efficiency=eta_p, tip_mach=tip_mach,
    )


def diameter_sweep(thrust: float, v_inf: float, diameters: np.ndarray, air: AirState,
                   eta_p: float, rpm: float | None = None,
                   tip_mach_limit: float | None = None) -> dict[str, np.ndarray]:
    """Required power against diameter -- the trade that picks the propeller.

    Power falls monotonically with diameter, so the choice is set by the
    constraints, not the optimum: ground clearance at the big end and tip Mach
    at the small end.  ``tip_mach_limit`` marks where compressibility starts
    eating the gain; ``feasible`` is False beyond it.
    """
    d = np.asarray(diameters, dtype=float)

    area = math.pi * d ** 2 / 4.0
    w = -0.5 * v_inf + np.sqrt(0.25 * v_inf ** 2
                               + max(thrust, 0.0) / (2.0 * air.density * area))
    ideal = thrust * (v_inf + w)
    shaft = ideal / max(eta_p, 1e-6)

    out = {
        "diameter": d, "disk_area": area, "induced_velocity": w,
        "ideal_power": ideal, "shaft_power": shaft,
        "shaft_power_hp": shaft / HORSEPOWER,
        "disk_loading": thrust / area,
        "ideal_efficiency": np.where(v_inf + w > 0, v_inf / (v_inf + w), 0.0),
    }
    if rpm is not None:
        tip = np.hypot(rpm_to_rad_s(rpm) * d / 2.0, v_inf) / air.sound_speed
        out["tip_mach"] = tip
        if tip_mach_limit is not None:
            out["feasible"] = tip <= tip_mach_limit
    return out


# ---------------------------------------------------------------------------
# 2. Adkins-Liebeck minimum-induced-loss design
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BladeDesign:
    """An optimum blade: its geometry, and what the method predicts of it."""

    radius: float
    hub_radius: float
    n_blades: int
    rpm: float
    v_inf: float
    x: np.ndarray                 # r/R at each design station
    chord: np.ndarray             # m
    twist: np.ndarray             # rad, blade angle (alpha + phi)
    phi: np.ndarray               # rad, flow angle
    a_axial: np.ndarray
    a_swirl: np.ndarray
    reynolds: np.ndarray
    zeta: float                   # displacement velocity ratio
    thrust: float                 # N, as predicted by the design method
    power: float                  # W
    efficiency: float
    design_cl: float
    design_alpha: float
    converged: bool
    iterations: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def activity_factor(self) -> float:
        return float(100000.0 / 16.0
                     * np.trapezoid(self.chord / self.diameter * self.x ** 3, self.x))

    def to_geometry(self, *, airfoil: str, thickness_root: float, thickness_tip: float,
                    tip_airfoil: str | None = None,
                    name: str = "Minimum-induced-loss design") -> BladeGeometry:
        """Turn the design into a blade the BEMT solver can analyse.

        Chord and twist become spline distributions through the design
        stations, so the analysis sees the geometry the design produced rather
        than a smoothed idealisation of it.
        """
        keep = np.linspace(0, self.x.size - 1, min(12, self.x.size)).astype(int)
        xs = tuple(float(v) for v in self.x[keep])
        return BladeGeometry(
            name=name, n_blades=self.n_blades, radius=self.radius,
            hub_radius_frac=self.hub_radius / self.radius,
            chord=Distribution("spline", control_x=xs,
                               control_y=tuple(float(c / self.radius)
                                               for c in self.chord[keep])),
            twist=Distribution("spline", control_x=xs,
                               control_y=tuple(float(t) for t in self.twist[keep])),
            thickness=Distribution("spline", control_x=(0.0, self.hub_radius / self.radius, 1.0),
                                   control_y=(float(thickness_root), float(thickness_root),
                                              float(thickness_tip))),
            root_airfoil=airfoil, tip_airfoil=tip_airfoil or airfoil,
            airfoil_blend_start=0.35,
        )

    def describe(self) -> str:
        status = "converged" if self.converged else "NOT converged"
        return (f"{self.n_blades}-blade, D = {self.diameter:.3f} m "
                f"({self.diameter / 0.0254:.1f} in) at {self.rpm:.0f} rpm, "
                f"V = {self.v_inf:.1f} m/s -> T = {self.thrust:.0f} N, "
                f"P = {self.power / 1000:.1f} kW ({self.power / HORSEPOWER:.0f} hp), "
                f"eta = {self.efficiency:.3f}, AF = {self.activity_factor:.0f} "
                f"[{status} in {self.iterations} iterations]")


def _compressible_section(polar: AirfoilPolar, cl_target: float, mach: np.ndarray,
                          thickness: np.ndarray, kappa: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Angle of attack and drag needed to *achieve* ``cl_target`` at each Mach.

    Compressibility amplifies lift by the Prandtl-Glauert factor, so a section
    asked to produce a given Cl needs a progressively smaller angle of attack
    as Mach rises -- and picks up wave drag once past drag divergence.  The
    design method must model this or it will hand the analysis a blade that
    over-produces, which is exactly the +17% seen when it does not.

    Returns ``(alpha, cd)`` per station.
    """
    grid = np.linspace(-8.0 * DEG, 16.0 * DEG, 601)
    cls, cds = polar.base_polar(grid)
    peak = int(np.argmax(cls))

    beta = np.sqrt(np.clip(1.0 - np.clip(mach, 0.0, 0.92) ** 2, 1e-3, 1.0))
    cl_inc = cl_target * beta                     # incompressible Cl that yields the target

    alpha = np.interp(cl_inc, cls[:peak + 1], grid[:peak + 1])
    cd = np.interp(alpha, grid, cds)

    m_cr = critical_mach(abs(cl_target), np.asarray(thickness, dtype=float), kappa)
    cd = cd + drag_divergence(np.asarray(mach, dtype=float), m_cr)
    return alpha, cd


def _design_point(polar: AirfoilPolar, design_cl: float | None,
                  design_alpha: float | None) -> tuple[float, float, float]:
    """Resolve the section's design point to ``(alpha, Cl, Cd)``.

    Given a target Cl, invert the polar for the angle that produces it; given an
    angle, read off Cl.  With neither, pick the angle of best L/D, which is what
    a designer would choose anyway.
    """
    alphas = np.linspace(-6.0 * DEG, 16.0 * DEG, 441)
    cls, cds = polar.base_polar(alphas)

    if design_alpha is not None:
        a = float(design_alpha)
    elif design_cl is not None:
        rising = np.argmax(cls)                 # invert on the pre-stall branch
        a = float(np.interp(design_cl, cls[:rising + 1], alphas[:rising + 1]))
    else:
        a = float(alphas[int(np.argmax(cls / np.maximum(cds, 1e-9)))])

    cl, cd = polar.base_polar(np.array([a]))
    return a, float(cl[0]), float(max(cd[0], 1e-5))


def adkins_liebeck_design(
        radius: float, n_blades: int, rpm: float, v_inf: float, *,
        air: AirState, airfoil: str | AirfoilPolar, hub_radius_frac: float,
        thickness_root: float, thickness_tip: float,
        thrust: float | None = None, power: float | None = None,
        design_cl: float | None = None, design_alpha: float | None = None,
        n_stations: int = 40, max_iter: int = 60, tol: float = 1e-7,
        compressible: bool = True) -> BladeDesign:
    r"""Design a minimum-induced-loss propeller (Adkins & Liebeck, 1994).

    Specify exactly one of ``thrust`` (N) or ``power`` (W): the first designs a
    propeller that delivers a required thrust, the second one that absorbs a
    given engine.

    The method iterates on the *displacement velocity ratio* :math:`\zeta`,
    which sets how fast the helical wake moves aft.  For a trial :math:`\zeta`
    the flow angle at every station follows from
    :math:`\tan\phi = (1 + \zeta/2)/x` with :math:`x = \Omega r / V`, and Betz's
    condition then fixes the product :math:`Wc`.  Integrating the resulting
    loading gives thrust and power coefficients, which are inverted for a new
    :math:`\zeta`.  It converges in a handful of iterations.

    Because the wake is helical and the blade count finite, Prandtl's tip-loss
    factor appears directly in the circulation -- this is a *design* constraint
    here, not an after-the-fact correction.

    With ``compressible=True`` (the default) the section's angle of attack and
    drag are recomputed at each station's local Mach number, so the blade is
    designed to *achieve* ``design_cl`` rather than to achieve it only in
    incompressible flow.  Turning it off reproduces the classical
    incompressible method, and the difference between the two is worth looking
    at on a fast propeller: it is around 17% of thrust on a light-aircraft
    cruise design.

    Requires ``v_inf > 0``: the formulation is built on the advance ratio, and
    a static design point has no minimum-induced-loss solution in this form.
    """
    if (thrust is None) == (power is None):
        raise ValueError("specify exactly one of thrust= or power=")
    if v_inf <= 0.0:
        raise ValueError("Adkins-Liebeck design needs a forward speed; "
                         "size a static propeller with momentum_sizing() instead")
    if design_cl is None and design_alpha is None:
        raise ValueError("give the section design point: design_cl or design_alpha")

    rho, mu = air.density, air.viscosity
    omega = rpm_to_rad_s(rpm)
    lam = v_inf / (omega * radius)                     # advance ratio V/(omega R)

    polar = get_airfoil(airfoil) if isinstance(airfoil, str) else airfoil
    alpha_d, cl_d, cd_d = _design_point(polar, design_cl, design_alpha)

    xi = np.linspace(hub_radius_frac, 0.999, n_stations)
    thickness = thickness_root + (thickness_tip - thickness_root) * xi
    kappa = float(polar.kappa)

    # Per-station section state; scalar until the compressible pass refines it.
    alpha_station = np.full_like(xi, alpha_d)
    eps = np.full_like(xi, cd_d / max(cl_d, 1e-9))
    x = xi / lam                                        # local speed ratio
    area = math.pi * radius ** 2

    tc = 2.0 * thrust / (rho * v_inf ** 2 * area) if thrust is not None else None
    pc = 2.0 * power / (rho * v_inf ** 3 * area) if power is not None else None

    zeta = 0.0
    converged = False
    iterations = 0
    phi = np.zeros_like(xi)
    wc = np.zeros_like(xi)
    a_ax = np.zeros_like(xi)
    a_sw = np.zeros_like(xi)

    for iterations in range(1, max_iter + 1):
        # Flow angle from the rigid-helicoid wake condition.
        phi_t = math.atan(lam * (1.0 + 0.5 * zeta))
        phi = np.arctan((1.0 + 0.5 * zeta) / np.maximum(x, 1e-9))

        # Prandtl tip loss, evaluated at the tip flow angle as the method requires.
        f = 0.5 * n_blades * (1.0 - xi) / max(math.sin(phi_t), 1e-9)
        F = (2.0 / math.pi) * np.arccos(np.clip(np.exp(-np.maximum(f, 0.0)), 0.0, 1.0))
        F = np.maximum(F, 1e-6)

        G = F * x * np.cos(phi) * np.sin(phi)

        a_ax = 0.5 * zeta * np.cos(phi) ** 2 * (1.0 - eps * np.tan(phi))
        a_sw = (0.5 * zeta / np.maximum(x, 1e-9)) * np.cos(phi) * np.sin(phi) \
            * (1.0 + eps / np.maximum(np.tan(phi), 1e-9))

        if compressible:
            # Local Mach follows from the velocity triangle alone -- no chord
            # needed -- so the section state can be refined inside the loop.
            w_local = v_inf * (1.0 + a_ax) / np.maximum(np.sin(phi), 1e-9)
            mach = np.clip(w_local / air.sound_speed, 0.0, 0.92)
            alpha_station, cd_station = _compressible_section(
                polar, cl_d, mach, thickness, kappa)
            eps = cd_station / max(cl_d, 1e-9)

        # Betz's condition fixes the product of chord and resultant velocity.
        wc = 4.0 * math.pi * lam * G * v_inf * radius * zeta / max(cl_d * n_blades, 1e-12)

        # Loading integrals (Adkins & Liebeck eqs. 11-14).
        i1 = 4.0 * xi * G * (1.0 - eps * np.tan(phi))
        i2 = lam * (i1 / (2.0 * np.maximum(xi, 1e-9))) \
            * (1.0 + eps / np.maximum(np.tan(phi), 1e-9)) * np.sin(phi) * np.cos(phi)
        j1 = 4.0 * xi * G * (1.0 + eps / np.maximum(np.tan(phi), 1e-9))
        j2 = 0.5 * j1 * (1.0 - eps * np.tan(phi)) * np.cos(phi) ** 2

        I1, I2 = np.trapezoid(i1, xi), np.trapezoid(i2, xi)
        J1, J2 = np.trapezoid(j1, xi), np.trapezoid(j2, xi)

        if tc is not None:
            half = I1 / (2.0 * I2) if abs(I2) > 1e-12 else 0.0
            disc = half ** 2 - tc / I2 if abs(I2) > 1e-12 else -1.0
            if disc < 0.0:
                break                    # this diameter cannot make that thrust
            zeta_new = half - math.sqrt(disc)
        else:
            half = J1 / (2.0 * J2) if abs(J2) > 1e-12 else 0.0
            disc = half ** 2 + pc / J2 if abs(J2) > 1e-12 else -1.0
            if disc < 0.0:
                break
            zeta_new = -half + math.sqrt(disc)

        if abs(zeta_new - zeta) < tol:
            zeta = zeta_new
            converged = True
            break
        zeta = zeta_new

    # Final geometry from the converged state.
    w_resultant = v_inf * (1.0 + a_ax) / np.maximum(np.sin(phi), 1e-9)
    chord = wc / np.maximum(w_resultant, 1e-9)
    twist = alpha_station + phi
    reynolds = rho * np.maximum(wc, 0.0) / mu

    tc_out = I1 * zeta - I2 * zeta ** 2
    pc_out = J1 * zeta + J2 * zeta ** 2
    thrust_out = 0.5 * rho * v_inf ** 2 * area * tc_out
    power_out = 0.5 * rho * v_inf ** 3 * area * pc_out

    return BladeDesign(
        radius=radius, hub_radius=hub_radius_frac * radius, n_blades=n_blades,
        rpm=rpm, v_inf=v_inf, x=xi, chord=chord, twist=twist, phi=phi,
        a_axial=a_ax, a_swirl=a_sw, reynolds=reynolds, zeta=zeta,
        thrust=float(thrust_out), power=float(power_out),
        efficiency=float(tc_out / pc_out) if pc_out > 1e-12 else float("nan"),
        design_cl=cl_d, design_alpha=float(np.mean(alpha_station)),
        converged=converged, iterations=iterations,
        meta={"advance_ratio": lam, "epsilon": float(np.mean(eps)),
              "design_cd": cd_d, "compressible": compressible,
              "alpha_station": alpha_station, "mach": (
                  np.clip(w_resultant / air.sound_speed, 0.0, 0.92)),
              "airfoil": getattr(polar, "name", str(airfoil)),
              "air": air.describe()},
    )


def verify_design(design: BladeDesign, air: AirState, *, airfoil: str,
                  thickness_root: float, thickness_tip: float,
                  n_elements: int = 60) -> dict[str, Any]:
    """Run a designed propeller back through the BEMT solver.

    The design method and the analysis are independent -- different equations,
    different assumptions -- so agreement between them is meaningful and
    disagreement is informative.  Expect a few percent; much more than that and
    the design is outside the method's comfortable range (usually too heavily
    loaded, or too close to stall at the design Cl).
    """
    from .bemt.core import OperatingPoint, SolverOptions
    from .bemt.solver import PropellerSolver

    geometry = design.to_geometry(airfoil=airfoil, thickness_root=thickness_root,
                                  thickness_tip=thickness_tip)
    solver = PropellerSolver(geometry, SolverOptions(n_elements=n_elements))
    result = solver.solve(OperatingPoint(rpm=design.rpm, v_inf=design.v_inf, air=air))
    return {
        "geometry": geometry, "result": result,
        "design_thrust": design.thrust, "bemt_thrust": result.thrust,
        "thrust_error": (result.thrust - design.thrust) / max(abs(design.thrust), 1e-9),
        "design_power": design.power, "bemt_power": result.power,
        "power_error": (result.power - design.power) / max(abs(design.power), 1e-9),
        "design_efficiency": design.efficiency, "bemt_efficiency": result.efficiency,
    }


# ---------------------------------------------------------------------------
# 3. Choosing the diameter: a minimum-induced-loss design at every candidate
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SizingCandidate:
    """One diameter in the trade, designed and judged."""

    diameter: float
    hub_ratio: float
    tip_mach: float
    status: str                     # "ok" or why this diameter was rejected
    efficiency: float               # Adkins-Liebeck design efficiency (nan if failed)
    froude_efficiency: float        # actuator-disk limit at the design's thrust
    thrust: float
    activity_factor: float
    beta75_deg: float
    design: "BladeDesign | None" = None


@dataclass(slots=True)
class SizingResult:
    status: str                     # "ok" | "infeasible"
    message: str
    candidates: list[SizingCandidate]
    best: SizingCandidate | None
    verification: dict[str, Any] | None
    limits: dict[str, float]


def size_propeller(*, power: float, prop_rpm: float, v_inf: float, air: AirState,
                   n_blades: int, airfoil: str, design_cl: float,
                   spinner_diameter: float, thickness_root: float, thickness_tip: float,
                   max_diameter: float, tip_mach_limit: float,
                   n_candidates: int = 21) -> SizingResult:
    """Pick the diameter, and design the blade, that best absorbs ``power``.

    Momentum theory says a bigger disk is always better, so on its own it
    cannot choose a propeller.  Here every candidate diameter gets its own
    Adkins-Liebeck minimum-induced-loss blade -- with compressible section
    drag, so a tip running into drag divergence pays for it -- and the
    candidates are compared on the efficiency of their *best possible* blade.
    The range runs from half the largest allowed diameter up to the smaller
    of ``max_diameter`` (ground clearance) and the diameter at which the
    helical tip Mach number reaches ``tip_mach_limit``.  The winner is then
    analysed by the independent BEMT solver as a check on the design.
    """
    omega = rpm_to_rad_s(prop_rpm)
    a = air.sound_speed
    vt2 = (tip_mach_limit * a) ** 2 - v_inf ** 2
    limits = {"max_diameter": max_diameter, "tip_mach_limit": tip_mach_limit}
    if vt2 <= 0.0:
        return SizingResult("infeasible", "The airspeed alone exceeds the tip Mach limit; "
                            "no propeller can satisfy it.", [], None, None, limits)
    d_tip = 2.0 * math.sqrt(vt2) / omega
    limits["tip_mach_diameter"] = d_tip
    d_hi = min(max_diameter, d_tip)
    d_lo = max(0.5 * d_hi, spinner_diameter / 0.45)
    limits["diameter_low"] = d_lo
    limits["diameter_high"] = d_hi
    if d_lo >= d_hi:
        return SizingResult(
            "infeasible",
            f"The spinner ({spinner_diameter / 0.0254:.1f} in) is too large for the "
            f"largest allowed diameter ({d_hi / 0.0254:.1f} in).", [], None, None, limits)

    candidates: list[SizingCandidate] = []
    for d in np.linspace(d_lo, d_hi, n_candidates):
        hub = spinner_diameter / d
        tip = math.hypot(omega * d / 2.0, v_inf) / a
        try:
            des = adkins_liebeck_design(
                d / 2.0, n_blades, prop_rpm, v_inf, air=air, airfoil=airfoil,
                hub_radius_frac=hub, thickness_root=thickness_root,
                thickness_tip=thickness_tip, power=power, design_cl=design_cl)
        except ValueError as exc:
            candidates.append(SizingCandidate(d, hub, tip, f"design_failed: {exc}",
                                              math.nan, math.nan, math.nan, math.nan, math.nan))
            continue
        area = math.pi * d * d / 4.0
        w = induced_velocity(des.thrust, v_inf, area, air.density)
        froude = v_inf / (v_inf + w) if (v_inf + w) > 0 else math.nan
        status = "ok"
        if not des.converged:
            status = "design_not_converged"
        elif not (np.all(np.isfinite(des.chord)) and np.all(des.chord > 0.0)):
            status = "design_invalid_chord"
        elif float(np.max(des.chord)) > 0.35 * d:
            status = "blade_too_wide"
        candidates.append(SizingCandidate(
            d, hub, tip, status, des.efficiency, froude, des.thrust,
            des.activity_factor, math.degrees(float(np.interp(0.75, des.x, des.twist))),
            des))

    good = [c for c in candidates if c.status == "ok" and math.isfinite(c.efficiency)]
    if not good:
        reasons = sorted({c.status for c in candidates})
        return SizingResult("infeasible", "No candidate diameter produced a valid design ("
                            + ", ".join(reasons) + ").", candidates, None, None, limits)
    best = max(good, key=lambda c: c.efficiency)
    check = verify_design(best.design, air, airfoil=airfoil, thickness_root=thickness_root,
                          thickness_tip=thickness_tip)
    binding = ("tip Mach limit" if d_tip < max_diameter else "maximum diameter")
    at_top = abs(best.diameter - d_hi) < 1e-9
    message = (f"Best diameter {best.diameter / 0.0254:.1f} in"
               + (f", set by the {binding}" if at_top else ", an interior optimum") + ".")
    return SizingResult("ok", message, candidates, best, check, limits)


__all__ = ["DiskSizing", "BladeDesign", "SizingCandidate", "SizingResult",
           "momentum_sizing", "diameter_sweep", "induced_velocity", "drag_from_weight",
           "adkins_liebeck_design", "verify_design", "size_propeller"]
