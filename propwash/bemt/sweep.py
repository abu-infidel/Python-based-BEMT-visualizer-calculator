"""Parameter sweeps: the part that actually needs a GPU.

A single BEMT solve is microseconds.  The interesting questions are all
*maps* -- how does thrust vary over pitch and RPM, where is the efficiency
island, what speed will the aircraft settle at -- and those are thousands to
millions of solves.  Everything here funnels into one batched call so the
backend can put the whole grid on the device at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from ..atmosphere import SEA_LEVEL, AirState
from ..geometry import BladeGeometry
from ..motor import MatchPoint, MotorSpec, match_rpm
from ..units import rpm_to_rad_s
from .core import SolverOptions
from .solver import PropellerSolver


@dataclass(slots=True)
class SweepResult:
    """A grid of solved operating points plus the axes that generated it."""

    axes: dict[str, np.ndarray] = field(default_factory=dict)
    axis_order: tuple[str, ...] = ()
    data: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.axes[k].size for k in self.axis_order)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def _scalar_fields(self) -> dict[str, np.ndarray]:
        """Arrays with exactly one value per operating point.

        Backends also return bookkeeping entries -- the backend's name, the
        elapsed time -- and the spanwise arrays carry an extra station axis, so
        anything that is not a same-shaped array is filtered out here rather
        than at every call site.
        """
        shape = self.shape
        return {k: v for k, v in self.data.items()
                if isinstance(v, np.ndarray) and v.shape == shape}

    def field_names(self) -> list[str]:
        return sorted(self._scalar_fields())

    def best(self, key: str = "efficiency") -> dict[str, float]:
        """Locate the maximum of ``key`` and report every field there."""
        arr = np.nan_to_num(np.asarray(self.data[key], dtype=float), nan=-np.inf)
        idx = np.unravel_index(int(np.argmax(arr)), arr.shape)
        out = {name: float(np.asarray(self.axes[name]).ravel()[idx[i]])
               for i, name in enumerate(self.axis_order)}
        for name, val in self._scalar_fields().items():
            out[name] = float(val[idx])
        return out

    def to_frame(self):
        """Flatten to a tidy pandas DataFrame (one row per operating point)."""
        import pandas as pd
        grids = np.meshgrid(*[self.axes[k] for k in self.axis_order], indexing="ij")
        cols = {k: g.ravel() for k, g in zip(self.axis_order, grids)}
        for name, val in self._scalar_fields().items():
            cols[name] = val.ravel()
        return pd.DataFrame(cols)


def _solver(geometry: BladeGeometry, options: SolverOptions | None,
            backend: str) -> PropellerSolver:
    return PropellerSolver(geometry, options or SolverOptions(), backend)


# ---------------------------------------------------------------------------
# One-dimensional sweeps
# ---------------------------------------------------------------------------

def j_sweep(geometry: BladeGeometry, rpm: float = 6000.0, j_max: float = 1.2,
            n: int = 60, air: AirState | None = None,
            options: SolverOptions | None = None, backend: str = "auto") -> SweepResult:
    """The textbook propeller chart: CT, CP and eta against advance ratio.

    Sweeping J at fixed RPM (rather than fixed speed) is the convention, because
    the coefficients then collapse onto a single curve independent of scale.
    """
    air = air or SEA_LEVEL
    solver = _solver(geometry, options, backend)
    j = np.linspace(0.0, j_max, n)
    v = j * (rpm / 60.0) * geometry.diameter
    data = solver.solve_many(np.full(n, float(rpm)), v, air)
    return SweepResult(axes={"j": j}, axis_order=("j",), data=data,
                       meta={"rpm": rpm, "blade": geometry.name,
                             "backend": data.get("backend", "?")})


def rpm_sweep(geometry: BladeGeometry, rpm_range: tuple[float, float] = (1000.0, 12000.0),
              v_inf: float = 0.0, n: int = 80, air: AirState | None = None,
              options: SolverOptions | None = None, backend: str = "auto") -> SweepResult:
    """Thrust and power against shaft speed at fixed flight speed."""
    air = air or SEA_LEVEL
    solver = _solver(geometry, options, backend)
    rpm = np.linspace(rpm_range[0], rpm_range[1], n)
    data = solver.solve_many(rpm, np.full(n, float(v_inf)), air)
    return SweepResult(axes={"rpm": rpm}, axis_order=("rpm",), data=data,
                       meta={"v_inf": v_inf, "blade": geometry.name})


# ---------------------------------------------------------------------------
# Two-dimensional maps
# ---------------------------------------------------------------------------

def pitch_rpm_map(geometry: BladeGeometry, pitch_deg: np.ndarray, rpm: np.ndarray,
                  v_inf: float = 0.0, air: AirState | None = None,
                  options: SolverOptions | None = None,
                  backend: str = "auto") -> SweepResult:
    """Thrust/power/efficiency over the collective-pitch x RPM plane.

    Pitch changes the *geometry*, so unlike an RPM sweep this cannot be a single
    batched call -- the blade is rediscretised per pitch row.  Each row is still
    one batched call, which keeps the GPU busy.
    """
    air = air or SEA_LEVEL
    options = options or SolverOptions()
    pitch = np.asarray(pitch_deg, dtype=float)
    rpm_arr = np.asarray(rpm, dtype=float)

    solver = _solver(geometry, options, backend)
    rows: list[dict[str, np.ndarray]] = []
    for p in pitch:
        solver.geometry = replace(geometry, pitch_offset=math.radians(float(p)))
        rows.append(solver.solve_many(rpm_arr, np.full(rpm_arr.size, float(v_inf)), air))

    data: dict[str, np.ndarray] = {}
    for key, sample in rows[0].items():
        if not isinstance(sample, np.ndarray):
            continue          # backend name, elapsed time, and other bookkeeping
        try:
            data[key] = np.stack([np.asarray(r[key]) for r in rows])
        except ValueError:
            continue

    return SweepResult(axes={"pitch_deg": pitch, "rpm": rpm_arr},
                       axis_order=("pitch_deg", "rpm"), data=data,
                       meta={"v_inf": v_inf, "blade": geometry.name})


def envelope_map(geometry: BladeGeometry, rpm: np.ndarray, v_inf: np.ndarray,
                 air: AirState | None = None, options: SolverOptions | None = None,
                 backend: str = "auto") -> SweepResult:
    """The full RPM x airspeed operating envelope in a single batched call."""
    air = air or SEA_LEVEL
    solver = _solver(geometry, options, backend)
    rpm_g, v_g = np.meshgrid(np.asarray(rpm, dtype=float),
                             np.asarray(v_inf, dtype=float), indexing="ij")
    data = solver.solve_many(rpm_g, v_g, air)
    return SweepResult(axes={"rpm": np.asarray(rpm, dtype=float),
                             "v_inf": np.asarray(v_inf, dtype=float)},
                       axis_order=("rpm", "v_inf"), data=data,
                       meta={"blade": geometry.name})


# ---------------------------------------------------------------------------
# Coupled solves: where does the system actually settle?
# ---------------------------------------------------------------------------

def match_operating_point(geometry: BladeGeometry, motor: MotorSpec,
                          v_inf: float = 0.0, throttle: float = 1.0,
                          air: AirState | None = None,
                          options: SolverOptions | None = None,
                          backend: str = "auto") -> MatchPoint:
    """Solve the propeller/motor torque balance.

    This is the answer to "how fast does it spin?".  The propeller's torque
    demand and the motor's torque supply cross at exactly one RPM, and that
    crossing is what the 3-D view animates.
    """
    air = air or SEA_LEVEL
    solver = _solver(geometry, options, backend)

    def prop_torque(rpm_arr: np.ndarray) -> np.ndarray:
        rpm_arr = np.atleast_1d(np.asarray(rpm_arr, dtype=float))
        out = solver.solve_many(np.maximum(rpm_arr, 1.0),
                                np.full(rpm_arr.size, float(v_inf)), air)
        return np.maximum(np.asarray(out["torque"], dtype=float).ravel(), 0.0)

    rpm, converged = match_rpm(prop_torque, motor, throttle, air=air)

    if rpm <= 0.0:
        return MatchPoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, throttle)

    final = solver.solve_many(np.array([rpm]), np.array([float(v_inf)]), air)
    thrust = float(np.asarray(final["thrust"]).ravel()[0])
    torque = float(np.asarray(final["torque"]).ravel()[0])
    shaft = torque * rpm_to_rad_s(rpm)

    if hasattr(motor, "fuel_flow_gph"):          # piston engine
        extras = {
            "fuel_gph": float(np.asarray(motor.fuel_flow_gph(rpm, throttle, air)).ravel()[0]),
            "fuel_lph": float(np.asarray(motor.fuel_flow_lph(rpm, throttle, air)).ravel()[0]),
            "power_fraction": float(np.asarray(
                motor.power_fraction(rpm, throttle, air)).ravel()[0]),
            "available_power": float(np.asarray(
                motor.shaft_power(rpm, throttle, air)).ravel()[0]),
        }
        p_in = extras["available_power"]
        return MatchPoint(
            rpm=rpm, thrust=thrust, torque=torque, shaft_power=shaft,
            electrical_power=p_in, current=0.0,
            motor_efficiency=1.0,
            system_efficiency=(thrust * v_inf / p_in) if p_in > 1e-9 else 0.0,
            converged=converged, throttle=throttle, extras=extras,
        )

    current = float(np.minimum(motor.current_at(np.array([rpm]), throttle),
                               motor.max_current)[0])
    p_elec = float(motor.electrical_power(np.array([rpm]), throttle)[0])
    return MatchPoint(
        rpm=rpm, thrust=thrust, torque=torque, shaft_power=shaft,
        electrical_power=p_elec, current=current,
        motor_efficiency=(shaft / p_elec) if p_elec > 1e-9 else 0.0,
        system_efficiency=(thrust * v_inf / p_elec) if p_elec > 1e-9 else 0.0,
        converged=converged, throttle=throttle,
    )


def thrust_required_speed(geometry: BladeGeometry, motor: MotorSpec,
                          drag_fn: Callable[[float], float] | None = None,
                          wing_area: float = 0.22, span: float = 1.20,
                          oswald: float = 0.85, cd0: float = 0.035,
                          mass: float = 1.2, throttle: float = 1.0,
                          air: AirState | None = None, v_min: float = 3.0,
                          v_max: float = 80.0, n_scan: int = 20, n_iter: int = 32,
                          options: SolverOptions | None = None,
                          backend: str = "auto") -> dict[str, Any]:
    """Find the level-flight speed where propeller thrust equals airframe drag.

    This closes the loop from blade geometry all the way to aircraft speed:
    change the pitch and the trim speed moves.

    The drag model is the standard two-term polar,

        D = q S C_D0  +  W^2 / (q pi e b^2)

    which is U-shaped -- induced drag diverges as speed falls, parasite drag
    grows as it rises.  Thrust available falls monotonically with speed, so the
    two curves can cross *twice*: once on the back side of the power curve and
    once at the genuine cruise trim.  Bisecting blindly from zero finds neither,
    because at V = 0 required thrust is infinite.  So the excess-thrust curve is
    scanned first and the bracket is taken at the *last* positive-to-negative
    crossing, which is the stable high-speed trim point.

    Pass ``drag_fn`` for a real drag polar; the built-in model is only meant to
    be plausible.
    """
    air = air or SEA_LEVEL
    weight = mass * 9.80665
    rho = air.density

    def drag(v: float) -> float:
        if drag_fn is not None:
            return float(drag_fn(v))
        q = 0.5 * rho * v * v
        if q < 1e-6:
            return float("inf")
        parasite = q * wing_area * cd0
        induced = weight ** 2 / (q * math.pi * oswald * span ** 2)
        return parasite + induced

    cache: dict[float, MatchPoint] = {}

    def excess(v: float) -> float:
        if v not in cache:
            cache[v] = match_operating_point(geometry, motor, v_inf=v,
                                             throttle=throttle, air=air,
                                             options=options, backend=backend)
        d = drag(v)
        return (cache[v].thrust - d) if math.isfinite(d) else -math.inf

    speeds = np.linspace(max(v_min, 0.5), v_max, max(n_scan, 4))
    values = [excess(float(v)) for v in speeds]

    crossing = None
    for i in range(len(speeds) - 1):
        if values[i] > 0.0 >= values[i + 1]:
            crossing = (float(speeds[i]), float(speeds[i + 1]))
    if crossing is None:
        best = int(np.argmax(values))
        return {
            "speed": float(speeds[best]), "converged": False,
            "reason": ("never enough thrust to fly level"
                       if values[best] <= 0.0 else
                       f"still climbing at v_max = {v_max:.0f} m/s"),
            "rpm": cache[float(speeds[best])].rpm,
            "thrust": cache[float(speeds[best])].thrust,
            "drag": drag(float(speeds[best])),
        }

    lo, hi = crossing
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    v_trim = 0.5 * (lo + hi)
    mp = cache[min(cache, key=lambda v: abs(v - v_trim))]
    return {
        "speed": v_trim, "converged": True, "rpm": mp.rpm, "thrust": mp.thrust,
        "drag": drag(v_trim), "current": mp.current,
        "electrical_power": mp.electrical_power,
        "advance_ratio": v_trim / max(mp.rpm / 60.0 * geometry.diameter, 1e-9),
    }


__all__ = ["SweepResult", "j_sweep", "rpm_sweep", "pitch_rpm_map", "envelope_map",
           "match_operating_point", "thrust_required_speed"]
