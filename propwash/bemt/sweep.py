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
from typing import Any, Callable, Sequence

import numpy as np

from ..atmosphere import SEA_LEVEL, AirState
from ..geometry import BladeGeometry
from ..motor import MatchPoint, MotorSpec, match_rpm
from ..units import rpm_to_rad_s
from .core import OperatingPoint, SolverOptions
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

    def field_names(self) -> list[str]:
        return sorted(k for k, v in self.data.items() if v.shape == self.shape)

    def best(self, key: str = "efficiency") -> dict[str, float]:
        """Locate the maximum of ``key`` and report every field there."""
        arr = np.nan_to_num(self.data[key], nan=-np.inf)
        flat = int(np.argmax(arr))
        idx = np.unravel_index(flat, arr.shape)
        out = {name: float(np.asarray(self.axes[name]).ravel()[idx[i]])
               for i, name in enumerate(self.axis_order)}
        for name, val in self.data.items():
            if val.shape == arr.shape:
                out[name] = float(val[idx])
        return out

    def to_frame(self):
        """Flatten to a tidy pandas DataFrame (one row per operating point)."""
        import pandas as pd
        grids = np.meshgrid(*[self.axes[k] for k in self.axis_order], indexing="ij")
        cols = {k: g.ravel() for k, g in zip(self.axis_order, grids)}
        for name, val in self.data.items():
            if val.shape == self.shape:
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
    for key in rows[0]:
        try:
            data[key] = np.stack([r[key] for r in rows])
        except (ValueError, TypeError):
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

    rpm, converged = match_rpm(prop_torque, motor, throttle)

    if rpm <= 0.0:
        return MatchPoint(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, throttle)

    final = solver.solve_many(np.array([rpm]), np.array([float(v_inf)]), air)
    thrust = float(np.asarray(final["thrust"]).ravel()[0])
    torque = float(np.asarray(final["torque"]).ravel()[0])
    shaft = torque * rpm_to_rad_s(rpm)
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
                          drag_area: float = 0.010, mass: float = 1.2,
                          throttle: float = 1.0, air: AirState | None = None,
                          v_max: float = 80.0, n_iter: int = 40,
                          options: SolverOptions | None = None,
                          backend: str = "auto") -> dict[str, float]:
    """Find the level-flight speed where propeller thrust equals airframe drag.

    This closes the loop all the way from blade geometry to aircraft speed: pick
    a pitch, and the trim speed moves.  ``drag_fn`` may be supplied for a real
    drag polar; otherwise a simple ``D = q * C_D A`` flat-plate model is used,
    with an induced-drag term from the wing loading.
    """
    air = air or SEA_LEVEL

    def drag(v: float) -> float:
        if drag_fn is not None:
            return float(drag_fn(v))
        q = 0.5 * air.density * v * v
        parasite = q * drag_area
        # Induced drag at level flight: lift equals weight.
        weight = mass * 9.80665
        induced = (weight ** 2) / max(q * math.pi * 0.85 * 1.2, 1e-6) * 1e-3
        return parasite + induced

    lo, hi = 0.0, float(v_max)
    last: MatchPoint | None = None

    def excess(v: float) -> tuple[float, MatchPoint]:
        mp = match_operating_point(geometry, motor, v_inf=v, throttle=throttle,
                                   air=air, options=options, backend=backend)
        return mp.thrust - drag(v), mp

    e_lo, mp_lo = excess(lo)
    e_hi, mp_hi = excess(hi)
    if e_lo <= 0.0:
        return {"speed": 0.0, "converged": False, "thrust": mp_lo.thrust,
                "drag": drag(0.0), "rpm": mp_lo.rpm}
    if e_hi > 0.0:
        return {"speed": float(v_max), "converged": False, "thrust": mp_hi.thrust,
                "drag": drag(v_max), "rpm": mp_hi.rpm}

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        e_mid, last = excess(mid)
        if e_mid > 0.0:
            lo = mid
        else:
            hi = mid

    v_trim = 0.5 * (lo + hi)
    mp = last if last is not None else mp_lo
    return {
        "speed": v_trim, "converged": True, "rpm": mp.rpm, "thrust": mp.thrust,
        "drag": drag(v_trim), "current": mp.current,
        "electrical_power": mp.electrical_power,
        "advance_ratio": v_trim / max(mp.rpm / 60.0 * geometry.diameter, 1e-9),
    }


__all__ = ["SweepResult", "j_sweep", "rpm_sweep", "pitch_rpm_map", "envelope_map",
           "match_operating_point", "thrust_required_speed"]
