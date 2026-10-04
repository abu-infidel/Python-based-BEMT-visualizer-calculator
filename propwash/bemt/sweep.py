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
from typing import Any

import numpy as np

from ..atmosphere import AirState
from ..geometry import BladeGeometry
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

def j_sweep(geometry: BladeGeometry, rpm: float, j_max: float, air: AirState,
            n: int = 60,
            options: SolverOptions | None = None, backend: str = "auto") -> SweepResult:
    """The textbook propeller chart: CT, CP and eta against advance ratio.

    Sweeping J at fixed RPM (rather than fixed speed) is the convention, because
    the coefficients then collapse onto a single curve independent of scale.
    """
    solver = _solver(geometry, options, backend)
    j = np.linspace(0.0, j_max, n)
    v = j * (rpm / 60.0) * geometry.diameter
    data = solver.solve_many(np.full(n, float(rpm)), v, air)
    return SweepResult(axes={"j": j}, axis_order=("j",), data=data,
                       meta={"rpm": rpm, "blade": geometry.name,
                             "backend": data.get("backend", "?")})


def rpm_sweep(geometry: BladeGeometry, rpm_range: tuple[float, float], v_inf: float,
              air: AirState, n: int = 80,
              options: SolverOptions | None = None, backend: str = "auto") -> SweepResult:
    """Thrust and power against shaft speed at fixed flight speed."""
    solver = _solver(geometry, options, backend)
    rpm = np.linspace(rpm_range[0], rpm_range[1], n)
    data = solver.solve_many(rpm, np.full(n, float(v_inf)), air)
    return SweepResult(axes={"rpm": rpm}, axis_order=("rpm",), data=data,
                       meta={"v_inf": v_inf, "blade": geometry.name})


# ---------------------------------------------------------------------------
# Two-dimensional maps
# ---------------------------------------------------------------------------

def pitch_rpm_map(geometry: BladeGeometry, pitch_deg: np.ndarray, rpm: np.ndarray,
                  v_inf: float, air: AirState,
                  options: SolverOptions | None = None,
                  backend: str = "auto") -> SweepResult:
    """Thrust/power/efficiency over the collective-pitch x RPM plane.

    Pitch changes the *geometry*, so unlike an RPM sweep this cannot be a single
    batched call -- the blade is rediscretised per pitch row.  Each row is still
    one batched call, which keeps the GPU busy.
    """
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
                 air: AirState, options: SolverOptions | None = None,
                 backend: str = "auto") -> SweepResult:
    """The full RPM x airspeed operating envelope in a single batched call."""
    solver = _solver(geometry, options, backend)
    rpm_g, v_g = np.meshgrid(np.asarray(rpm, dtype=float),
                             np.asarray(v_inf, dtype=float), indexing="ij")
    data = solver.solve_many(rpm_g, v_g, air)
    return SweepResult(axes={"rpm": np.asarray(rpm, dtype=float),
                             "v_inf": np.asarray(v_inf, dtype=float)},
                       axis_order=("rpm", "v_inf"), data=data,
                       meta={"blade": geometry.name})


__all__ = ["SweepResult", "j_sweep", "rpm_sweep", "pitch_rpm_map", "envelope_map"]
