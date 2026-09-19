"""High-level solver API.

:class:`PropellerSolver` owns the expensive, reusable parts -- the discretised
blade and the baked polar tables -- so that sweeping ten thousand operating
points does not re-tabulate the aerodynamics ten thousand times.  It also picks
the compute backend (CUDA if present, otherwise a CPU path) and falls back
without complaint when the GPU is missing.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..atmosphere import SEA_LEVEL, AirState
from ..geometry import BladeGeometry, BladeStations
from .core import (BEMTResult, OperatingPoint, SolverOptions, solve_batch,
                   solve_stations)


class PropellerSolver:
    """A blade plus its cached tables, ready to be asked many questions."""

    def __init__(self, geometry: BladeGeometry, options: SolverOptions | None = None,
                 backend: str = "auto") -> None:
        self.options = options or SolverOptions()
        self._geometry = geometry
        self._stations: BladeStations | None = None
        self._tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._backend_name = backend
        self._backend: Any = None

    # -- geometry cache ----------------------------------------------------
    @property
    def geometry(self) -> BladeGeometry:
        return self._geometry

    @geometry.setter
    def geometry(self, value: BladeGeometry) -> None:
        self._geometry = value
        self.invalidate()

    @property
    def stations(self) -> BladeStations:
        if self._stations is None:
            self._stations = self._geometry.discretize(self.options.n_elements,
                                                       self.options.spacing)
        return self._stations

    @property
    def tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._tables is None:
            self._tables = self.stations.polar_tables(self.options.n_alpha_table)
        return self._tables

    def invalidate(self) -> None:
        """Drop cached discretisation -- call after editing the geometry."""
        self._stations = None
        self._tables = None
        if self._backend is not None:
            self._backend.invalidate()

    def set_pitch(self, pitch_offset: float) -> None:
        """Change collective pitch.

        Only the twist array shifts, so the polar tables stay valid; this is the
        cheap path the GUI slider uses at interactive rates.
        """
        self._geometry = replace(self._geometry, pitch_offset=pitch_offset)
        if self._stations is not None:
            # Collective is a rigid rotation of every section, so the chord,
            # thickness and polar tables all survive untouched.
            self._stations.twist = self._geometry.twist_at(self._stations.x)
        if self._backend is not None:
            self._backend.invalidate()

    # -- backend -----------------------------------------------------------
    @property
    def backend(self):
        if self._backend is None:
            from ..accel import get_backend
            self._backend = get_backend(self._backend_name)
        return self._backend

    def use_backend(self, name: str) -> None:
        from ..accel import get_backend
        self._backend_name = name
        self._backend = get_backend(name)

    # -- solving -----------------------------------------------------------
    def solve(self, op: OperatingPoint) -> BEMTResult:
        """One operating point, with full spanwise diagnostics."""
        return solve_stations(self.stations, op, self.options, self.tables)

    def solve_many(self, rpm, v_inf, air: AirState | None = None) -> dict[str, np.ndarray]:
        """Many operating points; dispatched to the fastest available backend."""
        return self.backend.solve_batch(
            self.stations, np.asarray(rpm, dtype=float), np.asarray(v_inf, dtype=float),
            air or SEA_LEVEL, self.options, self.tables,
        )

    def solve_many_cpu(self, rpm, v_inf, air: AirState | None = None) -> dict[str, np.ndarray]:
        """Force the NumPy reference path (used to validate the GPU kernels)."""
        return solve_batch(self.stations, np.asarray(rpm, dtype=float),
                           np.asarray(v_inf, dtype=float), air or SEA_LEVEL,
                           self.options, self.tables)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def solve(geometry: BladeGeometry, op: OperatingPoint | None = None,
          options: SolverOptions | None = None) -> BEMTResult:
    """One-shot solve without keeping a solver around."""
    return PropellerSolver(geometry, options).solve(op or OperatingPoint())


def solve_grid(geometry: BladeGeometry, rpm, v_inf, air: AirState | None = None,
               options: SolverOptions | None = None,
               backend: str = "auto") -> dict[str, np.ndarray]:
    """Solve a full grid of operating points."""
    rpm_g, v_g = np.broadcast_arrays(np.asarray(rpm, dtype=float),
                                     np.asarray(v_inf, dtype=float))
    return PropellerSolver(geometry, options, backend).solve_many(rpm_g, v_g, air)


SPANWISE_COLUMNS = ("x", "r", "chord", "twist_deg", "phi_deg", "alpha_deg", "cl", "cd",
                    "ld", "w", "mach", "reynolds", "dt_dr", "dq_dr", "loss_factor",
                    "v_axial_induced", "v_swirl_induced")


def spanwise_frame(result: BEMTResult, stations: BladeStations | None = None):
    """Spanwise state as a pandas DataFrame (falls back to a dict of arrays)."""
    deg = 180.0 / np.pi
    data = {
        "x": result.x, "r": result.r,
        "phi_deg": result.phi * deg, "alpha_deg": result.alpha * deg,
        "cl": result.cl, "cd": result.cd,
        "ld": result.cl / np.maximum(result.cd, 1e-9),
        "w": result.w, "mach": result.mach, "reynolds": result.reynolds,
        "dt_dr": result.dt_dr, "dq_dr": result.dq_dr,
        "loss_factor": result.loss_factor,
        "v_axial_induced": result.v_axial_induced,
        "v_swirl_induced": result.v_swirl_induced,
        "converged": result.converged,
    }
    if stations is not None:
        data["chord"] = stations.chord
        data["twist_deg"] = stations.twist * deg
    try:
        import pandas as pd
        return pd.DataFrame(data)
    except ImportError:  # pragma: no cover - pandas is optional
        return data


__all__ = ["PropellerSolver", "solve", "solve_grid", "spanwise_frame", "SPANWISE_COLUMNS"]
