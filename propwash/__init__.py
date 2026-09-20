"""Propwash -- a GPU-accelerated blade-element-momentum propeller laboratory.

Quick start::

    from propwash import get_preset, OperatingPoint, solve

    prop = get_preset("APC 10x5 (sport)")
    result = solve(prop, OperatingPoint(rpm=8000, v_inf=12.0))
    print(result.summary())

The GUI is the point of the package, though::

    python -m propwash gui          # desktop (Qt + OpenGL)
    python -m propwash colab        # in-notebook (ipywidgets + Plotly)
    python -m propwash bench        # what can this machine actually do?

Submodules are imported lazily: importing :mod:`propwash` pulls in NumPy and
nothing else, so a Colab cell that only wants the solver does not pay for Qt,
Plotly or CUDA.
"""

from __future__ import annotations

from typing import Any

from .airfoil import (AIRFOIL_LIBRARY, AnalyticPolar, TablePolar, get_airfoil,
                      list_airfoils)
from .atmosphere import SEA_LEVEL, AirState, isa
from .bemt.core import BEMTResult, OperatingPoint, SolverOptions
from .bemt.solver import PropellerSolver, solve, solve_grid, spanwise_frame
from .bemt.sweep import (SweepResult, envelope_map, j_sweep, match_operating_point,
                         pitch_rpm_map, rpm_sweep, thrust_required_speed)
from .engine import (ENGINE_PRESETS, DEFAULT_ENGINE, PistonEngine, get_engine,
                     list_engines)
from .geometry import (AIRCRAFT_PRESETS, DRONE_PRESETS, PROP_PRESETS,
                       BladeGeometry, Distribution, get_preset, list_presets)
from .motor import MOTOR_PRESETS, MotorSpec, get_motor, list_motors
from .sizing import (BladeDesign, DiskSizing, adkins_liebeck_design,
                     diameter_sweep, drag_from_weight, momentum_sizing,
                     verify_design)
from .version import PROJECT_NAME, PROJECT_TAGLINE, __version__

_LAZY = {
    "get_backend": ("propwash.accel", "get_backend"),
    "list_backends": ("propwash.accel", "list_backends"),
    "describe_environment": ("propwash.accel", "describe_environment"),
    "benchmark": ("propwash.accel.bench", "benchmark"),
    "build_propeller_mesh": ("propwash.mesh", "build_propeller_mesh"),
    "PropellerMesh": ("propwash.mesh", "PropellerMesh"),
    "propeller_figure": ("propwash.viz.plotly3d", "propeller_figure"),
    "launch_gui": ("propwash.gui", "launch"),
    "launch_colab": ("propwash.colab", "launch"),
}


def __getattr__(name: str) -> Any:
    """Lazy re-export so heavy optional dependencies stay optional."""
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        import importlib
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY))


__all__ = [
    "__version__", "PROJECT_NAME", "PROJECT_TAGLINE",
    # atmosphere
    "AirState", "isa", "SEA_LEVEL",
    # sections
    "AnalyticPolar", "TablePolar", "AIRFOIL_LIBRARY", "get_airfoil", "list_airfoils",
    # geometry
    "BladeGeometry", "Distribution", "PROP_PRESETS", "get_preset", "list_presets",
    "AIRCRAFT_PRESETS", "DRONE_PRESETS",
    # powerplants
    "MotorSpec", "MOTOR_PRESETS", "get_motor", "list_motors",
    "PistonEngine", "ENGINE_PRESETS", "DEFAULT_ENGINE", "get_engine", "list_engines",
    # sizing and design
    "DiskSizing", "BladeDesign", "momentum_sizing", "diameter_sweep",
    "drag_from_weight", "adkins_liebeck_design", "verify_design",
    # solver
    "OperatingPoint", "SolverOptions", "BEMTResult", "PropellerSolver",
    "solve", "solve_grid", "spanwise_frame",
    # sweeps
    "SweepResult", "j_sweep", "rpm_sweep", "pitch_rpm_map", "envelope_map",
    "match_operating_point", "thrust_required_speed",
    # lazy
    *sorted(_LAZY),
]
