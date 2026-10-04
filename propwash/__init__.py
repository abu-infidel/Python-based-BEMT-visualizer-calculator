"""Propwash -- a full-size aircraft propeller calculator.

Every calculation starts from a case: a flat set of inputs with no defaults.
::

    from propwash import blank_case, run

    case = blank_case("analysis")       # every field None -- fill them in
    case.update({...})
    report = run(case)                  # validated, computed, never made up
    report.save("my_report.txt")        # structured text, docs/report-format/

Front ends: ``python -m propwash gui`` (desktop), ``propwash.colab.launch()``
(notebook) and the ``propwash`` command line.  Submodules that need Qt, Plotly
or CUDA are imported lazily, so ``import propwash`` costs only NumPy.
"""

from __future__ import annotations

from typing import Any

from .airfoil import AIRFOIL_LIBRARY, AnalyticPolar, TablePolar, get_airfoil, list_airfoils
from .analysis import run
from .atmosphere import AirState, isa
from .bemt.core import BEMTResult, OperatingPoint, SolverOptions
from .bemt.solver import PropellerSolver, solve, solve_grid, spanwise_frame
from .bemt.sweep import SweepResult, envelope_map, j_sweep, pitch_rpm_map, rpm_sweep
from .case import FIELDS, Case, CaseError, blank_case, parse_case, required_keys
from .engine import PistonEngine
from .geometry import (PLANFORMS, BladeGeometry, Distribution, full_size_blade,
                       list_planforms)
from .report import Report, ReportFormatError, parse_report
from .sizing import (BladeDesign, SizingResult, adkins_liebeck_design, size_propeller,
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
    "run_validation": ("propwash.validation", "run_validation"),
    "launch_gui": ("propwash.gui", "launch"),
    "launch_colab": ("propwash.colab", "launch"),
}


def __getattr__(name: str) -> Any:
    """Lazy re-export so heavy optional dependencies stay optional."""
    if name in _LAZY:
        import importlib
        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module 'propwash' has no attribute {name!r}")


__all__ = [
    "run", "blank_case", "parse_case", "required_keys", "Case", "CaseError", "FIELDS",
    "Report", "parse_report", "ReportFormatError",
    "AIRFOIL_LIBRARY", "AnalyticPolar", "TablePolar", "get_airfoil", "list_airfoils",
    "AirState", "isa", "BEMTResult", "OperatingPoint", "SolverOptions",
    "PropellerSolver", "solve", "solve_grid", "spanwise_frame", "SweepResult",
    "envelope_map", "j_sweep", "pitch_rpm_map", "rpm_sweep", "PistonEngine",
    "PLANFORMS", "BladeGeometry", "Distribution", "full_size_blade", "list_planforms",
    "BladeDesign", "SizingResult", "adkins_liebeck_design", "size_propeller",
    "verify_design", "PROJECT_NAME", "PROJECT_TAGLINE", "__version__", *_LAZY,
]
