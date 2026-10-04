"""Shared fixtures: the Cessna 172N reference propeller from the validation case."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from propwash.analysis import build_geometry                 # noqa: E402
from propwash.atmosphere import isa                          # noqa: E402
from propwash.bemt.core import OperatingPoint, SolverOptions  # noqa: E402
from propwash.case import parse_case                         # noqa: E402
from propwash.validation import C172N_CASE                   # noqa: E402


@pytest.fixture(scope="session")
def c172_case():
    return dict(C172N_CASE)


@pytest.fixture(scope="session")
def prop():
    geom, _ = build_geometry(parse_case(C172N_CASE))
    return geom


@pytest.fixture(scope="session")
def stations(prop):
    return prop.discretize(48)


@pytest.fixture(scope="session")
def tables(stations):
    return stations.polar_tables()


@pytest.fixture(scope="session")
def options():
    return SolverOptions(n_elements=48)


@pytest.fixture(scope="session")
def sea_level():
    return isa(0.0)


@pytest.fixture(scope="session")
def static_point(sea_level):
    return OperatingPoint(rpm=2340.0, v_inf=0.0, air=sea_level)


@pytest.fixture(scope="session")
def cruise_point():
    return OperatingPoint(rpm=2500.0, v_inf=58.6, air=isa(2438.4))
