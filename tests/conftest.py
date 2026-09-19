"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from propwash.atmosphere import isa                      # noqa: E402
from propwash.bemt.core import OperatingPoint, SolverOptions  # noqa: E402
from propwash.geometry import get_preset                 # noqa: E402


@pytest.fixture(scope="session")
def prop():
    return get_preset("APC 10x5 (sport)")


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
def hover_point():
    return OperatingPoint(rpm=8000.0, v_inf=0.0)


@pytest.fixture(scope="session")
def cruise_point():
    return OperatingPoint(rpm=8000.0, v_inf=12.0)
