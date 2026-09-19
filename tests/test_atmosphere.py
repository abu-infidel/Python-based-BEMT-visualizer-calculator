"""ISA values are tabulated to five figures; check against the published table."""

from __future__ import annotations

import math

import pytest

from propwash.atmosphere import (GAMMA, R_AIR, density_altitude, isa,
                                 sutherland_viscosity)

# altitude m -> (T K, p Pa, rho kg/m^3) from the ICAO standard atmosphere
ISA_TABLE = [
    (0, 288.150, 101325.0, 1.22500),
    (1000, 281.650, 89874.6, 1.11164),
    (5000, 255.650, 54019.9, 0.73612),
    (11000, 216.650, 22632.0, 0.36392),
    (20000, 216.650, 5474.89, 0.08803),
]


@pytest.mark.parametrize("h,t,p,rho", ISA_TABLE)
def test_isa_matches_published_table(h, t, p, rho):
    state = isa(h)
    assert state.temperature == pytest.approx(t, rel=1e-4)
    assert state.pressure == pytest.approx(p, rel=1e-4)
    assert state.density == pytest.approx(rho, rel=1e-4)


def test_sound_speed_consistent_with_temperature():
    state = isa(7000.0)
    assert state.sound_speed == pytest.approx(
        math.sqrt(GAMMA * R_AIR * state.temperature), rel=1e-9)


def test_viscosity_rises_with_temperature():
    assert sutherland_viscosity(300.0) > sutherland_viscosity(250.0)
    assert sutherland_viscosity(288.15) == pytest.approx(1.789e-5, rel=2e-3)


def test_hot_day_raises_density_altitude():
    hot = isa(1500.0, delta_isa=20.0)
    assert density_altitude(hot) > 1500.0
    assert hot.density < isa(1500.0).density


def test_humid_air_is_less_dense():
    assert isa(0.0, humidity=1.0).density < isa(0.0, humidity=0.0).density


def test_ideal_gas_law_holds():
    for h in (0.0, 3000.0, 9000.0, 18000.0):
        s = isa(h)
        assert s.pressure == pytest.approx(s.density * R_AIR * s.temperature, rel=1e-9)
