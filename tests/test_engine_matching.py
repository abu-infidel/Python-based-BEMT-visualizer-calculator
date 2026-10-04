"""Piston engine model and propeller/engine matching: statuses, never silent zeros."""

from __future__ import annotations

import math

import numpy as np
import pytest

from propwash import matching as mt
from propwash.atmosphere import isa
from propwash.bemt.core import SolverOptions
from propwash.engine import PistonEngine
from propwash.geometry import full_size_blade
from propwash.units import FOOT, HORSEPOWER

OPTS = SolverOptions(n_elements=40, n_bisect=40)


def _engine(hp=160.0, rpm=2700.0, gear=1.0, aspiration="normal", crit=None):
    return PistonEngine(rated_power=hp * HORSEPOWER, rated_rpm=rpm, gear_ratio=gear,
                        aspiration=aspiration, critical_altitude=crit)


def _prop(pitch_in=57.0, af=109.0, diameter_in=75.0, beta75=None, blades=2):
    kw = dict(beta75_deg=beta75) if beta75 is not None else dict(pitch=pitch_in * 0.0254)
    g = full_size_blade(diameter=diameter_in * 0.0254, n_blades=blades, activity_factor=af,
                        hub_ratio=0.12, root_airfoil="clark_y", tip_airfoil="clark_y",
                        thickness_root=0.20, thickness_tip=0.07, planform="standard", **kw)
    return mt.Propeller(g, OPTS)


# -- engine -------------------------------------------------------------------

def test_rated_power_is_delivered_at_rated_rpm():
    e = _engine()
    assert float(e.full_throttle_power(2700.0, isa(0.0))) == pytest.approx(160 * HORSEPOWER)


@pytest.mark.parametrize("hp", [40.0, 160.0, 450.0, 2000.0, 4500.0])
def test_any_power_rating_is_accepted(hp):
    """No upper limit: the old 400 hp slider cap is gone."""
    e = _engine(hp=hp)
    assert float(e.full_throttle_power(2700.0, isa(0.0))) == pytest.approx(hp * HORSEPOWER)


def test_gagg_farrar_lapse_is_steeper_than_density():
    air = isa(8000 * FOOT)
    assert _engine().altitude_factor(air) < air.density / 1.225


def test_turbocharged_engine_holds_power_to_critical_altitude():
    e = _engine(aspiration="turbocharged", crit=16000 * FOOT)
    assert e.altitude_factor(isa(10000 * FOOT)) == 1.0
    assert e.altitude_factor(isa(25000 * FOOT)) < 1.0


def test_turbocharged_engine_needs_a_critical_altitude():
    with pytest.raises(ValueError):
        _engine(aspiration="turbocharged")


def test_gearbox_divides_propeller_speed():
    e = _engine(hp=100.0, rpm=5800.0, gear=2.43)
    assert e.max_prop_rpm == pytest.approx(5800.0 / 2.43)


@pytest.mark.parametrize("bad", [dict(hp=0.0), dict(rpm=-1.0), dict(gear=0.0),
                                 dict(aspiration="steam")])
def test_engine_refuses_meaningless_inputs(bad):
    with pytest.raises(ValueError):
        _engine(**bad)


def test_fuel_flow_is_none_without_bsfc():
    assert _engine().fuel_flow(100_000.0) is None


# -- fixed pitch --------------------------------------------------------------

def test_static_rpm_is_in_the_type_certificate_range():
    m = mt.fixed_pitch(_prop(), _engine(), [0.0], isa(0.0))
    assert m.status[0] == mt.OK
    assert 2280.0 <= m.prop_rpm[0] <= 2400.0


def test_fine_propeller_at_high_speed_is_rpm_limited_not_zero():
    m = mt.fixed_pitch(_prop(pitch_in=50.0), _engine(), [60.0], isa(0.0))
    pt = m.at(0)
    assert pt["status"] == mt.RPM_LIMITED
    assert pt["prop_rpm"] == pytest.approx(2700.0)
    assert 0.0 < pt["throttle"] < 1.0 and pt["thrust"] > 0.0


def test_absurdly_coarse_propeller_is_reported_too_coarse():
    m = mt.fixed_pitch(_prop(af=600.0, beta75=60.0), _engine(hp=60.0), [0.0], isa(0.0))
    assert m.status[0] == mt.TOO_COARSE
    assert math.isnan(m.prop_rpm[0]) and math.isnan(m.solution["thrust"][0])


def test_set_rpm_beyond_full_throttle_is_flagged():
    m = mt.fixed_pitch_at_rpm(_prop(), _engine(), 2700.0, [30.0], isa(10000 * FOOT))
    assert m.status[0] == mt.POWER_NOT_AVAILABLE


def test_matched_point_balances_torque():
    e, p, air = _engine(), _prop(), isa(4000 * FOOT)
    m = mt.fixed_pitch(p, e, [50.0], air)
    q_engine = float(e.full_throttle_torque(m.prop_rpm[0], air))
    assert float(m.solution["torque"][0]) == pytest.approx(q_engine, rel=2e-3)


def test_set_power_finds_the_rpm_that_absorbs_it():
    e, p = _engine(), _prop()
    m = mt.fixed_pitch_at_power(p, e, 0.6 * e.rated_power, [55.0], isa(4000 * FOOT))
    assert m.status[0] == mt.OK
    assert float(m.solution["power"][0]) == pytest.approx(0.6 * e.rated_power, rel=2e-3)


# -- constant speed -----------------------------------------------------------

def test_constant_speed_absorbs_the_set_power():
    e, p = _engine(250.0, 2700.0), _prop(beta75=12.0, af=110.0, diameter_in=80.0, blades=3)
    m = mt.constant_speed(p, e, 2500.0, 0.75 * e.rated_power, [80.0], isa(6000 * FOOT),
                          math.radians(30.0))
    assert m.status[0] == mt.OK
    assert float(m.solution["power"][0]) == pytest.approx(0.75 * e.rated_power, rel=2e-3)
    assert 0.0 < m.delta_pitch[0] < math.radians(30.0)


def test_constant_speed_static_sits_on_the_fine_stop():
    e, p = _engine(250.0, 2700.0), _prop(beta75=22.0, af=110.0, diameter_in=80.0, blades=3)
    m = mt.constant_speed(p, e, 2700.0, e.rated_power, [0.0], isa(0.0), math.radians(25.0))
    assert m.status[0] == mt.ON_FINE_STOP
    assert m.prop_rpm[0] < 2700.0


def test_constant_speed_at_the_coarse_stop_overspeeds():
    e, p = _engine(250.0, 2700.0), _prop(beta75=10.0, af=110.0, diameter_in=80.0, blades=3)
    m = mt.constant_speed(p, e, 2500.0, e.rated_power * 0.9, [50.0], isa(0.0),
                          math.radians(4.0))
    assert m.status[0] == mt.ON_COARSE_STOP
    assert 2500.0 < m.prop_rpm[0] <= 2700.0


def test_spurious_windmill_brake_roots_are_not_operating_points():
    """At high speed and low RPM momentum theory has nonsense roots; never pick them."""
    e, p = _engine(250.0, 2700.0), _prop(beta75=10.0, af=110.0, diameter_in=80.0, blades=3)
    m = mt.constant_speed(p, e, 2500.0, e.rated_power * 0.9, [70.0], isa(0.0),
                          math.radians(6.0))
    assert m.status[0] == mt.RPM_LIMITED and m.prop_rpm[0] == pytest.approx(2700.0)


# -- trims ----------------------------------------------------------------------

def test_thrust_trim_reproduces_the_requested_thrust():
    e, p = _engine(), _prop()
    m = mt.thrust_trim_fixed(p, e, 900.0, [55.0], isa(4000 * FOOT))
    assert m.status[0] == mt.OK
    assert float(m.solution["thrust"][0]) == pytest.approx(900.0, rel=2e-3)


def test_impossible_thrust_is_not_reachable():
    m = mt.thrust_trim_fixed(_prop(), _engine(), 20000.0, [55.0], isa(0.0))
    assert m.status[0] == mt.THRUST_NOT_REACHABLE


def test_vectorised_match_over_speeds_has_no_silent_zeros():
    v = np.linspace(0.0, 90.0, 13)
    m = mt.fixed_pitch(_prop(), _engine(), v, isa(0.0))
    assert set(m.status) <= {mt.OK, mt.RPM_LIMITED, mt.WINDMILLING}
    rpm = m.prop_rpm
    assert np.all(np.diff(rpm[v > 20.0]) >= 0.0), "RPM rises with speed past the bucket"
    thrust = m.solution["thrust"]
    assert np.isfinite(thrust).all() and not np.any(thrust == 0.0)
    assert np.all(thrust[m.status == mt.WINDMILLING] < 0.0)
