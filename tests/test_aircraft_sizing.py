"""Aircraft propellers, piston engines, and propeller sizing/design."""

from __future__ import annotations

import math

import numpy as np
import pytest

from propwash.atmosphere import RHO0, isa
from propwash.bemt.core import OperatingPoint, SolverOptions
from propwash.bemt.solver import PropellerSolver
from propwash.bemt.sweep import match_operating_point, thrust_required_speed
from propwash.engine import get_engine, list_engines
from propwash.geometry import AIRCRAFT_PRESETS, DEFAULT_PRESET, DRONE_PRESETS, get_preset, list_presets
from propwash.sizing import (adkins_liebeck_design, diameter_sweep,
                             drag_from_weight, induced_velocity, momentum_sizing,
                             verify_design)
from propwash.units import HORSEPOWER, INCH, KNOT


# ---------------------------------------------------------------------------
# Presets: the aircraft ones are new, the drone ones must be untouched
# ---------------------------------------------------------------------------

def test_default_is_an_aircraft_propeller():
    assert DEFAULT_PRESET in AIRCRAFT_PRESETS
    assert "172" in DEFAULT_PRESET


def test_drone_presets_are_all_still_there():
    """The whole point of adding aircraft support was to lose nothing."""
    names = set(list_presets())
    for name in DRONE_PRESETS:
        assert name in names, f"{name} disappeared"
    assert len(list_presets("drone")) == 10
    assert set(list_presets("aircraft")).isdisjoint(list_presets("drone"))
    assert len(list_presets()) == len(DRONE_PRESETS) + len(AIRCRAFT_PRESETS)


@pytest.mark.parametrize("name", AIRCRAFT_PRESETS)
def test_aircraft_presets_are_plausible(name):
    g = get_preset(name)
    assert 1.5 < g.diameter < 2.3, "GA propellers are 60-90 inches"
    assert 0.4 < g.pitch_diameter_ratio() < 1.1
    assert 60.0 < g.activity_factor() < 160.0
    st = g.discretize(30)
    assert st.thickness[0] > st.thickness[-1], "root must be thicker than tip"


def test_cessna_172_pitch_matches_its_designation():
    g = get_preset("Cessna 172 (McCauley 75x57)")
    assert g.diameter / INCH == pytest.approx(75.0, abs=0.1)
    assert g.pitch_diameter_ratio() == pytest.approx(57.0 / 75.0, abs=0.03)


# ---------------------------------------------------------------------------
# Piston engine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list_engines())
def test_engine_curves_are_sane(name):
    e = get_engine(name)
    rpm = np.linspace(e.idle_rpm / e.gear_ratio, e.no_load_rpm(), 40)
    torque = e.shaft_torque(rpm)
    power = e.shaft_power(rpm)
    assert (torque >= 0.0).all()
    assert np.isfinite(power).all()
    assert power[-1] == pytest.approx(e.rated_power, rel=0.02), "rated power at redline"


def test_engine_torque_peaks_below_rated_speed():
    """That shape is what lets a fixed-pitch propeller over-rev a piston engine."""
    e = get_engine("Lycoming O-320-D2J (C172N/P)")
    rpm = np.linspace(800.0, 2700.0, 300)
    peak = rpm[int(np.argmax(e.shaft_torque(rpm)))]
    assert 1800.0 < peak < 2500.0
    assert e.shaft_torque(np.array([2700.0]))[0] < e.shaft_torque(np.array([peak]))[0]


def test_engine_produces_nothing_below_idle():
    e = get_engine("Lycoming O-320-D2J (C172N/P)")
    assert float(e.shaft_torque(np.array([300.0]))[0]) == 0.0


def test_gagg_farrar_lapse_is_steeper_than_density():
    """An unsupercharged engine loses more than the density ratio suggests."""
    e = get_engine("Lycoming O-320-D2J (C172N/P)")
    air = isa(2438.0)                       # 8,000 ft
    sigma = air.density / RHO0
    factor = e.altitude_factor(air)
    assert factor < sigma
    assert 0.72 < factor < 0.80, "about 75% power at 8,000 ft"


def test_gearbox_divides_propeller_speed():
    r = get_engine("Rotax 912ULS (LSA)")
    assert r.gear_ratio > 2.0
    assert r.no_load_rpm() == pytest.approx(r.redline_rpm / r.gear_ratio)
    assert float(r.shaft_torque(np.array([r.no_load_rpm()]))[0]) > 200.0


def test_fuel_flow_is_in_the_right_ballpark():
    """A 160 hp Lycoming burns roughly 13 gph at full power."""
    e = get_engine("Lycoming O-320-D2J (C172N/P)")
    assert 11.0 < float(e.fuel_flow_gph(np.array([2700.0]))[0]) < 15.0


# ---------------------------------------------------------------------------
# Matching an engine to a propeller
# ---------------------------------------------------------------------------

def test_engine_match_finds_a_cruise_point():
    """The bracket must start at idle: below it an engine makes no torque at all."""
    mp = match_operating_point(get_preset("Cessna 172 (McCauley 75x57)"),
                               get_engine("Lycoming O-320-D2J (C172N/P)"),
                               v_inf=110 * KNOT, air=isa(2438.0))
    assert mp.converged
    assert 2000.0 < mp.rpm < 2700.0
    assert "fuel_gph" in mp.extras
    assert 6.0 < mp.extras["fuel_gph"] < 14.0
    assert 0.5 < mp.extras["power_fraction"] < 1.0


def test_throttling_reduces_speed_power_and_fuel():
    g = get_preset("Cessna 172 (McCauley 75x57)")
    e = get_engine("Lycoming O-320-D2J (C172N/P)")
    full = match_operating_point(g, e, v_inf=110 * KNOT, throttle=1.0, air=isa(2438.0))
    part = match_operating_point(g, e, v_inf=110 * KNOT, throttle=0.7, air=isa(2438.0))
    assert part.rpm < full.rpm
    assert part.shaft_power < full.shaft_power
    assert part.extras["fuel_gph"] < full.extras["fuel_gph"]


def test_cessna_172_trims_near_its_published_cruise_speed():
    """End-to-end: blade geometry -> engine -> drag balance -> airspeed."""
    trim = thrust_required_speed(
        get_preset("Cessna 172 (McCauley 75x57)"),
        get_engine("Lycoming O-320-D2J (C172N/P)"),
        wing_area=16.17, span=11.0, cd0=0.032, oswald=0.75, mass=1043.0,
        air=isa(2438.0), v_max=90.0, n_scan=16)
    assert trim["converged"], trim.get("reason")
    assert 100.0 < trim["speed"] / KNOT < 135.0, "published C172N cruise is ~120 KTAS"


def test_electric_matching_still_works():
    """Adding engines must not have broken the drone path."""
    from propwash.motor import get_motor
    mp = match_operating_point(get_preset("APC 10x5 (sport)"),
                               get_motor("Sport 2820 kv1000"))
    assert mp.converged
    assert 5000.0 < mp.rpm < 11100.0
    assert mp.current > 0.0


# ---------------------------------------------------------------------------
# Compressibility -- the correction aircraft propellers actually need
# ---------------------------------------------------------------------------

def test_mach_ceiling_bounds_the_lift():
    from propwash.bemt.core import mach_lift_ceiling
    assert float(mach_lift_ceiling(0.2, 1.2)) == pytest.approx(1.2)
    assert float(mach_lift_ceiling(0.7, 1.2)) < 1.2
    assert float(mach_lift_ceiling(0.9, 1.2)) < float(mach_lift_ceiling(0.7, 1.2))
    assert float(mach_lift_ceiling(0.95, 1.2)) >= 0.35 * 1.2


def test_transonic_tip_does_not_exceed_its_section_maximum():
    """Prandtl-Glauert must not hand a tip more lift than the section can make."""
    g = get_preset("Cessna 172 (McCauley 75x57)")
    r = PropellerSolver(g).solve(OperatingPoint(rpm=2700.0, v_inf=0.0))
    tip = r.x > 0.7
    assert r.tip_mach > 0.7, "this case should really be transonic"
    assert r.cl[tip].max() < 1.35
    # The bug this guards against inflated L/D across the whole outer span, so
    # test the median there rather than a single element: the very last station
    # has a vanishing chord and legitimately runs high.
    ld = r.cl[tip] / np.maximum(r.cd[tip], 1e-9)
    assert np.median(ld) < 80.0, "a propeller section does not average L/D of 160"
    assert ld.max() < 130.0


def test_section_params_carries_cl_max():
    st = get_preset("Cessna 172 (McCauley 75x57)").discretize(20)
    thickness, re_ref, m_crit0, cl_max = st.section_params()
    assert cl_max.shape == (20,)
    assert (cl_max > 0.8).all() and (cl_max < 2.0).all()
    assert (thickness > 0.0).all() and (re_ref > 0.0).all() and (m_crit0 > 0.0).all()


# ---------------------------------------------------------------------------
# Momentum-theory disk sizing
# ---------------------------------------------------------------------------

def test_induced_velocity_matches_the_hover_limit():
    """At V = 0 the quadratic must collapse to sqrt(T / 2 rho A)."""
    area, thrust = 2.85, 3000.0
    w = induced_velocity(thrust, 0.0, area, RHO0)
    assert w == pytest.approx(math.sqrt(thrust / (2.0 * RHO0 * area)), rel=1e-9)


def test_momentum_sizing_satisfies_its_own_equation():
    s = momentum_sizing(930.0, 61.7, 1.905, eta_p=0.82)
    w, a = s.induced_velocity, s.disk_area
    # The induced velocity comes from a square root, so round-trip agreement is
    # limited by that, not by the formula.
    assert 2.0 * RHO0 * a * w * (s.v_inf + w) == pytest.approx(s.thrust, rel=1e-7)
    assert s.ideal_power == pytest.approx(s.thrust * (s.v_inf + w), rel=1e-12)
    assert s.shaft_power > s.ideal_power


def test_bigger_disk_needs_less_power():
    """The whole reason aircraft propellers are as big as clearance allows."""
    sweep = diameter_sweep(930.0, 61.7, np.linspace(1.0, 2.6, 30), rpm=2400.0)
    assert (np.diff(sweep["shaft_power"]) < 0.0).all()
    assert (np.diff(sweep["ideal_efficiency"]) > 0.0).all()
    assert (np.diff(sweep["tip_mach"]) > 0.0).all(), "but the tip gets faster"
    assert sweep["feasible"][0] and not sweep["feasible"][-1]


def test_drag_from_weight():
    assert drag_from_weight(1000.0, 10.0) == pytest.approx(980.665, rel=1e-6)


def test_momentum_power_is_a_lower_bound_on_a_real_propeller():
    """No propeller of that diameter can beat the actuator disk."""
    g = get_preset("Cessna 172 (McCauley 75x57)")
    air = isa(2438.0)
    r = PropellerSolver(g).solve(OperatingPoint(rpm=2400.0, v_inf=110 * KNOT, air=air))
    ideal = momentum_sizing(r.thrust, r.v_inf, g.diameter, air=air, eta_p=1.0)
    assert ideal.ideal_power < r.power


# ---------------------------------------------------------------------------
# Adkins-Liebeck design
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cruise_design():
    return adkins_liebeck_design(
        radius=75 * INCH / 2, n_blades=2, rpm=2400.0, v_inf=120 * KNOT,
        thrust=drag_from_weight(1043.0, 11.0), air=isa(2438.0), design_cl=0.65)


def test_design_converges_and_hits_its_target(cruise_design):
    d = cruise_design
    assert d.converged and d.iterations < 30
    assert d.thrust == pytest.approx(drag_from_weight(1043.0, 11.0), rel=0.01)
    assert 0.80 < d.efficiency < 0.95


def test_design_geometry_is_physical(cruise_design):
    d = cruise_design
    assert (d.chord > 0.0).all()
    assert d.chord.max() < 0.3 * d.radius
    assert (np.diff(d.twist) < 0.0).all(), "blade angle must fall monotonically outboard"
    assert (np.diff(d.phi) < 0.0).all()
    assert d.twist.max() < math.radians(80.0)
    assert 20.0 < d.activity_factor < 200.0


def test_design_agrees_with_the_independent_bemt_solver(cruise_design):
    """The design method and the analysis share no equations; agreement is real."""
    check = verify_design(cruise_design, air=isa(2438.0))
    assert abs(check["thrust_error"]) < 0.06
    assert abs(check["power_error"]) < 0.06
    assert check["bemt_efficiency"] == pytest.approx(cruise_design.efficiency, abs=0.02)


def test_compressible_design_agrees_better_than_incompressible():
    """The compressible correction is what closes a 17% gap on a fast propeller."""
    kw = dict(radius=75 * INCH / 2, n_blades=2, rpm=2400.0, v_inf=120 * KNOT,
              thrust=drag_from_weight(1043.0, 11.0), air=isa(2438.0), design_cl=0.65)
    classical = verify_design(adkins_liebeck_design(compressible=False, **kw),
                              air=isa(2438.0))
    corrected = verify_design(adkins_liebeck_design(compressible=True, **kw),
                              air=isa(2438.0))
    assert abs(corrected["thrust_error"]) < abs(classical["thrust_error"])
    assert abs(classical["thrust_error"]) > 0.10
    assert abs(corrected["thrust_error"]) < 0.05


def test_design_alpha_falls_as_mach_rises():
    """Less angle of attack is needed for the same Cl once compressible."""
    d = adkins_liebeck_design(radius=75 * INCH / 2, n_blades=2, rpm=2700.0,
                              v_inf=60.0, thrust=1000.0, design_cl=0.6)
    alpha = d.meta["alpha_station"]
    assert alpha[0] > alpha[-1]


def test_power_specified_design_absorbs_that_power():
    d = adkins_liebeck_design(radius=0.9, n_blades=2, rpm=2400.0, v_inf=55.0,
                              power=90.0 * HORSEPOWER, design_cl=0.6)
    assert d.converged
    assert d.power == pytest.approx(90.0 * HORSEPOWER, rel=0.02)


def test_design_rejects_ambiguous_and_static_requests():
    with pytest.raises(ValueError):
        adkins_liebeck_design(radius=0.9, n_blades=2, rpm=2400.0, v_inf=55.0)
    with pytest.raises(ValueError):
        adkins_liebeck_design(radius=0.9, n_blades=2, rpm=2400.0, v_inf=55.0,
                              thrust=800.0, power=50000.0)
    with pytest.raises(ValueError):
        adkins_liebeck_design(radius=0.9, n_blades=2, rpm=2400.0, v_inf=0.0,
                              thrust=800.0)


def test_design_converts_to_a_usable_geometry(cruise_design):
    g = cruise_design.to_geometry()
    assert g.n_blades == cruise_design.n_blades
    assert g.diameter == pytest.approx(cruise_design.diameter, rel=1e-9)
    r = PropellerSolver(g, SolverOptions(n_elements=48)).solve(
        OperatingPoint(rpm=cruise_design.rpm, v_inf=cruise_design.v_inf,
                       air=isa(2438.0)))
    assert r.converged_fraction == 1.0
    assert r.thrust > 0.0


def test_more_blades_need_less_chord_for_the_same_thrust():
    kw = dict(radius=0.95, rpm=2400.0, v_inf=60.0, thrust=1000.0, design_cl=0.65)
    two = adkins_liebeck_design(n_blades=2, **kw)
    three = adkins_liebeck_design(n_blades=3, **kw)
    assert three.chord.max() < two.chord.max()
    assert three.thrust == pytest.approx(two.thrust, rel=0.02)
