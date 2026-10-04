"""Solver tests on a full-size propeller: the residual, the bisection, the physics."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from propwash.atmosphere import isa
from propwash.bemt.core import (PHI_HI, PHI_LO, OperatingPoint, SolverOptions,
                                prandtl_loss, residual_phi, solve_batch, solve_stations)
from propwash.bemt.solver import PropellerSolver
from propwash.bemt.sweep import j_sweep, pitch_rpm_map
from propwash.units import rpm_to_rad_s

SL = isa(0.0)


def _context(stations, tables, rpm, v_inf, air, flags):
    from propwash.bemt.core import _batch_context
    return _batch_context(stations, np.array([rpm_to_rad_s(rpm)]),
                          np.array([float(v_inf)]), air, flags, tables)


def _solve(prop, rpm, v, air=SL, **opts):
    return solve_stations(prop.discretize(48), OperatingPoint(rpm=rpm, v_inf=v, air=air),
                          SolverOptions(n_elements=48, **opts))


@pytest.mark.parametrize("rpm,v", [(800, 0.0), (2300, 0.0), (2300, 40.0), (2700, 70.0),
                                   (500, 0.0)])
def test_residual_changes_sign_across_the_bracket(stations, tables, sea_level, options,
                                                  rpm, v):
    ctx = _context(stations, tables, rpm, v, sea_level, options.flags())
    r_lo = residual_phi(np.full((1, stations.n), PHI_LO), ctx)
    r_hi = residual_phi(np.full((1, stations.n), PHI_HI), ctx)
    assert (r_lo < 0.0).all() and (r_hi > 0.0).all()


def test_converged_phi_zeroes_the_residual(stations, tables, sea_level, options):
    res = solve_stations(stations, OperatingPoint(rpm=2300.0, v_inf=40.0, air=sea_level),
                         options, tables)
    ctx = _context(stations, tables, 2300.0, 40.0, sea_level, options.flags())
    residual = residual_phi(res.phi.reshape(1, -1), ctx)
    assert np.abs(residual / np.maximum(np.abs(ctx.omega_r), 1.0)).max() < 1e-9


def test_bisection_precision_improves_with_iterations(stations, tables, sea_level):
    op = OperatingPoint(rpm=2300.0, v_inf=30.0, air=sea_level)
    coarse, fine, finer = (solve_stations(stations, op, SolverOptions(n_bisect=n), tables).phi
                           for n in (10, 40, 60))
    assert np.abs(fine - finer).max() < np.abs(coarse - finer).max()


def test_everything_converges_in_the_full_size_envelope(prop, sea_level):
    rpm, v = np.meshgrid(np.linspace(1200.0, 2700.0, 6), np.linspace(0.0, 70.0, 6))
    out = PropellerSolver(prop, SolverOptions(n_elements=40)).solve_many_cpu(rpm, v, sea_level)
    assert np.min(out["converged_fraction"]) >= 0.95


def test_prandtl_loss_vanishes_at_tip_and_hub(prop):
    r = np.linspace(prop.hub_radius, prop.radius, 50)
    f = prandtl_loss(np.full(50, 0.3), r, prop.radius, prop.hub_radius, prop.n_blades, 3)
    assert f[0] < 0.05 and f[-1] < 0.05 and f[25] > 0.9


def test_tip_loss_reduces_thrust(prop):
    assert _solve(prop, 2400, 50.0).thrust < _solve(prop, 2400, 50.0, tip_loss=False).thrust


def test_static_thrust_scales_roughly_as_rpm_squared(prop):
    t1, t2 = _solve(prop, 1800, 0.0).thrust, _solve(prop, 2400, 0.0).thrust
    assert t2 / t1 == pytest.approx((2400 / 1800) ** 2, rel=0.15)


def test_thrust_falls_with_forward_speed(prop):
    thrust = [_solve(prop, 2500, v).thrust for v in (0.0, 30.0, 60.0)]
    assert thrust[0] > thrust[1] > thrust[2] > 0.0


def test_efficiency_is_bounded_and_peaks_inside_the_range(prop):
    sw = j_sweep(prop, rpm=2400.0, j_max=1.0, air=SL, n=21, backend="numpy",
                 options=SolverOptions(n_elements=40))
    eta = np.asarray(sw["efficiency"])
    assert eta.max() < 0.92
    assert 0 < int(np.argmax(eta)) < eta.size - 1


def test_figure_of_merit_is_physical(prop):
    assert 0.45 < _solve(prop, 2340, 0.0).figure_of_merit < 0.85


def test_static_power_exceeds_the_ideal_actuator_disk(prop):
    res = _solve(prop, 2340, 0.0)
    ideal = res.thrust ** 1.5 / math.sqrt(2.0 * SL.density * math.pi * prop.radius ** 2)
    assert res.power > ideal


def test_more_blades_make_more_thrust(prop):
    three = replace(prop, n_blades=3)
    assert _solve(three, 2400, 50.0).thrust > _solve(prop, 2400, 50.0).thrust


def test_more_pitch_makes_more_thrust_and_more_power(prop):
    coarse = replace(prop, pitch_offset=math.radians(2.0))
    a, b = _solve(prop, 2400, 50.0), _solve(coarse, 2400, 50.0)
    assert b.thrust > a.thrust and b.power > a.power


def test_thinner_air_makes_less_thrust(prop):
    assert _solve(prop, 2400, 50.0, ).thrust > _solve(prop, 2400, 50.0, isa(3000.0)).thrust


def test_thrust_loading_collapses_at_the_tip(prop):
    res = _solve(prop, 2400, 50.0)
    assert res.dt_dr[-1] < 0.2 * res.dt_dr.max()


def test_coefficients_are_self_consistent(prop):
    res = _solve(prop, 2400, 50.0)
    assert res.cp == pytest.approx(2.0 * math.pi * res.cq, rel=1e-12)
    assert res.efficiency == pytest.approx(res.ct * res.j / res.cp, rel=1e-9)


def test_batch_matches_single_point_exactly(stations, tables, options, sea_level):
    one = solve_stations(stations, OperatingPoint(rpm=2450.0, v_inf=55.0, air=sea_level),
                         options, tables)
    many = solve_batch(stations, np.array([1500.0, 2450.0]), np.array([0.0, 55.0]),
                       sea_level, options, tables)
    assert float(many["thrust"][1]) == pytest.approx(one.thrust, rel=1e-12)


def test_batch_preserves_grid_shape(stations, tables, options, sea_level):
    rpm, v = np.meshgrid(np.linspace(1000, 2700, 4), np.linspace(0, 60, 3), indexing="ij")
    out = solve_batch(stations, rpm, v, sea_level, options, tables)
    assert out["thrust"].shape == (4, 3) and out["phi"].shape == (4, 3, stations.n)


def test_per_case_collective_equals_rotating_the_blade(prop, sea_level):
    opts = SolverOptions(n_elements=40)
    st = prop.discretize(40)
    tab = st.polar_tables()
    d = math.radians(3.0)
    batched = solve_batch(st, np.array([2400.0]), np.array([50.0]), sea_level, opts, tab,
                          delta_pitch=np.array([d]))
    rotated = replace(prop, pitch_offset=d).discretize(40)
    direct = solve_batch(rotated, np.array([2400.0]), np.array([50.0]), sea_level, opts, tab)
    assert float(batched["thrust"][0]) == pytest.approx(float(direct["thrust"][0]), rel=1e-12)


def test_solver_caches_and_invalidates(prop):
    s = PropellerSolver(prop, SolverOptions(n_elements=30))
    st = s.stations
    assert s.stations is st
    s.geometry = replace(prop, n_blades=3)
    assert s.stations is not st


def test_set_pitch_shifts_every_station_rigidly(prop):
    s = PropellerSolver(prop, SolverOptions(n_elements=30))
    before = s.stations.twist.copy()
    s.set_pitch(math.radians(2.0))
    assert np.allclose(s.stations.twist - before, math.radians(2.0))


def test_pitch_rpm_map_rows_track_collective(prop):
    m = pitch_rpm_map(prop, np.array([-2.0, 0.0, 2.0]), np.array([2000.0, 2400.0]),
                      v_inf=40.0, air=SL, options=SolverOptions(n_elements=30),
                      backend="numpy")
    thrust = np.asarray(m["thrust"])
    assert thrust.shape == (3, 2)
    assert (np.diff(thrust[:, 1]) > 0).all()
