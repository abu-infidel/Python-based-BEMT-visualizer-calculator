"""Solver tests: the residual, the bisection, and whether the answers are physics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from propwash.atmosphere import isa
from propwash.bemt.core import (PHI_HI, PHI_LO, OperatingPoint, SolverOptions,
                                prandtl_loss, residual_phi, solve_batch,
                                solve_stations)
from propwash.bemt.solver import PropellerSolver
from propwash.geometry import get_preset
from propwash.units import rpm_to_rad_s


def _context(stations, tables, rpm, v_inf, air, flags):
    from propwash.bemt.core import _batch_context
    return _batch_context(stations, np.array([rpm_to_rad_s(rpm)]),
                          np.array([float(v_inf)]), air, flags, tables)


# ---------------------------------------------------------------------------
# The residual and the bracket
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rpm,v", [(3000, 0.0), (8000, 0.0), (8000, 10.0),
                                   (12000, 25.0), (600, 0.0)])
def test_residual_changes_sign_across_the_bracket(stations, tables, sea_level,
                                                  options, rpm, v):
    """The whole design rests on R(0+) < 0 < R(pi/2-); check it really holds."""
    ctx = _context(stations, tables, rpm, v, sea_level, options.flags())
    lo = np.full((1, stations.n), PHI_LO)
    hi = np.full((1, stations.n), PHI_HI)
    r_lo = residual_phi(lo, ctx)
    r_hi = residual_phi(hi, ctx)
    assert (r_lo < 0.0).all(), "residual should be negative at phi -> 0"
    assert (r_hi > 0.0).all(), "residual should be positive at phi -> pi/2"


def test_converged_phi_actually_zeroes_the_residual(stations, tables, sea_level,
                                                    options):
    result = solve_stations(stations, OperatingPoint(rpm=8000.0, v_inf=8.0),
                            options, tables)
    ctx = _context(stations, tables, 8000.0, 8.0, sea_level, options.flags())
    residual = residual_phi(result.phi.reshape(1, -1), ctx)
    scale = np.maximum(np.abs(ctx.omega_r), 1.0)
    assert np.abs(residual / scale).max() < 1e-9


def test_bisection_precision_improves_with_iterations(stations, tables):
    coarse = solve_stations(stations, OperatingPoint(rpm=8000.0),
                            SolverOptions(n_bisect=10), tables)
    fine = solve_stations(stations, OperatingPoint(rpm=8000.0),
                          SolverOptions(n_bisect=60), tables)
    finer = solve_stations(stations, OperatingPoint(rpm=8000.0),
                           SolverOptions(n_bisect=80), tables)
    assert np.abs(fine.phi - finer.phi).max() < np.abs(coarse.phi - finer.phi).max()
    assert np.abs(fine.phi - finer.phi).max() < 1e-12


def test_everything_converges_in_the_normal_envelope(prop):
    solver = PropellerSolver(prop)
    for rpm in (1000.0, 5000.0, 10000.0, 16000.0):
        for v in (0.0, 5.0, 15.0):
            result = solver.solve(OperatingPoint(rpm=rpm, v_inf=v))
            assert result.converged_fraction == 1.0, f"{rpm} rpm, {v} m/s"
            assert np.isfinite(result.thrust) and np.isfinite(result.power)


# ---------------------------------------------------------------------------
# Prandtl loss
# ---------------------------------------------------------------------------

def test_prandtl_loss_vanishes_at_the_tip_and_hub(prop):
    st = prop.discretize(200)
    phi = np.full(st.n, 0.15)
    f = prandtl_loss(phi, st.r, prop.radius, prop.hub_radius, prop.n_blades, 0b11)
    assert f[-1] < 0.35, "loss factor should collapse at the tip"
    assert f[0] < 0.6, "and at the hub"
    assert f[st.n // 2] > 0.97, "and be ~1 in the middle"
    assert (f > 0.0).all() and (f <= 1.0).all()


def test_tip_loss_reduces_thrust(prop):
    with_loss = PropellerSolver(prop, SolverOptions(tip_loss=True, hub_loss=True))
    without = PropellerSolver(prop, SolverOptions(tip_loss=False, hub_loss=False))
    op = OperatingPoint(rpm=8000.0)
    assert with_loss.solve(op).thrust < without.solve(op).thrust


# ---------------------------------------------------------------------------
# Physical behaviour
# ---------------------------------------------------------------------------

def test_static_thrust_scales_roughly_as_rpm_squared(prop):
    solver = PropellerSolver(prop)
    a = solver.solve(OperatingPoint(rpm=4000.0))
    b = solver.solve(OperatingPoint(rpm=8000.0))
    assert b.thrust / a.thrust == pytest.approx(4.0, rel=0.10)
    assert b.power / a.power == pytest.approx(8.0, rel=0.12)
    # CT is only *nearly* RPM-independent at model scale: doubling RPM doubles
    # Reynolds number, profile drag falls, and net thrust rises a few percent.
    assert b.ct == pytest.approx(a.ct, rel=0.08), "CT is nearly RPM-independent"
    assert b.ct > a.ct, "and the Reynolds trend should go this way"


def test_thrust_falls_with_forward_speed(prop):
    solver = PropellerSolver(prop)
    thrust = [solver.solve(OperatingPoint(rpm=9000.0, v_inf=v)).thrust
              for v in (0.0, 5.0, 10.0, 15.0, 20.0)]
    assert all(b < a for a, b in zip(thrust, thrust[1:]))


def test_efficiency_is_bounded_and_peaks_in_the_middle(prop):
    from propwash.bemt.sweep import j_sweep
    sweep = j_sweep(prop, rpm=9000.0, j_max=1.2, n=80)
    eta = np.asarray(sweep["efficiency"])
    assert (eta >= 0.0).all(), "efficiency is reported as zero once it stops meaning anything"
    assert eta.max() < 1.0, "a propeller cannot beat its own input power"
    peak = int(np.argmax(eta))
    assert 0 < peak < len(eta) - 1, "peak should be interior, not at an endpoint"
    assert 0.55 < eta[peak] < 0.92, f"implausible peak efficiency {eta[peak]}"


def test_figure_of_merit_is_physical(prop):
    """FoM is actual over ideal induced power; exceeding 1 breaks momentum theory."""
    solver = PropellerSolver(prop)
    for rpm in (3000.0, 7000.0, 12000.0):
        fom = solver.solve(OperatingPoint(rpm=rpm)).figure_of_merit
        assert 0.3 < fom < 1.0, f"{fom} at {rpm} rpm"


def test_static_thrust_stays_below_the_ideal_actuator_disk(prop):
    """Momentum theory gives the best any disk can do for a given power."""
    solver = PropellerSolver(prop)
    r = solver.solve(OperatingPoint(rpm=9000.0))
    area = math.pi * prop.radius ** 2
    ideal = (2.0 * 1.225 * area) ** (1.0 / 3.0) * r.power ** (2.0 / 3.0)
    assert r.thrust < ideal


def test_more_blades_make_more_thrust(prop):
    from dataclasses import replace
    op = OperatingPoint(rpm=8000.0)
    two = PropellerSolver(replace(prop, n_blades=2)).solve(op).thrust
    three = PropellerSolver(replace(prop, n_blades=3)).solve(op).thrust
    assert three > two
    assert three < 1.5 * two, "but not linearly -- induced losses grow too"


def test_more_pitch_makes_more_thrust_and_more_power(prop):
    from dataclasses import replace
    op = OperatingPoint(rpm=8000.0)
    flat = PropellerSolver(replace(prop, pitch_offset=math.radians(-4))).solve(op)
    coarse = PropellerSolver(replace(prop, pitch_offset=math.radians(4))).solve(op)
    assert coarse.thrust > flat.thrust
    assert coarse.power > flat.power


def test_thinner_air_makes_less_thrust(prop):
    solver = PropellerSolver(prop)
    low = solver.solve(OperatingPoint(rpm=8000.0, air=isa(0.0)))
    high = solver.solve(OperatingPoint(rpm=8000.0, air=isa(5000.0)))
    assert high.thrust < low.thrust
    assert high.thrust / low.thrust == pytest.approx(isa(5000.0).density / 1.225, rel=0.1)


def test_thrust_loading_collapses_at_the_tip(prop):
    result = PropellerSolver(prop).solve(OperatingPoint(rpm=9000.0))
    assert result.dt_dr[-1] < 0.5 * result.dt_dr.max()
    assert 0.6 < result.x[int(np.argmax(result.dt_dr))] < 0.95


def test_induced_velocity_matches_momentum_theory_order(prop):
    """Mean induced velocity should be near sqrt(T / 2 rho A) in hover."""
    result = PropellerSolver(prop).solve(OperatingPoint(rpm=9000.0))
    area = math.pi * prop.radius ** 2
    ideal = math.sqrt(result.thrust / (2.0 * 1.225 * area))
    weighted = float(np.average(result.v_axial_induced, weights=result.r))
    assert 0.6 * ideal < weighted < 1.9 * ideal


def test_coefficients_are_self_consistent(prop, sea_level):
    r = PropellerSolver(prop).solve(OperatingPoint(rpm=7500.0, v_inf=9.0,
                                                   air=sea_level))
    n = r.rpm / 60.0
    d = prop.diameter
    rho = sea_level.density          # ISA-derived, not exactly 1.225
    assert r.ct == pytest.approx(r.thrust / (rho * n ** 2 * d ** 4), rel=1e-9)
    assert r.cp == pytest.approx(r.power / (rho * n ** 3 * d ** 5), rel=1e-9)
    assert r.j == pytest.approx(r.v_inf / (n * d), rel=1e-9)
    assert r.efficiency == pytest.approx(r.ct * r.j / r.cp, rel=1e-6)


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------

def test_batch_matches_single_point_exactly(stations, tables, options, sea_level):
    rpm = np.array([2500.0, 6000.0, 11000.0])
    v = np.array([0.0, 7.0, 18.0])
    batch = solve_batch(stations, rpm, v, sea_level, options, tables)
    for i in range(rpm.size):
        one = solve_stations(stations, OperatingPoint(rpm=rpm[i], v_inf=v[i],
                                                      air=sea_level), options, tables)
        assert batch["thrust"][i] == pytest.approx(one.thrust, rel=0, abs=1e-12)
        assert batch["torque"][i] == pytest.approx(one.torque, rel=0, abs=1e-14)
        assert np.abs(batch["phi"][i] - one.phi).max() < 1e-15


def test_batch_preserves_grid_shape(stations, tables, options, sea_level):
    rpm, v = np.meshgrid(np.linspace(2000, 10000, 7), np.linspace(0, 20, 5),
                         indexing="ij")
    out = solve_batch(stations, rpm, v, sea_level, options, tables)
    assert out["thrust"].shape == (7, 5)
    assert out["phi"].shape == (7, 5, stations.n)


def test_solver_caches_and_invalidates(prop):
    from dataclasses import replace
    solver = PropellerSolver(prop)
    first = solver.stations
    assert solver.stations is first, "discretisation should be cached"
    solver.geometry = replace(prop, n_blades=3)
    assert solver.stations is not first, "and dropped when the blade changes"


def test_set_pitch_shifts_every_station_rigidly(prop):
    solver = PropellerSolver(prop)
    before = solver.stations.twist.copy()
    solver.set_pitch(math.radians(5.0))
    assert np.allclose(solver.stations.twist - before, math.radians(5.0))


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def test_sweep_best_ignores_backend_bookkeeping(prop):
    """Backends add a name and a timing to the result dict; those are not fields."""
    from propwash.bemt.sweep import j_sweep
    sweep = j_sweep(prop, rpm=9000.0, n=32)
    assert "backend" in sweep.data, "the backend name should still be carried"
    assert "backend" not in sweep.field_names()
    best = sweep.best("efficiency")
    assert isinstance(best["efficiency"], float)
    assert 0.0 <= best["efficiency"] < 1.0
    assert 0.0 < best["j"] < 1.2


def test_sweep_frame_is_one_row_per_operating_point(prop):
    from propwash.bemt.sweep import envelope_map
    sweep = envelope_map(prop, np.linspace(3000.0, 11000.0, 6),
                         np.linspace(0.0, 20.0, 5))
    assert sweep.shape == (6, 5)
    frame = sweep.to_frame()
    assert len(frame) == 30
    assert {"rpm", "v_inf", "thrust", "power"} <= set(frame.columns)
    assert frame["thrust"].notna().all()


def test_pitch_rpm_map_rows_track_collective(prop):
    """Thrust rises with collective, then falls once the blade stalls."""
    from propwash.bemt.sweep import pitch_rpm_map
    pitch = np.linspace(-6.0, 12.0, 10)
    sweep = pitch_rpm_map(prop, pitch, np.linspace(4000.0, 10000.0, 7))
    assert sweep.shape == (10, 7)

    thrust = np.asarray(sweep["thrust"])[:, -1]
    peak = int(np.argmax(thrust))
    assert peak > 0, "thrust must increase with pitch before it stalls"
    assert (np.diff(thrust[: peak + 1]) > 0.0).all()
    if peak < len(thrust) - 1:
        assert thrust[-1] < thrust[peak], "past stall it must come back down"
        assert np.asarray(sweep["stall_fraction"])[-1, -1] > \
            np.asarray(sweep["stall_fraction"])[0, -1]


def test_thrust_required_speed_finds_a_trim_point(prop):
    from propwash.bemt.sweep import thrust_required_speed
    from propwash.motor import get_motor
    trim = thrust_required_speed(prop, get_motor("Sport 2820 kv1000"),
                                 wing_area=0.22, span=1.2, cd0=0.035, mass=1.0,
                                 v_max=60.0)
    assert trim["converged"], trim.get("reason")
    assert 5.0 < trim["speed"] < 60.0
    assert trim["thrust"] == pytest.approx(trim["drag"], rel=0.08)


def test_thrust_required_speed_reports_an_unflyable_airframe():
    """Too much drag must be reported, not bisected into a wrong answer."""
    from propwash.bemt.sweep import thrust_required_speed
    from propwash.motor import get_motor
    trim = thrust_required_speed(get_preset("APC 6x4 (micro)"),
                                 get_motor("Micro 1104 kv7500"),
                                 wing_area=3.0, span=0.4, cd0=0.3, mass=8.0)
    assert not trim["converged"]
    assert trim["reason"]
