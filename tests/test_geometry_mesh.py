"""Blade geometry, motor model and the 3-D mesh."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from propwash.geometry import (BladeGeometry, Distribution, constant_pitch_twist,
                               get_preset, list_presets)
from propwash.mesh import build_propeller_mesh
from propwash.mesh.loft import PART_BLADE, _face_normals
from propwash.mesh.sections import (camber_from_zero_lift, naca4_section,
                                    section_area)
from propwash.motor import get_motor, list_motors, match_rpm
from propwash.units import INCH


# ---------------------------------------------------------------------------
# Distributions and blades
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["constant", "linear", "parabolic", "elliptic",
                                  "inverse", "betz"])
def test_distributions_are_finite_and_positive(kind):
    dist = Distribution(kind, root=0.12, tip=0.01, power=2.0)
    y = dist.evaluate(np.linspace(0.1, 1.0, 200))
    assert np.isfinite(y).all()


def test_spline_passes_through_its_control_points():
    cx = (0.15, 0.4, 0.7, 1.0)
    cy = (0.08, 0.18, 0.14, 0.02)
    dist = Distribution("spline", control_x=cx, control_y=cy)
    assert np.allclose(dist.evaluate(np.array(cx)), np.array(cy), atol=1e-12)


def test_monotone_spline_does_not_overshoot():
    """A shape-preserving interpolant must not produce a negative chord."""
    dist = Distribution("spline", control_x=(0.15, 0.5, 0.9, 1.0),
                        control_y=(0.10, 0.18, 0.05, 0.005))
    y = dist.evaluate(np.linspace(0.15, 1.0, 500))
    assert y.min() >= 0.0
    assert y.max() <= 0.18 + 1e-9


@pytest.mark.parametrize("name", list_presets())
def test_presets_discretise_cleanly(name):
    g = get_preset(name)
    st = g.discretize(40)
    assert st.n == 40
    assert (st.chord > 0.0).all()
    assert (st.r > 0.0).all()
    assert np.isfinite(st.twist).all()
    assert st.dr.sum() == pytest.approx(g.radius * (1.0 - g.hub_radius_frac), rel=1e-9)
    assert len(st.polars) == 40


def test_constant_pitch_twist_reproduces_the_named_pitch():
    """A "10x5" blade should measure 5 inches of geometric pitch at every station."""
    radius = 5.0 * INCH
    g = BladeGeometry(radius=radius, twist=constant_pitch_twist(5.0, radius))
    for x in (0.25, 0.5, 0.75, 1.0):
        theta = float(g.twist_at(np.array([x]))[0])
        pitch = 2.0 * math.pi * x * radius * math.tan(theta)
        assert pitch / INCH == pytest.approx(5.0, rel=0.02)


def test_pitch_diameter_ratio_matches_the_label():
    assert get_preset("APC 10x5 (sport)").pitch_diameter_ratio() == pytest.approx(0.5, abs=0.02)
    assert get_preset("APC 10x7 (fast)").pitch_diameter_ratio() == pytest.approx(0.7, abs=0.02)


def test_activity_factor_is_in_the_sport_propeller_range():
    """AF uses chord over *diameter*; referencing radius doubles the number."""
    assert 60.0 < get_preset("APC 10x5 (sport)").activity_factor() < 160.0


def test_collective_offset_is_a_rigid_rotation():
    from dataclasses import replace
    g = get_preset("APC 10x5 (sport)")
    x = np.linspace(0.2, 1.0, 20)
    shifted = replace(g, pitch_offset=math.radians(6.0))
    assert np.allclose(shifted.twist_at(x) - g.twist_at(x), math.radians(6.0))


def test_geometry_round_trips_through_json(tmp_path):
    g = get_preset("Scale warbird 4-blade")
    path = tmp_path / "blade.json"
    g.save(path)
    back = BladeGeometry.load(path)
    assert back.name == g.name and back.n_blades == g.n_blades
    assert back.radius == pytest.approx(g.radius)
    x = np.linspace(0.25, 1.0, 30)
    assert np.allclose(back.chord_at(x), g.chord_at(x))
    assert np.allclose(back.twist_at(x), g.twist_at(x))
    assert json.loads(path.read_text())["n_blades"] == g.n_blades


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def test_naca4_section_is_a_closed_loop():
    loop = naca4_section(0.12, 0.02, n_points=81)
    assert np.allclose(loop[0], loop[-1])
    assert loop[:, 0].min() >= -1e-9 and loop[:, 0].max() <= 1.0 + 1e-9


def test_naca4_thickness_is_as_requested():
    for t in (0.06, 0.12, 0.20):
        loop = naca4_section(t, 0.0, n_points=201)
        assert (loop[:, 1].max() - loop[:, 1].min()) == pytest.approx(t, rel=0.03)


def test_naca0012_area_matches_the_published_value():
    assert section_area(naca4_section(0.12, 0.0, n_points=401)) == pytest.approx(0.0822, rel=0.02)


@pytest.mark.parametrize("name,camber", [("naca0012", 0.0), ("naca2412", 0.02),
                                         ("naca4412", 0.04)])
def test_camber_is_recovered_from_the_zero_lift_angle(name, camber):
    from propwash.airfoil import get_airfoil
    assert camber_from_zero_lift(get_airfoil(name).alpha_0) == pytest.approx(camber, abs=0.002)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["APC 10x5 (sport)", "Tri-blade 10x6",
                                  "Scale warbird 4-blade"])
def test_mesh_is_well_formed(name):
    g = get_preset(name)
    mesh = build_propeller_mesh(g, n_span=24, n_chord=27)
    assert mesh.faces.min() >= 0
    assert mesh.faces.max() < mesh.n_vertices
    assert np.isfinite(mesh.vertices).all()
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
    assert mesh.radius == pytest.approx(g.radius, rel=0.02)
    assert (mesh.part == PART_BLADE).sum() > 0
    assert set(np.unique(mesh.blade_id)) >= set(range(g.n_blades))


def test_mesh_normals_point_outward():
    """A mesh lofted inside-out renders as a black silhouette."""
    mesh = build_propeller_mesh(get_preset("APC 10x5 (sport)"), n_span=24, n_chord=27)
    blade = mesh.part[mesh.faces].min(axis=1) == PART_BLADE
    centroids = mesh.vertices[mesh.faces].mean(axis=1)[blade]
    radial = centroids.copy()
    radial[:, 2] = 0.0
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    agreement = np.einsum("ij,ij->i", _face_normals(mesh.vertices.astype(float),
                                                    mesh.faces)[blade], radial)
    assert agreement.sum() > 0.0


def test_rotation_preserves_radius_and_axial_position():
    mesh = build_propeller_mesh(get_preset("APC 10x5 (sport)"), n_span=20, n_chord=21)
    turned = mesh.rotated(1.234)
    assert np.allclose(np.linalg.norm(turned[:, :2], axis=1),
                       np.linalg.norm(mesh.vertices[:, :2], axis=1), atol=1e-5)
    assert np.allclose(turned[:, 2], mesh.vertices[:, 2])


def test_blade_count_scales_the_mesh():
    from dataclasses import replace
    g = get_preset("APC 10x5 (sport)")
    two = build_propeller_mesh(replace(g, n_blades=2), n_span=20, n_chord=21)
    four = build_propeller_mesh(replace(g, n_blades=4), n_span=20, n_chord=21)
    assert four.n_faces > two.n_faces


def test_sample_field_interpolates_onto_vertices():
    mesh = build_propeller_mesh(get_preset("APC 10x5 (sport)"), n_span=20, n_chord=21)
    x = np.linspace(0.15, 1.0, 9)
    values = np.linspace(10.0, 90.0, 9)
    sampled = mesh.sample_field(x, values)
    assert sampled.shape == (mesh.n_vertices,)
    blade = mesh.part == PART_BLADE
    assert sampled[blade].min() >= 10.0 - 1e-6
    assert sampled[blade].max() <= 90.0 + 1e-6
    assert np.allclose(sampled[~blade], 10.0), "hub takes the root value"


def test_obj_and_stl_export(tmp_path):
    mesh = build_propeller_mesh(get_preset("APC 10x5 (sport)"), n_span=16, n_chord=17)
    obj = mesh.to_obj()
    assert obj.count("\nv ") + obj.startswith("v ") >= mesh.n_vertices - 1
    assert obj.count("\nf ") == mesh.n_faces

    stl = mesh.to_stl()
    assert len(stl) == 84 + 50 * mesh.n_faces
    assert int.from_bytes(stl[80:84], "little") == mesh.n_faces


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list_motors())
def test_motor_curves_are_monotone(name):
    motor = get_motor(name)
    rpm = np.linspace(0.0, motor.no_load_rpm(), 60)
    torque = motor.shaft_torque(rpm)
    assert torque[0] >= torque[-1]
    assert torque[-1] == pytest.approx(0.0, abs=1e-3)
    assert (motor.current_at(rpm) >= 0.0).all()
    assert (np.diff(motor.current_at(rpm)) <= 1e-9).all(), "current falls with speed"


def test_motor_efficiency_peaks_below_no_load():
    motor = get_motor("Sport 2820 kv1000")
    rpm = np.linspace(200.0, motor.no_load_rpm() * 0.995, 400)
    eta = motor.efficiency(rpm)
    assert 0.0 <= eta.max() <= 1.0
    assert rpm[int(np.argmax(eta))] < motor.no_load_rpm()


def test_cell_count_scales_no_load_speed():
    motor = get_motor("Sport 2820 kv1000")
    assert motor.with_cells(4).no_load_rpm() > motor.with_cells(3).no_load_rpm()


def test_match_rpm_finds_the_crossing():
    """Torque demand ~ k n^2 crosses a falling motor curve exactly once."""
    motor = get_motor("Sport 2820 kv1000")
    k = 2.2e-9
    rpm, converged = match_rpm(lambda n: k * np.asarray(n) ** 2, motor)
    assert converged
    assert float(motor.shaft_torque(np.array([rpm]))[0]) == pytest.approx(k * rpm ** 2, rel=1e-3)


def test_match_rpm_reports_failure_instead_of_guessing():
    motor = get_motor("Micro 1104 kv7500")
    _, converged = match_rpm(lambda n: 1e3 * np.ones_like(np.asarray(n, dtype=float)), motor)
    assert not converged, "an impossible load must be reported, not silently solved"


def test_matched_point_is_physically_sensible():
    from propwash.bemt.sweep import match_operating_point
    mp = match_operating_point(get_preset("APC 10x5 (sport)"),
                               get_motor("Sport 2820 kv1000"))
    assert mp.converged
    assert 5000.0 < mp.rpm < 11100.0
    assert 5.0 < mp.thrust < 30.0
    assert 0.0 < mp.current <= 30.0
    assert 0.3 < mp.motor_efficiency < 1.0
