"""Blade construction from full-size specifications, and the 3-D mesh."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from propwash.geometry import (PLANFORMS, BladeGeometry, Distribution, full_size_blade,
                               helical_twist, pitch_ratio_from_beta75)
from propwash.mesh import build_propeller_mesh, naca4_section


def _blade(**kw):
    spec = dict(diameter=1.905, n_blades=2, activity_factor=109.0, hub_ratio=0.12,
                root_airfoil="clark_y", tip_airfoil="clark_y", thickness_root=0.20,
                thickness_tip=0.07, planform="standard", pitch=1.4478)
    spec.update(kw)
    return full_size_blade(**spec)


@pytest.mark.parametrize("kind", ["linear", "spline"])
def test_distributions_are_finite(kind):
    d = (Distribution("linear", root=0.2, tip=0.05) if kind == "linear" else
         Distribution("spline", control_x=(0.0, 0.5, 1.0), control_y=(0.1, 0.2, 0.05)))
    y = d.evaluate(np.linspace(0, 1, 50))
    assert np.isfinite(y).all() and (y > 0).all()


@pytest.mark.parametrize("planform", list(PLANFORMS))
@pytest.mark.parametrize("af", [80.0, 110.0, 160.0, 220.0])
def test_activity_factor_is_reproduced(planform, af):
    assert _blade(planform=planform, activity_factor=af).activity_factor(2000) == \
        pytest.approx(af, rel=0.003)


def test_geometric_pitch_is_reproduced():
    g = _blade()
    assert g.geometric_pitch() == pytest.approx(1.4478, rel=1e-3)
    assert g.pitch_diameter_ratio() == pytest.approx(57.0 / 75.0, rel=1e-3)


def test_beta75_and_pitch_ratio_agree():
    g = _blade(pitch=None, beta75_deg=20.0)
    assert math.degrees(g.beta75()) == pytest.approx(20.0, abs=0.01)
    assert g.pitch_diameter_ratio() == pytest.approx(pitch_ratio_from_beta75(20.0), rel=1e-3)


def test_helical_twist_is_constant_pitch():
    tw = helical_twist(0.8)
    x = np.linspace(0.3, 1.0, 8)
    pitch = np.pi * x * np.tan(tw.evaluate(x))
    assert np.allclose(pitch, 0.8, rtol=2e-3)


@pytest.mark.parametrize("bad", [dict(diameter=-1.0), dict(activity_factor=0.0),
                                 dict(planform="nope"), dict(root_airfoil="nope"),
                                 dict(beta75_deg=20.0)])
def test_bad_specifications_are_refused(bad):
    with pytest.raises((ValueError, KeyError)):
        _blade(**bad)


def test_blade_needs_every_field():
    with pytest.raises(TypeError):
        BladeGeometry(n_blades=2, radius=1.0)            # no default blade exists


def test_collective_offset_is_a_rigid_rotation():
    g = _blade()
    x = np.linspace(0.2, 1.0, 9)
    rot = replace(g, pitch_offset=math.radians(3.0))
    assert np.allclose(rot.twist_at(x) - g.twist_at(x), math.radians(3.0))


def test_geometry_round_trips_through_json(tmp_path):
    g = _blade()
    path = tmp_path / "blade.json"
    g.save(path)
    back = BladeGeometry.load(path)
    x = np.linspace(0.15, 1.0, 20)
    assert np.allclose(back.chord_at(x), g.chord_at(x))
    assert np.allclose(back.twist_at(x), g.twist_at(x))


def test_naca4_section_is_a_closed_loop():
    xy = naca4_section(0.12, 0.02, 0.4, 41)
    assert np.allclose(xy[0], xy[-1], atol=1e-3)


def test_mesh_is_well_formed():
    mesh = build_propeller_mesh(_blade(), n_span=20, n_chord=21)
    assert mesh.faces.max() < mesh.vertices.shape[0]
    assert np.isfinite(mesh.vertices).all()
    assert mesh.radius == pytest.approx(1.905 / 2.0, rel=0.02)


def test_blade_count_scales_the_mesh():
    two = build_propeller_mesh(_blade(), n_span=16, n_chord=17, spinner=False)
    three = build_propeller_mesh(_blade(n_blades=3), n_span=16, n_chord=17, spinner=False)
    assert three.vertices.shape[0] == pytest.approx(1.5 * two.vertices.shape[0])


def test_rotation_preserves_radius():
    mesh = build_propeller_mesh(_blade(), n_span=16, n_chord=17)
    v = mesh.rotated(0.7)
    r0 = np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1])
    r1 = np.hypot(v[:, 0], v[:, 1])
    assert np.allclose(r0, r1, atol=1e-5)


def test_obj_and_stl_export():
    mesh = build_propeller_mesh(_blade(), n_span=12, n_chord=13)
    assert mesh.to_obj().startswith("#") or "v " in mesh.to_obj()
    assert len(mesh.to_stl()) > 84
