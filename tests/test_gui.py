"""Desktop GUI tests.

Skipped wholesale when Qt or a display is unavailable, which is the normal case
on a headless CI box without Xvfb.  When they do run they exercise the real
window: building it, driving the controls, and checking that the solver
responds the way the physics says it should.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture(scope="module")
def window(qapp):
    from propwash.gui.app import PropwashWindow
    return PropwashWindow(dark=True, n_elements=24, mesh_span=16, mesh_chord=17)


def test_window_builds_and_solves(window):
    assert window.result is not None
    assert window.result.thrust > 0.0
    assert window.mesh is not None and window.mesh.n_faces > 0
    # The default is a cruise-pitched aircraft propeller at V = 0, where BEMT
    # under-predicts power absorption and the match reports an over-rev rather
    # than converging.  That is the honest result, so assert it is *reported*,
    # not that it converges.
    assert window.match is not None
    assert window.match.rpm > 0.0


def test_metric_cards_are_populated(window):
    for name, card in window.cards.items():
        assert card.value.text() not in ("", None), name


def test_colour_legend_tracks_the_selected_field(window):
    """Colour without a scale is decoration, so the legend must follow the field."""
    for index in range(window.c_field.count()):
        window.c_field.setCurrentIndex(index)
        window._refresh_view()
        assert window.legend._label, window.c_field.currentData()
        assert window.legend._ramp is not None
        assert window.legend._vmax >= window.legend._vmin


def test_diverging_field_legend_is_centred_on_zero(window):
    window.c_field.setCurrentIndex(
        [window.c_field.itemData(i) for i in range(window.c_field.count())].index("alpha"))
    window._refresh_view()
    assert window.legend._vmin == pytest.approx(-window.legend._vmax)


def test_default_pairs_an_aircraft_propeller_with_a_piston_engine(window):
    """A 75-inch propeller on a 1000 kv drone motor is not a case worth showing."""
    from propwash.geometry import AIRCRAFT_PRESETS
    assert window.geometry.name in AIRCRAFT_PRESETS
    assert window.c_pptype.currentIndex() == 0
    assert "fuel_gph" in window.match.extras
    assert 1500.0 < window.result.rpm < 3000.0


def test_switching_to_a_model_propeller_switches_the_powerplant(window):
    window.c_preset.setCurrentText("APC 10x5 (sport)")
    window._run_solve()
    assert window.c_pptype.currentIndex() == 1
    assert window.result.rpm > 5000.0
    window.c_preset.setCurrentText("Cessna 172 (McCauley 75x57)")
    window._run_solve()


def test_collective_loads_the_powerplant_down(window):
    """More pitch always costs RPM; that holds stalled or not."""
    window.s_pitch.set_value(0.0, notify=True)
    window._run_solve()
    base_rpm = window.result.rpm

    window.s_pitch.set_value(5.0, notify=True)
    window._run_solve()
    assert window.result.rpm < base_rpm

    window.s_pitch.set_value(0.0, notify=True)
    window._run_solve()


def test_collective_adds_thrust_in_forward_flight(window):
    """Thrust rises with pitch while the blade is attached.

    At V = 0 a cruise-pitched propeller is already near stall, so adding
    collective can *reduce* thrust -- correct physics, but the wrong place to
    test the coupling.  In cruise the sections are attached and the sign is
    unambiguous.
    """
    window.s_speed.set_value(50.0, notify=True)
    window.s_pitch.set_value(-2.0, notify=True)
    window._run_solve()
    flat = window.result.thrust

    window.s_pitch.set_value(2.0, notify=True)
    window._run_solve()
    assert window.result.thrust > flat

    window.s_pitch.set_value(0.0, notify=True)
    window.s_speed.set_value(0.0, notify=True)
    window._run_solve()


def test_fixed_rpm_mode_releases_the_slider(window):
    window.b_match.setChecked(False)
    window._run_solve()
    assert window.match is None
    assert window.s_rpm.slider.isEnabled()
    window.s_rpm.set_value(7000.0, notify=True)
    window._run_solve()
    assert window.result.rpm == pytest.approx(7000.0, abs=60.0)

    window.b_match.setChecked(True)
    window._run_solve()
    assert not window.s_rpm.slider.isEnabled()


def test_labeled_slider_round_trips_engineering_units(qapp):
    from propwash.gui.panels import LabeledSlider
    seen = []
    slider = LabeledSlider("Pitch", -12.0, 18.0, 0.0, 0.25, "deg",
                           on_change=seen.append)
    slider.set_value(4.25, notify=True)
    assert slider.value == pytest.approx(4.25)
    assert seen == [pytest.approx(4.25)]
    assert "4.25" in slider.readout.text()

    slider.set_value(1.0, notify=False)
    assert slider.value == pytest.approx(1.0)
    assert len(seen) == 1, "a muted set must not fire the callback"


def test_vertex_lighting_keeps_colours_readable():
    """pyqtgraph's shaded shader crushes a sequential ramp to near-black."""
    from propwash.gui.gl_view import bake_lighting
    colors = np.tile(np.array([[0.2, 0.5, 0.9, 1.0]], dtype=np.float32), (64, 1))
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (64, 1))
    lit = bake_lighting(colors, normals)
    assert lit.shape == colors.shape
    assert (lit[:, :3] <= colors[:, :3] + 1e-6).all()
    assert lit[:, :3].max() > 0.35 * colors[:, :3].max(), "ambient floor must survive"
    assert np.allclose(lit[:, 3], 1.0)


def test_gl_field_colours_mark_the_hub_neutral():
    from propwash.bemt.core import OperatingPoint
    from propwash.bemt.solver import PropellerSolver
    from propwash.geometry import get_preset
    from propwash.gui.gl_view import field_vertex_colors
    from propwash.mesh import build_propeller_mesh
    from propwash.mesh.loft import PART_BLADE

    prop = get_preset("APC 10x5 (sport)")
    result = PropellerSolver(prop).solve(OperatingPoint(rpm=8000.0))
    mesh = build_propeller_mesh(prop, n_span=16, n_chord=17)

    colors, label, vmin, vmax = field_vertex_colors(mesh, result, "dt_dr")
    assert colors.shape == (mesh.n_vertices, 4)
    assert label and vmax > vmin
    # The hub colour is a warm grey (#b8b7b1), so test saturation rather than
    # exact channel equality: it must be near-neutral, the blade must not be.
    def saturation(rgb):
        return rgb.max(axis=1) - rgb.min(axis=1)

    hub = mesh.part != PART_BLADE
    assert saturation(colors[hub, :3]).max() < 0.05, "hub must read as neutral"
    assert saturation(colors[~hub, :3]).max() > 0.20, "blade must carry the ramp"
