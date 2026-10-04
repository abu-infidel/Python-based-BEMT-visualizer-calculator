"""Both front ends: blank forms, required-field marking, calculate, export."""

from __future__ import annotations

import os

import pytest

from propwash.case import FIELDS
from propwash.report import parse_report


# -- notebook app ------------------------------------------------------------------

@pytest.fixture
def notebook_app():
    pytest.importorskip("ipywidgets")
    from propwash.colab.app import PropwashApp
    return PropwashApp()


def test_notebook_form_starts_blank(notebook_app):
    assert all(v is None for v in notebook_app.values().values())


def test_notebook_inputs_are_free_text_not_sliders(notebook_app):
    import ipywidgets as W
    for f in FIELDS:
        w = notebook_app.inputs[f.key]
        assert isinstance(w, (W.Text, W.Dropdown)), f.key
        assert not isinstance(w, (W.FloatSlider, W.IntSlider, W.BoundedFloatText))


def test_notebook_marks_required_fields_from_the_choices(notebook_app):
    notebook_app.set_values({"case.mode": "analysis", "propeller.pitch_type": "constant_speed"})
    label = notebook_app.rows["propeller.fine_stop_deg"].children[0].value
    assert "*" in label
    assert notebook_app.rows["propeller.pitch_in"].layout.display == "none"


def test_notebook_calculates_and_exports(notebook_app, c172_case, tmp_path):
    notebook_app.set_values(dict(c172_case, **{"flight.airspeed_ktas": 0.0}))
    report = notebook_app.calculate()
    assert report.status in ("ok", "warning")
    path = notebook_app.export(path=str(tmp_path / "r.txt"))
    assert parse_report(path.read_text())["integrity"]["ok"]
    assert notebook_app.out_text.value.startswith("#PROPWASH-REPORT")


def test_notebook_accepts_values_far_beyond_old_slider_limits(notebook_app, c172_case):
    big = dict(c172_case, **{"engine.rated_power_hp": "1350", "engine.rated_rpm": "2000",
                             "propeller.diameter_in": "150", "propeller.blades": "4",
                             "propeller.pitch_in": "130", "flight.airspeed_ktas": "220",
                             "flight.altitude_ft": "15000"})
    notebook_app.set_values(big)
    report = notebook_app.calculate()
    assert report.get("INPUTS", "engine.rated_power_hp") == 1350.0
    assert report.sections["OPERATING_POINT"]["status"].value


def test_notebook_blank_calculate_lists_missing_fields(notebook_app):
    report = notebook_app.calculate()
    assert report.status == "error"
    assert any(n.code == "MISSING" for n in report.notices)


# -- desktop window ----------------------------------------------------------------

@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_desktop_window_round_trip(qt_app, c172_case, tmp_path):
    from propwash.gui.app import PropwashWindow
    win = PropwashWindow(dark=True)
    assert all(v is None for v in win.values().values())
    win.set_values(dict(c172_case, **{"flight.airspeed_ktas": 0.0}))
    report = win.calculate()
    assert report.status in ("ok", "warning")
    out = win.export(str(tmp_path / "desk.txt"))
    assert parse_report(open(out, encoding="utf-8").read())["integrity"]["ok"]
    inputs = win.save_inputs(str(tmp_path / "in.json"))
    win.clear()
    win.load_inputs(inputs)
    assert win.values()["propeller.pitch_in"] == "57"
