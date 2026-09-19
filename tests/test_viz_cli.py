"""Colour tokens, figure construction and the command line."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from propwash.bemt.core import OperatingPoint          # noqa: E402
from propwash.bemt.solver import PropellerSolver       # noqa: E402
from propwash.bemt.sweep import j_sweep, pitch_rpm_map  # noqa: E402
from propwash.geometry import get_preset               # noqa: E402
from propwash.mesh import build_propeller_mesh         # noqa: E402
from propwash.viz import palette                       # noqa: E402

REPO = Path(__file__).resolve().parent.parent
HEX = re.compile(r"^#[0-9a-f]{6}$")


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ramp", [palette.CATEGORICAL_LIGHT, palette.CATEGORICAL_DARK,
                                  palette.SEQ_BLUE, palette.SEQ_ORANGE, palette.SEQ_RED])
def test_every_token_is_valid_hex(ramp):
    assert all(HEX.match(c) for c in ramp)


def test_sequential_ramp_darkens_monotonically():
    """A sequential ramp encodes magnitude, so lightness must be ordered."""
    rgb = palette.ramp_array(palette.SEQ_BLUE)
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    assert (np.diff(luminance) < 0.0).all()


def test_diverging_ramp_has_a_neutral_middle():
    ramp = palette.diverging_ramp(dark=False, n_arm=7)
    assert ramp.shape == (15, 3)
    middle = ramp[7]
    assert middle.max() - middle.min() < 0.05, "midpoint must read as neutral, not a hue"
    assert ramp[0][2] > ramp[0][0], "cool arm is blue"
    assert ramp[-1][0] > ramp[-1][2], "warm arm is red"


def test_categorical_slots_are_stable_and_never_cycle_silently():
    assert palette.series_color(0) == palette.CATEGORICAL_LIGHT[0]
    assert palette.series_color(8) == palette.CATEGORICAL_LIGHT[0], "wraps after eight"
    assert palette.series_color(0, dark=True) == palette.CATEGORICAL_DARK[0]


def test_map_values_stays_in_range_and_handles_degenerate_input():
    out = palette.map_values(np.linspace(-5.0, 5.0, 64))
    assert out.shape == (64, 4)
    assert out.min() >= 0.0 and out.max() <= 1.0
    flat = palette.map_values(np.full(10, 3.0))
    assert np.isfinite(flat).all()
    with_nan = palette.map_values(np.array([np.nan, 1.0, 2.0]), vmin=0.0, vmax=2.0)
    assert np.isfinite(with_nan[1:]).all()


def test_plotly_colorscale_is_well_formed():
    scale = palette.plotly_colorscale(palette.SEQ_BLUE)
    assert scale[0][0] == 0.0 and scale[-1][0] == 1.0
    assert all(s[1].startswith("rgb(") for s in scale)


def test_every_view_field_has_a_style():
    from propwash.colab.app import VIEW_FIELDS
    from propwash.gui.app import VIEW_FIELDS as GUI_FIELDS
    for name in set(VIEW_FIELDS) | set(GUI_FIELDS):
        assert name in palette.FIELD_STYLES, f"{name} has no display style"
        assert palette.field_label(name)


def test_field_ramp_picks_diverging_for_signed_quantities():
    assert palette.field_ramp("alpha").shape[0] == 15
    assert palette.field_ramp("mach").shape[0] == len(palette.SEQ_BLUE)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def solved():
    prop = get_preset("APC 10x5 (sport)")
    solver = PropellerSolver(prop)
    return prop, solver, solver.solve(OperatingPoint(rpm=9000.0, v_inf=10.0))


def test_matplotlib_figures_build(solved, tmp_path):
    import matplotlib.pyplot as plt
    from propwash.viz import plots
    prop, solver, result = solved

    figures = [
        plots.spanwise_figure(result, solver.stations),
        plots.j_sweep_figure(j_sweep(prop, rpm=9000.0, n=24)),
        plots.blade_planform_figure(solver.stations),
        plots.polar_figure(solver.stations.polars[10]),
        plots.map_figure(pitch_rpm_map(prop, np.linspace(-4, 8, 6),
                                       np.linspace(3000, 11000, 8)),
                         "thrust", mark_best="thrust"),
    ]
    for i, fig in enumerate(figures):
        assert fig.axes
        path = tmp_path / f"fig{i}.png"
        fig.savefig(path, dpi=60)
        assert path.stat().st_size > 2000
        plt.close(fig)


def test_matching_figure_marks_the_crossing(tmp_path):
    import matplotlib.pyplot as plt
    from propwash.motor import get_motor
    from propwash.viz.plots import matching_figure
    rpm = np.linspace(500.0, 11000.0, 40)
    fig = matching_figure(rpm, 2.2e-9 * rpm ** 2,
                          get_motor("Sport 2820 kv1000").shaft_torque(rpm), 9000.0)
    assert fig.axes
    plt.close(fig)


def test_dark_mode_figures_use_the_dark_surface(solved):
    import matplotlib.pyplot as plt
    from propwash.viz.plots import spanwise_figure
    _, solver, result = solved
    fig = spanwise_figure(result, solver.stations, dark=True)
    assert fig.get_facecolor()[:3] == pytest.approx(
        palette.hex_to_rgb(palette.DARK["surface"]), abs=0.01)
    plt.close(fig)


def test_plotly_propeller_figure(solved):
    pytest.importorskip("plotly")
    from propwash.viz.plotly3d import propeller_figure, split_faces_by_part
    prop, _, result = solved
    mesh = build_propeller_mesh(prop, n_span=20, n_chord=21)

    blade_faces, hub_faces = split_faces_by_part(mesh)
    assert blade_faces.size and hub_faces.size
    assert blade_faces.shape[0] + hub_faces.shape[0] == mesh.n_faces

    fig = propeller_figure(mesh, result, "dt_dr", animate=True, n_frames=6)
    assert len(fig.data) == 2, "blade and spinner are separate traces"
    assert len(fig.frames) == 6
    assert fig.data[0].intensity is not None
    assert fig.data[1].intensity is None, "the spinner carries no solver data"


def test_plotly_diverging_field_is_centred_on_zero(solved):
    pytest.importorskip("plotly")
    from propwash.viz.plotly3d import propeller_figure
    prop, _, result = solved
    fig = propeller_figure(build_propeller_mesh(prop, n_span=16, n_chord=17),
                           result, "alpha")
    assert fig.data[0].cmin == pytest.approx(-fig.data[0].cmax)


def test_plotly_helper_figures(solved):
    pytest.importorskip("plotly")
    from propwash.viz.plotly3d import curve_figure, heatmap_figure, spanwise_line_figure
    prop, _, result = solved
    assert spanwise_line_figure(result, ("cl", "alpha")).data
    assert curve_figure(np.arange(5.0), {"a": np.arange(5.0)}, "x", "y").data
    assert heatmap_figure(pitch_rpm_map(prop, np.linspace(0, 6, 4),
                                        np.linspace(4000, 9000, 5)), "thrust").data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run(*args, timeout: int = 420):
    return subprocess.run([sys.executable, "-m", "propwash", *args],
                          capture_output=True, text=True, timeout=timeout, cwd=REPO)


def test_cli_version_and_help():
    assert _run("--version").returncode == 0
    assert "propeller" in _run("--help").stdout.lower()


def test_cli_env_reports_backends():
    out = _run("env")
    assert out.returncode == 0
    assert "Compute backends" in out.stdout
    assert "numpy" in out.stdout


def test_cli_presets_lists_everything():
    out = _run("presets")
    assert out.returncode == 0
    for token in ("APC 10x5 (sport)", "Clark Y", "Sport 2820 kv1000"):
        assert token in out.stdout


def test_cli_solve_writes_outputs(tmp_path):
    csv = tmp_path / "span.csv"
    js = tmp_path / "summary.json"
    out = _run("solve", "--rpm", "8000", "--speed", "10",
               "--spanwise", str(csv), "--json", str(js))
    assert out.returncode == 0, out.stderr
    assert csv.exists() and js.exists()
    import json
    assert json.loads(js.read_text())["thrust_N"] > 0.0


def test_cli_match_and_sweep(tmp_path):
    out = _run("match", "--motor", "Sport 2820 kv1000")
    assert out.returncode == 0 and "shaft speed" in out.stdout

    plot = tmp_path / "sweep.png"
    out = _run("sweep", "--points", "16", "--plot", str(plot))
    assert out.returncode == 0 and "Peak efficiency" in out.stdout
    assert plot.stat().st_size > 5000


def test_cli_mesh_export(tmp_path):
    out = _run("mesh", "--span", "16", "--chord", "17", "-o", str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert list(tmp_path.glob("*.obj")) and list(tmp_path.glob("*.stl"))


def test_cli_bench_is_small_but_real():
    out = _run("bench", "--cases", "64", "--elements", "16", "--bisect", "20")
    assert out.returncode == 0, out.stderr
    assert "numpy" in out.stdout and "Mres/s" in out.stdout


def test_cli_colab_explains_itself_outside_a_notebook():
    out = _run("colab")
    assert out.returncode == 0
    assert "notebook" in out.stdout.lower()
