"""The in-notebook GUI: ipywidgets controls driving a live Plotly 3-D view.

This is the interface on Colab, where there is no display server to open an
OpenGL window on.  Everything the desktop GUI does is here: the same solver,
the same mesh, the same colour tokens.

Two ideas shape the layout.

*Speed is an output, not an input.*  In **motor-matched** mode the RPM slider is
read-only -- the propeller spins at whatever speed makes its torque demand equal
the motor's torque supply.  Add pitch and it slows down; add cells and it speeds
up; climb to 3 km and it speeds up again because the air is thinner.  That
coupling is the thing the 3-D animation is showing, and the animation's frame
rate is proportional to the solved RPM so you *see* it.

*Only recompute what changed.*  Geometry edits rebuild the blade and the mesh;
operating-point edits reuse both.  Chart tabs render lazily, so dragging a
slider costs one solve and one Plotly vertex update.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from ..accel import list_backends
from ..atmosphere import isa
from ..bemt.core import BEMTResult, OperatingPoint, SolverOptions
from ..bemt.solver import PropellerSolver
from ..bemt.sweep import j_sweep, match_operating_point, pitch_rpm_map
from ..geometry import BladeGeometry, get_preset, list_presets
from ..mesh import build_propeller_mesh
from ..motor import MotorSpec, get_motor, list_motors
from ..units import INCH
from ..viz.palette import field_label, theme
from .bootstrap import enable_plotly_in_colab, probe

DEG = math.pi / 180.0

#: Fields offered for colouring the blade, in the order they make sense to try.
VIEW_FIELDS = ("dt_dr", "alpha", "cl", "mach", "dq_dr", "cd", "w",
               "loss_factor", "reynolds", "v_axial_induced", "v_swirl_induced")


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------

def _metric_card(label: str, value: str, sub: str = "", accent: str = "#2a78d6",
                 dark: bool = False) -> str:
    t = theme(dark)
    return f"""
    <div style="flex:1 1 118px;min-width:118px;background:{t['panel']};
                border-left:3px solid {accent};border-radius:6px;
                padding:8px 11px;margin:0;">
      <div style="font:600 10px/1.3 system-ui,sans-serif;letter-spacing:.055em;
                  text-transform:uppercase;color:{t['text_muted']}">{label}</div>
      <div style="font:600 19px/1.28 system-ui,sans-serif;color:{t['text']};
                  font-variant-numeric:tabular-nums">{value}</div>
      <div style="font:400 10px/1.3 system-ui,sans-serif;color:{t['text_secondary']}">{sub}</div>
    </div>"""


def _cards_html(cards: list[str], dark: bool = False) -> str:
    return (f"<div style='display:flex;flex-wrap:wrap;gap:7px;padding:2px 0 6px 0'>"
            f"{''.join(cards)}</div>")


def _banner(text: str, dark: bool = False, tone: str = "info") -> str:
    t = theme(dark)
    accent = {"info": "#2a78d6", "warn": "#eda100", "bad": "#e34948"}.get(tone, "#2a78d6")
    return (f"<div style='background:{t['panel']};border-left:3px solid {accent};"
            f"border-radius:6px;padding:7px 11px;font:400 11.5px/1.5 system-ui,"
            f"sans-serif;color:{t['text_secondary']}'>{text}</div>")


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------

class PropwashLab:
    """Interactive BEMT propeller laboratory for notebooks."""

    def __init__(self, preset: str | None = None, motor: str | None = None,
                 dark: bool = False, n_elements: int = 48,
                 mesh_span: int = 34, mesh_chord: int = 39,
                 backend: str = "auto") -> None:
        import ipywidgets as widgets

        enable_plotly_in_colab()
        self.W = widgets
        self.dark = bool(dark)
        self.mesh_span = int(mesh_span)
        self.mesh_chord = int(mesh_chord)

        self.geometry: BladeGeometry = get_preset(preset or list_presets()[0])
        self.motor: MotorSpec = get_motor(motor or list_motors()[2])
        self.options = SolverOptions(n_elements=int(n_elements))
        self.solver = PropellerSolver(self.geometry, self.options, backend)
        self.air = isa(0.0)
        self.result: BEMTResult | None = None
        self.match = None
        self.mesh = None

        self._busy = False
        self.tabs = None                  # created by ui(); charts check for it
        self._spin_thread: threading.Thread | None = None
        self._spinning = threading.Event()
        self._phase = 0.0
        self._chart_dirty = {"span": True, "sweep": True, "map": True, "match": True}

        self._build_widgets()
        self._rebuild_geometry(refresh=False)
        self.refresh(rebuild_mesh=True)

    # -- widget construction ----------------------------------------------
    def _build_widgets(self) -> None:
        w = self.W
        wide = w.Layout(width="97%")
        slim = w.Layout(width="97%")
        style = {"description_width": "104px"}

        # --- propeller -----------------------------------------------------
        self.w_preset = w.Dropdown(options=list_presets(), value=self.geometry.name,
                                   description="Preset", layout=wide, style=style)
        self.w_blades = w.IntSlider(value=self.geometry.n_blades, min=1, max=8, step=1,
                                    description="Blades", layout=slim, style=style,
                                    continuous_update=False)
        self.w_diameter = w.FloatSlider(value=self.geometry.diameter / INCH, min=3.0,
                                        max=30.0, step=0.25, description="Diameter [in]",
                                        layout=slim, style=style, readout_format=".2f",
                                        continuous_update=False)
        self.w_pitch = w.FloatSlider(value=0.0, min=-12.0, max=18.0, step=0.25,
                                     description="Collective [deg]", layout=slim,
                                     style=style, readout_format=".2f",
                                     continuous_update=False)
        self.w_chord_scale = w.FloatSlider(value=1.0, min=0.4, max=2.0, step=0.02,
                                           description="Chord scale", layout=slim,
                                           style=style, continuous_update=False)
        self.w_hub = w.FloatSlider(value=self.geometry.hub_radius_frac, min=0.05,
                                   max=0.40, step=0.01, description="Hub r/R",
                                   layout=slim, style=style, continuous_update=False)
        from ..airfoil import list_airfoils
        self.w_root_af = w.Dropdown(options=list_airfoils(), value=self.geometry.root_airfoil,
                                    description="Root section", layout=wide, style=style)
        self.w_tip_af = w.Dropdown(options=list_airfoils(), value=self.geometry.tip_airfoil,
                                   description="Tip section", layout=wide, style=style)

        # --- flight --------------------------------------------------------
        self.w_mode = w.ToggleButtons(
            options=[("Motor-matched", "match"), ("Fixed RPM", "fixed")],
            value="match", description="",
            tooltips=["Shaft speed is solved from the propeller/motor torque balance",
                      "You set the shaft speed directly"],
            layout=w.Layout(width="97%"))
        self.w_rpm = w.FloatSlider(value=8000.0, min=200.0, max=30000.0, step=50.0,
                                   description="Shaft speed", layout=slim, style=style,
                                   readout_format=".0f", continuous_update=False)
        self.w_speed = w.FloatSlider(value=0.0, min=0.0, max=80.0, step=0.5,
                                     description="Airspeed [m/s]", layout=slim,
                                     style=style, readout_format=".1f",
                                     continuous_update=False)
        self.w_alt = w.FloatSlider(value=0.0, min=0.0, max=8000.0, step=100.0,
                                   description="Altitude [m]", layout=slim, style=style,
                                   readout_format=".0f", continuous_update=False)
        self.w_disa = w.FloatSlider(value=0.0, min=-30.0, max=40.0, step=1.0,
                                    description="dISA [K]", layout=slim, style=style,
                                    continuous_update=False)

        # --- motor ---------------------------------------------------------
        self.w_motor = w.Dropdown(options=list_motors(), value=self.motor.name,
                                  description="Motor", layout=wide, style=style)
        self.w_kv = w.FloatSlider(value=self.motor.kv, min=100.0, max=9000.0, step=10.0,
                                  description="Kv [rpm/V]", layout=slim, style=style,
                                  readout_format=".0f", continuous_update=False)
        self.w_cells = w.IntSlider(value=self.motor.cells, min=1, max=12, step=1,
                                   description="LiPo cells", layout=slim, style=style,
                                   continuous_update=False)
        self.w_imax = w.FloatSlider(value=self.motor.max_current, min=2.0, max=120.0,
                                    step=1.0, description="Current limit [A]",
                                    layout=slim, style=style, continuous_update=False)
        self.w_throttle = w.FloatSlider(value=1.0, min=0.1, max=1.0, step=0.02,
                                        description="Throttle", layout=slim, style=style,
                                        readout_format=".2f", continuous_update=False)

        # --- solver --------------------------------------------------------
        backends = [i.name for i in list_backends() if i.available]
        self.w_backend = w.Dropdown(options=["auto"] + backends, value="auto",
                                    description="Backend", layout=wide, style=style)
        self.w_elements = w.IntSlider(value=self.options.n_elements, min=12, max=240,
                                      step=4, description="Elements", layout=slim,
                                      style=style, continuous_update=False)
        self.w_tiploss = w.Checkbox(value=True, description="Prandtl tip loss",
                                    indent=False)
        self.w_hubloss = w.Checkbox(value=True, description="Prandtl hub loss",
                                    indent=False)
        self.w_re = w.Checkbox(value=True, description="Reynolds correction", indent=False)
        self.w_mach = w.Checkbox(value=True, description="Compressibility", indent=False)

        # --- view ----------------------------------------------------------
        self.w_field = w.Dropdown(
            options=[(field_label(f), f) for f in VIEW_FIELDS],
            value="dt_dr", description="Colour by", layout=wide, style=style)
        self.w_spin = w.ToggleButton(value=False, description="Spin", icon="play",
                                     layout=w.Layout(width="46%"))
        self.w_dark = w.ToggleButton(value=self.dark, description="Dark",
                                     layout=w.Layout(width="46%"))
        self.w_slipstream = w.Checkbox(value=False, description="Show slipstream",
                                       indent=False)

        # --- outputs -------------------------------------------------------
        self.out_cards = w.HTML()
        self.out_status = w.HTML()
        self.live_view = True
        self.fig3d = self._make_figure_widget()
        self.out_span = w.Output()
        self.out_sweep = w.Output()
        self.out_map = w.Output()
        self.out_match = w.Output()
        self.out_bench = w.Output()
        self.w_bench_btn = w.Button(description="Run backend benchmark",
                                    button_style="", icon="bolt",
                                    layout=w.Layout(width="230px"))
        self.w_bench_n = w.IntSlider(value=4096, min=256, max=262144, step=256,
                                     description="Operating points",
                                     style={"description_width": "130px"},
                                     layout=w.Layout(width="420px"),
                                     continuous_update=False)
        self.w_export = w.Button(description="Export blade (OBJ/STL/CSV)", icon="download",
                                 layout=w.Layout(width="250px"))
        self.out_export = w.Output()

        self._wire()

    def _make_figure_widget(self):
        """A live FigureWidget when one is available, else an Output fallback.

        ``go.FigureWidget`` needs ipywidgets bindings (``anywidget`` on Plotly
        7).  When they are missing the view becomes a plain Output holding a
        normal figure, and the spin animation switches to Plotly's own
        client-side frames -- which needs no Python round-trip at all, so the
        fallback is arguably the smoother of the two.
        """
        import plotly.graph_objects as go
        try:
            fig = go.FigureWidget()
            self.live_view = True
            return fig
        except Exception:
            self.live_view = False
            return self.W.Output()

    def _wire(self) -> None:
        geo = (self.w_preset, self.w_blades, self.w_diameter, self.w_pitch,
               self.w_chord_scale, self.w_hub, self.w_root_af, self.w_tip_af)
        op = (self.w_mode, self.w_rpm, self.w_speed, self.w_alt, self.w_disa,
              self.w_motor, self.w_kv, self.w_cells, self.w_imax, self.w_throttle,
              self.w_backend, self.w_elements, self.w_tiploss, self.w_hubloss,
              self.w_re, self.w_mach)

        for widget in geo:
            widget.observe(self._on_geometry, names="value")
        for widget in op:
            widget.observe(self._on_operating, names="value")

        self.w_field.observe(self._on_view, names="value")
        self.w_slipstream.observe(self._on_view, names="value")
        self.w_dark.observe(self._on_theme, names="value")
        self.w_spin.observe(self._on_spin, names="value")
        self.w_bench_btn.on_click(self._on_bench)
        self.w_export.on_click(self._on_export)

    # -- state updates -----------------------------------------------------
    def _on_geometry(self, change) -> None:
        if self._busy:
            return
        if change["owner"] is self.w_preset:
            self._load_preset(change["new"])
        self._rebuild_geometry(refresh=False)
        self.refresh(rebuild_mesh=True)

    def _on_operating(self, change) -> None:
        if self._busy:
            return
        if change["owner"] is self.w_motor:
            self._load_motor(change["new"])
        if change["owner"] is self.w_backend:
            self.solver.use_backend(change["new"])
        if change["owner"] in (self.w_elements, self.w_tiploss, self.w_hubloss,
                               self.w_re, self.w_mach):
            self._apply_solver_options()
        self.refresh(rebuild_mesh=False)

    def _on_view(self, change) -> None:
        if not self._busy:
            self._refresh_3d()

    def _on_theme(self, change) -> None:
        self.dark = bool(change["new"])
        if not self._busy:
            self._mark_charts_dirty()
            self.refresh(rebuild_mesh=False)

    def _load_preset(self, name: str) -> None:
        self._busy = True
        try:
            g = get_preset(name)
            self.geometry = g
            self.w_blades.value = g.n_blades
            self.w_diameter.value = g.diameter / INCH
            self.w_hub.value = g.hub_radius_frac
            self.w_chord_scale.value = g.chord_scale
            self.w_pitch.value = math.degrees(g.pitch_offset)
            self.w_root_af.value = g.root_airfoil
            self.w_tip_af.value = g.tip_airfoil
        finally:
            self._busy = False

    def _load_motor(self, name: str) -> None:
        self._busy = True
        try:
            m = get_motor(name)
            self.motor = m
            self.w_kv.value = m.kv
            self.w_cells.value = m.cells
            self.w_imax.value = m.max_current
        finally:
            self._busy = False

    def _rebuild_geometry(self, refresh: bool = True) -> None:
        base = get_preset(self.w_preset.value)
        self.geometry = replace(
            base,
            n_blades=int(self.w_blades.value),
            radius=float(self.w_diameter.value) * INCH / 2.0,
            hub_radius_frac=float(self.w_hub.value),
            chord_scale=float(self.w_chord_scale.value),
            pitch_offset=float(self.w_pitch.value) * DEG,
            root_airfoil=self.w_root_af.value,
            tip_airfoil=self.w_tip_af.value,
        )
        self.solver.geometry = self.geometry
        if refresh:
            self.refresh(rebuild_mesh=True)

    def _apply_solver_options(self) -> None:
        self.options = SolverOptions(
            n_elements=int(self.w_elements.value),
            tip_loss=bool(self.w_tiploss.value),
            hub_loss=bool(self.w_hubloss.value),
            reynolds_correction=bool(self.w_re.value),
            mach_correction=bool(self.w_mach.value),
        )
        self.solver.options = self.options
        self.solver.invalidate()

    def _current_motor(self) -> MotorSpec:
        return replace(self.motor, kv=float(self.w_kv.value),
                       cells=int(self.w_cells.value),
                       voltage=3.7 * int(self.w_cells.value),
                       max_current=float(self.w_imax.value))

    # -- the solve ---------------------------------------------------------
    def refresh(self, rebuild_mesh: bool = False) -> None:
        """Re-solve and redraw everything that the last change invalidated."""
        t0 = time.perf_counter()
        self.air = isa(float(self.w_alt.value), delta_isa=float(self.w_disa.value))
        motor = self._current_motor()

        if self.w_mode.value == "match":
            self.match = match_operating_point(
                self.geometry, motor, v_inf=float(self.w_speed.value),
                throttle=float(self.w_throttle.value), air=self.air,
                options=self.options, backend=self.w_backend.value)
            rpm = self.match.rpm if self.match.rpm > 0 else 1.0
            self._busy = True
            try:
                self.w_rpm.value = float(np.clip(rpm, self.w_rpm.min, self.w_rpm.max))
                self.w_rpm.disabled = True
            finally:
                self._busy = False
        else:
            self.match = None
            self.w_rpm.disabled = False
            rpm = float(self.w_rpm.value)

        op = OperatingPoint(rpm=rpm, v_inf=float(self.w_speed.value), air=self.air)
        self.result = self.solver.solve(op)
        elapsed = time.perf_counter() - t0

        if rebuild_mesh or self.mesh is None:
            self.mesh = build_propeller_mesh(self.geometry, self.mesh_span,
                                             self.mesh_chord)

        self._mark_charts_dirty()
        self._refresh_cards(elapsed)
        self._refresh_3d(rebuilt=rebuild_mesh)
        self._refresh_active_chart()

    def _mark_charts_dirty(self) -> None:
        for key in self._chart_dirty:
            self._chart_dirty[key] = True

    # -- rendering ---------------------------------------------------------
    def _refresh_cards(self, elapsed: float) -> None:
        r = self.result
        if r is None:
            return
        dark = self.dark
        cards = [
            _metric_card("Shaft speed", f"{r.rpm:,.0f}",
                         "rpm — solved" if self.match else "rpm — set",
                         "#2a78d6", dark),
            _metric_card("Thrust", f"{r.thrust:,.2f} N",
                         f"{r.thrust / 9.80665 * 1000:,.0f} gf", "#eb6834", dark),
            _metric_card("Shaft power", f"{r.power:,.0f} W",
                         f"{r.torque:.4f} N m", "#1baf7a", dark),
        ]
        if r.v_inf > 0.05:
            cards.append(_metric_card("Efficiency", f"{r.efficiency * 100:.1f}%",
                                      f"J = {r.j:.3f}", "#eda100", dark))
        else:
            cards.append(_metric_card("Figure of merit", f"{r.figure_of_merit:.3f}",
                                      "hover (V = 0)", "#eda100", dark))
        if self.match is not None:
            cards.append(_metric_card("Current", f"{self.match.current:.1f} A",
                                      f"{self.match.electrical_power:.0f} W in",
                                      "#e87ba4", dark))
            cards.append(_metric_card("Motor eff.",
                                      f"{self.match.motor_efficiency * 100:.0f}%",
                                      f"throttle {self.w_throttle.value:.0%}",
                                      "#4a3aa7", dark))
        cards.append(_metric_card("Tip Mach", f"{r.tip_mach:.3f}",
                                  "helical", "#008300", dark))
        cards.append(_metric_card("Stalled span", f"{r.stall_fraction * 100:.0f}%",
                                  "past 14 deg",
                                  "#e34948" if r.stall_fraction > 0.25 else "#008300",
                                  dark))
        self.out_cards.value = _cards_html(cards, dark)

        notes = [f"{self.geometry.describe()}",
                 f"{self.air.describe()}",
                 f"solved {self.options.n_elements} elements on "
                 f"<b>{self.solver.backend.name}</b> in {elapsed * 1000:.0f} ms"]
        tone = "info"
        if r.converged_fraction < 0.999:
            notes.append(f"<b>{(1 - r.converged_fraction) * 100:.0f}% of the span is "
                         f"outside the thrusting branch</b> (windmilling or braking) "
                         f"— those elements are reported, not solved")
            tone = "warn"
        if self.match is not None and not self.match.converged:
            notes.append("<b>no torque balance</b> — the motor either cannot turn this "
                         "propeller or never loads up; try less pitch, a smaller "
                         "diameter or more cells")
            tone = "bad"
        self.out_status.value = _banner(" &nbsp;·&nbsp; ".join(notes), dark, tone)

    def _build_3d_figure(self, animate: bool):
        from ..viz.plotly3d import propeller_figure, slipstream_traces
        fig = propeller_figure(self.mesh, self.result, self.w_field.value,
                               dark=self.dark, animate=animate, n_frames=20,
                               rpm=self.result.rpm, height=560)
        if self.w_slipstream.value:
            for tr in slipstream_traces(self.result, self.geometry.radius,
                                        dark=self.dark):
                fig.add_trace(tr)
        return fig

    def _refresh_3d(self, rebuilt: bool = True) -> None:
        if self.mesh is None or self.result is None:
            return

        if not self.live_view:
            # Static-figure fallback: redraw, with client-side spin frames.
            from IPython.display import display
            with self.fig3d:
                self.fig3d.clear_output(wait=True)
                display(self._build_3d_figure(animate=bool(self.w_spin.value)))
            return

        if rebuilt or len(self.fig3d.data) == 0:
            fresh = self._build_3d_figure(animate=False)
            self.fig3d.data = []
            with self.fig3d.batch_update():
                for tr in fresh.data:
                    self.fig3d.add_trace(tr)
                self.fig3d.layout = fresh.layout
            return

        from ..viz.palette import plotly_colorscale
        from ..viz.plotly3d import mesh_intensity
        intensity, cb_title, ramp, vmin, vmax = mesh_intensity(
            self.mesh, self.result, self.w_field.value)
        with self.fig3d.batch_update():
            if intensity is not None:
                blade = self.fig3d.data[0]
                blade.intensity = intensity
                blade.cmin = vmin
                blade.cmax = vmax
                blade.colorscale = plotly_colorscale(ramp)
                blade.colorbar.title.text = cb_title

    def _refresh_active_chart(self) -> None:
        if self.tabs is None:
            return
        self._render_tab(self.tabs.selected_index)

    def _render_tab(self, index: int | None) -> None:
        if index is None:
            return
        names = ["3d", "span", "sweep", "map", "match", "bench", "export"]
        if index >= len(names):
            return
        name = names[index]
        if name in self._chart_dirty and self._chart_dirty[name]:
            {"span": self._draw_span, "sweep": self._draw_sweep,
             "map": self._draw_map, "match": self._draw_match}[name]()
            self._chart_dirty[name] = False

    def _draw(self, out, make: Callable[[], Any]) -> None:
        import matplotlib.pyplot as plt
        with out:
            out.clear_output(wait=True)
            try:
                fig = make()
                plt.show()
                plt.close(fig)
            except Exception as exc:
                print(f"{type(exc).__name__}: {exc}")

    def _draw_span(self) -> None:
        from ..viz.plots import spanwise_figure
        self._draw(self.out_span,
                   lambda: spanwise_figure(self.result, self.solver.stations,
                                           dark=self.dark))

    def _draw_sweep(self) -> None:
        from ..viz.plots import j_sweep_figure
        rpm = max(float(self.w_rpm.value), 100.0)
        self._draw(self.out_sweep,
                   lambda: j_sweep_figure(j_sweep(self.geometry, rpm=rpm, n=64,
                                                  air=self.air, options=self.options,
                                                  backend=self.w_backend.value),
                                          dark=self.dark))

    def _draw_map(self) -> None:
        from ..viz.plots import map_figure
        static = float(self.w_speed.value) < 0.05
        field = "figure_of_merit" if static else "efficiency"
        sweep = pitch_rpm_map(self.geometry,
                              np.linspace(-8.0, 14.0, 23),
                              np.linspace(1000.0, max(2000.0, self.w_rpm.max * 0.6), 36),
                              v_inf=float(self.w_speed.value), air=self.air,
                              options=self.options, backend=self.w_backend.value)
        self._draw(self.out_map,
                   lambda: map_figure(sweep, field, dark=self.dark, mark_best=field))

    def _draw_match(self) -> None:
        from ..viz.plots import matching_figure
        motor = self._current_motor()
        rpm = np.linspace(200.0, motor.no_load_rpm(float(self.w_throttle.value)) * 1.05, 70)
        prop_q = np.asarray(self.solver.solve_many(
            rpm, np.full(rpm.size, float(self.w_speed.value)), self.air)["torque"])
        motor_q = motor.shaft_torque(rpm, float(self.w_throttle.value))
        mrpm = self.match.rpm if self.match else None
        self._draw(self.out_match,
                   lambda: matching_figure(rpm, prop_q, motor_q, mrpm, dark=self.dark))

    # -- buttons -----------------------------------------------------------
    def _on_bench(self, _btn) -> None:
        from ..accel.bench import benchmark
        with self.out_bench:
            self.out_bench.clear_output(wait=True)
            print(f"Benchmarking {self.w_bench_n.value:,} operating points "
                  f"x {self.options.n_elements} elements ...\n")
            rep = benchmark(self.geometry, n_cases=int(self.w_bench_n.value),
                            n_elements=self.options.n_elements,
                            n_bisect=self.options.n_bisect, air=self.air, verbose=False)
            print(rep.format())

    def _on_export(self, _btn) -> None:
        from pathlib import Path
        with self.out_export:
            self.out_export.clear_output(wait=True)
            out = Path("exports")
            out.mkdir(exist_ok=True)
            stem = self.geometry.name.replace(" ", "_").replace("/", "-")
            (out / f"{stem}.obj").write_text(self.mesh.to_obj())
            (out / f"{stem}.stl").write_bytes(self.mesh.to_stl())
            self.geometry.save(out / f"{stem}.json")
            try:
                from ..bemt.solver import spanwise_frame
                spanwise_frame(self.result, self.solver.stations).to_csv(
                    out / f"{stem}_spanwise.csv", index=False)
            except Exception:
                pass
            print(f"Wrote to {out.resolve()}:")
            for p in sorted(out.glob(f"{stem}*")):
                print(f"  {p.name:<34} {p.stat().st_size / 1024:8.1f} kB")

    # -- animation ---------------------------------------------------------
    def _on_spin(self, change) -> None:
        if not self.live_view:
            # No live widget to push vertices into; rebuild with Plotly's own
            # play button instead.
            self._refresh_3d(rebuilt=True)
            return
        if change["new"]:
            self._start_spin()
        else:
            self._stop_spin()

    def _start_spin(self) -> None:
        if self._spin_thread is not None and self._spin_thread.is_alive():
            return
        self._spinning.set()
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    def _stop_spin(self) -> None:
        self._spinning.clear()

    def _spin_loop(self, fps: float = 18.0) -> None:
        """Rotate the mesh in a background thread.

        Real shaft speeds are 50-500 revolutions per second, so the display is
        slowed by a constant factor.  The factor is constant, which is the
        point: the *ratio* of on-screen speeds is the ratio of real RPMs, so
        adding pitch visibly slows the propeller down.
        """
        slowdown = 220.0
        dt = 1.0 / fps
        while self._spinning.is_set():
            try:
                rpm = float(self.result.rpm) if self.result else 0.0
                self._phase += 2.0 * math.pi * (rpm / 60.0) / slowdown * dt
                verts = self.mesh.rotated(self._phase)
                with self.fig3d.batch_update():
                    for trace in self.fig3d.data:
                        if trace.type == "mesh3d":
                            trace.x = verts[:, 0]
                            trace.y = verts[:, 1]
                            trace.z = verts[:, 2]
            except Exception:
                break
            time.sleep(dt)

    # -- assembly ----------------------------------------------------------
    def ui(self):
        """The assembled widget.  Display this."""
        w = self.W
        acc = w.Accordion(children=[
            w.VBox([self.w_preset, self.w_blades, self.w_diameter, self.w_pitch,
                    self.w_chord_scale, self.w_hub, self.w_root_af, self.w_tip_af]),
            w.VBox([self.w_mode, self.w_rpm, self.w_speed, self.w_alt, self.w_disa]),
            w.VBox([self.w_motor, self.w_kv, self.w_cells, self.w_imax, self.w_throttle]),
            w.VBox([self.w_backend, self.w_elements, self.w_tiploss, self.w_hubloss,
                    self.w_re, self.w_mach]),
            w.VBox([self.w_field, self.w_slipstream,
                    w.HBox([self.w_spin, self.w_dark])]),
        ])
        for i, title in enumerate(("Propeller", "Flight condition", "Motor",
                                   "Solver", "View")):
            acc.set_title(i, title)
        acc.selected_index = 0

        left = w.VBox([acc], layout=w.Layout(width="330px", flex="0 0 330px"))

        self.tabs = w.Tab(children=[
            self.fig3d,
            self.out_span,
            self.out_sweep,
            self.out_map,
            self.out_match,
            w.VBox([w.HBox([self.w_bench_btn, self.w_bench_n]), self.out_bench]),
            w.VBox([self.w_export, self.out_export]),
        ])
        for i, title in enumerate(("3-D blade", "Spanwise", "Advance ratio",
                                   "Pitch x RPM map", "Torque balance",
                                   "Benchmark", "Export")):
            self.tabs.set_title(i, title)
        self.tabs.observe(lambda ch: self._render_tab(ch["new"]), names="selected_index")

        right = w.VBox([self.out_cards, self.out_status, self.tabs],
                       layout=w.Layout(flex="1 1 auto", min_width="520px"))

        return w.HBox([left, right], layout=w.Layout(width="100%"))

    def display(self) -> None:
        from IPython.display import display
        display(self.ui())


def launch(preset: str | None = None, motor: str | None = None, dark: bool = False,
           show_environment: bool = True, **kw) -> PropwashLab:
    """Build the lab, display it, and hand the object back for scripting."""
    if show_environment:
        print(probe().report())
        print()
    lab = PropwashLab(preset=preset, motor=motor, dark=dark, **kw)
    lab.display()
    return lab


__all__ = ["PropwashLab", "launch", "VIEW_FIELDS"]
