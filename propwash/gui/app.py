"""The desktop application: Qt controls, an OpenGL blade and live 2-D plots.

Structure mirrors the notebook GUI on purpose -- same solver, same mesh, same
colours -- so that a result seen in one is the result seen in the other.

Responsiveness comes from two decisions:

* **Debounced solves.**  Dragging a slider fires dozens of change events; a
  single-shot timer coalesces them so the solver runs once when the drag
  settles rather than once per pixel.
* **Separated invalidation.**  Changing the operating point recolours existing
  geometry.  Only a geometry change rebuilds the mesh, which is the expensive
  part.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from typing import Any

import numpy as np

from ..airfoil import list_airfoils
from ..atmosphere import isa
from ..bemt.core import OperatingPoint, SolverOptions
from ..bemt.solver import PropellerSolver
from ..bemt.sweep import j_sweep, match_operating_point
from ..geometry import BladeGeometry, get_preset, list_presets
from ..mesh import build_propeller_mesh
from ..motor import get_motor, list_motors
from ..units import INCH
from ..viz.palette import field_label, series_color, theme
from .gl_view import PropellerGLView
from .panels import LabeledSlider, MetricCard, group, labelled
from .theme import configure_pyqtgraph, stylesheet

DEG = math.pi / 180.0

VIEW_FIELDS = ("dt_dr", "alpha", "cl", "mach", "dq_dr", "cd", "w",
               "loss_factor", "v_axial_induced", "v_swirl_induced")


def _qt():
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


class PropwashWindow:
    """Main window.  Not a QMainWindow subclass so the Qt import stays lazy."""

    def __init__(self, preset: str | None = None, motor: str | None = None,
                 dark: bool = True, n_elements: int = 48, mesh_span: int = 36,
                 mesh_chord: int = 41, backend: str = "auto") -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self.QtCore, self.QtGui, self.QtWidgets = QtCore, QtGui, QtWidgets

        self.dark = dark
        self.mesh_span, self.mesh_chord = int(mesh_span), int(mesh_chord)
        configure_pyqtgraph(dark)

        self.geometry: BladeGeometry = get_preset(preset or list_presets()[0])
        self.motor = get_motor(motor or list_motors()[2])
        self.options = SolverOptions(n_elements=int(n_elements))
        self.solver = PropellerSolver(self.geometry, self.options, backend)
        self.air = isa(0.0)
        self.result = None
        self.match = None
        self.mesh = None
        self._muted = False
        self._needs_mesh = True

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("Propwash -- BEMT propeller laboratory")
        self.window.resize(1500, 920)
        self.window.setStyleSheet(stylesheet(dark))

        self.view = PropellerGLView(dark)
        self._build_controls()
        self._build_plots()
        self._layout()

        # Coalesce rapid slider changes into one solve.
        self._timer = QtCore.QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._run_solve)

        self._rebuild_geometry()
        self._run_solve()

    # -- controls ----------------------------------------------------------
    def _build_controls(self) -> None:
        W = self.QtWidgets
        d = self.dark
        mark = lambda *_: self._schedule(False)     # noqa: E731
        remesh = lambda *_: self._schedule(True)    # noqa: E731

        self.c_preset = W.QComboBox()
        self.c_preset.addItems(list_presets())
        self.c_preset.setCurrentText(self.geometry.name)
        self.c_preset.currentTextChanged.connect(self._on_preset)

        self.s_blades = LabeledSlider("Blades", 1, 8, self.geometry.n_blades, 1,
                                      fmt="{:.0f}", on_change=remesh, dark=d)
        self.s_diameter = LabeledSlider("Diameter", 3, 30, self.geometry.diameter / INCH,
                                        0.25, "in", "{:.2f}", remesh, d)
        self.s_pitch = LabeledSlider("Collective", -12, 18, 0.0, 0.25, "deg",
                                     "{:+.2f}", remesh, d)
        self.s_chord = LabeledSlider("Chord scale", 0.4, 2.0, 1.0, 0.02, "x",
                                     "{:.2f}", remesh, d)
        self.s_hub = LabeledSlider("Hub", 0.05, 0.40, self.geometry.hub_radius_frac,
                                   0.01, "r/R", "{:.2f}", remesh, d)

        self.c_root = W.QComboBox(); self.c_root.addItems(list_airfoils())
        self.c_root.setCurrentText(self.geometry.root_airfoil)
        self.c_root.currentTextChanged.connect(remesh)
        self.c_tip = W.QComboBox(); self.c_tip.addItems(list_airfoils())
        self.c_tip.setCurrentText(self.geometry.tip_airfoil)
        self.c_tip.currentTextChanged.connect(remesh)

        self.b_match = W.QPushButton("Motor-matched")
        self.b_match.setCheckable(True); self.b_match.setChecked(True)
        self.b_match.setToolTip("Shaft speed is solved from the propeller/motor "
                                "torque balance instead of being set by hand")
        self.b_match.toggled.connect(mark)

        self.s_rpm = LabeledSlider("Shaft speed", 200, 30000, 8000, 50, "rpm",
                                   "{:,.0f}", mark, d)
        self.s_speed = LabeledSlider("Airspeed", 0, 80, 0.0, 0.5, "m/s", "{:.1f}", mark, d)
        self.s_alt = LabeledSlider("Altitude", 0, 8000, 0.0, 100, "m", "{:,.0f}", mark, d)
        self.s_disa = LabeledSlider("dISA", -30, 40, 0.0, 1, "K", "{:+.0f}", mark, d)

        self.c_motor = W.QComboBox(); self.c_motor.addItems(list_motors())
        self.c_motor.setCurrentText(self.motor.name)
        self.c_motor.currentTextChanged.connect(self._on_motor)
        self.s_kv = LabeledSlider("Kv", 100, 9000, self.motor.kv, 10, "rpm/V",
                                  "{:,.0f}", mark, d)
        self.s_cells = LabeledSlider("Cells", 1, 12, self.motor.cells, 1, "S",
                                     "{:.0f}", mark, d)
        self.s_imax = LabeledSlider("Current limit", 2, 120, self.motor.max_current,
                                    1, "A", "{:.0f}", mark, d)
        self.s_throttle = LabeledSlider("Throttle", 0.1, 1.0, 1.0, 0.02, "",
                                        "{:.0%}", mark, d)

        from ..accel import list_backends
        self.c_backend = W.QComboBox()
        self.c_backend.addItems(["auto"] + [i.name for i in list_backends() if i.available])
        self.c_backend.currentTextChanged.connect(self._on_backend)
        self.s_elements = LabeledSlider("Elements", 12, 240, self.options.n_elements,
                                        4, "", "{:.0f}", self._on_options, d)
        self.k_tip = W.QCheckBox("Prandtl tip loss"); self.k_tip.setChecked(True)
        self.k_hub = W.QCheckBox("Prandtl hub loss"); self.k_hub.setChecked(True)
        self.k_re = W.QCheckBox("Reynolds correction"); self.k_re.setChecked(True)
        self.k_mach = W.QCheckBox("Compressibility"); self.k_mach.setChecked(True)
        for box in (self.k_tip, self.k_hub, self.k_re, self.k_mach):
            box.toggled.connect(self._on_options)

        self.c_field = W.QComboBox()
        for name in VIEW_FIELDS:
            self.c_field.addItem(field_label(name), name)
        self.c_field.currentIndexChanged.connect(lambda *_: self._refresh_view())

        self.b_spin = W.QPushButton("Spin")
        self.b_spin.setCheckable(True)
        self.b_spin.toggled.connect(self.view.toggle)
        self.b_reset = W.QPushButton("Reset camera")
        self.b_reset.clicked.connect(self.view.reset_camera)
        self.b_export = W.QPushButton("Export OBJ / STL")
        self.b_export.clicked.connect(self._on_export)
        self.b_bench = W.QPushButton("Benchmark backends")
        self.b_bench.clicked.connect(self._on_bench)

        self.cards = {
            "rpm": MetricCard("Shaft speed", series_color(0, d), d),
            "thrust": MetricCard("Thrust", series_color(1, d), d),
            "power": MetricCard("Shaft power", series_color(2, d), d),
            "eta": MetricCard("Efficiency", series_color(3, d), d),
            "current": MetricCard("Current", series_color(4, d), d),
            "mach": MetricCard("Tip Mach", series_color(5, d), d),
            "stall": MetricCard("Stalled span", series_color(7, d), d),
        }

    # -- plots -------------------------------------------------------------
    def _build_plots(self) -> None:
        import pyqtgraph as pg
        t = theme(self.dark)
        self.plots: dict[str, Any] = {}
        self.curves: dict[str, Any] = {}

        def make(name: str, title: str, x_label: str, y_label: str,
                 si_prefix: bool = True):
            p = pg.PlotWidget(title=title)
            p.setLabel("bottom", x_label)
            p.setLabel("left", y_label)
            p.showGrid(x=True, y=True, alpha=0.18)
            p.getAxis("bottom").setPen(t["axis"])
            p.getAxis("left").setPen(t["axis"])
            # pyqtgraph rescales small numbers and hides the factor in the axis
            # label, so a 0.74 efficiency reads as "740".  Off for anything
            # dimensionless.
            if not si_prefix:
                p.getAxis("left").enableAutoSIPrefix(False)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            p.setMenuEnabled(False)
            self.plots[name] = p
            return p

        pen = lambda i, w=2: pg.mkPen(series_color(i, self.dark), width=w)  # noqa: E731

        p = make("angles", "Flow angles", "r / R", "deg", si_prefix=False)
        p.addLegend(offset=(-8, 8), labelTextColor=t["text_secondary"])
        self.curves["twist"] = p.plot(pen=pen(0), name="blade angle")
        self.curves["phi"] = p.plot(pen=pen(1), name="inflow")
        self.curves["alpha"] = p.plot(pen=pen(2), name="angle of attack")

        p = make("load", "Thrust loading", "r / R", "dT/dr [N/m]")
        self.curves["dt"] = p.plot(pen=pen(0),
                                   fillLevel=0.0, brush=(42, 120, 214, 55))

        p = make("cl", "Section lift coefficient", "r / R", "C_l", si_prefix=False)
        self.curves["cl"] = p.plot(pen=pen(0))

        p = make("torque", "Propeller / motor torque balance", "rpm", "torque [N m]")
        p.addLegend(offset=(-8, 8), labelTextColor=t["text_secondary"])
        self.curves["prop_q"] = p.plot(pen=pen(0), name="propeller demand")
        self.curves["motor_q"] = p.plot(pen=pen(1), name="motor supply")
        self.curves["match_pt"] = p.plot(pen=None, symbol="o", symbolSize=9,
                                         symbolBrush=t["text"], symbolPen=t["surface"])

        p = make("jsweep", "Efficiency vs advance ratio", "J", "eta", si_prefix=False)
        self.curves["eta"] = p.plot(pen=pen(2))
        self.curves["eta_pt"] = p.plot(pen=None, symbol="o", symbolSize=9,
                                       symbolBrush=t["text"], symbolPen=t["surface"])

    # -- layout ------------------------------------------------------------
    def _layout(self) -> None:
        W = self.QtWidgets
        d = self.dark

        panel = W.QWidget()
        col = W.QVBoxLayout(panel)
        col.setContentsMargins(8, 8, 8, 8)
        col.setSpacing(8)
        col.addWidget(group("Propeller", [
            labelled("Preset", self.c_preset, d), self.s_blades, self.s_diameter,
            self.s_pitch, self.s_chord, self.s_hub,
            labelled("Root section", self.c_root, d),
            labelled("Tip section", self.c_tip, d)]))
        col.addWidget(group("Flight condition", [
            self.b_match, self.s_rpm, self.s_speed, self.s_alt, self.s_disa]))
        col.addWidget(group("Motor", [
            labelled("Motor", self.c_motor, d), self.s_kv, self.s_cells,
            self.s_imax, self.s_throttle]))
        col.addWidget(group("Solver", [
            labelled("Backend", self.c_backend, d), self.s_elements,
            self.k_tip, self.k_hub, self.k_re, self.k_mach]))
        col.addWidget(group("View", [
            labelled("Colour by", self.c_field, d),
            [self.b_spin, self.b_reset], [self.b_export, self.b_bench]]))
        col.addStretch(1)

        scroll = W.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(352)
        scroll.setHorizontalScrollBarPolicy(self.QtCore.Qt.ScrollBarAlwaysOff)

        cards = W.QWidget()
        row = W.QHBoxLayout(cards)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)
        for card in self.cards.values():
            row.addWidget(card.widget)

        tabs = W.QTabWidget()
        tabs.addTab(self.view.widget, "3-D blade")

        span = W.QWidget()
        grid = W.QGridLayout(span)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.plots["angles"], 0, 0)
        grid.addWidget(self.plots["cl"], 0, 1)
        grid.addWidget(self.plots["load"], 1, 0)
        grid.addWidget(self.plots["jsweep"], 1, 1)
        tabs.addTab(span, "Spanwise and sweep")
        tabs.addTab(self.plots["torque"], "Torque balance")

        self.log = W.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:'JetBrains Mono',Menlo,monospace;"
                               "font-size:11px;")
        tabs.addTab(self.log, "Console")

        right = W.QWidget()
        rcol = W.QVBoxLayout(right)
        rcol.setContentsMargins(8, 10, 8, 4)
        rcol.setSpacing(8)
        rcol.addWidget(cards)
        rcol.addWidget(tabs, 1)

        central = W.QWidget()
        lay = W.QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(scroll)
        lay.addWidget(right, 1)
        self.window.setCentralWidget(central)
        self.status = self.window.statusBar()

    # -- events ------------------------------------------------------------
    def _schedule(self, remesh: bool) -> None:
        if self._muted:
            return
        self._needs_mesh = self._needs_mesh or remesh
        self._timer.start()

    def _on_preset(self, name: str) -> None:
        self._muted = True
        try:
            g = get_preset(name)
            self.s_blades.set_value(g.n_blades)
            self.s_diameter.set_value(g.diameter / INCH)
            self.s_hub.set_value(g.hub_radius_frac)
            self.s_chord.set_value(g.chord_scale)
            self.s_pitch.set_value(math.degrees(g.pitch_offset))
            self.c_root.setCurrentText(g.root_airfoil)
            self.c_tip.setCurrentText(g.tip_airfoil)
        finally:
            self._muted = False
        self._schedule(True)

    def _on_motor(self, name: str) -> None:
        self._muted = True
        try:
            m = get_motor(name)
            self.motor = m
            self.s_kv.set_value(m.kv)
            self.s_cells.set_value(m.cells)
            self.s_imax.set_value(m.max_current)
        finally:
            self._muted = False
        self._schedule(False)

    def _on_backend(self, name: str) -> None:
        self.solver.use_backend(name)
        self._schedule(False)

    def _on_options(self, *_a) -> None:
        self.options = SolverOptions(
            n_elements=int(self.s_elements.value),
            tip_loss=self.k_tip.isChecked(), hub_loss=self.k_hub.isChecked(),
            reynolds_correction=self.k_re.isChecked(),
            mach_correction=self.k_mach.isChecked())
        self.solver.options = self.options
        self.solver.invalidate()
        self._schedule(False)

    def _rebuild_geometry(self) -> None:
        base = get_preset(self.c_preset.currentText())
        self.geometry = replace(
            base, n_blades=int(self.s_blades.value),
            radius=self.s_diameter.value * INCH / 2.0,
            hub_radius_frac=self.s_hub.value,
            chord_scale=self.s_chord.value,
            pitch_offset=self.s_pitch.value * DEG,
            root_airfoil=self.c_root.currentText(),
            tip_airfoil=self.c_tip.currentText())
        self.solver.geometry = self.geometry

    def _current_motor(self):
        return replace(self.motor, kv=self.s_kv.value, cells=int(self.s_cells.value),
                       voltage=3.7 * int(self.s_cells.value),
                       max_current=self.s_imax.value)

    # -- the solve ---------------------------------------------------------
    def _run_solve(self) -> None:
        import time
        t0 = time.perf_counter()

        if self._needs_mesh:
            self._rebuild_geometry()

        self.air = isa(self.s_alt.value, delta_isa=self.s_disa.value)
        motor = self._current_motor()

        if self.b_match.isChecked():
            self.match = match_operating_point(
                self.geometry, motor, v_inf=self.s_speed.value,
                throttle=self.s_throttle.value, air=self.air,
                options=self.options, backend=self.c_backend.currentText())
            rpm = max(self.match.rpm, 1.0)
            self._muted = True
            try:
                self.s_rpm.set_value(rpm)
                self.s_rpm.set_enabled(False)
            finally:
                self._muted = False
        else:
            self.match = None
            self.s_rpm.set_enabled(True)
            rpm = self.s_rpm.value

        self.result = self.solver.solve(
            OperatingPoint(rpm=rpm, v_inf=self.s_speed.value, air=self.air))

        if self._needs_mesh or self.mesh is None:
            self.mesh = build_propeller_mesh(self.geometry, self.mesh_span,
                                             self.mesh_chord)
            self.view.set_mesh(self.mesh, self.result, self._field())
            self._needs_mesh = False

        self.view.set_rpm(self.result.rpm)
        self._refresh_view()
        self._refresh_cards()
        self._refresh_plots(motor)
        self.status.showMessage(
            f"{self.geometry.describe()}   |   {self.air.describe()}   |   "
            f"{self.solver.backend.name}, {self.options.n_elements} elements, "
            f"{(time.perf_counter() - t0) * 1000:.0f} ms")

    def _field(self) -> str:
        return self.c_field.currentData() or "dt_dr"

    def _refresh_view(self) -> None:
        if self.mesh is not None:
            self.view.set_field(self.result, self._field())

    def _refresh_cards(self) -> None:
        r = self.result
        self.cards["rpm"].set(f"{r.rpm:,.0f}",
                              "rpm - solved" if self.match else "rpm - set")
        self.cards["thrust"].set(f"{r.thrust:,.2f} N",
                                 f"{r.thrust / 9.80665 * 1000:,.0f} gf")
        self.cards["power"].set(f"{r.power:,.0f} W", f"{r.torque:.4f} N m")
        if r.v_inf > 0.05:
            self.cards["eta"].set(f"{r.efficiency * 100:.1f}%", f"J = {r.j:.3f}")
        else:
            self.cards["eta"].set(f"{r.figure_of_merit:.3f}", "figure of merit")
        if self.match is not None:
            self.cards["current"].set(f"{self.match.current:.1f} A",
                                      f"{self.match.electrical_power:.0f} W in")
        else:
            self.cards["current"].set("--", "fixed rpm")
        self.cards["mach"].set(f"{r.tip_mach:.3f}", "helical tip")
        self.cards["stall"].set(f"{r.stall_fraction * 100:.0f}%", "past 14 deg")

    def _refresh_plots(self, motor) -> None:
        r = self.result
        deg = 180.0 / math.pi
        st = self.solver.stations

        self.curves["twist"].setData(r.x, st.twist * deg)
        self.curves["phi"].setData(r.x, r.phi * deg)
        self.curves["alpha"].setData(r.x, r.alpha * deg)
        self.curves["dt"].setData(r.x, r.dt_dr)
        self.curves["cl"].setData(r.x, r.cl)

        rpm = np.linspace(200.0, motor.no_load_rpm(self.s_throttle.value) * 1.05, 60)
        prop_q = np.asarray(self.solver.solve_many(
            rpm, np.full(rpm.size, self.s_speed.value), self.air)["torque"])
        motor_q = motor.shaft_torque(rpm, self.s_throttle.value)
        self.curves["prop_q"].setData(rpm, prop_q)
        self.curves["motor_q"].setData(rpm, motor_q)
        if self.match is not None and self.match.rpm > 0:
            self.curves["match_pt"].setData([self.match.rpm],
                                            [float(np.interp(self.match.rpm, rpm, prop_q))])
        else:
            self.curves["match_pt"].setData([], [])

        sweep = j_sweep(self.geometry, rpm=max(r.rpm, 100.0), n=48, air=self.air,
                        options=self.options, backend=self.c_backend.currentText())
        eta = np.nan_to_num(sweep["efficiency"])
        self.curves["eta"].setData(sweep.axes["j"], eta)
        i = int(np.argmax(eta))
        self.curves["eta_pt"].setData([sweep.axes["j"][i]], [eta[i]])

    # -- buttons -----------------------------------------------------------
    def _on_export(self) -> None:
        from pathlib import Path
        out = Path("exports")
        out.mkdir(exist_ok=True)
        stem = self.geometry.name.replace(" ", "_").replace("/", "-")
        (out / f"{stem}.obj").write_text(self.mesh.to_obj())
        (out / f"{stem}.stl").write_bytes(self.mesh.to_stl())
        self.geometry.save(out / f"{stem}.json")
        self._log(f"Exported {stem}.obj / .stl / .json to {out.resolve()}")

    def _on_bench(self) -> None:
        from ..accel.bench import benchmark
        self._log("Benchmarking backends ...")
        self.QtWidgets.QApplication.processEvents()
        rep = benchmark(self.geometry, n_cases=4096,
                        n_elements=self.options.n_elements, air=self.air, verbose=False)
        self._log(rep.format())

    def _log(self, text: str) -> None:
        self.log.appendPlainText(text)
        self.status.showMessage(text.splitlines()[0][:160])

    def show(self) -> None:
        self.window.show()


def launch(preset: str | None = None, motor: str | None = None, dark: bool = True,
           **kw) -> int:
    """Create the QApplication, show the window and run the event loop."""
    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = PropwashWindow(preset=preset, motor=motor, dark=dark, **kw)
    win.show()
    return app.exec()


__all__ = ["PropwashWindow", "launch", "VIEW_FIELDS"]
