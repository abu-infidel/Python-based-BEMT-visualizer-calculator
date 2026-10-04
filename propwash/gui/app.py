"""The desktop window (PySide6 + pyqtgraph): the same form as the notebook app.

Left: every input as a free text field or a choice box, all empty to start,
with the fields the current choices require marked ``*``.  Right: the spinning
3-D propeller coloured by its thrust loading, the results, performance against
airspeed, and the structured report that **Export .txt** saves.
"""

from __future__ import annotations

import html
import json
import sys
from typing import Any

import numpy as np

from ..case import FIELDS, GROUPS, blank_case, required_keys
from ..report import Report, parse_report
from .theme import configure_pyqtgraph, stylesheet

GROUP_TITLES = {
    "case": "Calculation", "flight": "Flight condition", "engine": "Engine",
    "propeller": "Propeller", "operating": "Power setting",
    "airframe": "Airframe (optional)", "sizing": "Sizing constraints",
}


def _visible(values: dict[str, Any]) -> set[str]:
    from ..colab.app import visible_keys
    return visible_keys(values)


class PropwashWindow:
    """Main window.  ``values()``, ``calculate()`` and ``report`` are public."""

    def __init__(self, case: dict[str, Any] | None = None, dark: bool = True) -> None:
        from pyqtgraph.Qt import QtCore, QtWidgets

        self.QtWidgets, self.QtCore = QtWidgets, QtCore
        self.dark = dark
        self.report: Report | None = None
        configure_pyqtgraph(dark)

        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("Propwash -- full-size propeller calculator")
        self.win.setStyleSheet(stylesheet(dark))
        self.win.resize(1500, 920)

        # -- form ---------------------------------------------------------
        self.inputs: dict[str, Any] = {}
        self.labels: dict[str, Any] = {}
        self.forms: dict[str, Any] = {}
        self.boxes: dict[str, Any] = {}
        form_host = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(form_host)
        for g in GROUPS:
            box = QtWidgets.QGroupBox(GROUP_TITLES[g])
            form = QtWidgets.QFormLayout(box)
            for f in (f for f in FIELDS if f.group == g):
                if f.kind == "choice":
                    w = QtWidgets.QComboBox()
                    w.addItem("-- choose --", None)
                    for c in f.choices:
                        w.addItem(c, c)
                    w.currentIndexChanged.connect(lambda _i: self._refresh())
                else:
                    w = QtWidgets.QLineEdit()
                    hint = f.unit or "text"
                    if f.typical:
                        hint += f"   typ. {f.typical[0]:g}-{f.typical[1]:g}"
                    w.setPlaceholderText(hint)
                w.setToolTip(f"{f.key}\n{f.help}")
                lab = QtWidgets.QLabel()
                lab.setToolTip(f.help)
                form.addRow(lab, w)
                self.inputs[f.key], self.labels[f.key] = w, lab
            self.forms[g], self.boxes[g] = form, box
            col.addWidget(box)
        col.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(form_host)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(470)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_calc = QtWidgets.QPushButton("Calculate")
        self.btn_export = QtWidgets.QPushButton("Export .txt")
        self.btn_save = QtWidgets.QPushButton("Save inputs")
        self.btn_load = QtWidgets.QPushButton("Load inputs")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_export.setEnabled(False)
        for b, fn in ((self.btn_calc, self.calculate), (self.btn_export, self.export),
                      (self.btn_save, self.save_inputs), (self.btn_load, self.load_inputs),
                      (self.btn_clear, self.clear)):
            b.clicked.connect(lambda _=False, fn=fn: fn())
            buttons.addWidget(b)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(scroll, 1)
        lv.addLayout(buttons)
        self.status = QtWidgets.QLabel("Fill in the form and press Calculate.")
        self.status.setWordWrap(True)
        lv.addWidget(self.status)

        # -- results ------------------------------------------------------
        self.tabs = QtWidgets.QTabWidget()
        self.view = None
        try:
            from .gl_view import PropellerGLView
            self.view = PropellerGLView(dark)
            self.tabs.addTab(self.view.widget, "3-D propeller")
        except Exception as exc:                   # no OpenGL: everything else still works
            self.tabs.addTab(QtWidgets.QLabel(f"3-D view unavailable: {exc}"), "3-D propeller")
        self.results = QtWidgets.QTextBrowser()
        self.tabs.addTab(self.results, "Results")
        import pyqtgraph as pg
        self.plot_force = pg.PlotWidget(title="Thrust and drag against airspeed")
        self.plot_power = pg.PlotWidget(title="Shaft power against airspeed")
        for p, y in ((self.plot_force, "lbf"), (self.plot_power, "hp")):
            p.setLabel("bottom", "true airspeed", units="kt")
            p.setLabel("left", y)
            p.addLegend()
        curves = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(curves)
        cv.addWidget(self.plot_force)
        cv.addWidget(self.plot_power)
        self.tabs.addTab(curves, "Against airspeed")
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.tabs.addTab(self.text, "Report (.txt)")

        split = QtWidgets.QSplitter()
        split.addWidget(left)
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        self.win.setCentralWidget(split)

        if case:
            self.set_values(case)
        self._refresh()

    # -- form ----------------------------------------------------------------
    def values(self) -> dict[str, Any]:
        out = {}
        for key, w in self.inputs.items():
            if isinstance(w, self.QtWidgets.QComboBox):
                out[key] = w.currentData()
            else:
                t = w.text().strip()
                out[key] = t or None
        return out

    def set_values(self, values: dict[str, Any]) -> None:
        for key, v in values.items():
            w = self.inputs.get(key)
            if w is None:
                continue
            if isinstance(w, self.QtWidgets.QComboBox):
                i = w.findData(v)
                w.setCurrentIndex(i if i >= 0 else 0)
            else:
                w.setText("" if v is None else f"{v:g}" if isinstance(v, float) else str(v))
        self._refresh()

    def _refresh(self) -> None:
        vals = self.values()
        req = set(required_keys(vals))
        show = _visible(vals)
        for f in FIELDS:
            unit = f" [{f.unit}]" if f.unit else ""
            star = "<b style='color:#e74c3c'>*</b> " if f.key in req else ""
            self.labels[f.key].setText(f"{star}{html.escape(f.label)}{unit}")
            self.forms[f.group].setRowVisible(self.inputs[f.key], f.key in show)
        for g, box in self.boxes.items():
            box.setVisible(any(f.key in show for f in FIELDS if f.group == g))

    def clear(self) -> None:
        self.set_values(blank_case())
        self.report = None
        self.btn_export.setEnabled(False)
        self.status.setText("Cleared.")

    # -- actions -------------------------------------------------------------
    def calculate(self) -> Report:
        from ..analysis import run

        self.status.setText("Calculating...")
        self.QtWidgets.QApplication.processEvents()
        self.report = report = run(self.values())
        text = report.to_text()
        parsed = parse_report(text)
        self.text.setPlainText(text)
        self.results.setHtml(self._results_html(report, parsed))
        self.btn_export.setEnabled(True)
        self.status.setText(f"Status: {report.status}  ({len(report.notices)} notices; "
                            "see Results)")
        self._draw(report, parsed)
        return report

    @staticmethod
    def _results_html(report: Report, parsed: dict) -> str:
        parts = [f"<h3>Status: {report.status}</h3><ul>"]
        for n in report.notices:
            parts.append(f"<li><b>{n.severity.upper()}</b> {html.escape(n.code)} "
                         f"{html.escape(n.key)}: {html.escape(n.message)}</li>")
        parts.append("</ul>")
        for name in ("OPERATING_POINT", "AIRFRAME", "DESIGN_POINT", "SIZING", "PROPELLER",
                     "ENGINE", "ATMOSPHERE"):
            sec = parsed["sections"].get(name)
            if not sec:
                continue
            parts.append(f"<h4>{name}</h4><table>")
            for key, item in sec.items():
                v = item["value"]
                text = (f"n/a ({item['code']})" if v is None else
                        f"{v:,.4g} {item['unit']}" if isinstance(v, float) else
                        f"{v} {item['unit']}")
                parts.append(f"<tr><td><code>{key}</code></td><td>&nbsp;{html.escape(text)}"
                             "</td></tr>")
            parts.append("</table>")
        return "".join(parts)

    def _draw(self, report: Report, parsed: dict) -> None:
        geom = report.extras.get("geometry")
        if self.view is not None and geom is not None:
            from ..mesh import build_propeller_mesh
            mesh = build_propeller_mesh(geom, n_span=40, n_chord=41)
            self.view.set_mesh(mesh, report.extras.get("result"), "dt_dr")
            rpm = report.extras.get("prop_rpm")
            if rpm:
                self.view.set_rpm(float(rpm))
                self.view.start()
            else:
                self.view.stop()
        self.plot_force.clear()
        self.plot_power.clear()
        sweep = parsed["tables"].get("SPEED_SWEEP")
        if not sweep:
            return
        names = [c["name"] for c in sweep["columns"]]
        col = {n: np.array([np.nan if r[i] is None else r[i] for r in sweep["rows"]],
                           dtype=float) for i, n in enumerate(names) if n != "status"}
        x = col["airspeed_ktas"]
        self.plot_force.plot(x, col["thrust_lbf"], pen=(42, 120, 214), name="thrust")
        if "drag_lbf" in col:
            self.plot_force.plot(x, col["drag_lbf"], pen=(214, 90, 42), name="drag")
        self.plot_power.plot(x, col["shaft_power_hp"], pen=(42, 160, 90), name="shaft power")

    def export(self, path: str | None = None) -> str | None:
        if self.report is None:
            return None
        if path is None:
            path, _ = self.QtWidgets.QFileDialog.getSaveFileName(
                self.win, "Export report", "propwash_report.txt", "Text report (*.txt)")
            if not path:
                return None
        self.report.save(path)
        self.status.setText(f"Report saved to {path}")
        return path

    def save_inputs(self, path: str | None = None) -> str | None:
        if path is None:
            path, _ = self.QtWidgets.QFileDialog.getSaveFileName(
                self.win, "Save inputs", "propwash_inputs.json", "JSON (*.json)")
            if not path:
                return None
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.values(), fh, indent=2)
        return path

    def load_inputs(self, path: str | None = None) -> None:
        if path is None:
            path, _ = self.QtWidgets.QFileDialog.getOpenFileName(
                self.win, "Load inputs", "", "JSON (*.json)")
            if not path:
                return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.clear()
        self.set_values(data.get("inputs", data) if isinstance(data, dict) else {})

    def show(self) -> None:
        self.win.show()


def launch(case: dict[str, Any] | None = None, dark: bool = True) -> int:
    """Create the QApplication, show the window and run the event loop."""
    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = PropwashWindow(case=case, dark=dark)
    win.show()
    return app.exec()


__all__ = ["PropwashWindow", "launch"]
