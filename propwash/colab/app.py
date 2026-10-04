"""The notebook app (Colab / Jupyter): a form, a Calculate button, results.

Every input starts empty.  Text boxes accept any number -- there are no slider
limits -- and the form marks which fields the current choices make required.
Nothing is computed until **Calculate** is pressed, and the results can be
saved as the structured ``.txt`` report with **Export .txt**.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import re
from pathlib import Path
from typing import Any

from ..case import AIRFRAME_KEYS, FIELDS, GROUPS, blank_case, required_keys
from ..report import Report, parse_report
from .bootstrap import enable_plotly_in_colab, in_colab

GROUP_TITLES = {
    "case": "Calculation", "flight": "Flight condition", "engine": "Engine",
    "propeller": "Propeller", "operating": "Power setting",
    "airframe": "Airframe (optional: enables drag, climb, speed, ceilings)",
    "sizing": "Sizing constraints",
}
SEVERITY_COLOR = {"error": "#c0392b", "warning": "#b9770e", "info": "#2471a3", "ok": "#1e8449"}


def visible_keys(values: dict[str, Any]) -> set[str]:
    """Fields worth showing for the choices made so far."""
    keys = set(required_keys(values)) | {"case.name", "case.mode", "engine.bsfc_lb_hp_h"}
    if values.get("case.mode") != "sizing":
        keys |= set(AIRFRAME_KEYS)
    return keys


class PropwashApp:
    """Build with :func:`launch`; ``app.values()`` and ``app.report`` are public."""

    def __init__(self, dark: bool = False) -> None:
        import ipywidgets as W

        self.W = W
        self.dark = dark
        self.report: Report | None = None
        self.inputs: dict[str, Any] = {}
        self.rows: dict[str, Any] = {}

        label_w = "250px"
        for f in FIELDS:
            unit = f" [{f.unit}]" if f.unit else ""
            if f.kind == "choice":
                w = W.Dropdown(options=[("-- choose --", None)] + [(c, c) for c in f.choices],
                               value=None, layout=W.Layout(width="260px"))
                w.observe(self._on_choice, names="value")
            else:
                hint = f.unit or "text"
                if f.typical:
                    hint += f"  (typ. {f.typical[0]:g}-{f.typical[1]:g})"
                w = W.Text(value="", placeholder=hint, layout=W.Layout(width="260px"))
            label = W.HTML(layout=W.Layout(width=label_w))
            label.value = self._label(f, unit, False)
            help_ = W.HTML(f"<span style='color:#888;font-size:11px'>{html.escape(f.help)}</span>",
                           layout=W.Layout(width="420px"))
            self.inputs[f.key] = w
            self.rows[f.key] = W.HBox([label, w, help_], layout=W.Layout(align_items="center"))

        self.groups = {}
        for g in GROUPS:
            title = W.HTML(f"<h4 style='margin:10px 0 4px 0'>{GROUP_TITLES[g]}</h4>")
            members = [self.rows[f.key] for f in FIELDS if f.group == g]
            self.groups[g] = W.VBox([title, *members])

        self.btn_calc = W.Button(description="Calculate", button_style="primary", icon="play")
        self.btn_export = W.Button(description="Export .txt", icon="download", disabled=True)
        self.btn_save = W.Button(description="Save inputs .json", icon="save")
        self.btn_clear = W.Button(description="Clear form", icon="eraser")
        self.upload = W.FileUpload(accept=".json", multiple=False, description="Load .json")
        self.btn_calc.on_click(self.calculate)
        self.btn_export.on_click(self.export)
        self.btn_save.on_click(self.save_inputs)
        self.btn_clear.on_click(self.clear)
        self.upload.observe(self._on_upload, names="value")

        self.status = W.HTML()
        self.out_summary = W.Output()
        self.out_3d = W.Output()
        self.out_sweep = W.Output()
        self.out_blade = W.Output()
        self.out_text = W.Textarea(layout=W.Layout(width="100%", height="420px"),
                                   disabled=True)
        self.out_files = W.Output()
        self.tabs = W.Tab(children=[self.out_summary, self.out_3d, self.out_sweep,
                                    self.out_blade, self.out_text])
        for i, t in enumerate(("Results", "3-D propeller", "Against airspeed", "Blade",
                               "Report (.txt)")):
            self.tabs.set_title(i, t)

        self.form = W.VBox([self.groups[g] for g in GROUPS])
        buttons = W.HBox([self.btn_calc, self.btn_export, self.btn_save, self.upload,
                          self.btn_clear])
        header = W.HTML("<h3 style='margin:4px 0'>Propwash -- full-size propeller "
                        "calculator</h3><div style='color:#666'>All fields start empty. "
                        "Fields marked <b>*</b> are required for the choices made. Any value "
                        "is accepted; unusual ones produce a warning, not a limit.</div>")
        self.widget = W.VBox([header, self.form, buttons, self.status, self.out_files,
                              self.tabs])
        self._refresh()

    # -- form ----------------------------------------------------------------
    @staticmethod
    def _label(f, unit: str, required: bool) -> str:
        star = "<b style='color:#c0392b'>*</b> " if required else ""
        return f"<span title='{html.escape(f.key)}'>{star}{html.escape(f.label)}{unit}</span>"

    def values(self) -> dict[str, Any]:
        out = {}
        for key, w in self.inputs.items():
            v = w.value
            out[key] = None if (v is None or (isinstance(v, str) and not v.strip())) else v
        return out

    def set_values(self, values: dict[str, Any]) -> None:
        for key, v in values.items():
            if key not in self.inputs:
                continue
            w = self.inputs[key]
            if hasattr(w, "options"):
                w.value = v if v in [o[1] for o in w.options] else None
            else:
                w.value = "" if v is None else f"{v:g}" if isinstance(v, float) else str(v)
        self._refresh()

    def _on_choice(self, _change=None) -> None:
        self._refresh()

    def _refresh(self) -> None:
        vals = self.values()
        req = set(required_keys(vals))
        show = visible_keys(vals)
        for f in FIELDS:
            row = self.rows[f.key]
            row.layout.display = None if f.key in show else "none"
            unit = f" [{f.unit}]" if f.unit else ""
            row.children[0].value = self._label(f, unit, f.key in req)
        for g, box in self.groups.items():
            any_visible = any(f.key in show for f in FIELDS if f.group == g)
            box.layout.display = None if any_visible else "none"

    def clear(self, _=None) -> None:
        self.set_values(blank_case())
        self.report = None
        self.btn_export.disabled = True
        self.status.value = ""

    # -- actions -------------------------------------------------------------
    def calculate(self, _=None) -> Report:
        from ..analysis import run

        self.status.value = "<i>Calculating...</i>"
        self.report = report = run(self.values())
        self._show_status(report)
        self.btn_export.disabled = False     # an error report is a valid file too
        text = report.to_text()
        self.out_text.value = text
        parsed = parse_report(text)
        self._show_summary(parsed)
        self._show_figures(report, parsed)
        return report

    def _show_status(self, report: Report) -> None:
        color = SEVERITY_COLOR[report.status]
        items = "".join(
            f"<li><b style='color:{SEVERITY_COLOR[n.severity]}'>{n.severity.upper()}</b> "
            f"<code>{html.escape(n.code)}</code>"
            f"{(' <code>' + html.escape(n.key) + '</code>') if n.key else ''}: "
            f"{html.escape(n.message)}</li>" for n in report.notices)
        self.status.value = (f"<div style='border-left:4px solid {color};padding:4px 10px'>"
                             f"<b>Status: {report.status}</b><ul style='margin:4px 0'>{items}"
                             "</ul></div>")

    def _show_summary(self, parsed: dict) -> None:
        from IPython.display import HTML, display

        self.out_summary.clear_output()
        blocks = []
        for name in ("OPERATING_POINT", "AIRFRAME", "DESIGN_POINT", "SIZING", "PROPELLER",
                     "ENGINE", "ATMOSPHERE"):
            sec = parsed["sections"].get(name)
            if not sec:
                continue
            rows = []
            for key, item in sec.items():
                v = item["value"]
                if v is None:
                    text = f"<span style='color:#999'>n/a ({html.escape(item['code'] or '')})</span>"
                elif isinstance(v, float):
                    text = f"{v:,.4g} {html.escape(item['unit'])}"
                else:
                    text = f"{html.escape(str(v))} {html.escape(item['unit'])}"
                rows.append(f"<tr><td style='padding:1px 12px 1px 0'><code>{key}</code></td>"
                            f"<td>{text}</td></tr>")
            blocks.append(f"<h4 style='margin:8px 0 2px 0'>{name}</h4><table>"
                          + "".join(rows) + "</table>")
        with self.out_summary:
            display(HTML("".join(blocks) or "<i>No results (see the status above).</i>"))

    def _show_figures(self, report: Report, parsed: dict) -> None:
        from IPython.display import display

        for out in (self.out_3d, self.out_sweep, self.out_blade):
            out.clear_output()
        geom = report.extras.get("geometry")
        result = report.extras.get("result")
        try:
            if geom is not None:
                from ..mesh import build_propeller_mesh
                from ..viz.plotly3d import propeller_figure, spanwise_line_figure

                mesh = build_propeller_mesh(geom, n_span=36, n_chord=33)
                fig = propeller_figure(mesh, result, field="dt_dr", dark=self.dark,
                                       animate=result is not None,
                                       rpm=report.extras.get("prop_rpm"),
                                       title=f"{geom.n_blades}-blade, "
                                             f"{geom.diameter / 0.0254:.1f} in", height=560)
                with self.out_3d:
                    fig.show()
                if result is not None:
                    with self.out_blade:
                        spanwise_line_figure(result, ("cl", "alpha"), self.dark).show()
                        spanwise_line_figure(result, ("dt_dr", "dq_dr"), self.dark).show()
            sweep = parsed["tables"].get("SPEED_SWEEP")
            if sweep:
                import numpy as np

                from ..viz.plotly3d import curve_figure
                names = [c["name"] for c in sweep["columns"]]
                col = {n: np.array([np.nan if r[i] is None else r[i] for r in sweep["rows"]],
                                   dtype=float)
                       for i, n in enumerate(names) if n != "status"}
                x = col["airspeed_ktas"]
                with self.out_sweep:
                    curve_figure(x, {"thrust [lbf]": col["thrust_lbf"],
                                     **({"drag [lbf]": col["drag_lbf"]} if "drag_lbf" in col
                                        else {})},
                                 "true airspeed [kt]", "force [lbf]", self.dark).show()
                    curve_figure(x, {"shaft power [hp]": col["shaft_power_hp"]},
                                 "true airspeed [kt]", "power [hp]", self.dark).show()
                    curve_figure(x, {"propulsive efficiency": col["propulsive_efficiency"]},
                                 "true airspeed [kt]", "efficiency [-]", self.dark).show()
                    curve_figure(x, {"engine RPM": col["engine_rpm"]},
                                 "true airspeed [kt]", "RPM", self.dark).show()
            trade = parsed["tables"].get("DIAMETER_TRADE")
            if trade:
                import numpy as np

                from ..viz.plotly3d import curve_figure
                d = np.array([r[0] for r in trade["rows"]], dtype=float)
                eff = np.array([np.nan if r[4] is None else r[4] for r in trade["rows"]])
                froude = np.array([np.nan if r[5] is None else r[5] for r in trade["rows"]])
                with self.out_sweep:
                    curve_figure(d, {"best blade efficiency": eff, "momentum limit": froude},
                                 "diameter [in]", "efficiency [-]", self.dark).show()
        except Exception as exc:          # figures are a convenience; results stand
            with self.out_3d:
                display(f"Figure could not be drawn: {type(exc).__name__}: {exc}")

    def export(self, _=None, path: str | None = None) -> Path | None:
        """Write the structured report.  Downloads it too when running in Colab."""
        if self.report is None:
            self.status.value += "<div>Press Calculate first.</div>"
            return None
        if path is None:
            name = self.report.sections.get("META", {}).get("case_name")
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name.value))[:40] if (
                name and name.value) else "case"
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = f"propwash_{stem}_{stamp}.txt"
        out = Path(path)
        self.report.save(out)
        self._offer(out)
        return out

    def save_inputs(self, _=None, path: str | None = None) -> Path:
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(path or f"propwash_inputs_{stamp}.json")
        out.write_text(json.dumps(self.values(), indent=2) + "\n", encoding="utf-8")
        self._offer(out)
        return out

    def _offer(self, path: Path) -> None:
        from IPython.display import FileLink, display
        with self.out_files:
            print(f"saved {path.resolve()}")
            if in_colab():
                try:
                    from google.colab import files
                    files.download(str(path))
                except Exception:
                    pass
            else:
                display(FileLink(str(path)))

    def _on_upload(self, change) -> None:
        value = change["new"]
        items = list(value.values()) if isinstance(value, dict) else list(value)
        if not items:
            return
        content = items[0]["content"]
        data = json.loads(bytes(content).decode("utf-8"))
        if isinstance(data, dict) and isinstance(data.get("inputs"), dict):
            data = data["inputs"]
        self.clear()
        self.set_values(data)


def launch(dark: bool = False) -> PropwashApp:
    """Show the app in the current notebook cell and return it."""
    from IPython.display import display

    enable_plotly_in_colab()
    app = PropwashApp(dark=dark)
    display(app.widget)
    return app


__all__ = ["PropwashApp", "launch", "visible_keys"]
