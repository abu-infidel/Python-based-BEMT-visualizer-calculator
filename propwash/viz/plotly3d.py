"""Plotly 3-D propeller view -- the renderer that works inside a notebook.

Colab cannot open an OpenGL window, so this is the primary visualiser there:
a ``Mesh3d`` of the lofted blade, coloured per vertex by whichever solver field
is interesting, optionally animated through one revolution at a frame rate
scaled from the solved RPM.

The animation ships rotated vertex positions rather than moving the camera.
It costs more data, but a camera orbit rotates the axes, the colour bar legend
and any reference geometry along with the blade, which reads as the *world*
spinning rather than the propeller.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ..bemt.core import BEMTResult
from ..mesh.loft import PART_BLADE, PropellerMesh
from .palette import (SEQ_BLUE, diverging_ramp, field_label, plotly_colorscale,
                      ramp_array, theme)

SPINNER_COLOR_LIGHT = "#b8b7b1"
SPINNER_COLOR_DARK = "#4a4a45"


def split_faces_by_part(mesh: PropellerMesh) -> tuple[np.ndarray, np.ndarray]:
    """Separate blade triangles from hub/spinner triangles.

    A face counts as hub only when every one of its vertices is hub, so the
    blade-root triangles that meet the spinner stay with the blade.
    """
    hub_vertex = mesh.part != PART_BLADE
    hub_face = hub_vertex[mesh.faces].all(axis=1)
    return mesh.faces[~hub_face], mesh.faces[hub_face]


def _scene(mesh: PropellerMesh, dark: bool, show_axes: bool = False) -> dict:
    t = theme(dark)
    r = mesh.radius * 1.02
    z_lo, z_hi = float(mesh.vertices[:, 2].min()), float(mesh.vertices[:, 2].max())
    z_mid = 0.5 * (z_lo + z_hi)
    z_half = max(0.5 * (z_hi - z_lo) * 1.25, 0.12 * r)
    axis = dict(showgrid=show_axes, zeroline=False, showticklabels=show_axes,
                showbackground=False, title="", visible=show_axes,
                color=t["text_muted"])
    # A propeller is a disk: give the scene box the blade's real aspect ratio and
    # pull the camera in, or Plotly's default framing wastes most of the canvas.
    return dict(
        xaxis=dict(axis, range=[-r, r]),
        yaxis=dict(axis, range=[-r, r]),
        zaxis=dict(axis, range=[z_mid - z_half, z_mid + z_half]),
        aspectmode="manual",
        aspectratio=dict(x=1.0, y=1.0, z=max(0.16, min(0.45, z_half / r))),
        camera=dict(eye=dict(x=0.62, y=-0.82, z=0.46),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    projection=dict(type="perspective")),
        bgcolor=t["surface"],
    )


LIGHTING = dict(ambient=0.42, diffuse=0.82, specular=0.28, roughness=0.55,
                fresnel=0.10)
LIGHT_POSITION = dict(x=1.2, y=-1.6, z=2.0)


def mesh_intensity(mesh: PropellerMesh, result: BEMTResult | None, field: str
                   ) -> tuple[np.ndarray | None, str, np.ndarray, float, float]:
    """Per-vertex colour data for ``field`` plus its ramp and range.

    Diverging fields (angle of attack, Cl) are centred on zero so the neutral
    midpoint means "no lift" rather than "middle of the data".
    """
    from .palette import FIELD_STYLES

    if result is None or not hasattr(result, field):
        return None, "", ramp_array(SEQ_BLUE), 0.0, 1.0

    values = np.asarray(getattr(result, field), dtype=float)
    style = FIELD_STYLES.get(field, {"scale": 1.0, "ramp": "sequential"})
    values = values * float(style.get("scale", 1.0))

    vertex_values = mesh.sample_field(result.x, values)

    if style.get("ramp") == "diverging":
        centre = float(style.get("center", 0.0))
        half = max(abs(values.max() - centre), abs(values.min() - centre), 1e-9)
        return vertex_values, field_label(field), diverging_ramp(), centre - half, centre + half

    return vertex_values, field_label(field), ramp_array(SEQ_BLUE), \
        float(values.min()), float(values.max())


def propeller_figure(mesh: PropellerMesh, result: BEMTResult | None = None,
                     field: str = "dt_dr", dark: bool = False,
                     animate: bool = False, n_frames: int = 24,
                     rpm: float | None = None, title: str | None = None,
                     show_axes: bool = False, height: int = 620):
    """Build the interactive 3-D propeller figure.

    ``animate`` adds a play button whose frame duration is derived from ``rpm``
    (or the solved RPM) so the on-screen rotation rate is *proportional to the
    real shaft speed* -- a faster-spinning propeller visibly spins faster,
    which is the whole point of the view.
    """
    import plotly.graph_objects as go

    t = theme(dark)
    intensity, cb_title, ramp, vmin, vmax = mesh_intensity(mesh, result, field)
    spinner = SPINNER_COLOR_DARK if dark else SPINNER_COLOR_LIGHT

    v = mesh.vertices
    blade_faces, hub_faces = split_faces_by_part(mesh)

    shared = dict(flatshading=False, lighting=LIGHTING,
                  lightposition=LIGHT_POSITION, hoverinfo="skip", name="")

    blade_kw: dict[str, Any] = dict(
        i=blade_faces[:, 0], j=blade_faces[:, 1], k=blade_faces[:, 2], **shared)
    if intensity is not None:
        blade_kw.update(
            intensity=intensity, intensitymode="vertex",
            colorscale=plotly_colorscale(ramp), cmin=vmin, cmax=vmax,
            showscale=True,
            colorbar=dict(title=dict(text=cb_title, side="right",
                                     font=dict(size=11, color=t["text_secondary"])),
                          thickness=12, len=0.52, x=0.985, y=0.5, ypad=0,
                          tickfont=dict(size=10, color=t["text_secondary"]),
                          outlinewidth=0, bgcolor="rgba(0,0,0,0)"),
        )
    else:
        blade_kw.update(color=spinner, showscale=False)

    # The spinner carries no solver data, so it gets a flat neutral colour in a
    # trace of its own.  Folding it into the coloured mesh would either give it
    # a meaningless value or -- on a diverging scale -- paint it the most
    # saturated colour on screen.
    hub_kw = dict(i=hub_faces[:, 0], j=hub_faces[:, 1], k=hub_faces[:, 2],
                  color=spinner, showscale=False, **shared)

    def build(vertices: np.ndarray) -> list:
        traces = [go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                            **blade_kw)]
        if hub_faces.size:
            traces.append(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1],
                                    z=vertices[:, 2], **hub_kw))
        return traces

    fig = go.Figure(data=build(v))

    if animate:
        shaft_rpm = rpm if rpm is not None else (result.rpm if result else 6000.0)
        fig.frames = [go.Frame(name=str(n),
                               data=build(mesh.rotated(2.0 * math.pi * n / n_frames)))
                      for n in range(n_frames)]
        fig.update_layout(updatemenus=[_play_button(shaft_rpm, n_frames, dark)])

    fig.update_layout(
        scene=_scene(mesh, dark, show_axes),
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        margin=dict(l=0, r=0, t=46 if title else 8, b=0),
        height=height,
        title=dict(text=title or "", x=0.015, xanchor="left",
                   font=dict(size=14, color=t["text"])),
        font=dict(color=t["text_secondary"]),
        showlegend=False,
    )
    return fig


def _play_button(rpm: float, n_frames: int, dark: bool) -> dict:
    """Play/pause control whose frame duration tracks the real shaft speed.

    One animation revolution is slowed by a fixed factor from the real one --
    a 9,000 rpm propeller is 150 revolutions per second and would be a blur --
    but the *ratio* between two RPMs is preserved, so the speed-up you see when
    you add pitch or volts is the speed-up that is really happening.
    """
    slowdown = 220.0
    rev_seconds = max(60.0 / max(rpm, 1.0) * slowdown, 0.6)
    frame_ms = max(int(1000.0 * rev_seconds / max(n_frames, 1)), 20)
    t = theme(dark)
    return dict(
        type="buttons", showactive=False, direction="left",
        x=0.02, y=0.04, xanchor="left", yanchor="bottom",
        bgcolor=t["panel"], bordercolor=t["grid"], borderwidth=1,
        font=dict(color=t["text_secondary"], size=11),
        pad=dict(l=6, r=6, t=4, b=4),
        buttons=[
            dict(label=f"spin  ({rpm:,.0f} rpm)", method="animate",
                 args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                  fromcurrent=True, mode="immediate",
                                  transition=dict(duration=0))]),
            dict(label="stop", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0))]),
        ],
    )


def slipstream_traces(result: BEMTResult, radius: float, n_ring: int = 24,
                      scale: float = 0.006, dark: bool = False) -> list:
    """Cones showing the induced velocity field the blade leaves behind.

    Axial induction alone looks like a fan; adding the swirl component is what
    makes the slipstream read as a helix, and it is the part beginners are
    usually surprised by.
    """
    import plotly.graph_objects as go

    stride = max(result.x.size // 10, 1)
    r_s = result.r[::stride]
    va = result.v_axial_induced[::stride]
    vt = result.v_swirl_induced[::stride]

    ang = np.linspace(0.0, 2.0 * math.pi, n_ring, endpoint=False)
    xs, ys, zs, us, vs, ws = [], [], [], [], [], []
    for r_i, va_i, vt_i in zip(r_s, va, vt):
        ca, sa = np.cos(ang), np.sin(ang)
        xs.append(r_i * ca)
        ys.append(r_i * sa)
        zs.append(np.full(n_ring, -0.25 * radius))
        # Axial component along -Z (the slipstream goes aft), swirl tangential.
        us.append(-vt_i * sa * scale)
        vs.append(vt_i * ca * scale)
        ws.append(np.full(n_ring, -va_i * scale))

    return [go.Cone(
        x=np.concatenate(xs), y=np.concatenate(ys), z=np.concatenate(zs),
        u=np.concatenate(us), v=np.concatenate(vs), w=np.concatenate(ws),
        sizemode="absolute", sizeref=0.35 * radius, showscale=False,
        colorscale=plotly_colorscale(ramp_array(SEQ_BLUE)[3:]),
        opacity=0.55, anchor="tail", hoverinfo="skip", name="slipstream")]


def spanwise_line_figure(result: BEMTResult, fields: Sequence[str] = ("cl", "alpha"),
                         dark: bool = False, height: int = 260):
    """Compact Plotly line chart of spanwise fields, for the notebook GUI."""
    import plotly.graph_objects as go
    from .palette import FIELD_STYLES, series_color

    t = theme(dark)
    fig = go.Figure()
    for i, name in enumerate(fields):
        if not hasattr(result, name):
            continue
        style = FIELD_STYLES.get(name, {"scale": 1.0})
        y = np.asarray(getattr(result, name), dtype=float) * float(style.get("scale", 1.0))
        fig.add_trace(go.Scatter(
            x=result.x, y=y, mode="lines", name=field_label(name),
            line=dict(color=series_color(i, dark), width=2.2),
            hovertemplate="r/R %{x:.3f}<br>%{y:.4g}<extra></extra>"))

    fig.update_layout(
        height=height, margin=dict(l=54, r=16, t=28, b=40),
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        font=dict(color=t["text_secondary"], size=11),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.16, x=0, font=dict(size=10)),
        xaxis=dict(title="r / R", gridcolor=t["grid"], zeroline=False,
                   linecolor=t["axis"]),
        yaxis=dict(gridcolor=t["grid"], zeroline=True, zerolinecolor=t["axis"],
                   linecolor=t["axis"]),
    )
    return fig


def curve_figure(x: np.ndarray, series: dict[str, np.ndarray], x_title: str,
                 y_title: str, dark: bool = False, height: int = 280,
                 markers: dict[str, tuple[float, float]] | None = None):
    """Generic single-scale line chart used by the notebook dashboard."""
    import plotly.graph_objects as go
    from .palette import series_color

    t = theme(dark)
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=x, y=np.asarray(y, dtype=float), mode="lines", name=name,
            line=dict(color=series_color(i, dark), width=2.2),
            hovertemplate=f"{x_title} %{{x:.4g}}<br>{name} %{{y:.4g}}<extra></extra>"))

    for i, (name, (mx, my)) in enumerate((markers or {}).items()):
        fig.add_trace(go.Scatter(
            x=[mx], y=[my], mode="markers+text", name=name, text=[name],
            textposition="top center", showlegend=False,
            textfont=dict(size=10, color=t["text"]),
            marker=dict(size=10, color=t["text"], line=dict(width=2, color=t["surface"]))))

    fig.update_layout(
        height=height, margin=dict(l=58, r=16, t=30, b=42),
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        font=dict(color=t["text_secondary"], size=11),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.18, x=0, font=dict(size=10)),
        xaxis=dict(title=x_title, gridcolor=t["grid"], zeroline=False,
                   linecolor=t["axis"]),
        yaxis=dict(title=y_title, gridcolor=t["grid"], zeroline=True,
                   zerolinecolor=t["axis"], linecolor=t["axis"]),
    )
    return fig


def heatmap_figure(sweep, field: str = "thrust", dark: bool = False, height: int = 380):
    """2-D parameter map as a Plotly heatmap with contour lines."""
    import plotly.graph_objects as go
    from .plots import axis_label

    t = theme(dark)
    names = sweep.axis_order
    fig = go.Figure(go.Contour(
        x=sweep.axes[names[1]], y=sweep.axes[names[0]],
        z=np.asarray(sweep[field], dtype=float),
        colorscale=plotly_colorscale(SEQ_BLUE),
        contours=dict(showlines=True, coloring="heatmap"),
        line=dict(width=0.7, color=t["surface"]),
        colorbar=dict(title=dict(text=axis_label(field), side="right",
                                 font=dict(size=11, color=t["text_secondary"])),
                      thickness=13, len=0.85, outlinewidth=0,
                      tickfont=dict(size=10, color=t["text_secondary"])),
        hovertemplate=f"{axis_label(names[1])} %{{x:.4g}}<br>"
                      f"{axis_label(names[0])} %{{y:.4g}}<br>"
                      f"{axis_label(field)} %{{z:.4g}}<extra></extra>"))
    fig.update_layout(
        height=height, margin=dict(l=62, r=16, t=30, b=46),
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        font=dict(color=t["text_secondary"], size=11),
        xaxis=dict(title=axis_label(names[1]), linecolor=t["axis"]),
        yaxis=dict(title=axis_label(names[0]), linecolor=t["axis"]),
    )
    return fig


__all__ = ["propeller_figure", "mesh_intensity", "split_faces_by_part", "slipstream_traces",
           "spanwise_line_figure", "curve_figure", "heatmap_figure"]
