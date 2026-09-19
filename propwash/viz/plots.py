"""Matplotlib figures for the solver's output.

Every panel carries exactly one measure scale.  Where two quantities of
different units want comparing -- thrust and torque, Cl and Cd -- they get
adjacent small multiples rather than a second y-axis, because a twin axis lets
the author choose where the curves cross and is read as meaning by everyone
else.
"""

from __future__ import annotations

import math
import numpy as np

from ..bemt.core import BEMTResult
from ..geometry import BladeStations
from .palette import SEQ_BLUE, mpl_colormap, series_color, theme

DEG = 180.0 / math.pi
LINEWIDTH = 1.8
GRID_KW = dict(linewidth=0.6, alpha=0.9)

#: Axis and colour-bar captions, with units, for the names sweeps use as keys.
AXIS_LABELS = {
    "rpm": "shaft speed [rpm]", "v_inf": "airspeed [m/s]",
    "pitch_deg": "collective pitch [deg]", "j": "advance ratio $J$",
    "thrust": "thrust [N]", "torque": "torque [N m]", "power": "shaft power [W]",
    "efficiency": "propulsive efficiency", "figure_of_merit": "hover figure of merit",
    "ct": "$C_T$", "cq": "$C_Q$", "cp": "$C_P$", "tip_mach": "helical tip Mach",
    "disk_loading": "disk loading [N/m$^2$]", "stall_fraction": "stalled span fraction",
    "alpha": "angle of attack [deg]", "cl": "$C_l$", "cd": "$C_d$",
    "mach": "local Mach", "reynolds": "chord Reynolds number",
}


def axis_label(name: str) -> str:
    return AXIS_LABELS.get(name, name.replace("_", " "))


def style_axes(ax, dark: bool = False, grid_axis: str = "both") -> None:
    """Recessive frame: the data should be the only assertive thing on screen."""
    t = theme(dark)
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["text_secondary"], labelsize=8, length=3, width=0.8)
    ax.grid(True, axis=grid_axis, color=t["grid"], **GRID_KW)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["text_secondary"])
    ax.yaxis.label.set_color(t["text_secondary"])
    ax.title.set_color(t["text"])


def _legend(ax, dark: bool = False, **kw):
    t = theme(dark)
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=t["text_secondary"], **kw)
    return leg


def _direct_label(ax, x, y, text, color, dx=0.0, dy=0.0, fontsize=8):
    """Label a curve at its end -- identity without hunting through a legend."""
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), color=color,
                fontsize=fontsize, va="center", ha="left", annotation_clip=False)


def _label_curve_ends(ax, x_end, entries, dark=False, dx_frac=0.02,
                      min_gap_frac=0.075):
    """Direct-label several curves, pushing apart ends that would overlap.

    Curve ends often converge -- inflow angle and blade angle meet at the tip --
    and stacked labels are worse than no labels.  Sort by height and enforce a
    minimum vertical gap in axis-fraction units.
    """
    if not entries:
        return
    lo, hi = ax.get_ylim()
    span = max(hi - lo, 1e-12)
    gap = min_gap_frac * span

    ordered = sorted(entries, key=lambda e: e[1])
    placed: list[float] = []
    for _, y, _color in ordered:
        target = y if not placed else max(y, placed[-1] + gap)
        placed.append(target)

    x0, x1 = ax.get_xlim()
    dx = dx_frac * (x1 - x0)
    for (text, y, color), y_lab in zip(ordered, placed):
        ax.annotate(text, xy=(x_end, y), xytext=(x_end + dx, y_lab), color=color,
                    fontsize=8, va="center", ha="left", annotation_clip=False)
    ax.set_xlim(x0, x1 + 0.34 * (x1 - x0))


# ---------------------------------------------------------------------------
# Spanwise diagnostics
# ---------------------------------------------------------------------------

def spanwise_figure(result: BEMTResult, stations: BladeStations | None = None,
                    dark: bool = False, figsize: tuple[float, float] = (12.5, 7.0)):
    """Six single-scale panels describing what each blade element is doing."""
    import matplotlib.pyplot as plt

    t = theme(dark)
    fig, axes = plt.subplots(2, 3, figsize=figsize, facecolor=t["surface"])
    x = result.x

    # 1 -- angles (all degrees, one scale)
    ax = axes[0, 0]
    series = [("blade angle", (stations.twist * DEG) if stations is not None else None),
              ("inflow", result.phi * DEG),
              ("angle of attack", result.alpha * DEG)]
    ends = []
    for i, (name, y) in enumerate(series):
        if y is None:
            continue
        c = series_color(i, dark)
        ax.plot(x, y, color=c, linewidth=LINEWIDTH, label=name)
        ends.append((name, float(y[-1]), c))
    ax.axhline(0.0, color=t["axis"], linewidth=0.8)
    ax.set_title("Flow angles", fontsize=9, loc="left")
    ax.set_ylabel("degrees")
    style_axes(ax, dark)
    _label_curve_ends(ax, float(x[-1]), ends, dark)

    # 2 -- lift coefficient with the stall reference
    ax = axes[0, 1]
    c = series_color(0, dark)
    ax.plot(x, result.cl, color=c, linewidth=LINEWIDTH, label="$C_l$")
    ax.axhline(0.0, color=t["axis"], linewidth=0.8)
    stalled = result.alpha > math.radians(14.0)
    if stalled.any():
        ax.fill_between(x, result.cl.min(), result.cl.max(), where=stalled,
                        color=series_color(7, dark), alpha=0.12, linewidth=0,
                        label="past stall angle")
        _legend(ax, dark, loc="lower left")
    ax.set_title("Section lift coefficient", fontsize=9, loc="left")
    ax.set_ylabel("$C_l$")
    style_axes(ax, dark)

    # 3 -- lift-to-drag ratio
    ax = axes[0, 2]
    ld = result.cl / np.maximum(result.cd, 1e-9)
    ax.plot(x, ld, color=series_color(2, dark), linewidth=LINEWIDTH)
    ax.set_title("Section $L/D$", fontsize=9, loc="left")
    ax.set_ylabel("$C_l / C_d$")
    style_axes(ax, dark)

    # 4 -- thrust loading
    ax = axes[1, 0]
    c = series_color(0, dark)
    ax.fill_between(x, 0.0, result.dt_dr, color=c, alpha=0.18, linewidth=0)
    ax.plot(x, result.dt_dr, color=c, linewidth=LINEWIDTH)
    ax.axhline(0.0, color=t["axis"], linewidth=0.8)
    ax.set_title(f"Thrust loading  (total {result.thrust:.2f} N)", fontsize=9, loc="left")
    ax.set_ylabel("dT/dr  [N/m]")
    ax.set_xlabel("r / R")
    style_axes(ax, dark)

    # 5 -- torque loading
    ax = axes[1, 1]
    c = series_color(1, dark)
    ax.fill_between(x, 0.0, result.dq_dr, color=c, alpha=0.18, linewidth=0)
    ax.plot(x, result.dq_dr, color=c, linewidth=LINEWIDTH)
    ax.axhline(0.0, color=t["axis"], linewidth=0.8)
    ax.set_title(f"Torque loading  (total {result.torque:.4f} N m)", fontsize=9, loc="left")
    ax.set_ylabel("dQ/dr  [N]")
    ax.set_xlabel("r / R")
    style_axes(ax, dark)

    # 6 -- dimensionless: Mach and the Prandtl loss factor share [0, 1]
    ax = axes[1, 2]
    ends = []
    for i, (name, y) in enumerate((("local Mach", result.mach),
                                   ("tip/hub loss $F$", result.loss_factor))):
        c = series_color(i, dark)
        ax.plot(x, y, color=c, linewidth=LINEWIDTH, label=name)
        ends.append((name, float(y[-1]), c))
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Compressibility and tip loss", fontsize=9, loc="left")
    ax.set_xlabel("r / R")
    style_axes(ax, dark)
    _label_curve_ends(ax, float(x[-1]), ends, dark)

    fig.suptitle(
        f"{result.meta.get('blade', 'propeller')}  |  {result.rpm:,.0f} rpm, "
        f"V = {result.v_inf:.1f} m/s, J = {result.j:.3f}  |  "
        f"T = {result.thrust:.2f} N, P = {result.power:.0f} W, eta = {result.efficiency:.3f}",
        fontsize=10, color=t["text"], y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


# ---------------------------------------------------------------------------
# Performance curves
# ---------------------------------------------------------------------------

def j_sweep_figure(sweep, dark: bool = False, figsize: tuple[float, float] = (12.0, 3.6)):
    """The textbook chart: thrust, power and efficiency against advance ratio."""
    import matplotlib.pyplot as plt

    t = theme(dark)
    j = sweep.axes["j"]
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor=t["surface"])

    panels = (("Thrust coefficient", "$C_T$", sweep["ct"], 0),
              ("Power coefficient", "$C_P$", sweep["cp"], 1),
              ("Propulsive efficiency", r"$\eta$", sweep["efficiency"], 2))

    for ax, (title, ylab, y, slot) in zip(axes, panels):
        c = series_color(slot, dark)
        ax.plot(j, y, color=c, linewidth=LINEWIDTH)
        ax.axhline(0.0, color=t["axis"], linewidth=0.8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel(ylab)
        ax.set_xlabel("advance ratio $J$")
        style_axes(ax, dark)

    # Mark peak efficiency -- the number everyone is actually looking for.
    eta = np.nan_to_num(sweep["efficiency"])
    i = int(np.argmax(eta))
    ax = axes[2]
    ax.plot([j[i]], [eta[i]], "o", color=series_color(2, dark), markersize=6,
            markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=5)
    ax.annotate(f"peak {eta[i]:.3f}\nat J = {j[i]:.2f}", xy=(j[i], eta[i]),
                xytext=(6, -22), textcoords="offset points",
                fontsize=8, color=t["text_secondary"])
    ax.set_ylim(0.0, max(0.9, float(eta.max()) * 1.15))

    fig.suptitle(f"{sweep.meta.get('blade', 'propeller')} at "
                 f"{sweep.meta.get('rpm', 0):,.0f} rpm", fontsize=10,
                 color=t["text"], y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def map_figure(sweep, field: str = "thrust", dark: bool = False,
               figsize: tuple[float, float] = (6.4, 5.0), contours: bool = True,
               mark_best: str | None = None):
    """A 2-D parameter map as a single-hue heatmap with contour annotation."""
    import matplotlib.pyplot as plt

    t = theme(dark)
    ax_names = sweep.axis_order
    xa = sweep.axes[ax_names[1]]
    ya = sweep.axes[ax_names[0]]
    z = np.asarray(sweep[field], dtype=float)

    fig, ax = plt.subplots(figsize=figsize, facecolor=t["surface"])
    cmap = mpl_colormap(SEQ_BLUE)
    mesh = ax.pcolormesh(xa, ya, z, cmap=cmap, shading="gouraud")

    if contours:
        cs = ax.contour(xa, ya, z, levels=8, colors=t["surface"],
                        linewidths=0.7, alpha=0.65)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.3g")

    if mark_best:
        best = sweep.best(mark_best)
        bx, by = best[ax_names[1]], best[ax_names[0]]
        ax.plot([bx], [by], "o", markersize=8, markerfacecolor="none",
                markeredgecolor=t["surface"], markeredgewidth=2.0, zorder=6)
        # Flip the callout toward the interior when the optimum sits on an edge.
        right = bx > 0.5 * (xa[0] + xa[-1])
        top = by > 0.5 * (ya[0] + ya[-1])
        ax.annotate(f"best {axis_label(mark_best)} = {best[mark_best]:.3g}",
                    xy=(bx, by), xytext=(-10 if right else 10, -14 if top else 12),
                    textcoords="offset points", fontsize=8, color=t["surface"],
                    ha="right" if right else "left", zorder=6)

    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(axis_label(field), color=t["text_secondary"], fontsize=9)
    cb.ax.tick_params(colors=t["text_secondary"], labelsize=8)
    cb.outline.set_visible(False)

    ax.set_xlabel(axis_label(ax_names[1]))
    ax.set_ylabel(axis_label(ax_names[0]))
    ax.set_title(f"{axis_label(field)} over {axis_label(ax_names[0])} "
                 f"and {axis_label(ax_names[1])}", fontsize=10, loc="left",
                 color=t["text"])
    style_axes(ax, dark, grid_axis="both")
    ax.grid(False)
    fig.tight_layout()
    return fig


def polar_figure(polar, dark: bool = False, figsize: tuple[float, float] = (11.0, 3.4),
                 alpha_range: tuple[float, float] = (-25.0, 30.0)):
    """Cl, Cd and L/D against angle of attack for one section."""
    import matplotlib.pyplot as plt

    t = theme(dark)
    a = np.linspace(math.radians(alpha_range[0]), math.radians(alpha_range[1]), 601)
    cl, cd = polar.base_polar(a)
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor=t["surface"])

    for ax, (y, title, ylab, slot) in zip(axes, (
            (cl, "Lift", "$C_l$", 0),
            (cd, "Drag", "$C_d$", 1),
            (cl / np.maximum(cd, 1e-9), "Lift-to-drag", "$C_l / C_d$", 2))):
        c = series_color(slot, dark)
        ax.plot(a * DEG, y, color=c, linewidth=LINEWIDTH)
        ax.axhline(0.0, color=t["axis"], linewidth=0.8)
        ax.axvline(0.0, color=t["axis"], linewidth=0.8)
        ax.set_xlabel("angle of attack [deg]")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9, loc="left")
        style_axes(ax, dark)

    fig.suptitle(f"{polar.name}  (t/c = {polar.thickness:.3f}, "
                 f"Re_ref = {polar.re_ref:,.0f})", fontsize=10, color=t["text"], y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def matching_figure(rpm: np.ndarray, prop_torque: np.ndarray, motor_torque: np.ndarray,
                    match_rpm: float | None = None, dark: bool = False,
                    figsize: tuple[float, float] = (6.6, 4.2)):
    """Why the propeller spins at the speed it does.

    Two torque curves on one scale: the propeller's demand rising with RPM and
    the motor's supply falling toward its no-load speed.  Where they cross is
    the operating point -- and moving that crossing is what every slider in the
    GUI is really doing.
    """
    import matplotlib.pyplot as plt

    t = theme(dark)
    fig, ax = plt.subplots(figsize=figsize, facecolor=t["surface"])

    for i, (name, y) in enumerate((("propeller demand", prop_torque),
                                   ("motor supply", motor_torque))):
        c = series_color(i, dark)
        ax.plot(rpm, y, color=c, linewidth=LINEWIDTH, label=name)
        _direct_label(ax, rpm[-1], y[-1], name, c, dx=0.01 * (rpm[-1] - rpm[0]))

    if match_rpm:
        q = float(np.interp(match_rpm, rpm, prop_torque))
        ax.plot([match_rpm], [q], "o", markersize=8, color=t["text"],
                markeredgecolor=t["surface"], markeredgewidth=1.6, zorder=6)
        ax.annotate(f"operating point\n{match_rpm:,.0f} rpm", xy=(match_rpm, q),
                    xytext=(10, 14), textcoords="offset points", fontsize=8.5,
                    color=t["text"])
        ax.axvline(match_rpm, color=t["axis"], linewidth=0.8, linestyle=(0, (4, 4)))

    ax.set_xlabel("shaft speed [rpm]")
    ax.set_ylabel("torque [N m]")
    ax.set_title("Propeller / motor torque balance", fontsize=10, loc="left")
    ax.set_ylim(bottom=0.0)
    _legend(ax, dark, loc="upper left")
    style_axes(ax, dark)
    fig.tight_layout()
    return fig


def blade_planform_figure(stations: BladeStations, dark: bool = False,
                          figsize: tuple[float, float] = (11.0, 3.2)):
    """Planform, twist and thickness -- the three curves that define the blade."""
    import matplotlib.pyplot as plt

    t = theme(dark)
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor=t["surface"])
    x = stations.x

    ax = axes[0]
    c = series_color(0, dark)
    le = 0.25 * stations.chord * 1000
    te = -0.75 * stations.chord * 1000
    ax.fill_between(x, te, le, color=c, alpha=0.22, linewidth=0)
    ax.plot(x, le, color=c, linewidth=LINEWIDTH)
    ax.plot(x, te, color=c, linewidth=LINEWIDTH)
    ax.set_title("Planform (quarter-chord axis)", fontsize=9, loc="left")
    ax.set_ylabel("chordwise [mm]")

    axes[1].plot(x, stations.twist * DEG, color=series_color(1, dark), linewidth=LINEWIDTH)
    axes[1].set_title("Blade angle", fontsize=9, loc="left")
    axes[1].set_ylabel("degrees")

    axes[2].plot(x, stations.thickness * 100, color=series_color(2, dark),
                 linewidth=LINEWIDTH)
    axes[2].set_title("Thickness ratio", fontsize=9, loc="left")
    axes[2].set_ylabel("t/c [%]")

    for ax in axes:
        ax.set_xlabel("r / R")
        style_axes(ax, dark)

    fig.suptitle(stations.geometry.describe(), fontsize=10, color=t["text"], y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


__all__ = ["axis_label", "spanwise_figure", "j_sweep_figure", "map_figure", "polar_figure",
           "matching_figure", "blade_planform_figure", "style_axes"]
