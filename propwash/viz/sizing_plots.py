"""Charts for propeller sizing and design.

Two questions, two figures: *how big should the disk be* (a constrained trade,
not an optimum), and *what shape should the blade be* (the minimum-induced-loss
answer).  As everywhere else, one measure scale per panel -- power and Mach
number get their own axes rather than being stacked on a twin.
"""

from __future__ import annotations

import math

import numpy as np

from .palette import series_color, theme
from .plots import LINEWIDTH, style_axes

DEG = 180.0 / math.pi
INCH = 0.0254


def diameter_sweep_figure(sweep: dict, tip_mach_limit: float = 0.85,
                          chosen: float | None = None, dark: bool = False,
                          figsize: tuple[float, float] = (11.0, 3.6)):
    """Required power and tip Mach against diameter.

    Power falls monotonically with diameter, so there is no interior optimum --
    the choice is made by the constraints.  The shaded region is where the tip
    goes supercritical and the compressibility penalty starts eating the gain;
    ground clearance sets the other end.
    """
    import matplotlib.pyplot as plt

    t = theme(dark)
    d_in = sweep["diameter"] / INCH
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor=t["surface"])

    ax = axes[0]
    ax.plot(d_in, sweep["shaft_power_hp"], color=series_color(0, dark), linewidth=LINEWIDTH)
    ax.set_title("Required shaft power", fontsize=9, loc="left")
    ax.set_ylabel("shaft power [hp]")

    ax = axes[1]
    ax.plot(d_in, sweep["ideal_efficiency"] * 100.0, color=series_color(2, dark),
            linewidth=LINEWIDTH)
    ax.set_title("Froude (ideal) efficiency limit", fontsize=9, loc="left")
    ax.set_ylabel("percent")

    ax = axes[2]
    if "tip_mach" in sweep:
        ax.plot(d_in, sweep["tip_mach"], color=series_color(1, dark), linewidth=LINEWIDTH)
        ax.axhline(tip_mach_limit, color=t["axis"], linewidth=0.9, linestyle=(0, (4, 4)))
        over = sweep["tip_mach"] > tip_mach_limit
        if over.any():
            ax.fill_between(d_in, 0.0, sweep["tip_mach"], where=over,
                            color=series_color(7, dark), alpha=0.16, linewidth=0)
            first = d_in[np.argmax(over)]
            ax.annotate(f"tip goes supercritical\nabove {first:.0f} in",
                        xy=(first, tip_mach_limit), xytext=(-8, 14),
                        textcoords="offset points", ha="right", fontsize=8,
                        color=t["text_secondary"])
        ax.set_ylabel("helical tip Mach")
    else:
        ax.plot(d_in, sweep["disk_loading"], color=series_color(1, dark),
                linewidth=LINEWIDTH)
        ax.set_ylabel("disk loading [N/m$^2$]")
    ax.set_title("Tip Mach constraint", fontsize=9, loc="left")

    for ax in axes:
        ax.set_xlabel("diameter [in]")
        if chosen is not None:
            ax.axvline(chosen / INCH, color=t["text_muted"], linewidth=0.9,
                       linestyle=(0, (2, 3)))
        style_axes(ax, dark)

    fig.suptitle("Actuator-disk sizing: the trade is set by constraints, not an optimum",
                 fontsize=10, color=t["text"], y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def design_figure(design, compare=None, dark: bool = False,
                  figsize: tuple[float, float] = (11.0, 3.4)):
    """Chord, blade angle and loading of a minimum-induced-loss design.

    ``compare`` may be a :class:`~propwash.geometry.BladeStations` from an
    existing propeller, which is drawn alongside so the optimum can be read
    against something real.
    """
    import matplotlib.pyplot as plt

    t = theme(dark)
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor=t["surface"])

    ax = axes[0]
    ax.plot(design.x, design.chord / design.radius, color=series_color(0, dark),
            linewidth=LINEWIDTH, label="optimum")
    if compare is not None:
        ax.plot(compare.x, compare.chord / compare.geometry.radius,
                color=series_color(1, dark), linewidth=LINEWIDTH,
                linestyle=(0, (4, 3)), label=compare.geometry.name)
        ax.legend(frameon=False, fontsize=8, labelcolor=t["text_secondary"])
    ax.set_title("Chord", fontsize=9, loc="left")
    ax.set_ylabel("c / R")

    ax = axes[1]
    ax.plot(design.x, design.twist * DEG, color=series_color(0, dark),
            linewidth=LINEWIDTH, label="optimum")
    ax.plot(design.x, design.phi * DEG, color=series_color(2, dark),
            linewidth=LINEWIDTH, linestyle=(0, (1, 2)), label="flow angle")
    if compare is not None:
        ax.plot(compare.x, compare.twist * DEG, color=series_color(1, dark),
                linewidth=LINEWIDTH, linestyle=(0, (4, 3)), label=compare.geometry.name)
    ax.legend(frameon=False, fontsize=8, labelcolor=t["text_secondary"])
    ax.set_title("Blade angle", fontsize=9, loc="left")
    ax.set_ylabel("degrees")

    ax = axes[2]
    mach = design.meta.get("mach")
    if mach is not None:
        ax.plot(design.x, mach, color=series_color(1, dark), linewidth=LINEWIDTH)
        ax.set_ylabel("local Mach")
        ax.set_title("Section Mach at the design point", fontsize=9, loc="left")
    else:
        ax.plot(design.x, design.a_axial, color=series_color(0, dark), linewidth=LINEWIDTH)
        ax.set_ylabel("axial induction $a$")
        ax.set_title("Induction", fontsize=9, loc="left")

    for ax in axes:
        ax.set_xlabel("r / R")
        style_axes(ax, dark)

    fig.suptitle(design.describe(), fontsize=9.5, color=t["text"], y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


__all__ = ["diameter_sweep_figure", "design_figure"]
