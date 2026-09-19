"""Colour tokens shared by every renderer.

One definition feeds Matplotlib, Plotly, pyqtgraph and the raw OpenGL vertex
colours, so a curve is the same blue in the desktop GUI, the notebook and an
exported PNG.  Colours are assigned by *job*, not by taste:

* **categorical** -- identity (which curve is which).  Fixed slot order, never
  cycled; the ordering is what keeps adjacent pairs separable for colour-vision
  deficiency, so do not reshuffle it.
* **sequential** -- magnitude (thrust over a pitch/RPM map, Mach on the blade).
  One hue, light to dark.
* **diverging** -- polarity around a meaningful zero (angle of attack about the
  stall line, thrust about zero).  Two hues with a neutral grey midpoint.

Dark-mode values are separately chosen steps of the same hues, not an
automatic inversion.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Categorical -- fixed slot order
# ---------------------------------------------------------------------------

CATEGORICAL_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CATEGORICAL_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767")

# ---------------------------------------------------------------------------
# Sequential ramps (light -> dark)
# ---------------------------------------------------------------------------

SEQ_BLUE = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b")

SEQ_ORANGE = ("#fde0d1", "#fbcdb4", "#f9b997", "#f7a47a", "#f48f5f", "#f07a46",
              "#eb6834", "#d95926", "#c24d1f", "#a8421a", "#8d3615", "#722b10",
              "#58210c")

SEQ_RED = ("#fbd5d4", "#f6bcbb", "#f2a3a2", "#ed8a89", "#e97170", "#e45857",
           "#e34948", "#ce3b3a", "#b53130", "#9b2827", "#821f1e", "#681716",
           "#4f100f")

# ---------------------------------------------------------------------------
# Surfaces and ink
# ---------------------------------------------------------------------------

LIGHT = {
    "surface": "#fcfcfb", "panel": "#f5f5f2", "grid": "#e4e3de",
    "text": "#0b0b0b", "text_secondary": "#52514e", "text_muted": "#8a8984",
    "neutral": "#f0efec", "axis": "#b8b7b1",
}
DARK = {
    "surface": "#1a1a19", "panel": "#232322", "grid": "#33332f",
    "text": "#ffffff", "text_secondary": "#c3c2b7", "text_muted": "#8a8984",
    "neutral": "#383835", "axis": "#4a4a45",
}


def theme(dark: bool = False) -> dict[str, str]:
    return DARK if dark else LIGHT


def categorical(dark: bool = False) -> tuple[str, ...]:
    return CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT


def series_color(index: int, dark: bool = False) -> str:
    """Slot ``index`` of the categorical palette.

    Slots are assigned in fixed order and never recycled: a chart that needs a
    ninth series needs fewer series, not another hue.
    """
    palette = categorical(dark)
    return palette[int(index) % len(palette)]


# ---------------------------------------------------------------------------
# Ramp evaluation (pure NumPy -- no Matplotlib needed)
# ---------------------------------------------------------------------------

def hex_to_rgb(color: str) -> tuple[float, float, float]:
    c = color.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{int(round(max(0.0, min(1.0, v)) * 255)):02x}" for v in rgb[:3])


def ramp_array(ramp: Iterable[str]) -> np.ndarray:
    """``(n, 3)`` float RGB for a hex ramp."""
    return np.array([hex_to_rgb(c) for c in ramp], dtype=np.float64)


def diverging_ramp(dark: bool = False, n_arm: int = 7) -> np.ndarray:
    """Blue -> neutral -> red, with equal step counts per arm."""
    cool = ramp_array(SEQ_BLUE)[::-1]
    warm = ramp_array(SEQ_RED)
    mid = np.array(hex_to_rgb(theme(dark)["neutral"]))[None, :]
    lo = _resample(cool, n_arm)
    hi = _resample(warm, n_arm)
    return np.vstack([lo, mid, hi])


def _resample(ramp: np.ndarray, n: int) -> np.ndarray:
    t_src = np.linspace(0.0, 1.0, ramp.shape[0])
    t_dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t_dst, t_src, ramp[:, k]) for k in range(3)], axis=1)


def map_values(values: np.ndarray, ramp: Iterable[str] | np.ndarray = SEQ_BLUE,
               vmin: float | None = None, vmax: float | None = None,
               alpha: float = 1.0) -> np.ndarray:
    """Map scalars onto a ramp, returning ``(n, 4)`` RGBA in [0, 1].

    This is what colours the 3-D blade, so it has to work with no plotting
    library present.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    cols = ramp if isinstance(ramp, np.ndarray) else ramp_array(ramp)

    lo = float(np.nanmin(arr)) if vmin is None else float(vmin)
    hi = float(np.nanmax(arr)) if vmax is None else float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        hi = lo + 1.0

    t = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * (cols.shape[0] - 1)
    i0 = np.clip(np.floor(t).astype(int), 0, cols.shape[0] - 2)
    w = (t - i0)[:, None]
    rgb = cols[i0] * (1.0 - w) + cols[i0 + 1] * w

    out = np.empty((arr.size, 4), dtype=np.float32)
    out[:, :3] = rgb
    out[:, 3] = alpha
    return out


def plotly_colorscale(ramp: Iterable[str] | np.ndarray = SEQ_BLUE) -> list[list]:
    """``[[t, 'rgb(r,g,b)'], ...]`` for Plotly's ``colorscale``."""
    cols = ramp if isinstance(ramp, np.ndarray) else ramp_array(ramp)
    n = cols.shape[0]
    return [[i / (n - 1),
             f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})"]
            for i, c in enumerate(cols)]


def mpl_colormap(ramp: Iterable[str] | np.ndarray = SEQ_BLUE, name: str = "propwash"):
    """A Matplotlib ``LinearSegmentedColormap`` built from a ramp."""
    from matplotlib.colors import LinearSegmentedColormap
    cols = ramp if isinstance(ramp, np.ndarray) else ramp_array(ramp)
    return LinearSegmentedColormap.from_list(name, [tuple(c) for c in cols], N=256)


# ---------------------------------------------------------------------------
# Field presets: which ramp suits which solver quantity
# ---------------------------------------------------------------------------

FIELD_STYLES: dict[str, dict] = {
    "alpha":       {"label": "Angle of attack", "unit": "deg", "scale": 180.0 / np.pi,
                    "ramp": "diverging", "center": 0.0},
    "phi":         {"label": "Inflow angle", "unit": "deg", "scale": 180.0 / np.pi,
                    "ramp": "sequential"},
    "cl":          {"label": "Lift coefficient", "unit": "", "scale": 1.0,
                    "ramp": "diverging", "center": 0.0},
    "cd":          {"label": "Drag coefficient", "unit": "", "scale": 1.0,
                    "ramp": "sequential"},
    "dt_dr":       {"label": "Thrust loading", "unit": "N/m", "scale": 1.0,
                    "ramp": "sequential"},
    "dq_dr":       {"label": "Torque loading", "unit": "N", "scale": 1.0,
                    "ramp": "sequential"},
    "mach":        {"label": "Local Mach", "unit": "", "scale": 1.0, "ramp": "sequential"},
    "w":           {"label": "Resultant speed", "unit": "m/s", "scale": 1.0,
                    "ramp": "sequential"},
    "reynolds":    {"label": "Reynolds number", "unit": "", "scale": 1.0,
                    "ramp": "sequential"},
    "loss_factor": {"label": "Prandtl loss factor", "unit": "", "scale": 1.0,
                    "ramp": "sequential"},
    "v_axial_induced": {"label": "Induced axial velocity", "unit": "m/s", "scale": 1.0,
                        "ramp": "sequential"},
    "v_swirl_induced": {"label": "Induced swirl velocity", "unit": "m/s", "scale": 1.0,
                        "ramp": "sequential"},
}

FIELD_ORDER = tuple(FIELD_STYLES)


def field_ramp(field: str, dark: bool = False) -> np.ndarray:
    style = FIELD_STYLES.get(field, {"ramp": "sequential"})
    if style["ramp"] == "diverging":
        return diverging_ramp(dark)
    return ramp_array(SEQ_BLUE)


def field_label(field: str) -> str:
    style = FIELD_STYLES.get(field)
    if style is None:
        return field
    return f"{style['label']} [{style['unit']}]" if style["unit"] else style["label"]


__all__ = [
    "CATEGORICAL_LIGHT", "CATEGORICAL_DARK", "SEQ_BLUE", "SEQ_ORANGE", "SEQ_RED",
    "LIGHT", "DARK", "theme", "categorical", "series_color", "hex_to_rgb",
    "rgb_to_hex", "ramp_array", "diverging_ramp", "map_values",
    "plotly_colorscale", "mpl_colormap", "FIELD_STYLES", "FIELD_ORDER",
    "field_ramp", "field_label",
]
