"""OpenGL propeller viewport built on pyqtgraph's GL widgets.

Two things this does that a naive GLMeshItem does not:

* **Per-vertex colours from solver output.**  The mesh carries each vertex's
  ``r/R``, so any spanwise field maps straight onto the surface without
  rebuilding geometry.
* **Rotation at a rate proportional to real RPM.**  A propeller at 9,000 rpm is
  150 revolutions per second; shown honestly it is a grey disk.  The display is
  slowed by a fixed factor, so the ratio between two operating points survives
  and you can see the propeller speed up when you add volts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..bemt.core import BEMTResult
from ..mesh.loft import PART_BLADE, PropellerMesh
from ..viz.palette import (FIELD_STYLES, diverging_ramp, field_label, hex_to_rgb,
                           map_values, ramp_array, theme)
from ..viz.palette import SEQ_BLUE

#: Display revolutions are this many times slower than the real ones.
SPIN_SLOWDOWN = 220.0

#: Direction the baked key light comes from, in world space.
_LIGHT = np.array([0.34, -0.48, 0.81])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)
_AMBIENT = 0.42


def bake_lighting(colors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Multiply a Lambert term into vertex colours.

    pyqtgraph's ``shaded`` shader darkens vertex colours so heavily that a
    sequential ramp collapses to near-black.  Baking the lighting ourselves and
    drawing with no shader keeps the colour scale readable, which matters when
    the colour *is* the data.
    """
    lam = np.clip(normals @ _LIGHT, 0.0, 1.0)
    shade = (_AMBIENT + (1.0 - _AMBIENT) * lam).astype(np.float32)
    out = colors.copy()
    out[:, :3] = np.clip(out[:, :3] * shade[:, None], 0.0, 1.0)
    return out


def field_vertex_colors(mesh: PropellerMesh, result: BEMTResult | None, field: str,
                        dark: bool = False) -> tuple[np.ndarray, str, float, float]:
    """RGBA per vertex for ``field``, plus the caption and range for a legend."""
    t = theme(dark)
    hub_rgb = hex_to_rgb("#4a4a45" if dark else "#b8b7b1")

    if result is None or not hasattr(result, field):
        colors = np.tile(np.array([*hub_rgb, 1.0], dtype=np.float32),
                         (mesh.n_vertices, 1))
        return colors, "", 0.0, 1.0

    style = FIELD_STYLES.get(field, {"scale": 1.0, "ramp": "sequential"})
    values = np.asarray(getattr(result, field), dtype=float) * float(style.get("scale", 1.0))
    vertex_values = mesh.sample_field(result.x, values)

    if style.get("ramp") == "diverging":
        centre = float(style.get("center", 0.0))
        half = max(abs(values.max() - centre), abs(values.min() - centre), 1e-9)
        vmin, vmax = centre - half, centre + half
        ramp = diverging_ramp(dark)
    else:
        vmin, vmax = float(values.min()), float(values.max())
        ramp = ramp_array(SEQ_BLUE)

    colors = map_values(vertex_values, ramp, vmin, vmax)
    hub = mesh.part != PART_BLADE
    if hub.any():
        colors[hub, :3] = hub_rgb
    return colors.astype(np.float32), field_label(field), vmin, vmax


def build_gl_view(dark: bool = True):
    """Create the GL widget with a sensible initial camera."""
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtGui

    t = theme(dark)
    view = gl.GLViewWidget()
    r, g, b = hex_to_rgb(t["surface"])
    view.setBackgroundColor(QtGui.QColor.fromRgbF(r, g, b, 1.0))
    view.opts["distance"] = 0.45
    view.opts["elevation"] = 20
    view.opts["azimuth"] = -58
    view.opts["fov"] = 42
    return view


class PropellerGLView:
    """Owns the GL widget, the mesh item and the spin timer."""

    def __init__(self, dark: bool = True) -> None:
        import pyqtgraph.opengl as gl
        from pyqtgraph.Qt import QtCore

        self.dark = dark
        self.widget = build_gl_view(dark)
        self.mesh: PropellerMesh | None = None
        self.item: Any = None
        self.grid: Any = None
        self._angle = 0.0
        self._rpm = 0.0
        self._gl = gl

        self.timer = QtCore.QTimer()
        self.timer.setInterval(16)          # ~60 fps
        self.timer.timeout.connect(self._tick)
        self._spinning = False
        self._last = None

    # -- geometry ----------------------------------------------------------
    def set_mesh(self, mesh: PropellerMesh, result: BEMTResult | None = None,
                 field: str = "dt_dr") -> None:
        """Replace the geometry (blade changed)."""
        self.mesh = mesh
        colors, _, _, _ = field_vertex_colors(mesh, result, field, self.dark)
        colors = bake_lighting(colors, mesh.normals)

        if self.item is not None:
            self.widget.removeItem(self.item)

        data = self._gl.MeshData(vertexes=mesh.vertices.astype(np.float32),
                                 faces=mesh.faces.astype(np.uint32),
                                 vertexColors=colors)
        self.item = self._gl.GLMeshItem(meshdata=data, smooth=True, shader=None,
                                        drawEdges=False, computeNormals=False)
        self.widget.addItem(self.item)
        self._ensure_grid(mesh.radius)
        self._apply_rotation()

    def set_field(self, result: BEMTResult | None, field: str) -> tuple[str, float, float]:
        """Recolour without touching geometry (operating point changed)."""
        if self.mesh is None or self.item is None:
            return "", 0.0, 1.0
        colors, label, vmin, vmax = field_vertex_colors(self.mesh, result, field, self.dark)
        colors = bake_lighting(colors, self.mesh.normals)
        data = self._gl.MeshData(vertexes=self.mesh.vertices.astype(np.float32),
                                 faces=self.mesh.faces.astype(np.uint32),
                                 vertexColors=colors)
        self.item.setMeshData(meshdata=data)
        return label, vmin, vmax

    def _ensure_grid(self, radius: float) -> None:
        """A faint disk-plane grid so the propeller is not floating in a void."""
        if self.grid is not None:
            self.widget.removeItem(self.grid)
        grid = self._gl.GLGridItem()
        grid.setSize(x=6.0 * radius, y=6.0 * radius)
        grid.setSpacing(x=radius / 2.0, y=radius / 2.0)
        grid.translate(0, 0, -1.5 * radius)
        grid.setDepthValue(20)
        self.grid = grid
        self.widget.addItem(grid)
        self.widget.opts["distance"] = 5.0 * radius

    # -- animation ---------------------------------------------------------
    def set_rpm(self, rpm: float) -> None:
        self._rpm = float(max(rpm, 0.0))

    def start(self) -> None:
        import time
        if not self._spinning:
            self._spinning = True
            self._last = time.perf_counter()
            self.timer.start()

    def stop(self) -> None:
        self._spinning = False
        self.timer.stop()

    def toggle(self, on: bool) -> None:
        self.start() if on else self.stop()

    def _tick(self) -> None:
        import time
        now = time.perf_counter()
        dt = now - (self._last or now)
        self._last = now
        self._angle += 360.0 * (self._rpm / 60.0) / SPIN_SLOWDOWN * dt
        self._angle %= 360.0
        self._apply_rotation()

    def _apply_rotation(self) -> None:
        if self.item is None:
            return
        self.item.resetTransform()
        self.item.rotate(self._angle, 0.0, 0.0, 1.0)

    # -- misc --------------------------------------------------------------
    def set_dark(self, dark: bool) -> None:
        from pyqtgraph.Qt import QtGui
        self.dark = dark
        r, g, b = hex_to_rgb(theme(dark)["surface"])
        self.widget.setBackgroundColor(QtGui.QColor.fromRgbF(r, g, b, 1.0))

    def reset_camera(self) -> None:
        self.widget.opts["elevation"] = 20
        self.widget.opts["azimuth"] = -58
        if self.mesh is not None:
            self.widget.opts["distance"] = 5.0 * self.mesh.radius
        self.widget.update()


__all__ = ["PropellerGLView", "field_vertex_colors", "bake_lighting",
           "build_gl_view", "SPIN_SLOWDOWN"]
