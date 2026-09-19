"""Reusable control widgets for the desktop GUI.

A labelled slider that emits real engineering units, and a metric card, are the
only two things this file exists to provide -- but they appear forty times
between them, so they are worth doing once properly.
"""

from __future__ import annotations

from typing import Callable, Iterable

from ..viz.palette import theme


def _qt():
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


class LabeledSlider:
    """An integer QSlider presented in floating-point engineering units.

    Qt sliders are integer-only, so the value is carried as
    ``round(value / step)`` and converted at the boundary.  The readout shows
    the real number with its unit; callers never see the integer.
    """

    def __init__(self, label: str, minimum: float, maximum: float, value: float,
                 step: float = 1.0, unit: str = "", fmt: str = "{:.2f}",
                 on_change: Callable[[float], None] | None = None,
                 dark: bool = True) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self._step = float(step)
        self._fmt = fmt
        self._unit = unit
        self._callback = on_change
        self._muted = False

        t = theme(dark)
        self.widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(self.widget)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(2)

        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.name = QtWidgets.QLabel(label)
        self.name.setStyleSheet(f"color:{t['text_secondary']};font-size:11px;")
        self.readout = QtWidgets.QLabel()
        self.readout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.readout.setStyleSheet(
            f"color:{t['text']};font-size:11px;font-weight:600;"
            f"font-family:'JetBrains Mono','SF Mono',Menlo,monospace;")
        head.addWidget(self.name)
        head.addStretch(1)
        head.addWidget(self.readout)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(int(round(minimum / self._step)))
        self.slider.setMaximum(int(round(maximum / self._step)))
        self.slider.setValue(int(round(value / self._step)))
        self.slider.valueChanged.connect(self._emit)

        outer.addLayout(head)
        outer.addWidget(self.slider)
        self._update_readout()

    @property
    def value(self) -> float:
        return self.slider.value() * self._step

    def set_value(self, value: float, notify: bool = False) -> None:
        self._muted = not notify
        try:
            self.slider.setValue(int(round(float(value) / self._step)))
        finally:
            self._muted = False
        self._update_readout()

    def set_enabled(self, enabled: bool) -> None:
        self.slider.setEnabled(enabled)
        self.widget.setEnabled(True)

    def _emit(self, _raw: int) -> None:
        self._update_readout()
        if self._callback is not None and not self._muted:
            self._callback(self.value)

    def _update_readout(self) -> None:
        text = self._fmt.format(self.value)
        self.readout.setText(f"{text} {self._unit}".strip())


class MetricCard:
    """One headline number with a caption and a coloured accent bar."""

    def __init__(self, label: str, accent: str = "#2a78d6", dark: bool = True) -> None:
        _, _, QtWidgets = _qt()
        t = theme(dark)
        # A rounded box with a coloured left border draws the corner arcs in the
        # accent colour, which reads as stray punctuation.  Use a separate 3 px
        # strip widget instead so the bar stays a clean rectangle.
        self.widget = QtWidgets.QFrame()
        self.widget.setStyleSheet(
            f"QFrame#card{{background:{t['panel']};border-radius:6px;}}")
        self.widget.setObjectName("card")
        shell = QtWidgets.QHBoxLayout(self.widget)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        strip = QtWidgets.QFrame()
        strip.setFixedWidth(3)
        strip.setStyleSheet(f"background:{accent};border-top-left-radius:6px;"
                            f"border-bottom-left-radius:6px;")
        shell.addWidget(strip)

        body = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(body)
        lay.setContentsMargins(9, 6, 9, 6)
        lay.setSpacing(1)
        shell.addWidget(body, 1)

        self.label = QtWidgets.QLabel(label.upper())
        self.label.setObjectName("metricLabel")
        self.value = QtWidgets.QLabel("--")
        self.value.setObjectName("metricValue")
        self.sub = QtWidgets.QLabel("")
        self.sub.setObjectName("metricSub")

        lay.addWidget(self.label)
        lay.addWidget(self.value)
        lay.addWidget(self.sub)

    def set(self, value: str, sub: str = "") -> None:
        self.value.setText(value)
        self.sub.setText(sub)


def group(title: str, rows: Iterable) -> object:
    """A titled QGroupBox wrapping a vertical stack of widgets."""
    _, _, QtWidgets = _qt()
    box = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(8, 6, 8, 8)
    lay.setSpacing(4)
    for row in rows:
        if hasattr(row, "widget") and not isinstance(row, type):
            lay.addWidget(row.widget)
        elif isinstance(row, (list, tuple)):
            line = QtWidgets.QHBoxLayout()
            line.setContentsMargins(0, 0, 0, 0)
            for item in row:
                line.addWidget(item.widget if hasattr(item, "widget") else item)
            lay.addLayout(line)
        else:
            lay.addWidget(row)
    return box


def labelled(text: str, widget, dark: bool = True):
    """A horizontal ``label: control`` row."""
    _, _, QtWidgets = _qt()
    t = theme(dark)
    holder = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(holder)
    lay.setContentsMargins(0, 1, 0, 1)
    lab = QtWidgets.QLabel(text)
    lab.setStyleSheet(f"color:{t['text_secondary']};font-size:11px;")
    lab.setMinimumWidth(86)
    lay.addWidget(lab)
    lay.addWidget(widget, 1)
    return holder


__all__ = ["LabeledSlider", "MetricCard", "group", "labelled"]


class ColorLegend:
    """A horizontal colour bar for the 3-D view.

    The OpenGL viewport colours the blade by a solver field, and colour without
    a scale is decoration rather than data -- so the ramp, the field name and
    the numeric range are drawn underneath it.  The ramp is painted from the
    same token arrays the mesh is coloured with, so the bar cannot disagree
    with the blade.
    """

    def __init__(self, dark: bool = True, height: int = 38) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self._QtCore, self._QtGui = QtCore, QtGui
        self.dark = dark
        self._ramp = None
        self._label = ""
        self._vmin = 0.0
        self._vmax = 1.0

        legend = self

        class _Canvas(QtWidgets.QWidget):
            def paintEvent(self, _event):
                legend._paint(self)

        self.widget = _Canvas()
        self.widget.setFixedHeight(height)
        self.widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Fixed)

    def set_field(self, ramp, label: str, vmin: float, vmax: float) -> None:
        self._ramp = ramp
        self._label = label
        self._vmin = float(vmin)
        self._vmax = float(vmax)
        self.widget.update()

    def set_dark(self, dark: bool) -> None:
        self.dark = dark
        self.widget.update()

    def _paint(self, canvas) -> None:
        import numpy as np
        QtCore, QtGui = self._QtCore, self._QtGui
        t = theme(self.dark)

        painter = QtGui.QPainter(canvas)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = canvas.width(), canvas.height()
        pad = 12
        bar_h = 10
        bar = QtCore.QRectF(pad, 4, max(w - 2 * pad, 1), bar_h)

        if self._ramp is not None and len(self._ramp):
            gradient = QtGui.QLinearGradient(bar.left(), 0.0, bar.right(), 0.0)
            ramp = np.asarray(self._ramp, dtype=float)
            for i, rgb in enumerate(ramp):
                gradient.setColorAt(i / max(len(ramp) - 1, 1),
                                    QtGui.QColor.fromRgbF(*rgb[:3], 1.0))
            painter.fillRect(bar, QtGui.QBrush(gradient))

        painter.setPen(QtGui.QColor(t["text_secondary"]))
        font = painter.font()
        font.setPointSizeF(8.0)
        painter.setFont(font)

        text_rect = QtCore.QRectF(pad, bar.bottom() + 2, bar.width(), h - bar_h - 6)
        painter.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                         f"{self._vmin:,.3g}")
        painter.drawText(text_rect, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                         f"{self._vmax:,.3g}")
        painter.setPen(QtGui.QColor(t["text"]))
        painter.drawText(text_rect, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                         self._label or "")
        painter.end()


__all__.append("ColorLegend")
