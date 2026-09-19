"""Qt stylesheet and pyqtgraph defaults built from the shared colour tokens.

The desktop GUI must look like the notebook GUI, so both read the same palette
module rather than each hard-coding hexes.
"""

from __future__ import annotations

from ..viz.palette import categorical, theme as palette_theme

FONT_STACK = '"Inter", "Segoe UI", "SF Pro Text", system-ui, sans-serif'
MONO_STACK = '"JetBrains Mono", "SF Mono", Menlo, Consolas, monospace'


def stylesheet(dark: bool = True) -> str:
    t = palette_theme(dark)
    accent = categorical(dark)[0]
    return f"""
    QWidget {{
        background: {t['surface']};
        color: {t['text']};
        font-family: {FONT_STACK};
        font-size: 12px;
    }}
    QGroupBox {{
        background: {t['panel']};
        border: 1px solid {t['grid']};
        border-radius: 7px;
        margin-top: 14px;
        padding: 10px 8px 8px 8px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: {t['text_muted']};
        font-size: 10px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}
    QLabel#metricValue {{ font-size: 19px; font-weight: 600; font-family: {MONO_STACK}; }}
    QLabel#metricLabel {{ font-size: 9px; color: {t['text_muted']};
                          letter-spacing: 0.06em; }}
    QLabel#metricSub   {{ font-size: 10px; color: {t['text_secondary']}; }}
    QLabel#status      {{ color: {t['text_secondary']}; font-size: 11px; }}
    QSlider::groove:horizontal {{
        height: 3px; background: {t['grid']}; border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {accent}; width: 13px; height: 13px;
        margin: -5px 0; border-radius: 7px;
    }}
    QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background: {t['surface']}; border: 1px solid {t['grid']};
        border-radius: 5px; padding: 3px 7px; min-height: 20px;
    }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QPushButton {{
        background: {t['panel']}; border: 1px solid {t['grid']};
        border-radius: 5px; padding: 5px 13px;
    }}
    QPushButton:hover  {{ border-color: {accent}; }}
    QPushButton:checked {{ background: {accent}; color: #ffffff; border-color: {accent}; }}
    QCheckBox {{ spacing: 6px; }}
    QTabWidget::pane {{ border: 1px solid {t['grid']}; border-radius: 7px; top: -1px; }}
    QTabBar::tab {{
        background: transparent; padding: 6px 13px; margin-right: 2px;
        border-bottom: 2px solid transparent; color: {t['text_secondary']};
    }}
    QTabBar::tab:selected {{ color: {t['text']}; border-bottom-color: {accent}; }}
    QScrollArea {{ border: none; }}
    QStatusBar {{ color: {t['text_secondary']}; }}
    """


def configure_pyqtgraph(dark: bool = True) -> None:
    """Match pyqtgraph's global defaults to the palette."""
    import pyqtgraph as pg
    t = palette_theme(dark)
    pg.setConfigOption("background", t["surface"])
    pg.setConfigOption("foreground", t["text_secondary"])
    pg.setConfigOptions(antialias=True)


__all__ = ["stylesheet", "configure_pyqtgraph", "FONT_STACK", "MONO_STACK"]
