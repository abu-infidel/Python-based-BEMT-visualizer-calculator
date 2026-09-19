"""Rendering: shared colour tokens, Matplotlib figures and Plotly 3-D views."""

from .palette import (CATEGORICAL_DARK, CATEGORICAL_LIGHT, FIELD_ORDER,
                      FIELD_STYLES, field_label, field_ramp, map_values,
                      series_color, theme)

__all__ = ["theme", "series_color", "map_values", "field_ramp", "field_label",
           "FIELD_STYLES", "FIELD_ORDER", "CATEGORICAL_LIGHT", "CATEGORICAL_DARK"]
