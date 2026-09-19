"""Desktop GUI (PySide6 + pyqtgraph OpenGL).

Imported lazily: ``import propwash`` must not require Qt.
"""

__all__ = ["launch", "PropwashWindow"]


def __getattr__(name):
    if name in __all__:
        from . import app
        return getattr(app, name)
    raise AttributeError(name)
