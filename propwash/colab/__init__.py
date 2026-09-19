"""Notebook front end: environment bootstrap and the ipywidgets GUI."""

from .bootstrap import Environment, in_colab, in_notebook, probe, setup

__all__ = ["setup", "probe", "Environment", "in_colab", "in_notebook", "launch",
           "PropwashLab"]


def __getattr__(name):
    if name in ("launch", "PropwashLab"):
        from . import app
        return getattr(app, name)
    raise AttributeError(name)
