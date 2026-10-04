"""Notebook front end: environment bootstrap and the ipywidgets GUI."""

from .bootstrap import Environment, in_colab, in_notebook, probe, setup

__all__ = ["setup", "probe", "Environment", "in_colab", "in_notebook", "launch",
           "PropwashApp"]


def __getattr__(name):
    if name in ("launch", "PropwashApp"):
        from . import app
        return getattr(app, name)
    raise AttributeError(name)
