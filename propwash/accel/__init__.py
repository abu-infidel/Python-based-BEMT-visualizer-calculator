"""Compute backends: NumPy, Numba (CPU and CUDA), CuPy raw kernels, PyTorch."""

from .backend import (Backend, BackendInfo, describe_environment, get_backend,
                      list_backends)

__all__ = ["Backend", "BackendInfo", "get_backend", "list_backends",
           "describe_environment"]
