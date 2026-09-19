"""Compile the raw CUDA C kernel with NVRTC -- no GPU required.

NVRTC turns CUDA C into PTX and never touches a device, which makes it the
right tool for checking :mod:`propwash.accel.kernels_raw` in CI, on a laptop,
or on a Colab CPU runtime before switching to a GPU one.

    pip install nvidia-cuda-nvrtc-cu12
    python -m propwash cuda-check
"""

from __future__ import annotations

import ctypes
import glob
from dataclasses import dataclass

from .kernels_raw import CUDA_SOURCE

#: Where a pip-installed or system CUDA usually puts libnvrtc.
LIBRARY_PATTERNS = (
    "/usr/local/lib/python*/dist-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*",
    "/usr/local/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*",
    "/usr/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*",
    "/usr/local/cuda*/lib64/libnvrtc.so*",
    "/usr/lib/x86_64-linux-gnu/libnvrtc.so*",
)

DEFAULT_ARCHITECTURES = ("compute_75", "compute_80", "compute_89")


@dataclass
class CompileResult:
    architecture: str
    ok: bool
    log: str = ""
    ptx_bytes: int = 0

    def format(self) -> str:
        return (f"{self.architecture:<14} {'OK' if self.ok else 'FAILED':<7} "
                f"PTX {self.ptx_bytes:>8,} bytes")


def find_library() -> str | None:
    """Locate libnvrtc, skipping the builtins and the '.alt' variant."""
    for pattern in LIBRARY_PATTERNS:
        for hit in sorted(glob.glob(pattern)):
            if "builtins" not in hit and ".alt." not in hit:
                return hit
    return None


def compile_source(architecture: str = "compute_75", source: str | None = None
                   ) -> CompileResult:
    """Compile ``source`` (the packaged kernel by default) to PTX."""
    lib_path = find_library()
    if lib_path is None:
        raise RuntimeError("libnvrtc not found -- pip install nvidia-cuda-nvrtc-cu12")

    lib = ctypes.CDLL(lib_path)
    prog = ctypes.c_void_p()
    code = (source if source is not None else CUDA_SOURCE).encode()

    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), code, b"bemt_solve.cu",
                                0, None, None)
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed (code {rc})")

    options = [f"--gpu-architecture={architecture}".encode(), b"-std=c++11",
               b"-default-device"]
    argv = (ctypes.c_char_p * len(options))(*options)
    rc = lib.nvrtcCompileProgram(prog, len(options), argv)

    size = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    buf = ctypes.create_string_buffer(max(size.value, 1))
    lib.nvrtcGetProgramLog(prog, buf)
    log = buf.value.decode(errors="replace").strip()

    ptx_bytes = 0
    if rc == 0:
        ptx_size = ctypes.c_size_t()
        lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
        ptx = ctypes.create_string_buffer(ptx_size.value)
        lib.nvrtcGetPTX(prog, ptx)
        ptx_bytes = ptx_size.value

    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return CompileResult(architecture, rc == 0, log, ptx_bytes)


def check(architectures: tuple[str, ...] = DEFAULT_ARCHITECTURES,
          verbose: bool = True) -> list[CompileResult]:
    """Compile for several targets and report."""
    results = []
    for arch in architectures:
        result = compile_source(arch)
        results.append(result)
        if verbose:
            print(result.format())
            if result.log:
                print("  " + result.log.replace("\n", "\n  "))
    return results


__all__ = ["check", "compile_source", "find_library", "CompileResult",
           "DEFAULT_ARCHITECTURES", "LIBRARY_PATTERNS"]
