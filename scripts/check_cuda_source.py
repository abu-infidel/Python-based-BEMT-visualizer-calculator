#!/usr/bin/env python3
"""Compile the raw CUDA C kernel with NVRTC without needing a GPU.

NVRTC is a pure compiler: it turns CUDA C into PTX and never touches a device.
That makes it the right tool for CI, and for a laptop, to prove the kernel in
:mod:`propwash.accel.kernels_raw` is valid CUDA before anyone tries to run it
on Colab.  Install it with ``pip install nvidia-cuda-nvrtc-cu12``.
"""

from __future__ import annotations

import ctypes
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from propwash.accel.kernels_raw import CUDA_SOURCE  # noqa: E402

SEARCH = [
    "/usr/local/lib/python*/dist-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*",
    "/usr/local/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*",
    "/usr/lib/x86_64-linux-gnu/libnvrtc.so*",
    "/usr/local/cuda/lib64/libnvrtc.so*",
]


def find_nvrtc() -> str | None:
    for pattern in SEARCH:
        for hit in sorted(glob.glob(pattern)):
            if "builtins" not in hit and ".alt." not in hit:
                return hit
    return None


def compile_source(arch: str = "compute_75") -> tuple[bool, str, int]:
    lib_path = find_nvrtc()
    if lib_path is None:
        raise SystemExit("libnvrtc not found -- pip install nvidia-cuda-nvrtc-cu12")

    lib = ctypes.CDLL(lib_path)
    prog = ctypes.c_void_p()

    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), CUDA_SOURCE.encode(),
                                b"bemt_solve.cu", 0, None, None)
    if rc != 0:
        raise SystemExit(f"nvrtcCreateProgram failed with code {rc}")

    opts = [f"--gpu-architecture={arch}".encode(), b"-std=c++11", b"-default-device"]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)

    log_size = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(max(log_size.value, 1))
    lib.nvrtcGetProgramLog(prog, log)

    ptx_len = 0
    if rc == 0:
        ptx_size = ctypes.c_size_t()
        lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
        ptx = ctypes.create_string_buffer(ptx_size.value)
        lib.nvrtcGetPTX(prog, ptx)
        ptx_len = ptx_size.value

    lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return rc == 0, log.value.decode(errors="replace"), ptx_len


def main() -> int:
    arches = sys.argv[1:] or ["compute_60", "compute_75", "compute_80", "compute_89"]
    failed = False
    for arch in arches:
        ok, log, ptx_len = compile_source(arch)
        status = "OK" if ok else "FAILED"
        print(f"{arch:<14} {status:<7} PTX {ptx_len:>7} bytes")
        if log.strip():
            print("  " + log.strip().replace("\n", "\n  "))
        failed |= not ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
