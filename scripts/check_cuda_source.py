#!/usr/bin/env python3
"""Standalone wrapper: compile the CUDA kernel with NVRTC, no GPU required.

Equivalent to ``python -m propwash cuda-check``; kept as a script so CI can run
it straight from a checkout.

    pip install nvidia-cuda-nvrtc-cu12
    python scripts/check_cuda_source.py [compute_75 compute_90 ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from propwash.accel.nvrtc import DEFAULT_ARCHITECTURES, check  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    arches = tuple(args) if args else DEFAULT_ARCHITECTURES
    results = check(arches, verbose=True)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
