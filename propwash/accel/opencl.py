"""OpenCL backend -- GPU acceleration without CUDA.

CUDA only runs on NVIDIA hardware and only where the toolkit is installed.
OpenCL runs on AMD and Intel GPUs, on Apple silicon, on FPGAs, and -- via POCL
or Intel's CPU runtime -- on a plain CPU, which makes it the portable answer
and also means this path can be tested on a machine with no GPU at all.

The kernel is the same C as the CUDA one, rendered for OpenCL by
:func:`propwash.accel.kernels_raw.kernel_source`, so the two cannot drift.

Two device details are handled here rather than assumed:

* **Work-group size.** ``PW_THREADS`` is compiled into the kernel because it
  sizes the local-memory reduction buffers.  Devices disagree about the maximum
  work-group size -- CPUs often report far less than a GPU -- so the size is
  queried, clamped to the largest power of two that fits (the tree reduction
  requires a power of two), and the source is rendered for *that* size.
* **Double precision.** The bisection's sign test is exact-arithmetic
  sensitive.  Devices without ``cl_khr_fp64`` are rejected rather than silently
  demoted to float, which would quietly cost several digits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .kernels_raw import PW_THREADS, kernel_source

_cache: dict[str, Any] = {}


def _largest_power_of_two(n: int) -> int:
    return 1 << max(int(n).bit_length() - 1, 0)


def available_devices() -> list[tuple[Any, Any]]:
    """Every ``(platform, device)`` pair OpenCL can see."""
    import pyopencl as cl
    out = []
    for platform in cl.get_platforms():
        try:
            for device in platform.get_devices():
                out.append((platform, device))
        except cl.LogicError:            # pragma: no cover - driver dependent
            continue
    return out


def supports_fp64(device) -> bool:
    exts = device.get_info(__import__("pyopencl").device_info.EXTENSIONS)
    return "cl_khr_fp64" in exts or "cl_amd_fp64" in exts


def pick_device(prefer: str = "gpu"):
    """Choose a device, preferring a real GPU over a CPU runtime.

    Returns ``(platform, device)``.  Raises if nothing usable is present, which
    the backend turns into "unavailable" rather than an exception.
    """
    import pyopencl as cl

    candidates = [(p, d) for p, d in available_devices() if supports_fp64(d)]
    if not candidates:
        raise RuntimeError("no OpenCL device with double-precision support")

    def rank(pair):
        _, device = pair
        is_gpu = bool(device.type & cl.device_type.GPU)
        want_gpu = prefer.lower() != "cpu"
        return (0 if is_gpu == want_gpu else 1, -device.max_compute_units)

    return sorted(candidates, key=rank)[0]


def describe_device(device) -> str:
    import pyopencl as cl
    kind = cl.device_type.to_string(device.type, "%d")
    return (f"{device.name.strip()} ({kind}), {device.max_compute_units} CUs, "
            f"{device.global_mem_size / 2 ** 30:.1f} GiB")


def build(prefer: str = "gpu"):
    """Compile the kernel for the chosen device.

    Returns ``(context, queue, kernel, threads)``.  The *kernel object* is
    cached, not just the program: in PyOpenCL every ``program.name`` attribute
    lookup constructs a fresh ``cl.Kernel``, which is far from free when the
    solver is launched once per row of a sweep.
    """
    key = f"ocl:{prefer}"
    if key in _cache:
        return _cache[key]

    import pyopencl as cl

    platform, device = pick_device(prefer)
    context = cl.Context(devices=[device])
    queue = cl.CommandQueue(context)

    # The tree reduction halves the stride each pass, so the work-group size
    # must be a power of two, and it must fit both the device limit and the
    # local memory the three reduction buffers need (3 doubles per work item).
    max_group = int(device.max_work_group_size)
    by_memory = int(device.local_mem_size) // (3 * 8)
    threads = _largest_power_of_two(max(min(PW_THREADS, max_group, by_memory), 1))

    source = kernel_source("opencl", threads)
    program = cl.Program(context, source).build(options=["-cl-std=CL1.2"])
    kernel = cl.Kernel(program, "bemt_solve")

    # Declare the scalar argument types once.  Without this PyOpenCL has to
    # infer the type of every scalar on every call, which it warns about and
    # which costs real time when the kernel is launched in a sweep loop.
    kernel.set_scalar_arg_dtypes(
        [None] * 24 + [
            np.int32, np.int32, np.int32,                       # n_st, n_alpha, n_cases
            np.float64, np.float64,                             # alpha0, d_alpha
            np.float64, np.float64, np.float64,                 # rho, mu, a_sound
            np.float64, np.float64,                             # r_tip, r_hub
            np.int32, np.int32, np.int32,                       # n_blades, flags, n_bisect
            np.float64, np.float64,                             # phi_lo, phi_hi
            np.int32,                                           # want_span
        ])

    _cache[key] = (context, queue, kernel, threads)
    return _cache[key]


def solve_batch_opencl(stations, rpm: np.ndarray, v_inf: np.ndarray, air,
                       opts, tables, want_spanwise: bool = True,
                       prefer: str = "gpu") -> dict[str, np.ndarray]:
    """Run the batched BEMT solve on an OpenCL device."""
    import pyopencl as cl

    from ..bemt.core import PHI_HI, PHI_LO
    from ..units import rpm_to_rad_s
    from .backend import _kernel_arrays

    context, queue, kernel, threads = build(prefer)
    mf = cl.mem_flags

    a = _kernel_arrays(stations, tables)
    n_cases = int(np.asarray(rpm).size)
    n_st = int(stations.n)
    geom = stations.geometry

    def ro(array: np.ndarray):
        return cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=np.ascontiguousarray(array, dtype=np.float64))

    dev_in = [ro(a[k]) for k in ("r", "chord", "twist", "sigma", "thickness",
                                 "re_ref", "m_crit0", "cl_max", "dr",
                                 "cl_tab", "cd_tab")]
    dev_in.append(ro(rpm_to_rad_s(np.asarray(rpm, dtype=np.float64)).ravel()))
    dev_in.append(ro(np.asarray(v_inf, dtype=np.float64).ravel()))

    host_out = [np.empty(n_cases, dtype=np.float64) for _ in range(3)]
    span_n = n_cases * n_st if want_spanwise else 1
    host_span = [np.empty(span_n, dtype=np.float64) for _ in range(8)]

    dev_out = [cl.Buffer(context, mf.WRITE_ONLY, h.nbytes)
               for h in host_out + host_span]

    kernel(
        queue, (n_cases * threads,), (threads,),
        *dev_in, *dev_out,
        np.int32(n_st), np.int32(a["n_alpha"]), np.int32(n_cases),
        np.float64(a["alpha0"]), np.float64(a["d_alpha"]),
        np.float64(air.density), np.float64(air.viscosity),
        np.float64(air.sound_speed), np.float64(geom.radius),
        np.float64(geom.hub_radius), np.int32(geom.n_blades),
        np.int32(opts.flags()), np.int32(opts.n_bisect),
        np.float64(PHI_LO), np.float64(PHI_HI),
        np.int32(1 if want_spanwise else 0))

    for host, dev in zip(host_out + host_span, dev_out):
        cl.enqueue_copy(queue, host, dev)
    queue.finish()

    out = {"thrust": host_out[0], "torque": host_out[1], "converged": host_out[2]}
    if want_spanwise:
        for name, arr in zip(("phi", "alpha", "cl", "cd", "w", "dt_dr", "dq_dr",
                              "loss_factor"), host_span):
            out[name] = arr.reshape(n_cases, n_st)
    return out


__all__ = ["solve_batch_opencl", "build", "pick_device", "available_devices",
           "describe_device", "supports_fp64"]
