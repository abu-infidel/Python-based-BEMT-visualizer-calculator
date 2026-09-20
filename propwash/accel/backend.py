"""Compute backend registry and dispatch.

Every backend answers the same question -- "solve these operating points on this
blade" -- and returns the same dictionary of arrays.  :func:`get_backend`
picks the fastest one that actually works on this machine, which on a Colab
GPU runtime means CUDA and on a plain CPU runtime means Numba, silently.

Backends never raise on absence.  A missing CuPy or a driver/runtime version
mismatch is normal on a shared host, so probing is done once, the reason is
recorded, and the next backend down is used.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..atmosphere import SEA_LEVEL, AirState
from ..geometry import BladeStations
from ..bemt.core import (PHI_HI, PHI_LO, SolverOptions,
                         solve_batch as _numpy_solve_batch)
from ..units import rpm_to_rad_s

_SPAN_KEYS = ("phi", "alpha", "cl", "cd", "w", "dt_dr", "dq_dr", "loss_factor")


@dataclass
class BackendInfo:
    """What a backend is and whether it can run here."""

    name: str
    available: bool
    device: str = "cpu"
    detail: str = ""
    reason: str = ""

    def __str__(self) -> str:
        mark = "ok" if self.available else "unavailable"
        tail = self.detail if self.available else self.reason
        return f"{self.name:<12} [{mark}] {tail}"


class Backend:
    """Common interface.  Subclasses implement :meth:`_run`."""

    name = "base"
    priority = 0

    def __init__(self) -> None:
        self._info: BackendInfo | None = None
        self._device_cache: dict[str, Any] = {}

    # -- availability ------------------------------------------------------
    def probe(self) -> BackendInfo:
        if self._info is None:
            try:
                self._info = self._probe()
            except Exception as exc:  # pragma: no cover - environment dependent
                self._info = BackendInfo(self.name, False, reason=f"{type(exc).__name__}: {exc}")
        return self._info

    def _probe(self) -> BackendInfo:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return self.probe().available

    def invalidate(self) -> None:
        """Drop any device-side copies of the blade (geometry changed)."""
        self._device_cache.clear()

    # -- solving -----------------------------------------------------------
    def solve_batch(self, stations: BladeStations, rpm: np.ndarray, v_inf: np.ndarray,
                    air: AirState | None = None, opts: SolverOptions | None = None,
                    tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                    want_spanwise: bool = True) -> dict[str, np.ndarray]:
        opts = opts or SolverOptions()
        air = air or SEA_LEVEL
        tables = tables if tables is not None else stations.polar_tables(opts.n_alpha_table)

        rpm_b, v_b = np.broadcast_arrays(np.asarray(rpm, dtype=np.float64),
                                         np.asarray(v_inf, dtype=np.float64))
        shape = rpm_b.shape
        rpm_f = np.ascontiguousarray(rpm_b).ravel()
        v_f = np.ascontiguousarray(v_b).ravel()

        t0 = time.perf_counter()
        raw = self._run(stations, rpm_f, v_f, air, opts, tables, want_spanwise)
        elapsed = time.perf_counter() - t0

        out = _derive(raw, stations, rpm_f, v_f, air, opts)
        out = _reshape(out, shape, stations.n)
        out["backend"] = self.name
        out["elapsed_s"] = elapsed
        return out

    def _run(self, stations, rpm, v_inf, air, opts, tables, want_spanwise) -> dict[str, np.ndarray]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------

def _derive(raw: dict[str, np.ndarray], stations: BladeStations, rpm: np.ndarray,
            v_inf: np.ndarray, air: AirState, opts: SolverOptions) -> dict[str, np.ndarray]:
    """Turn per-element kernel output into the standard result dictionary.

    Kernels return thrust, torque and (optionally) spanwise state.  Everything
    else -- coefficients, efficiency, figure of merit, Mach, Reynolds -- is
    cheap algebra best done once, on the host, in one place.
    """
    geom = stations.geometry
    thrust = raw["thrust"]
    torque = raw["torque"]
    omega = rpm_to_rad_s(rpm)
    power = torque * omega

    n_rev = np.maximum(rpm / 60.0, 1e-9)
    d = geom.diameter
    area = math.pi * geom.radius ** 2
    rho = air.density

    with np.errstate(divide="ignore", invalid="ignore"):
        # See integrate_batch: efficiency is undefined once thrust or power
        # changes sign, so it is reported as zero rather than as a spike.
        useful = (power > 1e-9) & (thrust > 0.0)
        eta = np.where(useful, thrust * v_inf / np.maximum(power, 1e-12), 0.0)
        fom = np.where(useful, np.maximum(thrust, 0.0) ** 1.5
                       / math.sqrt(2.0 * rho * area) / np.maximum(power, 1e-12), 0.0)

    out: dict[str, np.ndarray] = {
        "thrust": thrust, "torque": torque, "power": power,
        "efficiency": np.clip(np.nan_to_num(eta), -2.0, 2.0),
        "figure_of_merit": np.clip(np.nan_to_num(fom), 0.0, 1.5),
        "ct": thrust / (rho * n_rev ** 2 * d ** 4),
        "cq": torque / (rho * n_rev ** 2 * d ** 5),
        "cp": power / (rho * n_rev ** 3 * d ** 5),
        "j": v_inf / (n_rev * d),
        "rpm": rpm, "v_inf": v_inf,
        "tip_mach": np.hypot(omega * geom.radius, v_inf) / air.sound_speed,
        "disk_loading": thrust / area,
        "converged_fraction": raw.get("converged", np.ones_like(thrust)),
    }

    if "phi" in raw:
        for key in _SPAN_KEYS:
            out[key] = raw[key]
        w = raw["w"]
        out["mach"] = w / air.sound_speed
        out["reynolds"] = rho * w * stations.chord[None, :] / air.viscosity
        out["v_axial_induced"] = w * np.sin(raw["phi"]) - v_inf[:, None]
        out["v_swirl_induced"] = np.maximum(
            omega[:, None] * stations.r[None, :] - w * np.cos(raw["phi"]), 0.0)
        out["stall_fraction"] = np.mean(raw["alpha"] > math.radians(14.0), axis=-1)

    return out


def _reshape(out: dict[str, np.ndarray], shape: tuple[int, ...], n_st: int) -> dict[str, np.ndarray]:
    for key, val in list(out.items()):
        if not isinstance(val, np.ndarray):
            continue
        if val.ndim == 1:
            out[key] = val.reshape(shape)
        elif val.ndim == 2 and val.shape[1] == n_st:
            out[key] = val.reshape(shape + (n_st,))
    return out


def _kernel_arrays(stations: BladeStations, tables) -> dict[str, np.ndarray]:
    """Flat, contiguous, float64 arrays ready to be handed to a kernel."""
    alpha_grid, cl_tab, cd_tab = tables
    thickness, re_ref, m_crit0, cl_max = stations.section_params()
    c = np.ascontiguousarray
    return {
        "r": c(stations.r, np.float64), "chord": c(stations.chord, np.float64),
        "twist": c(stations.twist, np.float64), "sigma": c(stations.solidity, np.float64),
        "thickness": c(thickness, np.float64), "re_ref": c(re_ref, np.float64),
        "m_crit0": c(m_crit0, np.float64), "cl_max": c(cl_max, np.float64),
        "dr": c(stations.dr, np.float64),
        "cl_tab": c(cl_tab, np.float64).ravel(), "cd_tab": c(cd_tab, np.float64).ravel(),
        "n_alpha": int(cl_tab.shape[1]), "alpha0": float(alpha_grid[0]),
        "d_alpha": float(alpha_grid[1] - alpha_grid[0]),
    }


# ---------------------------------------------------------------------------
# NumPy reference backend
# ---------------------------------------------------------------------------

class NumpyBackend(Backend):
    name = "numpy"
    priority = 10

    def _probe(self) -> BackendInfo:
        return BackendInfo(self.name, True, "cpu",
                           f"NumPy {np.__version__}, vectorised reference path")

    def solve_batch(self, stations, rpm, v_inf, air=None, opts=None, tables=None,
                    want_spanwise=True):
        opts = opts or SolverOptions()
        air = air or SEA_LEVEL
        t0 = time.perf_counter()
        out = _numpy_solve_batch(stations, rpm, v_inf, air, opts, tables)
        out["backend"] = self.name
        out["elapsed_s"] = time.perf_counter() - t0
        return out


# ---------------------------------------------------------------------------
# Numba CPU backend
# ---------------------------------------------------------------------------

class NumbaCPUBackend(Backend):
    name = "numba-cpu"
    priority = 20

    def _probe(self) -> BackendInfo:
        import numba
        return BackendInfo(self.name, True, "cpu",
                           f"Numba {numba.__version__}, {numba.config.NUMBA_NUM_THREADS} threads")

    def _run(self, stations, rpm, v_inf, air, opts, tables, want_spanwise):
        from .kernels import build_cpu_kernel
        kern = build_cpu_kernel()
        a = _kernel_arrays(stations, tables)

        n_cases, n_st = rpm.size, stations.n
        thrust = np.zeros(n_cases)
        torque = np.zeros(n_cases)
        conv = np.zeros(n_cases)
        span_n = n_cases * n_st if want_spanwise else 1
        span = [np.zeros(span_n) for _ in range(8)]

        kern(a["r"], a["chord"], a["twist"], a["sigma"], a["thickness"], a["re_ref"],
             a["m_crit0"], a["cl_max"], a["dr"], a["cl_tab"], a["cd_tab"],
             rpm_to_rad_s(rpm), v_inf, thrust, torque, conv, *span,
             a["n_alpha"], a["alpha0"], a["d_alpha"], air.density, air.viscosity,
             air.sound_speed, stations.geometry.radius, stations.geometry.hub_radius,
             stations.geometry.n_blades, opts.flags(), opts.n_bisect,
             PHI_LO, PHI_HI, 1 if want_spanwise else 0)

        out = {"thrust": thrust, "torque": torque, "converged": conv}
        if want_spanwise:
            for key, arr in zip(_SPAN_KEYS, span):
                out[key] = arr.reshape(n_cases, n_st)
        return out


# ---------------------------------------------------------------------------
# Numba CUDA backend
# ---------------------------------------------------------------------------

class NumbaCUDABackend(Backend):
    name = "numba-cuda"
    priority = 40

    def _probe(self) -> BackendInfo:
        from numba import cuda
        if not cuda.is_available():
            return BackendInfo(self.name, False, reason="no CUDA device visible to Numba")
        dev = cuda.get_current_device()
        cc = ".".join(str(v) for v in dev.compute_capability)
        name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
        free, total = cuda.current_context().get_memory_info()
        return BackendInfo(self.name, True, name,
                           f"SM {cc}, {total / 2**30:.1f} GiB ({free / 2**30:.1f} GiB free), "
                           f"{dev.MULTIPROCESSOR_COUNT} SMs")

    def _run(self, stations, rpm, v_inf, air, opts, tables, want_spanwise):
        from numba import cuda
        from .kernels import THREADS_PER_BLOCK, build_cuda_kernel, cuda_launch_config

        kern = build_cuda_kernel()
        a = _kernel_arrays(stations, tables)
        n_cases, n_st = rpm.size, stations.n

        key = id(stations)
        dev = self._device_cache.get(key)
        if dev is None:
            dev = {k: cuda.to_device(a[k]) for k in
                   ("r", "chord", "twist", "sigma", "thickness", "re_ref",
                    "m_crit0", "cl_max", "dr", "cl_tab", "cd_tab")}
            self._device_cache.clear()
            self._device_cache[key] = dev

        d_omega = cuda.to_device(np.ascontiguousarray(rpm_to_rad_s(rpm)))
        d_v = cuda.to_device(np.ascontiguousarray(v_inf))
        d_t = cuda.device_array(n_cases, np.float64)
        d_q = cuda.device_array(n_cases, np.float64)
        d_c = cuda.device_array(n_cases, np.float64)
        span_n = n_cases * n_st if want_spanwise else 1
        d_span = [cuda.device_array(span_n, np.float64) for _ in range(8)]

        blocks, threads = cuda_launch_config(n_cases, THREADS_PER_BLOCK)
        kern[blocks, threads](
            dev["r"], dev["chord"], dev["twist"], dev["sigma"], dev["thickness"],
            dev["re_ref"], dev["m_crit0"], dev["cl_max"], dev["dr"],
            dev["cl_tab"], dev["cd_tab"],
            d_omega, d_v, d_t, d_q, d_c, *d_span,
            a["n_alpha"], a["alpha0"], a["d_alpha"], air.density, air.viscosity,
            air.sound_speed, stations.geometry.radius, stations.geometry.hub_radius,
            stations.geometry.n_blades, opts.flags(), opts.n_bisect,
            PHI_LO, PHI_HI, 1 if want_spanwise else 0)
        cuda.synchronize()

        out = {"thrust": d_t.copy_to_host(), "torque": d_q.copy_to_host(),
               "converged": d_c.copy_to_host()}
        if want_spanwise:
            for key_name, arr in zip(_SPAN_KEYS, d_span):
                out[key_name] = arr.copy_to_host().reshape(n_cases, n_st)
        return out


# ---------------------------------------------------------------------------
# CuPy raw-CUDA-C backend
# ---------------------------------------------------------------------------

class CupyBackend(Backend):
    name = "cupy"
    priority = 50

    def _probe(self) -> BackendInfo:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n == 0:
            return BackendInfo(self.name, False, reason="no CUDA device visible to CuPy")
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        return BackendInfo(self.name, True, name,
                           f"CuPy {cp.__version__}, SM {props['major']}.{props['minor']}, "
                           f"{props['multiProcessorCount']} SMs")

    def _run(self, stations, rpm, v_inf, air, opts, tables, want_spanwise):
        import cupy as cp
        from .kernels_raw import get_raw_kernel

        kern, threads = get_raw_kernel()
        a = _kernel_arrays(stations, tables)
        n_cases, n_st = rpm.size, stations.n

        key = id(stations)
        dev = self._device_cache.get(key)
        if dev is None:
            dev = {k: cp.asarray(a[k], dtype=cp.float64) for k in
                   ("r", "chord", "twist", "sigma", "thickness", "re_ref",
                    "m_crit0", "cl_max", "dr", "cl_tab", "cd_tab")}
            self._device_cache.clear()
            self._device_cache[key] = dev

        d_omega = cp.asarray(rpm_to_rad_s(rpm), dtype=cp.float64)
        d_v = cp.asarray(v_inf, dtype=cp.float64)
        d_t = cp.empty(n_cases, dtype=cp.float64)
        d_q = cp.empty(n_cases, dtype=cp.float64)
        d_c = cp.empty(n_cases, dtype=cp.float64)
        span_n = n_cases * n_st if want_spanwise else 1
        d_span = [cp.empty(span_n, dtype=cp.float64) for _ in range(8)]

        kern((n_cases,), (threads,), (
            dev["r"], dev["chord"], dev["twist"], dev["sigma"], dev["thickness"],
            dev["re_ref"], dev["m_crit0"], dev["cl_max"], dev["dr"],
            dev["cl_tab"], dev["cd_tab"],
            d_omega, d_v, d_t, d_q, d_c, *d_span,
            np.int32(n_st), np.int32(a["n_alpha"]), np.int32(n_cases),
            np.float64(a["alpha0"]), np.float64(a["d_alpha"]),
            np.float64(air.density), np.float64(air.viscosity),
            np.float64(air.sound_speed), np.float64(stations.geometry.radius),
            np.float64(stations.geometry.hub_radius),
            np.int32(stations.geometry.n_blades), np.int32(opts.flags()),
            np.int32(opts.n_bisect), np.float64(PHI_LO), np.float64(PHI_HI),
            np.int32(1 if want_spanwise else 0)))
        cp.cuda.runtime.deviceSynchronize()

        out = {"thrust": cp.asnumpy(d_t), "torque": cp.asnumpy(d_q),
               "converged": cp.asnumpy(d_c)}
        if want_spanwise:
            for key_name, arr in zip(_SPAN_KEYS, d_span):
                out[key_name] = cp.asnumpy(arr).reshape(n_cases, n_st)
        return out


# ---------------------------------------------------------------------------
# Torch backend (vectorised, runs on CUDA or MPS without a custom kernel)
# ---------------------------------------------------------------------------

class OpenCLBackend(Backend):
    """Portable GPU path.  Runs on AMD, Intel, Apple -- and on a CPU runtime."""

    name = "opencl"

    def __init__(self, prefer: str = "gpu") -> None:
        super().__init__()
        self.prefer = prefer

    @property                                            # type: ignore[override]
    def priority(self) -> int:                           # noqa: D401
        """Rank above the CPU paths only when this is really a GPU.

        OpenCL on a CPU runtime (POCL, Intel's CPU device) is a correctness
        win -- it exercises the same kernel -- but it is usually *slower* than
        the threaded Numba path, so auto-selection should not prefer it.
        """
        info = self.probe()
        if not info.available:
            return 0
        # Below numba-cuda (40) on an NVIDIA box -- native CUDA is normally
        # faster there -- but above the CPU paths on any other GPU.
        return 35 if "GPU" in info.detail.upper() else 15

    def _probe(self) -> BackendInfo:
        from .opencl import available_devices, describe_device, pick_device
        if not available_devices():
            return BackendInfo(self.name, False, reason="no OpenCL platform found")
        try:
            _, device = pick_device(self.prefer)
        except RuntimeError as exc:
            return BackendInfo(self.name, False, reason=str(exc))
        import pyopencl as cl
        return BackendInfo(self.name, True, device.name.strip(),
                           f"PyOpenCL {cl.VERSION_TEXT}, {describe_device(device)}")

    def _run(self, stations, rpm, v_inf, air, opts, tables, want_spanwise):
        from .opencl import solve_batch_opencl
        return solve_batch_opencl(stations, rpm, v_inf, air, opts, tables,
                                  want_spanwise, self.prefer)


class TorchBackend(Backend):
    name = "torch"
    priority = 30

    def _probe(self) -> BackendInfo:
        import torch
        if torch.cuda.is_available():
            return BackendInfo(self.name, True, torch.cuda.get_device_name(0),
                               f"PyTorch {torch.__version__} CUDA")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return BackendInfo(self.name, True, "mps", f"PyTorch {torch.__version__} Metal")
        return BackendInfo(self.name, False, reason="PyTorch present but no GPU device")

    def _run(self, stations, rpm, v_inf, air, opts, tables, want_spanwise):
        from .torch_path import solve_batch_torch
        return solve_batch_torch(stations, rpm, v_inf, air, opts, tables, want_spanwise)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Backend] = {}


def _registry() -> dict[str, Backend]:
    if not _REGISTRY:
        for cls in (NumpyBackend, NumbaCPUBackend, TorchBackend, OpenCLBackend,
                    NumbaCUDABackend, CupyBackend):
            _REGISTRY[cls.name] = cls()
    return _REGISTRY


def list_backends() -> list[BackendInfo]:
    """Probe every backend once and report what this machine can do."""
    return [b.probe() for b in sorted(_registry().values(), key=lambda b: -b.priority)]


def get_backend(name: str = "auto") -> Backend:
    """Return a backend by name, or the best available one for ``"auto"``.

    An explicitly named backend that turns out to be unavailable falls back with
    a warning rather than an exception -- a notebook that asks for CUDA should
    still run on a CPU runtime.
    """
    reg = _registry()
    forced = os.environ.get("PROPWASH_BACKEND")
    if forced and name == "auto":
        name = forced

    if name != "auto":
        key = name.lower()
        if key not in reg:
            raise KeyError(f"unknown backend {name!r}; have {sorted(reg)}")
        backend = reg[key]
        if backend.available:
            return backend
        import warnings
        warnings.warn(f"backend {name!r} unavailable ({backend.probe().reason}); "
                      f"falling back to auto-selection", RuntimeWarning, stacklevel=2)

    for backend in sorted(reg.values(), key=lambda b: -b.priority):
        if backend.available:
            return backend
    return reg["numpy"]


def describe_environment() -> str:
    """Human-readable compute inventory, for the GUI status bar and the CLI."""
    lines = ["Propwash compute backends:"]
    for info in list_backends():
        lines.append(f"  {info}")
    lines.append(f"  -> selected: {get_backend().name}")
    return "\n".join(lines)


__all__ = ["Backend", "BackendInfo", "NumpyBackend", "NumbaCPUBackend",
           "NumbaCUDABackend", "CupyBackend", "TorchBackend", "OpenCLBackend",
           "get_backend", "list_backends", "describe_environment"]
