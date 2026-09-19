"""Numba kernels for CPU and CUDA, built from one set of source functions.

The trick in :func:`_compile_device_math` is worth explaining.  The functions in
:mod:`propwash.accel._device_math` call each other through module globals.  If
we simply decorated them in place, the CPU and GPU builds would fight over the
same names.  Instead we rebuild each function object against a *fresh* globals
dict in which its dependencies have already been replaced by their compiled
counterparts.  Same bytecode, two targets, zero duplicated physics.

The CUDA kernel uses one block per operating point and threads across blade
stations, finishing with a shared-memory tree reduction for thrust and torque.
That layout is deliberate:

* Blade stations of one propeller are exactly the values that must be summed,
  so the reduction is block-local and needs no global atomics.
* The bisection is a fixed trip count with no early exit, so every thread in the
  warp retires together -- no divergence, no tail effect.
* The polar tables are small (a few hundred kB) and read by every thread, so
  they sit happily in the read-only cache.
"""

from __future__ import annotations

import types
from typing import Any, Callable

from . import _device_math as _dm

# Imported at module scope, not inside the factory, for two reasons: the CUDA
# simulator swaps ``cuda`` in the kernel's *globals* (a closure cell would be
# invisible to it), and a missing Numba must degrade to "backend unavailable"
# rather than an ImportError at package import time.
try:
    from numba import cuda, float64
except Exception:  # pragma: no cover - Numba is an optional dependency
    cuda = None
    float64 = None

# Compiled in dependency order: each entry may only call the ones above it.
_FUNCTION_ORDER = ("wrap_pi", "interp_table", "prandtl_factor",
                   "apply_corrections", "element_state", "solve_phi")

_CONSTANTS = ("PI", "TWO_PI", "HALF_PI", "FLAG_TIP_LOSS", "FLAG_HUB_LOSS",
              "FLAG_REYNOLDS", "FLAG_MACH")

THREADS_PER_BLOCK = 256


def _compile_device_math(decorator: Callable[[Callable], Any]) -> dict[str, Any]:
    """Recompile the shared scalar physics for one Numba target."""
    import math

    ns: dict[str, Any] = {"math": math}
    for name in _CONSTANTS:
        ns[name] = getattr(_dm, name)

    for name in _FUNCTION_ORDER:
        original = getattr(_dm, name)
        rebound = types.FunctionType(original.__code__, ns, original.__name__,
                                     original.__defaults__, original.__closure__)
        ns[name] = decorator(rebound)
    return ns


# ---------------------------------------------------------------------------
# CPU (Numba, threaded over operating points)
# ---------------------------------------------------------------------------

_cpu_cache: dict[str, Any] = {}


def build_cpu_kernel(fastmath: bool = False):
    """JIT the multi-core CPU kernel (compiled once, then cached)."""
    key = f"cpu{int(fastmath)}"
    if key in _cpu_cache:
        return _cpu_cache[key]

    from numba import njit, prange

    dev = _compile_device_math(njit(inline="always", fastmath=fastmath, cache=False))
    solve_phi = dev["solve_phi"]
    element_state = dev["element_state"]

    @njit(parallel=True, fastmath=fastmath, cache=False, nogil=True)
    def kernel(r, chord, twist, sigma, thickness, re_ref, m_crit0, dr,
               cl_tab, cd_tab, omega, v_inf,
               thrust, torque, converged, phi_out, alpha_out, cl_out, cd_out,
               w_out, dt_out, dq_out, floss_out,
               n_alpha, alpha0, d_alpha, rho, mu, a_sound,
               r_tip, r_hub, n_blades, flags, n_bisect, phi_lo, phi_hi, want_span):
        n_cases = omega.shape[0]
        n_st = r.shape[0]
        for c in prange(n_cases):
            t_sum = 0.0
            q_sum = 0.0
            conv = 0.0
            for s in range(n_st):
                base = s * n_alpha
                omega_r = omega[c] * r[s]
                phi, brack = solve_phi(
                    twist[s], r[s], chord[s], sigma[s], thickness[s], re_ref[s],
                    m_crit0[s], base, n_alpha, alpha0, d_alpha, cl_tab, cd_tab,
                    omega_r, v_inf[c], rho, mu, a_sound, r_tip, r_hub, n_blades,
                    flags, n_bisect, phi_lo, phi_hi)
                _, w, cl, cd, cn, ct, floss, alpha = element_state(
                    phi, twist[s], r[s], chord[s], sigma[s], thickness[s],
                    re_ref[s], m_crit0[s], base, n_alpha, alpha0, d_alpha,
                    cl_tab, cd_tab, omega_r, v_inf[c], rho, mu, a_sound,
                    r_tip, r_hub, n_blades, flags)

                k = 0.5 * rho * n_blades * chord[s] * w * w
                dt = k * cn
                dq = k * ct * r[s]
                t_sum += dt * dr[s]
                q_sum += dq * dr[s]
                conv += brack

                if want_span == 1:
                    o = c * n_st + s
                    phi_out[o] = phi
                    alpha_out[o] = alpha
                    cl_out[o] = cl
                    cd_out[o] = cd
                    w_out[o] = w
                    dt_out[o] = dt
                    dq_out[o] = dq
                    floss_out[o] = floss

            thrust[c] = t_sum
            torque[c] = q_sum
            converged[c] = conv / n_st

    _cpu_cache[key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# CUDA (Numba, one block per operating point)
# ---------------------------------------------------------------------------

_cuda_cache: dict[str, Any] = {}


def build_cuda_kernel(fastmath: bool = False, nthreads: int = THREADS_PER_BLOCK):
    """JIT the CUDA kernel.  Raises if no CUDA toolchain is importable.

    ``nthreads`` is baked in as a compile-time constant because the shared-memory
    reduction buffers must be statically sized.  It is exposed so block size can
    be tuned per device, and so the CUDA simulator can run a tiny block.
    """
    if cuda is None:
        raise RuntimeError("Numba is not installed; the CUDA backend is unavailable")

    key = f"cuda{int(fastmath)}x{int(nthreads)}"
    if key in _cuda_cache:
        return _cuda_cache[key]

    dev = _compile_device_math(cuda.jit(device=True, inline=True, fastmath=fastmath))
    solve_phi = dev["solve_phi"]
    element_state = dev["element_state"]

    nthreads = int(nthreads)

    @cuda.jit(fastmath=fastmath)
    def kernel(r, chord, twist, sigma, thickness, re_ref, m_crit0, dr,
               cl_tab, cd_tab, omega, v_inf,
               thrust, torque, converged, phi_out, alpha_out, cl_out, cd_out,
               w_out, dt_out, dq_out, floss_out,
               n_alpha, alpha0, d_alpha, rho, mu, a_sound,
               r_tip, r_hub, n_blades, flags, n_bisect, phi_lo, phi_hi, want_span):
        case = cuda.blockIdx.x
        if case >= omega.shape[0]:
            return

        tid = cuda.threadIdx.x
        n_st = r.shape[0]

        s_t = cuda.shared.array(nthreads, float64)
        s_q = cuda.shared.array(nthreads, float64)
        s_c = cuda.shared.array(nthreads, float64)

        acc_t = 0.0
        acc_q = 0.0
        acc_c = 0.0

        om = omega[case]
        vv = v_inf[case]

        # Grid-stride over stations so any element count fits a fixed block.
        s = tid
        while s < n_st:
            base = s * n_alpha
            omega_r = om * r[s]
            phi, brack = solve_phi(
                twist[s], r[s], chord[s], sigma[s], thickness[s], re_ref[s],
                m_crit0[s], base, n_alpha, alpha0, d_alpha, cl_tab, cd_tab,
                omega_r, vv, rho, mu, a_sound, r_tip, r_hub, n_blades,
                flags, n_bisect, phi_lo, phi_hi)
            _, w, cl, cd, cn, ct, floss, alpha = element_state(
                phi, twist[s], r[s], chord[s], sigma[s], thickness[s],
                re_ref[s], m_crit0[s], base, n_alpha, alpha0, d_alpha,
                cl_tab, cd_tab, omega_r, vv, rho, mu, a_sound,
                r_tip, r_hub, n_blades, flags)

            k = 0.5 * rho * n_blades * chord[s] * w * w
            dt = k * cn
            dq = k * ct * r[s]
            acc_t += dt * dr[s]
            acc_q += dq * dr[s]
            acc_c += brack

            if want_span == 1:
                o = case * n_st + s
                phi_out[o] = phi
                alpha_out[o] = alpha
                cl_out[o] = cl
                cd_out[o] = cd
                w_out[o] = w
                dt_out[o] = dt
                dq_out[o] = dq
                floss_out[o] = floss

            s += nthreads

        s_t[tid] = acc_t
        s_q[tid] = acc_q
        s_c[tid] = acc_c
        cuda.syncthreads()

        # Shared-memory tree reduction over the block.
        stride = nthreads // 2
        while stride > 0:
            if tid < stride:
                s_t[tid] += s_t[tid + stride]
                s_q[tid] += s_q[tid + stride]
                s_c[tid] += s_c[tid + stride]
            cuda.syncthreads()
            stride //= 2

        if tid == 0:
            thrust[case] = s_t[0]
            torque[case] = s_q[0]
            converged[case] = s_c[0] / n_st

    _cuda_cache[key] = kernel
    return kernel


def cuda_launch_config(n_cases: int, nthreads: int = THREADS_PER_BLOCK) -> tuple[int, int]:
    """Blocks and threads for the CUDA kernel: one block per operating point."""
    return max(int(n_cases), 1), int(nthreads)


__all__ = ["build_cpu_kernel", "build_cuda_kernel", "cuda_launch_config",
           "THREADS_PER_BLOCK"]
