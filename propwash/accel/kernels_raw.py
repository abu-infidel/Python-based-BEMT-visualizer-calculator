"""Hand-written CUDA C kernel, compiled at runtime by CuPy's NVRTC bridge.

Numba's CUDA target is excellent and covers most of what this program needs.
This module exists for the cases it does not: it is the version you can read as
CUDA, profile with Nsight, tune the block size of, and hand to someone who
wants to see the actual device code.  It is a line-for-line transcription of
:mod:`propwash.accel._device_math`, and the test suite checks the two against
each other on any machine that has a GPU.

Layout mirrors the Numba kernel: one block per operating point, threads striding
over blade stations, and a shared-memory tree reduction for thrust and torque.
"""

from __future__ import annotations

from typing import Any

PW_THREADS = 256

CUDA_SOURCE = r"""
#define PW_PI       3.141592653589793
#define PW_TWO_PI   6.283185307179586
#define PW_THREADS  256

#define FLAG_TIP_LOSS 1
#define FLAG_HUB_LOSS 2
#define FLAG_REYNOLDS 4
#define FLAG_MACH     8

/* ---------------------------------------------------------------- helpers */

__device__ __forceinline__ double pw_wrap_pi(double a)
{
    while (a >  PW_PI) a -= PW_TWO_PI;
    while (a < -PW_PI) a += PW_TWO_PI;
    return a;
}

/* Linear interpolation on a uniform alpha grid.  `base` is station*n_alpha. */
__device__ __forceinline__ double pw_interp(double alpha, double alpha0,
                                            double d_alpha, int n_alpha,
                                            const double* __restrict__ table,
                                            int base)
{
    double a = pw_wrap_pi(alpha);
    double t = (a - alpha0) / d_alpha;
    int i = (int)floor(t);
    if (i < 0)             i = 0;
    else if (i > n_alpha - 2) i = n_alpha - 2;
    double w = t - (double)i;
    if (w < 0.0)      w = 0.0;
    else if (w > 1.0) w = 1.0;
    return table[base + i] * (1.0 - w) + table[base + i + 1] * w;
}

/* Combined Prandtl tip and hub loss factor. */
__device__ __forceinline__ double pw_prandtl(double phi, double r, double r_tip,
                                             double r_hub, int n_blades, int flags)
{
    double s = fabs(sin(phi));
    if (s < 1.0e-7) s = 1.0e-7;
    double f = 1.0;

    if (flags & FLAG_TIP_LOSS) {
        double ft = 0.5 * (double)n_blades * (r_tip - r) / (r * s);
        if (ft < 0.0) ft = 0.0;
        double e = exp(-ft);
        if (e > 1.0) e = 1.0;
        f *= (2.0 / PW_PI) * acos(e);
    }
    if ((flags & FLAG_HUB_LOSS) && r_hub > 0.0) {
        double fh = 0.5 * (double)n_blades * (r - r_hub) / (r_hub * s);
        if (fh < 0.0) fh = 0.0;
        double e = exp(-fh);
        if (e > 1.0) e = 1.0;
        f *= (2.0 / PW_PI) * acos(e);
    }
    if (f < 1.0e-4)      f = 1.0e-4;
    else if (f > 1.0)    f = 1.0;
    return f;
}

__device__ __forceinline__ void pw_corrections(double* cl, double* cd, double w,
                                               double chord, double rho, double mu,
                                               double a_sound, double thickness,
                                               double re_ref, double m_crit0, int flags)
{
    if (flags & FLAG_REYNOLDS) {
        double re = rho * w * chord / mu;
        if (re < 1.0e3) re = 1.0e3;
        double ratio = re_ref / re;
        double scale = pow(ratio, 0.2);
        double low = 2.0e4 / re - 1.0;
        if (low > 0.0) scale *= 1.0 + 0.45 * low;
        if (scale < 0.5)      scale = 0.5;
        else if (scale > 4.0) scale = 4.0;
        *cd *= scale;

        double cls = 1.0 - 0.14 * log10(ratio);
        if (cls < 0.45)       cls = 0.45;
        else if (cls > 1.12)  cls = 1.12;
        *cl *= cls;
    }
    if (flags & FLAG_MACH) {
        double m = w / a_sound;
        if (m < 0.0)        m = 0.0;
        else if (m > 0.92)  m = 0.92;
        double den = 1.0 - m * m;
        if (den < 1.0e-3) den = 1.0e-3;
        *cl /= sqrt(den);
        double m_dd = m_crit0 - 0.1 * fabs(*cl) - thickness;
        double dm = m - m_dd;
        if (dm > 0.0) *cd += 20.0 * dm * dm * dm * dm;
    }
    if (*cd < 1.0e-5) *cd = 1.0e-5;
}

/* Full element state at a trial phi.  Returns the residual; the rest is out-params.
 *
 *   R = omega*r * (4 F sin^2 phi - sigma Cn) - V * (sigma Ct + 4 F sin phi cos phi)
 */
__device__ __forceinline__ double pw_element(
        double phi, double twist, double r, double chord, double sigma,
        double thickness, double re_ref, double m_crit0, int base, int n_alpha,
        double alpha0, double d_alpha,
        const double* __restrict__ cl_tab, const double* __restrict__ cd_tab,
        double omega_r, double v_inf, double rho, double mu, double a_sound,
        double r_tip, double r_hub, int n_blades, int flags,
        double* w_out, double* cl_out, double* cd_out,
        double* cn_out, double* ct_out, double* f_out, double* alpha_out)
{
    double sp = sin(phi);
    double cp = cos(phi);
    double alpha = twist - phi;

    double f_loss = pw_prandtl(phi, r, r_tip, r_hub, n_blades, flags);

    double cl0 = pw_interp(alpha, alpha0, d_alpha, n_alpha, cl_tab, base);
    double cd0 = pw_interp(alpha, alpha0, d_alpha, n_alpha, cd_tab, base);

    /* Torque branch for W: strictly positive denominator on (0, pi/2), and
     * finite at V = 0 where the thrust branch degenerates. */
    double ct0 = cl0 * sp + cd0 * cp;
    double den = sigma * ct0 + 4.0 * f_loss * sp * cp;
    if (fabs(den) < 1.0e-9) den = (den >= 0.0) ? 1.0e-9 : -1.0e-9;
    double w = fabs(4.0 * f_loss * omega_r * sp / den);
    double vmin = fabs(v_inf);
    if (vmin < 1.0e-3) vmin = 1.0e-3;
    if (w < vmin) w = vmin;

    double cl = cl0, cd = cd0;
    pw_corrections(&cl, &cd, w, chord, rho, mu, a_sound, thickness, re_ref, m_crit0, flags);

    double cn = cl * cp - cd * sp;
    double ct = cl * sp + cd * cp;

    *w_out = w;   *cl_out = cl; *cd_out = cd;
    *cn_out = cn; *ct_out = ct; *f_out = f_loss; *alpha_out = alpha;

    return omega_r * (4.0 * f_loss * sp * sp - sigma * cn)
         - v_inf   * (sigma * ct + 4.0 * f_loss * sp * cp);
}

/* ----------------------------------------------------------------- kernel */

extern "C" __global__ void bemt_solve(
        const double* __restrict__ r,
        const double* __restrict__ chord,
        const double* __restrict__ twist,
        const double* __restrict__ sigma,
        const double* __restrict__ thickness,
        const double* __restrict__ re_ref,
        const double* __restrict__ m_crit0,
        const double* __restrict__ dr,
        const double* __restrict__ cl_tab,
        const double* __restrict__ cd_tab,
        const double* __restrict__ omega,
        const double* __restrict__ v_inf,
        double* __restrict__ thrust,
        double* __restrict__ torque,
        double* __restrict__ converged,
        double* __restrict__ phi_out,
        double* __restrict__ alpha_out,
        double* __restrict__ cl_out,
        double* __restrict__ cd_out,
        double* __restrict__ w_out,
        double* __restrict__ dt_out,
        double* __restrict__ dq_out,
        double* __restrict__ floss_out,
        int n_st, int n_alpha, int n_cases,
        double alpha0, double d_alpha, double rho, double mu, double a_sound,
        double r_tip, double r_hub, int n_blades, int flags, int n_bisect,
        double phi_lo, double phi_hi, int want_span)
{
    const int cs  = blockIdx.x;
    if (cs >= n_cases) return;
    const int tid = threadIdx.x;

    __shared__ double s_t[PW_THREADS];
    __shared__ double s_q[PW_THREADS];
    __shared__ double s_c[PW_THREADS];

    double acc_t = 0.0, acc_q = 0.0, acc_c = 0.0;
    const double om = omega[cs];
    const double vv = v_inf[cs];

    for (int s = tid; s < n_st; s += PW_THREADS) {
        const int base = s * n_alpha;
        const double omega_r = om * r[s];

        double w, cl, cd, cn, ct, fl, al;
        double lo = phi_lo, hi = phi_hi;

        double r_lo = pw_element(lo, twist[s], r[s], chord[s], sigma[s], thickness[s],
                                 re_ref[s], m_crit0[s], base, n_alpha, alpha0, d_alpha,
                                 cl_tab, cd_tab, omega_r, vv, rho, mu, a_sound,
                                 r_tip, r_hub, n_blades, flags,
                                 &w, &cl, &cd, &cn, &ct, &fl, &al);
        double r_hi = pw_element(hi, twist[s], r[s], chord[s], sigma[s], thickness[s],
                                 re_ref[s], m_crit0[s], base, n_alpha, alpha0, d_alpha,
                                 cl_tab, cd_tab, omega_r, vv, rho, mu, a_sound,
                                 r_tip, r_hub, n_blades, flags,
                                 &w, &cl, &cd, &cn, &ct, &fl, &al);

        const double bracketed = ((r_lo * r_hi) < 0.0) ? 1.0 : 0.0;

        /* Fixed trip count, no early exit: every thread in the warp retires
         * together, which is the whole point of bisecting instead of iterating. */
        for (int it = 0; it < n_bisect; ++it) {
            double mid = 0.5 * (lo + hi);
            double r_mid = pw_element(mid, twist[s], r[s], chord[s], sigma[s],
                                      thickness[s], re_ref[s], m_crit0[s], base,
                                      n_alpha, alpha0, d_alpha, cl_tab, cd_tab,
                                      omega_r, vv, rho, mu, a_sound, r_tip, r_hub,
                                      n_blades, flags,
                                      &w, &cl, &cd, &cn, &ct, &fl, &al);
            if ((r_lo * r_mid) <= 0.0) { hi = mid; r_hi = r_mid; }
            else                       { lo = mid; r_lo = r_mid; }
        }

        double phi = 0.5 * (lo + hi);
        if (bracketed == 0.0) {
            double vq = (vv > 1.0e-6) ? vv : 1.0e-6;
            double oq = (omega_r > 1.0e-6) ? omega_r : 1.0e-6;
            phi = atan2(vq, oq);
        }

        pw_element(phi, twist[s], r[s], chord[s], sigma[s], thickness[s],
                   re_ref[s], m_crit0[s], base, n_alpha, alpha0, d_alpha,
                   cl_tab, cd_tab, omega_r, vv, rho, mu, a_sound,
                   r_tip, r_hub, n_blades, flags,
                   &w, &cl, &cd, &cn, &ct, &fl, &al);

        const double k  = 0.5 * rho * (double)n_blades * chord[s] * w * w;
        const double dt = k * cn;
        const double dq = k * ct * r[s];
        acc_t += dt * dr[s];
        acc_q += dq * dr[s];
        acc_c += bracketed;

        if (want_span) {
            const int o = cs * n_st + s;
            phi_out[o]   = phi;
            alpha_out[o] = al;
            cl_out[o]    = cl;
            cd_out[o]    = cd;
            w_out[o]     = w;
            dt_out[o]    = dt;
            dq_out[o]    = dq;
            floss_out[o] = fl;
        }
    }

    s_t[tid] = acc_t;
    s_q[tid] = acc_q;
    s_c[tid] = acc_c;
    __syncthreads();

    for (int stride = PW_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_t[tid] += s_t[tid + stride];
            s_q[tid] += s_q[tid + stride];
            s_c[tid] += s_c[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        thrust[cs]    = s_t[0];
        torque[cs]    = s_q[0];
        converged[cs] = s_c[0] / (double)n_st;
    }
}
"""

_kernel_cache: dict[str, Any] = {}


def get_raw_kernel() -> tuple[Any, int]:
    """Compile (once) and return ``(kernel, threads_per_block)``.

    ``--use_fast_math`` is deliberately *not* passed: the bisection's sign test
    is exact-arithmetic sensitive, and a reassociated residual can flip a
    comparison near the root.
    """
    if "k" not in _kernel_cache:
        import cupy as cp
        module = cp.RawModule(code=CUDA_SOURCE, options=("-std=c++11",),
                              name_expressions=None)
        _kernel_cache["k"] = module.get_function("bemt_solve")
        _kernel_cache["m"] = module
    return _kernel_cache["k"], PW_THREADS


def kernel_source() -> str:
    """The CUDA C, for saving to a .cu file or printing in a notebook."""
    return CUDA_SOURCE


__all__ = ["CUDA_SOURCE", "PW_THREADS", "get_raw_kernel", "kernel_source"]
