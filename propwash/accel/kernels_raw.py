"""One hand-written C kernel, rendered for CUDA and for OpenCL.

Numba's CUDA target covers most of what this program needs.  This module exists
for the rest: the version you can read as C, profile with Nsight or Radeon GPU
Profiler, tune the work-group size of -- and, crucially, the version that runs
on hardware that has never heard of CUDA.

**Why a template rather than two files.**  The two kernels differ only in
qualifiers and in how they name the block and thread index; the physics is
identical.  Keeping two copies means the physics drifts, and a drift between a
CUDA path and an OpenCL path is close to undebuggable -- you would need both
kinds of hardware in front of you to notice.  So the body is written once with
a handful of tokens, and :func:`kernel_source` substitutes them per target.
``kernel_source("cuda")`` returns real CUDA C; ``kernel_source("opencl")``
returns real OpenCL C.  Both are compiled in the test suite.

Layout is the same either way: one block / work-group per operating point,
threads striding over blade stations, and a local-memory tree reduction for
thrust and torque.  Blade stations are exactly the values that must be summed,
so the reduction stays block-local and needs no global atomics.
"""

from __future__ import annotations

from typing import Any

PW_THREADS = 256

#: Substitutions per target.  Everything that differs between CUDA and OpenCL
#: lives here; the kernel body below is shared verbatim.
_TARGETS: dict[str, dict[str, str]] = {
    "cuda": {
        "PROLOGUE": "",
        "PW_GLOBAL": "",
        "PW_RESTRICT": "__restrict__",
        "PW_DEVFN": "__device__ __forceinline__",
        "PW_KERNEL": 'extern "C" __global__',
        "PW_LOCAL": "__shared__",
        "PW_BARRIER": "__syncthreads()",
        "PW_GROUP_ID": "blockIdx.x",
        "PW_LOCAL_ID": "threadIdx.x",
    },
    "opencl": {
        "PROLOGUE": "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
        "PW_GLOBAL": "__global",
        "PW_RESTRICT": "restrict",
        "PW_DEVFN": "inline",
        "PW_KERNEL": "__kernel",
        "PW_LOCAL": "__local",
        "PW_BARRIER": "barrier(CLK_LOCAL_MEM_FENCE)",
        "PW_GROUP_ID": "get_group_id(0)",
        "PW_LOCAL_ID": "get_local_id(0)",
    },
}

_TEMPLATE = r"""{PROLOGUE}
#define PW_PI       3.141592653589793
#define PW_TWO_PI   6.283185307179586
#define PW_THREADS  {THREADS}

#define FLAG_TIP_LOSS 1
#define FLAG_HUB_LOSS 2
#define FLAG_REYNOLDS 4
#define FLAG_MACH     8

/* ---------------------------------------------------------------- helpers */

{PW_DEVFN} double pw_wrap_pi(double a)
{{
    while (a >  PW_PI) a -= PW_TWO_PI;
    while (a < -PW_PI) a += PW_TWO_PI;
    return a;
}}

/* Linear interpolation on a uniform alpha grid.  `base` is station*n_alpha. */
{PW_DEVFN} double pw_interp(double alpha, double alpha0, double d_alpha,
                            int n_alpha, {PW_GLOBAL} const double * {PW_RESTRICT} table,
                            int base)
{{
    double a = pw_wrap_pi(alpha);
    double t = (a - alpha0) / d_alpha;
    int i = (int)floor(t);
    if (i < 0)                i = 0;
    else if (i > n_alpha - 2) i = n_alpha - 2;
    double w = t - (double)i;
    if (w < 0.0)      w = 0.0;
    else if (w > 1.0) w = 1.0;
    return table[base + i] * (1.0 - w) + table[base + i + 1] * w;
}}

/* Combined Prandtl tip and hub loss factor. */
{PW_DEVFN} double pw_prandtl(double phi, double r, double r_tip, double r_hub,
                             int n_blades, int flags)
{{
    double s = fabs(sin(phi));
    if (s < 1.0e-7) s = 1.0e-7;
    double f = 1.0;

    if (flags & FLAG_TIP_LOSS) {{
        double ft = 0.5 * (double)n_blades * (r_tip - r) / (r * s);
        if (ft < 0.0) ft = 0.0;
        double e = exp(-ft);
        if (e > 1.0) e = 1.0;
        f *= (2.0 / PW_PI) * acos(e);
    }}
    if ((flags & FLAG_HUB_LOSS) && r_hub > 0.0) {{
        double fh = 0.5 * (double)n_blades * (r - r_hub) / (r_hub * s);
        if (fh < 0.0) fh = 0.0;
        double e = exp(-fh);
        if (e > 1.0) e = 1.0;
        f *= (2.0 / PW_PI) * acos(e);
    }}
    if (f < 1.0e-4)   f = 1.0e-4;
    else if (f > 1.0) f = 1.0;
    return f;
}}

{PW_DEVFN} double pw_mach_cl_ceiling(double mach, double cl_max)
{{
    double excess = mach - 0.35;
    if (excess < 0.0) excess = 0.0;
    double factor = 1.0 - 0.9 * pow(excess, 1.5);
    if (factor < 0.35) factor = 0.35;
    return cl_max * factor;
}}

{PW_DEVFN} void pw_corrections(double *cl, double *cd, double w, double chord,
                               double rho, double mu, double a_sound,
                               double thickness, double re_ref, double m_crit0,
                               double cl_max, int flags)
{{
    if (flags & FLAG_REYNOLDS) {{
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
        if (cls < 0.45)      cls = 0.45;
        else if (cls > 1.12) cls = 1.12;
        *cl *= cls;
    }}
    if (flags & FLAG_MACH) {{
        double m = w / a_sound;
        if (m < 0.0)       m = 0.0;
        else if (m > 0.92) m = 0.92;
        double den = 1.0 - m * m;
        if (den < 1.0e-3) den = 1.0e-3;
        *cl /= sqrt(den);
        /* Compressibility raises the lift slope but lowers the stall ceiling. */
        double ceiling = pw_mach_cl_ceiling(m, cl_max);
        if (*cl >  ceiling) *cl =  ceiling;
        if (*cl < -ceiling) *cl = -ceiling;
        double m_dd = m_crit0 - 0.1 * fabs(*cl) - thickness;
        double dm = m - m_dd;
        if (dm > 0.0) *cd += 20.0 * dm * dm * dm * dm;
    }}
    if (*cd < 1.0e-5) *cd = 1.0e-5;
}}

/* Full element state at a trial phi.  Returns the residual; the rest is out-params.
 *
 *   R = omega*r * (4 F sin^2 phi - sigma Cn) - V * (sigma Ct + 4 F sin phi cos phi)
 */
{PW_DEVFN} double pw_element(
        double phi, double twist, double r, double chord, double sigma,
        double thickness, double re_ref, double m_crit0, double cl_max,
        int base, int n_alpha,
        double alpha0, double d_alpha,
        {PW_GLOBAL} const double * {PW_RESTRICT} cl_tab,
        {PW_GLOBAL} const double * {PW_RESTRICT} cd_tab,
        double omega_r, double v_inf, double rho, double mu, double a_sound,
        double r_tip, double r_hub, int n_blades, int flags,
        double *w_out, double *cl_out, double *cd_out,
        double *cn_out, double *ct_out, double *f_out, double *alpha_out)
{{
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
    pw_corrections(&cl, &cd, w, chord, rho, mu, a_sound, thickness, re_ref,
                   m_crit0, cl_max, flags);

    double cn = cl * cp - cd * sp;
    double ct = cl * sp + cd * cp;

    *w_out = w;   *cl_out = cl; *cd_out = cd;
    *cn_out = cn; *ct_out = ct; *f_out = f_loss; *alpha_out = alpha;

    return omega_r * (4.0 * f_loss * sp * sp - sigma * cn)
         - v_inf   * (sigma * ct + 4.0 * f_loss * sp * cp);
}}

/* ----------------------------------------------------------------- kernel */

{PW_KERNEL} void bemt_solve(
        {PW_GLOBAL} const double * {PW_RESTRICT} r,
        {PW_GLOBAL} const double * {PW_RESTRICT} chord,
        {PW_GLOBAL} const double * {PW_RESTRICT} twist,
        {PW_GLOBAL} const double * {PW_RESTRICT} sigma,
        {PW_GLOBAL} const double * {PW_RESTRICT} thickness,
        {PW_GLOBAL} const double * {PW_RESTRICT} re_ref,
        {PW_GLOBAL} const double * {PW_RESTRICT} m_crit0,
        {PW_GLOBAL} const double * {PW_RESTRICT} cl_max,
        {PW_GLOBAL} const double * {PW_RESTRICT} dr,
        {PW_GLOBAL} const double * {PW_RESTRICT} cl_tab,
        {PW_GLOBAL} const double * {PW_RESTRICT} cd_tab,
        {PW_GLOBAL} const double * {PW_RESTRICT} omega,
        {PW_GLOBAL} const double * {PW_RESTRICT} v_inf,
        {PW_GLOBAL} double * {PW_RESTRICT} thrust,
        {PW_GLOBAL} double * {PW_RESTRICT} torque,
        {PW_GLOBAL} double * {PW_RESTRICT} converged,
        {PW_GLOBAL} double * {PW_RESTRICT} phi_out,
        {PW_GLOBAL} double * {PW_RESTRICT} alpha_out,
        {PW_GLOBAL} double * {PW_RESTRICT} cl_out,
        {PW_GLOBAL} double * {PW_RESTRICT} cd_out,
        {PW_GLOBAL} double * {PW_RESTRICT} w_out,
        {PW_GLOBAL} double * {PW_RESTRICT} dt_out,
        {PW_GLOBAL} double * {PW_RESTRICT} dq_out,
        {PW_GLOBAL} double * {PW_RESTRICT} floss_out,
        int n_st, int n_alpha, int n_cases,
        double alpha0, double d_alpha, double rho, double mu, double a_sound,
        double r_tip, double r_hub, int n_blades, int flags, int n_bisect,
        double phi_lo, double phi_hi, int want_span)
{{
    const int cs  = {PW_GROUP_ID};
    if (cs >= n_cases) return;
    const int tid = {PW_LOCAL_ID};

    {PW_LOCAL} double s_t[PW_THREADS];
    {PW_LOCAL} double s_q[PW_THREADS];
    {PW_LOCAL} double s_c[PW_THREADS];

    double acc_t = 0.0, acc_q = 0.0, acc_c = 0.0;
    const double om = omega[cs];
    const double vv = v_inf[cs];

    for (int s = tid; s < n_st; s += PW_THREADS) {{
        const int base = s * n_alpha;
        const double omega_r = om * r[s];

        double w, cl, cd, cn, ct, fl, al;
        double lo = phi_lo, hi = phi_hi;

        double r_lo = pw_element(lo, twist[s], r[s], chord[s], sigma[s], thickness[s],
                                 re_ref[s], m_crit0[s], cl_max[s], base, n_alpha, alpha0, d_alpha,
                                 cl_tab, cd_tab, omega_r, vv, rho, mu, a_sound,
                                 r_tip, r_hub, n_blades, flags,
                                 &w, &cl, &cd, &cn, &ct, &fl, &al);
        double r_hi = pw_element(hi, twist[s], r[s], chord[s], sigma[s], thickness[s],
                                 re_ref[s], m_crit0[s], cl_max[s], base, n_alpha, alpha0, d_alpha,
                                 cl_tab, cd_tab, omega_r, vv, rho, mu, a_sound,
                                 r_tip, r_hub, n_blades, flags,
                                 &w, &cl, &cd, &cn, &ct, &fl, &al);

        const double bracketed = ((r_lo * r_hi) < 0.0) ? 1.0 : 0.0;

        /* Fixed trip count, no early exit: every thread in the warp / wavefront
         * retires together, which is the whole point of bisecting. */
        for (int it = 0; it < n_bisect; ++it) {{
            double mid = 0.5 * (lo + hi);
            double r_mid = pw_element(mid, twist[s], r[s], chord[s], sigma[s],
                                      thickness[s], re_ref[s], m_crit0[s], cl_max[s],
                                      base, n_alpha, alpha0, d_alpha, cl_tab, cd_tab,
                                      omega_r, vv, rho, mu, a_sound, r_tip, r_hub,
                                      n_blades, flags,
                                      &w, &cl, &cd, &cn, &ct, &fl, &al);
            if ((r_lo * r_mid) <= 0.0) {{ hi = mid; r_hi = r_mid; }}
            else                       {{ lo = mid; r_lo = r_mid; }}
        }}

        double phi = 0.5 * (lo + hi);
        if (bracketed == 0.0) {{
            double vq = (vv > 1.0e-6) ? vv : 1.0e-6;
            double oq = (omega_r > 1.0e-6) ? omega_r : 1.0e-6;
            phi = atan2(vq, oq);
        }}

        pw_element(phi, twist[s], r[s], chord[s], sigma[s], thickness[s],
                   re_ref[s], m_crit0[s], cl_max[s], base, n_alpha, alpha0, d_alpha,
                   cl_tab, cd_tab, omega_r, vv, rho, mu, a_sound,
                   r_tip, r_hub, n_blades, flags,
                   &w, &cl, &cd, &cn, &ct, &fl, &al);

        const double k  = 0.5 * rho * (double)n_blades * chord[s] * w * w;
        const double dt = k * cn;
        const double dq = k * ct * r[s];
        acc_t += dt * dr[s];
        acc_q += dq * dr[s];
        acc_c += bracketed;

        if (want_span) {{
            const int o = cs * n_st + s;
            phi_out[o]   = phi;
            alpha_out[o] = al;
            cl_out[o]    = cl;
            cd_out[o]    = cd;
            w_out[o]     = w;
            dt_out[o]    = dt;
            dq_out[o]    = dq;
            floss_out[o] = fl;
        }}
    }}

    s_t[tid] = acc_t;
    s_q[tid] = acc_q;
    s_c[tid] = acc_c;
    {PW_BARRIER};

    for (int stride = PW_THREADS / 2; stride > 0; stride >>= 1) {{
        if (tid < stride) {{
            s_t[tid] += s_t[tid + stride];
            s_q[tid] += s_q[tid + stride];
            s_c[tid] += s_c[tid + stride];
        }}
        {PW_BARRIER};
    }}

    if (tid == 0) {{
        thrust[cs]    = s_t[0];
        torque[cs]    = s_q[0];
        converged[cs] = s_c[0] / (double)n_st;
    }}
}}
"""


def kernel_source(target: str = "cuda", threads: int = PW_THREADS) -> str:
    """Render the kernel as real CUDA C or real OpenCL C.

    The result is a complete, compilable translation unit -- save it to a
    ``.cu`` or ``.cl`` file and hand it to nvcc or clBuildProgram.
    """
    key = target.lower()
    if key not in _TARGETS:
        raise KeyError(f"unknown target {target!r}; have {sorted(_TARGETS)}")
    return _TEMPLATE.format(THREADS=int(threads), **_TARGETS[key])


#: Rendered CUDA C, kept as a module constant for convenience.
CUDA_SOURCE = kernel_source("cuda")

#: Rendered OpenCL C.
OPENCL_SOURCE = kernel_source("opencl")

_kernel_cache: dict[str, Any] = {}


def get_raw_kernel() -> tuple[Any, int]:
    """Compile (once) and return the CuPy ``(kernel, threads_per_block)``.

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


__all__ = ["CUDA_SOURCE", "OPENCL_SOURCE", "PW_THREADS", "get_raw_kernel",
           "kernel_source"]
