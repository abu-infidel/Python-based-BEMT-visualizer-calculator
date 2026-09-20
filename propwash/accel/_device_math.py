"""Scalar element physics, written once and compiled for every target.

These are plain Python functions containing nothing but ``math`` calls and
scalar arithmetic.  :mod:`propwash.accel.kernels` compiles the *same source
objects* with ``numba.njit`` for the CPU and ``numba.cuda.jit(device=True)`` for
the GPU, so the two paths cannot drift apart -- a class of bug that is
otherwise extremely easy to introduce and extremely hard to find.

Array arguments are flat: polar tables are indexed ``station * n_alpha + i``
rather than 2-D, because that is what both Numba targets handle best and it
maps directly onto the raw CUDA C kernel in :mod:`propwash.accel.kernels_raw`.
"""

from __future__ import annotations

import math

PI = 3.141592653589793
TWO_PI = 6.283185307179586
HALF_PI = 1.5707963267948966

FLAG_TIP_LOSS = 1
FLAG_HUB_LOSS = 2
FLAG_REYNOLDS = 4
FLAG_MACH = 8


def wrap_pi(a):
    """Wrap an angle into (-pi, pi] without a modulo instruction."""
    x = a
    while x > PI:
        x -= TWO_PI
    while x < -PI:
        x += TWO_PI
    return x


def interp_table(alpha, alpha0, d_alpha, n_alpha, table, base):
    """Linear interpolation on a uniform grid; ``base = station * n_alpha``."""
    a = wrap_pi(alpha)
    t = (a - alpha0) / d_alpha
    i = int(math.floor(t))
    if i < 0:
        i = 0
    elif i > n_alpha - 2:
        i = n_alpha - 2
    w = t - i
    if w < 0.0:
        w = 0.0
    elif w > 1.0:
        w = 1.0
    return table[base + i] * (1.0 - w) + table[base + i + 1] * w


def prandtl_factor(phi, r, r_tip, r_hub, n_blades, flags):
    """Combined Prandtl tip and hub loss."""
    s = math.fabs(math.sin(phi))
    if s < 1.0e-7:
        s = 1.0e-7
    f = 1.0

    if flags & FLAG_TIP_LOSS:
        ft = 0.5 * n_blades * (r_tip - r) / (r * s)
        if ft < 0.0:
            ft = 0.0
        e = math.exp(-ft)
        if e > 1.0:
            e = 1.0
        f *= (2.0 / PI) * math.acos(e)

    if (flags & FLAG_HUB_LOSS) and r_hub > 0.0:
        fh = 0.5 * n_blades * (r - r_hub) / (r_hub * s)
        if fh < 0.0:
            fh = 0.0
        e = math.exp(-fh)
        if e > 1.0:
            e = 1.0
        f *= (2.0 / PI) * math.acos(e)

    if f < 1.0e-4:
        f = 1.0e-4
    elif f > 1.0:
        f = 1.0
    return f


def mach_lift_ceiling(mach, cl_max):
    """Maximum attainable |Cl| at a given Mach number (see the NumPy twin)."""
    excess = mach - 0.35
    if excess < 0.0:
        excess = 0.0
    factor = 1.0 - 0.9 * math.pow(excess, 1.5)
    if factor < 0.35:
        factor = 0.35
    return cl_max * factor


def apply_corrections(cl, cd, w, chord, rho, mu, a_sound, thickness, re_ref,
                      m_crit0, cl_max, flags):
    """Reynolds and compressibility scaling, identical to the NumPy twin."""
    if flags & FLAG_REYNOLDS:
        re = rho * w * chord / mu
        if re < 1.0e3:
            re = 1.0e3
        ratio = re_ref / re
        scale = ratio ** 0.2
        low = 2.0e4 / re - 1.0
        if low > 0.0:
            scale *= 1.0 + 0.45 * low
        if scale < 0.5:
            scale = 0.5
        elif scale > 4.0:
            scale = 4.0
        cd *= scale
        cls = 1.0 - 0.14 * math.log10(ratio)
        if cls < 0.45:
            cls = 0.45
        elif cls > 1.12:
            cls = 1.12
        cl *= cls

    if flags & FLAG_MACH:
        m = w / a_sound
        if m < 0.0:
            m = 0.0
        elif m > 0.92:
            m = 0.92
        denom = 1.0 - m * m
        if denom < 1.0e-3:
            denom = 1.0e-3
        cl /= math.sqrt(denom)
        # Compressibility raises the lift slope but lowers the stall ceiling.
        ceiling = mach_lift_ceiling(m, cl_max)
        if cl > ceiling:
            cl = ceiling
        elif cl < -ceiling:
            cl = -ceiling
        m_dd = m_crit0 - 0.1 * math.fabs(cl) - thickness
        dm = m - m_dd
        if dm > 0.0:
            cd += 20.0 * dm * dm * dm * dm

    if cd < 1.0e-5:
        cd = 1.0e-5
    return cl, cd


def element_state(phi, twist, r, chord, sigma, thickness, re_ref, m_crit0, cl_max,
                  base, n_alpha, alpha0, d_alpha, cl_tab, cd_tab,
                  omega_r, v_inf, rho, mu, a_sound, r_tip, r_hub, n_blades, flags):
    """Full element state at a trial inflow angle.

    Returns ``(residual, w, cl, cd, cn, ct, f_loss, alpha)``.  The residual is

        R = omega r (4 F sin^2 phi - sigma Cn) - V (sigma Ct + 4 F sin phi cos phi)

    which is zero exactly when the blade-element and momentum descriptions of
    the same element agree.
    """
    sp = math.sin(phi)
    cp = math.cos(phi)
    alpha = twist - phi

    f_loss = prandtl_factor(phi, r, r_tip, r_hub, n_blades, flags)

    cl0 = interp_table(alpha, alpha0, d_alpha, n_alpha, cl_tab, base)
    cd0 = interp_table(alpha, alpha0, d_alpha, n_alpha, cd_tab, base)

    # Resultant velocity from the torque branch: its denominator is strictly
    # positive on (0, pi/2), and unlike the thrust branch it stays finite at
    # V = 0, which is the static-thrust case everybody looks at first.
    ct0 = cl0 * sp + cd0 * cp
    denom = sigma * ct0 + 4.0 * f_loss * sp * cp
    if math.fabs(denom) < 1.0e-9:
        denom = 1.0e-9 if denom >= 0.0 else -1.0e-9
    w = 4.0 * f_loss * omega_r * sp / denom
    w = math.fabs(w)
    vmin = math.fabs(v_inf)
    if vmin < 1.0e-3:
        vmin = 1.0e-3
    if w < vmin:
        w = vmin

    cl, cd = apply_corrections(cl0, cd0, w, chord, rho, mu, a_sound,
                               thickness, re_ref, m_crit0, cl_max, flags)

    cn = cl * cp - cd * sp
    ct = cl * sp + cd * cp

    residual = (omega_r * (4.0 * f_loss * sp * sp - sigma * cn)
                - v_inf * (sigma * ct + 4.0 * f_loss * sp * cp))
    return residual, w, cl, cd, cn, ct, f_loss, alpha


def solve_phi(twist, r, chord, sigma, thickness, re_ref, m_crit0, cl_max,
              base, n_alpha, alpha0, d_alpha, cl_tab, cd_tab,
              omega_r, v_inf, rho, mu, a_sound, r_tip, r_hub, n_blades,
              flags, n_bisect, phi_lo, phi_hi):
    """Bracketed bisection for the inflow angle.

    Fixed trip count, no early exit, no data-dependent branch: every thread in a
    warp executes the identical instruction stream, which is what makes this
    worth running on a GPU at all.  Returns ``(phi, bracketed)``.
    """
    lo = phi_lo
    hi = phi_hi

    r_lo = element_state(lo, twist, r, chord, sigma, thickness, re_ref, m_crit0,
                         cl_max, base, n_alpha, alpha0, d_alpha, cl_tab, cd_tab,
                         omega_r, v_inf, rho, mu, a_sound, r_tip, r_hub,
                         n_blades, flags)[0]
    r_hi = element_state(hi, twist, r, chord, sigma, thickness, re_ref, m_crit0,
                         cl_max, base, n_alpha, alpha0, d_alpha, cl_tab, cd_tab,
                         omega_r, v_inf, rho, mu, a_sound, r_tip, r_hub,
                         n_blades, flags)[0]

    bracketed = 1 if (r_lo * r_hi) < 0.0 else 0

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        r_mid = element_state(mid, twist, r, chord, sigma, thickness, re_ref,
                              m_crit0, cl_max, base, n_alpha, alpha0, d_alpha,
                              cl_tab, cd_tab, omega_r, v_inf, rho, mu, a_sound,
                              r_tip, r_hub, n_blades, flags)[0]
        if (r_lo * r_mid) <= 0.0:
            hi = mid
            r_hi = r_mid
        else:
            lo = mid
            r_lo = r_mid

    phi = 0.5 * (lo + hi)
    if bracketed == 0:
        vv = v_inf if v_inf > 1.0e-6 else 1.0e-6
        oo = omega_r if omega_r > 1.0e-6 else 1.0e-6
        phi = math.atan2(vv, oo)
    return phi, bracketed


__all__ = ["wrap_pi", "interp_table", "prandtl_factor", "apply_corrections",
           "mach_lift_ceiling",
           "element_state", "solve_phi", "PI", "TWO_PI", "HALF_PI",
           "FLAG_TIP_LOSS", "FLAG_HUB_LOSS", "FLAG_REYNOLDS", "FLAG_MACH"]
