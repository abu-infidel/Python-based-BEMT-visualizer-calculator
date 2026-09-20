"""PyTorch implementation of the batched solve.

Numba and CuPy both need a CUDA toolchain.  PyTorch ships its own, so on a
runtime where those are awkward -- or on Apple Metal, where neither works --
this path still puts the sweep on the GPU.  It is a direct transcription of the
NumPy reference with ``torch`` in place of ``numpy``; the bisection is
identical, which is why a vectorised framework can run it at all.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..atmosphere import AirState
from ..bemt.core import (FLAG_HUB_LOSS, FLAG_MACH, FLAG_REYNOLDS, FLAG_TIP_LOSS,
                         PHI_HI, PHI_LO, SolverOptions)
from ..geometry import BladeStations
from ..units import rpm_to_rad_s


def pick_device(prefer: str = "auto"):
    import torch
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prandtl(torch, phi, r, r_tip, r_hub, n_blades, flags):
    s = torch.clamp(torch.abs(torch.sin(phi)), min=1e-7)
    f = torch.ones_like(phi)
    if flags & FLAG_TIP_LOSS:
        ft = torch.clamp(0.5 * n_blades * (r_tip - r) / torch.clamp(r * s, min=1e-12), min=0.0)
        f = f * (2.0 / math.pi) * torch.acos(torch.clamp(torch.exp(-ft), 0.0, 1.0))
    if (flags & FLAG_HUB_LOSS) and r_hub > 0.0:
        fh = torch.clamp(0.5 * n_blades * (r - r_hub) / max(r_hub, 1e-12) / s, min=0.0)
        f = f * (2.0 / math.pi) * torch.acos(torch.clamp(torch.exp(-fh), 0.0, 1.0))
    return torch.clamp(f, 1e-4, 1.0)


def _interp(torch, alpha, alpha0, d_alpha, cl_tab, cd_tab):
    """Uniform-grid lookup; ``cl_tab`` is (n_stations, n_alpha)."""
    n_alpha = cl_tab.shape[1]
    a = torch.remainder(alpha + math.pi, 2.0 * math.pi) - math.pi
    t = (a - alpha0) / d_alpha
    i0 = torch.clamp(torch.floor(t).long(), 0, n_alpha - 2)
    w = torch.clamp(t - i0.to(t.dtype), 0.0, 1.0)
    st = torch.arange(cl_tab.shape[0], device=alpha.device).view(1, -1).expand_as(i0)
    cl = cl_tab[st, i0] * (1.0 - w) + cl_tab[st, i0 + 1] * w
    cd = cd_tab[st, i0] * (1.0 - w) + cd_tab[st, i0 + 1] * w
    return cl, cd


def _corrections(torch, cl, cd, w, chord, rho, mu, a_sound, thickness, re_ref,
                 m_crit0, cl_max, flags):
    if flags & FLAG_REYNOLDS:
        re = torch.clamp(rho * w * chord / mu, min=1e3)
        ratio = re_ref / re
        scale = torch.clamp(ratio ** 0.2 * (1.0 + 0.45 * torch.clamp(2e4 / re - 1.0, min=0.0)),
                            0.5, 4.0)
        cd = cd * scale
        cl = cl * torch.clamp(1.0 - 0.14 * torch.log10(ratio), 0.45, 1.12)
    if flags & FLAG_MACH:
        m = torch.clamp(w / a_sound, 0.0, 0.92)
        cl = cl / torch.sqrt(torch.clamp(1.0 - m * m, min=1e-3))
        # Compressibility raises the lift slope but lowers the stall ceiling.
        ceiling = cl_max * torch.clamp(
            1.0 - 0.9 * torch.clamp(m - 0.35, min=0.0) ** 1.5, min=0.35)
        cl = torch.clamp(cl, -ceiling, ceiling)
        m_dd = m_crit0 - 0.1 * torch.abs(cl) - thickness
        cd = cd + 20.0 * torch.clamp(m - m_dd, min=0.0) ** 4
    return cl, torch.clamp(cd, min=1e-5)


def _state(torch, phi, g):
    sp, cp = torch.sin(phi), torch.cos(phi)
    alpha = g["twist"] - phi
    f = _prandtl(torch, phi, g["r"], g["r_tip"], g["r_hub"], g["n_blades"], g["flags"])
    cl0, cd0 = _interp(torch, alpha, g["alpha0"], g["d_alpha"], g["cl_tab"], g["cd_tab"])
    ct0 = cl0 * sp + cd0 * cp
    den = g["sigma"] * ct0 + 4.0 * f * sp * cp
    den = torch.where(den.abs() < 1e-9, torch.full_like(den, 1e-9), den)
    w = torch.clamp(torch.abs(4.0 * f * g["omega_r"] * sp / den),
                    min=1e-3).maximum(g["v_inf"].abs())
    cl, cd = _corrections(torch, cl0, cd0, w, g["chord"], g["rho"], g["mu"],
                          g["a_sound"], g["thickness"], g["re_ref"], g["m_crit0"],
                          g["cl_max"], g["flags"])
    cn = cl * cp - cd * sp
    ct = cl * sp + cd * cp
    res = (g["omega_r"] * (4.0 * f * sp * sp - g["sigma"] * cn)
           - g["v_inf"] * (g["sigma"] * ct + 4.0 * f * sp * cp))
    return res, w, cl, cd, cn, ct, f, alpha


def solve_batch_torch(stations: BladeStations, rpm: np.ndarray, v_inf: np.ndarray,
                      air: AirState, opts: SolverOptions, tables, want_spanwise: bool,
                      device: str = "auto", dtype: str = "float64") -> dict[str, np.ndarray]:
    import torch

    dev = pick_device(device)
    td = torch.float32 if (dtype == "float32" or dev.type == "mps") else torch.float64

    alpha_grid, cl_tab, cd_tab = tables
    thickness, re_ref, m_crit0, cl_max = stations.section_params()
    geom = stations.geometry

    def row(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=td, device=dev).view(1, -1)

    def col(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=td, device=dev).view(-1, 1)

    g: dict[str, Any] = {
        "r": row(stations.r), "chord": row(stations.chord), "twist": row(stations.twist),
        "sigma": row(stations.solidity), "thickness": row(thickness),
        "re_ref": row(re_ref), "m_crit0": row(m_crit0), "cl_max": row(cl_max),
        "cl_tab": torch.as_tensor(np.ascontiguousarray(cl_tab), dtype=td, device=dev),
        "cd_tab": torch.as_tensor(np.ascontiguousarray(cd_tab), dtype=td, device=dev),
        "alpha0": float(alpha_grid[0]), "d_alpha": float(alpha_grid[1] - alpha_grid[0]),
        "omega_r": col(rpm_to_rad_s(rpm)) * row(stations.r), "v_inf": col(v_inf),
        "rho": air.density, "mu": air.viscosity, "a_sound": air.sound_speed,
        "r_tip": geom.radius, "r_hub": geom.hub_radius, "n_blades": geom.n_blades,
        "flags": opts.flags(),
    }

    shape = (rpm.size, stations.n)
    lo = torch.full(shape, PHI_LO, dtype=td, device=dev)
    hi = torch.full(shape, PHI_HI, dtype=td, device=dev)
    r_lo = _state(torch, lo, g)[0]
    r_hi = _state(torch, hi, g)[0]
    bracketed = (r_lo * r_hi) < 0.0

    for _ in range(opts.n_bisect):
        mid = 0.5 * (lo + hi)
        r_mid = _state(torch, mid, g)[0]
        left = (r_lo * r_mid) <= 0.0
        hi = torch.where(left, mid, hi)
        lo = torch.where(left, lo, mid)
        r_hi = torch.where(left, r_mid, r_hi)
        r_lo = torch.where(left, r_lo, r_mid)

    phi = 0.5 * (lo + hi)
    phi = torch.where(bracketed, phi,
                      torch.atan2(torch.clamp(g["v_inf"], min=1e-6).expand_as(phi),
                                  torch.clamp(g["omega_r"], min=1e-6).expand_as(phi)))

    _, w, cl, cd, cn, ct, f, alpha = _state(torch, phi, g)
    dr = row(stations.dr)
    k = 0.5 * g["rho"] * g["n_blades"] * g["chord"] * w * w
    dt_dr = k * cn
    dq_dr = k * ct * g["r"]

    out = {
        "thrust": (dt_dr * dr).sum(dim=1).double().cpu().numpy(),
        "torque": (dq_dr * dr).sum(dim=1).double().cpu().numpy(),
        "converged": bracketed.to(td).mean(dim=1).double().cpu().numpy(),
    }
    if want_spanwise:
        for name, tensor in (("phi", phi), ("alpha", alpha), ("cl", cl), ("cd", cd),
                             ("w", w), ("dt_dr", dt_dr), ("dq_dr", dq_dr),
                             ("loss_factor", f)):
            out[name] = tensor.double().cpu().numpy()
    return out


__all__ = ["solve_batch_torch", "pick_device"]
