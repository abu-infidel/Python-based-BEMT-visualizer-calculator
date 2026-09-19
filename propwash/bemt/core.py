r"""The BEMT kernel: one residual, one bracketed root, no iteration limits.

Formulation
-----------
For a blade element at radius :math:`r` with solidity
:math:`\sigma = Bc/2\pi r`, equating the blade-element and momentum expressions
for thrust and for torque gives two expressions for the resultant velocity
:math:`W`:

.. math::
    W_T = \frac{4 F V \sin\phi}{4F\sin^2\phi - \sigma C_n}, \qquad
    W_Q = \frac{4 F \Omega r \sin\phi}{\sigma C_t + 4F\sin\phi\cos\phi}

Setting them equal and clearing the common factor :math:`4F\sin\phi` leaves a
single residual in the inflow angle :math:`\phi`:

.. math::
    \mathcal{R}(\phi) = \Omega r\,(4F\sin^2\phi - \sigma C_n)
                      - V\,(\sigma C_t + 4F\sin\phi\cos\phi)

This form is the whole reason the solver is fast and never diverges:

* **No division.**  The usual ``a = k/(1-k)`` induction-factor formulation blows
  up when ``k -> 1`` and is undefined at ``V = 0``.  This one is a smooth,
  bounded function of :math:`\phi` everywhere, and at :math:`V = 0` it collapses
  to the correct static-thrust relation :math:`4F\sin^2\phi = \sigma C_n`.
* **Guaranteed bracket.**  In the propeller state
  :math:`\mathcal{R}(0^+) < 0` (the ``-\Omega r \sigma C_n`` term dominates) and
  :math:`\mathcal{R}(\pi/2^-) > 0` (the ``4F\Omega r`` term dominates), so a
  root always exists in :math:`(0, \pi/2)`.
* **Fixed cost.**  Bisection to machine precision takes a known ~60 halvings.
  Every element takes the *same* number of steps with *no data-dependent
  branching*, which is precisely what a GPU warp wants.  A Newton or
  fixed-point scheme would be faster on a CPU and far worse on a GPU.

The trade is that only the normal thrusting branch is bracketed.  Windmilling
and propeller-brake states have their root outside :math:`(0, \pi/2)` and are
reported as non-converged rather than silently returning nonsense.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..airfoil import DEG
from ..atmosphere import SEA_LEVEL, AirState
from ..geometry import BladeStations
from ..units import rad_s_to_rpm, rpm_to_rad_s

# Correction bit-flags.  Packed into one int so the CUDA kernel takes a single
# scalar argument instead of half a dozen booleans.
FLAG_TIP_LOSS = 1 << 0
FLAG_HUB_LOSS = 1 << 1
FLAG_REYNOLDS = 1 << 2
FLAG_MACH = 1 << 3
FLAG_STALL_DELAY = 1 << 4
FLAG_SWIRL = 1 << 5

PHI_LO = 1.0e-6
PHI_HI = math.pi / 2.0 - 1.0e-6


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SolverOptions:
    """Numerical knobs and which physics corrections are switched on."""

    n_elements: int = 60
    spacing: str = "cosine"
    n_bisect: int = 60           # 60 halvings of (0, pi/2) == float64 exact
    n_alpha_table: int = 721     # polar table resolution (0.5 deg)

    tip_loss: bool = True
    hub_loss: bool = True
    reynolds_correction: bool = True
    mach_correction: bool = True
    stall_delay: bool = False
    swirl_in_thrust: bool = True  # include the swirl-recovery term in power

    def flags(self) -> int:
        f = 0
        f |= FLAG_TIP_LOSS if self.tip_loss else 0
        f |= FLAG_HUB_LOSS if self.hub_loss else 0
        f |= FLAG_REYNOLDS if self.reynolds_correction else 0
        f |= FLAG_MACH if self.mach_correction else 0
        f |= FLAG_STALL_DELAY if self.stall_delay else 0
        f |= FLAG_SWIRL if self.swirl_in_thrust else 0
        return f


@dataclass(slots=True)
class OperatingPoint:
    """Where the propeller is being asked to work."""

    rpm: float = 6000.0
    v_inf: float = 0.0            # m/s, axial freestream
    air: AirState = field(default_factory=lambda: SEA_LEVEL)

    @property
    def omega(self) -> float:
        return rpm_to_rad_s(self.rpm)

    @property
    def n_rev_s(self) -> float:
        return self.rpm / 60.0

    def with_rpm(self, rpm: float) -> "OperatingPoint":
        return OperatingPoint(rpm=rpm, v_inf=self.v_inf, air=self.air)

    def with_speed(self, v: float) -> "OperatingPoint":
        return OperatingPoint(rpm=self.rpm, v_inf=v, air=self.air)


# ---------------------------------------------------------------------------
# Physics helpers (vectorised; mirrored exactly in the CUDA kernels)
# ---------------------------------------------------------------------------

def prandtl_loss(phi: np.ndarray, r: np.ndarray, r_tip: float, r_hub: float,
                 n_blades: int, flags: int) -> np.ndarray:
    r"""Combined Prandtl tip and hub loss factor.

    Momentum theory assumes an actuator disk with infinitely many blades.  A
    real blade sheds a vortex at each end, so the circulation -- and with it the
    induced velocity -- collapses near the tip and the hub.  Prandtl's factor

    .. math:: F = \frac{2}{\pi}\arccos\left(e^{-f}\right),\quad
              f = \frac{B}{2}\frac{r_{tip}-r}{r\,|\sin\phi|}

    is the standard cheap correction.  Without it a propeller's predicted thrust
    is optimistic by roughly 5-15%, concentrated entirely at the tip.
    """
    s = np.maximum(np.abs(np.sin(phi)), 1.0e-7)
    f = np.ones_like(np.asarray(phi, dtype=float))

    if flags & FLAG_TIP_LOSS:
        ft = 0.5 * n_blades * (r_tip - r) / np.maximum(r * s, 1e-12)
        f = f * (2.0 / math.pi) * np.arccos(np.clip(np.exp(-np.maximum(ft, 0.0)), 0.0, 1.0))

    if (flags & FLAG_HUB_LOSS) and r_hub > 0.0:
        fh = 0.5 * n_blades * (r - r_hub) / np.maximum(r_hub * s, 1e-12)
        f = f * (2.0 / math.pi) * np.arccos(np.clip(np.exp(-np.maximum(fh, 0.0)), 0.0, 1.0))

    return np.clip(f, 1.0e-4, 1.0)


def _interp_tables(alpha: np.ndarray, alpha0: float, d_alpha: float,
                   cl_tab: np.ndarray, cd_tab: np.ndarray,
                   station_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-grid table lookup -- the host twin of the device interpolation."""
    na = cl_tab.shape[1]
    a = (np.asarray(alpha, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi
    t = (a - alpha0) / d_alpha
    i0 = np.clip(np.floor(t).astype(np.int64), 0, na - 2)
    w = t - i0
    cl = cl_tab[station_index, i0] * (1.0 - w) + cl_tab[station_index, i0 + 1] * w
    cd = cd_tab[station_index, i0] * (1.0 - w) + cd_tab[station_index, i0 + 1] * w
    return cl, cd


def _corrected_coeffs(cl: np.ndarray, cd: np.ndarray, w: np.ndarray, chord: np.ndarray,
                      rho: float, mu: float, a_sound: float,
                      thickness: np.ndarray, re_ref: np.ndarray, m_crit0: np.ndarray,
                      flags: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply Reynolds and compressibility scaling to raw table values."""
    if flags & FLAG_REYNOLDS:
        re = np.maximum(rho * w * chord / mu, 1.0e3)
        re_scale = (re_ref / re) ** 0.2 * (1.0 + 0.45 * np.maximum(2.0e4 / re - 1.0, 0.0))
        cd = cd * np.clip(re_scale, 0.5, 4.0)
        cl = cl * np.clip(1.0 - 0.14 * np.log10(np.maximum(re_ref / re, 1e-12)), 0.45, 1.12)

    if flags & FLAG_MACH:
        m = np.clip(w / a_sound, 0.0, 0.92)
        cl = cl / np.sqrt(np.maximum(1.0 - m * m, 1.0e-3))
        m_dd = m_crit0 - 0.1 * np.abs(cl) - thickness
        cd = cd + 20.0 * np.maximum(m - m_dd, 0.0) ** 4

    return cl, np.maximum(cd, 1.0e-5)


def residual_phi(phi: np.ndarray, ctx: "_ElementContext") -> np.ndarray:
    r"""Evaluate :math:`\mathcal{R}(\phi)` for every element at once."""
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)

    alpha = ctx.twist - phi

    f_loss = prandtl_loss(phi, ctx.r, ctx.r_tip, ctx.r_hub, ctx.n_blades, ctx.flags)

    # W from the torque branch: well conditioned for every propeller-like case
    # (its denominator is strictly positive for phi in (0, pi/2)), and unlike the
    # thrust branch it does not vanish when V = 0.
    cl0, cd0 = _interp_tables(alpha, ctx.alpha0, ctx.d_alpha, ctx.cl_tab, ctx.cd_tab, ctx.idx)
    ct0 = cl0 * sin_p + cd0 * cos_p
    denom = ctx.sigma * ct0 + 4.0 * f_loss * sin_p * cos_p
    w_guess = 4.0 * f_loss * ctx.omega_r * sin_p / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    w_guess = np.maximum(np.abs(w_guess), np.maximum(np.abs(ctx.v_inf), 1e-3))

    cl, cd = _corrected_coeffs(cl0, cd0, w_guess, ctx.chord, ctx.rho, ctx.mu,
                               ctx.a_sound, ctx.thickness, ctx.re_ref, ctx.m_crit0,
                               ctx.flags)

    cn = cl * cos_p - cd * sin_p
    ct = cl * sin_p + cd * cos_p

    return (ctx.omega_r * (4.0 * f_loss * sin_p * sin_p - ctx.sigma * cn)
            - ctx.v_inf * (ctx.sigma * ct + 4.0 * f_loss * sin_p * cos_p))


@dataclass(slots=True)
class _ElementContext:
    """Everything the residual needs, pre-broadcast to the element shape."""

    r: np.ndarray
    chord: np.ndarray
    twist: np.ndarray
    sigma: np.ndarray
    thickness: np.ndarray
    re_ref: np.ndarray
    m_crit0: np.ndarray
    idx: np.ndarray
    alpha0: float
    d_alpha: float
    cl_tab: np.ndarray
    cd_tab: np.ndarray
    omega_r: np.ndarray
    v_inf: float
    rho: float
    mu: float
    a_sound: float
    r_tip: float
    r_hub: float
    n_blades: int
    flags: int


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BEMTResult:
    """Integrated performance plus the full spanwise state that produced it."""

    # scalars
    thrust: float = 0.0            # N
    torque: float = 0.0            # N m
    power: float = 0.0             # W
    efficiency: float = 0.0        # propulsive efficiency eta = T V / P
    figure_of_merit: float = 0.0   # hover FoM (static only)
    ct: float = 0.0                # T / (rho n^2 D^4)
    cq: float = 0.0                # Q / (rho n^2 D^5)
    cp: float = 0.0                # P / (rho n^3 D^5)
    j: float = 0.0                 # advance ratio
    rpm: float = 0.0
    v_inf: float = 0.0
    tip_mach: float = 0.0
    disk_loading: float = 0.0      # N/m^2
    converged_fraction: float = 1.0

    # spanwise arrays
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    alpha: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mach: np.ndarray = field(default_factory=lambda: np.zeros(0))
    reynolds: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dt_dr: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dq_dr: np.ndarray = field(default_factory=lambda: np.zeros(0))
    a_axial: np.ndarray = field(default_factory=lambda: np.zeros(0))
    a_swirl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v_axial_induced: np.ndarray = field(default_factory=lambda: np.zeros(0))
    v_swirl_induced: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loss_factor: np.ndarray = field(default_factory=lambda: np.zeros(0))
    converged: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def thrust_kgf(self) -> float:
        return self.thrust / 9.80665

    @property
    def stall_fraction(self) -> float:
        """Fraction of the blade past its peak-lift angle -- the stall warning."""
        if self.alpha.size == 0:
            return 0.0
        return float(np.mean(self.alpha > 14.0 * DEG))

    def summary(self) -> dict[str, float]:
        return {
            "rpm": self.rpm, "v_inf": self.v_inf, "J": self.j,
            "thrust_N": self.thrust, "thrust_kgf": self.thrust_kgf,
            "torque_Nm": self.torque, "power_W": self.power,
            "efficiency": self.efficiency, "figure_of_merit": self.figure_of_merit,
            "CT": self.ct, "CQ": self.cq, "CP": self.cp,
            "tip_mach": self.tip_mach, "disk_loading_Nm2": self.disk_loading,
            "stall_fraction": self.stall_fraction,
            "converged_fraction": self.converged_fraction,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<BEMTResult {self.rpm:.0f} rpm, V={self.v_inf:.1f} m/s, "
                f"T={self.thrust:.2f} N, P={self.power:.0f} W, eta={self.efficiency:.3f}>")


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

def solve_stations(stations: BladeStations, op: OperatingPoint,
                   opts: SolverOptions | None = None,
                   tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> BEMTResult:
    """Solve one operating point on a discretised blade (reference NumPy path)."""
    opts = opts or SolverOptions()
    geom = stations.geometry
    flags = opts.flags()

    alpha_grid, cl_tab, cd_tab = tables if tables is not None else stations.polar_tables(opts.n_alpha_table)
    thickness, re_ref, m_crit0 = stations.section_params()

    ctx = _ElementContext(
        r=stations.r, chord=stations.chord, twist=stations.twist,
        sigma=stations.solidity, thickness=thickness, re_ref=re_ref, m_crit0=m_crit0,
        idx=np.arange(stations.n), alpha0=float(alpha_grid[0]),
        d_alpha=float(alpha_grid[1] - alpha_grid[0]),
        cl_tab=cl_tab, cd_tab=cd_tab,
        omega_r=op.omega * stations.r, v_inf=float(op.v_inf),
        rho=op.air.density, mu=op.air.viscosity, a_sound=op.air.sound_speed,
        r_tip=geom.radius, r_hub=geom.hub_radius, n_blades=geom.n_blades, flags=flags,
    )

    lo = np.full(stations.n, PHI_LO)
    hi = np.full(stations.n, PHI_HI)
    r_lo = residual_phi(lo, ctx)
    r_hi = residual_phi(hi, ctx)

    bracketed = (r_lo * r_hi) < 0.0

    # Branch-free bisection: identical work for every element, so this maps
    # one-to-one onto a CUDA thread block with no warp divergence.
    for _ in range(opts.n_bisect):
        mid = 0.5 * (lo + hi)
        r_mid = residual_phi(mid, ctx)
        take_left = (r_lo * r_mid) <= 0.0
        hi = np.where(take_left, mid, hi)
        lo = np.where(take_left, lo, mid)
        r_hi = np.where(take_left, r_mid, r_hi)
        r_lo = np.where(take_left, r_lo, r_mid)

    phi = 0.5 * (lo + hi)
    # Elements with no sign change are outside the thrusting branch; fall back
    # to the zero-induction inflow angle and flag them rather than lie.
    phi = np.where(bracketed, phi,
                   np.arctan2(max(op.v_inf, 1e-6), np.maximum(ctx.omega_r, 1e-6)))

    return _assemble(phi, bracketed, ctx, stations, op, opts)


def _assemble(phi: np.ndarray, converged: np.ndarray, ctx: _ElementContext,
              stations: BladeStations, op: OperatingPoint,
              opts: SolverOptions) -> BEMTResult:
    """Turn a converged inflow angle field into every quantity of interest."""
    geom = stations.geometry
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    alpha = ctx.twist - phi
    f_loss = prandtl_loss(phi, ctx.r, ctx.r_tip, ctx.r_hub, ctx.n_blades, ctx.flags)

    cl0, cd0 = _interp_tables(alpha, ctx.alpha0, ctx.d_alpha, ctx.cl_tab, ctx.cd_tab, ctx.idx)
    ct0 = cl0 * sin_p + cd0 * cos_p
    denom = ctx.sigma * ct0 + 4.0 * f_loss * sin_p * cos_p
    w = 4.0 * f_loss * ctx.omega_r * sin_p / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    w = np.maximum(np.abs(w), np.maximum(np.abs(ctx.v_inf), 1e-3))

    cl, cd = _corrected_coeffs(cl0, cd0, w, ctx.chord, ctx.rho, ctx.mu, ctx.a_sound,
                               ctx.thickness, ctx.re_ref, ctx.m_crit0, ctx.flags)
    cn = cl * cos_p - cd * sin_p
    ct = cl * sin_p + cd * cos_p

    rho, b = ctx.rho, ctx.n_blades
    dt_dr = 0.5 * rho * b * ctx.chord * w * w * cn
    dq_dr = 0.5 * rho * b * ctx.chord * w * w * ct * ctx.r

    # Induced velocities follow directly from the converged triangle.
    u_axial = w * sin_p
    u_tang = w * cos_p
    v_a = u_axial - ctx.v_inf
    v_t = np.maximum(ctx.omega_r - u_tang, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        a_axial = np.where(np.abs(ctx.v_inf) > 1e-6,
                           v_a / np.maximum(ctx.v_inf, 1e-9), np.inf)
        a_swirl = np.where(ctx.omega_r > 1e-9, v_t / np.maximum(ctx.omega_r, 1e-9), 0.0)

    thrust = float(np.sum(dt_dr * stations.dr))
    torque = float(np.sum(dq_dr * stations.dr))
    power = torque * op.omega

    n = op.n_rev_s
    d = geom.diameter
    denom_ct = rho * n * n * d ** 4
    denom_cq = rho * n * n * d ** 5
    denom_cp = rho * n ** 3 * d ** 5

    j = ctx.v_inf / (n * d) if n > 1e-9 else 0.0
    # Propulsive efficiency only means anything while the propeller is both
    # producing thrust and absorbing power.  Past the zero-thrust advance ratio
    # T V / P goes negative and then, as power crosses zero too, diverges --
    # reporting that as "efficiency" is worse than reporting nothing.
    eta = (thrust * ctx.v_inf / power) if (power > 1e-9 and thrust > 0.0) else 0.0

    # Hover figure of merit: ideal induced power over actual shaft power.
    area = math.pi * geom.radius ** 2
    fom = 0.0
    if power > 1e-9 and thrust > 0.0:
        fom = float((thrust ** 1.5) / math.sqrt(2.0 * rho * area) / power)

    mach = w / ctx.a_sound
    reynolds = rho * w * ctx.chord / ctx.mu

    return BEMTResult(
        thrust=thrust, torque=torque, power=power, efficiency=eta,
        figure_of_merit=fom,
        ct=thrust / denom_ct if denom_ct > 1e-12 else 0.0,
        cq=torque / denom_cq if denom_cq > 1e-12 else 0.0,
        cp=power / denom_cp if denom_cp > 1e-12 else 0.0,
        j=j, rpm=op.rpm, v_inf=ctx.v_inf,
        tip_mach=float(math.hypot(op.omega * geom.radius, ctx.v_inf) / ctx.a_sound),
        disk_loading=thrust / area if area > 0 else 0.0,
        converged_fraction=float(np.mean(converged)),
        x=stations.x, r=ctx.r, phi=phi, alpha=alpha, cl=cl, cd=cd, w=w,
        mach=mach, reynolds=reynolds, dt_dr=dt_dr, dq_dr=dq_dr,
        a_axial=a_axial, a_swirl=a_swirl,
        v_axial_induced=v_a, v_swirl_induced=v_t,
        loss_factor=f_loss, converged=converged,
        meta={
            "n_elements": stations.n,
            "blade": geom.name,
            "n_blades": geom.n_blades,
            "diameter": d,
            "air": op.air.describe(),
            "flags": ctx.flags,
            "backend": "numpy",
        },
    )


__all__ = [
    "SolverOptions", "OperatingPoint", "BEMTResult", "solve_stations",
    "residual_phi", "prandtl_loss", "FLAG_TIP_LOSS", "FLAG_HUB_LOSS",
    "FLAG_REYNOLDS", "FLAG_MACH", "FLAG_STALL_DELAY", "FLAG_SWIRL",
    "PHI_LO", "PHI_HI", "rad_s_to_rpm",
]


# ---------------------------------------------------------------------------
# Batched solve -- many operating points at once
# ---------------------------------------------------------------------------

def _batch_context(stations: BladeStations, omega: np.ndarray, v_inf: np.ndarray,
                   air: AirState, flags: int,
                   tables: tuple[np.ndarray, np.ndarray, np.ndarray]) -> _ElementContext:
    """Broadcast a blade and a list of operating points to shape (cases, stations)."""
    alpha_grid, cl_tab, cd_tab = tables
    thickness, re_ref, m_crit0 = stations.section_params()
    geom = stations.geometry

    col = lambda a: np.asarray(a, dtype=float).reshape(-1, 1)   # noqa: E731
    row = lambda a: np.asarray(a, dtype=float).reshape(1, -1)   # noqa: E731

    return _ElementContext(
        r=row(stations.r), chord=row(stations.chord), twist=row(stations.twist),
        sigma=row(stations.solidity), thickness=row(thickness), re_ref=row(re_ref),
        m_crit0=row(m_crit0),
        idx=np.broadcast_to(np.arange(stations.n).reshape(1, -1),
                            (col(omega).shape[0], stations.n)),
        alpha0=float(alpha_grid[0]), d_alpha=float(alpha_grid[1] - alpha_grid[0]),
        cl_tab=cl_tab, cd_tab=cd_tab,
        omega_r=col(omega) * row(stations.r), v_inf=col(v_inf),
        rho=air.density, mu=air.viscosity, a_sound=air.sound_speed,
        r_tip=geom.radius, r_hub=geom.hub_radius, n_blades=geom.n_blades, flags=flags,
    )


def solve_batch(stations: BladeStations, rpm: np.ndarray, v_inf: np.ndarray,
                air: AirState | None = None, opts: SolverOptions | None = None,
                tables: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Solve ``n`` operating points in one vectorised pass.

    ``rpm`` and ``v_inf`` are broadcast against each other, so a 40x40 pitch/RPM
    map is one call.  Returns integrated scalars as ``(n,)`` arrays and the
    spanwise state as ``(n, n_stations)``.  This is the CPU reference that the
    CUDA kernels are checked against.
    """
    opts = opts or SolverOptions()
    air = air or SEA_LEVEL
    flags = opts.flags()
    tables = tables if tables is not None else stations.polar_tables(opts.n_alpha_table)

    rpm_b, v_b = np.broadcast_arrays(np.asarray(rpm, dtype=float),
                                     np.asarray(v_inf, dtype=float))
    shape = rpm_b.shape
    rpm_f = rpm_b.ravel()
    v_f = v_b.ravel()
    omega = rpm_to_rad_s(rpm_f)

    ctx = _batch_context(stations, omega, v_f, air, flags, tables)
    n_cases = rpm_f.size

    lo = np.full((n_cases, stations.n), PHI_LO)
    hi = np.full((n_cases, stations.n), PHI_HI)
    r_lo = residual_phi(lo, ctx)
    r_hi = residual_phi(hi, ctx)
    bracketed = (r_lo * r_hi) < 0.0

    for _ in range(opts.n_bisect):
        mid = 0.5 * (lo + hi)
        r_mid = residual_phi(mid, ctx)
        take_left = (r_lo * r_mid) <= 0.0
        hi = np.where(take_left, mid, hi)
        lo = np.where(take_left, lo, mid)
        r_hi = np.where(take_left, r_mid, r_hi)
        r_lo = np.where(take_left, r_lo, r_mid)

    phi = 0.5 * (lo + hi)
    fallback = np.arctan2(np.maximum(ctx.v_inf, 1e-6), np.maximum(ctx.omega_r, 1e-6))
    phi = np.where(bracketed, phi, fallback)

    out = integrate_batch(phi, bracketed, ctx, stations, rpm_f, v_f, air)
    for key, val in out.items():
        if val.ndim == 1:
            out[key] = val.reshape(shape)
        else:
            out[key] = val.reshape(shape + (stations.n,))
    return out


def integrate_batch(phi: np.ndarray, converged: np.ndarray, ctx: _ElementContext,
                    stations: BladeStations, rpm: np.ndarray, v_inf: np.ndarray,
                    air: AirState) -> dict[str, np.ndarray]:
    """Reduce a batch of converged inflow-angle fields to performance arrays."""
    geom = stations.geometry
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    alpha = ctx.twist - phi
    f_loss = prandtl_loss(phi, ctx.r, ctx.r_tip, ctx.r_hub, ctx.n_blades, ctx.flags)

    cl0, cd0 = _interp_tables(alpha, ctx.alpha0, ctx.d_alpha, ctx.cl_tab, ctx.cd_tab, ctx.idx)
    ct0 = cl0 * sin_p + cd0 * cos_p
    denom = ctx.sigma * ct0 + 4.0 * f_loss * sin_p * cos_p
    w = 4.0 * f_loss * ctx.omega_r * sin_p / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    w = np.maximum(np.abs(w), np.maximum(np.abs(ctx.v_inf), 1e-3))

    cl, cd = _corrected_coeffs(cl0, cd0, w, ctx.chord, ctx.rho, ctx.mu, ctx.a_sound,
                               ctx.thickness, ctx.re_ref, ctx.m_crit0, ctx.flags)
    cn = cl * cos_p - cd * sin_p
    ct = cl * sin_p + cd * cos_p

    k = 0.5 * ctx.rho * ctx.n_blades * ctx.chord * w * w
    dt_dr = k * cn
    dq_dr = k * ct * ctx.r

    thrust = np.sum(dt_dr * stations.dr, axis=-1)
    torque = np.sum(dq_dr * stations.dr, axis=-1)
    omega = rpm_to_rad_s(rpm)
    power = torque * omega

    n_rev = np.maximum(rpm / 60.0, 1e-9)
    d = geom.diameter
    area = math.pi * geom.radius ** 2
    rho = ctx.rho

    with np.errstate(divide="ignore", invalid="ignore"):
        useful = (power > 1e-9) & (thrust > 0.0)
        eta = np.where(useful, thrust * v_inf / np.maximum(power, 1e-12), 0.0)
        fom = np.where(useful, np.maximum(thrust, 0.0) ** 1.5
                       / math.sqrt(2.0 * rho * area) / np.maximum(power, 1e-12), 0.0)

    return {
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
        "converged_fraction": np.mean(converged, axis=-1),
        "stall_fraction": np.mean(alpha > 14.0 * DEG, axis=-1),
        # spanwise
        "phi": phi, "alpha": alpha, "cl": cl, "cd": cd, "w": w,
        "dt_dr": dt_dr, "dq_dr": dq_dr, "loss_factor": f_loss,
        "mach": w / ctx.a_sound, "reynolds": rho * w * ctx.chord / ctx.mu,
        "v_axial_induced": w * sin_p - ctx.v_inf,
        "v_swirl_induced": np.maximum(ctx.omega_r - w * cos_p, 0.0),
    }


__all__ += ["solve_batch", "integrate_batch"]
