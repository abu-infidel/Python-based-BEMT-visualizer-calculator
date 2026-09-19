"""Unit helpers.

The solver is strictly SI internally (m, kg, s, N, W, rad).  Cockpit-style
inputs (inches, RPM, knots) are converted at the boundary so that nothing
downstream has to think about it.
"""

from __future__ import annotations

import math

# --- length -----------------------------------------------------------------
INCH = 0.0254
FOOT = 0.3048

# --- speed ------------------------------------------------------------------
KNOT = 0.514444
MPH = 0.44704
KMH = 1.0 / 3.6

# --- force / mass -----------------------------------------------------------
LBF = 4.4482216152605
GRAM_FORCE = 9.80665e-3
KGF = 9.80665

# --- power ------------------------------------------------------------------
HORSEPOWER = 745.699872


def rpm_to_rad_s(rpm: float) -> float:
    """Revolutions per minute -> radians per second."""
    return rpm * 2.0 * math.pi / 60.0


def rad_s_to_rpm(omega: float) -> float:
    """Radians per second -> revolutions per minute."""
    return omega * 60.0 / (2.0 * math.pi)


def inch_pitch_to_twist(pitch_m: float, radius_m: float) -> float:
    """Geometric twist angle (rad) of a constant-pitch helix at ``radius_m``.

    A "10x7" propeller advances 7 inches per revolution if it were a screw in
    a solid; the local blade angle that produces that is ``atan(p / (2 pi r))``.
    """
    if radius_m <= 0.0:
        return math.pi / 2.0
    return math.atan2(pitch_m, 2.0 * math.pi * radius_m)


def advance_ratio(v_inf: float, rpm: float, diameter: float) -> float:
    """Propeller advance ratio ``J = V / (n D)`` with ``n`` in rev/s."""
    n = rpm / 60.0
    if n <= 0.0 or diameter <= 0.0:
        return 0.0
    return v_inf / (n * diameter)


def tip_mach(rpm: float, radius: float, v_inf: float, a_sound: float) -> float:
    """Helical tip Mach number (rotational and axial components combined)."""
    u_tip = rpm_to_rad_s(rpm) * radius
    return math.hypot(u_tip, v_inf) / max(a_sound, 1e-9)
