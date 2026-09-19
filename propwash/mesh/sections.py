"""Airfoil section coordinates for lofting the 3-D blade.

The solver never needs section *shape* -- only its Cl/Cd behaviour -- but the
visualiser does.  Rather than ship coordinate files for every preset, the
section is reconstructed from two numbers the polar already knows:

* thickness comes straight from the blade's thickness distribution;
* camber is inferred from the zero-lift angle, because for a NACA 4-digit
  section ``alpha_0 ~= -1.05 * (100 m)`` degrees where ``m`` is maximum camber.
  NACA 2412 has ``alpha_0 = -2.1 deg``, NACA 4412 has ``-4.2 deg``; the fit is
  good to a few tenths of a degree across the family.

So the shape you see is consistent with the aerodynamics being solved, without
a directory of .dat files.
"""

from __future__ import annotations

import math

import numpy as np


def cosine_spacing(n: int) -> np.ndarray:
    """Chordwise stations clustered at the leading and trailing edges."""
    beta = np.linspace(0.0, math.pi, n)
    return 0.5 * (1.0 - np.cos(beta))


def naca4_thickness(x: np.ndarray, t: float, sharp_te: bool = True) -> np.ndarray:
    """NACA 4-digit half-thickness distribution."""
    a4 = -0.1036 if sharp_te else -0.1015
    return 5.0 * t * (0.2969 * np.sqrt(np.clip(x, 0.0, 1.0))
                      - 0.1260 * x - 0.3516 * x ** 2
                      + 0.2843 * x ** 3 + a4 * x ** 4)


def naca4_camber(x: np.ndarray, m: float, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean line and its slope for a NACA 4-digit section."""
    x = np.clip(x, 0.0, 1.0)
    if m <= 1e-9:
        return np.zeros_like(x), np.zeros_like(x)
    p = float(np.clip(p, 0.05, 0.95))
    fore = x < p
    yc = np.where(fore,
                  m / p ** 2 * (2.0 * p * x - x ** 2),
                  m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x ** 2))
    dy = np.where(fore,
                  2.0 * m / p ** 2 * (p - x),
                  2.0 * m / (1.0 - p) ** 2 * (p - x))
    return yc, dy


def naca4_section(thickness: float, camber: float = 0.0, camber_pos: float = 0.4,
                  n_points: int = 61, sharp_te: bool = True) -> np.ndarray:
    """Closed section loop, TE -> upper -> LE -> lower -> TE.

    Returns ``(n, 2)`` of ``(x/c, y/c)`` with ``x/c`` in [0, 1].  The loop is
    ordered so that consecutive points form a consistent winding, which is what
    the lofter needs to emit outward-facing triangles.
    """
    n_half = max(int(n_points) // 2 + 1, 8)
    x = cosine_spacing(n_half)
    yt = naca4_thickness(x, max(thickness, 0.005), sharp_te)
    yc, dy = naca4_camber(x, camber, camber_pos)
    theta = np.arctan(dy)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # TE -> LE along the upper surface, then LE -> TE along the lower.
    upper = np.stack([xu[::-1], yu[::-1]], axis=1)
    lower = np.stack([xl[1:], yl[1:]], axis=1)
    loop = np.vstack([upper, lower])

    # Close the loop exactly; a gap here shows up as a visible seam.
    if not np.allclose(loop[0], loop[-1], atol=1e-9):
        loop = np.vstack([loop, loop[:1]])
    return loop


def camber_from_zero_lift(alpha_0_rad: float) -> float:
    """Invert the thin-airfoil relation between zero-lift angle and camber."""
    deg = -math.degrees(alpha_0_rad)
    return float(np.clip(deg / 105.0, 0.0, 0.09))


def section_for_polar(polar, n_points: int = 61) -> np.ndarray:
    """Build the coordinates that go with a given :class:`AnalyticPolar`."""
    camber = camber_from_zero_lift(getattr(polar, "alpha_0", 0.0))
    return naca4_section(getattr(polar, "thickness", 0.12), camber,
                         camber_pos=0.4, n_points=n_points)


def section_area(loop: np.ndarray) -> float:
    """Enclosed area of a section loop (shoelace) -- used for blade mass."""
    x, y = loop[:, 0], loop[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


__all__ = ["cosine_spacing", "naca4_thickness", "naca4_camber", "naca4_section",
           "camber_from_zero_lift", "section_for_polar", "section_area"]
