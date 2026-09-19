"""International Standard Atmosphere plus the fluid properties BEMT needs.

Only the troposphere and lower stratosphere are modelled (0 - 32 km), which
covers every propeller anybody is going to type into this program.  Viscosity
comes from Sutherland's law because Reynolds number materially changes the
drag polar at model-aircraft scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ISA sea-level reference values
T0 = 288.15         # K
P0 = 101325.0       # Pa
RHO0 = 1.225        # kg/m^3
G0 = 9.80665        # m/s^2
R_AIR = 287.05287   # J/(kg K)
GAMMA = 1.4

# Sutherland's law constants for air
_SUTH_MU0 = 1.716e-5
_SUTH_T0 = 273.15
_SUTH_S = 110.4

# (base altitude m, base temperature K, lapse rate K/m)
_LAYERS = (
    (0.0, 288.15, -0.0065),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 0.001),
    (32000.0, 228.65, 0.0028),
)


@dataclass(frozen=True, slots=True)
class AirState:
    """Everything the aerodynamics needs to know about the working fluid."""

    altitude: float = 0.0        # m, geopotential
    temperature: float = T0      # K
    pressure: float = P0         # Pa
    density: float = RHO0        # kg/m^3
    viscosity: float = 1.789e-5  # Pa s (dynamic)
    sound_speed: float = 340.294  # m/s

    @property
    def kinematic_viscosity(self) -> float:
        return self.viscosity / self.density

    def reynolds(self, speed: float, chord: float) -> float:
        """Chord Reynolds number for a section seeing ``speed``."""
        return self.density * abs(speed) * chord / self.viscosity

    def mach(self, speed: float) -> float:
        return abs(speed) / self.sound_speed

    def describe(self) -> str:
        return (
            f"{self.altitude:,.0f} m ISA{_offset_str(self)}  "
            f"rho={self.density:.4f} kg/m^3  T={self.temperature:.1f} K  "
            f"a={self.sound_speed:.1f} m/s  nu={self.kinematic_viscosity:.3e} m^2/s"
        )


def _offset_str(state: AirState) -> str:
    std = isa(state.altitude, delta_isa=0.0)
    d = state.temperature - std.temperature
    if abs(d) < 0.05:
        return ""
    return f"{d:+.1f}K"


def sutherland_viscosity(temperature: float) -> float:
    """Dynamic viscosity of air (Pa s) from Sutherland's law."""
    t = max(temperature, 1.0)
    return _SUTH_MU0 * ((t / _SUTH_T0) ** 1.5) * (_SUTH_T0 + _SUTH_S) / (t + _SUTH_S)


def isa(altitude: float = 0.0, delta_isa: float = 0.0, humidity: float = 0.0) -> AirState:
    """Build an :class:`AirState` at ``altitude`` metres.

    ``delta_isa`` offsets temperature from standard (hot-and-high performance).
    ``humidity`` is relative humidity in [0, 1]; moist air is *less* dense than
    dry air, which is worth a percent or so of thrust on a muggy day.
    """
    h = min(max(altitude, -1000.0), 32000.0)

    temperature = T0
    pressure = P0
    for i, (h_b, t_b, lapse) in enumerate(_LAYERS):
        h_top = _LAYERS[i + 1][0] if i + 1 < len(_LAYERS) else 32000.0
        if h <= h_top or i == len(_LAYERS) - 1:
            dh = h - h_b
            if abs(lapse) < 1e-12:
                temperature = t_b
                pressure = _layer_base_pressure(i) * math.exp(-G0 * dh / (R_AIR * t_b))
            else:
                temperature = t_b + lapse * dh
                pressure = _layer_base_pressure(i) * (temperature / t_b) ** (-G0 / (lapse * R_AIR))
            break

    temperature += delta_isa
    density = pressure / (R_AIR * temperature)

    if humidity > 0.0:
        # Buck equation for saturation vapour pressure, then the partial-pressure
        # correction.  Water vapour has a lower molar mass than dry air.
        t_c = temperature - 273.15
        p_sat = 611.21 * math.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))
        p_v = min(max(humidity, 0.0), 1.0) * p_sat
        density = (pressure - p_v) / (R_AIR * temperature) + p_v / (461.495 * temperature)

    return AirState(
        altitude=altitude,
        temperature=temperature,
        pressure=pressure,
        density=density,
        viscosity=sutherland_viscosity(temperature),
        sound_speed=math.sqrt(GAMMA * R_AIR * temperature),
    )


def density_altitude(state: AirState) -> float:
    """Altitude at which the standard atmosphere has this density."""
    ratio = state.density / RHO0
    lapse = -_LAYERS[0][2]
    # Invert the troposphere density relation rho/rho0 = (1 - Lh/T0)^(g/(L R) - 1)
    exponent = G0 / (lapse * R_AIR) - 1.0
    return (T0 / lapse) * (1.0 - ratio ** (1.0 / exponent))


def _layer_base_pressure(index: int) -> float:
    """Pressure at the base of ISA layer ``index`` (cached by construction)."""
    if index == 0:
        return P0
    p = P0
    for i in range(index):
        h_b, t_b, lapse = _LAYERS[i]
        h_top = _LAYERS[i + 1][0]
        dh = h_top - h_b
        if abs(lapse) < 1e-12:
            p *= math.exp(-G0 * dh / (R_AIR * t_b))
        else:
            t_top = t_b + lapse * dh
            p *= (t_top / t_b) ** (-G0 / (lapse * R_AIR))
    return p


SEA_LEVEL = isa(0.0)
