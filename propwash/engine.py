"""Piston aero-engine model.

The engine answers one question for the propeller: how much torque is there to
turn it with, at this RPM and this altitude?  Every number that describes a
particular engine is an input -- there are no preset engines.  The only fixed
numbers are the *model* constants below, which describe how a typical
certificated piston engine behaves rather than any one engine, and every report
prints them.

Torque curve
    Full-throttle crankshaft torque is a parabola in RPM that peaks at
    :data:`PEAK_TORQUE_FRACTION` of rated speed and is :data:`TORQUE_DROOP`
    lower again at rated speed, anchored so power at rated RPM equals rated
    power.  Aircraft piston engines have flat torque curves, so this matters
    little near rated RPM and is only a rough guide far below it.

Altitude
    A normally aspirated engine loses power faster than air density, because it
    still pumps exhaust against an ambient pressure that falls more slowly.
    The Gagg-Farrar relation ``P/P_SL = sigma - (1 - sigma)/7.55`` captures it.
    A turbocharged (or turbonormalised) engine holds sea-level power up to its
    critical altitude and lapses by the same relation above it, with sigma
    taken relative to the density at the critical altitude.

RPM limit
    Rated RPM is also the RPM limit (true of the certificated direct-drive
    engines this is built for).  A propeller that would drive the engine past
    it at full throttle is reported as RPM-limited, with the throttle needed to
    hold the limit -- never silently clipped.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .atmosphere import RHO0, AirState, isa
from .units import HORSEPOWER, rpm_to_rad_s

#: Crankshaft RPM of peak full-throttle torque, as a fraction of rated RPM.
PEAK_TORQUE_FRACTION = 0.78
#: Fractional torque lost between the torque peak and rated RPM.
TORQUE_DROOP = 0.08
#: Gagg-Farrar altitude-lapse constant.
GAGG_FARRAR = 7.55
#: Lowest crankshaft speed, as a fraction of rated, the matching will consider
#: a running engine.  A propeller that holds the engine below this at full
#: throttle is reported as too coarse rather than given a made-up answer.
MIN_RUNNING_FRACTION = 0.25

#: Model constants, printed in every report so a result can be reproduced.
MODEL_CONSTANTS = {
    "peak_torque_fraction": PEAK_TORQUE_FRACTION,
    "torque_droop": TORQUE_DROOP,
    "gagg_farrar_constant": GAGG_FARRAR,
    "min_running_fraction": MIN_RUNNING_FRACTION,
}

ASPIRATIONS = ("normal", "turbocharged")

#: lb/(hp h) -> kg/J
BSFC_US_TO_SI = 0.45359237 / (HORSEPOWER * 3600.0)


@dataclass(frozen=True, slots=True)
class PistonEngine:
    """A piston aero engine, described entirely by its inputs.

    ``rated_power`` is watts at sea level (at or below the critical altitude
    for a turbocharged engine).  ``rated_rpm`` is crankshaft RPM and is also
    the RPM limit.  ``gear_ratio`` is crankshaft revolutions per propeller
    revolution (1 for direct drive).  ``bsfc`` (kg/J) is optional: without it
    the fuel-flow outputs are reported as not available.
    """

    rated_power: float
    rated_rpm: float
    gear_ratio: float
    aspiration: str
    critical_altitude: float | None = None    # m, turbocharged only
    bsfc: float | None = None                  # kg/J
    name: str = ""

    def __post_init__(self) -> None:
        if not (math.isfinite(self.rated_power) and self.rated_power > 0.0):
            raise ValueError("rated power must be a positive number")
        if not (math.isfinite(self.rated_rpm) and self.rated_rpm > 0.0):
            raise ValueError("rated RPM must be a positive number")
        if not (math.isfinite(self.gear_ratio) and self.gear_ratio > 0.0):
            raise ValueError("gear ratio must be a positive number")
        if self.aspiration not in ASPIRATIONS:
            raise ValueError(f"aspiration must be one of {ASPIRATIONS}")
        if self.aspiration == "turbocharged":
            if self.critical_altitude is None or not math.isfinite(self.critical_altitude):
                raise ValueError("a turbocharged engine needs its critical altitude")
        if self.bsfc is not None and not (math.isfinite(self.bsfc) and self.bsfc > 0.0):
            raise ValueError("BSFC must be a positive number when given")

    # -- speeds ------------------------------------------------------------
    @property
    def rated_power_hp(self) -> float:
        return self.rated_power / HORSEPOWER

    @property
    def max_prop_rpm(self) -> float:
        """Propeller RPM at the engine's RPM limit."""
        return self.rated_rpm / self.gear_ratio

    @property
    def min_prop_rpm(self) -> float:
        """Lowest propeller RPM the matching treats as a running engine."""
        return MIN_RUNNING_FRACTION * self.rated_rpm / self.gear_ratio

    def prop_rpm(self, engine_rpm):
        return np.asarray(engine_rpm, dtype=float) / self.gear_ratio

    def engine_rpm(self, prop_rpm):
        return np.asarray(prop_rpm, dtype=float) * self.gear_ratio

    # -- altitude ----------------------------------------------------------
    def altitude_factor(self, air: AirState) -> float:
        """Full-throttle power at this air state over rated power."""
        if self.aspiration == "turbocharged":
            rho_ref = isa(self.critical_altitude).density
            if air.density >= rho_ref:
                return 1.0
        else:
            rho_ref = RHO0
        sigma = air.density / rho_ref
        return float(max(sigma - (1.0 - sigma) / GAGG_FARRAR, 0.0))

    # -- curves (propeller shaft) ------------------------------------------
    def _shape(self, engine_rpm) -> np.ndarray:
        n = np.asarray(engine_rpm, dtype=float) / self.rated_rpm
        span = 1.0 - PEAK_TORQUE_FRACTION
        return np.clip(1.0 - TORQUE_DROOP * ((n - PEAK_TORQUE_FRACTION) / span) ** 2,
                       0.0, None)

    @property
    def peak_torque(self) -> float:
        """Sea-level full-throttle crankshaft torque at the torque peak (N m)."""
        return self.rated_power / ((1.0 - TORQUE_DROOP) * rpm_to_rad_s(self.rated_rpm))

    def full_throttle_torque(self, prop_rpm, air: AirState) -> np.ndarray:
        """Torque available at the propeller shaft at full throttle (N m)."""
        crank = self.engine_rpm(prop_rpm)
        return (self.peak_torque * self._shape(crank) * self.altitude_factor(air)
                * self.gear_ratio)

    def full_throttle_power(self, prop_rpm, air: AirState) -> np.ndarray:
        """Shaft power available at full throttle (W)."""
        return self.full_throttle_torque(prop_rpm, air) * rpm_to_rad_s(
            np.asarray(prop_rpm, dtype=float))

    # -- consumption -------------------------------------------------------
    def fuel_flow(self, shaft_power: float) -> float | None:
        """Fuel mass flow (kg/s) at a shaft power, or None without a BSFC."""
        if self.bsfc is None or shaft_power is None:
            return None
        return self.bsfc * max(float(shaft_power), 0.0)

    def describe(self) -> str:
        gear = "direct drive" if abs(self.gear_ratio - 1.0) < 1e-9 else \
            f"{self.gear_ratio:.3f}:1 reduction"
        turbo = (f", turbocharged to {self.critical_altitude / 0.3048:,.0f} ft"
                 if self.aspiration == "turbocharged" else ", normally aspirated")
        return (f"{self.name + ': ' if self.name else ''}{self.rated_power_hp:.0f} hp "
                f"@ {self.rated_rpm:.0f} rpm, {gear}{turbo}")

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["PistonEngine", "ASPIRATIONS", "MODEL_CONSTANTS", "BSFC_US_TO_SI",
           "PEAK_TORQUE_FRACTION", "TORQUE_DROOP", "GAGG_FARRAR",
           "MIN_RUNNING_FRACTION"]
