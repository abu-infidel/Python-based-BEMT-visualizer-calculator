"""Piston aero-engine model, for aircraft propellers.

:mod:`propwash.motor` answers "how fast does it spin?" for an electric drone.
This module answers the same question for a light aircraft, and the answer has
a different shape.

A brushless motor's torque falls linearly to zero at its no-load speed, so a
propeller that is too fine simply spins up and stops there.  A normally
aspirated piston engine has a torque curve that *peaks below* rated speed and
droops only gently, so a fine propeller will happily drive it past redline --
which is a real failure mode (an over-revving fixed-pitch prop), not a
convergence problem, and is reported as such.

Two further differences matter:

* **Altitude.** An unsupercharged engine loses power faster than density alone
  suggests, because it still has to pump exhaust against a pressure that falls
  more slowly than intake density.  The Gagg-Farrar relation,
  ``P/P_SL = sigma - (1 - sigma)/7.55``, captures it in one term.
* **Gearing.** Direct-drive engines (Lycoming, Continental) turn the propeller
  at crankshaft speed, so propeller tip Mach is what limits RPM.  High-revving
  engines (Rotax) need a reduction gearbox, and the propeller then sees
  crankshaft speed divided by the gear ratio.

The interface deliberately mirrors :class:`~propwash.motor.MotorSpec`
(``shaft_torque``, ``shaft_power``, ``no_load_rpm``) so the same
:func:`~propwash.motor.match_rpm` bisection drives both.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .atmosphere import RHO0, AirState
from .units import HORSEPOWER, rpm_to_rad_s

#: Avgas density, kg/L, for turning a mass flow into something a pilot reads.
AVGAS_DENSITY = 0.72

#: Brake specific fuel consumption, kg/(W s).  0.50 lb/(hp hr) is typical of a
#: normally aspirated Lycoming at cruise power.
DEFAULT_BSFC = 0.50 * 0.45359237 / (HORSEPOWER * 3600.0)


@dataclass(slots=True)
class PistonEngine:
    """A normally aspirated four-stroke aero engine.

    ``rated_power`` is in watts at sea level; the presets are built from the
    horsepower figures on the type certificate.
    """

    name: str = "Lycoming O-320-D2J"
    rated_power: float = 160.0 * HORSEPOWER   # W, sea level
    rated_rpm: float = 2700.0                 # crankshaft
    idle_rpm: float = 600.0
    redline_rpm: float = 2700.0
    peak_torque_frac: float = 0.78            # N at peak torque / N rated
    torque_droop: float = 0.08                # torque lost between peak and rated
    gear_ratio: float = 1.0                   # crank rev per propeller rev
    bsfc: float = DEFAULT_BSFC
    displacement_l: float = 5.24
    cylinders: int = 4
    supercharged: bool = False

    # -- basic quantities --------------------------------------------------
    @property
    def rated_power_hp(self) -> float:
        return self.rated_power / HORSEPOWER

    @property
    def peak_torque(self) -> float:
        """Crankshaft torque at the top of the torque curve (N m).

        Anchored so that power at rated RPM comes out equal to ``rated_power``.
        """
        omega_rated = rpm_to_rad_s(self.rated_rpm)
        return self.rated_power / max((1.0 - self.torque_droop) * omega_rated, 1e-9)

    def prop_rpm(self, crank_rpm) -> np.ndarray:
        """Propeller speed for a given crankshaft speed."""
        return np.asarray(crank_rpm, dtype=float) / self.gear_ratio

    def crank_rpm(self, prop_rpm) -> np.ndarray:
        """Crankshaft speed for a given propeller speed."""
        return np.asarray(prop_rpm, dtype=float) * self.gear_ratio

    # -- the curves --------------------------------------------------------
    def altitude_factor(self, air: AirState | None = None) -> float:
        """Gagg-Farrar power lapse with density ratio.

        At 8,000 ft (sigma = 0.79) this gives 0.76 of sea-level power, against
        0.79 for density alone -- the few percent that makes a normally
        aspirated aircraft feel tired before the numbers say it should.
        """
        if air is None or self.supercharged:
            return 1.0
        sigma = air.density / RHO0
        return float(np.clip(sigma - (1.0 - sigma) / 7.55, 0.0, 1.2))

    def crank_torque(self, crank_rpm, throttle: float = 1.0,
                     air: AirState | None = None) -> np.ndarray:
        """Crankshaft torque (N m).

        A parabola in RPM peaking at ``peak_torque_frac`` of rated speed.  Below
        idle the engine cannot sustain itself, so torque is zero -- which also
        keeps the propeller-matching bisection well behaved.
        """
        n = np.asarray(crank_rpm, dtype=float) / max(self.rated_rpm, 1e-9)
        p = self.peak_torque_frac
        span = max(1.0 - p, 1e-3)

        shape = 1.0 - self.torque_droop * ((n - p) / span) ** 2
        torque = self.peak_torque * np.clip(shape, 0.0, None)

        torque = torque * float(np.clip(throttle, 0.0, 1.0)) * self.altitude_factor(air)
        return np.where(np.asarray(crank_rpm, dtype=float) < self.idle_rpm, 0.0, torque)

    # -- the MotorSpec-compatible interface (propeller shaft) --------------
    def shaft_torque(self, rpm, throttle: float = 1.0,
                     air: AirState | None = None) -> np.ndarray:
        """Torque delivered to the *propeller* (N m), after any reduction gear."""
        return self.crank_torque(self.crank_rpm(rpm), throttle, air) * self.gear_ratio

    def shaft_power(self, rpm, throttle: float = 1.0,
                    air: AirState | None = None) -> np.ndarray:
        return self.shaft_torque(rpm, throttle, air) * rpm_to_rad_s(
            np.asarray(rpm, dtype=float))

    def no_load_rpm(self, throttle: float = 1.0) -> float:
        """Upper bracket for propeller matching: propeller RPM at redline.

        A piston engine has no true no-load speed -- it would destroy itself
        first.  Redline is the honest ceiling, and a propeller that does not
        load the engine down to it is over-revving.
        """
        return self.redline_rpm / self.gear_ratio

    @property
    def max_current(self) -> float:      # pragma: no cover - interface parity
        """Present for interface parity with MotorSpec; not meaningful here."""
        return float("inf")

    # -- consumption -------------------------------------------------------
    def fuel_flow(self, rpm, throttle: float = 1.0,
                  air: AirState | None = None) -> np.ndarray:
        """Fuel mass flow, kg/s."""
        return self.bsfc * np.maximum(self.shaft_power(rpm, throttle, air), 0.0)

    def fuel_flow_lph(self, rpm, throttle: float = 1.0,
                      air: AirState | None = None) -> np.ndarray:
        """Fuel flow in litres per hour -- what the gauge shows."""
        return self.fuel_flow(rpm, throttle, air) * 3600.0 / AVGAS_DENSITY

    def fuel_flow_gph(self, rpm, throttle: float = 1.0,
                      air: AirState | None = None) -> np.ndarray:
        return self.fuel_flow_lph(rpm, throttle, air) / 3.785411784

    def power_fraction(self, rpm, throttle: float = 1.0,
                       air: AirState | None = None) -> np.ndarray:
        """Percentage of rated power -- the number a pilot sets cruise by."""
        return self.shaft_power(rpm, throttle, air) / max(self.rated_power, 1e-9)

    # -- helpers -----------------------------------------------------------
    def with_power(self, horsepower: float) -> "PistonEngine":
        return replace(self, rated_power=horsepower * HORSEPOWER)

    def describe(self) -> str:
        gear = "" if abs(self.gear_ratio - 1.0) < 1e-6 else f", {self.gear_ratio:.2f}:1 gearbox"
        return (f"{self.name}: {self.rated_power_hp:.0f} hp @ {self.rated_rpm:.0f} rpm, "
                f"{self.cylinders} cyl, {self.displacement_l:.2f} L{gear}, "
                f"prop redline {self.no_load_rpm():.0f} rpm")


ENGINE_PRESETS: dict[str, PistonEngine] = {
    e.name: e for e in (
        PistonEngine("Lycoming O-320-D2J (C172N/P)", 160.0 * HORSEPOWER, 2700.0,
                     redline_rpm=2700.0, displacement_l=5.24, cylinders=4),
        PistonEngine("Lycoming IO-360-L2A (C172S)", 180.0 * HORSEPOWER, 2700.0,
                     redline_rpm=2700.0, displacement_l=5.92, cylinders=4),
        PistonEngine("Continental O-200-A (C150)", 100.0 * HORSEPOWER, 2750.0,
                     redline_rpm=2750.0, displacement_l=3.29, cylinders=4),
        PistonEngine("Lycoming IO-540-K (PA-32)", 300.0 * HORSEPOWER, 2700.0,
                     redline_rpm=2700.0, displacement_l=8.85, cylinders=6),
        PistonEngine("Rotax 912ULS (LSA)", 100.0 * HORSEPOWER, 5800.0,
                     idle_rpm=1400.0, redline_rpm=5800.0, gear_ratio=2.43,
                     displacement_l=1.35, cylinders=4, peak_torque_frac=0.88,
                     torque_droop=0.06, bsfc=0.46 * 0.45359237 / (HORSEPOWER * 3600.0)),
        PistonEngine("Continental O-470-R (C182)", 230.0 * HORSEPOWER, 2600.0,
                     redline_rpm=2600.0, displacement_l=7.70, cylinders=6),
    )
}

DEFAULT_ENGINE = "Lycoming O-320-D2J (C172N/P)"


def get_engine(name: str) -> PistonEngine:
    if name not in ENGINE_PRESETS:
        raise KeyError(f"unknown engine {name!r}; have {list(ENGINE_PRESETS)}")
    return replace(ENGINE_PRESETS[name])


def list_engines() -> list[str]:
    return list(ENGINE_PRESETS)


__all__ = ["PistonEngine", "ENGINE_PRESETS", "DEFAULT_ENGINE", "get_engine",
           "list_engines", "AVGAS_DENSITY", "DEFAULT_BSFC"]
