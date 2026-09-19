"""Brushless motor model and propeller/motor matching.

This is what turns the app from a propeller calculator into an answer to the
question the user actually asked: *what makes the propeller spin faster?*

A propeller does not have an RPM of its own.  It has a power demand that rises
roughly as RPM-cubed; a motor has a power output that falls as RPM rises toward
its no-load speed.  The operating RPM is where the two curves cross.  Change
the pitch, the diameter, the blade count, the air density or the throttle and
the crossing moves -- which is exactly the coupling the 3-D view animates.

The motor is the standard first-order DC model, which describes a brushless
outrunner driven by an ESC remarkably well:

    omega = Kv (V_applied - I R_m),    Q_shaft = (I - I_0) / Kv
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .units import rad_s_to_rpm, rpm_to_rad_s


@dataclass(slots=True)
class MotorSpec:
    """First-order brushless DC motor.

    ``kv`` is in the units everybody quotes it in: RPM per volt, unloaded.
    """

    name: str = "Generic 2820 kv1000"
    kv: float = 1000.0          # rpm/V
    resistance: float = 0.085   # ohm, phase-to-phase / 2
    no_load_current: float = 0.7  # A
    voltage: float = 11.1       # V, pack nominal
    max_current: float = 30.0   # A, ESC/motor limit
    internal_resistance: float = 0.012  # ohm, battery + wiring
    cells: int = 3

    @property
    def kv_rad(self) -> float:
        """Kv in rad/s per volt."""
        return self.kv * 2.0 * math.pi / 60.0

    @property
    def kt(self) -> float:
        """Torque constant (N m / A) -- the reciprocal of Kv in SI."""
        return 1.0 / self.kv_rad

    def terminal_voltage(self, current: np.ndarray, throttle: float = 1.0) -> np.ndarray:
        """Voltage at the motor after battery sag and throttle duty cycle."""
        v = self.voltage * float(np.clip(throttle, 0.0, 1.0))
        return np.maximum(v - np.asarray(current, dtype=float) * self.internal_resistance, 0.0)

    def current_at(self, rpm: np.ndarray, throttle: float = 1.0) -> np.ndarray:
        """Current drawn at a given shaft speed.

        Solves ``V - I(R_m + R_int) = omega / Kv`` for ``I``, i.e. the battery
        sag and the winding drop together.
        """
        omega = rpm_to_rad_s(np.asarray(rpm, dtype=float))
        v_open = self.voltage * float(np.clip(throttle, 0.0, 1.0))
        r_total = self.resistance + self.internal_resistance
        return np.maximum((v_open - omega * self.kt) / r_total, 0.0)

    def shaft_torque(self, rpm: np.ndarray, throttle: float = 1.0) -> np.ndarray:
        """Torque the motor can deliver at ``rpm`` (N m), after iron/idle losses."""
        i = np.minimum(self.current_at(rpm, throttle), self.max_current)
        return np.maximum((i - self.no_load_current) * self.kt, 0.0)

    def shaft_power(self, rpm: np.ndarray, throttle: float = 1.0) -> np.ndarray:
        return self.shaft_torque(rpm, throttle) * rpm_to_rad_s(np.asarray(rpm, dtype=float))

    def electrical_power(self, rpm: np.ndarray, throttle: float = 1.0) -> np.ndarray:
        i = np.minimum(self.current_at(rpm, throttle), self.max_current)
        return i * self.terminal_voltage(i, throttle)

    def efficiency(self, rpm: np.ndarray, throttle: float = 1.0) -> np.ndarray:
        p_in = self.electrical_power(rpm, throttle)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.clip(np.nan_to_num(self.shaft_power(rpm, throttle) / np.maximum(p_in, 1e-9)),
                           0.0, 1.0)

    def no_load_rpm(self, throttle: float = 1.0) -> float:
        return self.kv * self.voltage * float(np.clip(throttle, 0.0, 1.0))

    def with_cells(self, cells: int) -> "MotorSpec":
        """Rebuild on a different LiPo cell count (3.7 V nominal per cell)."""
        return replace(self, cells=cells, voltage=3.7 * cells)

    def describe(self) -> str:
        return (f"{self.name}: {self.kv:.0f} kv, {self.cells}S ({self.voltage:.1f} V), "
                f"Rm={self.resistance * 1000:.0f} mohm, I0={self.no_load_current:.1f} A, "
                f"no-load {self.no_load_rpm():.0f} rpm")


MOTOR_PRESETS: dict[str, MotorSpec] = {
    m.name: m for m in (
        MotorSpec("Micro 1104 kv7500", kv=7500, resistance=0.35, no_load_current=0.35,
                  voltage=7.4, max_current=8.0, cells=2),
        MotorSpec("Racing 2207 kv2400", kv=2400, resistance=0.055, no_load_current=1.1,
                  voltage=22.2, max_current=45.0, cells=6),
        MotorSpec("Sport 2820 kv1000", kv=1000, resistance=0.085, no_load_current=0.7,
                  voltage=11.1, max_current=30.0, cells=3),
        MotorSpec("Trainer 3536 kv910", kv=910, resistance=0.065, no_load_current=1.2,
                  voltage=14.8, max_current=40.0, cells=4),
        MotorSpec("Heavy-lift 5010 kv340", kv=340, resistance=0.120, no_load_current=0.5,
                  voltage=22.2, max_current=30.0, cells=6),
        MotorSpec("Glider 2814 kv1400", kv=1400, resistance=0.070, no_load_current=0.9,
                  voltage=11.1, max_current=25.0, cells=3),
    )
}

DEFAULT_MOTOR = "Sport 2820 kv1000"


def get_motor(name: str) -> MotorSpec:
    if name not in MOTOR_PRESETS:
        raise KeyError(f"unknown motor {name!r}; have {list(MOTOR_PRESETS)}")
    return replace(MOTOR_PRESETS[name])


def list_motors() -> list[str]:
    return list(MOTOR_PRESETS)


@dataclass(slots=True)
class MatchPoint:
    """Where the propeller's power demand meets the motor's power supply."""

    rpm: float
    thrust: float
    torque: float
    shaft_power: float
    electrical_power: float
    current: float
    motor_efficiency: float
    system_efficiency: float
    converged: bool
    throttle: float = 1.0

    def describe(self) -> str:
        return (f"{self.rpm:.0f} rpm | T={self.thrust:.2f} N ({self.thrust / 9.80665 * 1000:.0f} gf) | "
                f"{self.current:.1f} A | {self.electrical_power:.0f} W electrical | "
                f"motor eta={self.motor_efficiency * 100:.0f}%")


def match_rpm(prop_torque: Callable[[np.ndarray], np.ndarray], motor: MotorSpec,
              throttle: float = 1.0, rpm_max: float | None = None,
              tol: float = 0.5, max_iter: int = 60) -> tuple[float, bool]:
    """Find the RPM where propeller torque demand equals motor torque supply.

    Both curves are monotone in opposite directions over the operating range --
    propeller torque rises as RPM^2, motor torque falls linearly -- so the
    difference has exactly one sign change and bisection is unconditionally
    safe.  No initial guess required, which matters when the GUI is calling this
    sixty times a second with wildly different geometry.
    """
    hi = rpm_max if rpm_max is not None else motor.no_load_rpm(throttle)
    hi = max(hi, 1.0)
    lo = 1.0

    f = lambda n: float(motor.shaft_torque(np.array([n]), throttle)[0]   # noqa: E731
                        - np.asarray(prop_torque(np.array([n]))).ravel()[0])

    f_lo, f_hi = f(lo), f(hi)
    if f_lo <= 0.0:
        return 0.0, False           # motor cannot even break the prop loose
    if f_hi > 0.0:
        return hi, False            # prop never absorbs the available power

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            break
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), True


__all__ = ["MotorSpec", "MOTOR_PRESETS", "DEFAULT_MOTOR", "get_motor", "list_motors",
           "MatchPoint", "match_rpm", "rad_s_to_rpm", "rpm_to_rad_s"]
