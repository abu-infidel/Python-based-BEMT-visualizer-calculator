"""Checks of the model against published data for a real aircraft.

The reference is the Cessna 172N with its standard fixed-pitch propeller, the
McCauley 1C160/DTM7557 (75 in diameter, 57 in pitch), and the 160 hp Lycoming
O-320-H2AD.  ``propwash validate`` runs every check and prints the table that
``docs/VALIDATION.md`` reproduces.

Three kinds of check, kept apart on purpose:

``calibration``
    A number used to set an unpublished input.  The 1C160/DTM7557 blade's
    activity factor is not published, so it was chosen so that full-throttle
    static RPM lands in the middle of the type-certificate range.  Agreement
    here is by construction and proves nothing on its own.
``validation``
    Propeller-only predictions that do not depend on the airframe: the power
    the propeller absorbs at the RPM and true airspeed of each published
    cruise-table row, against the published percentage of rated power.
``indicative``
    Whole-aircraft results that also depend on an estimated drag polar (not
    published by Cessna) and on installation effects the model leaves out --
    slipstream scrubbing over the cowling, cooling drag.  They show how close
    a complete calculation gets, with a looser tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .analysis import run
from .report import Report

#: Every input of the reference case.  Values from the POH / type certificate
#: except where ``C172N_ASSUMPTIONS`` says otherwise.
C172N_CASE: dict[str, Any] = {
    "case.name": "Cessna 172N, McCauley 1C160/DTM7557, Lycoming O-320-H2AD",
    "case.mode": "analysis",
    "flight.altitude_ft": 0.0, "flight.isa_deviation_c": 0.0, "flight.airspeed_ktas": 0.0,
    "engine.rated_power_hp": 160.0, "engine.rated_rpm": 2700.0, "engine.gear_ratio": 1.0,
    "engine.aspiration": "normal", "engine.bsfc_lb_hp_h": 0.42,
    "propeller.diameter_in": 75.0, "propeller.blades": 2, "propeller.activity_factor": 109.0,
    "propeller.spinner_diameter_in": 9.0, "propeller.planform": "standard",
    "propeller.root_airfoil": "clark_y", "propeller.tip_airfoil": "clark_y",
    "propeller.root_thickness_pct": 20.0, "propeller.tip_thickness_pct": 7.0,
    "propeller.pitch_type": "fixed", "propeller.pitch_in": 57.0,
    "operating.mode": "full_throttle",
    "airframe.weight_lb": 2300.0, "airframe.wing_area_ft2": 174.0,
    "airframe.wing_span_ft": 35.83, "airframe.cd0": 0.0341, "airframe.oswald_e": 0.77,
    "airframe.cl_max": 1.56,
}

C172N_ASSUMPTIONS = {
    "propeller.activity_factor": "not published; set so static RPM is mid-range (calibration)",
    "propeller.planform": "not published; typical fixed-pitch metal outline",
    "propeller.root_thickness_pct": "not published; typical metal blade",
    "propeller.tip_thickness_pct": "not published; typical metal blade",
    "propeller.root_airfoil": "flat-faced Clark Y-type section assumed",
    "propeller.spinner_diameter_in": "approximate",
    "airframe.cd0": "estimated drag polar used in performance examples, not from Cessna",
    "airframe.oswald_e": "estimated drag polar used in performance examples, not from Cessna",
    "airframe.cl_max": "from the POH flaps-up stall speed, 50 KCAS at 2,300 lb",
    "engine.bsfc_lb_hp_h": "implied by the POH fuel flows (8.4 gal/h at 75%)",
}

#: Published cruise-table rows: pressure altitude (ft), RPM, % rated power,
#: true airspeed (kt).  Cessna 172N POH, cruise performance, 2,300 lb, ISA.
POH_CRUISE = (
    (8000.0, 2650.0, 75.0, 122.0),
    (8000.0, 2500.0, 64.0, 114.0),
    (4000.0, 2300.0, 57.0, 105.0),
    (4000.0, 2200.0, 51.0, 99.0),
    (4000.0, 2100.0, 46.0, 93.0),
)

SOURCES = {
    "poh": "Cessna 172N Pilot's Operating Handbook, Section 5",
    "tcds": "FAA Type Certificate Data Sheet 3A12 (172N with McCauley 1C160/DTM7557)",
}


@dataclass(slots=True)
class Check:
    name: str
    kind: str                 # calibration | validation | indicative
    reference: float
    model: float | None
    unit: str
    tolerance_pct: float | None   # None for a range check
    source: str
    low: float | None = None
    high: float | None = None

    @property
    def error_pct(self) -> float | None:
        if self.model is None:
            return None
        return (self.model - self.reference) / self.reference * 100.0

    @property
    def passed(self) -> bool:
        if self.model is None:
            return False
        if self.low is not None and self.high is not None:
            return self.low <= self.model <= self.high
        return abs(self.error_pct) <= self.tolerance_pct


def _value(report: Report, section: str, key: str):
    return report.sections.get(section, {}).get(key).value \
        if key in report.sections.get(section, {}) else None


def run_validation() -> list[Check]:
    checks: list[Check] = []

    static = run(dict(C172N_CASE, **{"flight.airspeed_ktas": 0.0}))
    checks.append(Check("Static RPM, full throttle, sea level", "calibration", 2340.0,
                        _value(static, "OPERATING_POINT", "engine_rpm"), "rpm", None,
                        SOURCES["tcds"] + ": 2,280-2,400 rpm", 2280.0, 2400.0))

    no_airframe = {k: (None if k.startswith("airframe.") else v)
                   for k, v in C172N_CASE.items()}
    for alt, rpm, pct, ktas in POH_CRUISE:
        rep = run(dict(no_airframe, **{"flight.altitude_ft": alt, "flight.airspeed_ktas": ktas,
                                       "operating.mode": "set_rpm",
                                       "operating.engine_rpm": rpm}))
        model = _value(rep, "OPERATING_POINT", "power_pct_rated")
        if model is None:                       # needs more than full throttle: still report it
            need = _value(rep, "OPERATING_POINT", "required_power_hp")
            model = None if need is None else need / 160.0 * 100.0
        checks.append(Check(f"Power absorbed, {alt:,.0f} ft, {rpm:.0f} rpm, {ktas:.0f} KTAS",
                            "validation", pct, model, "% rated", 8.0, SOURCES["poh"]))

    climb = run(dict(C172N_CASE, **{"flight.airspeed_ktas": 74.0}))
    checks.append(Check("Maximum level speed, sea level", "indicative", 125.0,
                        _value(climb, "AIRFRAME", "level_speed_at_setting_ktas"), "KTAS", 5.0,
                        SOURCES["poh"]))
    checks.append(Check("Maximum rate of climb, sea level", "indicative", 770.0,
                        _value(climb, "AIRFRAME", "best_rate_of_climb_fpm"), "ft/min", 25.0,
                        SOURCES["poh"]))
    checks.append(Check("Service ceiling", "indicative", 14200.0,
                        _value(climb, "AIRFRAME", "service_ceiling_ft"), "ft", 25.0,
                        SOURCES["poh"]))
    return checks


def format_checks(checks: list[Check]) -> str:
    head = (f"{'check':<52} {'kind':<12} {'reference':>10} {'model':>10} {'error':>8}  "
            f"{'limit':>8}  result")
    lines = [head, "-" * len(head)]
    for c in checks:
        ref = (f"{c.low:,.0f}-{c.high:,.0f}" if c.low is not None else f"{c.reference:,.1f}")
        mod = "-" if c.model is None else f"{c.model:,.1f}"
        err = "-" if c.error_pct is None or c.low is not None else f"{c.error_pct:+.1f}%"
        lim = "range" if c.low is not None else f"+/-{c.tolerance_pct:.0f}%"
        lines.append(f"{c.name:<52} {c.kind:<12} {ref:>10} {mod:>10} {err:>8}  {lim:>8}  "
                     f"{'PASS' if c.passed else 'FAIL'}")
    return "\n".join(lines)


__all__ = ["C172N_CASE", "C172N_ASSUMPTIONS", "POH_CRUISE", "SOURCES", "Check",
           "run_validation", "format_checks"]
