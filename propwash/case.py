"""The calculator's inputs: what can be entered, what must be, and how it is checked.

Nothing here has a default value.  A case is a flat mapping of field keys to
values (numbers, choice strings, or the text a user typed); :func:`parse_case`
returns either a validated :class:`Case` or the list of what is missing or
wrong.  The rules:

* **Required means required.**  A blank required field is reported as missing,
  never filled in.  Which fields are required depends on the choices made
  (fixed or constant-speed pitch, turbocharged or not, analysis or sizing).
* **No upper limits.**  Numbers must be physically meaningful (a diameter must
  be positive, a hub smaller than the propeller), but any magnitude is
  accepted.  A value outside the range seen on full-size aircraft produces a
  warning, not a refusal.
* **Optional groups are all-or-nothing.**  The airframe is optional, but a
  half-entered airframe is an error, not a partial calculation.

Inputs use the units a pilot or propeller catalogue uses (inches, horsepower,
feet, knots); everything is converted to SI once, here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .airfoil import list_airfoils
from .geometry import list_planforms

MODES = ("analysis", "sizing")
PITCH_TYPES = ("fixed", "constant_speed")
ASPIRATIONS = ("normal", "turbocharged")
OPERATING_MODES = ("full_throttle", "set_rpm", "set_power")


@dataclass(frozen=True, slots=True)
class Field:
    key: str
    label: str
    unit: str
    kind: str                                   # "float" | "int" | "choice" | "text"
    group: str
    help: str
    choices: tuple[str, ...] = ()
    typical: tuple[float, float] | None = None  # full-size range; outside -> warning
    minimum: float | None = None                # physical floor (exclusive unless min_inclusive)
    min_inclusive: bool = False


def _f(key, label, unit, group, help, typical=None, minimum=0.0, inclusive=False,
       kind="float"):
    return Field(key, label, unit, kind, group, help, (), typical, minimum, inclusive)


def _c(key, label, group, help, choices):
    return Field(key, label, "", "choice", group, help, tuple(choices))


FIELDS: tuple[Field, ...] = (
    Field("case.name", "Case name", "", "text", "case",
          "Free text, copied into the report."),
    _c("case.mode", "Calculation", "case",
       "'analysis' evaluates a propeller you describe; 'sizing' designs one for a "
       "cruise point.", MODES),

    # -- flight condition (both modes) --------------------------------------
    _f("flight.altitude_ft", "Pressure altitude", "ft", "flight",
       "ISA pressure altitude of the operating (or design) point.",
       typical=(0.0, 30000.0), minimum=-3280.0, inclusive=True),
    _f("flight.isa_deviation_c", "ISA temperature deviation", "degC", "flight",
       "Outside air temperature minus the ISA temperature at that altitude (0 = standard day).",
       typical=(-40.0, 40.0), minimum=None),
    _f("flight.airspeed_ktas", "True airspeed", "kt", "flight",
       "True airspeed of the operating point (0 = static). Sizing needs a cruise speed > 0.",
       typical=(0.0, 300.0), minimum=0.0, inclusive=True),

    # -- engine (both modes) ------------------------------------------------
    _f("engine.rated_power_hp", "Rated power", "hp", "engine",
       "Maximum continuous brake horsepower at sea level (at or below critical "
       "altitude if turbocharged).", typical=(60.0, 600.0)),
    _f("engine.rated_rpm", "Rated engine RPM", "rpm", "engine",
       "Crankshaft RPM at rated power; also treated as the RPM limit.",
       typical=(2000.0, 6000.0)),
    _f("engine.gear_ratio", "Reduction gear ratio", "-", "engine",
       "Crankshaft revolutions per propeller revolution (1 = direct drive).",
       typical=(1.0, 3.0)),
    _c("engine.aspiration", "Aspiration", "engine",
       "'normal' lapses with altitude from sea level; 'turbocharged' holds rated "
       "power to its critical altitude.", ASPIRATIONS),
    _f("engine.critical_altitude_ft", "Critical altitude", "ft", "engine",
       "Turbocharged only: highest altitude at which rated power is available.",
       typical=(5000.0, 25000.0), minimum=0.0, inclusive=True),
    _f("engine.bsfc_lb_hp_h", "Brake specific fuel consumption", "lb/(hp*h)", "engine",
       "Optional. Without it, fuel-flow outputs are reported as not available.",
       typical=(0.38, 0.60)),

    # -- propeller (analysis; sizing uses blades, spinner, thickness) -------
    _f("propeller.diameter_in", "Diameter", "in", "propeller",
       "Tip-to-tip diameter.", typical=(48.0, 160.0)),
    _f("propeller.blades", "Number of blades", "-", "propeller",
       "Blade count.", typical=(2.0, 6.0), kind="int"),
    _f("propeller.activity_factor", "Activity factor (per blade)", "-", "propeller",
       "AF = (100000/16) * integral of (c/D) x^3 dx from x = 0.15 to 1.",
       typical=(70.0, 220.0)),
    _f("propeller.spinner_diameter_in", "Spinner diameter", "in", "propeller",
       "Sets where the blade's working span begins.", typical=(6.0, 30.0)),
    _c("propeller.planform", "Blade planform", "propeller",
       "Chord distribution shape; it is scaled to the activity factor.",
       list_planforms()),
    _c("propeller.root_airfoil", "Root section", "propeller",
       "Airfoil family at the blade root.", list_airfoils()),
    _c("propeller.tip_airfoil", "Tip section", "propeller",
       "Airfoil family at the tip (blended smoothly from the root).", list_airfoils()),
    _f("propeller.root_thickness_pct", "Root thickness ratio", "%", "propeller",
       "Section thickness-to-chord at the spinner.", typical=(12.0, 35.0)),
    _f("propeller.tip_thickness_pct", "Tip thickness ratio", "%", "propeller",
       "Section thickness-to-chord at the tip.", typical=(4.0, 12.0)),
    _c("propeller.pitch_type", "Pitch type", "propeller",
       "'fixed' or 'constant_speed'.", PITCH_TYPES),
    _f("propeller.pitch_in", "Geometric pitch", "in", "propeller",
       "Fixed pitch only: the 'P' in 'D x P', measured at 0.75 R from the blade face.",
       typical=(40.0, 110.0)),
    _f("propeller.fine_stop_deg", "Fine (low) pitch stop", "deg", "propeller",
       "Constant speed only: blade angle at 0.75 R on the low-pitch stop.",
       typical=(8.0, 25.0), minimum=None),
    _f("propeller.coarse_stop_deg", "Coarse (high) pitch stop", "deg", "propeller",
       "Constant speed only: blade angle at 0.75 R on the high-pitch stop.",
       typical=(25.0, 50.0), minimum=None),

    # -- operating setting --------------------------------------------------
    _c("operating.mode", "Power setting", "operating",
       "'full_throttle'; 'set_rpm' (fixed pitch: hold an engine RPM); "
       "'set_power' (a percentage of rated power).", OPERATING_MODES),
    _f("operating.engine_rpm", "Engine RPM", "rpm", "operating",
       "Fixed pitch 'set_rpm', every constant-speed case (the governor setting), and sizing.",
       typical=(1500.0, 6000.0)),
    _f("operating.power_pct", "Power", "% rated", "operating",
       "'set_power' and sizing: shaft power as a percentage of rated power.",
       typical=(40.0, 100.0)),

    # -- airframe (optional, all-or-nothing) --------------------------------
    _f("airframe.weight_lb", "Weight", "lb", "airframe",
       "Optional group: enables drag, climb, speed and ceiling results.",
       typical=(800.0, 15000.0)),
    _f("airframe.wing_area_ft2", "Wing area", "ft^2", "airframe",
       "Reference wing area.", typical=(80.0, 600.0)),
    _f("airframe.wing_span_ft", "Wing span", "ft", "airframe",
       "Tip-to-tip span.", typical=(20.0, 80.0)),
    _f("airframe.cd0", "Zero-lift drag coefficient CD0", "-", "airframe",
       "Parasite drag coefficient referred to the wing area.", typical=(0.018, 0.060)),
    _f("airframe.oswald_e", "Oswald efficiency e", "-", "airframe",
       "Span efficiency of the drag polar CD = CD0 + CL^2/(pi e AR).",
       typical=(0.60, 0.90)),
    _f("airframe.cl_max", "CLmax (flaps up)", "-", "airframe",
       "Clean maximum lift coefficient; bounds the slow end of the climb search.",
       typical=(1.2, 2.0)),

    # -- sizing -------------------------------------------------------------
    _f("sizing.max_diameter_in", "Maximum diameter", "in", "sizing",
       "Largest diameter the installation allows (ground and fuselage clearance).",
       typical=(48.0, 160.0)),
    _f("sizing.tip_mach_limit", "Tip Mach limit", "-", "sizing",
       "Highest helical tip Mach number to accept at the design point.",
       typical=(0.75, 0.92)),
    _f("sizing.design_cl", "Section design lift coefficient", "-", "sizing",
       "Lift coefficient every station is designed to run at.", typical=(0.3, 0.8)),
    _c("sizing.airfoil", "Section for the designed blade", "sizing",
       "Airfoil family used along the designed blade.", list_airfoils()),
)

FIELD_BY_KEY: dict[str, Field] = {f.key: f for f in FIELDS}
GROUPS = ("case", "flight", "engine", "propeller", "operating", "airframe", "sizing")
AIRFRAME_KEYS = tuple(f.key for f in FIELDS if f.group == "airframe")


@dataclass(slots=True)
class Problem:
    code: str          # MISSING | INVALID | WARNING-style codes, see report-format README
    key: str
    message: str


@dataclass(slots=True)
class Case:
    """Validated inputs: raw values as entered plus their SI conversions."""

    values: dict[str, Any]
    si: dict[str, Any]
    warnings: list[Problem] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return self.values["case.mode"]

    @property
    def has_airframe(self) -> bool:
        return all(self.values.get(k) is not None for k in AIRFRAME_KEYS)


class CaseError(ValueError):
    """Raised by :func:`parse_case` with every problem found, not just the first."""

    def __init__(self, problems: list[Problem], values: dict[str, Any]):
        self.problems = problems
        self.values = values
        super().__init__("; ".join(f"{p.key}: {p.message}" for p in problems))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _coerce(f: Field, raw: Any) -> Any:
    """Turn what was typed into a value of the right kind, or raise ValueError."""
    if f.kind == "text":
        return str(raw).strip()
    if f.kind == "choice":
        val = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        if val not in f.choices:
            raise ValueError(f"must be one of: {', '.join(f.choices)}")
        return val
    if isinstance(raw, bool):
        raise ValueError("must be a number")
    if isinstance(raw, (int, float)):
        num = float(raw)
    else:
        text = str(raw).strip().replace(",", "").replace("_", "")
        try:
            num = float(text)
        except ValueError:
            raise ValueError(f"'{raw}' is not a number") from None
    if not math.isfinite(num):
        raise ValueError("must be a finite number")
    if f.kind == "int":
        if abs(num - round(num)) > 1e-9:
            raise ValueError("must be a whole number")
        return int(round(num))
    return num


def required_keys(values: dict[str, Any]) -> list[str]:
    """Which fields this combination of choices needs."""
    mode = values.get("case.mode")
    need = ["case.mode", "flight.altitude_ft", "flight.isa_deviation_c",
            "flight.airspeed_ktas", "engine.rated_power_hp", "engine.rated_rpm",
            "engine.gear_ratio", "engine.aspiration"]
    if values.get("engine.aspiration") == "turbocharged":
        need.append("engine.critical_altitude_ft")
    if mode == "sizing":
        need += ["propeller.blades", "propeller.spinner_diameter_in",
                 "propeller.root_thickness_pct", "propeller.tip_thickness_pct",
                 "operating.engine_rpm", "operating.power_pct",
                 "sizing.max_diameter_in", "sizing.tip_mach_limit", "sizing.design_cl",
                 "sizing.airfoil"]
        return need
    need += ["propeller.diameter_in", "propeller.blades", "propeller.activity_factor",
             "propeller.spinner_diameter_in", "propeller.planform",
             "propeller.root_airfoil", "propeller.tip_airfoil",
             "propeller.root_thickness_pct", "propeller.tip_thickness_pct",
             "propeller.pitch_type", "operating.mode"]
    pitch = values.get("propeller.pitch_type")
    op = values.get("operating.mode")
    if pitch == "fixed":
        need.append("propeller.pitch_in")
        if op == "set_rpm":
            need.append("operating.engine_rpm")
    elif pitch == "constant_speed":
        need += ["propeller.fine_stop_deg", "propeller.coarse_stop_deg",
                 "operating.engine_rpm"]
    if op == "set_power":
        need.append("operating.power_pct")
    return need


def parse_case(raw: dict[str, Any]) -> Case:
    """Validate a case.  Raises :class:`CaseError` listing every problem."""
    problems: list[Problem] = []
    values: dict[str, Any] = {}

    unknown = [k for k in raw if k not in FIELD_BY_KEY]
    for k in unknown:
        problems.append(Problem("UNKNOWN_FIELD", k, "is not a recognised input"))

    for f in FIELDS:
        r = raw.get(f.key)
        if _blank(r):
            values[f.key] = None
            continue
        try:
            values[f.key] = _coerce(f, r)
        except ValueError as exc:
            values[f.key] = None
            problems.append(Problem("INVALID", f.key, f"{f.label}: {exc}"))

    for key in required_keys(values):
        if values.get(key) is None and not any(p.key == key for p in problems):
            problems.append(Problem("MISSING", key, f"{FIELD_BY_KEY[key].label} is required"))

    # physical floors
    for f in FIELDS:
        v = values.get(f.key)
        if v is None or f.kind not in ("float", "int") or f.minimum is None:
            continue
        bad = v < f.minimum if f.min_inclusive else v <= f.minimum
        if bad:
            rel = ">=" if f.min_inclusive else ">"
            problems.append(Problem("INVALID", f.key,
                                    f"{f.label} must be {rel} {f.minimum:g} {f.unit}".rstrip()))

    _cross_checks(values, problems)
    if problems:
        raise CaseError(problems, values)

    warnings = []
    for key in required_keys(values) + [k for k in ("engine.bsfc_lb_hp_h",) if values.get(k)]:
        f = FIELD_BY_KEY[key]
        v = values.get(key)
        if f.typical and isinstance(v, (int, float)) and not (f.typical[0] <= v <= f.typical[1]):
            warnings.append(Problem(
                "OUTSIDE_TYPICAL_RANGE", key,
                f"{f.label} = {v:g} {f.unit} is outside the usual full-size range "
                f"{f.typical[0]:g}-{f.typical[1]:g}; the result is computed but the model "
                f"has not been checked there.".replace("  ", " ")))
    if values.get("case.mode") == "analysis" and values.get("airframe.weight_lb") is not None:
        for key in AIRFRAME_KEYS:
            f = FIELD_BY_KEY[key]
            v = values[key]
            if f.typical and not (f.typical[0] <= v <= f.typical[1]):
                warnings.append(Problem(
                    "OUTSIDE_TYPICAL_RANGE", key,
                    f"{f.label} = {v:g} {f.unit} is outside the usual range "
                    f"{f.typical[0]:g}-{f.typical[1]:g}."))
    return Case(values=values, si=_to_si(values), warnings=warnings)


def _cross_checks(v: dict[str, Any], problems: list[Problem]) -> None:
    def bad(key, msg):
        problems.append(Problem("INVALID", key, msg))

    mode = v.get("case.mode")
    if mode == "analysis":
        d, s = v.get("propeller.diameter_in"), v.get("propeller.spinner_diameter_in")
        if d and s and s >= 0.9 * d:
            bad("propeller.spinner_diameter_in", "spinner must be smaller than 90% of the diameter")
        pitch, op = v.get("propeller.pitch_type"), v.get("operating.mode")
        if pitch == "constant_speed" and op == "set_rpm":
            bad("operating.mode", "a constant-speed propeller needs 'full_throttle' or "
                "'set_power' (its RPM is the governor setting)")
        fine, coarse = v.get("propeller.fine_stop_deg"), v.get("propeller.coarse_stop_deg")
        if pitch == "constant_speed" and fine is not None and coarse is not None:
            if not 0.0 < fine < 89.0:
                bad("propeller.fine_stop_deg", "fine stop must be between 0 and 89 deg")
            if not fine < coarse < 89.5:
                bad("propeller.coarse_stop_deg", "coarse stop must be above the fine stop and below 89.5 deg")
        given = [k for k in AIRFRAME_KEYS if v.get(k) is not None]
        if 0 < len(given) < len(AIRFRAME_KEYS):
            for k in AIRFRAME_KEYS:
                if v.get(k) is None:
                    problems.append(Problem("MISSING", k, f"{FIELD_BY_KEY[k].label} is "
                                            "required once any airframe field is given"))
        e = v.get("airframe.oswald_e")
        if e is not None and e > 1.0:
            bad("airframe.oswald_e", "Oswald efficiency cannot exceed 1")
    if mode == "sizing":
        if v.get("flight.airspeed_ktas") is not None and v["flight.airspeed_ktas"] <= 0.0:
            bad("flight.airspeed_ktas", "sizing designs for a cruise point; airspeed must be > 0")
        m = v.get("sizing.tip_mach_limit")
        if m is not None and m >= 1.0:
            bad("sizing.tip_mach_limit", "tip Mach limit must be below 1 (subsonic tips)")
        dmax, s = v.get("sizing.max_diameter_in"), v.get("propeller.spinner_diameter_in")
        if dmax and s and s >= 0.45 * dmax:
            bad("propeller.spinner_diameter_in", "spinner must be under 45% of the maximum diameter")
    pp = v.get("operating.power_pct")
    if pp is not None and pp > 100.0:
        bad("operating.power_pct", "power above 100% of rated is not a continuous setting")
    tr, tt = v.get("propeller.root_thickness_pct"), v.get("propeller.tip_thickness_pct")
    for key, val in (("propeller.root_thickness_pct", tr), ("propeller.tip_thickness_pct", tt)):
        if val is not None and val >= 60.0:
            bad(key, "thickness ratio must be below 60%")
    alt = v.get("flight.altitude_ft")
    if alt is not None and alt > 104986.0:
        bad("flight.altitude_ft", "the ISA model used here ends at 104,986 ft (32 km)")
    crit = v.get("engine.critical_altitude_ft")
    if crit is not None and crit > 104986.0:
        bad("engine.critical_altitude_ft", "the ISA model used here ends at 104,986 ft (32 km)")


def _to_si(v: dict[str, Any]) -> dict[str, Any]:
    from .engine import BSFC_US_TO_SI
    from .units import FOOT, HORSEPOWER, INCH, KNOT

    LB = 0.45359237 * 9.80665
    conv = {
        "flight.altitude_ft": FOOT, "flight.airspeed_ktas": KNOT,
        "engine.rated_power_hp": HORSEPOWER, "engine.critical_altitude_ft": FOOT,
        "engine.bsfc_lb_hp_h": BSFC_US_TO_SI,
        "propeller.diameter_in": INCH, "propeller.spinner_diameter_in": INCH,
        "propeller.pitch_in": INCH, "propeller.root_thickness_pct": 0.01,
        "propeller.tip_thickness_pct": 0.01, "operating.power_pct": 0.01,
        "airframe.weight_lb": LB, "airframe.wing_area_ft2": FOOT ** 2,
        "airframe.wing_span_ft": FOOT, "sizing.max_diameter_in": INCH,
        "propeller.fine_stop_deg": math.pi / 180.0, "propeller.coarse_stop_deg": math.pi / 180.0,
    }
    return {k: (val * conv[k] if (k in conv and val is not None) else val)
            for k, val in v.items()}


def blank_case(mode: str | None = None) -> dict[str, Any]:
    """Every field, empty -- the starting point of a form or a template file."""
    out: dict[str, Any] = {f.key: None for f in FIELDS}
    if mode is not None:
        out["case.mode"] = mode
    return out


__all__ = ["Field", "FIELDS", "FIELD_BY_KEY", "GROUPS", "MODES", "PITCH_TYPES",
           "ASPIRATIONS", "OPERATING_MODES", "AIRFRAME_KEYS", "Problem", "Case",
           "CaseError", "parse_case", "required_keys", "blank_case"]
