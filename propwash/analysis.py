"""From a filled-in case to a structured report.

:func:`run` is the single entry point every front end uses -- the Colab app,
the desktop GUI, the command line and the tests -- so they cannot disagree.
It never returns a made-up number: anything that cannot be computed is
``None`` in the report with a reason code, and every limit that was hit is a
notice.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import numpy as np

from . import matching as mt
from .atmosphere import RHO0, AirState, density_altitude, isa
from .bemt.core import SolverOptions
from .case import FIELD_BY_KEY, FIELDS, Case, CaseError, parse_case
from .engine import MODEL_CONSTANTS, PistonEngine
from .geometry import AF_LOWER, BladeGeometry, full_size_blade
from .report import Report, Table
from .units import FOOT, HORSEPOWER, INCH, KNOT, LBF
from .version import __version__

G0 = 9.80665
LB = 0.45359237
FPM = FOOT / 60.0
GAL = 3.785411784e-3          # m^3 per US gallon
AVGAS_KG_PER_L = 0.72
SERVICE_CEILING_ROC = 100.0 * FPM

#: Solver settings used for every report (printed in [MODEL]).
OPTIONS = SolverOptions(n_elements=60, spacing="cosine", n_bisect=44, n_alpha_table=721,
                        tip_loss=True, hub_loss=True, reynolds_correction=True,
                        mach_correction=True)

STATUS_TEXT = {
    mt.OK: "Operating point found inside all limits.",
    mt.RPM_LIMITED: "At full throttle the propeller would overspeed the engine; the point is "
                    "shown at the RPM limit with the throttle reduced to hold it.",
    mt.WINDMILLING: "The airflow drives the propeller: it absorbs no power and makes drag.",
    mt.TOO_COARSE: "The propeller is too coarse: it holds the engine below a running speed.",
    mt.POWER_NOT_AVAILABLE: "The requested setting needs more power than the engine has here.",
    mt.ON_FINE_STOP: "The blades are on the fine pitch stop; RPM is below the governor setting.",
    mt.ON_COARSE_STOP: "The blades are on the coarse pitch stop; the governor cannot hold RPM.",
    mt.THRUST_NOT_REACHABLE: "No setting within the limits makes the thrust needed.",
    mt.BELOW_RUNNING_RANGE: "Less is needed than the propeller gives at the engine's lowest "
                            "running speed.",
}
STATUS_SEVERITY = {
    mt.OK: None, mt.RPM_LIMITED: "warning", mt.WINDMILLING: "warning",
    mt.TOO_COARSE: "error", mt.POWER_NOT_AVAILABLE: "error", mt.ON_FINE_STOP: "info",
    mt.ON_COARSE_STOP: "warning", mt.THRUST_NOT_REACHABLE: "warning",
    mt.BELOW_RUNNING_RANGE: "warning",
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(raw: dict[str, Any]) -> Report:
    """Validate ``raw`` and compute everything it allows.  Never raises on bad input."""
    report = Report()
    report.put("META", "format", "PROPWASH-REPORT")
    report.put("META", "format_version", 1)
    report.put("META", "generator", f"propwash {__version__}")
    report.put("META", "created_utc", _dt.datetime.now(_dt.timezone.utc)
               .strftime("%Y-%m-%dT%H:%M:%SZ"))
    try:
        case = parse_case(raw)
    except CaseError as exc:
        report.put("META", "case_name", exc.values.get("case.name") or None,
                   code=None if exc.values.get("case.name") else "NOT_GIVEN")
        report.put("META", "mode", exc.values.get("case.mode"),
                   code=None if exc.values.get("case.mode") else "NOT_GIVEN")
        _inputs(report, exc.values)
        for p in exc.problems:
            report.notice(p.code, "error", p.message, p.key)
        return report
    report.put("META", "case_name", case.values.get("case.name") or None,
               code=None if case.values.get("case.name") else "NOT_GIVEN")
    report.put("META", "mode", case.mode)
    _inputs(report, case.values)
    for w in case.warnings:
        report.notice(w.code, "warning", w.message, w.key)
    _model(report)

    air = isa(case.si["flight.altitude_ft"], case.si["flight.isa_deviation_c"])
    engine = PistonEngine(
        rated_power=case.si["engine.rated_power_hp"], rated_rpm=case.values["engine.rated_rpm"],
        gear_ratio=case.values["engine.gear_ratio"], aspiration=case.values["engine.aspiration"],
        critical_altitude=case.si.get("engine.critical_altitude_ft"),
        bsfc=case.si.get("engine.bsfc_lb_hp_h"))
    _atmosphere(report, air)
    _engine(report, engine, air)
    if engine.bsfc is None:
        report.notice("NO_BSFC", "info", "No BSFC was given, so fuel flow is not reported.",
                      "engine.bsfc_lb_hp_h")

    if case.mode == "analysis":
        _analysis(report, case, engine, air)
    else:
        _sizing(report, case, engine, air)
    return report


# ---------------------------------------------------------------------------
# Common sections
# ---------------------------------------------------------------------------

def _inputs(report: Report, values: dict[str, Any]) -> None:
    for f in FIELDS:
        v = values.get(f.key)
        unit = f.unit if f.kind in ("float", "int") else ""
        report.put("INPUTS", f.key, v, unit, code=None if v is not None else "NOT_GIVEN")


def _model(report: Report) -> None:
    s = "MODEL"
    report.put(s, "method", "blade element momentum theory, Prandtl tip and hub loss")
    report.put(s, "atmosphere", "ISA 1976 with temperature offset")
    report.put(s, "blade_elements", OPTIONS.n_elements)
    report.put(s, "element_spacing", OPTIONS.spacing)
    report.put(s, "reynolds_correction", OPTIONS.reynolds_correction)
    report.put(s, "compressibility_correction", OPTIONS.mach_correction)
    for k, v in MODEL_CONSTANTS.items():
        report.put(s, f"engine_{k}", v)
    report.put(s, "service_ceiling_climb_rate", 100.0, "ft/min")


def _atmosphere(report: Report, air: AirState) -> None:
    s = "ATMOSPHERE"
    report.put(s, "pressure_altitude_ft", air.altitude / FOOT, "ft")
    report.put(s, "density_altitude_ft", density_altitude(air) / FOOT, "ft")
    report.put(s, "temperature_c", air.temperature - 273.15, "degC")
    report.put(s, "pressure_pa", air.pressure, "Pa")
    report.put(s, "density_kg_m3", air.density, "kg/m^3")
    report.put(s, "density_ratio", air.density / RHO0, "-")
    report.put(s, "speed_of_sound_m_s", air.sound_speed, "m/s")
    report.put(s, "dynamic_viscosity_pa_s", air.viscosity, "Pa*s")


def _engine(report: Report, engine: PistonEngine, air: AirState) -> None:
    s = "ENGINE"
    report.put(s, "rated_power_hp", engine.rated_power_hp, "hp")
    report.put(s, "rated_power_kw", engine.rated_power / 1000.0, "kW")
    report.put(s, "rated_engine_rpm", engine.rated_rpm, "rpm")
    report.put(s, "gear_ratio", engine.gear_ratio, "-")
    report.put(s, "max_prop_rpm", engine.max_prop_rpm, "rpm")
    report.put(s, "min_running_prop_rpm", engine.min_prop_rpm, "rpm")
    report.put(s, "aspiration", engine.aspiration)
    report.put(s, "altitude_power_factor", engine.altitude_factor(air), "-")
    report.put(s, "full_throttle_power_at_rated_rpm_hp",
               float(engine.full_throttle_power(engine.max_prop_rpm, air)) / HORSEPOWER, "hp")


def _fuel(report: Report, section: str, engine: PistonEngine, power: float | None,
          valid: bool) -> None:
    flow = engine.fuel_flow(power) if (valid and power is not None) else None
    code = "NO_BSFC" if engine.bsfc is None else "NO_OPERATING_POINT"
    if flow is None:
        for k, u in (("fuel_flow_kg_h", "kg/h"), ("fuel_flow_lb_h", "lb/h"),
                     ("fuel_flow_gal_h", "US gal/h")):
            report.put(section, k, None, u, code)
        return
    report.put(section, "fuel_flow_kg_h", flow * 3600.0, "kg/h")
    report.put(section, "fuel_flow_lb_h", flow * 3600.0 / LB, "lb/h")
    report.put(section, "fuel_flow_gal_h", flow * 3600.0 / AVGAS_KG_PER_L / 1000.0 / GAL,
               "US gal/h")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class _Setting:
    """The selected power setting, applicable at any airspeed."""

    def __init__(self, case: Case, prop: mt.Propeller, engine: PistonEngine,
                 delta_max: float | None):
        self.case, self.prop, self.engine, self.delta_max = case, prop, engine, delta_max
        self.pitch = case.values["propeller.pitch_type"]
        self.mode = case.values["operating.mode"]
        rpm = case.values.get("operating.engine_rpm")
        self.prop_rpm = None if rpm is None else rpm / engine.gear_ratio
        pct = case.si.get("operating.power_pct")
        self.power = None if pct is None else pct * engine.rated_power

    def match(self, v, air: AirState, full_throttle: bool = False) -> mt.Match:
        mode = "full_throttle" if full_throttle else self.mode
        prop, engine = self.prop, self.engine
        if self.pitch == "fixed":
            if mode == "full_throttle":
                return mt.fixed_pitch(prop, engine, v, air, 1.0)
            if mode == "set_rpm":
                return mt.fixed_pitch_at_rpm(prop, engine, self.prop_rpm, v, air)
            return mt.fixed_pitch_at_power(prop, engine, self.power, v, air)
        power = (float(engine.full_throttle_power(self.prop_rpm, air))
                 if mode == "full_throttle" else self.power)
        return mt.constant_speed(prop, engine, self.prop_rpm, power, v, air, self.delta_max)

    def trim(self, thrust, v, air: AirState) -> mt.Match:
        if self.pitch == "fixed":
            return mt.thrust_trim_fixed(self.prop, self.engine, thrust, v, air)
        return mt.thrust_trim_constant_speed(self.prop, self.engine, self.prop_rpm, thrust,
                                             v, air, self.delta_max)

    def thrust(self, v, air: AirState, full_throttle: bool = False) -> np.ndarray:
        """Thrust at each speed; NaN where the setting has no valid point."""
        m = self.match(v, air, full_throttle)
        t = np.where(m.valid(), m.solution["thrust"], np.nan)
        return t


def build_geometry(case: Case) -> tuple[BladeGeometry, float | None]:
    v, si = case.values, case.si
    common = dict(diameter=si["propeller.diameter_in"], n_blades=v["propeller.blades"],
                  activity_factor=v["propeller.activity_factor"],
                  hub_ratio=si["propeller.spinner_diameter_in"] / si["propeller.diameter_in"],
                  root_airfoil=v["propeller.root_airfoil"],
                  tip_airfoil=v["propeller.tip_airfoil"],
                  thickness_root=si["propeller.root_thickness_pct"],
                  thickness_tip=si["propeller.tip_thickness_pct"],
                  planform=v["propeller.planform"], name=v.get("case.name") or "")
    if v["propeller.pitch_type"] == "fixed":
        return full_size_blade(pitch=si["propeller.pitch_in"], **common), None
    geom = full_size_blade(beta75_deg=v["propeller.fine_stop_deg"], **common)
    return geom, si["propeller.coarse_stop_deg"] - si["propeller.fine_stop_deg"]


def _analysis(report: Report, case: Case, engine: PistonEngine, air: AirState) -> None:
    geom, delta_max = build_geometry(case)
    prop = mt.Propeller(geom, OPTIONS)
    setting = _Setting(case, prop, engine, delta_max)
    v_op = case.si["flight.airspeed_ktas"]

    _propeller_section(report, case, geom)
    _check_rpm_limit(report, case, engine)

    m = setting.match([v_op], air)
    pt = m.at(0)
    _operating_point(report, case, engine, geom, air, pt)
    report.extras["geometry"] = geom
    if pt["status"] in mt.VALID_STATUSES:
        from dataclasses import replace

        from .bemt.core import OperatingPoint, solve_stations
        g_op = replace(geom, pitch_offset=geom.pitch_offset + pt["delta_pitch"])
        st = g_op.discretize(OPTIONS.n_elements, OPTIONS.spacing)
        report.extras["geometry"] = g_op
        report.extras["prop_rpm"] = pt["prop_rpm"]
        report.extras["result"] = solve_stations(
            st, OperatingPoint(rpm=pt["prop_rpm"], v_inf=v_op, air=air), OPTIONS)

    v_level = None
    if case.has_airframe:
        v_level = _airframe(report, case, setting, engine, air, pt)
    else:
        report.notice("AIRFRAME_NOT_GIVEN", "info",
                      "No airframe was given, so drag, climb, speed and ceiling results "
                      "are not computed.")
    _speed_sweep(report, case, setting, engine, geom, air, v_op, v_level)


def _propeller_section(report: Report, case: Case, geom: BladeGeometry) -> None:
    s = "PROPELLER"
    report.put(s, "diameter_m", geom.diameter, "m")
    report.put(s, "diameter_in", geom.diameter / INCH, "in")
    report.put(s, "blades", geom.n_blades)
    report.put(s, "hub_ratio", geom.hub_radius_frac, "-")
    report.put(s, "activity_factor_per_blade", geom.activity_factor(), "-")
    report.put(s, "total_activity_factor", geom.activity_factor() * geom.n_blades, "-")
    report.put(s, "activity_factor_lower_limit_r_over_r", AF_LOWER, "-")
    report.put(s, "mean_solidity", geom.mean_solidity(), "-")
    report.put(s, "max_chord_m", float(np.max(geom.chord_at(np.linspace(0, 1, 201))))
               * geom.radius, "m")
    report.put(s, "planform", case.values["propeller.planform"])
    report.put(s, "pitch_type", case.values["propeller.pitch_type"])
    beta = math.degrees(geom.beta75())
    if case.values["propeller.pitch_type"] == "fixed":
        report.put(s, "beta75_deg", beta, "deg")
        report.put(s, "geometric_pitch_in", geom.geometric_pitch() / INCH, "in")
        report.put(s, "pitch_to_diameter", geom.pitch_diameter_ratio(), "-")
    else:
        report.put(s, "fine_stop_beta75_deg", beta, "deg")
        report.put(s, "coarse_stop_beta75_deg", case.values["propeller.coarse_stop_deg"], "deg")
        report.put(s, "pitch_to_diameter_at_fine_stop", geom.pitch_diameter_ratio(), "-")
    report.put(s, "twist_law", "helical (constant geometric pitch)")


def _check_rpm_limit(report: Report, case: Case, engine: PistonEngine) -> None:
    rpm = case.values.get("operating.engine_rpm")
    if rpm is not None and rpm > engine.rated_rpm:
        report.notice("RPM_ABOVE_LIMIT", "warning",
                      f"Engine RPM {rpm:g} is above the rated/limit RPM {engine.rated_rpm:g}; "
                      "the torque curve is extrapolated.", "operating.engine_rpm")


def _status_notice(report: Report, status: str, where: str, key: str = "") -> None:
    sev = STATUS_SEVERITY.get(status)
    if sev:
        report.notice(status.upper(), sev, f"{where}: {STATUS_TEXT[status]}", key)


def _operating_point(report: Report, case: Case, engine: PistonEngine, geom: BladeGeometry,
                     air: AirState, pt: dict) -> None:
    s = "OPERATING_POINT"
    status = pt["status"]
    valid = status in mt.VALID_STATUSES
    na = None if valid else "NO_OPERATING_POINT"
    v = pt["v_inf"]
    report.put(s, "status", status)
    report.put(s, "status_message", STATUS_TEXT[status])
    report.put(s, "power_setting", case.values["operating.mode"])
    report.put(s, "airspeed_ktas", v / KNOT, "kt")
    report.put(s, "airspeed_m_s", v, "m/s")
    _status_notice(report, status, "Operating point")

    def put(key, value, unit, code=None):
        report.put(s, key, value if valid else None, unit, code or na)

    if status == mt.POWER_NOT_AVAILABLE:
        need, have = float(pt["power"]), float(pt["available_power"])
        report.put(s, "required_power_hp", need / HORSEPOWER, "hp", "NOT_DEFINED")
        report.put(s, "full_throttle_power_hp", have / HORSEPOWER, "hp", "NOT_DEFINED")
        if math.isfinite(need) and math.isfinite(have):
            report.notice("SHORTFALL", "info",
                          f"The setting needs {need / HORSEPOWER:.1f} hp; full throttle gives "
                          f"{have / HORSEPOWER:.1f} hp here.")
    rpm = pt["prop_rpm"]
    n = rpm / 60.0 if valid else math.nan
    put("engine_rpm", rpm * engine.gear_ratio, "rpm")
    put("prop_rpm", rpm, "rpm")
    put("throttle_pct", pt["throttle"] * 100.0, "%")
    power = float(pt["power"])
    put("shaft_power_hp", power / HORSEPOWER, "hp")
    put("shaft_power_kw", power / 1000.0, "kW")
    put("power_pct_rated", power / engine.rated_power * 100.0, "%")
    put("available_power_hp", float(pt["available_power"]) / HORSEPOWER, "hp")
    thrust = float(pt["thrust"])
    put("thrust_n", thrust, "N")
    put("thrust_lbf", thrust / LBF, "lbf")
    torque = float(pt["torque"])
    put("torque_nm", torque, "N*m")
    put("torque_lbf_ft", torque / (LBF * FOOT), "lbf*ft")
    put("advance_ratio_j", v / (n * geom.diameter) if valid and n > 0 else None, "-")
    if v > 0.0:
        eff = thrust * v / power if (valid and power > 0.0 and thrust > 0.0) else None
        put("propulsive_efficiency", eff, "-", None if eff is not None or not valid
            else "NOT_PRODUCING_THRUST")
        report.put(s, "figure_of_merit", None, "-", "NA_IN_FORWARD_FLIGHT")
    else:
        report.put(s, "propulsive_efficiency", None, "-", "NA_STATIC")
        fom = (thrust ** 1.5 / math.sqrt(2.0 * air.density * math.pi * geom.radius ** 2) / power
               if valid and power > 0.0 and thrust > 0.0 else None)
        put("figure_of_merit", fom, "-", None if fom is not None or not valid
            else "NOT_PRODUCING_THRUST")
    put("ct", float(pt["ct"]), "-")
    put("cp", float(pt["cp"]), "-")
    put("cq", float(pt["cq"]), "-")
    tip_speed = rpm * math.pi * geom.diameter / 60.0 if valid else None
    put("tip_speed_m_s", tip_speed, "m/s")
    put("helical_tip_mach", float(pt["tip_mach"]), "-")
    beta = math.degrees(geom.beta75() + pt["delta_pitch"])
    put("beta75_deg", beta, "deg")
    put("disk_loading_n_m2", float(pt["disk_loading"]), "N/m^2")
    put("disk_loading_lb_ft2", float(pt["disk_loading"]) / (LBF / FOOT ** 2), "lb/ft^2")
    put("thrust_per_power_lbf_hp", thrust / LBF / (power / HORSEPOWER)
        if valid and power > 0.0 else None, "lbf/hp",
        None if (valid and power > 0.0) or not valid else "NOT_ABSORBING_POWER")
    _fuel(report, s, engine, power if valid else None, valid)
    alpha = pt.get("alpha")
    stalled = None
    if valid and alpha is not None:
        stalled = _stalled_fraction(geom, pt)
    put("stalled_span_fraction", stalled, "-")
    put("converged_span_fraction", float(pt["converged_fraction"]), "-")

    if valid:
        tip_m = float(pt["tip_mach"])
        if tip_m > 1.0:
            report.notice("TIP_SUPERSONIC", "warning",
                          f"Helical tip Mach {tip_m:.2f} is supersonic; the section model "
                          "is not valid there and the result is unreliable.")
        elif tip_m > 0.88:
            report.notice("TIP_MACH_HIGH", "warning",
                          f"Helical tip Mach {tip_m:.2f}: compressibility losses are large "
                          "and noise is severe.")
        if float(pt["converged_fraction"]) < 0.9:
            report.notice("BEMT_PARTIAL_CONVERGENCE", "warning",
                          "More than 10% of blade elements had no momentum solution; "
                          "they were given zero induction.")
        if stalled is not None and stalled > 0.25 and v > 0.0:
            report.notice("BLADE_STALL", "warning",
                          f"{stalled * 100:.0f}% of the span is past stall in forward flight.")
        _spanwise(report, geom, pt)


def _stalled_fraction(geom: BladeGeometry, pt: dict) -> float:
    """Fraction of stations whose angle of attack is past their own stall angle."""
    st = geom.discretize(OPTIONS.n_elements, OPTIONS.spacing)
    stall = np.array([p.alpha_0 + p.max_lift() / p.cl_alpha for p in st.polars])
    return float(np.mean(np.asarray(pt["alpha"]) > stall))


def _spanwise(report: Report, geom: BladeGeometry, pt: dict) -> None:
    st = geom.discretize(OPTIONS.n_elements, OPTIONS.spacing)
    deg = 180.0 / math.pi
    twist = st.twist + pt["delta_pitch"]
    cols = [("r_over_r", "-"), ("radius_m", "m"), ("chord_m", "m"), ("thickness_ratio", "-"),
            ("blade_angle_deg", "deg"), ("inflow_angle_deg", "deg"), ("alpha_deg", "deg"),
            ("cl", "-"), ("cd", "-"), ("relative_speed_m_s", "m/s"), ("mach", "-"),
            ("reynolds", "-"), ("thrust_per_span_n_m", "N/m"), ("torque_per_span_nm_m", "N*m/m"),
            ("loss_factor", "-")]
    thickness = st.section_params()[0]
    rows = []
    for i in range(st.n):
        rows.append([st.x[i], st.r[i], st.chord[i], thickness[i], twist[i] * deg,
                     pt["phi"][i] * deg, pt["alpha"][i] * deg, pt["cl"][i], pt["cd"][i],
                     pt["w"][i], pt["mach"][i], pt["reynolds"][i], pt["dt_dr"][i],
                     pt["dq_dr"][i], pt["loss_factor"][i]])
    report.tables["SPANWISE"] = Table(cols, rows, "Blade state at the operating point, "
                                      "root to tip; per-span loads are for all blades.")


# ---------------------------------------------------------------------------
# Airframe performance
# ---------------------------------------------------------------------------

class _Airframe:
    def __init__(self, case: Case):
        si = case.si
        self.weight = si["airframe.weight_lb"]
        self.area = si["airframe.wing_area_ft2"]
        self.span = si["airframe.wing_span_ft"]
        self.cd0 = case.values["airframe.cd0"]
        self.e = case.values["airframe.oswald_e"]
        self.cl_max = case.values["airframe.cl_max"]
        self.ar = self.span ** 2 / self.area
        self.k = 1.0 / (math.pi * self.e * self.ar)

    def stall_speed(self, air: AirState) -> float:
        return math.sqrt(2.0 * self.weight / (air.density * self.area * self.cl_max))

    def drag(self, v, air: AirState):
        v = np.asarray(v, dtype=float)
        q = 0.5 * air.density * v * v
        with np.errstate(divide="ignore"):
            cl = self.weight / (q * self.area)
        return q * self.area * (self.cd0 + self.k * cl * cl)

    def speed_bound(self, power: float, air: AirState) -> float:
        """A speed no level flight can exceed: all power into parasite drag at eta = 1."""
        return (2.0 * max(power, 1.0) / (air.density * self.area * self.cd0)) ** (1.0 / 3.0)


def _climb_curve(setting: _Setting, af: _Airframe, air: AirState, full: bool,
                 n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    vs = af.stall_speed(air)
    p_max = setting.engine.rated_power * setting.engine.altitude_factor(air) * 1.2
    v = np.linspace(vs, max(af.speed_bound(p_max, air), 1.05 * vs), n)
    t = setting.thrust(v, air, full)
    roc = (t - af.drag(v, air)) * v / af.weight
    return v, roc


def _best_climb(setting: _Setting, af: _Airframe, air: AirState, full: bool):
    """Best rate of climb and its speed.  Returns (roc, v) or (None, None)."""
    v, roc = _climb_curve(setting, af, air, full)
    if not np.any(np.isfinite(roc)):
        return None, None
    r = np.where(np.isfinite(roc), roc, -np.inf)
    i = int(np.argmax(r))
    lo, hi = v[max(i - 1, 0)], v[min(i + 1, v.size - 1)]
    best_v, best_r = v[i], r[i]
    for _ in range(2):
        grid = np.linspace(lo, hi, 7)
        t = setting.thrust(grid, air, full)
        rr = (t - af.drag(grid, air)) * grid / af.weight
        rr = np.where(np.isfinite(rr), rr, -np.inf)
        j = int(np.argmax(rr))
        if rr[j] > best_r:
            best_v, best_r = grid[j], rr[j]
        step = grid[1] - grid[0]
        lo, hi = max(grid[j] - step, v[0]), grid[j] + step
    return float(best_r), float(best_v)


def _level_speed(setting: _Setting, af: _Airframe, air: AirState, full: bool):
    """Fastest speed where thrust equals drag.  Returns (v, status)."""
    v, roc = _climb_curve(setting, af, air, full, n=24)
    ok = np.isfinite(roc)
    if not np.any(ok):
        return None, "NO_VALID_POINTS"
    pos = ok & (roc > 0.0)
    if not np.any(pos):
        return None, "CANNOT_HOLD_LEVEL_FLIGHT"
    i = int(np.nonzero(pos)[0][-1])
    if i == v.size - 1:
        return None, "NOT_BRACKETED"
    lo, hi = float(v[i]), float(v[i + 1])
    for _ in range(3):
        grid = np.linspace(lo, hi, 9)
        t = setting.thrust(grid, air, full)
        ex = np.where(np.isfinite(t), t - af.drag(grid, air), -np.inf)
        j = int(np.nonzero(ex > 0.0)[0][-1]) if np.any(ex > 0.0) else 0
        lo, hi = float(grid[j]), float(grid[min(j + 1, grid.size - 1)])
    return 0.5 * (lo + hi), "ok"


def _ceiling(setting: _Setting, af: _Airframe, case: Case, target: float):
    """Altitude (m) where the best climb rate at full throttle falls to ``target``."""
    dt = case.si["flight.isa_deviation_c"]

    def g(h):
        roc, _ = _best_climb(setting, af, isa(h, dt), True)
        return -1.0 if roc is None else roc - target

    lo, g_lo = 0.0, g(0.0)
    if g_lo <= 0.0:
        return None, "CANNOT_CLIMB_AT_SEA_LEVEL"
    hi = None
    for h in (6000.0, 12000.0, 20000.0, 32000.0):
        gh = g(h)
        if gh <= 0.0:
            hi, g_hi = h, gh
            break
        lo, g_lo = h, gh
    if hi is None:
        return None, "ABOVE_MODEL_RANGE"
    # Illinois false position: the climb-rate curve is nearly straight.
    side = 0
    for _ in range(30):
        h = hi - g_hi * (hi - lo) / (g_hi - g_lo) if g_hi != g_lo else 0.5 * (lo + hi)
        gh = g(h)
        if abs(gh) < 0.01 or hi - lo < 3.0:
            return h, "ok"
        if gh > 0.0:
            lo, g_lo = h, gh
            if side == 1:
                g_hi *= 0.5
            side = 1
        else:
            hi, g_hi = h, gh
            if side == -1:
                g_lo *= 0.5
            side = -1
    return 0.5 * (lo + hi), "ok"


def _airframe(report: Report, case: Case, setting: _Setting, engine: PistonEngine,
              air: AirState, pt: dict) -> float | None:
    """Airframe results; returns the level speed at the setting (m/s) if there is one."""
    s = "AIRFRAME"
    af = _Airframe(case)
    sigma = air.density / RHO0
    v = pt["v_inf"]
    vs = af.stall_speed(air)
    report.put(s, "aspect_ratio", af.ar, "-")
    report.put(s, "induced_drag_factor_k", af.k, "-")
    report.put(s, "stall_speed_ktas", vs / KNOT, "kt")
    report.put(s, "stall_speed_keas", vs * math.sqrt(sigma) / KNOT, "kt")

    if v <= 0.0:
        for key, unit in (("lift_coefficient", "-"), ("drag_n", "N"), ("drag_lbf", "lbf"),
                          ("thrust_minus_drag_lbf", "lbf"), ("rate_of_climb_fpm", "ft/min")):
            report.put(s, key, None, unit, "NA_STATIC")
    else:
        q = 0.5 * air.density * v * v
        drag = float(af.drag(v, air))
        report.put(s, "lift_coefficient", af.weight / (q * af.area), "-")
        report.put(s, "drag_n", drag, "N")
        report.put(s, "drag_lbf", drag / LBF, "lbf")
        below = v < vs
        if below:
            report.notice("SPEED_BELOW_STALL", "warning",
                          f"The airspeed is below the stall speed ({vs / KNOT:.0f} KTAS); "
                          "climb and trim results at this speed are not computed.",
                          "flight.airspeed_ktas")
        valid = pt["status"] in mt.VALID_STATUSES
        code = "NA_BELOW_STALL_SPEED" if below else (None if valid else "NO_OPERATING_POINT")
        ex = float(pt["thrust"]) - drag if valid else None
        report.put(s, "thrust_minus_drag_lbf", None if below or ex is None else ex / LBF,
                   "lbf", code)
        report.put(s, "rate_of_climb_fpm",
                   None if below or ex is None else ex * v / af.weight / FPM, "ft/min", code)
        # level flight at this airspeed
        if below:
            for key, unit in (("level_trim_status", ""), ("level_trim_engine_rpm", "rpm"),
                              ("level_trim_power_hp", "hp"), ("level_trim_power_pct", "%"),
                              ("level_trim_beta75_deg", "deg"),
                              ("level_trim_efficiency", "-")):
                report.put(s, key, None, unit, "NA_BELOW_STALL_SPEED")
        else:
            trim = setting.trim(drag, [v], air).at(0)
            ok = trim["status"] in mt.VALID_STATUSES
            tc = None if ok else "NO_OPERATING_POINT"
            report.put(s, "level_trim_status", trim["status"])
            p = float(trim["power"])
            report.put(s, "level_trim_engine_rpm", trim["prop_rpm"] * engine.gear_ratio
                       if ok else None, "rpm", tc)
            report.put(s, "level_trim_power_hp", p / HORSEPOWER if ok else None, "hp", tc)
            report.put(s, "level_trim_power_pct", p / engine.rated_power * 100.0
                       if ok else None, "%", tc)
            report.put(s, "level_trim_beta75_deg",
                       math.degrees(setting.prop.geometry.beta75() + trim["delta_pitch"])
                       if ok else None, "deg", tc)
            report.put(s, "level_trim_efficiency", drag * v / p if ok and p > 0 else None,
                       "-", tc)
            _fuel_trim = engine.fuel_flow(p) if ok else None
            report.put(s, "level_trim_fuel_flow_gal_h",
                       None if _fuel_trim is None else
                       _fuel_trim * 3600.0 / AVGAS_KG_PER_L / 1000.0 / GAL, "US gal/h",
                       None if _fuel_trim is not None else
                       ("NO_BSFC" if engine.bsfc is None else "NO_OPERATING_POINT"))
            if not ok:
                _status_notice(report, trim["status"], "Level flight at the given airspeed")

    # at the selected power setting, this altitude
    vmax, why = _level_speed(setting, af, air, False)
    report.put(s, "level_speed_at_setting_ktas", None if vmax is None else vmax / KNOT, "kt",
               None if vmax is not None else why)
    report.put(s, "level_speed_at_setting_keas",
               None if vmax is None else vmax * math.sqrt(sigma) / KNOT, "kt",
               None if vmax is not None else why)
    roc, vy = _best_climb(setting, af, air, False)
    good = roc is not None and roc > 0.0
    report.put(s, "best_rate_of_climb_fpm", roc / FPM if roc is not None else None,
               "ft/min", None if roc is not None else "NO_VALID_POINTS")
    report.put(s, "best_climb_speed_ktas", vy / KNOT if good else None, "kt",
               None if good else "CANNOT_CLIMB")
    report.put(s, "best_climb_speed_keas", vy * math.sqrt(sigma) / KNOT if good else None,
               "kt", None if good else "CANNOT_CLIMB")
    # full throttle, any altitude
    for key, target in (("service_ceiling_ft", SERVICE_CEILING_ROC),
                        ("absolute_ceiling_ft", 0.0)):
        h, why = _ceiling(setting, af, case, target)
        report.put(s, key, None if h is None else h / FOOT, "ft",
                   None if h is not None else why)
    report.put(s, "ceilings_power_setting", "full_throttle")
    return vmax


# ---------------------------------------------------------------------------
# Speed sweep
# ---------------------------------------------------------------------------

def _speed_sweep(report: Report, case: Case, setting: _Setting, engine: PistonEngine,
                 geom: BladeGeometry, air: AirState, v_op: float,
                 v_level: float | None) -> None:
    if setting.pitch == "fixed":
        n_ref = setting.prop_rpm if setting.mode == "set_rpm" else engine.max_prop_rpm
        v_pitch = n_ref / 60.0 * geom.geometric_pitch()
    else:
        coarse = math.radians(case.values["propeller.coarse_stop_deg"])
        v_pitch = setting.prop_rpm / 60.0 * 2.0 * math.pi * 0.75 * geom.radius * math.tan(coarse)
    # Up to the propeller's pitch speed (where a fixed-pitch blade stops making
    # thrust), and past the operating and level-flight speeds.
    v_end = max(1.25 * v_op, v_pitch, 1.15 * (v_level or 0.0))
    v = np.linspace(0.0, v_end, 26)
    m = setting.match(v, air)
    af = _Airframe(case) if case.has_airframe else None
    cols = [("airspeed_ktas", "kt"), ("status", ""), ("engine_rpm", "rpm"),
            ("throttle_pct", "%"), ("beta75_deg", "deg"), ("thrust_lbf", "lbf"),
            ("shaft_power_hp", "hp"), ("propulsive_efficiency", "-"),
            ("advance_ratio_j", "-"), ("helical_tip_mach", "-")]
    if af:
        cols += [("drag_lbf", "lbf"), ("rate_of_climb_fpm", "ft/min")]
    rows = []
    for i in range(v.size):
        pt = m.at(i)
        ok = pt["status"] in mt.VALID_STATUSES
        thrust, power = float(pt["thrust"]), float(pt["power"])
        eff = (thrust * v[i] / power) if ok and v[i] > 0 and power > 0 and thrust > 0 else None
        n = pt["prop_rpm"] / 60.0
        row = [v[i] / KNOT, pt["status"],
               pt["prop_rpm"] * engine.gear_ratio if ok else None,
               pt["throttle"] * 100.0 if ok else None,
               math.degrees(geom.beta75() + pt["delta_pitch"]) if ok else None,
               thrust / LBF if ok else None, power / HORSEPOWER if ok else None, eff,
               v[i] / (n * geom.diameter) if ok and n > 0 else None,
               float(pt["tip_mach"]) if ok else None]
        if af:
            if v[i] >= af.stall_speed(air):
                d = float(af.drag(v[i], air))
                row += [d / LBF, (thrust - d) * v[i] / af.weight / FPM if ok else None]
            else:
                row += [None, None]
        rows.append([None if (isinstance(x, float) and not math.isfinite(x)) else x
                     for x in row])
    report.tables["SPEED_SWEEP"] = Table(
        cols, rows, f"Performance against airspeed at this altitude and the selected power "
        f"setting ({case.values['operating.mode']}). Nulls: no valid operating point, or "
        "below the stall speed for the airframe columns.")


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def _sizing(report: Report, case: Case, engine: PistonEngine, air: AirState) -> None:
    from .sizing import size_propeller

    v, si = case.values, case.si
    prop_rpm = v["operating.engine_rpm"] / engine.gear_ratio
    power = si["operating.power_pct"] * engine.rated_power
    available = float(engine.full_throttle_power(prop_rpm, air))
    speed = si["flight.airspeed_ktas"]
    _check_rpm_limit(report, case, engine)

    s = "DESIGN_POINT"
    report.put(s, "airspeed_ktas", speed / KNOT, "kt")
    report.put(s, "engine_rpm", v["operating.engine_rpm"], "rpm")
    report.put(s, "prop_rpm", prop_rpm, "rpm")
    report.put(s, "shaft_power_hp", power / HORSEPOWER, "hp")
    report.put(s, "available_power_hp", available / HORSEPOWER, "hp")
    if power > available * (1.0 + 1e-9):
        report.put(s, "status", mt.POWER_NOT_AVAILABLE)
        report.notice("POWER_NOT_AVAILABLE", "error",
                      f"The design point asks for {power / HORSEPOWER:.0f} hp but only "
                      f"{available / HORSEPOWER:.0f} hp is available at this altitude and RPM.",
                      "operating.power_pct")
        return
    report.put(s, "status", "ok")

    res = size_propeller(
        power=power, prop_rpm=prop_rpm, v_inf=speed, air=air,
        n_blades=v["propeller.blades"], airfoil=v["sizing.airfoil"],
        design_cl=v["sizing.design_cl"], spinner_diameter=si["propeller.spinner_diameter_in"],
        thickness_root=si["propeller.root_thickness_pct"],
        thickness_tip=si["propeller.tip_thickness_pct"],
        max_diameter=si["sizing.max_diameter_in"], tip_mach_limit=v["sizing.tip_mach_limit"])

    s = "SIZING"
    report.put(s, "status", res.status)
    report.put(s, "message", res.message)
    lim = res.limits
    report.put(s, "max_diameter_in", lim["max_diameter"] / INCH, "in")
    report.put(s, "tip_mach_limited_diameter_in",
               lim["tip_mach_diameter"] / INCH if "tip_mach_diameter" in lim else None, "in",
               None if "tip_mach_diameter" in lim else "INFEASIBLE")
    report.put(s, "searched_from_diameter_in",
               lim["diameter_low"] / INCH if "diameter_low" in lim else None, "in",
               None if "diameter_low" in lim else "INFEASIBLE")
    if res.best is None:
        report.notice("SIZING_INFEASIBLE", "error", res.message)
        _trade_table(report, res)
        return
    b, des = res.best, res.best.design
    geom = des.to_geometry(airfoil=v["sizing.airfoil"],
                           thickness_root=si["propeller.root_thickness_pct"],
                           thickness_tip=si["propeller.tip_thickness_pct"])
    report.put(s, "diameter_in", b.diameter / INCH, "in")
    report.put(s, "diameter_m", b.diameter, "m")
    report.put(s, "blades", v["propeller.blades"])
    report.put(s, "hub_ratio", b.hub_ratio, "-")
    report.put(s, "activity_factor_per_blade", geom.activity_factor(), "-")
    report.put(s, "beta75_deg", math.degrees(geom.beta75()), "deg")
    report.put(s, "geometric_pitch_at_75_in", geom.geometric_pitch() / INCH, "in")
    report.put(s, "pitch_to_diameter", geom.pitch_diameter_ratio(), "-")
    report.put(s, "helical_tip_mach", b.tip_mach, "-")
    report.put(s, "design_efficiency", b.efficiency, "-")
    report.put(s, "momentum_limit_efficiency", b.froude_efficiency, "-")
    report.put(s, "design_thrust_lbf", b.thrust / LBF, "lbf")
    chk = res.verification
    report.extras.update(geometry=chk["geometry"], result=chk["result"], prop_rpm=prop_rpm)
    report.put(s, "bemt_thrust_lbf", chk["bemt_thrust"] / LBF, "lbf")
    report.put(s, "bemt_power_hp", chk["bemt_power"] / HORSEPOWER, "hp")
    report.put(s, "bemt_efficiency", chk["bemt_efficiency"], "-")
    report.put(s, "bemt_thrust_error_pct", chk["thrust_error"] * 100.0, "%")
    report.put(s, "bemt_power_error_pct", chk["power_error"] * 100.0, "%")
    report.put(s, "design_iterations", des.iterations)
    if abs(chk["power_error"]) > 0.05 or abs(chk["thrust_error"]) > 0.05:
        report.notice("DESIGN_CHECK_DISAGREES", "warning",
                      "The BEMT check differs from the design method by more than 5%; the "
                      "design is outside the method's comfortable range.")
    if b.tip_mach > 0.88:
        report.notice("TIP_MACH_HIGH", "warning",
                      f"The chosen propeller runs a helical tip Mach of {b.tip_mach:.2f}.")
    _trade_table(report, res)
    deg = 180.0 / math.pi
    rows = []
    thick = geom.thickness_at(des.x)
    for i in range(des.x.size):
        rows.append([des.x[i], des.x[i] * des.radius, des.chord[i], des.chord[i] / INCH,
                     des.twist[i] * deg, des.phi[i] * deg, thick[i], des.reynolds[i],
                     float(des.meta["mach"][i])])
    report.tables["BLADE_DESIGN"] = Table(
        [("r_over_r", "-"), ("radius_m", "m"), ("chord_m", "m"), ("chord_in", "in"),
         ("blade_angle_deg", "deg"), ("inflow_angle_deg", "deg"), ("thickness_ratio", "-"),
         ("reynolds", "-"), ("mach", "-")], rows,
        "Minimum-induced-loss blade at the chosen diameter, hub to tip.")


def _trade_table(report: Report, res) -> None:
    rows = []
    for c in res.candidates:
        fin = lambda x: x if (x is not None and math.isfinite(x)) else None  # noqa: E731
        rows.append([c.diameter / INCH, c.diameter, c.tip_mach, c.status.split(":")[0],
                     fin(c.efficiency), fin(c.froude_efficiency),
                     fin(c.thrust / LBF if math.isfinite(c.thrust) else math.nan),
                     fin(c.activity_factor), fin(c.beta75_deg)])
    report.tables["DIAMETER_TRADE"] = Table(
        [("diameter_in", "in"), ("diameter_m", "m"), ("helical_tip_mach", "-"),
         ("status", ""), ("design_efficiency", "-"), ("momentum_limit_efficiency", "-"),
         ("thrust_lbf", "lbf"), ("activity_factor", "-"), ("beta75_deg", "deg")], rows,
        "Each row is the best (minimum-induced-loss) blade for that diameter.")


def field_help() -> list[dict[str, Any]]:
    """The input specification as plain data (for templates and front ends)."""
    return [{"key": f.key, "label": f.label, "unit": f.unit, "kind": f.kind,
             "group": f.group, "choices": list(f.choices), "help": f.help,
             "typical": list(f.typical) if f.typical else None}
            for f in FIELD_BY_KEY.values()]


__all__ = ["run", "build_geometry", "field_help", "OPTIONS", "STATUS_TEXT"]
