"""End-to-end calculations: every mode, extreme inputs, and no silent zeros."""

from __future__ import annotations

import math

import pytest

from propwash.analysis import run
from propwash.report import parse_report, values


def _numbers(report):
    """Every non-null number in every section."""
    d = parse_report(report.to_text())
    for name, sec in d["sections"].items():
        for key, item in sec.items():
            if isinstance(item["value"], (int, float)) and not isinstance(item["value"], bool):
                yield name, key, item["value"]


def _assert_sane(report):
    for name, key, v in _numbers(report):
        assert math.isfinite(v), f"{name}.{key} = {v}"
    op = report.sections.get("OPERATING_POINT", {})
    if op and op["status"].value in ("ok", "rpm_limited", "on_fine_stop", "on_coarse_stop"):
        for key in ("engine_rpm", "thrust_n", "shaft_power_hp", "torque_nm"):
            assert op[key].value not in (None, 0, 0.0), key


@pytest.mark.parametrize("over", [
    {},                                                            # static
    {"flight.airspeed_ktas": 100.0, "flight.altitude_ft": 6000.0},
    {"operating.mode": "set_rpm", "operating.engine_rpm": 2400.0,
     "flight.airspeed_ktas": 105.0, "flight.altitude_ft": 4000.0},
    {"operating.mode": "set_power", "operating.power_pct": 65.0,
     "flight.airspeed_ktas": 110.0, "flight.altitude_ft": 8000.0},
])
def test_fixed_pitch_modes(c172_case, over):
    no_airframe = {k: (None if k.startswith("airframe.") else v) for k, v in c172_case.items()}
    rep = run(dict(no_airframe, **over))
    assert rep.status in ("ok", "warning")
    _assert_sane(rep)


def _cs_case(c172_case, **over):
    case = {k: (None if k.startswith("airframe.") else v) for k, v in c172_case.items()}
    case.update({"engine.rated_power_hp": 310.0, "engine.rated_rpm": 2700.0,
                 "propeller.diameter_in": 80.0, "propeller.blades": 3,
                 "propeller.tip_airfoil": "naca_16",
                 "propeller.activity_factor": 105.0, "propeller.planform": "paddle",
                 "propeller.pitch_type": "constant_speed", "propeller.pitch_in": None,
                 "propeller.fine_stop_deg": 13.0, "propeller.coarse_stop_deg": 38.0,
                 "propeller.spinner_diameter_in": 13.0, "operating.engine_rpm": 2400.0,
                 "operating.mode": "set_power", "operating.power_pct": 65.0,
                 "flight.altitude_ft": 8000.0, "flight.airspeed_ktas": 165.0})
    case.update(over)
    return case


def test_constant_speed_cruise(c172_case):
    rep = run(_cs_case(c172_case))
    op = values(parse_report(rep.to_text()), "OPERATING_POINT")
    assert op["status"] == "ok"
    assert op["power_pct_rated"] == pytest.approx(65.0, rel=3e-3)
    assert 13.0 < op["beta75_deg"] < 38.0
    assert 0.80 < op["propulsive_efficiency"] < 0.90
    _assert_sane(rep)


def test_constant_speed_static_reports_the_fine_stop(c172_case):
    rep = run(_cs_case(c172_case, **{"flight.airspeed_ktas": 0.0, "flight.altitude_ft": 0.0,
                                     "operating.mode": "full_throttle"}))
    op = values(parse_report(rep.to_text()), "OPERATING_POINT")
    assert op["status"] in ("ok", "on_fine_stop")
    _assert_sane(rep)


def test_turbocharged_holds_power_at_altitude(c172_case):
    case = _cs_case(c172_case, **{"engine.aspiration": "turbocharged",
                                  "engine.critical_altitude_ft": 18000.0,
                                  "flight.altitude_ft": 16000.0, "operating.power_pct": 80.0,
                                  "flight.airspeed_ktas": 180.0})
    assert values(parse_report(run(case).to_text()), "OPERATING_POINT")["status"] == "ok"


def test_unreachable_power_is_an_error_with_the_shortfall(c172_case):
    no_airframe = {k: (None if k.startswith("airframe.") else v) for k, v in c172_case.items()}
    rep = run(dict(no_airframe, **{"operating.mode": "set_power", "operating.power_pct": 95.0,
                                   "flight.altitude_ft": 12000.0,
                                   "flight.airspeed_ktas": 100.0}))
    d = parse_report(rep.to_text())
    op = values(d, "OPERATING_POINT")
    assert op["status"] == "power_not_available" and op["thrust_n"] is None
    assert d["sections"]["OPERATING_POINT"]["thrust_n"]["code"] == "NO_OPERATING_POINT"
    assert rep.status == "error"


@pytest.mark.parametrize("over", [
    {"engine.rated_power_hp": 1500.0, "engine.rated_rpm": 2000.0,
     "propeller.diameter_in": 156.0, "propeller.blades": 4, "propeller.pitch_in": 140.0,
     "propeller.activity_factor": 140.0, "flight.airspeed_ktas": 200.0},     # big
    {"engine.rated_power_hp": 45.0, "engine.rated_rpm": 3300.0, "propeller.diameter_in": 54.0,
     "propeller.pitch_in": 30.0, "flight.airspeed_ktas": 60.0},             # small
    {"engine.rated_power_hp": 100.0, "engine.rated_rpm": 5800.0, "engine.gear_ratio": 2.43,
     "propeller.diameter_in": 68.0, "propeller.pitch_in": 52.0},            # geared
    {"propeller.pitch_in": 140.0, "flight.airspeed_ktas": 0.0},             # absurdly coarse
    {"propeller.pitch_in": 20.0, "flight.airspeed_ktas": 150.0},            # absurdly fine
    {"flight.altitude_ft": 30000.0, "flight.isa_deviation_c": 25.0},        # thin hot air
    {"flight.altitude_ft": -1000.0, "flight.isa_deviation_c": -30.0},       # dense cold air
])
def test_extreme_inputs_never_crash_or_invent_numbers(c172_case, over):
    rep = run(dict(c172_case, **over))
    _assert_sane(rep)
    d = parse_report(rep.to_text())
    assert d["integrity"]["ok"]
    assert values(d, "OPERATING_POINT")["status"]


def test_airframe_results(c172_case):
    rep = run(dict(c172_case, **{"flight.airspeed_ktas": 74.0}))
    af = values(parse_report(rep.to_text()), "AIRFRAME")
    assert af["level_speed_at_setting_ktas"] == pytest.approx(125.0, rel=0.05)
    assert 600.0 < af["best_rate_of_climb_fpm"] < 1000.0
    assert af["service_ceiling_ft"] < af["absolute_ceiling_ft"]
    assert af["level_trim_status"] == "ok"


def test_sizing_mode(c172_case):
    case = {k: None for k in c172_case}
    case.update({"case.mode": "sizing", "flight.altitude_ft": 8000.0,
                 "flight.isa_deviation_c": 0.0, "flight.airspeed_ktas": 122.0,
                 "engine.rated_power_hp": 160.0, "engine.rated_rpm": 2700.0,
                 "engine.gear_ratio": 1.0, "engine.aspiration": "normal",
                 "propeller.blades": 2, "propeller.spinner_diameter_in": 9.0,
                 "propeller.root_thickness_pct": 20.0, "propeller.tip_thickness_pct": 7.0,
                 "operating.engine_rpm": 2650.0, "operating.power_pct": 75.0,
                 "sizing.max_diameter_in": 76.0, "sizing.tip_mach_limit": 0.88,
                 "sizing.design_cl": 0.5, "sizing.airfoil": "clark_y"})
    d = parse_report(run(case).to_text())
    sz = values(d, "SIZING")
    assert sz["status"] == "ok" and 60.0 < sz["diameter_in"] <= 76.0
    assert abs(sz["bemt_power_error_pct"]) < 5.0
    assert len(d["tables"]["DIAMETER_TRADE"]["rows"]) == 21

    impossible = dict(case, **{"sizing.tip_mach_limit": 0.15})
    d = parse_report(run(impossible).to_text())
    assert values(d, "SIZING")["status"] == "infeasible"
