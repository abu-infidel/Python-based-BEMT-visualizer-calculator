"""Inputs (no defaults, no caps) and the structured report format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from propwash.analysis import run
from propwash.case import FIELDS, CaseError, blank_case, parse_case, required_keys
from propwash.report import (MAGIC, Report, ReportFormatError, parse_report, table_records,
                             values)

REPO = Path(__file__).resolve().parent.parent


# -- inputs ---------------------------------------------------------------------

def test_blank_case_has_every_field_and_no_values():
    case = blank_case()
    assert set(case) == {f.key for f in FIELDS}
    assert all(v is None for v in case.values())


def test_blank_case_is_refused_with_every_missing_field_listed():
    with pytest.raises(CaseError) as err:
        parse_case(blank_case("analysis"))
    missing = {p.key for p in err.value.problems if p.code == "MISSING"}
    assert {"flight.altitude_ft", "engine.rated_power_hp", "propeller.diameter_in"} <= missing


def test_required_fields_follow_the_choices(c172_case):
    fixed = required_keys(c172_case)
    cs = required_keys(dict(c172_case, **{"propeller.pitch_type": "constant_speed"}))
    turbo = required_keys(dict(c172_case, **{"engine.aspiration": "turbocharged"}))
    assert "propeller.pitch_in" in fixed and "propeller.pitch_in" not in cs
    assert {"propeller.fine_stop_deg", "propeller.coarse_stop_deg"} <= set(cs)
    assert "engine.critical_altitude_ft" in turbo


@pytest.mark.parametrize("hp", [401.0, 1200.0, 3500.0])
def test_large_engines_are_accepted_with_a_warning_not_a_cap(c172_case, hp):
    case = parse_case(dict(c172_case, **{"engine.rated_power_hp": hp,
                                          "propeller.diameter_in": 140.0}))
    assert case.values["engine.rated_power_hp"] == hp
    if hp > 600.0:
        assert any(w.key == "engine.rated_power_hp" for w in case.warnings)


def test_typed_text_is_accepted(c172_case):
    case = parse_case({k: (f"{v:,}" if isinstance(v, float) else v)
                       for k, v in c172_case.items()})
    assert case.values["airframe.weight_lb"] == 2300.0


@pytest.mark.parametrize("key,bad", [("propeller.diameter_in", "-75"),
                                     ("propeller.diameter_in", "seventy"),
                                     ("propeller.blades", "2.5"),
                                     ("engine.aspiration", "diesel"),
                                     ("operating.power_pct", "130"),
                                     ("flight.altitude_ft", "200000")])
def test_meaningless_values_are_errors(c172_case, key, bad):
    with pytest.raises(CaseError) as err:
        parse_case(dict(c172_case, **{key: bad}))
    assert any(p.key == key for p in err.value.problems)


def test_airframe_is_all_or_nothing(c172_case):
    with pytest.raises(CaseError) as err:
        parse_case(dict(c172_case, **{"airframe.cd0": None}))
    assert any(p.key == "airframe.cd0" and p.code == "MISSING" for p in err.value.problems)
    no_airframe = {k: (None if k.startswith("airframe.") else v) for k, v in c172_case.items()}
    assert not parse_case(no_airframe).has_airframe


def test_unknown_fields_are_reported(c172_case):
    with pytest.raises(CaseError) as err:
        parse_case(dict(c172_case, **{"propeller.colour": "red"}))
    assert err.value.problems[0].code == "UNKNOWN_FIELD"


# -- report format -------------------------------------------------------------

@pytest.fixture(scope="module")
def static_report(c172_case):
    return run(dict(c172_case, **{"flight.airspeed_ktas": 0.0}))


def test_report_round_trips(static_report):
    text = static_report.to_text()
    assert text.startswith(MAGIC + "\n") and text.endswith("[END]\n")
    d = parse_report(text)
    assert d["integrity"]["ok"]
    assert values(d, "OPERATING_POINT")["status"] == "ok"
    rpm = values(d, "OPERATING_POINT")["engine_rpm"]
    assert rpm == pytest.approx(static_report.get("OPERATING_POINT", "engine_rpm"), rel=1e-5)


def test_nulls_carry_a_reason(static_report):
    d = parse_report(static_report.to_text())
    eff = d["sections"]["OPERATING_POINT"]["propulsive_efficiency"]
    assert eff["value"] is None and eff["code"] == "NA_STATIC"
    for sec in d["sections"].values():
        for item in sec.values():
            assert item["value"] is not None or item["code"]


def test_tables_parse_to_records(static_report):
    d = parse_report(static_report.to_text())
    span = table_records(d, "SPANWISE")
    assert len(span) == 60 and 0.0 < span[0]["r_over_r"] < span[-1]["r_over_r"] <= 1.0
    sweep = table_records(d, "SPEED_SWEEP")
    assert sweep[0]["airspeed_ktas"] == 0 and all("status" in r for r in sweep)


def test_edited_file_fails_the_checksum(static_report):
    text = static_report.to_text().replace('status = "ok"', 'status = "warning"', 1)
    with pytest.raises(ReportFormatError):
        parse_report(text)
    assert parse_report(text, verify=False)["integrity"]["ok"] is False


@pytest.mark.parametrize("damage", [lambda t: t.replace("[END]\n", ""),
                                    lambda t: "hello\n" + t,
                                    lambda t: t.replace("#PROPWASH-REPORT 1", "#PROPWASH-REPORT 9")])
def test_damaged_files_are_refused(static_report, damage):
    with pytest.raises(ReportFormatError):
        parse_report(damage(static_report.to_text()), verify=False)


def test_writer_escapes_strings_and_never_writes_nan():
    r = Report()
    r.put("META", "case_name", 'quote " and, comma')
    r.put("RESULTS", "x", float("nan"), "m")
    d = parse_report(r.to_text())
    assert d["sections"]["META"]["case_name"]["value"] == 'quote " and, comma'
    assert d["sections"]["RESULTS"]["x"] == {"value": None, "unit": "m", "code": "NOT_COMPUTED"}


def test_error_reports_are_still_valid_files():
    rep = run(blank_case("analysis"))
    d = parse_report(rep.to_text())
    assert values(d, "META")["status"] == "error"
    assert any(n["code"] == "MISSING" for n in d["notices"])


@pytest.mark.parametrize("name", ["example_analysis.txt", "example_sizing.txt",
                                  "example_error.txt"])
def test_documented_examples_parse(name):
    path = REPO / "docs" / "report-format" / name
    d = parse_report(path.read_text(encoding="utf-8"))
    assert d["integrity"]["ok"]


def test_json_view_matches_parser(static_report):
    assert json.loads(json.dumps(static_report.to_dict()))["format"]["version"] == 1
