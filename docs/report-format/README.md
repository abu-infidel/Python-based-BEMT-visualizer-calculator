# Propwash report format (`PROPWASH-REPORT`, version 1)

Propwash writes its results, when you ask it to (**Export .txt** in either GUI,
or `propwash run case.json -o report.txt`), as a UTF-8 text file in the format
specified here. The format is meant to be read by other programs, including
LLM-based tools, without guesswork:

- every value has an explicit unit;
- a missing value is `null` together with a code that says why;
- tables are plain CSV between two marker lines;
- the file carries a checksum, so a damaged or hand-edited file is detected;
- the reference reader is ~150 lines of standard-library Python
  ([`propwash/report.py`](../../propwash/report.py), `parse_report`).

Three complete example files are in this folder:
[`example_analysis.txt`](example_analysis.txt) (fixed-pitch propeller with an
airframe), [`example_sizing.txt`](example_sizing.txt) (sizing mode) and
[`example_error.txt`](example_error.txt) (an incomplete form). Regenerate them
with `python docs/report-format/make_examples.py`.

---

## 1. Grammar

The file is a sequence of lines separated by `\n` (LF). In order:

```
#PROPWASH-REPORT 1                 <- magic line, always first; "1" is the major version
# comment lines may appear anywhere outside tables, and inside tables before the header
                                    <- blank lines are ignored
[SECTION_NAME]                     <- starts a key/value section
key = value [unit] {CODE}          <- one value per line
[TABLE.NAME]                       <- starts a table
# optional comment lines describing the table
col_a[unit],col_b[unit],...        <- header row: name[unit] cells
1.25,"ok",null                     <- data rows: comma-separated values
[END_TABLE]                        <- ends the table
[NOTICES]                          <- messages about the calculation (section 5)
[INTEGRITY]
body_sha256 = "<64 hex chars>"     <- checksum (section 6)
[END]                              <- last line; nothing but blank lines may follow
```

### Key/value lines

```
key = value
key = value [unit]
key = null [unit] {CODE}
```

| Part | Rule |
|---|---|
| `key` | lower case, `[a-z][a-z0-9_]*`, may contain dots (`flight.altitude_ft`). Unique within its section. |
| `value` | a JSON scalar: a number (`2700`, `0.0563213`, `1.731365e-05`), a double-quoted JSON string (`"ok"`, with `\"` and `\\` escapes), `true`, `false` or `null`. Numbers never use thousands separators. A number written without a decimal point (`2700`) is still just a number. |
| `[unit]` | optional, in square brackets after one space. Absent for strings and booleans. `[-]` means dimensionless. Units are listed in section 7. |
| `{CODE}` | present only when the value is `null`, in braces after one space: why the value is not available (section 4). |

A regular expression that matches every key/value line:

```
^(?P<key>[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*) = (?P<value>null|true|false|-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|"(?:[^"\\]|\\.)*")(?: \[(?P<unit>[^\]]*)\])?(?: \{(?P<code>[A-Z][A-Z0-9_]*)\})?$
```

### Tables

- `[TABLE.NAME]` opens a table. The first non-comment line is the header.
  Each header cell is `name[unit]`, and `[]` is an empty unit, used for text
  columns.
- Each following line until `[END_TABLE]` is one row. Cells are the same JSON
  scalars as values. They are separated by commas, and a comma inside a quoted
  string does not separate cells. Every row has as many cells as the header.
- `null` in a table cell means "no value for this row"; the table's comment and
  the row's `status` column (where there is one) say why.

### Section names

Upper case, `[A-Z][A-Z0-9_]*`. `TABLE.` is a reserved prefix. Sections appear
in the order listed in section 3, and a section that does not apply to the
calculation is omitted.

---

## 2. How to read a file safely

1. Check that the first line starts with `#PROPWASH-REPORT `. Read the major
   version after it. Refuse a version higher than the one you support.
2. Verify the checksum (section 6). If it fails, the file was edited or
   truncated; decide whether you still want it.
3. Read `META.status`. If it is `"error"`, there are no results: read
   `[NOTICES]` and stop.
4. Read the sections you need by key. **Ignore keys and sections you do not
   know.** Later minor revisions may add keys, sections, table columns and
   codes, but they will not rename or remove any (see section 8).
5. Before you use any number from `OPERATING_POINT`, check
   `OPERATING_POINT.status` (section 5.2). Before you use a table row, check its
   `status` column.

Python, using the reference reader:

```python
from propwash.report import parse_report, values, table_records

data = parse_report(open("report.txt", encoding="utf-8").read())   # verifies the checksum
if values(data, "META")["status"] == "error":
    raise RuntimeError([n["message"] for n in data["notices"]])
op = values(data, "OPERATING_POINT")            # {"status": "ok", "thrust_n": 1318.56, ...}
sweep = table_records(data, "SPEED_SWEEP")      # [{"airspeed_ktas": 0, "status": "ok", ...}, ...]
```

`parse_report` returns:

```json
{
  "format":   {"name": "PROPWASH-REPORT", "version": 1},
  "sections": {"OPERATING_POINT": {"thrust_n": {"value": 1318.56, "unit": "N", "code": null}, "...": "..."}},
  "tables":   {"SPANWISE": {"columns": [{"name": "r_over_r", "unit": "-"}], "rows": [[0.1201]]}},
  "notices":  [{"code": "NO_BSFC", "severity": "info", "field": "engine.bsfc_lb_hp_h", "message": "..."}],
  "integrity": {"ok": true, "expected": "...", "actual": "..."}
}
```

`propwash parse report.txt --json` prints the same structure, so a program in
any language can read the file through the command line.

A minimal reader without Propwash (key/value sections only):

```python
import json, re
LINE = re.compile(r'^([a-z][a-z0-9_.]*) = (null|true|false|"(?:[^"\\]|\\.)*"|\S+)(?: \[([^\]]*)\])?(?: \{([A-Z0-9_]+)\})?$')
sections, current, in_table = {}, None, False
for line in open("report.txt", encoding="utf-8").read().split("\n"):
    if line.startswith("[TABLE."): in_table = True; continue
    if line == "[END_TABLE]": in_table = False; continue
    if in_table or not line or line.startswith("#"): continue
    if line.startswith("["): current = line.strip("[]"); sections[current] = {}; continue
    key, raw, unit, code = LINE.match(line).groups()
    sections[current][key] = (json.loads(raw), unit, code)
```

---

## 3. Sections

The sections appear in this order. *Mode* says which calculation mode
(`META.mode`) writes the section.

| Section | Mode | Contents |
|---|---|---|
| `META` | all | format, generator, time, case name, mode, overall status |
| `INPUTS` | all | every input field exactly as validated, `null {NOT_GIVEN}` where blank |
| `MODEL` | both, not on input errors | solver settings and model constants used |
| `ATMOSPHERE` | both | the air at the operating (or design) point |
| `ENGINE` | both | engine description and its power at this altitude |
| `PROPELLER` | analysis | the propeller as built from the inputs |
| `OPERATING_POINT` | analysis | the solved operating point |
| `AIRFRAME` | analysis, only when an airframe was given | drag, climb, level-flight trim, speeds, ceilings |
| `DESIGN_POINT` | sizing | the cruise point being designed for |
| `SIZING` | sizing | the chosen propeller and its independent BEMT check |
| `TABLE.SPANWISE` | analysis, valid operating point | blade state from root to tip |
| `TABLE.SPEED_SWEEP` | analysis | performance against airspeed at the selected power setting |
| `TABLE.DIAMETER_TRADE` | sizing | every candidate diameter that was designed |
| `TABLE.BLADE_DESIGN` | sizing, feasible | the optimum blade's chord and twist |
| `NOTICES` | all | errors, warnings and information (section 5) |
| `INTEGRITY` | all | checksum |

### META

| key | type | meaning |
|---|---|---|
| `format` | string | always `"PROPWASH-REPORT"` |
| `format_version` | integer | major version, same as the magic line |
| `generator` | string | `"propwash <version>"` |
| `created_utc` | string | ISO 8601, `YYYY-MM-DDThh:mm:ssZ` |
| `case_name` | string or null | free text from the input form |
| `mode` | string or null | `"analysis"` or `"sizing"` |
| `status` | string | `"ok"`, `"warning"` (results valid, read the notices) or `"error"` (no usable results, or the requested condition cannot exist) |
| `notice_count` | integer | number of notices |

### INPUTS

One line per input field, keyed by the field name, in the units the form uses:
`flight.altitude_ft [ft]`, `engine.rated_power_hp [hp]`, `propeller.diameter_in [in]`, and so on.
`propwash fields` prints every field with its unit, type, choices and meaning,
and `propwash template` writes a blank input file (section 9).

### MODEL

`method`, `atmosphere`, `blade_elements`, `element_spacing`,
`reynolds_correction`, `compressibility_correction`,
`engine_peak_torque_fraction`, `engine_torque_droop`,
`engine_gagg_farrar_constant`, `engine_min_running_fraction`,
`service_ceiling_climb_rate [ft/min]`. These numbers belong to the model, not
the aircraft, and are printed so that a result can be reproduced.

### ATMOSPHERE

`pressure_altitude_ft [ft]`, `density_altitude_ft [ft]`, `temperature_c [degC]`,
`pressure_pa [Pa]`, `density_kg_m3 [kg/m^3]`, `density_ratio [-]` (ρ/ρ₀),
`speed_of_sound_m_s [m/s]`, `dynamic_viscosity_pa_s [Pa*s]`.

### ENGINE

| key | unit | meaning |
|---|---|---|
| `rated_power_hp`, `rated_power_kw` | hp, kW | rated power at sea level |
| `rated_engine_rpm` | rpm | crankshaft RPM at rated power; also the RPM limit |
| `gear_ratio` | - | crankshaft revolutions per propeller revolution |
| `max_prop_rpm` | rpm | propeller RPM at the engine limit |
| `min_running_prop_rpm` | rpm | lowest propeller RPM the matching treats as a running engine |
| `aspiration` | | `"normal"` or `"turbocharged"` |
| `altitude_power_factor` | - | full-throttle power here / rated power |
| `full_throttle_power_at_rated_rpm_hp` | hp | what the engine can give here at its RPM limit |

### PROPELLER (analysis)

`diameter_m [m]`, `diameter_in [in]`, `blades`, `hub_ratio [-]`,
`activity_factor_per_blade [-]` (as built, integrated from r/R 0.15 to 1),
`total_activity_factor [-]`, `activity_factor_lower_limit_r_over_r [-]`,
`mean_solidity [-]`, `max_chord_m [m]`, `planform`, `pitch_type`, `twist_law`.
Fixed pitch adds `beta75_deg [deg]` (the blade angle at 0.75 R), `geometric_pitch_in [in]` and
`pitch_to_diameter [-]`. Constant speed adds `fine_stop_beta75_deg [deg]`,
`coarse_stop_beta75_deg [deg]` and `pitch_to_diameter_at_fine_stop [-]`.

### OPERATING_POINT (analysis)

| key | unit | meaning |
|---|---|---|
| `status` | | see section 5.2. **Check this first.** |
| `status_message` | | one sentence explaining the status |
| `power_setting` | | `"full_throttle"`, `"set_rpm"` or `"set_power"` |
| `airspeed_ktas`, `airspeed_m_s` | kt, m/s | true airspeed |
| `engine_rpm`, `prop_rpm` | rpm | crankshaft and propeller speed |
| `throttle_pct` | % | share of full-throttle torque in use |
| `shaft_power_hp`, `shaft_power_kw` | hp, kW | power absorbed by the propeller |
| `power_pct_rated` | % | shaft power / rated power |
| `available_power_hp` | hp | full-throttle power at this RPM and altitude |
| `required_power_hp`, `full_throttle_power_hp` | hp | only when `status` is `power_not_available`: what the setting needs, and the most the engine gives |
| `thrust_n`, `thrust_lbf` | N, lbf | propeller thrust |
| `torque_nm`, `torque_lbf_ft` | N*m, lbf*ft | shaft torque |
| `advance_ratio_j` | - | J = V/(nD) |
| `propulsive_efficiency` | - | T·V/P; `null {NA_STATIC}` at zero airspeed |
| `figure_of_merit` | - | static only; `null {NA_IN_FORWARD_FLIGHT}` otherwise |
| `ct`, `cp`, `cq` | - | T/(ρn²D⁴), P/(ρn³D⁵), Q/(ρn²D⁵), with n in rev/s |
| `tip_speed_m_s` | m/s | rotational tip speed |
| `helical_tip_mach` | - | tip Mach including airspeed |
| `beta75_deg` | deg | blade angle at 0.75 R at this point (it moves for constant speed) |
| `disk_loading_n_m2`, `disk_loading_lb_ft2` | N/m^2, lb/ft^2 | thrust per disk area |
| `thrust_per_power_lbf_hp` | lbf/hp | |
| `fuel_flow_kg_h`, `fuel_flow_lb_h`, `fuel_flow_gal_h` | kg/h, lb/h, US gal/h | BSFC × shaft power; `null {NO_BSFC}` without a BSFC; gallons of avgas at 0.72 kg/L |
| `stalled_span_fraction` | - | share of blade stations past their stall angle |
| `converged_span_fraction` | - | share of stations with a momentum-theory solution |

### AIRFRAME (analysis, only when all six airframe inputs are given)

| key | unit | meaning |
|---|---|---|
| `aspect_ratio`, `induced_drag_factor_k` | - | AR = b²/S, k = 1/(π e AR) |
| `stall_speed_ktas`, `stall_speed_keas` | kt | at this weight and altitude, from `airframe.cl_max` |
| `lift_coefficient`, `drag_n`, `drag_lbf` | -, N, lbf | at the operating airspeed (`null {NA_STATIC}` when static) |
| `thrust_minus_drag_lbf`, `rate_of_climb_fpm` | lbf, ft/min | at the operating point |
| `level_trim_status` | | status (section 5.2) of the level-flight trim at the operating airspeed |
| `level_trim_engine_rpm`, `level_trim_power_hp`, `level_trim_power_pct`, `level_trim_beta75_deg`, `level_trim_efficiency`, `level_trim_fuel_flow_gal_h` | | what holding level flight at this airspeed takes |
| `level_speed_at_setting_ktas`, `level_speed_at_setting_keas` | kt | fastest level-flight speed at the selected power setting and this altitude |
| `best_rate_of_climb_fpm`, `best_climb_speed_ktas`, `best_climb_speed_keas` | ft/min, kt | best climb at the selected power setting |
| `service_ceiling_ft`, `absolute_ceiling_ft` | ft | where the best climb falls to 100 ft/min and to 0, at full throttle |
| `ceilings_power_setting` | | always `"full_throttle"` |

KEAS is equivalent airspeed (TAS × √σ), the closest the model gets to indicated airspeed.

### DESIGN_POINT (sizing)

`airspeed_ktas [kt]`, `engine_rpm [rpm]`, `prop_rpm [rpm]`, `shaft_power_hp [hp]`,
`available_power_hp [hp]`, `status` (`"ok"` or `"power_not_available"`).

### SIZING

| key | unit | meaning |
|---|---|---|
| `status` | | `"ok"` or `"infeasible"` |
| `message` | | why this diameter, or why nothing works |
| `max_diameter_in` | in | the constraint from the input |
| `tip_mach_limited_diameter_in` | in | the diameter at which the helical tip Mach reaches its limit |
| `searched_from_diameter_in` | in | the smallest candidate designed |
| `diameter_in`, `diameter_m` | in, m | the chosen diameter |
| `blades`, `hub_ratio`, `activity_factor_per_blade` | | |
| `beta75_deg`, `geometric_pitch_at_75_in`, `pitch_to_diameter` | deg, in, - | |
| `helical_tip_mach` | - | at the design point |
| `design_efficiency` | - | the Adkins-Liebeck minimum-induced-loss design's efficiency |
| `momentum_limit_efficiency` | - | the actuator-disk (Froude) limit for the same thrust and disk |
| `design_thrust_lbf` | lbf | thrust predicted by the design method |
| `bemt_thrust_lbf`, `bemt_power_hp`, `bemt_efficiency` | | the designed blade analysed by the independent BEMT solver |
| `bemt_thrust_error_pct`, `bemt_power_error_pct` | % | disagreement between the two methods |
| `design_iterations` | | |

### Tables

`TABLE.SPANWISE`: `r_over_r[-]`, `radius_m[m]`, `chord_m[m]`, `thickness_ratio[-]`,
`blade_angle_deg[deg]`, `inflow_angle_deg[deg]`, `alpha_deg[deg]`, `cl[-]`, `cd[-]`,
`relative_speed_m_s[m/s]`, `mach[-]`, `reynolds[-]`, `thrust_per_span_n_m[N/m]`,
`torque_per_span_nm_m[N*m/m]`, `loss_factor[-]`. Rows run root to tip. The
per-span loads are for all blades together, so integrating them over radius
gives the totals.

`TABLE.SPEED_SWEEP`: `airspeed_ktas[kt]`, `status[]`, `engine_rpm[rpm]`,
`throttle_pct[%]`, `beta75_deg[deg]`, `thrust_lbf[lbf]`, `shaft_power_hp[hp]`,
`propulsive_efficiency[-]`, `advance_ratio_j[-]`, `helical_tip_mach[-]`, plus
`drag_lbf[lbf]` and `rate_of_climb_fpm[ft/min]` when an airframe was given. It
has 26 rows, from 0 to the larger of 1.25 × the operating airspeed, the propeller's pitch
speed and 1.15 × the level-flight speed. The values are `null` wherever the row's
`status` is not a valid one (section 5.2), and the airframe columns are `null`
below the stall speed.

`TABLE.DIAMETER_TRADE`: `diameter_in[in]`, `diameter_m[m]`, `helical_tip_mach[-]`,
`status[]` (`"ok"`, `"design_not_converged"`, `"design_invalid_chord"`,
`"blade_too_wide"`, `"design_failed"`), `design_efficiency[-]`,
`momentum_limit_efficiency[-]`, `thrust_lbf[lbf]`, `activity_factor[-]`, `beta75_deg[deg]`.

`TABLE.BLADE_DESIGN`: `r_over_r[-]`, `radius_m[m]`, `chord_m[m]`, `chord_in[in]`,
`blade_angle_deg[deg]`, `inflow_angle_deg[deg]`, `thickness_ratio[-]`,
`reynolds[-]`, `mach[-]`, hub to tip.

---

## 4. Why a value is `null`: codes

| code | meaning |
|---|---|
| `NOT_GIVEN` | the input was left blank (inputs, case name) |
| `NO_OPERATING_POINT` | the operating point (or trim) has no valid solution; see its status |
| `NA_STATIC` | undefined at zero airspeed (efficiency, drag, climb) |
| `NA_IN_FORWARD_FLIGHT` | defined only at zero airspeed (figure of merit) |
| `NA_BELOW_STALL_SPEED` | the airspeed is below the stall speed |
| `NOT_PRODUCING_THRUST` | efficiency undefined because thrust is not positive |
| `NOT_ABSORBING_POWER` | ratio undefined because power is not positive |
| `NO_BSFC` | fuel flow needs a BSFC input |
| `CANNOT_HOLD_LEVEL_FLIGHT` | thrust never reaches drag at this setting and altitude |
| `CANNOT_CLIMB` | the best climb rate is not positive |
| `CANNOT_CLIMB_AT_SEA_LEVEL` | a ceiling would be below sea level |
| `ABOVE_MODEL_RANGE` | a ceiling lies above the 32 km atmosphere model |
| `NOT_BRACKETED`, `NO_VALID_POINTS` | the speed search found no usable crossing |
| `INFEASIBLE` | sizing found no feasible diameter |
| `NOT_DEFINED` | the quantity does not exist in this situation |
| `NOT_COMPUTED` | a number could not be computed (should not happen; please report it) |
| `NONE` | used by `NOTICES` for a notice that refers to no input field |

Consumers should treat an unknown code like `NOT_COMPUTED`.

---

## 5. Notices and statuses

### 5.1 `[NOTICES]`

Each notice is four lines with a shared prefix `nNNN.`:

```
n001.code = "RPM_LIMITED"
n001.severity = "warning"
n001.field = null {NONE}            <- or the input key it is about, e.g. "propeller.pitch_in"
n001.message = "Operating point: At full throttle the propeller would overspeed the engine; ..."
```

`severity` is `"error"` (no usable result for what was asked), `"warning"`
(results are valid but something needs attention) or `"info"`.

| code | severity | meaning |
|---|---|---|
| `MISSING` | error | a required input is blank (`field` names it) |
| `INVALID` | error | an input is not a number, out of its physical domain, or inconsistent |
| `UNKNOWN_FIELD` | error | the case contains a key Propwash does not know |
| `OUTSIDE_TYPICAL_RANGE` | warning | an input is unusual for a full-size aircraft; it is computed anyway (there are no caps) |
| `RPM_ABOVE_LIMIT` | warning | a set RPM is above the engine's rated/limit RPM |
| `RPM_LIMITED`, `WINDMILLING`, `TOO_COARSE`, `POWER_NOT_AVAILABLE`, `ON_FINE_STOP`, `ON_COARSE_STOP`, `THRUST_NOT_REACHABLE`, `BELOW_RUNNING_RANGE` | varies | the operating point or trim has that status (5.2) |
| `SHORTFALL` | info | how much power a setting needs against how much there is |
| `TIP_MACH_HIGH` | warning | helical tip Mach above 0.88 |
| `TIP_SUPERSONIC` | warning | helical tip Mach above 1: the section model is not valid |
| `BLADE_STALL` | warning | more than 25% of the span is stalled in forward flight |
| `BEMT_PARTIAL_CONVERGENCE` | warning | more than 10% of blade stations had no momentum solution |
| `SPEED_BELOW_STALL` | warning | the operating airspeed is below the stall speed |
| `NO_BSFC` | info | fuel flow not reported |
| `AIRFRAME_NOT_GIVEN` | info | airframe results not computed |
| `SIZING_INFEASIBLE` | error | no diameter satisfies the constraints |
| `DESIGN_CHECK_DISAGREES` | warning | the BEMT check of a design differs from the design method by more than 5% |

### 5.2 Operating-point statuses

Used by `OPERATING_POINT.status`, `AIRFRAME.level_trim_status` and the
`status` column of `TABLE.SPEED_SWEEP`.

| status | numbers valid? | meaning |
|---|---|---|
| `ok` | yes | a genuine operating point inside every limit |
| `rpm_limited` | yes | fixed pitch at full throttle would overspeed; shown at the RPM limit with the throttle reduced (`throttle_pct` < 100) |
| `on_fine_stop` | yes | constant speed: blades on the low-pitch stop, RPM below the governor setting |
| `on_coarse_stop` | yes | constant speed: blades on the high-pitch stop, RPM above the governor setting |
| `windmilling` | no | the airflow drives the propeller; it makes drag |
| `too_coarse` | no | the propeller holds the engine below a running speed |
| `power_not_available` | no | the setting needs more power than the engine has here |
| `thrust_not_reachable` | no | no setting within the limits makes the thrust needed |
| `below_running_range` | no | less is needed than the propeller gives at the engine's lowest running speed |

When the status is not valid, the performance values are
`null {NO_OPERATING_POINT}`. Propwash never puts a zero or an estimate in
their place.

---

## 6. Integrity

`body_sha256` is the SHA-256 (lower-case hex) of the UTF-8 bytes of everything
before the `[INTEGRITY]` line, up to and including the line feed that ends the
`[NOTICES]` block. The single blank line that precedes `[INTEGRITY]` is not
included. Equivalently:

```python
body = text[: text.index("\n[INTEGRITY]\n")]
assert hashlib.sha256(body.encode("utf-8")).hexdigest() == expected
```

---

## 7. Units

`m`, `m/s`, `kg/m^3`, `Pa`, `Pa*s`, `N`, `N*m`, `N/m`, `N*m/m`, `N/m^2`, `W`,
`kW`, `hp` (mechanical horsepower, 745.699872 W), `kt` (knot, 0.514444 m/s),
`ft`, `ft^2`, `ft/min`, `in`, `lb` (pound mass, 0.45359237 kg), `lbf`, `lbf*ft`, `lb/ft^2`,
`lbf/hp`, `lb/(hp*h)`, `lb/h`, `kg/h`, `US gal/h`, `rpm`, `deg`, `degC`, `%`,
`% rated`, `-` (dimensionless). Values that come in pairs (`thrust_n` and `thrust_lbf`) are
the same quantity in two units, so read whichever you prefer.

---

## 8. Versioning

- The magic line carries the **major** version. It changes only if existing keys
  are renamed or removed, a unit changes, or the grammar changes. A reader
  must refuse a major version it does not know.
- Within a major version, new sections, keys, table columns, notice codes,
  null codes and statuses may be added. A reader must ignore what it does not
  know and must not depend on column positions; use the header names.

---

## 9. Feeding Propwash: the input file

The inputs are a flat JSON object of field keys to values (numbers, or the
choice strings). `propwash template [--mode analysis|sizing]` writes one with
every field `null`, and `propwash fields` documents each field. A program that
generates cases for Propwash writes this file and runs
`propwash run case.json -o report.txt`. Blank (`null`) required fields come back
as `MISSING` notices in an error report, and are never filled in for you.

```json
{
  "case.mode": "analysis",
  "flight.altitude_ft": 6000, "flight.isa_deviation_c": 0, "flight.airspeed_ktas": 110,
  "engine.rated_power_hp": 160, "engine.rated_rpm": 2700, "engine.gear_ratio": 1,
  "engine.aspiration": "normal", "engine.bsfc_lb_hp_h": 0.42,
  "propeller.diameter_in": 75, "propeller.blades": 2, "propeller.activity_factor": 109,
  "propeller.spinner_diameter_in": 9, "propeller.planform": "standard",
  "propeller.root_airfoil": "clark_y", "propeller.tip_airfoil": "clark_y",
  "propeller.root_thickness_pct": 20, "propeller.tip_thickness_pct": 7,
  "propeller.pitch_type": "fixed", "propeller.pitch_in": 57,
  "operating.mode": "set_rpm", "operating.engine_rpm": 2500
}
```

The `INPUTS` section of every report repeats the inputs in this same key
space, so a report can be turned back into a case:
`{k: v["value"] for k, v in report["sections"]["INPUTS"].items()}`.
