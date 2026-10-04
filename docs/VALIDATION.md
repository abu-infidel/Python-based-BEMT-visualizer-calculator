# Validation against a real aircraft

`python -m propwash validate` runs every check below and prints the table.
`tests/test_validation.py` runs the same checks, so a change to the physics
that breaks agreement fails the test suite.

![Validation](images/validation.png)

## The reference aircraft

A **Cessna 172N**: a Lycoming O-320-H2AD rated at 160 hp at 2,700 rpm, driving
the standard fixed-pitch **McCauley 1C160/DTM7557** propeller (75 in diameter,
57 in pitch) directly. The full input case is `propwash.validation.C172N_CASE`.
`propwash validate --write-case c172n.json` saves it as a case file you can
open in either GUI.

**Published sources**

- Cessna 172N Pilot's Operating Handbook, Section 5: cruise performance at
  2,300 lb on a standard day, maximum speed, maximum rate of climb, service
  ceiling, and flaps-up stall speed.
- FAA Type Certificate Data Sheet 3A12: static RPM at full throttle for this
  propeller, 2,280–2,400 rpm.

**Inputs not published, and what was used**

| input | value | basis |
|---|---|---|
| blade activity factor | 109 | **calibrated**: set so full-throttle static RPM is mid-range (2,340) |
| planform | `standard` | typical fixed-pitch metal outline |
| thickness, root → tip | 20% → 7% | typical metal blade |
| section | Clark Y | flat-faced section, as on blades of this type |
| spinner diameter | 9 in | approximate |
| drag polar | CD0 0.0341, e 0.77 | an estimate used in performance examples, not from Cessna |
| CLmax (flaps up) | 1.56 | from the POH stall speed: 50 KCAS at 2,300 lb |
| BSFC | 0.42 lb/(hp·h) | implied by the POH fuel flows (8.4 gal/h at 75%) |

## The checks

| check | kind | reference | model | error | limit |
|---|---|---:|---:|---:|---:|
| Static RPM, full throttle, sea level | calibration | 2,280–2,400 | 2,339 | — | range |
| Power absorbed, 8,000 ft, 2,650 rpm, 122 KTAS | validation | 75% | 79.2% | +5.7% | ±8% |
| Power absorbed, 8,000 ft, 2,500 rpm, 114 KTAS | validation | 64% | 65.1% | +1.7% | ±8% |
| Power absorbed, 4,000 ft, 2,300 rpm, 105 KTAS | validation | 57% | 54.8% | −3.9% | ±8% |
| Power absorbed, 4,000 ft, 2,200 rpm, 99 KTAS | validation | 51% | 49.2% | −3.6% | ±8% |
| Power absorbed, 4,000 ft, 2,100 rpm, 93 KTAS | validation | 46% | 43.9% | −4.5% | ±8% |
| Maximum level speed, sea level | indicative | 125 KTAS | 124.9 KTAS | −0.1% | ±5% |
| Maximum rate of climb, sea level | indicative | 770 ft/min | 895 ft/min | +16.3% | ±25% |
| Service ceiling | indicative | 14,200 ft | 17,391 ft | +22.5% | ±25% |

The three kinds are kept apart on purpose:

- **Calibration** agrees by construction. The 1C160/DTM7557's blade width
  (activity factor) is not published, so it was set from the static RPM. That
  check says nothing on its own.
- **Validation** rows are propeller-only predictions and do not depend on the
  airframe. For each published cruise row, the model computes the power the
  propeller absorbs at that row's RPM and true airspeed, and compares it with the
  published percentage of rated power. All five agree within 6%.
- **Indicative** rows are whole-aircraft results. They also depend on the
  estimated drag polar and on installation effects the model does not include.
  Top speed matches. Climb and ceiling are over-predicted by 16% and 22%. The
  most likely cause is installation losses that strip theory leaves out:
  slipstream scrubbing over the cowling and fuselage, and cooling drag. Both
  matter most at the low speeds where climb happens, and a ceiling magnifies
  any error in excess power.

## What was adjusted to get here

Being explicit about this matters more than the numbers.

1. **Activity factor 109**: calibrated to the TCDS static RPM, as above.
2. **Clark Y zero-lift angle −4.2°**. Propeller pitch is measured off the flat
   face of the blade, which sits about 2° nose down from the
   leading-to-trailing-edge chord of a Clark Y. With the chord-referenced value
   (−3.6°), every cruise row came out 8–16% low on power. −4.2° is inside the
   published range for the section (−3.6° chord-referenced to about −5.5°
   face-referenced), and it was chosen with the cruise rows in view. It is
   therefore a second, physically constrained calibration, not an independent
   result. It changes the Clark Y section for every propeller, not just this
   one.

Nothing else was tuned to this aircraft. The Reynolds, compressibility,
thickness and engine-lapse models are general, and their constants are in
[PHYSICS.md](PHYSICS.md) and in every report's `[MODEL]` section.

## Consistency checks, not against flight data

- Every compute backend (Numba CPU, OpenCL, PyTorch, and Numba CUDA under the
  simulator) agrees with the NumPy reference to 1e-12 or better.
- The Adkins–Liebeck design method and the independent BEMT solver agree
  within 1% on thrust and power for the sizing example.
- ISA reproduces the published standard-atmosphere table.
- `tests/test_analysis.py` sweeps extreme inputs: 45 hp to 1,500 hp,
  54–156 in diameters, geared engines, absurdly coarse and fine pitches, and
  hot-and-high and cold, dense air. It asserts that no report contains a
  non-finite number and that no valid operating point reports zero RPM,
  thrust, power or torque.
