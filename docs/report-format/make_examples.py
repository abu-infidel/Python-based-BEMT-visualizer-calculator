"""Regenerate the example reports in this folder: ``python docs/report-format/make_examples.py``."""

from pathlib import Path

from propwash.analysis import run
from propwash.case import blank_case
from propwash.validation import C172N_CASE

HERE = Path(__file__).resolve().parent

analysis = dict(C172N_CASE, **{"flight.altitude_ft": 6000.0, "flight.airspeed_ktas": 110.0,
                               "operating.mode": "set_rpm", "operating.engine_rpm": 2500.0})
sizing = blank_case("sizing")
sizing.update({"case.name": "Sizing example: 160 hp cruise propeller",
               "flight.altitude_ft": 8000.0, "flight.isa_deviation_c": 0.0,
               "flight.airspeed_ktas": 122.0, "engine.rated_power_hp": 160.0,
               "engine.rated_rpm": 2700.0, "engine.gear_ratio": 1.0,
               "engine.aspiration": "normal", "engine.bsfc_lb_hp_h": 0.42,
               "propeller.blades": 2, "propeller.spinner_diameter_in": 9.0,
               "propeller.root_thickness_pct": 20.0, "propeller.tip_thickness_pct": 7.0,
               "operating.engine_rpm": 2650.0, "operating.power_pct": 75.0,
               "sizing.max_diameter_in": 76.0, "sizing.tip_mach_limit": 0.88,
               "sizing.design_cl": 0.5, "sizing.airfoil": "clark_y"})
error = blank_case("analysis")
error.update({"case.name": "Error example: incomplete form", "propeller.diameter_in": "-75",
              "engine.rated_power_hp": "180"})

for name, case in (("example_analysis.txt", analysis), ("example_sizing.txt", sizing),
                   ("example_error.txt", error)):
    run(case).save(HERE / name)
    print("wrote", HERE / name)
