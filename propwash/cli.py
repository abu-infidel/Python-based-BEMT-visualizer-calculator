"""Command line interface.

Every calculation goes through the same case file the GUIs fill in::

    propwash template -o my_case.json      # every input, all blank
    # ...fill it in...
    propwash run my_case.json -o my_report.txt
    propwash parse my_report.txt --json    # read a report back (for other programs)
    propwash validate                      # model vs Cessna 172N published data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .version import PROJECT_NAME, PROJECT_TAGLINE, __version__


def _load_case(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "inputs" in data and isinstance(data["inputs"], dict):
        data = data["inputs"]
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object of field -> value")
    return data


def cmd_template(args) -> int:
    from .case import blank_case
    text = json.dumps(blank_case(args.mode), indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"blank case -> {Path(args.output).resolve()}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_fields(args) -> int:
    from .case import FIELDS
    for f in FIELDS:
        kind = "choice: " + " | ".join(f.choices) if f.kind == "choice" else f.kind
        unit = f" [{f.unit}]" if f.unit else ""
        typical = f"  typical {f.typical[0]:g}-{f.typical[1]:g}" if f.typical else ""
        print(f"{f.key:<34} {f.label}{unit}  ({kind}){typical}")
        print(f"{'':<34} {f.help}")
    return 0


def _summary(report) -> str:
    from .report import parse_report
    d = parse_report(report.to_text())
    out = [f"status: {report.status}"]
    for name in ("OPERATING_POINT", "AIRFRAME", "DESIGN_POINT", "SIZING"):
        sec = d["sections"].get(name)
        if not sec:
            continue
        out.append(f"\n[{name}]")
        for key, item in sec.items():
            val = item["value"]
            if val is None:
                text = f"n/a ({item['code']})"
            elif isinstance(val, float):
                text = f"{val:,.4g} {item['unit']}".rstrip()
            else:
                text = f"{val} {item['unit']}".rstrip()
            out.append(f"  {key:<36} {text}")
    if d["notices"]:
        out.append("\n[NOTICES]")
        for n in d["notices"]:
            field = f" ({n['field']})" if n["field"] else ""
            out.append(f"  {n['severity'].upper():<8} {n['code']}{field}: {n['message']}")
    return "\n".join(out)


def cmd_run(args) -> int:
    from .analysis import run
    report = run(_load_case(args.case))
    print(_summary(report))
    if args.output:
        report.save(args.output)
        print(f"\nreport -> {Path(args.output).resolve()}")
    return 1 if report.status == "error" else 0


def cmd_parse(args) -> int:
    from .report import ReportFormatError, parse_report
    try:
        data = parse_report(Path(args.report).read_text(encoding="utf-8"),
                            verify=not args.no_verify)
    except ReportFormatError as exc:
        print(f"{args.report}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"{data['format']['name']} v{data['format']['version']}, "
              f"checksum {'ok' if data['integrity']['ok'] else 'NOT VERIFIED'}")
        for name, sec in data["sections"].items():
            print(f"[{name}] {len(sec)} values")
        for name, t in data["tables"].items():
            print(f"[TABLE.{name}] {len(t['rows'])} rows x {len(t['columns'])} columns")
        print(f"{len(data['notices'])} notices")
    return 0


def cmd_validate(args) -> int:
    from .validation import C172N_CASE, format_checks, run_validation
    if args.write_case:
        Path(args.write_case).write_text(json.dumps(C172N_CASE, indent=2) + "\n")
        print(f"Cessna 172N reference case -> {Path(args.write_case).resolve()}\n")
    print("Propwash against published Cessna 172N data (see docs/VALIDATION.md)\n")
    checks = run_validation()
    print(format_checks(checks))
    return 0 if all(c.passed for c in checks) else 1


def cmd_env(args) -> int:
    from .colab.bootstrap import probe
    print(probe().report())
    return 0


def cmd_bench(args) -> int:
    from .accel.bench import benchmark
    from .analysis import build_geometry
    from .atmosphere import isa
    from .case import parse_case

    source = args.case
    if source is None:
        from .validation import C172N_CASE
        raw, label = C172N_CASE, "Cessna 172N reference propeller"
    else:
        raw, label = _load_case(source), source
    geom, _ = build_geometry(parse_case(raw))
    print(f"\nbenchmark blade: {label}")
    rep = benchmark(geom, isa(0.0), n_cases=args.cases, n_bisect=args.bisect, verbose=False)
    print(rep.format())
    print()
    return 0


def cmd_cuda_check(args) -> int:
    from .accel.kernels_raw import kernel_source
    from .accel.nvrtc import DEFAULT_ARCHITECTURES, check

    if args.dump:
        Path(args.dump).write_text(kernel_source())
        print(f"CUDA source -> {Path(args.dump).resolve()}")
    arches = tuple(args.arch) if args.arch else DEFAULT_ARCHITECTURES
    print("\nCompiling the raw CUDA C kernel with NVRTC (no GPU needed)\n")
    try:
        results = check(arches, verbose=True)
    except RuntimeError as exc:
        print(f"{exc}")
        return 1
    return 0 if all(r.ok for r in results) else 1


def cmd_gui(args) -> int:
    try:
        from .gui import launch
    except ImportError as exc:
        print(f"The desktop GUI needs PySide6 and pyqtgraph: {exc}")
        print('  pip install "propwash[desktop]"   # or: pip install PySide6 pyqtgraph PyOpenGL')
        return 1
    return launch(case=_load_case(args.case) if args.case else None, dark=not args.light)


def cmd_colab(args) -> int:
    from .colab.bootstrap import in_notebook, probe
    if in_notebook():
        from .colab import launch
        launch()
        return 0
    print(probe().report())
    print("\nThe notebook app needs a notebook.  In Colab or Jupyter run:\n")
    print("    import propwash.colab as pc")
    print("    pc.setup()")
    print("    app = pc.launch()\n")
    print("Or open notebooks/Propwash_Propeller_Lab.ipynb in Colab (see docs/COLAB.md).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="propwash", description=f"{PROJECT_NAME} {__version__} -- {PROJECT_TAGLINE}")
    p.add_argument("--version", action="version", version=f"{PROJECT_NAME} {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("template", help="write a blank case file (every input empty)")
    sp.add_argument("--mode", choices=("analysis", "sizing"))
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_template)

    sp = sub.add_parser("fields", help="describe every input field")
    sp.set_defaults(func=cmd_fields)

    sp = sub.add_parser("run", help="calculate a case file; -o writes the .txt report")
    sp.add_argument("case")
    sp.add_argument("-o", "--output", help="write the structured report here")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("parse", help="read a report file back")
    sp.add_argument("report")
    sp.add_argument("--json", action="store_true", help="print as JSON")
    sp.add_argument("--no-verify", action="store_true", help="ignore a checksum mismatch")
    sp.set_defaults(func=cmd_parse)

    sp = sub.add_parser("validate", help="compare with published Cessna 172N data")
    sp.add_argument("--write-case", help="also save the reference case to this JSON file")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("env", help="report the runtime and compute backends")
    sp.set_defaults(func=cmd_env)

    sp = sub.add_parser("bench", help="benchmark and cross-check the compute backends")
    sp.add_argument("--case", help="case file whose propeller to use "
                    "(otherwise the Cessna 172N reference propeller)")
    sp.add_argument("--cases", type=int, default=4096, help="operating points per backend")
    sp.add_argument("--bisect", type=int, default=60, help="bisection steps per element")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("cuda-check", help="compile the raw CUDA C kernel with NVRTC")
    sp.add_argument("--dump", help="also write the CUDA source to this .cu file")
    sp.add_argument("--arch", nargs="*", metavar="compute_XX")
    sp.set_defaults(func=cmd_cuda_check)

    sp = sub.add_parser("gui", help="launch the desktop GUI")
    sp.add_argument("--case", help="open with this case file filled in")
    sp.add_argument("--light", action="store_true", help="light theme")
    sp.set_defaults(func=cmd_gui)

    sp = sub.add_parser("colab", help="notebook app instructions, or launch it")
    sp.set_defaults(func=cmd_colab)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
