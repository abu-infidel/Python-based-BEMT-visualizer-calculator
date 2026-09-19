"""Command line interface.

Uses argparse rather than a framework so the package has no CLI dependency;
``rich`` is used for tables when it happens to be installed and plain text
otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .version import PROJECT_NAME, PROJECT_TAGLINE, __version__


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _console():
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def _table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    console = _console()
    if console is None:
        widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
                  for i, c in enumerate(columns)]
        print(f"\n{title}")
        print("  " + "  ".join(c.ljust(w) for c, w in zip(columns, widths)))
        print("  " + "  ".join("-" * w for w in widths))
        for row in rows:
            print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, widths)))
        return

    from rich.table import Table
    table = Table(title=title, title_justify="left", header_style="bold",
                  box=None, pad_edge=False)
    for c in columns:
        table.add_column(c)
    for row in rows:
        table.add_row(*[str(v) for v in row])
    console.print(table)


def _geometry_from_args(args):
    from dataclasses import replace
    from .geometry import get_preset
    from .units import INCH

    g = get_preset(args.prop)
    kw = {}
    if getattr(args, "blades", None):
        kw["n_blades"] = int(args.blades)
    if getattr(args, "diameter", None):
        kw["radius"] = float(args.diameter) * INCH / 2.0
    if getattr(args, "pitch", None):
        kw["pitch_offset"] = math.radians(float(args.pitch))
    return replace(g, **kw) if kw else g


def _air_from_args(args):
    from .atmosphere import isa
    return isa(float(getattr(args, "altitude", 0.0) or 0.0),
               delta_isa=float(getattr(args, "disa", 0.0) or 0.0))


def _options_from_args(args):
    from .bemt.core import SolverOptions
    return SolverOptions(n_elements=int(getattr(args, "elements", 60) or 60))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_env(args) -> int:
    from .colab.bootstrap import probe
    print(probe().report())
    return 0


def cmd_presets(args) -> int:
    from .airfoil import get_airfoil, list_airfoils
    from .geometry import get_preset, list_presets
    from .motor import get_motor, list_motors

    _table("Propellers", ["name", "blades", "diameter", "P/D", "AF", "solidity"],
           [[n, str(g.n_blades), f"{g.diameter * 1000:.0f} mm",
             f"{g.pitch_diameter_ratio():.2f}", f"{g.activity_factor():.0f}",
             f"{g.mean_solidity():.3f}"]
            for n in list_presets() for g in (get_preset(n),)])

    _table("Airfoil sections", ["key", "name", "t/c", "alpha_0", "Cl_max", "L/D max"],
           [[k, a.name, f"{a.thickness:.3f}", f"{s['alpha_0_deg']:.1f} deg",
             f"{s['cl_max']:.2f}", f"{s['ld_max']:.0f}"]
            for k in list_airfoils()
            for a in (get_airfoil(k),) for s in (a.summary(),)])

    _table("Motors", ["name", "Kv", "cells", "volts", "Rm", "I max", "no-load rpm"],
           [[n, f"{m.kv:.0f}", str(m.cells), f"{m.voltage:.1f} V",
             f"{m.resistance * 1000:.0f} mohm", f"{m.max_current:.0f} A",
             f"{m.no_load_rpm():,.0f}"]
            for n in list_motors() for m in (get_motor(n),)])
    return 0


def cmd_solve(args) -> int:
    from .bemt.core import OperatingPoint
    from .bemt.solver import PropellerSolver, spanwise_frame

    geom = _geometry_from_args(args)
    solver = PropellerSolver(geom, _options_from_args(args), args.backend)
    result = solver.solve(OperatingPoint(rpm=args.rpm, v_inf=args.speed,
                                         air=_air_from_args(args)))

    print(f"\n{geom.describe()}")
    print(f"{result.meta['air']}\n")
    summary = result.summary()
    _table(f"Operating point: {args.rpm:,.0f} rpm, V = {args.speed:.1f} m/s",
           ["quantity", "value"],
           [[k, f"{v:,.5g}"] for k, v in summary.items()])

    if args.spanwise:
        frame = spanwise_frame(result, solver.stations)
        if hasattr(frame, "to_csv"):
            path = Path(args.spanwise)
            frame.to_csv(path, index=False)
            print(f"\nSpanwise data -> {path.resolve()}")
        else:
            print("\npandas is not installed; cannot write the spanwise CSV")

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"Summary -> {Path(args.json).resolve()}")
    return 0


def cmd_match(args) -> int:
    from .bemt.sweep import match_operating_point
    from .motor import get_motor

    geom = _geometry_from_args(args)
    motor = get_motor(args.motor)
    if args.cells:
        motor = motor.with_cells(int(args.cells))
    if args.kv:
        from dataclasses import replace
        motor = replace(motor, kv=float(args.kv))

    mp = match_operating_point(geom, motor, v_inf=args.speed, throttle=args.throttle,
                               air=_air_from_args(args), options=_options_from_args(args),
                               backend=args.backend)
    print(f"\n{geom.describe()}")
    print(f"{motor.describe()}\n")
    if not mp.converged:
        print("No torque balance found -- the motor either cannot turn this "
              "propeller or never loads up.\n")
    _table("Matched operating point", ["quantity", "value"], [
        ["shaft speed", f"{mp.rpm:,.0f} rpm"],
        ["thrust", f"{mp.thrust:.3f} N  ({mp.thrust / 9.80665 * 1000:,.0f} gf)"],
        ["shaft torque", f"{mp.torque:.4f} N m"],
        ["shaft power", f"{mp.shaft_power:,.1f} W"],
        ["electrical power", f"{mp.electrical_power:,.1f} W"],
        ["current", f"{mp.current:.2f} A"],
        ["motor efficiency", f"{mp.motor_efficiency * 100:.1f} %"],
        ["system efficiency", f"{mp.system_efficiency * 100:.1f} %"],
    ])
    return 0


def cmd_sweep(args) -> int:
    from .bemt.sweep import j_sweep

    geom = _geometry_from_args(args)
    sweep = j_sweep(geom, rpm=args.rpm, j_max=args.j_max, n=args.points,
                    air=_air_from_args(args), options=_options_from_args(args),
                    backend=args.backend)
    j = sweep.axes["j"]
    eta = np.nan_to_num(np.asarray(sweep["efficiency"]))
    best = int(np.argmax(eta))

    stride = max(len(j) // 18, 1)
    _table(f"{geom.name} at {args.rpm:,.0f} rpm",
           ["J", "CT", "CP", "eta", "thrust [N]", "power [W]"],
           [[f"{j[i]:.3f}", f"{sweep['ct'][i]:.4f}", f"{sweep['cp'][i]:.4f}",
             f"{eta[i]:.3f}", f"{sweep['thrust'][i]:.2f}", f"{sweep['power'][i]:.0f}"]
            for i in range(0, len(j), stride)])
    print(f"\nPeak efficiency {eta[best]:.3f} at J = {j[best]:.3f} "
          f"(V = {j[best] * args.rpm / 60 * geom.diameter:.1f} m/s)")

    if args.csv:
        try:
            sweep.to_frame().to_csv(args.csv, index=False)
            print(f"Sweep -> {Path(args.csv).resolve()}")
        except ImportError:
            print("pandas is not installed; cannot write the CSV")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        from .viz.plots import j_sweep_figure
        j_sweep_figure(sweep).savefig(args.plot, dpi=140, bbox_inches="tight")
        print(f"Chart -> {Path(args.plot).resolve()}")
    return 0


def cmd_mesh(args) -> int:
    from .mesh import build_propeller_mesh

    geom = _geometry_from_args(args)
    mesh = build_propeller_mesh(geom, n_span=args.span, n_chord=args.chord)
    out = Path(args.output or ".")
    out.mkdir(parents=True, exist_ok=True)
    stem = geom.name.replace(" ", "_").replace("/", "-")

    (out / f"{stem}.obj").write_text(mesh.to_obj())
    (out / f"{stem}.stl").write_bytes(mesh.to_stl())
    geom.save(out / f"{stem}.json")

    print(f"\n{geom.describe()}")
    print(f"{mesh.describe()}\n")
    for p in sorted(out.glob(f"{stem}.*")):
        print(f"  {p.name:<36} {p.stat().st_size / 1024:9.1f} kB")
    return 0


def cmd_bench(args) -> int:
    from .accel.bench import benchmark
    geom = _geometry_from_args(args)
    print()
    rep = benchmark(geom, n_cases=args.cases, n_elements=args.elements,
                    n_bisect=args.bisect, air=_air_from_args(args), verbose=False)
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
    return launch(preset=args.prop, motor=args.motor, dark=not args.light,
                  n_elements=args.elements)


def cmd_colab(args) -> int:
    from .colab.bootstrap import in_notebook, probe
    if in_notebook():
        from .colab import launch
        launch(preset=args.prop, motor=args.motor, dark=args.dark)
        return 0
    print(probe().report())
    print("\nThe notebook GUI needs a notebook.  In Colab or Jupyter run:\n")
    print("    !pip install -q propwash            # or: %pip install -e .")
    print("    import propwash.colab as pc")
    print("    pc.setup()                          # installs and reports")
    print("    lab = pc.launch()                   # the GUI\n")
    print("Or just open notebooks/Propwash_Propeller_Lab.ipynb in Colab.")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="propwash",
        description=f"{PROJECT_NAME} {__version__} -- {PROJECT_TAGLINE}")
    p.add_argument("--version", action="version", version=f"{PROJECT_NAME} {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_prop(sp, default_rpm: float | None = None):
        sp.add_argument("--prop", default="APC 10x5 (sport)", help="propeller preset")
        sp.add_argument("--blades", type=int, help="override blade count")
        sp.add_argument("--diameter", type=float, help="override diameter [in]")
        sp.add_argument("--pitch", type=float, help="collective pitch offset [deg]")
        sp.add_argument("--altitude", type=float, default=0.0, help="altitude [m]")
        sp.add_argument("--disa", type=float, default=0.0, help="ISA temperature offset [K]")
        sp.add_argument("--elements", type=int, default=60, help="blade elements")
        sp.add_argument("--backend", default="auto",
                        help="numpy | numba-cpu | numba-cuda | cupy | torch | auto")
        if default_rpm is not None:
            sp.add_argument("--rpm", type=float, default=default_rpm)

    sp = sub.add_parser("env", help="report the runtime and available backends")
    sp.set_defaults(func=cmd_env)

    sp = sub.add_parser("presets", help="list propellers, sections and motors")
    sp.set_defaults(func=cmd_presets)

    sp = sub.add_parser("solve", help="solve one operating point")
    add_prop(sp, 6000.0)
    sp.add_argument("--speed", type=float, default=0.0, help="airspeed [m/s]")
    sp.add_argument("--spanwise", help="write spanwise data to this CSV")
    sp.add_argument("--json", help="write the summary to this JSON file")
    sp.set_defaults(func=cmd_solve)

    sp = sub.add_parser("match", help="solve the propeller/motor torque balance")
    add_prop(sp)
    sp.add_argument("--motor", default="Sport 2820 kv1000")
    sp.add_argument("--cells", type=int, help="override LiPo cell count")
    sp.add_argument("--kv", type=float, help="override motor Kv")
    sp.add_argument("--throttle", type=float, default=1.0)
    sp.add_argument("--speed", type=float, default=0.0, help="airspeed [m/s]")
    sp.set_defaults(func=cmd_match)

    sp = sub.add_parser("sweep", help="advance-ratio sweep")
    add_prop(sp, 8000.0)
    sp.add_argument("--j-max", type=float, default=1.2, dest="j_max")
    sp.add_argument("--points", type=int, default=60)
    sp.add_argument("--csv", help="write the sweep to this CSV")
    sp.add_argument("--plot", help="write a chart to this PNG")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("mesh", help="export the 3-D blade as OBJ and STL")
    add_prop(sp)
    sp.add_argument("--span", type=int, default=44)
    sp.add_argument("--chord", type=int, default=49)
    sp.add_argument("-o", "--output", default="exports")
    sp.set_defaults(func=cmd_mesh)

    sp = sub.add_parser("bench", help="benchmark and cross-validate the backends")
    add_prop(sp)
    sp.add_argument("--cases", type=int, default=4096)
    sp.add_argument("--bisect", type=int, default=60)
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("cuda-check", help="compile the raw CUDA C kernel with NVRTC")
    sp.add_argument("--dump", help="also write the CUDA source to this .cu file")
    sp.add_argument("--arch", nargs="*", metavar="compute_XX",
                    help="target architectures (default: 75, 80, 89)")
    sp.set_defaults(func=cmd_cuda_check)

    sp = sub.add_parser("gui", help="launch the desktop GUI")
    sp.add_argument("--prop", default=None)
    sp.add_argument("--motor", default=None)
    sp.add_argument("--elements", type=int, default=48)
    sp.add_argument("--light", action="store_true", help="light theme")
    sp.set_defaults(func=cmd_gui)

    sp = sub.add_parser("colab", help="notebook GUI instructions, or launch it")
    sp.add_argument("--prop", default=None)
    sp.add_argument("--motor", default=None)
    sp.add_argument("--dark", action="store_true")
    sp.set_defaults(func=cmd_colab)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
