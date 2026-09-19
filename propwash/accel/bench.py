"""Benchmark and cross-validate every available backend.

Two things matter and they are reported together: how fast a backend is, and
whether it still gets the right answer.  A GPU path that is a hundred times
faster and half a percent wrong is a bug, not a feature, so every backend is
diffed against the NumPy reference on the same grid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..atmosphere import SEA_LEVEL, AirState
from ..bemt.core import SolverOptions
from ..geometry import BladeGeometry, get_preset
from .backend import get_backend, list_backends


@dataclass
class BenchRow:
    backend: str
    device: str
    n_cases: int
    n_elements: int
    seconds: float
    cases_per_s: float
    element_solves_per_s: float
    residual_evals_per_s: float
    max_thrust_error: float = 0.0
    max_rel_error: float = 0.0
    note: str = ""

    def format(self) -> str:
        err = "reference" if self.note == "reference" else f"{self.max_rel_error:.2e}"
        return (f"{self.backend:<12} {self.device[:22]:<22} {self.seconds * 1e3:9.1f} ms "
                f"{self.cases_per_s:11,.0f} case/s {self.residual_evals_per_s / 1e6:8.1f} Mres/s  "
                f"rel.err {err}")


@dataclass
class BenchReport:
    rows: list[BenchRow] = field(default_factory=list)
    n_cases: int = 0
    n_elements: int = 0
    n_bisect: int = 0
    blade: str = ""

    def format(self) -> str:
        head = (f"Propwash backend benchmark -- {self.blade}\n"
                f"  {self.n_cases:,} operating points x {self.n_elements} elements "
                f"x {self.n_bisect + 2} residual evaluations "
                f"= {self.n_cases * self.n_elements * (self.n_bisect + 2):,} evaluations\n")
        body = "\n".join("  " + r.format() for r in self.rows)
        if len(self.rows) > 1:
            base = self.rows[0].seconds
            fastest = min(self.rows, key=lambda r: r.seconds)
            body += (f"\n\n  fastest: {fastest.backend} "
                     f"({base / fastest.seconds:.1f}x the NumPy reference)")
        return head + body


def benchmark(geometry: BladeGeometry | None = None, n_cases: int = 4096,
              n_elements: int = 60, n_bisect: int = 60, air: AirState | None = None,
              backends: list[str] | None = None, warmup: bool = True,
              verbose: bool = True) -> BenchReport:
    """Run every available backend over the same grid and compare.

    ``warmup`` runs each backend once on a small grid first, so JIT compilation
    and CUDA context creation are not counted as solve time.
    """
    geometry = geometry or get_preset("APC 10x5 (sport)")
    air = air or SEA_LEVEL
    opts = SolverOptions(n_elements=n_elements, n_bisect=n_bisect)
    stations = geometry.discretize(n_elements, opts.spacing)
    tables = stations.polar_tables(opts.n_alpha_table)

    side = max(int(np.sqrt(n_cases)), 2)
    rpm_g, v_g = np.meshgrid(np.linspace(1500.0, 14000.0, side),
                             np.linspace(0.0, 35.0, side), indexing="ij")
    n_total = rpm_g.size

    infos = {i.name: i for i in list_backends()}
    names = backends or [i.name for i in list_backends() if i.available]
    names = sorted(set(names), key=lambda n: (n != "numpy", n))

    report = BenchReport(n_cases=n_total, n_elements=n_elements, n_bisect=n_bisect,
                         blade=geometry.describe())
    reference: np.ndarray | None = None

    for name in names:
        info = infos.get(name)
        if info is not None and not info.available:
            continue
        backend = get_backend(name)

        try:
            if warmup:
                backend.solve_batch(stations, rpm_g[:2, :2], v_g[:2, :2], air, opts,
                                    tables, want_spanwise=False)
            t0 = time.perf_counter()
            out = backend.solve_batch(stations, rpm_g, v_g, air, opts, tables,
                                      want_spanwise=False)
            dt = max(time.perf_counter() - t0, 1e-9)
        except Exception as exc:  # pragma: no cover - environment dependent
            if verbose:
                print(f"  {name:<12} failed: {type(exc).__name__}: {exc}")
            continue

        thrust = np.asarray(out["thrust"], dtype=float)
        row = BenchRow(
            backend=name, device=(info.device if info else "?"),
            n_cases=n_total, n_elements=n_elements, seconds=dt,
            cases_per_s=n_total / dt,
            element_solves_per_s=n_total * n_elements / dt,
            residual_evals_per_s=n_total * n_elements * (n_bisect + 2) / dt,
        )
        if reference is None:
            reference = thrust
            row.note = "reference"
        else:
            diff = np.abs(thrust - reference)
            row.max_thrust_error = float(diff.max())
            row.max_rel_error = float((diff / np.maximum(np.abs(reference), 1e-9)).max())
        report.rows.append(row)
        if verbose:
            print("  " + row.format())

    return report


def sizing_sweep(sizes=(256, 1024, 4096, 16384, 65536), **kw) -> dict[str, list[float]]:
    """Throughput against problem size -- where a GPU starts to pay for itself."""
    out: dict[str, list[float]] = {}
    for n in sizes:
        rep = benchmark(n_cases=n, verbose=False, **kw)
        for row in rep.rows:
            out.setdefault(row.backend, []).append(row.cases_per_s)
    out["sizes"] = list(sizes)
    return out


__all__ = ["benchmark", "sizing_sweep", "BenchReport", "BenchRow"]
