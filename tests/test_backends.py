"""Cross-validation of every compute path against the NumPy reference.

A fast backend that disagrees with the reference is a bug, so these are
equality tests, not performance tests.  Paths whose hardware is absent are
skipped rather than silently passing.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from propwash.accel import get_backend, list_backends
from propwash.accel._device_math import element_state, solve_phi
from propwash.bemt.core import PHI_HI, PHI_LO, solve_batch, solve_stations
from propwash.bemt.core import OperatingPoint
from propwash.units import rpm_to_rad_s

REPO = Path(__file__).resolve().parent.parent


def _flat(tables):
    alpha, cl, cd = tables
    return (float(alpha[0]), float(alpha[1] - alpha[0]), int(cl.shape[1]),
            np.ascontiguousarray(cl).ravel(), np.ascontiguousarray(cd).ravel())


# ---------------------------------------------------------------------------
# The shared scalar physics
# ---------------------------------------------------------------------------

def test_device_math_reproduces_the_numpy_solver(stations, tables, options,
                                                 sea_level, prop):
    """The scalar functions the kernels compile must match the vector reference.

    This is the test that keeps the CPU and GPU paths from drifting: both
    Numba targets are compiled from exactly these function objects.
    """
    reference = solve_stations(stations, OperatingPoint(rpm=8000.0, v_inf=6.0,
                                                        air=sea_level),
                               options, tables)
    a0, da, n_alpha, cl_flat, cd_flat = _flat(tables)
    thickness, re_ref, m_crit0 = stations.section_params()
    omega = rpm_to_rad_s(8000.0)

    worst = 0.0
    for i in range(stations.n):
        phi, bracketed = solve_phi(
            stations.twist[i], stations.r[i], stations.chord[i], stations.solidity[i],
            thickness[i], re_ref[i], m_crit0[i], i * n_alpha, n_alpha, a0, da,
            cl_flat, cd_flat, omega * stations.r[i], 6.0, sea_level.density,
            sea_level.viscosity, sea_level.sound_speed, prop.radius, prop.hub_radius,
            prop.n_blades, options.flags(), options.n_bisect, PHI_LO, PHI_HI)
        assert bracketed == 1
        worst = max(worst, abs(float(reference.phi[i]) - phi))
    assert worst < 1e-13, f"scalar and vector paths differ by {worst} rad"


def test_element_state_residual_is_zero_at_the_solution(stations, tables, options,
                                                        sea_level, prop):
    a0, da, n_alpha, cl_flat, cd_flat = _flat(tables)
    thickness, re_ref, m_crit0 = stations.section_params()
    omega = rpm_to_rad_s(7000.0)
    i = stations.n // 2

    phi, _ = solve_phi(
        stations.twist[i], stations.r[i], stations.chord[i], stations.solidity[i],
        thickness[i], re_ref[i], m_crit0[i], i * n_alpha, n_alpha, a0, da,
        cl_flat, cd_flat, omega * stations.r[i], 0.0, sea_level.density,
        sea_level.viscosity, sea_level.sound_speed, prop.radius, prop.hub_radius,
        prop.n_blades, options.flags(), options.n_bisect, PHI_LO, PHI_HI)
    residual = element_state(
        phi, stations.twist[i], stations.r[i], stations.chord[i], stations.solidity[i],
        thickness[i], re_ref[i], m_crit0[i], i * n_alpha, n_alpha, a0, da,
        cl_flat, cd_flat, omega * stations.r[i], 0.0, sea_level.density,
        sea_level.viscosity, sea_level.sound_speed, prop.radius, prop.hub_radius,
        prop.n_blades, options.flags())[0]
    assert abs(residual) < 1e-9 * max(omega * stations.r[i], 1.0)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

def test_numpy_backend_is_always_available():
    assert get_backend("numpy").available


def test_auto_selects_something_that_works():
    backend = get_backend("auto")
    assert backend.available
    assert backend.name in {i.name for i in list_backends()}


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        get_backend("quantum")


def test_unavailable_backend_warns_and_falls_back():
    """Asking for CUDA on a CPU box should degrade, not crash the notebook."""
    infos = {i.name: i for i in list_backends()}
    missing = [n for n, i in infos.items() if not i.available]
    if not missing:
        pytest.skip("every backend is available on this machine")
    with pytest.warns(RuntimeWarning):
        backend = get_backend(missing[0])
    assert backend.available


def test_probe_never_raises():
    for info in list_backends():
        assert isinstance(info.available, bool)
        assert info.available or info.reason, "an unavailable backend must say why"


# ---------------------------------------------------------------------------
# Compiled backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [i.name for i in list_backends()
                                  if i.available and i.name != "numpy"])
def test_backend_matches_the_numpy_reference(name, stations, tables, options,
                                             sea_level):
    rpm, v = np.meshgrid(np.linspace(2000.0, 12000.0, 9),
                         np.linspace(0.0, 24.0, 7), indexing="ij")
    reference = solve_batch(stations, rpm, v, sea_level, options, tables)
    out = get_backend(name).solve_batch(stations, rpm, v, sea_level, options, tables)

    for key, tol in (("thrust", 1e-9), ("torque", 1e-9), ("power", 1e-9)):
        ref = np.asarray(reference[key])
        got = np.asarray(out[key])
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-9)
        assert rel.max() < tol, f"{name}.{key} differs by {rel.max():.2e}"

    assert np.abs(np.asarray(out["phi"]) - np.asarray(reference["phi"])).max() < 1e-12


def test_benchmark_runs_and_cross_validates(prop):
    from propwash.accel.bench import benchmark
    report = benchmark(prop, n_cases=64, n_elements=16, n_bisect=20, verbose=False)
    assert report.rows, "at least the NumPy reference should have run"
    assert report.rows[0].backend == "numpy"
    for row in report.rows[1:]:
        assert row.max_rel_error < 1e-9, f"{row.backend} disagrees with NumPy"
        assert row.seconds > 0.0 and row.cases_per_s > 0.0


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------

CUDASIM_SCRIPT = textwrap.dedent("""
    import math, sys
    import numpy as np
    sys.path.insert(0, %r)
    from propwash.accel.kernels import build_cuda_kernel, cuda_launch_config
    from propwash.bemt.core import solve_batch, SolverOptions, PHI_LO, PHI_HI
    from propwash.geometry import get_preset
    from propwash.atmosphere import SEA_LEVEL as A

    NT = 8
    g = get_preset("APC 10x5 (sport)")
    st = g.discretize(6)
    opts = SolverOptions(n_bisect=14)
    ag, cl_t, cd_t = st.polar_tables()
    th, rr, mc = st.section_params()
    na = cl_t.shape[1]
    cl_f = np.ascontiguousarray(cl_t).ravel()
    cd_f = np.ascontiguousarray(cd_t).ravel()

    rpm = np.array([4000.0, 9000.0, 7000.0])
    v = np.array([0.0, 12.0, 4.0])
    nc, ns = rpm.size, st.n

    kern = build_cuda_kernel(nthreads=NT)
    T = np.zeros(nc); Q = np.zeros(nc); C = np.zeros(nc)
    span = [np.zeros(nc * ns) for _ in range(8)]
    blocks, threads = cuda_launch_config(nc, NT)
    kern[blocks, threads](
        st.r, st.chord, st.twist, st.solidity, th, rr, mc, st.dr, cl_f, cd_f,
        rpm * 2 * math.pi / 60, v, T, Q, C, *span, na, float(ag[0]),
        float(ag[1] - ag[0]), A.density, A.viscosity, A.sound_speed,
        g.radius, g.hub_radius, g.n_blades, opts.flags(), opts.n_bisect,
        PHI_LO, PHI_HI, 1)

    ref = solve_batch(st, rpm, v, A, opts)
    dt = np.abs(T - ref["thrust"]).max()
    dq = np.abs(Q - ref["torque"]).max()
    dphi = np.abs(span[0].reshape(nc, ns) - ref["phi"]).max()
    assert dt < 1e-12, dt
    assert dq < 1e-14, dq
    assert dphi < 1e-15, dphi
    assert np.allclose(C, 1.0)
    print("CUDASIM-OK", dt, dq, dphi)
""")


def test_cuda_kernel_logic_under_the_simulator():
    """Run the real CUDA kernel on the CPU via Numba's simulator.

    This exercises the shared-memory reduction, the grid-stride loop over
    stations and the block-per-case mapping without needing a GPU, so the
    kernel is covered on CI and on a Colab CPU runtime.
    """
    pytest.importorskip("numba")
    env = {"NUMBA_ENABLE_CUDASIM": "1", "PATH": "/usr/bin:/bin",
           "HOME": "/tmp", "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-c", CUDASIM_SCRIPT % str(REPO)],
                         capture_output=True, text=True, timeout=600, env=env)
    assert "CUDASIM-OK" in out.stdout, (out.stdout[-3000:], out.stderr[-3000:])


def test_raw_cuda_source_compiles_to_ptx():
    """NVRTC compiles CUDA C without a device, so the kernel is checked anywhere."""
    from propwash.accel.nvrtc import check, find_library
    if find_library() is None:
        pytest.skip("libnvrtc not installed (pip install nvidia-cuda-nvrtc-cu12)")
    for result in check(("compute_75", "compute_80", "compute_90"), verbose=False):
        assert result.ok, f"{result.architecture}: {result.log}"
        assert result.ptx_bytes > 10_000


def test_raw_cuda_source_mirrors_the_python_kernel():
    """Both kernels must implement the same constants and structure."""
    from propwash.accel.kernels_raw import CUDA_SOURCE, PW_THREADS
    from propwash.accel.kernels import THREADS_PER_BLOCK

    assert PW_THREADS == THREADS_PER_BLOCK
    assert f"#define PW_THREADS  {PW_THREADS}" in CUDA_SOURCE
    for token in ("pw_prandtl", "pw_interp", "pw_corrections", "pw_element",
                  "__syncthreads", "bemt_solve", "FLAG_TIP_LOSS"):
        assert token in CUDA_SOURCE, f"{token} missing from the CUDA kernel"
