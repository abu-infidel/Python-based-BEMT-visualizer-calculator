"""The polar must be finite and continuous over the whole circle.

If it is not, the solver's bisection walks into a NaN and every downstream
number is garbage -- so these are load-bearing tests, not decoration.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from propwash.airfoil import (DEG, TablePolar, get_airfoil, list_airfoils,
                              stack_tables, viterna_extrapolate)


@pytest.mark.parametrize("name", list_airfoils())
def test_polar_is_finite_over_the_full_circle(name):
    polar = get_airfoil(name)
    alpha = np.linspace(-math.pi, math.pi, 4001)
    cl, cd = polar.base_polar(alpha)
    assert np.isfinite(cl).all()
    assert np.isfinite(cd).all()
    assert (cd > 0.0).all()
    assert np.abs(cl).max() < 3.0, "deep-stall lift should stay bounded"


@pytest.mark.parametrize("name", list_airfoils())
def test_polar_is_continuous(name):
    """No step larger than a sensible per-sample bound anywhere on the circle."""
    polar = get_airfoil(name)
    alpha = np.linspace(-math.pi, math.pi, 20001)
    cl, cd = polar.base_polar(alpha)
    assert np.abs(np.diff(cl)).max() < 0.02
    assert np.abs(np.diff(cd)).max() < 0.02


@pytest.mark.parametrize("name,alpha0,cl_max", [
    ("naca0012", 0.0, 1.35), ("naca2412", -2.1, 1.52), ("naca4412", -4.2, 1.62),
])
def test_preset_parameters_are_recovered(name, alpha0, cl_max):
    summary = get_airfoil(name).summary()
    assert summary["alpha_0_deg"] == pytest.approx(alpha0, abs=0.25)
    assert summary["cl_max"] >= cl_max - 0.05


def test_symmetric_section_is_symmetric():
    polar = get_airfoil("naca0012")
    a = np.linspace(-12.0, 12.0, 121) * DEG
    cl_p, cd_p = polar.base_polar(a)
    cl_n, cd_n = polar.base_polar(-a)
    assert np.allclose(cl_p, -cl_n, atol=1e-9)
    assert np.allclose(cd_p, cd_n, atol=1e-9)


def test_lift_slope_is_near_thin_airfoil_theory():
    slope = get_airfoil("naca0012").summary()["cl_alpha_per_rad"]
    assert 5.5 < slope < 6.5, "should be close to 2 pi"


def test_viterna_is_bounded_near_zero():
    """The a2 cos^2/sin term is singular at alpha=0 and must stay clamped."""
    alpha = np.linspace(-0.2, 0.2, 401)
    cl, cd = viterna_extrapolate(0.24, 1.4, 0.05, 12.0, alpha)
    assert np.isfinite(cl).all() and np.abs(cl).max() < 5.0
    assert np.isfinite(cd).all()


def test_reynolds_correction_raises_drag_at_low_re():
    polar = get_airfoil("clarky")
    a = np.array([5.0 * DEG])
    _, cd_high = polar.evaluate(a, reynolds=np.array([1.0e6]))
    _, cd_low = polar.evaluate(a, reynolds=np.array([2.0e4]))
    assert cd_low[0] > cd_high[0]
    assert cd_low[0] / cd_high[0] < 12.0, "the compounded factor must stay clamped"


def test_mach_correction_adds_wave_drag():
    polar = get_airfoil("naca16509")
    a = np.array([3.0 * DEG])
    _, cd_sub = polar.evaluate(a, mach=np.array([0.3]))
    _, cd_tran = polar.evaluate(a, mach=np.array([0.9]))
    assert cd_tran[0] > cd_sub[0] * 1.5


def test_table_lookup_matches_direct_evaluation():
    polar = get_airfoil("clarky")
    table = polar.tabulate(2881)
    alpha = np.linspace(-math.pi, math.pi, 997)
    cl_ref, cd_ref = polar.base_polar(alpha)
    cl_tab, cd_tab = table.lookup(alpha)
    assert np.abs(cl_tab - cl_ref).max() < 5e-3
    assert np.abs(cd_tab - cd_ref).max() < 5e-3


def test_stacked_tables_have_the_kernel_layout():
    polars = [get_airfoil("clarky"), get_airfoil("e63"), get_airfoil("naca0012")]
    alpha, cl, cd = stack_tables(polars, 721)
    assert alpha.shape == (721,)
    assert cl.shape == cd.shape == (3, 721)
    assert cl.flags["C_CONTIGUOUS"] and cd.flags["C_CONTIGUOUS"]
    assert np.allclose(np.diff(alpha), alpha[1] - alpha[0]), "grid must be uniform"


def test_table_polar_extends_measured_data_to_the_full_circle():
    a = np.linspace(-10.0, 14.0, 25) * DEG
    cl = 6.0 * a
    cd = 0.01 + 0.02 * cl ** 2
    polar = TablePolar(name="synthetic", alpha_data=a, cl_data=cl, cd_data=cd)
    wide = np.linspace(-math.pi, math.pi, 1001)
    cl_w, cd_w = polar.base_polar(wide)
    assert np.isfinite(cl_w).all() and np.isfinite(cd_w).all()
    assert abs(float(polar.base_polar(np.array([0.0]))[0][0])) < 1e-6


def test_unknown_airfoil_raises():
    with pytest.raises(KeyError):
        get_airfoil("not-an-airfoil")
