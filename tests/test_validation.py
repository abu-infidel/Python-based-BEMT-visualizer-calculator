"""The model against published Cessna 172N data (docs/VALIDATION.md)."""

from __future__ import annotations

import pytest

from propwash.validation import run_validation


@pytest.fixture(scope="module")
def checks():
    return run_validation()


def test_every_check_passes(checks):
    failed = [c.name for c in checks if not c.passed]
    assert not failed, failed


def test_propeller_only_predictions_are_within_eight_percent(checks):
    for c in checks:
        if c.kind == "validation":
            assert abs(c.error_pct) <= 8.0, c.name


def test_static_rpm_is_inside_the_certificate_range(checks):
    static = next(c for c in checks if c.kind == "calibration")
    assert 2280.0 <= static.model <= 2400.0


def test_top_speed_is_within_five_percent(checks):
    vmax = next(c for c in checks if "level speed" in c.name)
    assert abs(vmax.error_pct) <= 5.0
