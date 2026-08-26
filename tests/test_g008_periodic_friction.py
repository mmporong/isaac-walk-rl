from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_g008_periodic_friction", ROOT / "scripts" / "evaluate_g008_periodic_friction.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stripe_field_alternates_and_supports_phase_offsets() -> None:
    assert MODULE.stripe_is_low(0.00, 0.5)
    assert MODULE.stripe_is_low(0.49, 0.5)
    assert not MODULE.stripe_is_low(0.50, 0.5)
    assert not MODULE.stripe_is_low(0.99, 0.5)
    assert MODULE.stripe_is_low(1.00, 0.5)
    assert not MODULE.stripe_is_low(0.25, 0.5, phase_m=0.25)
    with pytest.raises(ValueError, match="positive"):
        MODULE.stripe_is_low(0.0, 0.0)


def test_default_sweep_is_ordered_and_physically_consistent() -> None:
    MODULE.validate_sweep(MODULE.DEFAULT_SWEEP)
    mixed = [case for case in MODULE.DEFAULT_SWEEP if case["mixed"]]
    assert [case["low_static"] for case in mixed] == sorted(
        (case["low_static"] for case in mixed), reverse=True
    )
    assert all(case["low_dynamic"] <= case["low_static"] for case in mixed)


def test_threshold_summary_uses_first_contiguous_failure() -> None:
    cases = [
        {
            "case_id": "uniform_nominal",
            "mixed": False,
            "low_static": 0.8,
            "low_dynamic": 0.6,
            "all_directions_gate_pass": True,
            "directions": [],
        },
        {
            "case_id": "mixed_070_050",
            "mixed": True,
            "low_static": 0.7,
            "low_dynamic": 0.5,
            "all_directions_gate_pass": True,
            "directions": [{"id": "forward", "gate_pass": True}],
        },
        {
            "case_id": "mixed_060_040",
            "mixed": True,
            "low_static": 0.6,
            "low_dynamic": 0.4,
            "all_directions_gate_pass": False,
            "directions": [
                {"id": "forward", "gate_pass": False},
                {"id": "right_turn", "gate_pass": True},
            ],
        },
        {
            "case_id": "mixed_050_030",
            "mixed": True,
            "low_static": 0.5,
            "low_dynamic": 0.3,
            "all_directions_gate_pass": False,
            "directions": [{"id": "forward", "gate_pass": False}],
        },
    ]
    summary = MODULE.summarize_threshold(cases)
    assert summary["contiguous_pass_floor"] == {
        "case_id": "mixed_070_050",
        "low_static": 0.7,
        "low_dynamic": 0.5,
    }
    assert summary["first_failure"]["case_id"] == "mixed_060_040"
    assert summary["first_failure"]["failed_directions"] == ["forward"]
    assert summary["monotonic_gate_sequence"] is True
