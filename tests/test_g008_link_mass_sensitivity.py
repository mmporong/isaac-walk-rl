from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "evaluate_g008_link_mass_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("evaluate_g008_link_mass_sensitivity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mass_case_matrix_covers_four_groups_without_nominal_duplicates() -> None:
    cases = MODULE.MASS_CASES
    assert cases[0] == {"id": "nominal", "group": None, "factor": 1.0}
    assert len(cases) == 25
    for group in MODULE.LINK_GROUP_PATTERNS:
        group_cases = [case for case in cases if case["group"] == group]
        assert [case["factor"] for case in group_cases] == list(MODULE.MASS_FACTORS)


def test_protocol_requires_balanced_policy_case_direction_assignment() -> None:
    MODULE.validate_protocol(800, 300, 50)
    with pytest.raises(ValueError):
        MODULE.validate_protocol(700, 300, 50)


def test_group_summary_keeps_failed_directions() -> None:
    cases = [
        {
            "group": "hip",
            "factor": 0.8,
            "all_directions_gate_pass": False,
            "directions": [
                {"id": "forward", "gate_pass": True},
                {"id": "right_turn", "gate_pass": False},
            ],
        },
        {
            "group": "hip",
            "factor": 1.2,
            "all_directions_gate_pass": True,
            "directions": [{"id": "forward", "gate_pass": True}],
        },
    ]
    summary = MODULE.summarize_group_cases(cases)
    assert summary["hip"]["passing_factors"] == [1.2]
    assert summary["hip"]["failing_factors"] == [
        {"factor": 0.8, "failed_directions": ["right_turn"]}
    ]
