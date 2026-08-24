from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PACKAGE = ModuleType("isaac_walk_g006")
PACKAGE.__path__ = [str(ROOT / "src" / "isaac_walk_g006")]
sys.modules.setdefault("isaac_walk_g006", PACKAGE)

from isaac_walk_g006.evaluation.protocol import (  # noqa: E402
    EXPECTED_SUCCESS_CRITERIA,
    body_xy_to_world,
    build_guardrail_trials,
    build_push_trials,
    tile_boundary_violation,
    validate_success_criteria,
    wilson_interval,
)


INITIAL_STATES = [{"id": index, "root_relative_pos_m": [0.0, 0.0, 0.4]} for index in range(10)]


def test_trial_matrices_are_exact_and_unique() -> None:
    push = build_push_trials(INITIAL_STATES)
    guardrail = build_guardrail_trials(INITIAL_STATES)
    assert len(push) == len({trial["trial_id"] for trial in push}) == 1080
    assert len({trial["stratum_id"] for trial in push}) == 108
    assert all(sum(trial["stratum_id"] == key for trial in push) == 10 for key in {item["stratum_id"] for item in push})
    assert len(guardrail) == len({trial["trial_id"] for trial in guardrail}) == 90


@pytest.mark.parametrize(
    ("quaternion", "body", "expected"),
    [
        ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0), (1.0, 0.0, 0.0)),
        ((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)), (1.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, 0.0, 1.0), (1.0, 0.0), (-1.0, 0.0, 0.0)),
    ],
)
def test_wxyz_yaw_only_body_to_world(quaternion, body, expected) -> None:
    assert body_xy_to_world(body, quaternion) == pytest.approx(expected, abs=1e-12)


def test_tile_boundary_is_strictly_greater_than_11_5() -> None:
    assert not tile_boundary_violation((11.5, -11.5), (0.0, 0.0))
    assert tile_boundary_violation((11.500001, 0.0), (0.0, 0.0))


def test_wilson_zero_denominator_and_known_bounds() -> None:
    assert wilson_interval(0, 0) == (None, None)
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.236593, rel=1e-5)
    assert high == pytest.approx(0.763407, rel=1e-5)


def test_success_criteria_is_exact_and_fail_closed() -> None:
    assert validate_success_criteria(dict(EXPECTED_SUCCESS_CRITERIA)) == EXPECTED_SUCCESS_CRITERIA
    missing = dict(EXPECTED_SUCCESS_CRITERIA)
    missing.pop("roll_abs_rad_max")
    with pytest.raises(ValueError, match="success_criteria"):
        validate_success_criteria(missing)
    changed = dict(EXPECTED_SUCCESS_CRITERIA, horizon_completed_step=599)
    with pytest.raises(ValueError, match="success_criteria"):
        validate_success_criteria(changed)
