from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PACKAGE = ModuleType("isaac_walk_g006")
PACKAGE.__path__ = [str(ROOT / "src" / "isaac_walk_g006")]
sys.modules.setdefault("isaac_walk_g006", PACKAGE)

from isaac_walk_g006.evaluation.protocol import PushRecoveryStateMachine  # noqa: E402


def machine() -> PushRecoveryStateMachine:
    return PushRecoveryStateMachine(0.30, 0.30, 0.35, 0.35)


def run_to(
    state: PushRecoveryStateMachine,
    end: int,
    *,
    good_from: int | None = None,
    terminate_at: int | None = None,
    auto_reset_at: int | None = None,
    roll_bad_at: set[int] | None = None,
    pitch_bad_at: set[int] | None = None,
) -> None:
    roll_bad_at = roll_bad_at or set()
    pitch_bad_at = pitch_bad_at or set()
    for completed_step in range(state.last_completed_step + 1, end + 1):
        if completed_step == 201:
            state.mark_push(200)
        good = good_from is not None and completed_step >= good_from
        state.observe(
            completed_step,
            tracking_error=0.0 if good else 1.0,
            angular_error=0.0 if good else 1.0,
            roll=0.4 if completed_step in roll_bad_at else 0.0,
            pitch=-0.4 if completed_step in pitch_bad_at else 0.0,
            terminated=completed_step == terminate_at,
            auto_reset_detected=completed_step == auto_reset_at,
        )
        if not state.active:
            break


def test_push_is_injected_after_200_completed_steps_and_sample_201_is_immediate() -> None:
    state = machine()
    run_to(state, 201, good_from=201)
    assert state.pushed is True
    assert state.sample_steps == [201]
    assert state.dwell == 1


def test_fall_before_push_counts_as_failed_recovery_and_physical_failure() -> None:
    state = machine()
    run_to(state, 99, terminate_at=99)
    result = state.finalize()
    assert result["eligible"] is True
    assert result["failed"] is True
    assert result["prepush_failure"] is True
    assert result["survived_to_horizon"] is False
    assert result["physical_failure"] is True


def test_auto_reset_poison_protocol_blocks_instead_of_quiet_exclusion() -> None:
    state = machine()
    run_to(state, 49, auto_reset_at=49)
    result = state.finalize()
    assert result["protocol_blocked"] is True
    assert result["excluded_reason"] == "auto_reset_poison"
    assert result["recovered"] is False


def test_25_step_dwell_may_end_at_completed_step_450_then_survive_to_600() -> None:
    state = machine()
    run_to(state, 600, good_from=426)
    result = state.finalize()
    assert result["criterion_met"] is True
    assert result["recovery_step"] == 450
    assert result["survived_to_horizon"] is True
    assert result["recovered"] is True


def test_dwell_starting_427_misses_window_but_physical_survival_is_true() -> None:
    state = machine()
    run_to(state, 600, good_from=427)
    result = state.finalize()
    assert result["criterion_met"] is False
    assert result["recovered"] is False
    assert result["recovery_failed"] is True
    assert result["survived_to_horizon"] is True
    assert result["physical_failure"] is False


def test_late_fall_after_criterion_revokes_final_recovered() -> None:
    state = machine()
    run_to(state, 300, good_from=201, terminate_at=300)
    result = state.finalize()
    assert result["criterion_met"] is True
    assert result["recovery_step"] == 225
    assert result["survived_to_horizon"] is False
    assert result["recovered"] is False
    assert result["failed"] is True


def test_roll_or_pitch_exceedance_resets_consecutive_dwell() -> None:
    state = machine()
    run_to(state, 600, good_from=201, roll_bad_at={220}, pitch_bad_at={240})
    result = state.finalize()
    assert result["criterion_met"] is True
    assert result["recovery_step"] == 265
    assert result["recovered"] is True


def test_postpush_tile_escape_is_protocol_blocked() -> None:
    state = machine()
    run_to(state, 200)
    state.mark_push(200)
    state.observe(201, boundary_violation=True)
    result = state.finalize()
    assert result["eligible"] is True
    assert result["protocol_blocked"] is True
    assert result["excluded_reason"] == "tile_boundary"
    assert result["recovered"] is False


def test_completed_step_600_sets_survival_even_without_recovery() -> None:
    state = machine()
    run_to(state, 600)
    result = state.finalize()
    assert result["criterion_met"] is False
    assert result["survived_to_horizon"] is True
    assert result["recovered"] is False
