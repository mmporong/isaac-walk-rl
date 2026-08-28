from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaac_walk_g009.recover_contracts import StableHoldLatch


def test_vectorized_latch_requires_consecutive_steps_and_emits_one_shot_pulse() -> None:
    latch = StableHoldLatch(num_envs=3, required_consecutive_steps=3)

    state, pulse = latch.update([True, True, False])
    assert state == (False, False, False)
    assert pulse == (False, False, False)
    assert latch.counters == (1, 1, 0)

    latch.update([True, False, True])
    state, pulse = latch.update([True, True, True])
    assert state == (True, False, False)
    assert pulse == (True, False, False)
    assert latch.counters == (3, 1, 2)

    state, pulse = latch.update([True, True, True])
    assert state == (True, False, True)
    assert pulse == (False, False, True)

    state, pulse = latch.update([True, True, True])
    assert state == (True, True, True)
    assert pulse == (False, True, False)


def test_false_observation_breaks_the_consecutive_run_without_revoking_a_latch() -> None:
    latch = StableHoldLatch(num_envs=1, required_consecutive_steps=2)
    latch.update([True])
    latch.update([False])
    assert latch.counters == (0,)
    assert latch.latched == (False,)

    latch.update([True])
    assert latch.update([True]) == ((True,), (True,))
    assert latch.update([False]) == ((True,), (False,))
    assert latch.counters == (0,)


def test_reset_can_clear_selected_environments_or_the_whole_batch() -> None:
    latch = StableHoldLatch(num_envs=3, required_consecutive_steps=1)
    latch.update([True, True, True])

    latch.reset([0, 2])
    assert latch.latched == (False, True, False)
    assert latch.counters == (0, 1, 0)

    latch.reset()
    assert latch.latched == (False, False, False)
    assert latch.counters == (0, 0, 0)


def test_nonfinite_input_fails_closed_per_environment() -> None:
    latch = StableHoldLatch(num_envs=3, required_consecutive_steps=2)
    latch.update([True, True, True])
    state, pulse = latch.update([True, math.nan, math.inf])

    assert state == (True, False, False)
    assert pulse == (True, False, False)
    assert latch.counters == (2, 0, 0)


@pytest.mark.parametrize("kwargs", [{"num_envs": 0}, {"num_envs": 1, "required_consecutive_steps": 0}])
def test_invalid_latch_dimensions_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        StableHoldLatch(**kwargs)


def test_update_and_reset_validate_vector_shape_and_indices() -> None:
    latch = StableHoldLatch(num_envs=2, required_consecutive_steps=2)
    with pytest.raises(ValueError, match="length"):
        latch.update([True])
    with pytest.raises(IndexError):
        latch.reset([2])
