import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from record_g009_r0_rev32_hd import sample_steps


def test_actual_state_sampling_is_full_speed_and_unique():
    steps = sample_steps(400)
    assert len(steps) == 240
    assert steps[0] == 0
    assert steps[-1] == 398
    assert len(set(steps)) == len(steps)
    assert set(b - a for a, b in zip(steps, steps[1:])) == {1, 2}
    assert len(steps) / 30 == 400 / 50
    assert max(abs(step / 50 - frame / 30) for frame, step in enumerate(steps)) <= 0.01


@pytest.mark.parametrize("horizon,fps", [(0, 30), (400, 60), (400, 0)])
def test_fake_fps_and_invalid_horizon_are_rejected(horizon, fps):
    with pytest.raises(ValueError):
        sample_steps(horizon, fps=fps)
