import importlib.util
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_g008_directions", ROOT / "scripts" / "evaluate_g008_directions.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

RECORD_SPEC = importlib.util.spec_from_file_location(
    "record_g008_directions", ROOT / "scripts" / "record_g008_directions.py"
)
RECORD_MODULE = importlib.util.module_from_spec(RECORD_SPEC)
assert RECORD_SPEC.loader is not None
RECORD_SPEC.loader.exec_module(RECORD_MODULE)


def test_protocol_requires_four_way_balanced_environment_count():
    MODULE.validate_protocol(64, 250, 50)
    with pytest.raises(ValueError, match="multiple"):
        MODULE.validate_protocol(63, 250, 50)
    with pytest.raises(ValueError, match="warmup"):
        MODULE.validate_protocol(64, 250, 250)


def test_direction_sign_check_covers_forward_reverse_and_yaw():
    assert MODULE.command_sign_pass((0.6, 0.0, 0.0), (0.2, 0.1, -0.1))
    assert MODULE.command_sign_pass((-0.4, 0.0, 0.0), (-0.2, 0.0, 0.0))
    assert MODULE.command_sign_pass((0.0, 0.0, 0.5), (0.0, 0.0, 0.2))
    assert MODULE.command_sign_pass((0.0, 0.0, -0.5), (0.0, 0.0, -0.2))
    assert not MODULE.command_sign_pass((-0.4, 0.0, 0.0), (0.2, 0.0, 0.0))


def test_finalization_applies_tracking_and_attitude_gate():
    accumulator = MODULE.new_accumulator(4)
    accumulator.update(
        {
            "sample_count": 4,
            "linear_error_sq_sum": 0.04,
            "yaw_error_sq_sum": 0.04,
            "achieved_vx_sum": 2.0,
            "torque_norm_sum": 8.0,
            "mechanical_power_sum": 12.0,
            "roll_abs_max": 0.2,
            "pitch_abs_max": 0.3,
        }
    )
    result = MODULE.finalize_accumulator(accumulator, (0.6, 0.0, 0.0))
    assert result["linear_tracking_rmse_mps"] == pytest.approx(0.1)
    assert result["yaw_tracking_rmse_radps"] == pytest.approx(0.1)
    assert result["torque_l2_norm_mean_nm"] == pytest.approx(2.0)
    assert result["gate_pass"] is True

    accumulator["roll_abs_max"] = 0.36
    assert MODULE.finalize_accumulator(accumulator, (0.6, 0.0, 0.0))["gate_pass"] is False


def test_recording_sequence_has_all_directions_and_stays_inside_episode():
    names = [name for name, _, _ in RECORD_MODULE.SEQUENCE]
    assert [name for name in names if name != "stand"] == ["forward", "backward", "left_turn", "right_turn"]
    assert sum(length for _, length, _ in RECORD_MODULE.SEQUENCE) == 900
    assert RECORD_MODULE.command_at_step(50) == ("forward", (0.6, 0.0, 0.0))
    assert RECORD_MODULE.command_at_step(899) == ("right_turn", (0.0, 0.0, -0.5))
    with pytest.raises(IndexError):
        RECORD_MODULE.command_at_step(900)
