from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rev31_attribution", ROOT / "scripts" / "bootstrap_train_g009_rev31_attribution.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeRecorder:
    def __init__(self, env):
        self.env = env
        self.active_terms = []
        self.seen_q = None

    def record_pre_reset(self, env_ids, force_export_or_skip=None):
        self.seen_q = self.env.scene["robot"].data.joint_pos.clone()
        self.env.scene["robot"].data.joint_pos[env_ids] = 0.0
        return "original-result"


def fake_env(*, nonfinite=False):
    joint_names = ["FL_calf_joint", "FR_calf_joint"]
    q = torch.tensor([[-2.6, 1.2], [-2.6, 0.0]], dtype=torch.float32)
    if nonfinite:
        q[0, 0] = float("nan")
    robot = SimpleNamespace(
        joint_names=joint_names,
        data=SimpleNamespace(
            joint_pos=q,
            joint_vel=torch.tensor([[0.3, -0.4], [0.1, 0.2]]),
            joint_pos_limits=torch.tensor([[[-2.5, 1.0], [-1.0, 1.0]]] * 2),
            applied_torque=torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
            root_pos_w=torch.tensor([[0.0, 0.0, 0.2], [1.0, 0.0, 0.2]]),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        ),
    )
    action_term = SimpleNamespace(
        _joint_names=joint_names,
        _joint_ids=[0, 1],
        processed_actions=torch.tensor([[-2.3, 0.8], [-2.4, 0.0]]),
    )
    contact = SimpleNamespace(
        body_names=["base", "FL_foot"],
        data=SimpleNamespace(net_forces_w=torch.tensor([[[0.0, 0.0, 98.1], [0.0, 0.0, 49.05]]] * 2)),
    )
    env = SimpleNamespace(
        scene={"robot": robot, "contact_forces": contact},
        action_manager=SimpleNamespace(get_term=lambda name: action_term),
        termination_manager=SimpleNamespace(get_term=lambda name: torch.tensor([True, True])),
        _g009_recover_fall_class=torch.tensor([0, 1]),
        _g009_r0_body_mass=torch.tensor([[10.0], [10.0]]),
        episode_length_buf=torch.tensor([7, 9]),
    )
    env.recorder_manager = FakeRecorder(env)

    def step(actions):
        result = env.recorder_manager.record_pre_reset(torch.tensor([0, 1]))
        return ("obs", "reward", "done", {"original": result})

    env.step = step
    return env


def test_capture_occurs_before_reset_preserves_results_and_rng() -> None:
    env = fake_env()
    observer = MODULE.TrainingAttributionObserver(env, event_cap=10)
    original_pre_reset, original_step = MODULE.install_pre_reset_observer(env, observer)
    rng_before = torch.get_rng_state().clone()

    result = env.step(torch.zeros((2, 2)))

    assert result == ("obs", "reward", "done", {"original": "original-result"})
    assert torch.equal(rng_before, torch.get_rng_state())
    assert observer.rng_neutral is True
    assert observer.control_step == 1
    assert observer.samples[0]["violations"][0]["q_rad"] == pytest.approx(-2.6)
    assert env.recorder_manager.seen_q[0, 0].item() == pytest.approx(-2.6)
    assert env.scene["robot"].data.joint_pos[0, 0].item() == 0.0
    assert callable(original_pre_reset) and callable(original_step)


def test_lower_upper_attribution_and_joint_vs_environment_counts() -> None:
    env = fake_env()
    observer = MODULE.TrainingAttributionObserver(env, event_cap=10)
    observer.control_step = 25
    observer.capture(torch.tensor([0, 1]))

    report = observer.report()
    assert report["hard_termination_env_count"] == 2
    assert report["hard_termination_joint_event_count"] == 3
    assert report["hard_termination_by_joint"] == {"FL_calf_joint": 2, "FR_calf_joint": 1}
    first = report["event_samples"][0]
    assert first["iteration_index_zero_based"] == 1
    assert first["iteration_number_one_based"] == 2
    assert first["episode_step"] == 7
    assert first["violations"][0]["limit_side"] == "lower"
    assert first["violations"][0]["hard_limit_excess_rad"] == pytest.approx(0.1)
    assert first["violations"][0]["q_rad"] < first["violations"][0]["lower_rad"]
    assert first["violations"][1]["limit_side"] == "upper"
    assert first["violations"][1]["hard_limit_excess_rad"] == pytest.approx(0.2)
    assert first["violations"][1]["q_rad"] > first["violations"][1]["upper_rad"]
    assert first["max_foot_net_contact_bodyweights"] == pytest.approx(0.5)
    assert first["max_nonfoot_net_contact_bodyweights"] == pytest.approx(1.0)


def test_sample_cap_keeps_totals_and_reports_overflow() -> None:
    env = fake_env()
    observer = MODULE.TrainingAttributionObserver(env, event_cap=1)
    observer.capture(torch.tensor([0, 1]))

    report = observer.report()
    assert report["hard_termination_env_count"] == 2
    assert report["hard_termination_joint_event_count"] == 3
    assert report["event_sample_count"] == 1
    assert report["event_sample_overflow_count"] == 1


def test_nonfinite_terminal_value_fails_closed_and_records_error() -> None:
    env = fake_env(nonfinite=True)
    observer = MODULE.TrainingAttributionObserver(env, event_cap=10)
    MODULE.install_pre_reset_observer(env, observer)

    with pytest.raises(RuntimeError, match="non-finite attribution value"):
        env.step(torch.zeros((2, 2)))

    assert observer.error is not None
    assert "non-finite attribution value" in observer.error


def test_exact_calf_hydra_override_is_stripped_before_official_trainer() -> None:
    argv = [
        "--task", MODULE.FIXED_TASK,
        "--num_envs", "1024",
        "--max_iterations", "50",
        "--seed", "42",
        "--device", "cuda:0",
        "--run_name", "go2_flat_rev31_s42_20260908-1023",
        "--headless",
        "env.events.reset_base.params.calf_angle=-2.28",
    ]
    custom, forwarded = MODULE.parse_and_strip_custom_args(argv)
    assert custom.calf_reset == -2.28
    assert not any(value.startswith(MODULE.CALF_HYDRA_PREFIX) for value in forwarded)


@pytest.mark.parametrize(
    "tokens",
    [
        ["env.events.reset_base.params.calf_angle=-2.20"],
        [
            "env.events.reset_base.params.calf_angle=-2.28",
            "env.events.reset_base.params.calf_angle=-2.37",
        ],
        ["agent.max_iterations=1"],
    ],
)
def test_unapproved_or_duplicate_hydra_overrides_are_rejected(tokens) -> None:
    argv = [
        "--task", MODULE.FIXED_TASK,
        "--num_envs", "1024",
        "--max_iterations", "50",
        "--seed", "42",
        "--device", "cuda:0",
        "--run_name", "go2_flat_rev31_s42_20260908-1023",
        "--headless",
        *tokens,
    ]
    with pytest.raises(ValueError):
        MODULE.parse_and_strip_custom_args(argv)
