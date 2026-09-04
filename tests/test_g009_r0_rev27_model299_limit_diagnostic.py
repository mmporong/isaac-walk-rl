from __future__ import annotations

import copy
import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_g009_r0_rev26_model299_joint_limits.py"
SPEC = importlib.util.spec_from_file_location("g009_rejected_checkpoint_hard_limits", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def rejected_report() -> dict:
    return {
        "task": DIAGNOSTIC.DEFAULT_TASK,
        "seed": 42,
        "num_envs": 1024,
        "max_iterations": 300,
        "headless": True,
        "last_iteration": 299,
        "iteration_target": 300,
        "passed": False,
        "run_health_passed": False,
        "qualification_passed": None,
        "effective_hydra_overrides": [],
        "resume": {"enabled": False},
        "qualification_mode": {
            "enabled": True,
            "preflight_passed": True,
            "policy_qualification_status": "not_run",
        },
        "success_checks": dict(DIAGNOSTIC.EXPECTED_TRAINING_CHECKS),
        "training_safety_aggregate": {
            "hard_joint_limit": {"maximum": 0.125, "nonzero_sample_count": 57},
            "numeric_invalid": {"maximum": 0.0, "nonzero_sample_count": 0},
            "qualification_passed": False,
        },
    }


def test_rejected_contract_accepts_exact_single_failure() -> None:
    binding = DIAGNOSTIC.validate_rejected_training_contract(rejected_report())
    assert binding == {
        "hard_joint_limit_maximum": 0.125,
        "hard_joint_limit_nonzero_sample_count": 57,
        "numeric_invalid_maximum": 0.0,
        "only_failed_required_check": "qualification_training_safety_zero",
        "training_qualification_passed": False,
        "held_out_qualification_status": "not_run",
    }


def test_committed_rejected_report_matches_preregistered_hash_and_contract() -> None:
    report_path = ROOT / DIAGNOSTIC.PREREGISTRATION["training_report"]["path"]
    assert DIAGNOSTIC.qualification.file_sha256(report_path) == (
        DIAGNOSTIC.PREREGISTRATION["training_report"]["sha256"]
    )
    DIAGNOSTIC.validate_rejected_training_contract(
        DIAGNOSTIC.qualification._read_json(report_path)
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["success_checks"].update({"gpu_recovered_to_baseline": False}), "only"),
        (lambda value: value["success_checks"].update({"qualification_training_safety_zero": True}), "only"),
        (lambda value: value["training_safety_aggregate"]["hard_joint_limit"].update({"maximum": 0.0}), "positive"),
        (lambda value: value["training_safety_aggregate"]["hard_joint_limit"].update({"nonzero_sample_count": 0}), "positive"),
        (lambda value: value["training_safety_aggregate"]["numeric_invalid"].update({"maximum": 1.0}), "zero"),
        (lambda value: value.update({"passed": True}), "passed"),
        (lambda value: value.update({"qualification_passed": False}), "not run"),
    ],
)
def test_rejected_contract_fails_closed(mutation, message: str) -> None:
    report = copy.deepcopy(rejected_report())
    mutation(report)
    with pytest.raises(ValueError, match=message):
        DIAGNOSTIC.validate_rejected_training_contract(report)


def row(*, pose: str, env: int, step: int, joint: str, excess: float) -> dict:
    return {
        "pose_id": pose,
        "env_index": env,
        "rollout_step": step,
        "episode_step": step,
        "joint_name": joint,
        "threshold_excess_rad": excess,
    }


def test_aggregate_preserves_joint_order_counts_maxima_and_uncapped_totals() -> None:
    rows = [
        row(pose="supine", env=5, step=2, joint="hip", excess=0.03),
        row(pose="prone", env=0, step=1, joint="knee", excess=0.02),
        row(pose="prone", env=0, step=1, joint="hip", excess=0.01),
        row(pose="supine", env=5, step=3, joint="hip", excess=0.04),
    ]
    result = DIAGNOSTIC.aggregate_attribution_rows(rows, ("hip", "knee"), event_cap=2)
    assert result["joint_order"] == ["hip", "knee"]
    assert result["pose_joint_event_counts"]["prone"] == {"hip": 1, "knee": 1}
    assert result["pose_joint_event_counts"]["supine"] == {"hip": 2, "knee": 0}
    assert result["pose_joint_max_threshold_excess_rad"]["supine"]["hip"] == 0.04
    assert result["hard_limit_episode_count"] == 2
    assert result["hard_limit_joint_event_count"] == 4
    assert result["event_sample_count"] == 2
    assert result["event_sample_dropped_count"] == 2
    assert [(item["rollout_step"], item["joint_name"]) for item in result["event_samples"]] == [
        (1, "hip"),
        (1, "knee"),
    ]


def test_aggregate_rejects_invalid_order_row_and_cap() -> None:
    with pytest.raises(ValueError, match="unique"):
        DIAGNOSTIC.aggregate_attribution_rows([], ("hip", "hip"), 1)
    with pytest.raises(ValueError, match="event_cap"):
        DIAGNOSTIC.aggregate_attribution_rows([], ("hip",), -1)
    with pytest.raises(ValueError, match="pose/joint"):
        DIAGNOSTIC.aggregate_attribution_rows(
            [row(pose="unknown", env=0, step=1, joint="hip", excess=0.1)], ("hip",), 1
        )


def args(tmp_path: Path, **overrides) -> Namespace:
    checkpoint = tmp_path / "model_299.pt"
    training_report = tmp_path / "training.json"
    checkpoint.write_bytes(b"checkpoint")
    training_report.write_text("{}", encoding="utf-8")
    values = {
        "checkpoint": checkpoint,
        "training_report": training_report,
        "task": DIAGNOSTIC.DEFAULT_TASK,
        "seed": 42,
        "num_envs": 1024,
        "horizon_steps": 400,
        "action_mode": "stochastic",
        "device": "cuda:0",
        "event_sample_cap": 512,
        "headless": True,
        "output": tmp_path / "reports" / "runs" / "diagnostic.json",
    }
    values.update(overrides)
    return Namespace(**values)


def test_argument_constraints_and_no_overwrite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(DIAGNOSTIC, "REPO_ROOT", tmp_path)
    (tmp_path / "reports" / "runs").mkdir(parents=True)
    DIAGNOSTIC.validate_args(args(tmp_path))
    for override in (
        {"seed": 1042},
        {"num_envs": 256},
        {"horizon_steps": 399},
        {"action_mode": "deterministic"},
        {"device": "cpu"},
        {"headless": False},
        {"event_sample_cap": 511},
    ):
        with pytest.raises(ValueError):
            DIAGNOSTIC.validate_args(args(tmp_path, **override))
    existing = tmp_path / "reports" / "runs" / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        DIAGNOSTIC.validate_args(args(tmp_path, output=existing))
    assert existing.read_text(encoding="utf-8") == "keep"
    with pytest.raises(ValueError, match="direct child"):
        DIAGNOSTIC.validate_args(args(tmp_path, output=tmp_path / "reports" / "runs" / "nested" / "x.json"))
    with pytest.raises(ValueError, match="direct child"):
        DIAGNOSTIC.validate_args(args(tmp_path, output=tmp_path / "reports" / "runs" / "x.txt"))


def test_qualification_claim_invariants() -> None:
    report = {
        "status": "diagnostic_complete",
        "protocol_mode": "diagnostic_only",
        "qualification_eligible": False,
        "historical_training_event_attribution": False,
        "completion_is_safety_pass": False,
        "result": "hard_limit_reproduced",
        "claim_limits": {
            "official_evaluation": False,
            "success_rate_estimate": False,
            "checkpoint_qualification": False,
            "completion_is_safety_pass": False,
        },
    }
    DIAGNOSTIC.validate_diagnostic_report_claims(report)
    for key, unsafe in (
        ("status", "pass"),
        ("protocol_mode", "official_qualification"),
        ("qualification_eligible", True),
        ("historical_training_event_attribution", True),
        ("result", "pass"),
    ):
        modified = copy.deepcopy(report)
        modified[key] = unsafe
        with pytest.raises(ValueError):
            DIAGNOSTIC.validate_diagnostic_report_claims(modified)


@pytest.mark.parametrize(
    "result", ("hard_limit_reproduced", "soft_only_reproduced", "not_reproduced")
)
def test_all_preregistered_diagnostic_outcomes_are_non_qualifying(result: str) -> None:
    report = {
        "status": "diagnostic_complete",
        "protocol_mode": "diagnostic_only",
        "qualification_eligible": False,
        "historical_training_event_attribution": False,
        "completion_is_safety_pass": False,
        "result": result,
        "claim_limits": {
            "official_evaluation": False,
            "success_rate_estimate": False,
            "checkpoint_qualification": False,
            "completion_is_safety_pass": False,
        },
    }
    DIAGNOSTIC.validate_diagnostic_report_claims(report)


def test_pre_reset_wrapper_preserves_zero_terms_and_rng() -> None:
    class Recorder:
        active_terms = []

        def record_pre_reset(self, env_ids, force_export_or_skip=None):
            return (tuple(env_ids), force_export_or_skip)

    class Observer:
        rng_neutral = True

        def __init__(self) -> None:
            self.captured = []

        def capture(self, env_ids) -> None:
            self.captured.append(tuple(env_ids))

    recorder = Recorder()
    observer = Observer()
    original = DIAGNOSTIC.install_pre_reset_observer(recorder, observer)
    assert recorder.record_pre_reset([1, 3], "sentinel") == ((1, 3), "sentinel")
    assert observer.captured == [(1, 3)]
    assert observer.rng_neutral is True
    assert recorder.active_terms == []
    recorder.record_pre_reset = original
    assert recorder.record_pre_reset([2]) == ((2,), None)


def test_pre_reset_wrapper_rejects_active_recorder_terms() -> None:
    class Recorder:
        active_terms = ["unexpected"]

        def record_pre_reset(self, env_ids, force_export_or_skip=None):
            return None

    with pytest.raises(RuntimeError, match="zero active recorder terms"):
        DIAGNOSTIC.install_pre_reset_observer(Recorder(), object())


def fake_limit_env() -> SimpleNamespace:
    action_term = SimpleNamespace(
        _joint_ids=[0],
        _joint_names=["hip"],
        raw_actions=torch.tensor([[1.0], [0.0], [0.0], [0.0]]),
        processed_actions=torch.tensor([[0.8], [0.0], [0.0], [0.0]]),
    )
    robot = SimpleNamespace(
        joint_names=["hip"],
        data=SimpleNamespace(
            joint_pos=torch.tensor([[1.2], [1.2], [1.2], [1.2]]),
            soft_joint_pos_limits=torch.tensor([[[-1.0, 1.0]]] * 4),
            joint_pos_limits=torch.tensor([[[-1.0, 1.0]]] * 4),
            root_pos_w=torch.zeros((4, 3)),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4),
            root_lin_vel_w=torch.zeros((4, 3)),
            root_ang_vel_w=torch.zeros((4, 3)),
        ),
    )
    return SimpleNamespace(
        num_envs=4,
        scene={"robot": robot},
        action_manager=SimpleNamespace(get_term=lambda name: action_term),
        termination_manager=SimpleNamespace(
            get_term=lambda name: torch.tensor([True, False, False, False])
        ),
        _g009_recover_fall_class=torch.tensor([0, 1, 2, 3]),
        episode_length_buf=torch.tensor([1, 1, 1, 1]),
    )


def test_soft_sampling_includes_initial_reset_and_uses_fresh_active_mask() -> None:
    env = fake_limit_env()
    observer = DIAGNOSTIC.PreResetLimitObserver(env, event_cap=8)
    observer.sample_active_soft_limits(
        active=torch.tensor([True, False, False, False]), state_step=0
    )
    env.scene["robot"].data.joint_pos[:] = 1.3
    observer.sample_active_soft_limits(
        active=torch.tensor([False, True, False, False]), state_step=1
    )
    aggregate = observer.aggregate()["soft_joint_limit"]
    assert aggregate["initial_reset_state_sampled"] is True
    assert aggregate["sampled_state_steps"] == [0, 1]
    assert aggregate["pose_joint_sample_counts"]["prone"]["hip"] == 1
    assert aggregate["pose_joint_sample_counts"]["supine"]["hip"] == 1
    assert aggregate["episode_count"] == 2


def test_terminal_event_separates_unclipped_sample_clipped_action_and_target() -> None:
    env = fake_limit_env()
    env.scene["robot"].data.joint_pos[0, 0] = 1.02
    observer = DIAGNOSTIC.PreResetLimitObserver(env, event_cap=8)
    observer.sample_active_soft_limits(active=torch.tensor([True] * 4), state_step=0)
    observer.set_step_context(
        active=torch.tensor([True] * 4),
        actions=torch.tensor([[2.5], [0.0], [0.0], [0.0]]),
        rollout_step=1,
    )
    observer.capture(torch.tensor([0]))
    event = observer.aggregate()["event_samples"][0]
    assert event["sampled_policy_action"] == 2.5
    assert event["sampled_policy_action_stage"] == "before RslRlVecEnvWrapper clamp"
    assert event["clipped_normalized_action"] == 1.0
    assert event["clipped_normalized_action_stage"] == "ActionManager raw_actions after wrapper clamp"
    assert event["processed_target_rad"] == pytest.approx(0.8)
    assert "last_policy_action" not in event


def test_invalid_cpu_binding_does_not_construct_app_launcher(monkeypatch, tmp_path: Path) -> None:
    constructed = []

    class FakeAppLauncher:
        def __init__(self, args) -> None:
            constructed.append(args)

    isaaclab = types.ModuleType("isaaclab")
    app = types.ModuleType("isaaclab.app")
    setattr(app, "AppLauncher", FakeAppLauncher)
    setattr(isaaclab, "app", app)
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)
    parsed = SimpleNamespace(
        training_report=tmp_path / "invalid.json",
        checkpoint=tmp_path / "invalid.pt",
    )
    monkeypatch.setattr(DIAGNOSTIC, "parse_args", lambda argv: parsed)
    monkeypatch.setattr(
        DIAGNOSTIC,
        "validate_rejected_training_binding",
        lambda report, checkpoint: (_ for _ in ()).throw(ValueError("invalid binding")),
    )
    with pytest.raises(ValueError, match="invalid binding"):
        DIAGNOSTIC.main([])
    assert constructed == []


def test_checkpoint_toctou_mismatch_does_not_call_runner_load(monkeypatch, tmp_path: Path) -> None:
    class Runner:
        def __init__(self) -> None:
            self.load_calls = []

        def load(self, path: str) -> None:
            self.load_calls.append(path)

    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"replacement")
    runner = Runner()
    monkeypatch.setattr(
        DIAGNOSTIC.qualification,
        "file_sha256",
        lambda path: "replacement-sha256",
    )
    with pytest.raises(ValueError, match="before runner.load"):
        DIAGNOSTIC.load_prevalidated_checkpoint(
            runner,
            checkpoint,
            {"checkpoint": {"sha256": "pre-app-bound-sha256"}},
        )
    assert runner.load_calls == []


def test_rev27_preregistration_fixes_hashes_runtime_and_claim_limits() -> None:
    prereg = DIAGNOSTIC.PREREGISTRATION
    assert prereg["training_report"]["sha256"] == (
        "71a3b45129b79f2beaed14fab486423aafdd0f92e860022cf22c2c6242391234"
    )
    assert prereg["checkpoint"]["sha256"] == (
        "75b38ac4c8f8b2ed17d73893350c1ca484ca3ef3c0d633273553e13efdf44c95"
    )
    assert prereg["runtime"] == {
        "task": DIAGNOSTIC.DEFAULT_TASK,
        "seed": 42,
        "forbidden_held_out_seed": 1042,
        "num_envs": 1024,
        "environments_per_pose": 256,
        "poses": ["prone", "supine", "left_side", "right_side"],
        "horizon_steps": 400,
        "action_mode": "stochastic",
        "device": "cuda:0",
        "headless": True,
        "policy_updates": 0,
        "optimizer_updates": 0,
    }
    assert not any(prereg["claim_limits"].values())
    assert prereg["diagnostic_source_binding_path_manifest_sha256"] == (
        "35ef5f9201560da6e294039a6d88c60b39653e6ccb6d9e4418a89e69cc496505"
    )
