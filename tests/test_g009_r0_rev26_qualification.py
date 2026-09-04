from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = _load("evaluate_g009_r0_rev26", "scripts/evaluate_g009_r0.py")
summary = _load("summarize_g009_r0_rev26", "scripts/summarize_g009_r0_rev26_qualification.py")
bootstrap = _load("bootstrap_train_g009_rev26", "scripts/bootstrap_train_g009.py")


def _pose(successes: int = 205) -> dict:
    recovery_samples = [4.0] * successes
    return {
        "episode_count": 256,
        "success_count": successes,
        "success_rate": successes / 256,
        "termination_counts": {
            "stable_success": successes,
            "time_out": 256 - successes,
            "numeric_invalid": 0,
            "hard_joint_limit": 0,
            "other": 0,
        },
        "recovery_time_s": {
            "count": successes,
            "min": 4.0,
            "median": 4.0,
            "mean": 4.0,
            "p95": 4.0,
            "max": 4.0,
        },
        "recovery_time_samples_s": recovery_samples,
        "max_raw_hard_joint_limit_violation_rad": 0.0,
        "gate_pass": successes >= 205,
    }


def _evaluation_report(training_binding: dict, checkpoint_sha256: str) -> dict:
    poses = [{"pose_id": pose, **_pose()} for pose in evaluation.POSE_NAMES]
    return {
        "status": "pass",
        "protocol_mode": "official_qualification",
        "official_protocol": evaluation.OFFICIAL_PROTOCOL,
        "task": evaluation.DEFAULT_TASK,
        "seed": 1042,
        "num_envs": 1024,
        "episodes_per_pose": 256,
        "observation_corruption": True,
        "checkpoint": {"sha256": checkpoint_sha256},
        "training_binding": training_binding,
        "source_state": {
            "before": {"commit": training_binding["repository"]["commit"], "clean": True},
            "after": {"commit": training_binding["repository"]["commit"], "clean": True},
        },
        "source_bindings": {
            "evaluator": {
                "path": "scripts/evaluate_g009_r0.py",
                "sha256": evaluation.file_sha256(ROOT / "scripts" / "evaluate_g009_r0.py"),
            },
            "config": {
                "path": "configs/g009_r0.json",
                "sha256": evaluation.file_sha256(ROOT / "configs" / "g009_r0.json"),
            },
        },
        "poses": poses,
        "aggregate": {
            "episode_count": 1024,
            "success_count": 820,
            "success_rate": 820 / 1024,
            "safety_termination_count": 0,
            "other_termination_count": 0,
            "all_pose_gate_pass": True,
        },
        "physics_readback": {"policy_observation_dim": 140, "action_dim": 12},
    }


def test_rev26_preregistration_is_the_authoritative_contract() -> None:
    config = evaluation._load_qualification_config(
        ROOT / "configs" / "g009_r0_rev26_qualification.json"
    )

    assert config["task"] == evaluation.DEFAULT_TASK
    assert config["evaluation"]["seed"] == 1042
    assert config["evaluation"]["environments_per_pose"] == 256
    assert tuple(config["source_binding_paths"]) == evaluation.QUALIFICATION_SOURCE_PATHS


def test_rev26_preregistration_rejects_contract_and_manifest_mutation(tmp_path: Path) -> None:
    source = json.loads(
        (ROOT / "configs" / "g009_r0_rev26_qualification.json").read_text(encoding="utf-8")
    )
    path = tmp_path / "rev26.json"
    source["evaluation"]["seed"] = 42
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation contract mismatch: seed"):
        evaluation._load_qualification_config(path)

    source["evaluation"]["seed"] = 1042
    source["source_binding_paths"].append("z.py")
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest mismatch"):
        evaluation._load_qualification_config(path)

    source["source_binding_path_manifest_sha256"] = evaluation.hashlib.sha256(
        json.dumps(source["source_binding_paths"], ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest mismatch"):
        evaluation._load_qualification_config(path)


def test_rev26_preregistration_rejects_nested_training_contract_mutation(tmp_path: Path) -> None:
    source = json.loads(
        (ROOT / "configs" / "g009_r0_rev26_qualification.json").read_text(encoding="utf-8")
    )
    source["training"]["ppo_num_mini_batches"] = 8
    path = tmp_path / "rev26.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="training contract mismatch: ppo_num_mini_batches"):
        evaluation._load_qualification_config(path)


def test_rev26_evaluation_accepts_exact_205_of_256_boundary() -> None:
    binding = {"repository": {"commit": "c" * 40}}
    checks = summary.validate_evaluation_report(
        _evaluation_report(binding, "a" * 64),
        checkpoint_sha256="a" * 64,
        training_binding=binding,
    )

    assert all(checks.values())


@pytest.mark.parametrize(
    ("mutation", "failed_gate"),
    [
        (lambda report: report["poses"][0].update(episode_count=255), "pose_prone"),
        (lambda report: report["poses"][1]["recovery_time_s"].update(count=204), "pose_supine_reported_consistent"),
        (lambda report: report.update(observation_corruption=False), "actor_corruption_enabled"),
        (lambda report: report["physics_readback"].update(policy_observation_dim=83), "policy_observation_dimension"),
        (lambda report: report["physics_readback"].update(action_dim=11), "action_dimension"),
        (lambda report: report["source_state"]["after"].update(clean=False), "evaluation_source_stable"),
        (lambda report: report.update(source_bindings={}), "evaluation_sources_bound"),
        (lambda report: report["poses"].__setitem__(1, report["poses"][0]), "pose_structure"),
        (lambda report: report["poses"][0].update(success_rate=1.0), "pose_prone_reported_consistent"),
        (lambda report: report["aggregate"].update(success_count=999), "aggregate_recomputed"),
        (lambda report: report["poses"][0]["termination_counts"].update(other=1), "pose_prone"),
    ],
)
def test_rev26_evaluation_rejects_fail_open_mutations(mutation, failed_gate: str) -> None:
    binding = {"repository": {"commit": "c" * 40}}
    report = _evaluation_report(binding, "a" * 64)
    mutation(report)

    checks = summary.validate_evaluation_report(
        report, checkpoint_sha256="a" * 64, training_binding=binding
    )

    assert checks[failed_gate] is False


def test_rev26_summary_publish_is_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    summary.write_json_no_overwrite(output, {"status": "pass"})

    with pytest.raises(FileExistsError):
        summary.write_json_no_overwrite(output, {"status": "replacement"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "pass"}


def test_bootstrap_validates_pinned_commit_and_train_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = tmp_path / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("print('train')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "scripts/reinforcement_learning/rsl_rl/train.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    monkeypatch.setattr(bootstrap, "EXPECTED_ISAACLAB_COMMIT", commit)
    monkeypatch.setattr(bootstrap, "EXPECTED_TRAIN_SHA256", bootstrap._file_sha256(upstream))

    assert bootstrap.validate_upstream(tmp_path) == upstream.resolve()

    upstream.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked worktree"):
        bootstrap.validate_upstream(tmp_path)

    subprocess.run(["git", "add", "scripts/reinforcement_learning/rsl_rl/train.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "tampered"], cwd=tmp_path, check=True)
    tampered_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    monkeypatch.setattr(bootstrap, "EXPECTED_ISAACLAB_COMMIT", tampered_commit)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        bootstrap.validate_upstream(tmp_path)


def test_bootstrap_git_failure_preserves_exit_and_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=23, stdout="partial stdout\n", stderr="fatal: fixture failure\n"
        ),
    )

    with pytest.raises(RuntimeError, match=r"exit=23.*fatal: fixture failure"):
        bootstrap._git_stdout(tmp_path, "status", "--porcelain=v1")


def test_evaluator_revalidates_upstream_source_and_cleanliness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = tmp_path / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("print('train')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "scripts/reinforcement_learning/rsl_rl/train.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    digest = evaluation.file_sha256(upstream)
    monkeypatch.setattr(evaluation, "EXPECTED_ISAACLAB_COMMIT", commit)
    monkeypatch.setattr(evaluation, "EXPECTED_TRAIN_SHA256", digest)
    binding = {
        "isaac_lab_expected_commit": commit,
        "isaac_lab_commit": commit,
        "official_train_path": str(upstream),
        "official_train_expected_sha256": digest,
        "official_train_sha256": digest,
        "tracked_clean": True,
    }

    assert evaluation._validate_upstream_binding(binding)["tracked_clean"] is True

    upstream.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked_clean"):
        evaluation._validate_upstream_binding(binding)
