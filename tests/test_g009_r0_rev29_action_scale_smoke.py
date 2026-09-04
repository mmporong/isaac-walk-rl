from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "configs" / "g009_r0_rev29_action_scale_smoke.json"
HARNESS = ROOT / "scripts" / "run_training.ps1"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = _load(
    "validate_g009_r0_rev29_action_scale_smoke_test",
    "scripts/validate_g009_r0_rev29_action_scale_smoke.py",
)
summary = _load(
    "summarize_g009_r0_rev29_action_scale_smoke_test",
    "scripts/summarize_g009_r0_rev29_action_scale_smoke.py",
)


def test_preregistration_locks_single_variable_budget_and_claim_order() -> None:
    preregistration = validation.load_preregistration()
    readback = validation.validate_semantics(preregistration)

    assert preregistration["single_experimental_variable"] == {
        "name": "normalized_joint_position_action_scale",
        "rejected_rev28_value": 0.7,
        "candidate_value": 0.65,
    }
    assert readback["action_scale"] == 0.65
    assert readback["agent_yaml"]["entropy_coef"] == 0.0
    assert preregistration["training"]["transitions"] == 1024 * 24 * 50 == 1_228_800
    assert preregistration["training"]["optimizer_mini_batch_updates"] == 50 * 5 * 4 == 1000
    assert preregistration["claim_limits"]["action_scale_change_proved_causal"] is False
    assert preregistration["execution_order"] == {
        "prelaunch_validator_required": True,
        "smoke_must_pass_before_full_300_iteration_training": True,
        "held_out_seed_1042_forbidden_until_full_300_training_safety_zero": True,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["single_experimental_variable"].update(candidate_value=0.66),
        lambda value: value["frozen_contract"].update(ppo_entropy_coefficient=0.01),
        lambda value: value["runtime_readback"]["env_yaml"].update(action_scale=0.7),
    ],
)
def test_preregistration_rejects_variable_or_frozen_mutation(mutation) -> None:
    preregistration = validation.load_preregistration()
    mutation(preregistration)
    with pytest.raises(ValueError):
        validation.validate_semantics(preregistration)


def test_canonical_manifest_matches_rev29_contract() -> None:
    binding = validation.validate_canonical_manifest()
    manifest = json.loads((ROOT / "configs" / "g009_r0.json").read_text(encoding="utf-8"))
    assert binding["sha256"] == validation.file_sha256(ROOT / "configs" / "g009_r0.json")
    assert manifest["contract"]["contract_id"] == "g009_r0_recover_rev29"
    assert manifest["contract"]["action"]["scale"] == 0.65
    assert manifest["contract"]["ppo"]["entropy_coefficient"] == 0.0


def test_historical_evidence_is_hash_bound() -> None:
    evidence = validation.validate_historical_evidence(validation.load_preregistration())
    assert set(evidence) == {
        "rev27_diagnostic_report",
        "rev28_training_report",
        "rev28_rejection_report",
    }


def _zero_series() -> dict:
    return {
        "sample_count": 50,
        "latest": 0.0,
        "minimum": 0.0,
        "maximum": 0.0,
        "mean": 0.0,
        "nonzero_sample_count": 0,
    }


def _git_fixture_run(command, **_kwargs):
    if "show" in command:
        relative = command[-1].split(":", 1)[1]
        return SimpleNamespace(returncode=0, stdout=(ROOT / relative).read_bytes())
    return SimpleNamespace(returncode=0, stdout=b"")


def _raw_report(checkpoint: Path) -> dict:
    preregistration = validation.load_preregistration()
    paths = preregistration["source_binding_paths"]
    files = {path: summary.file_sha256(ROOT / path) for path in paths}
    bundle_payload = "\n".join(f"{path}:{files[path]}" for path in paths)
    bundle = hashlib.sha256(bundle_payload.encode()).hexdigest()
    commit = "c" * 40
    run_directory = checkpoint.parent
    params = run_directory / "params"
    params.mkdir(parents=True, exist_ok=True)
    agent_yaml = params / "agent.yaml"
    agent_yaml.write_text(
        "seed: 42\ndevice: cuda:0\nnum_steps_per_env: 24\nmax_iterations: 50\n"
        "policy:\n  init_noise_std: 0.5\nalgorithm:\n  entropy_coef: 0.0\n"
        "  num_learning_epochs: 5\n  num_mini_batches: 4\n",
        encoding="utf-8",
    )
    env_yaml = params / "env.yaml"
    env_yaml.write_text(
        "actions:\n  joint_pos:\n    scale: 0.65\n    rescale_to_limits: true\n    alpha: 0.2\n"
        "scene:\n  robot:\n    soft_joint_pos_limit_factor: 0.9\n    spawn:\n"
        "      articulation_props:\n        solver_position_iteration_count: 8\n"
        "        solver_velocity_iteration_count: 0\n      rigid_props:\n"
        "        max_depenetration_velocity: 1.0\n",
        encoding="utf-8",
    )
    snapshot = {"repository_commit": commit, "sha256": bundle, "files": files}
    static = {
        "action_scale": 0.65,
        "agent_yaml": copy.deepcopy(preregistration["runtime_readback"]["agent_yaml"]),
        "env_yaml": copy.deepcopy(preregistration["runtime_readback"]["env_yaml"]),
    }
    return {
        "task": preregistration["training"]["task"],
        "num_envs": 1024,
        "max_iterations": 50,
        "seed": 42,
        "headless": True,
        "resume": {"enabled": False, "load_run": None, "checkpoint": None},
        "effective_hydra_overrides": [],
        "qualification_mode": {"enabled": False, "preflight_passed": None, "policy_qualification_status": "not_run"},
        "entropy_smoke_mode": {"enabled": False, "preflight_passed": None, "runtime_algorithm_entropy_coef": None, "held_out_evaluation_status": None, "full_300_iteration_training_status": None, "policy_qualification_status": "not_run"},
        "action_scale_smoke_mode": {
            "enabled": True,
            "preflight_passed": True,
            "runtime_action_scale": 0.65,
            "held_out_evaluation_status": "forbidden_until_full_300_training_safety_zero",
            "full_300_iteration_training_status": "forbidden_until_smoke_accepted",
            "policy_qualification_status": "not_run",
        },
        "repository": {"commit": commit, "dirty": False},
        "source_bundle": {
            "sha256": bundle,
            "hash_domain": "executed_worktree_bytes",
            "matches_repository_commit": True,
            "files": files,
            "commit_blob_sha256": {"bundle": bundle, "files": files},
            "prelaunch": {**snapshot, "hash_domain": "executed_worktree_bytes", "matches_validated_snapshot": True},
            "postrun": {**snapshot, "hash_domain": "executed_worktree_bytes", "stable": True},
        },
        "action_scale_smoke_contract": {
            "path": "configs/g009_r0_rev29_action_scale_smoke.json",
            "sha256": summary.file_sha256(PREREGISTRATION),
            "source_binding_path_manifest_sha256": preregistration["source_binding_path_manifest_sha256"],
            "prelaunch_validation": {
                "schema_version": "g009.r0.rev29.action_scale_smoke_prelaunch_validation.v1",
                "status": "pass",
                "evidence_id": "G009-5-E022",
                "preregistration": {"path": "configs/g009_r0_rev29_action_scale_smoke.json", "sha256": summary.file_sha256(PREREGISTRATION)},
                "canonical_static_readback": static,
                "source_state": {
                    "repository_commit": commit,
                    "repository_clean": True,
                    "source_paths_logically_equal_to_head": True,
                    "executed_worktree_sha256": {"bundle": bundle, "files": files},
                    "commit_blob_sha256": {"bundle": bundle, "files": files},
                },
                "upstream": {
                    "isaac_lab_commit": "90b79bb2d44feb8d833f260f2bf37da3487180ba",
                    "tracked_clean": True,
                    "official_train_sha256": "8b995f75ac57ce7403973ff1f3f2715fbff9563ef2cdcdc321a7edc5dd15f5df",
                },
            },
        },
        "tensorboard": {
            "series_summary": {
                "Episode_Termination/hard_joint_limit": _zero_series(),
                "Episode_Termination/numeric_invalid": _zero_series(),
                "Policy/mean_noise_std": {"sample_count": 50, "latest": 0.48, "minimum": 0.47, "maximum": 0.5, "mean": 0.485, "nonzero_sample_count": 50},
            }
        },
        "training_safety_gate": {"required": True, "passed": True},
        "gpu": {"protected_run_safety": {"required": True, "passed": True, "mode": "action_scale_smoke", "temperature_threshold_c": 90.0, "sustained_sample_count": 3, "fatal_matches": [], "descendants_exited": True}},
        "artifacts": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": summary.file_sha256(checkpoint),
            "tensorboard_directory": str(run_directory),
            "agent_yaml": str(agent_yaml),
            "agent_yaml_sha256": summary.file_sha256(agent_yaml),
            "env_yaml": str(env_yaml),
            "env_yaml_sha256": summary.file_sha256(env_yaml),
        },
        "runtime_agent_config": {"source": "official train.py params/agent.yaml", "readback": copy.deepcopy(preregistration["runtime_readback"]["agent_yaml"]), "passed": True},
        "runtime_env_config": {"source": "official train.py params/env.yaml", "readback": copy.deepcopy(preregistration["runtime_readback"]["env_yaml"]), "passed": True},
        "success_checks": {
            "process_exit_zero": True,
            "no_traceback_or_error": True,
            "requested_iteration_reached": True,
            "log_directory_exists": True,
            "tensorboard_exists": True,
            "checkpoint_exists": True,
            "gpu_measurement_complete": True,
            "gpu_recovered_to_baseline": True,
            "qualification_training_safety_zero": None,
            "qualification_gpu_safety": None,
            "entropy_smoke_training_safety_zero": None,
            "entropy_smoke_gpu_safety": None,
            "entropy_smoke_source_snapshot_stable": None,
            "entropy_smoke_agent_yaml_readback": None,
            "action_scale_smoke_training_safety_zero": True,
            "action_scale_smoke_gpu_safety": True,
            "action_scale_smoke_source_snapshot_stable": True,
            "action_scale_smoke_agent_yaml_readback": True,
            "action_scale_smoke_env_yaml_readback": True,
            "requested_training_safety_gate_zero": None,
        },
        "run_health_passed": True,
        "passed": True,
        "qualification_passed": None,
    }


def test_summary_accepts_exact_zero_and_binds_runtime_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "run" / "model_49.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(summary, "checkpoint_std_vector", lambda _path: ([0.5] * 12, 49))
    monkeypatch.setattr(summary.subprocess, "run", _git_fixture_run)
    evidence = summary.validate_report(_raw_report(checkpoint), validation.load_preregistration())
    assert evidence["training_safety"]["hard_joint_limit"]["maximum"] == 0.0
    assert evidence["runtime_env_config"]["action_scale"] == 0.65
    assert evidence["exploration_noise_monitor"]["acceptance_gate"].startswith("finite_only")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["tensorboard"]["series_summary"]["Episode_Termination/hard_joint_limit"].update(maximum=0.01, nonzero_sample_count=1), "hard_joint_limit maximum"),
        (lambda report: report["runtime_env_config"]["readback"].update(action_scale=0.7), "runtime env readback mismatch"),
        (lambda report: report["gpu"]["protected_run_safety"].update(mode="entropy_smoke"), "protected GPU mode mismatch"),
        (lambda report: report["source_bundle"]["postrun"].update(stable=False), "source changed during training"),
        (lambda report: report["action_scale_smoke_mode"].update(held_out_evaluation_status="not_run"), "held-out gate opened"),
    ],
)
def test_summary_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    checkpoint = tmp_path / "run" / "model_49.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(summary, "checkpoint_std_vector", lambda _path: ([0.5] * 12, 49))
    monkeypatch.setattr(summary.subprocess, "run", _git_fixture_run)
    report = _raw_report(checkpoint)
    mutation(report)
    with pytest.raises(ValueError, match=message):
        summary.validate_report(report, validation.load_preregistration())


def test_summary_rejects_mutated_preregistration(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "model_49.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture")
    mutated = copy.deepcopy(validation.load_preregistration())
    mutated["acceptance_gate"]["tensorboard_exact_sample_count"] = 1
    with pytest.raises(ValueError, match="not canonical"):
        summary.validate_report(_raw_report(checkpoint), mutated)


def test_summary_output_is_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    summary.write_json_no_overwrite(output, {"status": "pass"})
    with pytest.raises(ValueError, match="overwrite"):
        summary.write_json_no_overwrite(output, {"status": "replacement"})


def test_harness_has_separate_fail_closed_action_scale_mode() -> None:
    source = HARNESS.read_text(encoding="utf-8-sig")
    assert "[switch]$ActionScaleSmoke" in source
    assert "$protectedGpuRun = [bool]($Qualification -or $EntropySmoke -or $ActionScaleSmoke)" in source
    assert "action_scale_smoke_training_safety_zero" in source
    assert "action_scale_smoke_env_yaml_readback" in source
    assert "params\\env.yaml" in source
    assert "model_49.pt" in source


def test_harness_rejects_noncanonical_action_scale_budget_before_launch() -> None:
    completed = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(HARNESS),
            "-Task",
            "Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0",
            "-NumEnvs",
            "512",
            "-MaxIterations",
            "50",
            "-Seed",
            "42",
            "-RunName",
            "rev29_guard_test",
            "-ActionScaleSmoke",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode != 0
    assert "num_envs=1024" in completed.stdout + completed.stderr
