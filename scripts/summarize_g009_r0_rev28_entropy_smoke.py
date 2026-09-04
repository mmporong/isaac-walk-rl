#!/usr/bin/env python3
"""Validate and summarize one completed G009 R0 rev28 entropy smoke report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = REPO_ROOT / "configs" / "g009_r0_rev28_entropy_smoke.json"
SCHEMA_VERSION = "g009.r0.rev28.entropy_smoke_summary.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_mapping(value: Any, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(message)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_portable_path(value: str) -> Path:
    prefix = "%USERPROFILE%"
    if value.upper().startswith(prefix):
        value = str(Path(os.environ.get("USERPROFILE", Path.home()))) + value[len(prefix) :]
    return Path(value).resolve()


def _series(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    tensorboard = require_mapping(report.get("tensorboard"), "TensorBoard evidence missing")
    summaries = require_mapping(
        tensorboard.get("series_summary"), "TensorBoard series summaries missing"
    )
    return require_mapping(
        summaries.get(name), f"required TensorBoard series missing: {name}"
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_zero_series(series: Mapping[str, Any], *, expected_count: int, label: str) -> None:
    require(series.get("sample_count") == expected_count, f"{label} sample_count mismatch")
    require(_finite_number(series.get("maximum")) and float(series["maximum"]) == 0.0, f"{label} maximum must be zero")
    require(series.get("nonzero_sample_count") == 0, f"{label} nonzero sample count must be zero")


def checkpoint_std_vector(path: Path) -> tuple[list[float], int]:
    import torch

    checkpoint = require_mapping(
        torch.load(path, map_location="cpu", weights_only=True),
        "checkpoint root must be a mapping",
    )
    state = require_mapping(checkpoint.get("model_state_dict"), "checkpoint model_state_dict missing")
    std = state.get("std")
    if std is None or not hasattr(std, "detach"):
        raise ValueError("checkpoint std tensor missing")
    values = [float(value) for value in std.detach().cpu().reshape(-1).tolist()]
    require(len(values) == 12, "checkpoint std vector must contain 12 values")
    require(all(math.isfinite(value) for value in values), "checkpoint std vector must be finite")
    iteration = checkpoint.get("iter")
    if not isinstance(iteration, int) or isinstance(iteration, bool):
        raise ValueError("checkpoint iter missing")
    return values, int(iteration)


def validate_report(
    report: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    *,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    canonical_preregistration = json.loads(
        DEFAULT_PREREGISTRATION.read_text(encoding="utf-8")
    )
    require(dict(preregistration) == canonical_preregistration, "supplied preregistration is not canonical")
    training = require_mapping(preregistration.get("training"), "training contract missing")
    gate = require_mapping(preregistration.get("acceptance_gate"), "acceptance gate missing")
    require(report.get("task") == training["task"], "task mismatch")
    require(report.get("num_envs") == 1024, "num_envs mismatch")
    require(report.get("max_iterations") == 50, "iteration budget mismatch")
    require(report.get("seed") == 42 and report.get("headless") is True, "seed/headless mismatch")
    require(report.get("resume") == {"enabled": False, "load_run": None, "checkpoint": None}, "smoke must be scratch")
    require(report.get("effective_hydra_overrides") == [], "Hydra overrides are forbidden")
    qualification = require_mapping(report.get("qualification_mode"), "qualification mode missing")
    require(qualification.get("enabled") is False, "smoke cannot enable qualification")
    require(qualification.get("policy_qualification_status") == "not_run", "qualification status changed")
    smoke = require_mapping(report.get("entropy_smoke_mode"), "entropy smoke mode missing")
    require(smoke.get("enabled") is True, "entropy smoke mode missing")
    require(smoke.get("preflight_passed") is True, "entropy smoke preflight did not pass")
    require(smoke.get("held_out_evaluation_status") == "forbidden_until_full_300_training_safety_zero", "held-out gate opened")
    require(smoke.get("full_300_iteration_training_status") == "forbidden_until_smoke_accepted", "full-run gate opened")
    require(smoke.get("runtime_algorithm_entropy_coef") == 0.0, "runtime entropy readback mismatch")

    source = require_mapping(report.get("source_bundle"), "source bundle missing")
    require(source.get("matches_repository_commit") is True, "source bundle is not bound to HEAD")
    source_files = require_mapping(source.get("files"), "source binding files missing")
    require(list(source_files.keys()) == preregistration["source_binding_paths"], "source binding exact set mismatch")
    contract = require_mapping(
        report.get("entropy_smoke_contract"), "entropy smoke contract binding missing"
    )
    preregistration_path = DEFAULT_PREREGISTRATION
    require(contract.get("path") == "configs/g009_r0_rev28_entropy_smoke.json", "preregistration path mismatch")
    require(contract.get("sha256") == file_sha256(preregistration_path), "preregistration SHA-256 mismatch")
    require(contract.get("source_binding_path_manifest_sha256") == preregistration["source_binding_path_manifest_sha256"], "source manifest mismatch")
    repository = require_mapping(report.get("repository"), "repository binding missing")
    commit_value = repository.get("commit")
    if not isinstance(commit_value, str) or len(commit_value) != 40:
        raise ValueError("repository commit mismatch")
    commit = commit_value
    require(repository.get("dirty") is False, "training repository was dirty")
    preflight = require_mapping(contract.get("prelaunch_validation"), "preflight snapshot missing")
    require(preflight.get("schema_version") == "g009.r0.rev28.entropy_smoke_prelaunch_validation.v1", "preflight schema mismatch")
    require(preflight.get("status") == "pass" and preflight.get("evidence_id") == "G009-5-E021", "preflight identity mismatch")
    preflight_prereg = require_mapping(preflight.get("preregistration"), "preflight preregistration binding missing")
    require(preflight_prereg.get("path") == "configs/g009_r0_rev28_entropy_smoke.json", "preflight preregistration path mismatch")
    require(preflight_prereg.get("sha256") == contract.get("sha256"), "preflight preregistration SHA mismatch")
    require(dict(require_mapping(preflight.get("canonical_static_readback"), "preflight static readback missing")) == {
        "entropy_coef": 0.0,
        "init_noise_std": 0.5,
        "num_steps_per_env": 24,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
    }, "preflight static readback mismatch")
    preflight_source = require_mapping(preflight.get("source_state"), "preflight source snapshot missing")
    require(preflight_source.get("repository_commit") == commit, "preflight commit mismatch")
    require(preflight_source.get("repository_clean") is True, "preflight repository cleanliness mismatch")
    require(preflight_source.get("source_paths_logically_equal_to_head") is True, "preflight logical source binding mismatch")

    paths = preregistration["source_binding_paths"]
    actual_blob_files: dict[str, str] = {}
    for relative in preregistration["source_binding_paths"]:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{relative}"],
            check=False,
            capture_output=True,
        )
        require(completed.returncode == 0, f"cannot read bound git blob: {relative}")
        actual_blob_files[relative] = hashlib.sha256(completed.stdout).hexdigest()
    blob_payload = "\n".join(f"{path}:{actual_blob_files[path]}" for path in paths)
    actual_blob_bundle = hashlib.sha256(blob_payload.encode("utf-8")).hexdigest()
    report_blobs = require_mapping(source.get("commit_blob_sha256"), "commit blob domain missing")
    require(dict(require_mapping(report_blobs.get("files"), "commit blob files missing")) == actual_blob_files, "commit blob file hashes mismatch")
    require(report_blobs.get("bundle") == actual_blob_bundle, "commit blob bundle mismatch")
    preflight_blobs = require_mapping(preflight_source.get("commit_blob_sha256"), "preflight commit blob domain missing")
    require(dict(preflight_blobs) == dict(report_blobs), "preflight commit blob mismatch")

    actual_worktree_files = {path: file_sha256(REPO_ROOT / path) for path in paths}
    worktree_payload = "\n".join(f"{path}:{actual_worktree_files[path]}" for path in paths)
    actual_worktree_bundle = hashlib.sha256(worktree_payload.encode("utf-8")).hexdigest()
    require(source.get("hash_domain") == "executed_worktree_bytes", "worktree hash domain mismatch")
    require(dict(source_files) == actual_worktree_files, "executed worktree file hashes mismatch")
    require(source.get("sha256") == actual_worktree_bundle, "executed worktree bundle mismatch")
    preflight_worktree = require_mapping(preflight_source.get("executed_worktree_sha256"), "preflight worktree domain missing")
    require(dict(require_mapping(preflight_worktree.get("files"), "preflight worktree files missing")) == actual_worktree_files, "preflight worktree files mismatch")
    require(preflight_worktree.get("bundle") == actual_worktree_bundle, "preflight worktree bundle mismatch")
    logical_diff = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--quiet", commit, "--", *paths],
        check=False,
    )
    require(logical_diff.returncode == 0, "worktree sources differ logically from bound commit")
    for phase in ("prelaunch", "postrun"):
        snapshot = require_mapping(source.get(phase), f"{phase} source snapshot missing")
        require(snapshot.get("repository_commit") == commit, f"{phase} commit mismatch")
        require(snapshot.get("hash_domain") == "executed_worktree_bytes", f"{phase} hash domain mismatch")
        require(snapshot.get("sha256") == actual_worktree_bundle, f"{phase} bundle mismatch")
        require(dict(require_mapping(snapshot.get("files"), f"{phase} files missing")) == actual_worktree_files, f"{phase} files mismatch")
    require(require_mapping(source.get("prelaunch"), "prelaunch missing").get("matches_validated_snapshot") is True, "prelaunch snapshot was not validated")
    require(require_mapping(source.get("postrun"), "postrun missing").get("stable") is True, "source changed during training")

    hard = _series(report, "Episode_Termination/hard_joint_limit")
    numeric = _series(report, "Episode_Termination/numeric_invalid")
    noise = _series(report, "Policy/mean_noise_std")
    expected_count = gate["tensorboard_exact_sample_count"]
    _validate_zero_series(hard, expected_count=expected_count, label="hard_joint_limit")
    _validate_zero_series(numeric, expected_count=expected_count, label="numeric_invalid")
    require(noise.get("sample_count") == expected_count, "mean_noise_std sample_count mismatch")
    for field in ("latest", "minimum", "maximum", "mean"):
        require(_finite_number(noise.get(field)), f"mean_noise_std {field} must be finite")
    require(float(noise["latest"]) <= 0.5513023734, "mean_noise_std worsened versus rev26 step49")

    safety = require_mapping(report.get("training_safety_gate"), "training safety gate missing")
    require(safety.get("required") is True, "training safety gate not required")
    require(safety.get("passed") is True, "training safety gate failed")
    gpu = require_mapping(report.get("gpu"), "GPU evidence missing")
    gpu_safety = require_mapping(gpu.get("protected_run_safety"), "protected GPU safety missing")
    require(gpu_safety.get("required") is True, "protected GPU safety missing")
    require(gpu_safety.get("passed") is True, "protected GPU safety failed")
    require(gpu_safety.get("temperature_threshold_c") == 90.0, "GPU temperature threshold mismatch")
    require(gpu_safety.get("sustained_sample_count") == 3, "GPU sustained sample gate mismatch")
    require(gpu_safety.get("fatal_matches") == [], "GPU fatal event found")
    require(gpu_safety.get("descendants_exited") is True, "GPU descendants remain")

    artifacts = require_mapping(report.get("artifacts"), "artifacts missing")
    checkpoint_value = artifacts.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise ValueError("checkpoint path missing")
    reported_checkpoint = resolve_portable_path(checkpoint_value)
    tensorboard_value = artifacts.get("tensorboard_directory")
    if not isinstance(tensorboard_value, str):
        raise ValueError("TensorBoard directory missing")
    run_directory = resolve_portable_path(tensorboard_value)
    require(reported_checkpoint.parent == run_directory, "checkpoint is outside reported run directory")
    if checkpoint_path is not None:
        require(reported_checkpoint == checkpoint_path.resolve(), "checkpoint path mismatch")
    require(reported_checkpoint.name == "model_49.pt" and reported_checkpoint.is_file(), "model_49.pt missing")
    checkpoint_hash = file_sha256(reported_checkpoint)
    require(checkpoint_hash == artifacts.get("checkpoint_sha256"), "checkpoint SHA-256 mismatch")

    agent_yaml_value = artifacts.get("agent_yaml")
    if not isinstance(agent_yaml_value, str):
        raise ValueError("agent.yaml path missing")
    agent_yaml = resolve_portable_path(agent_yaml_value)
    require(agent_yaml == run_directory / "params" / "agent.yaml", "agent.yaml provenance mismatch")
    require(agent_yaml.is_file(), "agent.yaml missing")
    require(file_sha256(agent_yaml) == artifacts.get("agent_yaml_sha256"), "agent.yaml SHA-256 mismatch")
    runtime_config = require_mapping(report.get("runtime_agent_config"), "runtime agent config missing")
    require(runtime_config.get("source") == "official train.py params/agent.yaml", "runtime config source mismatch")
    readback = require_mapping(runtime_config.get("readback"), "runtime agent readback missing")
    expected_readback = {
        "entropy_coef": 0.0,
        "init_noise_std": 0.5,
        "num_steps_per_env": 24,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "max_iterations": 50,
        "device": "cuda:0",
    }
    import yaml

    agent_payload = require_mapping(
        yaml.safe_load(agent_yaml.read_text(encoding="utf-8")), "agent.yaml root mismatch"
    )
    algorithm = require_mapping(agent_payload.get("algorithm"), "agent.yaml algorithm missing")
    policy = require_mapping(agent_payload.get("policy"), "agent.yaml policy missing")
    actual_yaml_readback = {
        "entropy_coef": algorithm.get("entropy_coef"),
        "init_noise_std": policy.get("init_noise_std"),
        "num_steps_per_env": agent_payload.get("num_steps_per_env"),
        "num_learning_epochs": algorithm.get("num_learning_epochs"),
        "num_mini_batches": algorithm.get("num_mini_batches"),
        "max_iterations": agent_payload.get("max_iterations"),
        "device": agent_payload.get("device"),
    }
    require(actual_yaml_readback == expected_readback, "agent.yaml runtime values mismatch")
    require(dict(readback) == expected_readback, "runtime agent readback mismatch")
    require(runtime_config.get("passed") is True, "runtime agent config gate failed")
    std_vector, checkpoint_iteration = checkpoint_std_vector(reported_checkpoint)
    require(checkpoint_iteration == 49, "checkpoint iteration mismatch")

    checks = require_mapping(report.get("success_checks"), "harness success checks missing")
    expected_checks = {
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
        "entropy_smoke_training_safety_zero": True,
        "entropy_smoke_gpu_safety": True,
        "entropy_smoke_source_snapshot_stable": True,
        "entropy_smoke_agent_yaml_readback": True,
        "requested_training_safety_gate_zero": None,
    }
    require(dict(checks) == expected_checks, "harness success checks mismatch")
    require(report.get("run_health_passed") is True and report.get("passed") is True, "raw smoke report failed")
    require(report.get("qualification_passed") is None, "smoke cannot claim qualification")

    historical = require_mapping(
        preregistration.get("historical_evidence"), "historical evidence missing"
    )
    baseline_noise = float(historical["rev26_step49_mean_noise_std"])
    latest_noise = float(noise["latest"])
    return {
        "training_safety": {
            "hard_joint_limit": dict(hard),
            "numeric_invalid": dict(numeric),
        },
        "exploration_noise": {
            "series": dict(noise),
            "checkpoint_std_vector": std_vector,
            "checkpoint_std_vector_mean": sum(std_vector) / len(std_vector),
            "rev26_step49_mean_noise_std": baseline_noise,
            "latest_directional_delta_from_rev26_step49": latest_noise - baseline_noise,
            "comparison_semantics": (
                "non-worsening smoke acceptance gate only; equality is allowed and neither "
                "strict improvement nor causality is claimed"
            ),
        },
        "checkpoint": {
            "path": artifacts["checkpoint"],
            "sha256": checkpoint_hash,
            "iteration": checkpoint_iteration,
        },
        "gpu_safety": dict(gpu_safety),
    }


def write_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    require(not path.exists(), f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    try:
        preregistration_path = args.preregistration.resolve()
        require(
            preregistration_path == DEFAULT_PREREGISTRATION.resolve(),
            "only the canonical rev28 preregistration path is accepted",
        )
        report_path = args.report.resolve()
        preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        evidence = validate_report(
            report,
            preregistration,
            checkpoint_path=args.checkpoint.resolve() if args.checkpoint else None,
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "passed": True,
            "evidence_id": preregistration["evidence_id"],
            "revision": "rev28",
            "decision": "entropy_smoke_accepted",
            "raw_report": {"path": str(report_path), "sha256": file_sha256(report_path)},
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": file_sha256(preregistration_path),
            },
            "evidence": evidence,
            "claim_limits": {
                "policy_qualification": False,
                "recovery_success_measured": False,
                "held_out_evaluation_status": "forbidden_until_full_300_training_safety_zero",
                "full_300_iteration_training": "allowed_next_not_run",
            },
        }
        write_json_no_overwrite(args.output, summary)
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    except Exception as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
