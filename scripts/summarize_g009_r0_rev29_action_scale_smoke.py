#!/usr/bin/env python3
"""Validate and summarize one completed G009 R0 rev29 action-scale smoke report."""

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
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from validate_g009_r0_rev29_action_scale_smoke import (  # noqa: E402
    DEFAULT_PREREGISTRATION,
    EVIDENCE_ID,
    file_sha256,
    require,
)


SCHEMA_VERSION = "g009.r0.rev29.action_scale_smoke_summary.v1"


def require_mapping(value: Any, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(message)
    return value


def require_string(value: Any, message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    return value


def resolve_portable_path(value: str) -> Path:
    prefix = "%USERPROFILE%"
    if value.upper().startswith(prefix):
        value = str(Path(os.environ.get("USERPROFILE", Path.home()))) + value[len(prefix) :]
    return Path(value).resolve()


def _series(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    tensorboard = require_mapping(report.get("tensorboard"), "TensorBoard evidence missing")
    summaries = require_mapping(tensorboard.get("series_summary"), "TensorBoard series summaries missing")
    return require_mapping(summaries.get(name), f"required TensorBoard series missing: {name}")


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


def _expected_agent_readback(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    return dict(require_mapping(require_mapping(preregistration["runtime_readback"], "runtime readback missing")["agent_yaml"], "agent readback contract missing"))


def _expected_env_readback(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    return dict(require_mapping(require_mapping(preregistration["runtime_readback"], "runtime readback missing")["env_yaml"], "env readback contract missing"))


def validate_report(
    report: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    *,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    canonical_preregistration = json.loads(DEFAULT_PREREGISTRATION.read_text(encoding="utf-8"))
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
    entropy_smoke = require_mapping(report.get("entropy_smoke_mode"), "entropy smoke mode missing")
    require(entropy_smoke.get("enabled") is False, "rev28 entropy mode must remain disabled")
    smoke = require_mapping(report.get("action_scale_smoke_mode"), "action-scale smoke mode missing")
    require(smoke.get("enabled") is True, "action-scale smoke mode missing")
    require(smoke.get("preflight_passed") is True, "action-scale smoke preflight did not pass")
    require(smoke.get("runtime_action_scale") == 0.65, "runtime action scale mismatch")
    require(smoke.get("held_out_evaluation_status") == "forbidden_until_full_300_training_safety_zero", "held-out gate opened")
    require(smoke.get("full_300_iteration_training_status") == "forbidden_until_smoke_accepted", "full-run gate opened")

    source = require_mapping(report.get("source_bundle"), "source bundle missing")
    require(source.get("matches_repository_commit") is True, "source bundle is not bound to HEAD")
    source_files = require_mapping(source.get("files"), "source binding files missing")
    paths = preregistration["source_binding_paths"]
    require(list(source_files.keys()) == paths, "source binding exact set mismatch")
    contract = require_mapping(report.get("action_scale_smoke_contract"), "action-scale smoke contract binding missing")
    require(contract.get("path") == "configs/g009_r0_rev29_action_scale_smoke.json", "preregistration path mismatch")
    require(contract.get("sha256") == file_sha256(DEFAULT_PREREGISTRATION), "preregistration SHA-256 mismatch")
    require(contract.get("source_binding_path_manifest_sha256") == preregistration["source_binding_path_manifest_sha256"], "source manifest mismatch")

    repository = require_mapping(report.get("repository"), "repository binding missing")
    commit_value = repository.get("commit")
    if not isinstance(commit_value, str) or len(commit_value) != 40:
        raise ValueError("repository commit mismatch")
    commit = commit_value
    require(repository.get("dirty") is False, "training repository was dirty")
    preflight = require_mapping(contract.get("prelaunch_validation"), "preflight snapshot missing")
    require(preflight.get("schema_version") == "g009.r0.rev29.action_scale_smoke_prelaunch_validation.v1", "preflight schema mismatch")
    require(preflight.get("status") == "pass" and preflight.get("evidence_id") == EVIDENCE_ID, "preflight identity mismatch")
    preflight_prereg = require_mapping(preflight.get("preregistration"), "preflight preregistration binding missing")
    require(preflight_prereg.get("path") == "configs/g009_r0_rev29_action_scale_smoke.json", "preflight preregistration path mismatch")
    require(preflight_prereg.get("sha256") == contract.get("sha256"), "preflight preregistration SHA mismatch")
    expected_static = {
        "action_scale": 0.65,
        "agent_yaml": _expected_agent_readback(preregistration),
        "env_yaml": _expected_env_readback(preregistration),
    }
    require(dict(require_mapping(preflight.get("canonical_static_readback"), "preflight static readback missing")) == expected_static, "preflight static readback mismatch")
    upstream = require_mapping(preflight.get("upstream"), "pinned upstream evidence missing")
    require(upstream.get("isaac_lab_commit") == "90b79bb2d44feb8d833f260f2bf37da3487180ba", "Isaac Lab commit mismatch")
    require(upstream.get("tracked_clean") is True, "Isaac Lab tracked state was dirty")
    require(upstream.get("official_train_sha256") == "8b995f75ac57ce7403973ff1f3f2715fbff9563ef2cdcdc321a7edc5dd15f5df", "official train.py SHA mismatch")

    preflight_source = require_mapping(preflight.get("source_state"), "preflight source snapshot missing")
    require(preflight_source.get("repository_commit") == commit, "preflight commit mismatch")
    require(preflight_source.get("repository_clean") is True, "preflight repository cleanliness mismatch")
    actual_blob_files: dict[str, str] = {}
    for relative in paths:
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
    require(dict(require_mapping(preflight_source.get("commit_blob_sha256"), "preflight commit blob domain missing")) == dict(report_blobs), "preflight commit blob mismatch")

    actual_worktree_files = {path: file_sha256(REPO_ROOT / path) for path in paths}
    worktree_payload = "\n".join(f"{path}:{actual_worktree_files[path]}" for path in paths)
    actual_worktree_bundle = hashlib.sha256(worktree_payload.encode("utf-8")).hexdigest()
    require(source.get("hash_domain") == "executed_worktree_bytes", "worktree hash domain mismatch")
    require(dict(source_files) == actual_worktree_files, "executed worktree file hashes mismatch")
    require(source.get("sha256") == actual_worktree_bundle, "executed worktree bundle mismatch")
    preflight_worktree = require_mapping(preflight_source.get("executed_worktree_sha256"), "preflight worktree domain missing")
    require(dict(require_mapping(preflight_worktree.get("files"), "preflight worktree files missing")) == actual_worktree_files, "preflight worktree files mismatch")
    require(preflight_worktree.get("bundle") == actual_worktree_bundle, "preflight worktree bundle mismatch")
    logical_diff = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--quiet", commit, "--", *paths], check=False)
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

    safety = require_mapping(report.get("training_safety_gate"), "training safety gate missing")
    require(safety.get("required") is True and safety.get("passed") is True, "training safety gate failed")
    gpu = require_mapping(report.get("gpu"), "GPU evidence missing")
    gpu_safety = require_mapping(gpu.get("protected_run_safety"), "protected GPU safety missing")
    require(gpu_safety.get("required") is True and gpu_safety.get("passed") is True, "protected GPU safety failed")
    require(gpu_safety.get("mode") == "action_scale_smoke", "protected GPU mode mismatch")
    require(gpu_safety.get("temperature_threshold_c") == 90.0, "GPU temperature threshold mismatch")
    require(gpu_safety.get("sustained_sample_count") == 3, "GPU sustained sample gate mismatch")
    require(gpu_safety.get("fatal_matches") == [], "GPU fatal event found")
    require(gpu_safety.get("descendants_exited") is True, "GPU descendants remain")

    artifacts = require_mapping(report.get("artifacts"), "artifacts missing")
    run_directory = resolve_portable_path(
        require_string(artifacts.get("tensorboard_directory"), "TensorBoard directory missing")
    )
    checkpoint = resolve_portable_path(
        require_string(artifacts.get("checkpoint"), "checkpoint path missing")
    )
    require(checkpoint.parent == run_directory, "checkpoint is outside reported run directory")
    if checkpoint_path is not None:
        require(checkpoint == checkpoint_path.resolve(), "checkpoint path mismatch")
    require(checkpoint.name == "model_49.pt" and checkpoint.is_file(), "model_49.pt missing")
    checkpoint_hash = file_sha256(checkpoint)
    require(checkpoint_hash == artifacts.get("checkpoint_sha256"), "checkpoint SHA-256 mismatch")

    import yaml

    agent_yaml = resolve_portable_path(
        require_string(artifacts.get("agent_yaml"), "agent.yaml path missing")
    )
    env_yaml = resolve_portable_path(
        require_string(artifacts.get("env_yaml"), "env.yaml path missing")
    )
    require(agent_yaml == run_directory / "params" / "agent.yaml" and agent_yaml.is_file(), "agent.yaml provenance mismatch")
    require(env_yaml == run_directory / "params" / "env.yaml" and env_yaml.is_file(), "env.yaml provenance mismatch")
    require(file_sha256(agent_yaml) == artifacts.get("agent_yaml_sha256"), "agent.yaml SHA-256 mismatch")
    require(file_sha256(env_yaml) == artifacts.get("env_yaml_sha256"), "env.yaml SHA-256 mismatch")
    runtime_agent = require_mapping(report.get("runtime_agent_config"), "runtime agent config missing")
    runtime_env = require_mapping(report.get("runtime_env_config"), "runtime env config missing")
    expected_agent = _expected_agent_readback(preregistration)
    expected_env = _expected_env_readback(preregistration)
    require(runtime_agent.get("source") == "official train.py params/agent.yaml", "runtime agent source mismatch")
    require(dict(require_mapping(runtime_agent.get("readback"), "runtime agent readback missing")) == expected_agent, "runtime agent readback mismatch")
    require(runtime_agent.get("passed") is True, "runtime agent config gate failed")
    require(runtime_env.get("source") == "official train.py params/env.yaml", "runtime env source mismatch")
    require(dict(require_mapping(runtime_env.get("readback"), "runtime env readback missing")) == expected_env, "runtime env readback mismatch")
    require(runtime_env.get("passed") is True, "runtime env config gate failed")

    agent_payload = require_mapping(yaml.safe_load(agent_yaml.read_text(encoding="utf-8")), "agent.yaml root mismatch")
    algorithm = require_mapping(agent_payload.get("algorithm"), "agent.yaml algorithm missing")
    policy = require_mapping(agent_payload.get("policy"), "agent.yaml policy missing")
    actual_agent = {
        "entropy_coef": algorithm.get("entropy_coef"),
        "init_noise_std": policy.get("init_noise_std"),
        "num_steps_per_env": agent_payload.get("num_steps_per_env"),
        "num_learning_epochs": algorithm.get("num_learning_epochs"),
        "num_mini_batches": algorithm.get("num_mini_batches"),
        "max_iterations": agent_payload.get("max_iterations"),
        "device": agent_payload.get("device"),
    }
    require(actual_agent == expected_agent, "agent.yaml runtime values mismatch")
    env_payload = require_mapping(yaml.unsafe_load(env_yaml.read_text(encoding="utf-8")), "env.yaml root mismatch")
    action = require_mapping(require_mapping(env_payload.get("actions"), "env actions missing").get("joint_pos"), "joint_pos action missing")
    robot = require_mapping(require_mapping(env_payload.get("scene"), "env scene missing").get("robot"), "env robot missing")
    articulation = require_mapping(require_mapping(robot.get("spawn"), "robot spawn missing").get("articulation_props"), "articulation props missing")
    rigid = require_mapping(require_mapping(robot.get("spawn"), "robot spawn missing").get("rigid_props"), "rigid props missing")
    actual_env = {
        "action_scale": action.get("scale"),
        "rescale_to_limits": action.get("rescale_to_limits"),
        "action_ema_alpha": action.get("alpha"),
        "asset_soft_joint_limit_factor": robot.get("soft_joint_pos_limit_factor"),
        "articulation_solver_position_iteration_count": articulation.get("solver_position_iteration_count"),
        "articulation_solver_velocity_iteration_count": articulation.get("solver_velocity_iteration_count"),
        "max_depenetration_velocity_m_s": rigid.get("max_depenetration_velocity"),
    }
    require(actual_env == expected_env, "env.yaml runtime values mismatch")
    std_vector, checkpoint_iteration = checkpoint_std_vector(checkpoint)
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
    }
    require(dict(checks) == expected_checks, "harness success checks mismatch")
    require(report.get("run_health_passed") is True and report.get("passed") is True, "raw smoke report failed")
    require(report.get("qualification_passed") is None, "smoke cannot claim qualification")
    baseline_noise = float(preregistration["historical_evidence"]["rev28_step49_mean_noise_std"])
    latest_noise = float(noise["latest"])
    return {
        "training_safety": {"hard_joint_limit": dict(hard), "numeric_invalid": dict(numeric)},
        "exploration_noise_monitor": {
            "series": dict(noise),
            "checkpoint_std_vector": std_vector,
            "checkpoint_std_vector_mean": sum(std_vector) / len(std_vector),
            "rev28_step49_mean_noise_std": baseline_noise,
            "latest_delta_from_rev28_step49": latest_noise - baseline_noise,
            "acceptance_gate": "finite_only; action scale is the sole intervention",
        },
        "runtime_agent_config": expected_agent,
        "runtime_env_config": expected_env,
        "checkpoint": {"path": artifacts["checkpoint"], "sha256": checkpoint_hash, "iteration": checkpoint_iteration},
        "gpu_safety": dict(gpu_safety),
        "upstream": dict(upstream),
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
        require(preregistration_path == DEFAULT_PREREGISTRATION.resolve(), "only the canonical rev29 preregistration path is accepted")
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
            "evidence_id": EVIDENCE_ID,
            "revision": "rev29",
            "decision": "action_scale_smoke_accepted",
            "raw_report": {"path": str(report_path), "sha256": file_sha256(report_path)},
            "preregistration": {"path": str(preregistration_path), "sha256": file_sha256(preregistration_path)},
            "evidence": evidence,
            "claim_limits": {
                "policy_qualification": False,
                "recovery_success_measured": False,
                "action_scale_change_proved_causal": False,
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
