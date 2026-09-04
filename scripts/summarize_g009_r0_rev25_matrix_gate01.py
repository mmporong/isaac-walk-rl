#!/usr/bin/env python3
"""Fail-closed synthesis for the G009 rev25 matrix policy Gate01."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = REPO_ROOT / "configs/g009_r0_rev25_matrix_gate01.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports/runs/g009_r0_rev25_matrix_gate01_s42.json"
TASK = "Isaac-G009-Recover-Flat-Go2-R0-MatrixGate01-v0"
ENTRYPOINT = "scripts/bootstrap_matrix_gate01_g009.py"
SOURCE_PATHS = (
    "configs/g009_r0.json",
    "configs/g009_r0_rev25_matrix_gate01.json",
    "scripts/bootstrap_benchmark_g009.py",
    ENTRYPOINT,
    "scripts/run_training.ps1",
    "scripts/summarize_g009_r0_rev25_matrix_gate01.py",
    "src/isaac_walk_g009/agent_cfg.py",
    "src/isaac_walk_g009/matrix_gate01.py",
    "src/isaac_walk_g009/matrix_observation_adapter.py",
    "src/isaac_walk_g009/mdp/__init__.py",
    "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/mdp/recover.py",
    "src/isaac_walk_g009/recover_contracts.py",
    "src/isaac_walk_g009/recover_env_cfg.py",
    "src/isaac_walk_g009/registry.py",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def portable_path(value: str) -> Path:
    return Path(value.replace("%USERPROFILE%", str(Path.home()))).resolve()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def repository_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_output(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
    ).stdout


def matrix_telemetry_path(run_name: str) -> Path:
    return Path.home() / "IsaacLab/logs/harness" / f"{run_name}.matrix_gate01.json"


def validate_source(report: Mapping[str, Any]) -> bool:
    repository = report.get("repository", {})
    bundle = report.get("source_bundle", {})
    files = bundle.get("files", {}) if isinstance(bundle, Mapping) else {}
    try:
        head = repository_head()
        prereg = load_json(PREREG_PATH)
        manifest_paths = prereg.get("source_binding_paths")
        manifest_sha256 = prereg.get("source_binding_path_manifest_sha256")
        canonical_paths = sorted(SOURCE_PATHS)
        recomputed_manifest = sha256(
            json.dumps(canonical_paths, separators=(",", ":")).encode("utf-8")
        )
        if not (
            repository.get("commit") == head
            and isinstance(manifest_paths, list)
            and manifest_paths == canonical_paths
            and manifest_sha256 == recomputed_manifest
            and set(files) == set(SOURCE_PATHS)
        ):
            return False
        dirty = git_output("status", "--porcelain=v1", "--", *SOURCE_PATHS).strip()
        if dirty:
            return False
        for relative in SOURCE_PATHS:
            digest = files.get(relative)
            path = REPO_ROOT / relative
            if not (isinstance(digest, str) and SHA256.fullmatch(digest) and path.is_file()):
                return False
            tracked = git_output("ls-files", "--error-unmatch", "--", relative).decode().strip()
            if tracked != relative:
                return False
            committed_blob_oid = git_output("rev-parse", f"{head}:{relative}").decode().strip()
            worktree_blob_oid = git_output("hash-object", "--", relative).decode().strip()
            if committed_blob_oid != worktree_blob_oid or sha256(path.read_bytes()) != digest:
                return False
        payload = "\n".join(f"{path}:{files[path]}" for path in canonical_paths).encode()
        return bundle.get("sha256") == sha256(payload)
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError, ValueError):
        return False


def finite_zero(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value == 0


def actor_matrix_optimizer_evidence(optimizer: object) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if not isinstance(optimizer, Mapping):
        return evidence
    for value in optimizer.get("state", {}).values():
        moment = value.get("exp_avg") if isinstance(value, Mapping) else None
        if isinstance(moment, torch.Tensor) and tuple(moment.shape) == (512, 140):
            step = value.get("step")
            if not isinstance(step, (int, float, torch.Tensor)):
                continue
            evidence.append(
                {
                    "step": int(step.item()) if isinstance(step, torch.Tensor) else int(step),
                    "matrix_columns_nonzero": bool(moment[:, 83:140].count_nonzero().item()),
                    "matrix_columns_l2": float(torch.linalg.vector_norm(moment[:, 83:140]).item()),
                }
            )
    return evidence


def load_checkpoint_if_source_valid(path: Path, source_valid: bool) -> Mapping[str, Any]:
    if not source_valid or not path.is_file():
        return {}
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return {}
    return value if isinstance(value, Mapping) else {}


def checkpoint_artifact_binding(
    report: Mapping[str, Any],
) -> tuple[Path, Path, bool]:
    artifacts = report.get("artifacts", {})
    checkpoint = portable_path(artifacts.get("checkpoint", ""))
    tensorboard = portable_path(artifacts.get("tensorboard_directory", ""))
    valid = (
        checkpoint.is_file()
        and tensorboard.is_dir()
        and checkpoint.parent == tensorboard
        and checkpoint.name == "model_0.pt"
        and artifacts.get("checkpoint_sha256") == sha256(checkpoint.read_bytes())
        and any(tensorboard.glob("events.out.tfevents.*"))
    )
    return checkpoint, tensorboard, valid


def validate_telemetry_identity(
    telemetry: Mapping[str, Any],
    prereg: Mapping[str, Any],
    report: Mapping[str, Any],
    telemetry_path: Path,
    head: str,
) -> bool:
    policy = prereg.get("policy_observation", {})
    run_name = report.get("run_name")
    return (
        telemetry.get("schema_version") == "g009.r0.rev25.matrix_gate01_runtime.v1"
        and telemetry.get("evidence_id") == prereg.get("evidence_id") == "G009-5-E018"
        and telemetry.get("run_name") == run_name
        and isinstance(run_name, str)
        and telemetry_path.name == f"{run_name}.matrix_gate01.json"
        and telemetry.get("repository_commit") == head
        and telemetry.get("benchmark_completed") is True
        and telemetry.get("terrain_filter_paths") == prereg.get("terrain_filter_paths")
        and telemetry.get("matrix_observation_dimension") == policy.get("matrix_dimension") == 57
        and telemetry.get("policy_observation_dimension") == policy.get("total_dimension") == 140
        and telemetry.get("critic_observation_dimension") == prereg.get("critic_observation_dimension") == 164
        and telemetry.get("expected_policy_matrix_slice_from_term_order") == [83, 140]
        and telemetry.get("raw_authority_frame") == policy.get("raw_authority_frame") == "world"
        and telemetry.get("policy_projection_frame") == policy.get("policy_projection_frame") == "base"
        and math.isclose(
            telemetry.get("nominal_body_weight_n", float("nan")),
            policy.get("nominal_body_weight_n", float("nan")),
            abs_tol=1.0e-12,
        )
        and telemetry.get("bounding") == policy.get("bounding_id") == "elementwise_tanh"
    )


def synthesize(training_report_path: Path) -> dict[str, Any]:
    prereg = load_json(PREREG_PATH)
    report = load_json(training_report_path)
    require(prereg.get("task") == TASK and prereg.get("evidence_id") == "G009-5-E018", "preregistration identity mismatch")
    require(report.get("task") == TASK, "training task mismatch")
    require(report.get("num_envs") == 1024 and report.get("max_iterations") == 1 and report.get("seed") == 42, "training budget mismatch")
    require(report.get("headless") is True and report.get("resume", {}).get("enabled") is False, "fresh headless contract mismatch")
    require(report.get("effective_hydra_overrides") == [], "Hydra override forbidden")

    telemetry_path = matrix_telemetry_path(report["run_name"])
    telemetry = load_json(telemetry_path)
    runtime = telemetry.get("runtime", {})
    live_contract = runtime.get("live_contract", {})
    solver_rows = live_contract.get("solver_position_velocity", []) if isinstance(live_contract, Mapping) else []
    depenetration = live_contract.get("max_depenetration_velocity_m_s", []) if isinstance(live_contract, Mapping) else []
    ordered_body_names = prereg.get("ordered_body_names")
    ordered_body_names_sha256 = prereg.get("ordered_body_names_sha256")
    recomputed_body_order_sha256 = sha256(
        json.dumps(ordered_body_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ) if isinstance(ordered_body_names, list) else None
    source_valid = validate_source(report)
    checkpoint_path, _tensorboard_path, checkpoint_binding_valid = checkpoint_artifact_binding(report)
    checkpoint = load_checkpoint_if_source_valid(
        checkpoint_path, source_valid and checkpoint_binding_valid
    )
    state = checkpoint.get("model_state_dict", {}) if isinstance(checkpoint, Mapping) else {}
    actor_shape = tuple(state.get("actor.0.weight", torch.empty(0)).shape)
    critic_shape = tuple(state.get("critic.0.weight", torch.empty(0)).shape)
    optimizer = checkpoint.get("optimizer_state_dict", {}) if isinstance(checkpoint, Mapping) else {}
    actor_matrix_optimizer_state = actor_matrix_optimizer_evidence(optimizer)
    safety = report.get("training_safety_aggregate", {})
    numeric = safety.get("numeric_invalid", {})
    hard = safety.get("hard_joint_limit", {})
    success = report.get("success_checks", {})
    entrypoint = report.get("training_entrypoint", {})
    entrypoint_path = str(entrypoint.get("path", "")).replace("\\", "/").lower()
    telemetry_scope = prereg.get("telemetry_scope", {})

    gates = {
        "process_exit_zero": report.get("exit_code") == 0,
        "wrapper_run_health": report.get("run_health_passed") is True and report.get("passed") is True,
        "requested_iteration_reached": success.get("requested_iteration_reached") is True,
        "checkpoint_and_tensorboard": checkpoint_binding_valid and success.get("checkpoint_exists") is True and success.get("tensorboard_exists") is True,
        "source_bundle_matches_commit": source_valid,
        "entrypoint_bound": entrypoint.get("repository_internal") is True and entrypoint_path.endswith("/" + ENTRYPOINT.lower()),
        "matrix_telemetry_identity": validate_telemetry_identity(telemetry, prereg, report, telemetry_path, repository_head()) and telemetry_scope.get("enabled_only_for_this_one_iteration_gate") is True and telemetry_scope.get("production_training_reuse_forbidden") is True,
        "matrix_runtime_calls_positive": isinstance(runtime.get("call_count"), int) and runtime["call_count"] > 0,
        "matrix_source_and_output_finite": runtime.get("all_source_finite") is True and runtime.get("all_output_finite") is True,
        "matrix_source_unchanged": runtime.get("source_unchanged") is True,
        "matrix_positive_magnitude": isinstance(runtime.get("positive_magnitude_count"), int) and runtime["positive_magnitude_count"] > 0,
        "matrix_projection_nonzero_and_variable": runtime.get("nonzero_output_count", 0) > 0 and runtime.get("output_variance_maximum", 0.0) > 0.0,
        "matrix_projection_bounded": runtime.get("output_minimum", -2.0) >= -1.0 and runtime.get("output_maximum", 2.0) <= 1.0,
        "ordered_body_names_bound": isinstance(ordered_body_names, list) and len(ordered_body_names) == 19 and ordered_body_names_sha256 == recomputed_body_order_sha256 and telemetry.get("ordered_body_names") == ordered_body_names and telemetry.get("ordered_body_names_sha256") == ordered_body_names_sha256 and runtime.get("ordered_body_names") == ordered_body_names and runtime.get("ordered_body_names_sha256") == ordered_body_names_sha256 and runtime.get("body_order_consistent") is True,
        "matrix_shapes": runtime.get("source_shapes") == ["1024x19x1x3"] and runtime.get("output_shapes") == ["1024x57"],
        "matrix_dtype_device": runtime.get("source_dtypes") == ["torch.float32"] and runtime.get("source_devices") == ["cuda:0"],
        "live_solver_8_0": len(solver_rows) == 1024 and all(row == [8, 0] for row in solver_rows),
        "live_max_depenetration_1_0": len(depenetration) == 1024 * 19 and all(isinstance(value, (int, float)) and math.isclose(value, 1.0, abs_tol=1.0e-12) for value in depenetration),
        "live_action_scale_ema": isinstance(live_contract, Mapping) and math.isclose(live_contract.get("action_scale", float("nan")), 0.70, abs_tol=1.0e-12) and math.isclose(live_contract.get("action_ema_alpha", float("nan")), 0.2, abs_tol=1.0e-12),
        "actor_checkpoint_input_dimension": actor_shape == (512, 140),
        "critic_checkpoint_input_dimension": critic_shape == (512, 164),
        "optimizer_state_present": isinstance(optimizer, Mapping) and bool(optimizer.get("state")),
        "matrix_actor_columns_received_gradient": len(actor_matrix_optimizer_state) == 1 and actor_matrix_optimizer_state[0]["step"] == 20 and actor_matrix_optimizer_state[0]["matrix_columns_nonzero"] is True and actor_matrix_optimizer_state[0]["matrix_columns_l2"] > 0.0,
        "numeric_invalid_zero": finite_zero(numeric.get("maximum")),
        "hard_joint_limit_zero": finite_zero(hard.get("maximum")),
        "training_safety_gate_zero": success.get("requested_training_safety_gate_zero") is True,
    }
    preregistered_gates = prereg.get("pass_gates", {})
    manifest_keys = sorted([*gates, "preregistered_gate_manifest"])
    manifest_sha256 = sha256(
        json.dumps(manifest_keys, separators=(",", ":")).encode("utf-8")
    )
    gates["preregistered_gate_manifest"] = (
        isinstance(preregistered_gates, Mapping)
        and set(preregistered_gates) == set(manifest_keys)
        and all(value is True for value in preregistered_gates.values())
        and prereg.get("pass_gate_key_manifest_sha256") == manifest_sha256
    )
    passed = all(gates.values())
    return {
        "schema_version": "g009.r0.rev25.matrix_gate01_synthesis.v1",
        "evidence_id": "G009-5-E018",
        "status": "complete",
        "training_report": {"path": training_report_path.resolve().relative_to(REPO_ROOT).as_posix(), "sha256": sha256(training_report_path.read_bytes())},
        "matrix_telemetry": {"path": str(telemetry_path), "sha256": sha256(telemetry_path.read_bytes()), "runtime": runtime},
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path.read_bytes()) if checkpoint_path.is_file() else None, "actor_first_layer_shape": list(actor_shape), "critic_first_layer_shape": list(critic_shape), "actor_matrix_optimizer_state": actor_matrix_optimizer_state},
        "protocol": {"num_envs": 1024, "steps_per_env": 24, "iterations": 1, "ppo_epochs": 5, "mini_batches": 4, "optimizer_mini_batch_updates": 20},
        "safety": {"numeric_invalid_maximum": numeric.get("maximum"), "hard_joint_limit_maximum": hard.get("maximum")},
        "gates": gates,
        "decision": {"passed": passed, "outcome": "matrix_gate01_passed" if passed else "matrix_gate01_failed", "policy_qualification": "not_run", "recovery_success": "not_measured"},
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    require(path.resolve() == DEFAULT_OUTPUT.resolve(), "canonical output path required")
    require(not path.exists(), f"canonical output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"canonical output already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), f"canonical output already exists: {args.output}")
    value = synthesize(args.training_report)
    write_json(args.output, value)
    print(json.dumps({"output": str(args.output), "decision": value["decision"]}, ensure_ascii=False))
    return 0 if value["decision"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
