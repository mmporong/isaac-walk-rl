#!/usr/bin/env python3
"""Strictly synthesize GPU dynamics and CPU separation calibration evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any


EXPECTED_NUM_ENVS = 8
EXPECTED_ROLLOUT_STEPS = 150
EXPECTED_POSE_ORDER = ["prone", "supine", "left_side", "right_side"]
EXPECTED_TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
EXPECTED_SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SINGLE_RUN_CONTRACT_SHA256 = (
    "4e0499699a24a272cccb9687f417d97770fcbc229186e2aedde6914e45beab66"
)
EXPECTED_REPORT_IDENTITY = {
    "goal_id": "g009",
    "stage_id": "R0",
    "probe": "flat_recover_runtime_calibration",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
STRICT_REQUIRED_CHECKS = frozenset(
    {
        "reset_pose_hold_action_diagnostics_finite",
        "reset_pose_hold_actions_unsaturated",
        "reset_pose_hold_reachable_targets_match_reset_positions",
        "nonfoot_peak_force_body_attribution_complete",
        "nonfoot_peak_force_bounded",
    }
)
REV14_CONTRACT_SHA256 = "744c53d3c8d1e608f849af405c7d0fad314b01234fc4cb9a4ab1000c69140506"
REV14_REQUIRED_CHECKS = STRICT_REQUIRED_CHECKS | {
    "rigid_body_max_depenetration_velocity_matches_contract"
}
REV15_CONTRACT_SHA256 = "5f29ba19458404b5009d3734294c57e79294efecc7fe03bf8c71c71656129832"
REV15_REQUIRED_CHECKS = REV14_REQUIRED_CHECKS
STRICT_REQUIRED_CHECKS_BY_CONTRACT: dict[str, frozenset[str]] = {
    "0679a10d025156f53452e04b50c40b530318cf4c5e904cfc34152b9dea700da4": (
        STRICT_REQUIRED_CHECKS
    ),
    "d4b48d2b5fc1ea7684684a6324ba22fbfae767effeae45668c7310df382392e0": (
        STRICT_REQUIRED_CHECKS
    ),
    "ebee855c503c77bce93c0884535d4fdf66ee5a01538fa59eef0e1b7aabba7558": (
        STRICT_REQUIRED_CHECKS
    ),
    REV14_CONTRACT_SHA256: REV14_REQUIRED_CHECKS,
    REV15_CONTRACT_SHA256: REV15_REQUIRED_CHECKS,
}


class SynthesisError(ValueError):
    """Raised when input evidence cannot support the synthesis claim."""


def _read_report(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise SynthesisError(f"report is not UTF-8: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise SynthesisError(f"invalid JSON report {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise SynthesisError(f"report must contain a JSON object: {resolved}")
    return value, {
        "absolute_path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _nested(report: dict[str, Any], *keys: str) -> Any:
    value: Any = report
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise SynthesisError(f"missing required field: {'.'.join(keys)}")
        value = value[key]
    return value


def _require_true(value: Any, label: str) -> None:
    if value is not True:
        raise SynthesisError(f"{label} must be true")


def _require_not_run_qualification(report: dict[str, Any], label: str) -> None:
    status = _nested(report, "qualification", "status")
    passed = _nested(report, "qualification", "passed")
    if status != "not_run" or passed is not None:
        raise SynthesisError(
            f"{label} qualification must be status=not_run and passed=null"
        )


def _validate_source_bundle(report: dict[str, Any], label: str) -> dict[str, Any]:
    bundle = _nested(report, "source_bundle")
    if not isinstance(bundle, dict):
        raise SynthesisError(f"{label} source_bundle must be an object")
    commit = bundle.get("git_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT_PATTERN.fullmatch(commit):
        raise SynthesisError(f"{label} source_bundle git_commit is invalid")
    _require_true(bundle.get("git_commit_valid"), f"{label} source_bundle git_commit_valid")
    _require_true(bundle.get("all_files_present"), f"{label} source_bundle all_files_present")
    _require_true(bundle.get("clean"), f"{label} source_bundle clean")
    if bundle.get("missing_files") != []:
        raise SynthesisError(f"{label} source_bundle missing_files must be empty")
    if bundle.get("dirty_source_paths") != []:
        raise SynthesisError(f"{label} source_bundle dirty_source_paths must be empty")
    paths = bundle.get("source_binding_paths")
    files = bundle.get("source_binding_files")
    if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths):
        raise SynthesisError(f"{label} source_binding_paths must be a non-empty string list")
    if not isinstance(files, dict) or set(files) != set(paths):
        raise SynthesisError(f"{label} source_binding_files must match source_binding_paths")
    if not all(isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) for value in files.values()):
        raise SynthesisError(f"{label} source_binding_files contains an invalid sha256")
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    expected_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if bundle.get("source_bundle_sha256") != expected_sha256:
        raise SynthesisError(f"{label} source_bundle_sha256 mismatch")
    return bundle


def _validate_shared_identity(
    gpu: dict[str, Any], cpu: dict[str, Any]
) -> dict[str, Any]:
    if gpu.get("schema_version") != 3 or cpu.get("schema_version") != 3:
        raise SynthesisError("GPU/CPU probe schema_version must equal 3")
    for field, expected in EXPECTED_REPORT_IDENTITY.items():
        if gpu.get(field) != expected:
            raise SynthesisError(f"GPU {field} must equal {expected}")
        if cpu.get(field) != expected:
            raise SynthesisError(f"CPU {field} must equal {expected}")
    fields = (
        "contract_sha256",
        "task",
        "seed",
        "num_envs",
        "rollout_steps",
        "pose_name_order",
    )
    for field in fields:
        if field not in gpu or field not in cpu:
            raise SynthesisError(f"missing required field: {field}")
        if gpu[field] != cpu[field]:
            raise SynthesisError(f"GPU/CPU {field} mismatch")
    contract_sha256 = gpu["contract_sha256"]
    if not isinstance(contract_sha256, str) or not _SHA256_PATTERN.fullmatch(
        contract_sha256
    ):
        raise SynthesisError("contract_sha256 must be 64 lowercase hexadecimal characters")
    if not isinstance(gpu["task"], str) or not gpu["task"]:
        raise SynthesisError("task must be a non-empty source task ID")
    if gpu["num_envs"] != EXPECTED_NUM_ENVS:
        raise SynthesisError(f"num_envs must equal {EXPECTED_NUM_ENVS}")
    if gpu["rollout_steps"] != EXPECTED_ROLLOUT_STEPS:
        raise SynthesisError(f"rollout_steps must equal {EXPECTED_ROLLOUT_STEPS}")
    if gpu["pose_name_order"] != EXPECTED_POSE_ORDER:
        raise SynthesisError(f"pose_name_order must equal {EXPECTED_POSE_ORDER}")
    gpu_source_bundle = _validate_source_bundle(gpu, "GPU")
    cpu_source_bundle = _validate_source_bundle(cpu, "CPU")
    if gpu_source_bundle != cpu_source_bundle:
        raise SynthesisError("GPU/CPU source_bundle mismatch")
    identity = {field: gpu[field] for field in fields}
    identity["source_bundle"] = gpu_source_bundle
    return identity


def _validate_repeated_shared_identity(
    gpu_reports: list[dict[str, Any]], cpu_reports: list[dict[str, Any]]
) -> dict[str, Any]:
    """Require one immutable execution identity across every repeated run."""

    reference = gpu_reports[0]
    reference_bundle = _validate_source_bundle(reference, "GPU report 1")
    fields = (
        "contract_sha256",
        "task",
        "seed",
        "headless",
        "num_envs",
        "rollout_steps",
        "pose_name_order",
    )
    for device_label, reports in (("GPU", gpu_reports), ("CPU", cpu_reports)):
        for index, report in enumerate(reports, start=1):
            label = f"{device_label} report {index}"
            if report.get("schema_version") != 3:
                raise SynthesisError(f"{label} schema_version must equal 3")
            for field, expected in EXPECTED_REPORT_IDENTITY.items():
                if report.get(field) != expected:
                    raise SynthesisError(f"{label} {field} must equal {expected}")
            for field in fields:
                if field not in report:
                    raise SynthesisError(f"{label} missing required field: {field}")
                if report[field] != reference[field]:
                    raise SynthesisError(f"all reports {field} mismatch")
            bundle = _validate_source_bundle(report, label)
            if bundle != reference_bundle:
                raise SynthesisError("all reports source_bundle mismatch")

    contract_sha256 = reference["contract_sha256"]
    if not isinstance(contract_sha256, str) or not _SHA256_PATTERN.fullmatch(
        contract_sha256
    ):
        raise SynthesisError("contract_sha256 must be 64 lowercase hexadecimal characters")
    if reference["task"] != EXPECTED_TASK:
        raise SynthesisError(f"task must equal {EXPECTED_TASK}")
    if (
        isinstance(reference["seed"], bool)
        or not isinstance(reference["seed"], int)
        or reference["seed"] != EXPECTED_SEED
    ):
        raise SynthesisError(f"seed must be integer {EXPECTED_SEED}")
    if not isinstance(reference["headless"], bool) or reference["headless"] is not True:
        raise SynthesisError("headless must be boolean true")
    if reference["num_envs"] != EXPECTED_NUM_ENVS:
        raise SynthesisError(f"num_envs must equal {EXPECTED_NUM_ENVS}")
    if reference["rollout_steps"] != EXPECTED_ROLLOUT_STEPS:
        raise SynthesisError(f"rollout_steps must equal {EXPECTED_ROLLOUT_STEPS}")
    if reference["pose_name_order"] != EXPECTED_POSE_ORDER:
        raise SynthesisError(f"pose_name_order must equal {EXPECTED_POSE_ORDER}")
    identity = {field: reference[field] for field in fields}
    identity["source_bundle"] = reference_bundle
    return identity


def _validate_strict_device_report(
    report: dict[str, Any], *, label: str, expected_device: str
) -> None:
    device = _nested(report, "device")
    if expected_device == "gpu":
        if not isinstance(device, str) or not device.lower().startswith("cuda"):
            raise SynthesisError(f"{label} device must be CUDA")
    elif not isinstance(device, str) or device.lower() != "cpu":
        raise SynthesisError(f"{label} device must be cpu")
    _require_true(_nested(report, "run_health", "passed"), f"{label} run_health")
    _require_true(
        _nested(report, "runtime_contract", "passed"),
        f"{label} runtime_contract",
    )
    checks = _nested(report, "checks")
    if not isinstance(checks, dict) or not checks:
        raise SynthesisError(f"{label} checks must be a non-empty boolean map")
    if not all(isinstance(key, str) and key for key in checks):
        raise SynthesisError(f"{label} checks keys must be non-empty strings")
    contract_sha256 = report.get("contract_sha256")
    if not isinstance(contract_sha256, str):
        raise SynthesisError(f"{label} contract_sha256 must be a string")
    required_checks = STRICT_REQUIRED_CHECKS_BY_CONTRACT.get(contract_sha256)
    if required_checks is None:
        raise SynthesisError(
            f"{label} contract_sha256 has no registered strict check contract"
        )
    missing_checks = sorted(required_checks - checks.keys())
    if missing_checks:
        raise SynthesisError(
            f"{label} checks missing required checks: {', '.join(missing_checks)}"
        )
    non_boolean_checks = sorted(
        key for key, value in checks.items() if not isinstance(value, bool)
    )
    if non_boolean_checks:
        raise SynthesisError(
            f"{label} checks must contain only booleans: {', '.join(non_boolean_checks)}"
        )
    failed_checks = sorted(key for key, value in checks.items() if value is not True)
    if failed_checks:
        raise SynthesisError(
            f"{label} runtime_contract aggregate disagrees with failed checks: "
            f"{', '.join(failed_checks)}"
        )
    if contract_sha256 == REV15_CONTRACT_SHA256:
        progression = _nested(report, "progression_gate")
        if not isinstance(progression, dict):
            raise SynthesisError(f"{label} rev15 progression_gate must be an object")
        _require_true(progression.get("passed"), f"{label} rev15 progression_gate")
        _require_true(report.get("passed"), f"{label} rev15 top-level passed")
        if report.get("passed_semantics") != "progression_gate_not_policy_qualification":
            raise SynthesisError(f"{label} rev15 passed_semantics mismatch")
        if progression.get("device") != device:
            raise SynthesisError(f"{label} rev15 progression gate device mismatch")
        cpu_required = expected_device == "cpu"
        if progression.get("cpu_contact_separation_required_this_run") is not cpu_required:
            raise SynthesisError(
                f"{label} rev15 progression gate device authority scope mismatch"
            )
        blocking_checks = progression.get("blocking_checks")
        if not isinstance(blocking_checks, dict):
            raise SynthesisError(
                f"{label} rev15 progression gate blocking_checks must be an object"
            )
        _require_true(
            blocking_checks.get("runtime_contract"),
            f"{label} rev15 progression runtime_contract",
        )
        if cpu_required:
            if progression.get("status") != "passed":
                raise SynthesisError(f"{label} rev15 CPU progression status mismatch")
            _require_true(
                blocking_checks.get("cpu_contact_separation"),
                f"{label} rev15 progression cpu_contact_separation",
            )
        else:
            if progression.get("status") != (
                "passed_runtime_contract_cpu_authority_not_evaluated"
            ):
                raise SynthesisError(f"{label} rev15 GPU progression status mismatch")
            if "cpu_contact_separation" in blocking_checks:
                raise SynthesisError(
                    f"{label} rev15 GPU progression gate must not claim CPU authority"
                )
    _require_not_run_qualification(report, label)
    if expected_device != "cpu":
        return
    separation = _nested(report, "required_crosschecks", "cpu_contact_separation")
    if not isinstance(separation, dict):
        raise SynthesisError(f"{label} separation crosscheck must be an object")
    if separation.get("authority_device") != "cpu":
        raise SynthesisError(f"{label} separation authority_device must be cpu")
    _require_true(
        separation.get("this_run_is_authority"),
        f"{label} separation this_run_is_authority",
    )
    _require_true(separation.get("data_available"), f"{label} separation data_available")
    if separation.get("status") != "observed":
        raise SynthesisError(f"{label} separation status must be observed")
    _require_true(separation.get("passed"), f"{label} authoritative separation passed")


def _strict_force_evidence(
    report: dict[str, Any], *, label: str, input_evidence: dict[str, str]
) -> tuple[float, dict[str, Any]]:
    threshold = _nested(report, "calibration_thresholds", "max_nonfoot_force_bodyweights")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or threshold <= 0
    ):
        raise SynthesisError(f"{label} force threshold must be a positive finite number")
    _require_true(
        _nested(report, "checks", "nonfoot_peak_force_bounded"),
        f"{label} nonfoot_peak_force_bounded",
    )
    metrics = _nested(report, "pose_mode_metrics")
    if not isinstance(metrics, list) or len(metrics) != EXPECTED_NUM_ENVS:
        raise SynthesisError(f"{label} pose_mode_metrics must contain 8 entries")
    env_indices: set[int] = set()
    worst: dict[str, Any] | None = None
    for metric in metrics:
        if not isinstance(metric, dict):
            raise SynthesisError(f"{label} pose_mode_metrics entries must be objects")
        env_index = metric.get("env_index")
        force = metric.get("max_nonfoot_force_bodyweights")
        step = metric.get("max_nonfoot_force_physics_step")
        body_index = metric.get("max_nonfoot_force_body_index")
        body_name = metric.get("max_nonfoot_force_body_name")
        if isinstance(env_index, bool) or not isinstance(env_index, int):
            raise SynthesisError(f"{label} force env_index must be an integer")
        env_indices.add(env_index)
        if (
            isinstance(force, bool)
            or not isinstance(force, (int, float))
            or not math.isfinite(float(force))
            or force < 0
        ):
            raise SynthesisError(f"{label} max non-foot force must be finite and nonnegative")
        if metric.get("pose_id") not in EXPECTED_POSE_ORDER:
            raise SynthesisError(f"{label} force pose_id is invalid")
        if not isinstance(metric.get("action_mode"), str) or not metric["action_mode"]:
            raise SynthesisError(f"{label} force action_mode is invalid")
        if force > 0:
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise SynthesisError(f"{label} observed force physics step is invalid")
            if isinstance(body_index, bool) or not isinstance(body_index, int) or body_index < 0:
                raise SynthesisError(f"{label} observed force body index is invalid")
            if not isinstance(body_name, str) or not body_name:
                raise SynthesisError(f"{label} observed force body name is invalid")
        candidate = {
            "bodyweights": float(force),
            "env_index": env_index,
            "pose_id": metric["pose_id"],
            "action_mode": metric["action_mode"],
            "body_index": body_index,
            "body_name": body_name,
            "physics_step": step,
            "input": input_evidence,
        }
        if worst is None or candidate["bodyweights"] > worst["bodyweights"]:
            worst = candidate
    if env_indices != set(range(EXPECTED_NUM_ENVS)):
        raise SynthesisError(f"{label} force env indices must equal 0..7")
    assert worst is not None
    if worst["bodyweights"] > float(threshold):
        raise SynthesisError(f"{label} worst non-foot force exceeds threshold")
    return float(threshold), worst


def _validate_execution_lineage(
    report: dict[str, Any], *, input_path: Path, label: str
) -> dict[str, Any]:
    execution = _nested(report, "execution")
    if not isinstance(execution, dict):
        raise SynthesisError(f"{label} execution must be an object")
    execution_id = execution.get("execution_id")
    if not isinstance(execution_id, str):
        raise SynthesisError(f"{label} execution_id must be UUID4 hex")
    try:
        parsed_uuid = uuid.UUID(hex=execution_id)
    except (ValueError, AttributeError) as exc:
        raise SynthesisError(f"{label} execution_id must be UUID4 hex") from exc
    if parsed_uuid.version != 4 or parsed_uuid.hex != execution_id:
        raise SynthesisError(f"{label} execution_id must be UUID4 hex")

    started_at_utc = execution.get("started_at_utc")
    if not isinstance(started_at_utc, str) or not started_at_utc:
        raise SynthesisError(f"{label} started_at_utc must be a valid UTC timestamp")
    try:
        parsed_time = datetime.fromisoformat(started_at_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SynthesisError(
            f"{label} started_at_utc must be a valid UTC timestamp"
        ) from exc
    if parsed_time.tzinfo is None or parsed_time.utcoffset() != timedelta(0):
        raise SynthesisError(f"{label} started_at_utc must be a valid UTC timestamp")
    if execution.get("no_overwrite") is not True:
        raise SynthesisError(f"{label} execution no_overwrite must be true")

    resolved_input = Path(input_path).resolve(strict=True)
    reports_dir = (REPO_ROOT / "reports" / "runs").resolve()
    if resolved_input.parent != reports_dir or resolved_input.suffix != ".json":
        raise SynthesisError(
            f"{label} input must be canonical reports/runs/<filename>.json"
        )
    canonical_relative = Path("reports", "runs", resolved_input.name).as_posix()
    if execution.get("output_path_repo_relative") != canonical_relative:
        raise SynthesisError(f"{label} execution output path binding mismatch")
    return {
        "execution_id": execution_id,
        "started_at_utc": started_at_utc,
        "no_overwrite": True,
        "output_path_repo_relative": canonical_relative,
    }


def synthesize_reports(gpu_path: Path, cpu_path: Path) -> dict[str, Any]:
    """Validate two immutable evidence files and return a bounded synthesis."""

    gpu_path = Path(gpu_path)
    cpu_path = Path(cpu_path)
    gpu, gpu_evidence = _read_report(gpu_path)
    cpu, cpu_evidence = _read_report(cpu_path)
    identity = _validate_shared_identity(gpu, cpu)

    gpu_device = _nested(gpu, "device")
    cpu_device = _nested(cpu, "device")
    if not isinstance(gpu_device, str) or not gpu_device.lower().startswith("cuda"):
        raise SynthesisError("GPU report device must be CUDA")
    if not isinstance(cpu_device, str) or cpu_device.lower() != "cpu":
        raise SynthesisError("CPU report device must be cpu")
    _require_true(_nested(gpu, "run_health", "passed"), "GPU run_health")
    _require_true(
        _nested(gpu, "runtime_contract", "passed"), "GPU runtime_contract"
    )
    _require_not_run_qualification(gpu, "GPU report")
    _require_not_run_qualification(cpu, "CPU report")

    separation = _nested(
        cpu, "required_crosschecks", "cpu_contact_separation"
    )
    if not isinstance(separation, dict):
        raise SynthesisError("CPU separation crosscheck must be an object")
    if separation.get("authority_device") != "cpu":
        raise SynthesisError("CPU separation authority_device must be cpu")
    _require_true(
        separation.get("this_run_is_authority"),
        "CPU separation this_run_is_authority",
    )
    _require_true(separation.get("data_available"), "CPU separation data_available")
    if separation.get("status") != "observed":
        raise SynthesisError("CPU separation status must be observed")
    _require_true(separation.get("passed"), "CPU authoritative separation passed")

    return {
        "schema_version": 2,
        "goal_id": "g009",
        "stage_id": "R0",
        "synthesis": "gpu_dynamics_cpu_separation_runtime_calibration",
        "verified_identity": identity,
        "inputs": {
            "gpu": gpu_evidence,
            "cpu": cpu_evidence,
        },
        "evidence_verdicts": {
            "gpu_run_health_passed": True,
            "gpu_runtime_contract_passed": True,
            "cpu_authoritative_separation_passed": True,
        },
        "runtime_calibration_passed": True,
        "learned_policy_qualified": False,
        "learned_policy_qualification": {
            "status": "not_run",
            "passed": False,
            "reason": "runtime calibration does not evaluate a learned checkpoint",
        },
    }


def synthesize_repeated_reports(
    gpu_paths: list[Path], cpu_paths: list[Path]
) -> dict[str, Any]:
    """Strictly synthesize three GPU and three CPU rev11 calibration runs."""

    if len(gpu_paths) != 3 or len(cpu_paths) != 3:
        raise SynthesisError("strict repeated synthesis requires exactly 3 GPU and 3 CPU reports")
    resolved_paths = [
        Path(path).resolve(strict=True) for path in [*gpu_paths, *cpu_paths]
    ]
    if len(set(resolved_paths)) != 6:
        raise SynthesisError("strict repeated synthesis requires 6 distinct input paths")

    gpu_loaded = [_read_report(Path(path)) for path in gpu_paths]
    cpu_loaded = [_read_report(Path(path)) for path in cpu_paths]
    gpu_reports = [report for report, _ in gpu_loaded]
    cpu_reports = [report for report, _ in cpu_loaded]
    gpu_evidence = [evidence for _, evidence in gpu_loaded]
    cpu_evidence = [evidence for _, evidence in cpu_loaded]
    identity = _validate_repeated_shared_identity(gpu_reports, cpu_reports)

    execution_lineage: dict[str, list[dict[str, Any]]] = {"gpu": [], "cpu": []}
    execution_ids: list[str] = []
    for device, loaded in (("gpu", gpu_loaded), ("cpu", cpu_loaded)):
        for index, (report, evidence) in enumerate(loaded, start=1):
            lineage = _validate_execution_lineage(
                report,
                input_path=Path(evidence["absolute_path"]),
                label=f"{device.upper()} report {index}",
            )
            execution_ids.append(lineage["execution_id"])
            execution_lineage[device].append({**lineage, "input": evidence})
    if len(set(execution_ids)) != 6:
        raise SynthesisError("strict repeated synthesis requires 6 unique execution_id values")

    thresholds: list[float] = []
    force_candidates: list[dict[str, Any]] = []
    for index, (report, evidence) in enumerate(gpu_loaded, start=1):
        label = f"GPU report {index}"
        _validate_strict_device_report(report, label=label, expected_device="gpu")
        threshold, worst = _strict_force_evidence(
            report, label=label, input_evidence=evidence
        )
        thresholds.append(threshold)
        force_candidates.append({"device": "gpu", "run_index": index, **worst})
    for index, (report, evidence) in enumerate(cpu_loaded, start=1):
        label = f"CPU report {index}"
        _validate_strict_device_report(report, label=label, expected_device="cpu")
        threshold, worst = _strict_force_evidence(
            report, label=label, input_evidence=evidence
        )
        thresholds.append(threshold)
        force_candidates.append({"device": "cpu", "run_index": index, **worst})
    if len(set(thresholds)) != 1:
        raise SynthesisError("all reports force threshold mismatch")
    worst_force = max(force_candidates, key=lambda item: item["bodyweights"])

    return {
        "schema_version": 3,
        "goal_id": "g009",
        "stage_id": "R0",
        "synthesis": "gpu_dynamics_cpu_separation_runtime_calibration",
        "synthesis_mode": "strict_repeated_3x_gpu_3x_cpu",
        "verified_identity": identity,
        "inputs": {
            "gpu": gpu_evidence,
            "cpu": cpu_evidence,
        },
        "execution_lineage": execution_lineage,
        "device_pass_counts": {
            "gpu": {"passed": 3, "required": 3},
            "cpu": {"passed": 3, "required": 3},
        },
        "force_verification": {
            "threshold_bodyweights": thresholds[0],
            "all_runs_bounded": True,
            "worst_case": worst_force,
        },
        "evidence_verdicts": {
            "gpu_run_health_passed_3_of_3": True,
            "gpu_runtime_contract_passed_3_of_3": True,
            "cpu_run_health_passed_3_of_3": True,
            "cpu_runtime_contract_passed_3_of_3": True,
            "cpu_authoritative_separation_passed_3_of_3": True,
            "all_runs_nonfoot_peak_force_bounded": True,
        },
        "runtime_calibration_passed": True,
        "learned_policy_qualified": False,
        "learned_policy_qualification": {
            "status": "not_run",
            "passed": False,
            "reason": "runtime calibration does not evaluate a learned checkpoint",
        },
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing synthesis: {resolved}")
    if temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite existing temporary synthesis: {temporary}"
        )
    created_temporary = False
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            created_temporary = True
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, resolved)
    finally:
        if created_temporary and temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-report", required=True, type=Path, nargs="+")
    parser.add_argument("--cpu-report", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def synthesize_cli_inputs(
    gpu_paths: list[Path], cpu_paths: list[Path]
) -> dict[str, Any]:
    """Dispatch CLI inputs without allowing new contracts to bypass the 3+3 gate."""

    if len(gpu_paths) == 1 and len(cpu_paths) == 1:
        gpu, _ = _read_report(Path(gpu_paths[0]))
        cpu, _ = _read_report(Path(cpu_paths[0]))
        gpu_contract = gpu.get("contract_sha256")
        cpu_contract = cpu.get("contract_sha256")
        if (
            gpu_contract != LEGACY_SINGLE_RUN_CONTRACT_SHA256
            or cpu_contract != LEGACY_SINGLE_RUN_CONTRACT_SHA256
        ):
            raise SynthesisError(
                "CLI 1+1 synthesis is restricted to the exact legacy contract; "
                "newer contracts require strict 3+3 reports"
            )
        return synthesize_reports(gpu_paths[0], cpu_paths[0])
    return synthesize_repeated_reports(gpu_paths, cpu_paths)


def main() -> int:
    args = build_parser().parse_args()
    synthesis = synthesize_cli_inputs(args.gpu_report, args.cpu_report)
    _write_json_atomic(args.output, synthesis)
    print(json.dumps({"output": str(args.output.resolve()), **synthesis}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
