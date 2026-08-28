#!/usr/bin/env python3
"""Strictly synthesize GPU dynamics and CPU separation calibration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_NUM_ENVS = 8
EXPECTED_ROLLOUT_STEPS = 150
EXPECTED_POSE_ORDER = ["prone", "supine", "left_side", "right_side"]
EXPECTED_REPORT_IDENTITY = {
    "goal_id": "g009",
    "stage_id": "R0",
    "probe": "flat_recover_runtime_calibration",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


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


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-report", required=True, type=Path)
    parser.add_argument("--cpu-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    synthesis = synthesize_reports(args.gpu_report, args.cpu_report)
    _write_json_atomic(args.output, synthesis)
    print(json.dumps({"output": str(args.output.resolve()), **synthesis}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
