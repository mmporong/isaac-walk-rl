#!/usr/bin/env python3
"""Fail-closed synthesis for the G009-5-E017 rev24 GPU throughput smoke."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "reports/runs"
PREREG_PATH = REPO_ROOT / "configs/g009_r0_rev24_gpu_throughput.json"
ACTIVE_CONFIG_PATH = REPO_ROOT / "configs/g009_r0.json"
DEFAULT_OUTPUT = RUNS_DIR / "g009_r0_rev24_gpu_throughput_synthesis_s42.json"
SCHEMA = "g009.r0.rev24.gpu_throughput_synthesis.v1"
EVIDENCE_ID = "G009-5-E017"
REVISION = "rev24"
TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
SEED = 42
ITERATIONS = 5
ENV_ORDER = (1024, 2048)
ENTRYPOINT = "scripts/bootstrap_benchmark_g009.py"
ACTIVE_CONTRACT_SHA256 = "64eb108bb736d9ba8c1727c3a56ddc3fefaafaba25a98f93c1f8505704c5dd91"
ISAACLAB_COMMIT = "90b79bb2d44feb8d833f260f2bf37da3487180ba"
BENCHMARK_SHA256 = "2d5a88b9c07bfb38852082a0b9bf00f4213043b16ce0294776646ab06d351c82"
REQUIRED_SOURCE_PATHS = (
    "configs/g009_r0.json",
    "configs/g009_r0_rev24_gpu_throughput.json",
    "scripts/bootstrap_benchmark_g009.py",
    "scripts/run_training.ps1",
    "scripts/summarize_g009_r0_rev24_gpu_throughput.py",
    "src/isaac_walk_g009/agent_cfg.py",
    "src/isaac_walk_g009/mdp/__init__.py",
    "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/mdp/recover.py",
    "src/isaac_walk_g009/recover_contracts.py",
    "src/isaac_walk_g009/recover_env_cfg.py",
    "src/isaac_walk_g009/registry.py",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_SUCCESS_CHECKS = (
    "process_exit_zero",
    "no_traceback_or_error",
    "requested_iteration_reached",
    "log_directory_exists",
    "tensorboard_exists",
    "checkpoint_exists",
    "gpu_measurement_complete",
    "gpu_recovered_to_baseline",
)


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(payload)


def repository_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _finite_number(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and (minimum is None or number >= minimum) and (maximum is None or number <= maximum)


def _timestamp(value: Any, label: str) -> datetime:
    require(isinstance(value, str), f"{label} timestamp missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} timestamp invalid") from error
    require(parsed.tzinfo is not None, f"{label} timestamp must include timezone")
    return parsed


def load_preregistration() -> dict[str, Any]:
    value, _ = _read_json(PREREG_PATH, "rev24 preregistration")
    require(value.get("schema_version") == "g009.r0.rev24.gpu_throughput_preregistration.v1", "preregistration schema mismatch")
    require(value.get("evidence_id") == EVIDENCE_ID and value.get("revision") == REVISION, "preregistration identity mismatch")
    experiment = value.get("experiment", {})
    protocol = value.get("protocol", {})
    bindings = value.get("source_bindings", {})
    report_contract = value.get("report_contract", {})
    require(
        experiment.get("purpose") == "headless scratch PPO GPU throughput smoke"
        and experiment.get("task") == TASK
        and experiment.get("seed") == SEED
        and experiment.get("qualification_claim") is False
        and experiment.get("recovery_success_claim") is False,
        "preregistered claim boundary mismatch",
    )
    require(
        protocol.get("ordered_environment_counts") == list(ENV_ORDER)
        and protocol.get("max_iterations") == ITERATIONS
        and protocol.get("num_steps_per_env") == 24
        and protocol.get("ppo_num_learning_epochs") == 5
        and protocol.get("ppo_num_mini_batches") == 4
        and protocol.get("headless") is True
        and protocol.get("scratch") is True
        and protocol.get("resume_allowed") is False,
        "preregistered protocol mismatch",
    )
    require(
        bindings.get("isaaclab") == {"version": "v2.1.1", "commit": ISAACLAB_COMMIT}
        and bindings.get("official_benchmark")
        == {"path": "scripts/benchmarks/benchmark_rsl_rl.py", "sha256": BENCHMARK_SHA256}
        and bindings.get("repository_entrypoint") == ENTRYPOINT
        and bindings.get("active_contract")
        == {"path": "configs/g009_r0.json", "contract_sha256": ACTIVE_CONTRACT_SHA256},
        "preregistered source binding mismatch",
    )
    require(
        report_contract.get("schema_version") == 1
        and report_contract.get("repository_clean_required") is True
        and report_contract.get("source_bundle_matches_commit_required") is True
        and report_contract.get("success_checks_required_true") == list(REQUIRED_SUCCESS_CHECKS)
        and report_contract.get("steps_per_second", {}).get("exact_sample_count") == ITERATIONS
        and report_contract.get("gpu", {}).get("peak_memory_fraction_maximum") == 0.9,
        "preregistered report gate mismatch",
    )
    gpu_contract = report_contract.get("gpu", {})
    require(
        gpu_contract.get("visible_gpu_policy") == "exactly_one"
        and gpu_contract.get("power_draw_policy") == "finite_nonnegative_required"
        and gpu_contract.get("unavailable_power_draw_blocks") is True,
        "preregistered GPU measurement policy mismatch",
    )
    return value


def validate_active_contract() -> dict[str, str]:
    value, raw = _read_json(ACTIVE_CONFIG_PATH, "active G009 R0 config")
    require(value.get("contract_sha256") == ACTIVE_CONTRACT_SHA256, "active contract SHA mismatch")
    contract = value.get("contract")
    require(isinstance(contract, Mapping), "active contract payload missing")
    require(_canonical_sha256(contract) == ACTIVE_CONTRACT_SHA256, "active contract payload hash mismatch")
    return {
        "path": ACTIVE_CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
        "contract_sha256": ACTIVE_CONTRACT_SHA256,
        "file_sha256": _sha256(raw),
    }


def _binding(path: Path, raw: bytes) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    require(resolved.parent == RUNS_DIR.resolve(), "input must be a direct reports/runs JSON")
    return {"path": resolved.relative_to(REPO_ROOT.resolve()).as_posix(), "sha256": _sha256(raw)}


def _entrypoint_matches(value: Any) -> bool:
    if not isinstance(value, Mapping) or value.get("repository_internal") is not True:
        return False
    path = str(value.get("path", "")).replace("\\", "/").lower()
    if not (path.endswith("/" + ENTRYPOINT.lower()) or path == ENTRYPOINT.lower()):
        return False
    expected_hash = _sha256((REPO_ROOT / ENTRYPOINT).read_bytes())
    return value.get("sha256") == expected_hash


def _source_bundle_valid(repository: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    if not isinstance(repository.get("commit"), str) or not COMMIT_PATTERN.fullmatch(repository["commit"]):
        return False
    if repository["commit"] != repository_head():
        return False
    if repository.get("dirty") is not False or source.get("matches_repository_commit") is not True:
        return False
    files = source.get("files")
    if not isinstance(files, Mapping) or set(files) != set(REQUIRED_SOURCE_PATHS):
        return False
    for path in REQUIRED_SOURCE_PATHS:
        recorded = files.get(path)
        local = REPO_ROOT / path
        if not isinstance(recorded, str) or not SHA256_PATTERN.fullmatch(recorded) or not local.is_file():
            return False
        if _sha256(local.read_bytes()) != recorded:
            return False
    bundle_sha = source.get("sha256")
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files)).encode("utf-8")
    return (
        isinstance(bundle_sha, str)
        and SHA256_PATTERN.fullmatch(bundle_sha) is not None
        and bundle_sha == _sha256(payload)
    )


def _numeric_invalid(report: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = report.get("training_safety_aggregate")
    series = aggregate.get("numeric_invalid") if isinstance(aggregate, Mapping) else None
    if series is None:
        return {"availability": "unavailable", "blocking": False, "passed": True, "maximum": None}
    maximum = series.get("maximum") if isinstance(series, Mapping) else None
    passed = _finite_number(maximum) and float(maximum) == 0.0
    return {"availability": "available", "blocking": not passed, "passed": passed, "maximum": maximum}


def _power(gpu: Mapping[str, Any]) -> dict[str, Any]:
    value = gpu.get("peak_power_draw_w")
    if value is None:
        return {"availability": "unavailable", "blocking": True, "passed": False, "peak_power_draw_w": None}
    passed = _finite_number(value, minimum=0.0)
    return {"availability": "available", "blocking": not passed, "passed": passed, "peak_power_draw_w": value}


def evaluate_report(report: Mapping[str, Any], env_count: int, binding: Mapping[str, str]) -> dict[str, Any]:
    require(report.get("schema_version") == 1, "run report schema mismatch")
    require(report.get("task") == TASK, "task mismatch")
    require(report.get("num_envs") == env_count, "environment count mismatch")
    require(report.get("seed") == SEED, "seed mismatch")
    require(report.get("max_iterations") == ITERATIONS, "iteration count mismatch")
    require(report.get("headless") is True, "headless mismatch")
    require(_entrypoint_matches(report.get("training_entrypoint")), "training entrypoint mismatch")
    resume = report.get("resume")
    require(isinstance(resume, Mapping) and resume.get("enabled") is False, "throughput smoke must be scratch, not resumed")
    require(report.get("effective_hydra_overrides") == [], "PPO rollout/update override mismatch")
    qualification = report.get("qualification_mode")
    safety_gate = report.get("training_safety_gate")
    require(
        isinstance(qualification, Mapping)
        and qualification.get("enabled") is False
        and qualification.get("policy_qualification_status") == "not_run"
        and report.get("qualification_passed") is None,
        "throughput smoke must not claim qualification",
    )
    require(
        isinstance(safety_gate, Mapping)
        and safety_gate.get("requested") is False
        and safety_gate.get("required") is False
        and safety_gate.get("passed") is None,
        "throughput smoke training-safety mode mismatch",
    )

    repository = report.get("repository") if isinstance(report.get("repository"), Mapping) else {}
    source = report.get("source_bundle") if isinstance(report.get("source_bundle"), Mapping) else {}
    success = report.get("success_checks") if isinstance(report.get("success_checks"), Mapping) else {}
    performance = report.get("performance") if isinstance(report.get("performance"), Mapping) else {}
    gpu = report.get("gpu") if isinstance(report.get("gpu"), Mapping) else {}
    resolution = report.get("log_directory_resolution") if isinstance(report.get("log_directory_resolution"), Mapping) else {}
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), Mapping) else {}
    samples = performance.get("steps_per_second_samples")
    steps_valid = (
        isinstance(samples, list)
        and len(samples) == ITERATIONS
        and all(_finite_number(item, minimum=0.0) and float(item) > 0.0 for item in samples)
    )
    success_checks = {name: success.get(name) is True for name in REQUIRED_SUCCESS_CHECKS}
    gpu_lifecycle = gpu.get("measurement_complete") is True and gpu.get("recovered_to_baseline") is True
    total = gpu.get("total_mib")
    peak = gpu.get("peak_used_mib")
    memory_values_valid = _finite_number(total, minimum=0.0) and float(total) > 0.0 and _finite_number(peak, minimum=0.0)
    memory_fraction = float(peak) / float(total) if memory_values_valid else None
    gpu_checks = {
        "single_visible_gpu": gpu.get("device_count") == 1,
        "total_mib_valid": _finite_number(total, minimum=0.0) and float(total) > 0.0 if _finite_number(total, minimum=0.0) else False,
        "peak_used_mib_valid": _finite_number(peak, minimum=0.0),
        "peak_utilization_gpu_percent_valid": _finite_number(gpu.get("peak_utilization_gpu_percent"), minimum=0.0, maximum=100.0),
        "peak_temperature_c_valid": _finite_number(gpu.get("peak_temperature_c"), minimum=0.0),
        "peak_memory_fraction_at_most_0_90": memory_fraction is not None and memory_fraction <= 0.9,
    }
    numeric_invalid = _numeric_invalid(report)
    power = _power(gpu)
    steps_mean = statistics.fmean(float(item) for item in samples) if steps_valid else None
    steps_median = statistics.median(float(item) for item in samples) if steps_valid else None
    steps_summary_valid = (
        steps_valid
        and _finite_number(performance.get("mean_steps_per_second"), minimum=0.0)
        and _finite_number(performance.get("median_steps_per_second"), minimum=0.0)
        and math.isclose(float(performance["mean_steps_per_second"]), round(steps_mean, 2), abs_tol=1.0e-9)
        and math.isclose(float(performance["median_steps_per_second"]), round(steps_median, 2), abs_tol=1.0e-9)
    )
    source_bundle_valid = _source_bundle_valid(repository, source)
    resolution_candidates = resolution.get("candidates")
    log_resolution_valid = (
        resolution.get("mode") == "single_new_run_name_directory"
        and isinstance(resolution_candidates, list)
        and len(resolution_candidates) == 1
        and resolution.get("selected") == resolution_candidates[0]
        and artifacts.get("tensorboard_directory") == resolution.get("selected")
    )
    started_at = _timestamp(report.get("started_at"), "started_at")
    ended_at = _timestamp(report.get("ended_at"), "ended_at")
    timing_valid = ended_at > started_at and _finite_number(report.get("wall_time_seconds"), minimum=0.0)
    gates = {
        "run_health": report.get("run_health_passed") is True and report.get("passed") is True and report.get("exit_code") == 0,
        "repository_clean": repository.get("dirty") is False,
        "source_bundle_matches_commit": source_bundle_valid,
        "success_checks": all(success_checks.values()),
        "gpu_measurement_and_recovery": gpu_lifecycle,
        "steps_per_second": steps_summary_valid,
        "gpu_required_measurements": all(gpu_checks.values()),
        "gpu_power": power["passed"],
        "numeric_invalid": numeric_invalid["passed"],
        "timing": timing_valid,
        "log_directory_resolution": log_resolution_valid,
    }
    return {
        "num_envs": env_count,
        "input_report": dict(binding),
        "passed": all(gates.values()),
        "gates": gates,
        "success_checks": success_checks,
        "steps_per_second": {
            "samples": samples if isinstance(samples, list) else None,
            "sample_count": len(samples) if isinstance(samples, list) else 0,
            "mean": performance.get("mean_steps_per_second"),
            "median": performance.get("median_steps_per_second"),
        },
        "gpu": {
            "device_count": gpu.get("device_count"),
            "total_mib": total,
            "peak_used_mib": peak,
            "peak_memory_fraction": memory_fraction,
            "peak_utilization_gpu_percent": gpu.get("peak_utilization_gpu_percent"),
            "mean_utilization_gpu_percent": gpu.get("mean_utilization_gpu_percent"),
            "peak_temperature_c": gpu.get("peak_temperature_c"),
            "power": power,
            "required_measurement_checks": gpu_checks,
        },
        "numeric_invalid": numeric_invalid,
        "timing": {
            "started_at": report.get("started_at"),
            "ended_at": report.get("ended_at"),
            "wall_time_seconds": report.get("wall_time_seconds"),
        },
    }


def load_report(path: Path, env_count: int) -> tuple[dict[str, Any], dict[str, str]]:
    value, raw = _read_json(path, f"{env_count}-environment report")
    return value, _binding(path, raw)


def synthesize(input_1024: Path, input_2048: Path | None = None) -> dict[str, Any]:
    prereg = load_preregistration()
    active_contract = validate_active_contract()
    report_1024, binding_1024 = load_report(input_1024, 1024)
    row_1024 = evaluate_report(report_1024, 1024, binding_1024)
    rows = [row_1024]
    stage_2048_authorized = row_1024["passed"] is True

    if input_2048 is not None:
        require(stage_2048_authorized, "2048 report is forbidden until the 1024 gate passes")
        report_2048, binding_2048 = load_report(input_2048, 2048)
        require(binding_2048["path"] != binding_1024["path"], "duplicate input report path")
        require(binding_2048["sha256"] != binding_1024["sha256"], "duplicate input report SHA-256")
        row_2048 = evaluate_report(report_2048, 2048, binding_2048)
        require(
            _timestamp(row_1024["timing"]["ended_at"], "1024 ended_at")
            < _timestamp(row_2048["timing"]["started_at"], "2048 started_at"),
            "2048 execution must start after the 1024 execution ended",
        )
        rows.append(row_2048)

    complete = len(rows) == 2
    both_pass = complete and all(row["passed"] for row in rows)
    stable_max_envs = 2048 if both_pass else (1024 if row_1024["passed"] else None)
    return {
        "schema_version": SCHEMA,
        "evidence_id": EVIDENCE_ID,
        "revision": REVISION,
        "status": "complete" if complete else "partial",
        "experiment_class": "headless_scratch_ppo_throughput_smoke",
        "claim_limits": {
            "policy_qualification": "not_run",
            "recovery_success": "not_measured",
            "statement": prereg["experiment"]["claim_limit"],
        },
        "protocol": prereg["protocol"],
        "source_bindings": prereg["source_bindings"],
        "active_contract": active_contract,
        "rows": rows,
        "sequence_gate": {
            "environment_order": list(ENV_ORDER),
            "stage_1024_passed": row_1024["passed"],
            "stage_2048_authorized": stage_2048_authorized,
            "stage_2048_input_status": "provided" if input_2048 is not None else "missing",
        },
        "decision": {
            "passed": both_pass,
            "stable_max_envs": stable_max_envs,
            "outcome": "throughput_2048_passed" if both_pass else ("awaiting_2048_input" if row_1024["passed"] else "throughput_1024_failed"),
        },
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    require(resolved == DEFAULT_OUTPUT.resolve(), "output must be the canonical rev24 synthesis path")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-1024", required=True, type=Path)
    parser.add_argument("--input-2048", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    value = synthesize(args.input_1024, args.input_2048)
    write_json(args.output, value)
    print(json.dumps({"output": str(args.output), "decision": value["decision"]}, ensure_ascii=False))
    return 0 if value["decision"]["passed"] or value["decision"]["outcome"] == "awaiting_2048_input" else 1


if __name__ == "__main__":
    raise SystemExit(main())
