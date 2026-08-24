#!/usr/bin/env python3
"""Strictly validate and summarize the G005 reward-ablation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable


METRIC_NAMES = (
    "lin_vel_rmse_mps",
    "yaw_rate_rmse_radps",
    "torque_l2_mean",
    "absolute_mechanical_power_w",
    "action_rate_l2_mean",
    "feet_air_time_raw_mean",
    "mean_air_time_at_first_contact_s",
    "fall_trial_rate",
    "survival_rate",
)
SUMMARY_METRIC_NAMES = ("sample_count",) + METRIC_NAMES + (
    "first_contact_count",
    "fall_count",
    "timeout_count",
    "reset_count",
    "fall_timeout_overlap_count",
    "trials_started",
)


class ValidationError(ValueError):
    """Raised when experiment evidence violates the fixed protocol."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sample_std(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.stdev(values) if len(values) >= 2 else None


def sample_variance(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.variance(values) if len(values) >= 2 else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _experiment(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest.get("experiment", manifest)


def _training_seeds(manifest: dict[str, Any]) -> list[int]:
    experiment = _experiment(manifest)
    seeds = experiment.get("training_seeds", experiment.get("seeds"))
    _require(isinstance(seeds, list) and seeds, "manifest training_seeds/seeds must be a non-empty list")
    return [int(seed) for seed in seeds]


def _variant_weights(variant: dict[str, Any]) -> dict[str, float]:
    weights = variant.get("weights")
    _require(isinstance(weights, dict) and weights, f"variant {variant.get('name', variant.get('id'))!r} has no weights")
    return {str(key): float(value) for key, value in weights.items()}


def validate_one_factor_variants(manifest: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, float]]]:
    variants = _experiment(manifest).get("variants")
    _require(isinstance(variants, list) and len(variants) == 4, "exactly four variants are required")
    by_name: dict[str, dict[str, float]] = {}
    for variant in variants:
        name = variant.get("id", variant.get("name"))
        _require(isinstance(name, str) and name, "each variant requires id or name")
        _require(name not in by_name, f"duplicate variant: {name}")
        by_name[name] = _variant_weights(variant)
    _require("baseline" in by_name, "baseline variant is required")
    baseline = by_name["baseline"]
    _require(len(baseline) == 3, "baseline must contain exactly three reward weights")
    changed_keys: list[str] = []
    for name, weights in by_name.items():
        _require(set(weights) == set(baseline), f"variant {name} reward keys differ from baseline")
        if name == "baseline":
            continue
        changed = [key for key in baseline if weights[key] != baseline[key]]
        _require(len(changed) == 1, f"variant {name} must change exactly one reward weight")
        changed_keys.append(changed[0])
    _require(len(set(changed_keys)) == 3, "the three ablations must cover three distinct reward weights")
    return list(by_name), by_name


def compute_trial_rates(fall_count: int, timeout_count: int, reset_count: int, trials_started: int) -> dict[str, float | int | None]:
    """Compute rates while preserving fall/timeout overlap as separate event counts."""
    _require(min(fall_count, timeout_count, reset_count, trials_started) >= 0, "trial counts cannot be negative")
    _require(reset_count <= fall_count + timeout_count, "reset_count cannot exceed termination-event count")
    _require(reset_count >= max(fall_count, timeout_count), "reset_count must be the union of fall and timeout events")
    _require(max(fall_count, timeout_count, reset_count) <= trials_started, "event counts cannot exceed trials_started")
    fall_rate = safe_ratio(float(fall_count), float(trials_started))
    return {
        "fall_count": fall_count,
        "timeout_count": timeout_count,
        "reset_count": reset_count,
        "trials_started": trials_started,
        "fall_trial_rate": fall_rate,
        "survival_rate": None if fall_rate is None else 1.0 - fall_rate,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _resolve_report_path(queue_path: Path, raw_path: str) -> Path:
    if raw_path.upper().startswith("%USERPROFILE%"):
        raw_path = str(Path.home()) + raw_path[len("%USERPROFILE%") :]
    raw_path = os.path.expandvars(raw_path)
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (queue_path.parent / path).resolve()


def _validate_metric_block(metrics: dict[str, Any], key: tuple[str, int], scope: str) -> None:
    _require(isinstance(metrics, dict), f"{scope} metrics missing: {key}")
    count_names = (
        "sample_count",
        "first_contact_count",
        "fall_count",
        "timeout_count",
        "reset_count",
        "fall_timeout_overlap_count",
        "trials_started",
    )
    for count_name in count_names:
        _require(type(metrics.get(count_name)) is int and metrics[count_name] >= 0, f"invalid {count_name} in {scope}: {key}")
    _require(metrics["sample_count"] > 0, f"invalid sample_count in {scope}: {key}")
    for metric in METRIC_NAMES:
        _require(metric in metrics, f"metric {metric} missing in {scope}: {key}")
        if metrics[metric] is not None:
            _require(type(metrics[metric]) in (int, float) and math.isfinite(metrics[metric]), f"invalid {metric} in {scope}: {key}")
    nonnegative_metrics = (
        "lin_vel_rmse_mps",
        "yaw_rate_rmse_radps",
        "torque_l2_mean",
        "absolute_mechanical_power_w",
        "action_rate_l2_mean",
        "mean_air_time_at_first_contact_s",
    )
    for metric in nonnegative_metrics:
        if metrics[metric] is not None:
            _require(metrics[metric] >= 0, f"negative physical metric {metric} in {scope}: {key}")
    rates = compute_trial_rates(metrics["fall_count"], metrics["timeout_count"], metrics["reset_count"], metrics["trials_started"])
    _require(
        metrics["fall_timeout_overlap_count"] <= min(metrics["fall_count"], metrics["timeout_count"]),
        f"fall/timeout overlap exceeds an event count in {scope}: {key}",
    )
    _require(
        metrics["fall_timeout_overlap_count"] == metrics["fall_count"] + metrics["timeout_count"] - metrics["reset_count"],
        f"inconsistent fall/timeout overlap in {scope}: {key}",
    )
    for rate_name in ("fall_trial_rate", "survival_rate"):
        expected = rates[rate_name]
        actual = metrics[rate_name]
        _require(
            (expected is None and actual is None) or (expected is not None and actual is not None and math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12)),
            f"inconsistent {rate_name} in {scope}: {key}",
        )


def _close(actual: float | None, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=1e-6)


def _validate_runtime_evidence(report: dict[str, Any], key: tuple[str, int]) -> None:
    runtime = report.get("runtime_evidence")
    _require(isinstance(runtime, dict), f"runtime_evidence missing: {key}")
    _require(runtime.get("exit_code") == 0, f"runtime exit_code is not zero: {key}")
    _require(runtime.get("app_close_completed") is True, f"runtime app close is incomplete: {key}")
    _require(runtime.get("finalized_after_process_exit") is True, f"runtime finalization is incomplete: {key}")
    _require(runtime.get("gpu_recovered_to_baseline") is True, f"GPU recovery is incomplete: {key}")
    _require(runtime.get("process_recovered") is True, f"process recovery is incomplete: {key}")
    gpu_after = runtime.get("gpu_after")
    _require(isinstance(gpu_after, dict) and gpu_after.get("measurement_complete") is True, f"GPU measurement is incomplete: {key}")
    fatal_scan = runtime.get("fatal_scan")
    _require(isinstance(fatal_scan, dict) and fatal_scan.get("measurement_complete") is True, f"fatal scan is incomplete: {key}")
    _require(fatal_scan.get("count") == 0 and fatal_scan.get("patterns") == [], f"fatal patterns found: {key}")


def _validate_overall_consistency(
    overall: dict[str, Any], by_command: list[dict[str, Any]], expected_trials_per_condition: int, key: tuple[str, int]
) -> None:
    for item in by_command:
        _require(
            item["trials_started"] == expected_trials_per_condition,
            f"per-condition trial mismatch for {item['command']['id']}: {key}",
        )
    additive = (
        "sample_count",
        "first_contact_count",
        "fall_count",
        "timeout_count",
        "reset_count",
        "fall_timeout_overlap_count",
        "trials_started",
    )
    for metric in additive:
        _require(overall[metric] == sum(item[metric] for item in by_command), f"overall {metric} mismatch: {key}")

    sample_total = overall["sample_count"]
    contact_total = overall["first_contact_count"]
    mean_by_sample = (
        "torque_l2_mean",
        "absolute_mechanical_power_w",
        "action_rate_l2_mean",
        "feet_air_time_raw_mean",
    )
    for metric in mean_by_sample:
        expected = None if sample_total == 0 else sum(item[metric] * item["sample_count"] for item in by_command) / sample_total
        _require(_close(overall[metric], expected), f"overall weighted {metric} mismatch: {key}")
    for metric in ("lin_vel_rmse_mps", "yaw_rate_rmse_radps"):
        expected = None
        if sample_total:
            expected = math.sqrt(sum(item[metric] ** 2 * item["sample_count"] for item in by_command) / sample_total)
        _require(_close(overall[metric], expected), f"overall weighted {metric} mismatch: {key}")
    expected_air = None
    if contact_total:
        expected_air = sum(
            item["mean_air_time_at_first_contact_s"] * item["first_contact_count"]
            for item in by_command
            if item["first_contact_count"] > 0
        ) / contact_total
    _require(_close(overall["mean_air_time_at_first_contact_s"], expected_air), f"overall contact air-time mismatch: {key}")
    for metric in ("fall_trial_rate", "survival_rate"):
        expected = sum(item[metric] * item["trials_started"] for item in by_command) / overall["trials_started"]
        _require(_close(overall[metric], expected), f"overall weighted {metric} mismatch: {key}")


def _expected_commands(protocol: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    grid = protocol.get("command_grid")
    _require(isinstance(grid, dict), "evaluation_protocol.command_grid is required")
    expected_axes = {
        "vx_mps": [-1.0, 0.0, 1.0],
        "vy_mps": [-0.5, 0.0, 0.5],
        "yaw_rate_radps": [-0.5, 0.0, 0.5],
    }
    for name, expected in expected_axes.items():
        _require(grid.get(name) == expected, f"command_grid.{name} differs from the fixed protocol")
    _require(grid.get("exclude") == [[0.0, 0.0, 0.0]], "command_grid.exclude differs from the fixed protocol")
    _require(
        int(grid.get("environments_per_condition", -1)) == int(protocol["environments_per_condition"]),
        "command_grid environments_per_condition mismatch",
    )
    commands: dict[str, dict[str, float | str]] = {}
    for vx in grid["vx_mps"]:
        for vy in grid["vy_mps"]:
            for yaw in grid["yaw_rate_radps"]:
                if [vx, vy, yaw] == [0.0, 0.0, 0.0]:
                    continue
                command = {
                    "id": f"vx{vx:+.1f}_vy{vy:+.1f}_yaw{yaw:+.1f}",
                    "vx_mps": vx,
                    "vy_mps": vy,
                    "yaw_rate_radps": yaw,
                }
                commands[command["id"]] = command
    return commands


def validate_evidence(
    manifest: dict[str, Any], queue: dict[str, Any], queue_path: Path
) -> tuple[dict[tuple[str, int], dict[str, Any]], str, dict[str, dict[str, float]]]:
    variant_names, weights = validate_one_factor_variants(manifest)
    manifest_config_sha = canonical_sha256(manifest)
    variants_by_name = {
        variant.get("id", variant.get("name")): variant for variant in _experiment(manifest)["variants"]
    }
    canonical_variant_hashes = {name: canonical_sha256(variant) for name, variant in variants_by_name.items()}
    declared_variant_hashes = manifest.get("variant_sha256")
    _require(isinstance(declared_variant_hashes, dict), "manifest variant_sha256 is required")
    for name, expected_hash in canonical_variant_hashes.items():
        _require(declared_variant_hashes.get(name) == expected_hash, f"manifest variant hash mismatch: {name}")
    seeds = _training_seeds(manifest)
    _require(len(seeds) == 3 and len(set(seeds)) == 3, "exactly three distinct training seeds are required")
    protocol = manifest.get("evaluation_protocol")
    _require(isinstance(protocol, dict), "manifest evaluation_protocol is required")
    protocol_sha = canonical_sha256(protocol)
    protocol_seed = int(protocol["seed"])
    protocol_envs = int(protocol["num_envs"])
    protocol_horizon = int(protocol["horizon_steps"])
    protocol_step_dt = float(protocol["step_dt"])
    protocol_conditions = int(protocol["command_grid_conditions"])
    expected_commands = _expected_commands(protocol)
    _require(len(expected_commands) == protocol_conditions, "command grid condition count mismatch")
    _require(protocol_envs == protocol_conditions * int(protocol["environments_per_condition"]), "protocol num_envs mismatch")
    jobs = queue.get("jobs")
    _require(queue.get("schema_version") == 2, "queue schema_version must be 2")
    _require("canonical_config_sha256" not in queue, "legacy canonical_config_sha256 is not accepted")
    config_reference = queue.get("config_path")
    _require(isinstance(config_reference, str) and config_reference, "queue config_path is required")
    config_path = _resolve_report_path(queue_path, config_reference)
    _require(config_path.is_file(), "queue config_path is not usable")
    manifest_file_bytes = config_path.read_bytes()
    try:
        manifest_from_file = json.loads(manifest_file_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"queue config_path is not valid UTF-8 JSON: {exc}") from exc
    _require(canonical_json_bytes(manifest_from_file) == canonical_json_bytes(manifest), "manifest argument differs from queue config file")
    manifest_file_sha = hashlib.sha256(manifest_file_bytes).hexdigest()
    _require(isinstance(jobs, list), "queue jobs must be a list")
    _require(queue.get("mode") == "production", "queue mode must be production")
    _require(queue.get("status") == "complete", "queue status must be complete")
    _require(queue.get("config_sha256") == manifest_config_sha, "queue canonical config hash mismatch")
    _require(queue.get("config_file_sha256") == manifest_file_sha, "queue config_file_sha256 mismatch")
    _require(queue.get("protocol_sha256") == protocol_sha, "queue protocol hash mismatch")
    expected = {(variant, seed) for variant in variant_names for seed in seeds}
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    seen: set[tuple[str, int]] = set()
    for job in jobs:
        variant = job.get("variant", job.get("variant_id"))
        seed = int(job.get("training_seed", job.get("seed", -1)))
        key = (variant, seed)
        _require(key in expected, f"unexpected queue job: {key}")
        _require(key not in seen, f"duplicate queue job: {key}")
        seen.add(key)
        _require(job.get("id") == f"{variant}-s{seed}", f"job id mismatch: {key}")
        _require(job.get("status") == "complete", f"job is not complete: {key}")
        _require(job.get("config_sha256") == manifest_config_sha, f"job canonical config hash mismatch: {key}")
        _require(job.get("config_file_sha256") == manifest_file_sha, f"job config_file_sha256 mismatch: {key}")
        _require(job.get("protocol_sha256") == protocol_sha, f"job protocol hash mismatch: {key}")
        report_raw = job.get("evaluation_report_path", job.get("eval_report_path", job.get("report_path")))
        _require(isinstance(report_raw, str) and report_raw, f"job report_path missing: {key}")
        report = _read_json(_resolve_report_path(queue_path, report_raw))
        _require(report.get("schema_version") == 1, f"evaluation report schema_version must be 1: {key}")
        _require(report.get("variant") == variant, f"report variant mismatch: {key}")
        _require(int(report.get("training_seed", -1)) == seed, f"report training seed mismatch: {key}")
        _require(report.get("protocol_sha256") == protocol_sha, f"protocol hash mismatch: {key}")
        _require(report.get("protocol_compliant") is True, f"non-protocol report cannot complete G005: {key}")
        _require(int(report.get("evaluation_seed", -1)) == protocol_seed, f"evaluation seed mismatch: {key}")
        _require(report.get("task") == protocol["task"], f"evaluation task mismatch: {key}")
        _require(int(report.get("num_envs", -1)) == protocol_envs, f"evaluation num_envs mismatch: {key}")
        _require(int(report.get("horizon_steps", -1)) == protocol_horizon, f"evaluation horizon mismatch: {key}")
        _require(math.isclose(float(report.get("step_dt", -1.0)), protocol_step_dt), f"evaluation step_dt mismatch: {key}")
        _require(report.get("effective_weights") == weights[variant], f"effective weights mismatch: {key}")
        denominators = report.get("denominators")
        _require(isinstance(denominators, dict), f"explicit denominators missing: {key}")
        for denominator_name in (
            "sample_count",
            "fall_trial_rate",
            "survival_rate",
            "trials_started",
            "fall_timeout_overlap_count",
        ):
            _require(
                isinstance(denominators.get(denominator_name), str) and denominators[denominator_name],
                f"denominator {denominator_name} missing: {key}",
            )
        _require(report.get("config_sha256") == manifest_config_sha, f"canonical config hash mismatch: {key}")
        _require(report.get("config_file_sha256") == manifest_file_sha, f"config_file_sha256 mismatch: {key}")
        _validate_runtime_evidence(report, key)
        _require(
            job.get("variant_config_sha256") == canonical_variant_hashes[variant]
            and report.get("variant_config_sha256") == canonical_variant_hashes[variant],
            f"variant_config_sha256 mismatch: {key}",
        )
        checkpoint = report.get("checkpoint")
        _require(isinstance(checkpoint, dict), f"checkpoint reference missing: {key}")
        checkpoint_reference = checkpoint.get("reference")
        _require(isinstance(checkpoint_reference, str) and checkpoint_reference, f"checkpoint reference missing: {key}")
        checkpoint_path = _resolve_report_path(queue_path, checkpoint_reference)
        _require(checkpoint_path.is_file(), f"checkpoint reference is not usable: {key}")
        actual_checkpoint_sha = file_sha256(checkpoint_path)
        _require(
            job.get("checkpoint_sha256") == actual_checkpoint_sha
            and report.get("checkpoint_sha256") == actual_checkpoint_sha
            and checkpoint.get("sha256") == actual_checkpoint_sha,
            f"checkpoint_sha256 mismatch: {key}",
        )
        report_metrics = report.get("metrics", {})
        overall = report_metrics.get("overall")
        _validate_metric_block(overall, key, "overall")
        by_command = report_metrics.get("by_command")
        _require(isinstance(by_command, list) and len(by_command) == protocol_conditions, f"by_command completeness mismatch: {key}")
        command_ids: set[str] = set()
        for item in by_command:
            command = item.get("command", {})
            command_id = command.get("id")
            _require(
                isinstance(command_id, str)
                and command_id not in command_ids
                and command == expected_commands.get(command_id),
                f"duplicate/invalid command definition: {key}",
            )
            command_ids.add(command_id)
            _validate_metric_block(item, key, f"command {command_id}")
        _require(command_ids == set(expected_commands), f"command coverage mismatch: {key}")
        _validate_overall_consistency(overall, by_command, int(protocol["environments_per_condition"]), key)
        reports[key] = report
    _require(seen == expected, f"queue completeness mismatch; missing={sorted(expected - seen)}, extra={sorted(seen - expected)}")
    return reports, protocol_sha, weights


def _practical_thresholds(manifest: dict[str, Any]) -> dict[str, float]:
    raw = manifest.get("practical_thresholds", {})
    common = raw.get("tracking_energy_relative", raw.get("relative", 0.05))
    return {
        "tracking_relative": float(raw.get("tracking_relative", common)),
        "energy_relative": float(raw.get("energy_relative", common)),
        "fall_absolute": float(raw.get("fall_rate_absolute", raw.get("fall_absolute", 0.02))),
    }


def summarize(manifest: dict[str, Any], queue: dict[str, Any], queue_path: Path) -> dict[str, Any]:
    reports, protocol_sha, weights = validate_evidence(manifest, queue, queue_path)
    seeds = _training_seeds(manifest)
    variants = list(weights)
    thresholds = _practical_thresholds(manifest)
    results: dict[str, Any] = {}
    baseline_values: dict[str, list[float | None]] = {}

    for variant in variants:
        metric_results: dict[str, Any] = {}
        for metric in SUMMARY_METRIC_NAMES:
            values = [reports[(variant, seed)]["metrics"]["overall"][metric] for seed in seeds]
            numeric = [float(value) for value in values if value is not None]
            metric_results[metric] = {
                "values_by_seed": {str(seed): value for seed, value in zip(seeds, values)},
                "mean": statistics.fmean(numeric) if len(numeric) == len(values) else None,
                "sample_std": sample_std(numeric) if len(numeric) == len(values) else None,
                "sample_variance": sample_variance(numeric) if len(numeric) == len(values) else None,
            }
            if variant == "baseline":
                baseline_values[metric] = values
        results[variant] = {"effective_weights": weights[variant], "metrics": metric_results}

    for variant in variants:
        if variant == "baseline":
            continue
        for metric in SUMMARY_METRIC_NAMES:
            values = [results[variant]["metrics"][metric]["values_by_seed"][str(seed)] for seed in seeds]
            base = baseline_values[metric]
            deltas = [None if value is None or baseline is None else float(value) - float(baseline) for value, baseline in zip(values, base)]
            numeric = [value for value in deltas if value is not None]
            mean_delta = statistics.fmean(numeric) if len(numeric) == len(deltas) else None
            baseline_mean = results["baseline"]["metrics"][metric]["mean"]
            relative_delta = None if mean_delta is None or baseline_mean in (None, 0) else mean_delta / abs(baseline_mean)
            if metric in ("lin_vel_rmse_mps", "yaw_rate_rmse_radps"):
                threshold_kind = "tracking_relative"
            elif metric in ("torque_l2_mean", "absolute_mechanical_power_w", "action_rate_l2_mean"):
                threshold_kind = "energy_relative"
            elif metric == "fall_trial_rate":
                threshold_kind = "fall_absolute"
            else:
                threshold_kind = None
            practical_value = mean_delta if threshold_kind == "fall_absolute" else relative_delta
            practical_threshold = None if threshold_kind is None else thresholds[threshold_kind]
            results[variant]["metrics"][metric]["paired_vs_baseline"] = {
                "deltas_by_seed": {str(seed): value for seed, value in zip(seeds, deltas)},
                "mean_delta": mean_delta,
                "sample_std_delta": sample_std(numeric) if len(numeric) == len(deltas) else None,
                "sample_variance_delta": sample_variance(numeric) if len(numeric) == len(deltas) else None,
                "relative_mean_delta": relative_delta,
                "practical_threshold_kind": threshold_kind,
                "practical_threshold": practical_threshold,
                "exceeds_practical_threshold": None
                if practical_value is None or practical_threshold is None
                else abs(practical_value) >= practical_threshold,
            }

    return {
        "schema_version": 1,
        "experiment_name": _experiment(manifest).get("name", _experiment(manifest).get("experiment_name")),
        "protocol_sha256": protocol_sha,
        "job_completeness": {"expected": 12, "complete": len(reports), "variants": 4, "seeds_per_variant": 3},
        "practical_thresholds": thresholds,
        "warnings": [
            "n=3이므로 표본 표준편차와 paired delta는 탐색적 근거이며 통계적 검정력이 제한됩니다.",
            "absolute_mechanical_power_w는 |applied_torque * joint_velocity| 합의 시뮬레이션 proxy이며 전기 에너지 측정값이 아닙니다.",
        ],
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--queue", "--queue-state", dest="queue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = _read_json(args.manifest.resolve())
    queue_path = args.queue.resolve()
    result = summarize(manifest, _read_json(queue_path), queue_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["job_completeness"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
