#!/usr/bin/env python3
"""Validate and summarize the rejected G009 R0 rev13 CPU runtime experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REV13_INPUTS = tuple(
    REPO_ROOT / "reports/runs" / f"g009_r0_runtime_probe_rev13_cpu_rep{index:02d}_s42.json"
    for index in range(1, 4)
)
DEFAULT_BASELINE = REPO_ROOT / "reports/runs/g009_r0_runtime_probe_rev12_cpu_rep01_s42.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports/runs/g009_r0_runtime_probe_rev13_cpu_failure_synthesis_s42.json"
EXPECTED_PEAK_BW = 15.97161865234375
EXPECTED_BASELINE_PEAK_BW = 9.332860946655273
EXPECTED_THRESHOLD_BW = 15.0
FAILED_CHECK = "nonfoot_peak_force_bounded"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def right_side_reset_pose_hold(report: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in report.get("pose_mode_metrics", [])
        if item.get("pose_id") == "right_side" and item.get("action_mode") == "reset_pose_hold"
    ]
    require(len(matches) == 1, "exactly one right_side/reset_pose_hold metric is required")
    return matches[0]


def solver_counts(report: dict[str, Any]) -> tuple[int, int]:
    articulations = (
        report.get("physics_readback", {})
        .get("articulation_solver_iterations", {})
        .get("articulations")
    )
    require(
        isinstance(articulations, list) and len(articulations) == 8,
        "eight articulation readbacks are required",
    )
    pairs = {
        (item.get("solver_position_iteration_count"), item.get("solver_velocity_iteration_count"))
        for item in articulations
    }
    require(len(pairs) == 1, "articulation solver readbacks are inconsistent")
    position, velocity = next(iter(pairs))
    require(type(position) is int and type(velocity) is int, "solver iteration counts must be integers")
    return position, velocity


def termination_totals(report: dict[str, Any]) -> dict[str, int]:
    totals = {"numeric_invalid": 0, "hard_joint_limit": 0}
    metrics = report.get("pose_mode_metrics")
    if not isinstance(metrics, list) or len(metrics) != 8:
        raise ValueError("eight pose/action metrics are required")
    for metric in metrics:
        if not isinstance(metric, dict):
            raise ValueError("each pose/action metric must be an object")
        counts = metric.get("termination_counts", {})
        if not isinstance(counts, dict):
            raise ValueError("termination_counts must be an object")
        for name in totals:
            value = counts.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"invalid termination count: {name}")
            totals[name] += value
    return totals


def false_checks(report: dict[str, Any]) -> list[str]:
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise ValueError("checks are required")
    if not all(type(value) is bool for value in checks.values()):
        raise ValueError("all checks must be booleans")
    return sorted(name for name, value in checks.items() if not value)


def validate_common_probe(report: dict[str, Any], *, position: int, velocity: int) -> None:
    require(report.get("schema_version") == 3, "runtime probe schema must be 3")
    require(report.get("goal_id") == "g009" and report.get("stage_id") == "R0", "goal/stage mismatch")
    require(report.get("probe") == "flat_recover_runtime_calibration", "probe mismatch")
    require(report.get("task") == "Isaac-G009-Recover-Flat-Go2-R0-v0", "task mismatch")
    require(report.get("seed") == 42 and report.get("device") == "cpu", "seed/device mismatch")
    require(report.get("headless") is True, "runtime probe must be headless")
    require(solver_counts(report) == (position, velocity), "actual articulation solver counts mismatch")
    require(report.get("qualification", {}).get("status") == "not_run", "qualification must be not_run")


def validate_rev13(report: dict[str, Any]) -> dict[str, Any]:
    validate_common_probe(report, position=8, velocity=1)
    execution_id = report.get("execution", {}).get("execution_id")
    require(isinstance(execution_id, str), "execution_id is required")
    try:
        parsed = uuid.UUID(hex=execution_id)
    except ValueError as exc:
        raise ValueError("execution_id must be UUID4 hex") from exc
    require(parsed.version == 4 and parsed.hex == execution_id, "execution_id must be lowercase UUID4 hex")
    require(report.get("source_bundle", {}).get("clean") is True, "source bundle must be clean")
    require(report.get("run_health", {}).get("passed") is True, "run health must pass")
    require(report.get("runtime_contract", {}).get("passed") is False, "runtime contract must fail")
    require(report.get("passed") is False, "overall runtime probe must fail")
    require(false_checks(report) == [FAILED_CHECK], f"only {FAILED_CHECK} may fail")
    require(
        termination_totals(report) == {"numeric_invalid": 0, "hard_joint_limit": 0},
        "safety termination count changed",
    )
    threshold = report.get("calibration_thresholds", {}).get("max_nonfoot_force_bodyweights")
    require(threshold == EXPECTED_THRESHOLD_BW, "non-foot peak threshold must remain 15 BW")
    metric = right_side_reset_pose_hold(report)
    require(metric.get("max_nonfoot_force_body_name") == "base", "peak body must be base")
    require(metric.get("max_nonfoot_force_physics_step") == 129, "peak physics step must be 129")
    require(metric.get("max_nonfoot_force_time_s") == 0.645, "peak time must be 0.645 s")
    peak = metric.get("max_nonfoot_force_bodyweights")
    if not isinstance(peak, (int, float)):
        raise ValueError("rev13 peak must be numeric")
    if not math.isclose(peak, EXPECTED_PEAK_BW, abs_tol=1e-12):
        raise ValueError("rev13 peak changed")
    return {
        "execution_id": execution_id,
        "started_at_utc": report["execution"]["started_at_utc"],
        "peak_bodyweights": float(peak),
        "peak_body": "base",
        "peak_time_s": 0.645,
        "peak_physics_step": 129,
        "failed_checks": [FAILED_CHECK],
        "numeric_invalid_terminations": 0,
        "hard_joint_limit_terminations": 0,
    }


def summarize(rev13_paths: Iterable[Path], baseline_path: Path) -> dict[str, Any]:
    paths = tuple(rev13_paths)
    require(len(paths) == 3, "exactly three rev13 CPU reports are required")
    require(len({path.resolve() for path in paths}) == 3, "rev13 report paths must be distinct")
    reports = [read_json(path) for path in paths]
    runs = [validate_rev13(report) for report in reports]
    require(len({run["execution_id"] for run in runs}) == 3, "three distinct execution IDs are required")

    contract_hashes = {report.get("contract_sha256") for report in reports}
    source_commits = {report.get("source_bundle", {}).get("git_commit") for report in reports}
    source_bundles = {report.get("source_bundle", {}).get("source_bundle_sha256") for report in reports}
    require(
        len(contract_hashes) == len(source_commits) == len(source_bundles) == 1,
        "rev13 lineage must match across all runs",
    )
    contract_sha256 = next(iter(contract_hashes))
    source_commit = next(iter(source_commits))
    source_bundle_sha256 = next(iter(source_bundles))
    require(isinstance(contract_sha256, str) and len(contract_sha256) == 64, "invalid contract hash")
    require(isinstance(source_commit, str) and len(source_commit) == 40, "invalid source commit")
    require(isinstance(source_bundle_sha256, str) and len(source_bundle_sha256) == 64, "invalid source bundle hash")

    baseline = read_json(baseline_path)
    validate_common_probe(baseline, position=8, velocity=0)
    require(baseline.get("run_health", {}).get("passed") is True, "rev12 baseline run health must pass")
    require(
        baseline.get("runtime_contract", {}).get("passed") is True
        and baseline.get("passed") is True,
        "rev12 baseline must pass",
    )
    require(false_checks(baseline) == [], "rev12 baseline cannot contain a failed check")
    require(
        termination_totals(baseline) == {"numeric_invalid": 0, "hard_joint_limit": 0},
        "rev12 baseline safety changed",
    )
    baseline_metric = right_side_reset_pose_hold(baseline)
    baseline_peak = float(baseline_metric["max_nonfoot_force_bodyweights"])
    require(math.isclose(baseline_peak, EXPECTED_BASELINE_PEAK_BW, abs_tol=1e-12), "rev12 baseline peak changed")

    delta = EXPECTED_PEAK_BW - baseline_peak
    percent = delta / baseline_peak * 100.0
    rev13_metric = right_side_reset_pose_hold(reports[0])
    comparison_fields = {
        "max_root_angular_speed_rad_s": (6.586225986480713, 9.659443855285645),
        "max_joint_speed_rad_s": (10.620381355285645, 7.199735164642334),
        "excess_contact_delta_v_m_s": (1.1110174655914307, 1.0345003604888916),
        "peak_step_excess_contact_delta_v_m_s": (0.9402111768722534, 0.8841875791549683),
    }
    for field, (expected_baseline, expected_rev13) in comparison_fields.items():
        require(math.isclose(float(baseline_metric[field]), expected_baseline, abs_tol=1e-12), f"rev12 {field} changed")
        require(math.isclose(float(rev13_metric[field]), expected_rev13, abs_tol=1e-12), f"rev13 {field} changed")
    inputs = [
        {"path": repo_path(path), "sha256": file_sha256(path), **run}
        for path, run in zip(paths, runs, strict=True)
    ]
    return {
        "schema_version": 1,
        "goal_id": "g009",
        "stage_number": "G009-5",
        "stage_id": "R0",
        "experiment": "rev13_cpu_runtime_failure",
        "status": "rejected",
        "diagnostic_only": True,
        "public_claim_eligible": False,
        "learned_policy_qualified": False,
        "qualification_status": "not_run",
        "conclusion": "rev13 is rejected before GPU, Gate01, Gate10, or PPO",
        "lineage": {
            "source_commit": source_commit,
            "source_bundle_sha256": source_bundle_sha256,
            "contract_sha256": contract_sha256,
            "actual_articulation_solver_iterations": {"position": 8, "velocity": 1},
        },
        "repeatability": {
            "required_runs": 3,
            "validated_runs": 3,
            "distinct_execution_ids": True,
            "identical_failure": True,
            "inputs": inputs,
        },
        "failure": {
            "failed_check": FAILED_CHECK,
            "threshold_bodyweights": EXPECTED_THRESHOLD_BW,
            "right_side_reset_pose_hold_peak_bodyweights": EXPECTED_PEAK_BW,
            "peak_body": "base",
            "peak_time_s": 0.645,
            "peak_physics_step": 129,
            "numeric_invalid_terminations": 0,
            "hard_joint_limit_terminations": 0,
        },
        "rev12_comparison": {
            "input": {"path": repo_path(baseline_path), "sha256": file_sha256(baseline_path)},
            "solver_velocity_iterations": 0,
            "right_side_reset_pose_hold_peak_bodyweights": baseline_peak,
            "absolute_increase_bodyweights": delta,
            "relative_increase_percent": percent,
            "right_side_reset_pose_hold": {
                field: {
                    "rev12": baseline_value,
                    "rev13": rev13_value,
                    "relative_change_percent": (rev13_value / baseline_value - 1.0) * 100.0,
                }
                for field, (baseline_value, rev13_value) in comparison_fields.items()
            },
            "peak_time_s": {"rev12": 0.655, "rev13": 0.645},
            "careful_interpretation": (
                "The lower total and peak-step excess delta-v do not indicate improvement: the force peak "
                "and root angular-rate peak increased. This is consistent with, but does not prove, a more "
                "temporally concentrated and rotational contact response."
            ),
        },
        "blocked_stages": {
            "gpu_runtime": True,
            "gate01": True,
            "gate10": True,
            "ppo_training": True,
        },
    }


def write_summary(rev13_paths: Iterable[Path], baseline_path: Path, output_path: Path) -> dict[str, Any]:
    require(not output_path.exists(), f"refusing to overwrite output: {output_path}")
    summary = summarize(rev13_paths, baseline_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev13", nargs=3, type=Path, default=DEFAULT_REV13_INPUTS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = write_summary(args.rev13, args.baseline, args.output)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
