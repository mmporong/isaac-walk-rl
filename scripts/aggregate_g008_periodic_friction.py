#!/usr/bin/env python3
"""Aggregate isolated G008 periodic-friction case reports into one threshold report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_g008_periodic_friction import DEFAULT_SWEEP, summarize_threshold


REPO_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def portable_report_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def summarize_qualified_threshold(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases or cases[0]["mixed"]:
        raise ValueError("threshold cases must start with a uniform baseline")
    baseline = cases[0]
    mixed_passes = [case for case in cases[1:] if case["all_directions_gate_pass"]]
    if not baseline["all_directions_gate_pass"]:
        return {
            "baseline_gate_pass": False,
            "baseline_failure": {
                "case_id": baseline["case_id"],
                "failed_directions": [
                    item["id"] for item in baseline["directions"] if not item["gate_pass"]
                ],
            },
            "contiguous_pass_floor": None,
            "first_failure": None,
            "lowest_tested_passing": None,
            "isolated_mixed_passes": [case["case_id"] for case in mixed_passes],
            "monotonic_gate_sequence": False,
        }
    summary = summarize_threshold(cases)
    return {"baseline_gate_pass": True, "baseline_failure": None, **summary}


def summarize_direction_threshold(cases: list[dict[str, Any]], direction_id: str) -> dict[str, Any]:
    projected = []
    for case in cases:
        matches = [direction for direction in case["directions"] if direction["id"] == direction_id]
        if len(matches) != 1:
            raise ValueError(f"direction {direction_id} missing or duplicated in {case['case_id']}")
        direction = matches[0]
        projected.append(
            {
                "case_id": case["case_id"],
                "mixed": case["mixed"],
                "low_static": case["low_static"],
                "low_dynamic": case["low_dynamic"],
                "all_directions_gate_pass": direction["gate_pass"],
                "directions": [direction],
            }
        )
    return summarize_qualified_threshold(projected)


def aggregate(paths: list[Path], failure_paths: list[Path] | None = None) -> dict[str, Any]:
    failure_paths = failure_paths or []
    expected_case_ids = [case["id"] for case in DEFAULT_SWEEP]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_case = {report["case"]["id"]: (path, report) for path, report in zip(paths, reports)}
    if len(by_case) != len(paths):
        raise ValueError("duplicate friction case reports")
    failure_reports = [json.loads(path.read_text(encoding="utf-8")) for path in failure_paths]
    failures_by_case = {
        report["case"]["id"]: (path, report) for path, report in zip(failure_paths, failure_reports)
    }
    if len(failures_by_case) != len(failure_paths):
        raise ValueError("duplicate friction failure reports")
    if set(by_case) & set(failures_by_case):
        raise ValueError("a friction case cannot be both complete and failed")
    if set(by_case) | set(failures_by_case) != set(expected_case_ids):
        raise ValueError(
            f"case set mismatch: complete={sorted(by_case)}, failed={sorted(failures_by_case)}"
        )

    ordered = [by_case[case_id] for case_id in expected_case_ids if case_id in by_case]
    reference = ordered[0][1]
    invariant_fields = (
        "task",
        "headless",
        "device",
        "evaluation_seed",
        "num_envs",
        "horizon_steps",
        "warmup_steps",
        "step_dt_s",
        "observation_corruption",
        "gate",
        "evaluation_source_sha256",
        "direction_evaluator_source_sha256",
    )
    for _, report in ordered[1:]:
        for field in invariant_fields:
            if report[field] != reference[field]:
                raise ValueError(f"invariant mismatch for {field}")

    policy_ids = [policy["policy_id"] for policy in reference["policies"]]
    policies = []
    for policy_id in policy_ids:
        cases = []
        checkpoint = None
        for _, report in ordered:
            matches = [policy for policy in report["policies"] if policy["policy_id"] == policy_id]
            if len(matches) != 1:
                raise ValueError(f"policy {policy_id} missing or duplicated")
            policy = matches[0]
            if checkpoint is None:
                checkpoint = policy["checkpoint"]
            elif policy["checkpoint"] != checkpoint:
                raise ValueError(f"checkpoint mismatch for policy {policy_id}")
            cases.append(policy["case"])
        policies.append(
            {
                "policy_id": policy_id,
                "checkpoint": checkpoint,
                "threshold_summary": summarize_qualified_threshold(cases),
                "direction_thresholds": {
                    direction["id"]: summarize_direction_threshold(cases, direction["id"])
                    for direction in cases[0]["directions"]
                },
                "cases": cases,
            }
        )

    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "protocol": "spatial_periodic_friction_stripes_sweep_v1",
        "task": reference["task"],
        "terrain_mode": reference["terrain_mode"],
        "contact_model": reference["contact_model"] | {"low_material": "varies_by_case"},
        "headless": reference["headless"],
        "device": reference["device"],
        "evaluation_seed": reference["evaluation_seed"],
        "num_envs_per_case": reference["num_envs"],
        "environments_per_policy": reference["environments_per_policy"],
        "environments_per_policy_direction": reference["environments_per_policy_direction"],
        "horizon_steps": reference["horizon_steps"],
        "warmup_steps": reference["warmup_steps"],
        "step_dt_s": reference["step_dt_s"],
        "observation_corruption": reference["observation_corruption"],
        "gate": reference["gate"],
        "fall_detection": reference["fall_detection"],
        "sweep": [dict(case) for case in DEFAULT_SWEEP],
        "policies": policies,
        "case_reports": [
            {
                "path": portable_report_path(path),
                "sha256": file_sha256(path),
                "case_id": report["case"]["id"],
                "wall_time_seconds": report["wall_time_seconds"],
            }
            for path, report in ordered
        ],
        "failed_evaluations": [
            {
                "path": portable_report_path(path),
                "sha256": file_sha256(path),
                "case_id": report["case"]["id"],
                "status": report["status"],
                "completed_steps_before_native_termination": report[
                    "completed_steps_before_native_termination"
                ],
                "attempt_count": len(report["attempts"]),
                "failure_classification": report["failure_classification"],
            }
            for case_id in expected_case_ids
            if case_id in failures_by_case
            for path, report in [failures_by_case[case_id]]
        ],
        "evaluation_source_sha256": reference["evaluation_source_sha256"],
        "direction_evaluator_source_sha256": reference["direction_evaluator_source_sha256"],
        "aggregation_source_sha256": file_sha256(Path(__file__)),
        "interpretation_contract": {
            "contiguous_pass_floor": "lowest coefficient reached before the first all-direction failure",
            "baseline_qualification": "a mixed-friction tolerance is withheld when the uniform nominal case fails the same gate",
            "lowest_tested_passing": "lowest isolated passing case, reported separately when results are non-monotonic",
            "non_monotonic_warning": "a later isolated pass does not erase an earlier failure",
            "failed_evaluation": "a simulator-native termination is unresolved evidence and is excluded from policy pass/fail thresholds",
            "claim_scope": "single-seed Isaac Sim stress test, not a real-floor friction guarantee",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--failure-reports", nargs="*", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in args.inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in args.failure_reports:
        if not path.is_file():
            raise FileNotFoundError(path)
    report = aggregate(
        [path.resolve() for path in args.inputs],
        [path.resolve() for path in args.failure_reports],
    )
    _write_json_atomic(args.output.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
