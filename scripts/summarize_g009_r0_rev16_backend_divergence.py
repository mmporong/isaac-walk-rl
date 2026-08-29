#!/usr/bin/env python3
"""Fail-closed sequential synthesis for the G009 R0 rev16 backend probe."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "reports/runs"
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import probe_g009_r0_rev16_backend_divergence as raw_probe

GROUP_SIZE = 3
ALLOWED_INPUT_COUNTS = (3, 6, 9, 12)
GROUP_ORDER = (("A", "cpu"), ("A", "cuda:0"), ("B", "cpu"), ("B", "cuda:0"))
EXPECTED_REFERENCE = {
    "A": (
        "g009_r0_recover_rev12",
        "d4b48d2b5fc1ea7684684a6324ba22fbfae767effeae45668c7310df382392e0",
        "rev12",
    ),
    "B": (
        "g009_r0_recover_rev15",
        "5f29ba19458404b5009d3734294c57e79294efecc7fe03bf8c71c71656129832",
        "rev15",
    ),
}
HISTORICAL_REPORTS = {
    ("A", "cpu"): (
        "reports/runs/g009_r0_runtime_probe_rev12_cpu_rep01_s42.json",
        "fb8bad2190389c3e964d1807a0f54ea700ddfd6919765105c04b93bfa8c7dd75",
    ),
    ("A", "cuda:0"): (
        "reports/runs/g009_r0_runtime_probe_rev12_gpu_rep01_s42.json",
        "e485a3fcab5d8f8e6a793d30f76fb0a3ce346e27ed89a158409862e3e32414d1",
    ),
    ("B", "cpu"): (
        "reports/runs/g009_r0_runtime_probe_rev15_cpu_rep01_s42.json",
        "426f4fe1085aeddad52c77d98fc74a55907dcc90d7084ebe8b4fde736b60e9d5",
    ),
    ("B", "cuda:0"): (
        "reports/runs/g009_r0_runtime_probe_rev15_gpu_rep01_s42.json",
        "e24674a1ed33c38fbe5f12d19dc068167b9787e75323efbe55629bf059839b91",
    ),
}
POSE_ORDER = ("prone", "supine", "left_side", "right_side")
ACTION_ORDER = ("zero_normalized", "reset_pose_hold")
POSE_FIELDS = (
    "env_index",
    "pose_id",
    "action_mode",
    "max_nonfoot_force_bodyweights",
    "max_nonfoot_force_physics_step",
    "max_nonfoot_force_body_index",
    "max_nonfoot_force_body_name",
    "max_root_angular_speed_rad_s",
    "max_joint_speed_rad_s",
    "min_contact_separation_m",
    "termination_counts",
)
PROJECTION_PAIR_EXACT_FIELDS = tuple(
    field for field in POSE_FIELDS if field != "max_nonfoot_force_bodyweights"
)
TRACE_FIELDS = (
    "input_action",
    "raw_action",
    "processed_ema_target_rad",
    "ema_previous_before_rad",
    "ema_previous_after_rad",
)
SAFETY_ZERO = {"numeric_invalid": 0, "hard_joint_limit": 0}
PREDECESSOR_REQUIREMENTS = {
    ("A", "cpu"): None,
    ("A", "cuda:0"): (3, "A.cuda:0"),
    ("B", "cpu"): (6, "B.cpu"),
    ("B", "cuda:0"): (9, "B.cuda:0"),
}
FORCE_THRESHOLD_BODYWEIGHTS = 15.0
PROJECTION_PAIR_FORCE_BW_ABS_TOLERANCE = 4.0e-6
CONCENTRATION_RATIO_THRESHOLD = 1.20
TRACE_TOLERANCE = 1.0e-6
DIVERGENCE_TOLERANCES = {
    "base_force_bodyweights": 1.0e-6,
    "root_state_w": 1.0e-6,
    "joint_position_rad": 1.0e-6,
    "joint_velocity_rad_s": 1.0e-6,
    "applied_torque_nm": 1.0e-6,
    "raw_action": 1.0e-6,
    "processed_ema_target_rad": 1.0e-6,
    "ema_previous_before_rad": 1.0e-6,
    "ema_previous_after_rad": 1.0e-6,
}
CONTACT_EXERCISE_THRESHOLD_N = float(
    raw_probe.runtime_probe.CONTACT_EXERCISE_THRESHOLD_N
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_dict(value: Any, message: str) -> dict[str, Any]:
    require(isinstance(value, dict), message)
    assert isinstance(value, dict)
    return value


def require_list(value: Any, message: str) -> list[Any]:
    require(isinstance(value, list), message)
    assert isinstance(value, list)
    return value


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be a JSON number")
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def _hex(value: Any, length: int, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value),
        f"{label} must be lowercase hexadecimal",
    )
    return value


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = path.resolve(strict=True)
    require(
        resolved.parent == RUNS_DIR.resolve(),
        f"input must be a direct child of reports/runs: {path}",
    )
    raw = resolved.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    assert isinstance(value, dict)
    return value, {
        "path": f"reports/runs/{resolved.name}",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _historical_pose_projection(
    report: dict[str, Any], device: str
) -> list[dict[str, Any]]:
    metrics = require_list(
        report.get("pose_mode_metrics"), "historical target missing 8 pose metrics"
    )
    require(
        len(metrics) == 8,
        "historical target missing 8 pose metrics",
    )
    return [
        {
            field: (
                None
                if field == "min_contact_separation_m" and device != "cpu"
                else item.get(field)
            )
            for field in POSE_FIELDS
        }
        for item in metrics
        if isinstance(item, dict)
    ]


def load_historical_target(arm: str, device: str) -> dict[str, Any]:
    relative_path, expected_sha = HISTORICAL_REPORTS[(arm, device)]
    path = REPO_ROOT / relative_path
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == expected_sha,
        "historical target report hash mismatch",
    )
    report = json.loads(raw.decode("utf-8"))
    require(isinstance(report, dict), "historical target root must be an object")
    assert isinstance(report, dict)
    pose_metrics = _historical_pose_projection(report, device)
    historical_comparison_pose_metrics = copy_pose_metrics(pose_metrics)
    candidate_checks = _runtime_candidate_checks(pose_metrics, device)
    checks = {
        "reference_report_sha256_exact": True,
        "reference_contract_sha256_exact": True,
        "pose_metric_fingerprint_within_1e_6": True,
    }
    return {
        "reference_contract_id": EXPECTED_REFERENCE[arm][0],
        "reference_contract_sha256": EXPECTED_REFERENCE[arm][1],
        "reference_report_path": relative_path,
        "reference_report_sha256": expected_sha,
        "historical_projection_derivation": dict(
            raw_probe.HISTORICAL_PROJECTION_DERIVATION
        ),
        "pose_metrics": pose_metrics,
        "historical_comparison_pose_metrics": historical_comparison_pose_metrics,
        "projection_pair_crosscheck": _projection_pair_crosscheck(
            pose_metrics,
            historical_comparison_pose_metrics,
        ),
        "checks": checks,
        "passed": True,
        "runtime_candidate_checks": candidate_checks,
        "runtime_candidate_passed": all(candidate_checks.values()),
        "matches_historical_reference": True,
        "progression_allowed": False,
    }


def copy_pose_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **item,
            "termination_counts": dict(item["termination_counts"]),
        }
        for item in metrics
    ]


def _metric_matches(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    if set(observed) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual = observed[key]
        if type(expected_value) in (int, float) and type(actual) in (int, float):
            if not math.isclose(
                float(actual), float(expected_value), rel_tol=0.0, abs_tol=1.0e-6
            ):
                return False
        elif actual != expected_value:
            return False
    return True


def _runtime_candidate_checks(
    metrics: list[dict[str, Any]], device: str
) -> dict[str, bool]:
    return {
        "all_cells_finite": all(
            math.isfinite(float(item["max_nonfoot_force_bodyweights"]))
            and math.isfinite(float(item["max_root_angular_speed_rad_s"]))
            and math.isfinite(float(item["max_joint_speed_rad_s"]))
            for item in metrics
        ),
        "max_nonfoot_force_at_most_15_bodyweights": all(
            float(item["max_nonfoot_force_bodyweights"]) <= FORCE_THRESHOLD_BODYWEIGHTS
            for item in metrics
        ),
        "safety_termination_zero": all(
            item["termination_counts"].get("numeric_invalid") == 0
            and item["termination_counts"].get("hard_joint_limit") == 0
            for item in metrics
        ),
        "cpu_separation_at_least_minus_1cm": (
            all(
                item["min_contact_separation_m"] is not None
                and float(item["min_contact_separation_m"]) >= -0.01
                for item in metrics
            )
            if device == "cpu"
            else True
        ),
    }


def _projection_pair_crosscheck(
    candidate_metrics: list[dict[str, Any]],
    historical_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Independently verify that both projections describe the same events."""

    require(
        len(candidate_metrics) == 8 and len(historical_metrics) == 8,
        "projection pair must contain 8 pose metrics per projection",
    )
    max_force_bw_abs_delta = 0.0
    for row_index, (candidate, historical) in enumerate(
        zip(candidate_metrics, historical_metrics, strict=True)
    ):
        require(
            set(candidate) == set(POSE_FIELDS) and set(historical) == set(POSE_FIELDS),
            f"projection pair row {row_index} field set mismatch",
        )
        for field in PROJECTION_PAIR_EXACT_FIELDS:
            require(
                type(candidate[field]) is type(historical[field])
                and candidate[field] == historical[field],
                f"projection pair row {row_index} shared field {field} mismatch",
            )

        candidate_value = candidate["max_nonfoot_force_bodyweights"]
        historical_value = historical["max_nonfoot_force_bodyweights"]
        require(
            type(candidate_value) in (int, float)
            and type(historical_value) in (int, float),
            f"projection pair row {row_index} force BW values must be numeric",
        )
        candidate_bw = float(candidate_value)
        historical_bw = float(historical_value)
        require(
            math.isfinite(candidate_bw)
            and math.isfinite(historical_bw)
            and candidate_bw >= 0.0
            and historical_bw >= 0.0,
            f"projection pair row {row_index} force BW values must be finite and nonnegative",
        )
        force_bw_abs_delta = abs(candidate_bw - historical_bw)
        require(
            force_bw_abs_delta <= PROJECTION_PAIR_FORCE_BW_ABS_TOLERANCE,
            f"projection pair row {row_index} force BW compatibility tolerance exceeded",
        )
        require(
            (candidate_bw <= FORCE_THRESHOLD_BODYWEIGHTS)
            == (historical_bw <= FORCE_THRESHOLD_BODYWEIGHTS),
            f"projection pair row {row_index} force threshold classification mismatch",
        )
        max_force_bw_abs_delta = max(max_force_bw_abs_delta, force_bw_abs_delta)

    return {
        "force_bw_abs_tolerance": PROJECTION_PAIR_FORCE_BW_ABS_TOLERANCE,
        "force_threshold_bodyweights": FORCE_THRESHOLD_BODYWEIGHTS,
        "max_force_bw_abs_delta": max_force_bw_abs_delta,
        "all_shared_fields_exact": True,
        "all_force_bw_finite_nonnegative": True,
        "all_force_bw_within_tolerance": True,
        "all_force_threshold_classifications_equal": True,
        "passed": True,
    }


def _validate_execution(report: dict[str, Any], evidence: dict[str, str]) -> str:
    execution = require_dict(
        report.get("execution"), "execution provenance is required"
    )
    execution_id = execution.get("execution_id")
    require(isinstance(execution_id, str), "execution_id must be UUID4 hex")
    assert isinstance(execution_id, str)
    try:
        parsed = uuid.UUID(hex=execution_id)
    except ValueError as exc:
        raise ValueError("execution_id must be UUID4 hex") from exc
    require(
        parsed.version == 4 and parsed.hex == execution_id,
        "execution_id must be lowercase UUID4 hex",
    )
    require(execution.get("no_overwrite") is True, "raw report must use no-overwrite")
    require(
        execution.get("output_path_repo_relative") == evidence["path"],
        "raw report output path binding mismatch",
    )
    timestamp = execution.get("started_at_utc")
    require(
        isinstance(timestamp, str) and timestamp.endswith("Z"),
        "execution timestamp must be UTC RFC3339",
    )
    assert isinstance(timestamp, str)
    try:
        parsed_time = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("execution timestamp must be valid UTC RFC3339") from exc
    require(parsed_time.tzinfo == timezone.utc, "execution timestamp must be UTC")
    return execution_id


def _validate_source_bundle(report: dict[str, Any]) -> tuple[str, str]:
    bundle = require_dict(report.get("source_bundle"), "source bundle is required")
    require(bundle.get("git_commit_valid") is True, "source commit is not validated")
    require(
        bundle.get("all_files_present") is True and bundle.get("clean") is True,
        "source bundle must be complete and clean",
    )
    require(
        bundle.get("missing_files") == [] and bundle.get("dirty_source_paths") == [],
        "source bundle contains unresolved paths",
    )
    commit = _hex(bundle.get("git_commit"), 40, "source commit")
    digest = _hex(bundle.get("source_bundle_sha256"), 64, "source bundle hash")
    paths = require_list(
        bundle.get("source_binding_paths"), "source paths are required"
    )
    files = require_dict(
        bundle.get("source_binding_files"), "source files are required"
    )
    require(
        bool(paths) and len(paths) == len(set(paths)),
        "source paths must be unique",
    )
    require(
        set(files) == set(paths),
        "source file/path binding mismatch",
    )
    require(
        set(paths) == set(raw_probe.SOURCE_BINDING_PATHS),
        "rev16 source binding path set mismatch",
    )
    for path, value in files.items():
        require(isinstance(path, str), "source path must be a string")
        _hex(value, 64, f"source file hash: {path}")
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    require(
        hashlib.sha256(payload.encode()).hexdigest() == digest,
        "source bundle digest mismatch",
    )
    return commit, digest


def _expected_pose_identity() -> list[tuple[int, str, str]]:
    return [
        (index, pose, action)
        for index, (action, pose) in enumerate(
            (action, pose) for action in ACTION_ORDER for pose in POSE_ORDER
        )
    ]


def _validated_pose_projection(value: Any, label: str) -> list[dict[str, Any]]:
    metrics = require_list(value, f"{label} must contain 8 pose metrics")
    require(len(metrics) == 8, f"{label} must contain 8 pose metrics")
    require(
        all(isinstance(item, dict) for item in metrics),
        f"{label} pose metrics must all be objects",
    )
    typed_metrics = [item for item in metrics if isinstance(item, dict)]
    require(
        all(set(item) == set(POSE_FIELDS) for item in typed_metrics),
        f"{label} pose metric field set mismatch",
    )
    observed_identity = [
        (item["env_index"], item["pose_id"], item["action_mode"])
        for item in typed_metrics
    ]
    require(
        observed_identity == _expected_pose_identity(),
        f"{label} pose/action mapping mismatch",
    )
    return typed_metrics


def _validate_historical_summary(
    report: dict[str, Any], arm: str, device: str
) -> tuple[bool, bool]:
    summary = require_dict(
        report.get("historical_runtime_summary"),
        "historical_runtime_summary is required",
    )
    contract_id, contract_sha, _ = EXPECTED_REFERENCE[arm]
    require(
        summary.get("reference_contract_id") == contract_id,
        "historical reference contract id mismatch",
    )
    require(
        summary.get("reference_contract_sha256") == contract_sha,
        "historical reference contract hash mismatch",
    )
    reference_path, reference_sha = HISTORICAL_REPORTS[(arm, device)]
    require(
        summary.get("reference_report_path") == reference_path,
        "historical report path mismatch",
    )
    require(
        summary.get("reference_report_sha256") == reference_sha,
        "historical report hash mismatch",
    )
    require(
        summary.get("historical_projection_derivation")
        == raw_probe.HISTORICAL_PROJECTION_DERIVATION,
        "historical projection derivation mismatch",
    )
    projection = summary
    candidate_metrics = _validated_pose_projection(
        projection.get("pose_metrics"),
        "runtime candidate summary",
    )
    historical_metrics = _validated_pose_projection(
        projection.get("historical_comparison_pose_metrics"),
        "historical comparison summary",
    )
    projection_pair_crosscheck = _projection_pair_crosscheck(
        candidate_metrics,
        historical_metrics,
    )
    require(
        projection.get("projection_pair_crosscheck") == projection_pair_crosscheck,
        "projection pair crosscheck does not match independent verification",
    )
    checks = require_dict(
        projection["checks"], "historical checks must be a non-empty bool map"
    )
    target = load_historical_target(arm, device)
    target_metrics = target["pose_metrics"]
    assert isinstance(target_metrics, list)
    reproduced = all(
        _metric_matches(observed, expected)
        for observed, expected in zip(historical_metrics, target_metrics, strict=True)
        if isinstance(expected, dict)
    ) and len(historical_metrics) == len(target_metrics)
    expected_checks = {
        "reference_report_sha256_exact": True,
        "reference_contract_sha256_exact": True,
        "pose_metric_fingerprint_within_1e_6": reproduced,
    }
    require(
        checks == expected_checks,
        "historical check map does not match independent verification",
    )
    candidate_checks = _runtime_candidate_checks(candidate_metrics, device)
    require(
        projection.get("runtime_candidate_checks") == candidate_checks,
        "runtime candidate checks do not match independent verification",
    )
    runtime_candidate_passed = all(candidate_checks.values())
    require(
        type(projection["passed"]) is bool
        and type(summary.get("matches_historical_reference")) is bool
        and type(projection["runtime_candidate_passed"]) is bool
        and projection["progression_allowed"] is False,
        "historical decisions must preserve diagnostic-only semantics",
    )
    require(
        projection["passed"] is reproduced
        and summary["matches_historical_reference"] is reproduced,
        "historical reproduction flags do not match exact target comparison",
    )
    require(
        projection["runtime_candidate_passed"] is runtime_candidate_passed,
        "runtime candidate decision mismatch",
    )
    return reproduced, runtime_candidate_passed


def _max_consecutive_steps(steps: list[int]) -> int:
    longest = 0
    current = 0
    prior = None
    for step in steps:
        current = current + 1 if prior is not None and step == prior + 1 else 1
        longest = max(longest, current)
        prior = step
    return longest


def _validate_physics_rows(report: dict[str, Any]) -> dict[str, Any]:
    timing = require_dict(
        report.get("telemetry_timing"), "telemetry timing is required"
    )
    physics_dt = finite_number(timing.get("physics_dt_s"), "physics dt")
    require(math.isclose(physics_dt, 0.005, abs_tol=1e-12), "physics dt changed")
    require(
        timing.get("control_decimation") == 4
        and timing.get("history_order") == "newest_to_oldest",
        "history timing contract changed",
    )
    require(
        timing.get("peak_window_radius_physics_steps") == 8,
        "peak window radius changed",
    )
    rows = require_list(
        report.get("physics_substep_telemetry"),
        "exactly 600 physics rows are required",
    )
    require(
        len(rows) == 600,
        "exactly 600 physics rows are required",
    )
    base_impulses: dict[int, float] = {}
    base_forces: dict[int, float] = {}
    any_contact_steps: list[int] = []
    base_contact_steps: list[int] = []
    for expected_step, row in enumerate(rows, 1):
        require(isinstance(row, dict), "physics row must be an object")
        assert isinstance(row, dict)
        require(
            row.get("physics_step") == expected_step,
            "physics steps must be contiguous 1..600",
        )
        expected_control = (expected_step + 3) // 4
        expected_slot = expected_control * 4 - expected_step
        require(
            row.get("control_step") == expected_control
            and row.get("contact_force_history_slot") == expected_slot,
            "physics history slot mapping mismatch",
        )
        require(
            row.get("history_slot_order") == "newest_first",
            "physics history order mismatch",
        )
        base_forces[expected_step] = finite_number(
            row.get("base_force_bodyweights"), "base force BW"
        )
        base_impulses[expected_step] = finite_number(
            row.get("base_impulse_n_s"), "base impulse"
        )
        base_force_magnitude = finite_number(
            row.get("base_force_magnitude_n"), "base force magnitude"
        )
        foot_force = finite_number(row.get("foot_total_force_n"), "foot force")
        nonfoot_force = finite_number(
            row.get("nonfoot_total_force_n"), "non-foot force"
        )
        require(
            base_forces[expected_step] >= 0.0 and base_impulses[expected_step] >= 0.0,
            "force/impulse must be nonnegative",
        )
        require(
            math.isclose(
                base_impulses[expected_step],
                base_forces[expected_step]
                * finite_number(
                    require_dict(
                        report.get("runtime_topology"), "runtime topology is required"
                    ).get("body_weight_n"),
                    "body weight",
                )
                * physics_dt,
                rel_tol=2e-5,
                abs_tol=1e-7,
            ),
            "base force/impulse derivation mismatch",
        )
        if foot_force + nonfoot_force >= CONTACT_EXERCISE_THRESHOLD_N:
            any_contact_steps.append(expected_step)
        if base_force_magnitude >= CONTACT_EXERCISE_THRESHOLD_N:
            base_contact_steps.append(expected_step)
    peak_step = max(base_forces, key=base_forces.__getitem__)
    require(9 <= peak_step <= 592, "peak must have a complete 17-step window")
    window_sum = sum(
        base_impulses[step] for step in range(peak_step - 8, peak_step + 9)
    )
    require(window_sum > 0.0, "peak impulse window must be positive")
    require(bool(any_contact_steps), "at least one contact step is required")
    require(bool(base_contact_steps), "at least one base contact step is required")
    threshold_exposure: dict[str, dict[str, float | int]] = {}
    for threshold in (5, 10, 15):
        exceeded = [
            step for step, force in base_forces.items() if force > float(threshold)
        ]
        threshold_exposure[f"over_{threshold}_bodyweights"] = {
            "step_count": len(exceeded),
            "duration_s": len(exceeded) * physics_dt,
            "max_consecutive_steps": _max_consecutive_steps(exceeded),
            "max_consecutive_duration_s": _max_consecutive_steps(exceeded) * physics_dt,
            "integrated_base_impulse_n_s": sum(
                base_impulses[step] for step in exceeded
            ),
        }
    first_any = any_contact_steps[0]
    first_base = base_contact_steps[0]
    return {
        "peak_base_force_bodyweights": base_forces[peak_step],
        "peak_base_force_physics_step": peak_step,
        "peak_base_force_time_s": peak_step * physics_dt,
        "peak_base_impulse_n_s": base_impulses[peak_step],
        "window_base_impulse_n_s": window_sum,
        "concentration_index": base_impulses[peak_step] / window_sum,
        "contact_exposure": {
            "thresholds": threshold_exposure,
            "first_any_contact_physics_step": first_any,
            "first_any_contact_time_s": first_any * physics_dt,
            "first_base_contact_physics_step": first_base,
            "first_base_contact_time_s": first_base * physics_dt,
            "base_contact_to_peak_physics_steps": peak_step - first_base,
            "base_contact_to_peak_duration_s": (peak_step - first_base) * physics_dt,
        },
        "_base_force_series": [base_forces[step] for step in range(1, 601)],
    }


def _flatten_numbers(value: Any, label: str) -> list[float]:
    require(isinstance(value, list), f"{label} must be a list")
    assert isinstance(value, list)
    result: list[float] = []
    stack = list(reversed(value))
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            result.append(finite_number(item, label))
    require(bool(result), f"{label} must not be empty")
    return result


def _validate_control_rows(
    report: dict[str, Any],
) -> tuple[
    dict[str, list[float]],
    dict[int, float],
    dict[int, float],
    list[dict[str, list[float]]],
]:
    rows = require_list(
        report.get("control_step_telemetry"), "exactly 150 control rows are required"
    )
    require(
        len(rows) == 150,
        "exactly 150 control rows are required",
    )
    traces = {field: [] for field in TRACE_FIELDS}
    root_angular_by_step: dict[int, float] = {}
    joint_speed_by_step: dict[int, float] = {}
    control_series: list[dict[str, list[float]]] = []
    for expected_step, row in enumerate(rows, 1):
        require(
            isinstance(row, dict) and row.get("control_step") == expected_step,
            "control steps must be contiguous 1..150",
        )
        assert isinstance(row, dict)
        root = _flatten_numbers(row.get("root_state_w"), "root state")
        require(len(root) == 13, "root state must have 13 values")
        root_angular_by_step[expected_step] = math.sqrt(
            sum(value * value for value in root[10:13])
        )
        joints = _flatten_numbers(row.get("joint_velocity_rad_s"), "joint velocity")
        joint_speed_by_step[expected_step] = max(abs(value) for value in joints)
        flags = require_dict(
            row.get("termination_flags"), "termination flags are required"
        )
        require(
            flags.get("numeric_invalid") is False
            and flags.get("hard_joint_limit") is False,
            "safety termination occurred",
        )
        for field in TRACE_FIELDS:
            traces[field].extend(_flatten_numbers(row.get(field), field))
        control_series.append(
            {
                field: _flatten_numbers(row.get(field), field)
                for field in DIVERGENCE_TOLERANCES
                if field != "base_force_bodyweights"
            }
        )
    return traces, root_angular_by_step, joint_speed_by_step, control_series


def _validate_predecessor_binding(
    report: dict[str, Any], arm: str, device: str, commit: str, bundle_sha: str
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    requirement = PREDECESSOR_REQUIREMENTS[(arm, device)]
    binding = report.get("predecessor_synthesis")
    if requirement is None:
        require(binding is None, "A.cpu predecessor binding must be null")
        return None, []
    binding = require_dict(binding, "predecessor synthesis binding is required")
    expected_count, expected_next = requirement
    require(
        binding.get("evidence_synthesis_valid") is True
        and binding.get("validated_run_count") == expected_count
        and binding.get("next_group") == expected_next,
        "predecessor synthesis sequence mismatch",
    )
    require(
        binding.get("source_commit") == commit
        and binding.get("source_bundle_sha256") == bundle_sha,
        "predecessor synthesis source binding mismatch",
    )
    path = binding.get("path")
    require(
        isinstance(path, str)
        and path.startswith("reports/runs/")
        and Path(path).parent.as_posix() == "reports/runs",
        "predecessor synthesis path binding mismatch",
    )
    assert isinstance(path, str)
    _hex(binding.get("sha256"), 64, "predecessor synthesis hash")
    producer_binding = raw_probe.validate_predecessor_synthesis(
        REPO_ROOT / path,
        arm=arm,
        device=device,
        source_bundle=report["source_bundle"],
    )
    require(
        isinstance(producer_binding, dict), "producer predecessor validation failed"
    )
    assert isinstance(producer_binding, dict)
    for key, value in binding.items():
        require(
            producer_binding.get(key) == value,
            f"producer predecessor binding mismatch: {key}",
        )
    input_reports = _validate_predecessor_file(
        binding,
        expected_count=expected_count,
        expected_next=expected_next,
        commit=commit,
        bundle_sha=bundle_sha,
    )
    return binding, input_reports


def _validate_predecessor_file(
    binding: dict[str, Any],
    *,
    expected_count: int,
    expected_next: str,
    commit: str,
    bundle_sha: str,
) -> list[dict[str, str]]:
    relative_path = binding["path"]
    assert isinstance(relative_path, str)
    resolved = (REPO_ROOT / relative_path).resolve(strict=True)
    require(
        resolved.parent == (REPO_ROOT / "reports/runs").resolve(),
        "predecessor synthesis must resolve to a direct child of reports/runs",
    )
    raw = resolved.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == binding["sha256"],
        "predecessor synthesis byte hash mismatch",
    )
    value = json.loads(raw.decode("utf-8"))
    require(
        isinstance(value, dict), "predecessor synthesis JSON root must be an object"
    )
    assert isinstance(value, dict)
    run_matrix = require_dict(
        value.get("run_matrix"), "predecessor synthesis run_matrix is required"
    )
    require(
        value.get("evidence_synthesis_valid") is True,
        "predecessor synthesis evidence is invalid",
    )
    require(
        run_matrix.get("validated_run_count") == expected_count
        and value.get("next_group") == expected_next,
        "predecessor synthesis file sequence mismatch",
    )
    require(
        value.get("source_commit") == commit
        and value.get("source_bundle_sha256") == bundle_sha,
        "predecessor synthesis file source binding mismatch",
    )
    input_reports = require_list(
        value.get("input_reports"), "predecessor synthesis input_reports are required"
    )
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(input_reports):
        item = require_dict(item, f"predecessor input report {index} must be an object")
        path = item.get("path")
        sha = item.get("sha256")
        require(
            isinstance(path, str)
            and path.startswith("reports/runs/")
            and Path(path).parent.as_posix() == "reports/runs",
            f"predecessor input report {index} path binding mismatch",
        )
        assert isinstance(path, str)
        normalized.append({"path": path, "sha256": _hex(sha, 64, "raw report hash")})
    require(
        len(normalized) == expected_count,
        "predecessor input report count mismatch",
    )
    return normalized


def validate_raw_report(
    report: dict[str, Any],
    evidence: dict[str, str],
    expected_arm: str,
    expected_device: str,
    expected_replicate: int,
) -> dict[str, Any]:
    raw_probe.validate_report_contract(report)
    require(
        report.get("schema_version") == "g009.r0.rev16.backend_divergence.v1",
        "raw schema mismatch",
    )
    require(
        report.get("goal_id") == "g009"
        and report.get("stage_id") == "R0"
        and report.get("revision") == "rev16",
        "goal/stage/revision mismatch",
    )
    require(
        report.get("status") == "complete"
        and report.get("diagnostic_capture_complete") is True,
        "raw diagnostic is incomplete",
    )
    require(
        report.get("diagnostic_only") is True
        and report.get("qualification_eligible") is False,
        "raw report escaped diagnostic governance",
    )
    require(
        report.get("replicate_index") == expected_replicate,
        "replicate order/index mismatch",
    )
    require(report.get("device") == expected_device, "device/order mismatch")
    require(
        report.get("runtime_device") == expected_device,
        "runtime device readback mismatch",
    )
    require(
        report.get("headless") is True
        and report.get("seed") == 42
        and report.get("task") == raw_probe.DEFAULT_TASK
        and report.get("num_envs") == 8
        and report.get("rollout_steps") == 150,
        "execution conditions changed",
    )
    contract = require_dict(report.get("contract"), "contract is required")
    require(
        contract == raw_probe.rev16_contract(expected_arm, expected_device),
        "rev16 contract content mismatch",
    )
    arm = require_dict(contract.get("arm"), "arm/order mismatch")
    require(arm.get("id") == expected_arm, "arm/order mismatch")
    expected_iterations = 8 if expected_arm == "A" else 16
    require(
        arm.get("articulation_solver_position_iteration_count") == expected_iterations
        and arm.get("articulation_solver_velocity_iteration_count") == 0
        and arm.get("max_depenetration_velocity_m_s") == 1.0,
        "arm physics tuple mismatch",
    )
    require(
        report.get("contract_sha256") == canonical_sha256(contract),
        "contract hash mismatch",
    )
    reference_id, reference_sha, _ = EXPECTED_REFERENCE[expected_arm]
    historical = require_dict(
        contract.get("historical_reference"), "historical lineage mismatch"
    )
    require(
        historical.get("contract_id") == reference_id
        and historical.get("canonical_sha256") == reference_sha,
        "historical lineage mismatch",
    )
    require(
        historical.get("historical_checkpoint_loaded") is False
        and historical.get("historical_training_resumed") is False
        and historical.get("current_rev16_execution_is_fresh") is True,
        "historical execution was resumed",
    )
    governance = require_dict(report.get("governance"), "governance mismatch")
    require(
        governance.get("diagnostic_only") is True
        and governance.get("qualification_eligible") is False
        and governance.get("learned") is False,
        "governance mismatch",
    )
    require(
        governance.get("ppo") == {"allowed": False, "status": "not_run"}
        and governance.get("gate01") == {"allowed": False, "status": "forbidden"}
        and governance.get("gate10") == {"allowed": False, "status": "forbidden"},
        "PPO/Gate governance mismatch",
    )
    cell = require_dict(report.get("controlled_cell"), "controlled cell mismatch")
    require(
        cell.get("source_env_index") == 7
        and cell.get("pose_id") == "right_side"
        and cell.get("action_mode") == "reset_pose_hold"
        and cell.get("target_body_name") == "base",
        "controlled cell mismatch",
    )
    require(
        report.get("safety_termination_counts") == SAFETY_ZERO,
        "safety termination count must be zero",
    )
    live = require_dict(
        report.get("live_physics_readback"), "live physics readback failed"
    )
    live_checks = require_dict(live.get("checks"), "live physics readback failed")
    require(
        bool(live_checks) and all(value is True for value in live_checks.values()),
        "live physics readback failed",
    )
    expected_clock = require_dict(
        report.get("physics_step_clock"), "physics step clock is required"
    )
    require(
        expected_clock.get("callback_count") == 600
        and expected_clock.get("passed") is True,
        "physics step clock must prove exactly 600 pre-step callbacks",
    )
    if expected_device == "cpu":
        authority = require_dict(
            report.get("cpu_contact_authority"), "CPU contact authority incomplete"
        )
        events = require_list(
            authority.get("events"), "CPU contact authority events are required"
        )
        require(
            authority.get("passed") is True
            and authority.get("physics_step_clock") == expected_clock
            and len(events) > 0
            and type(authority.get("callback_event_count")) is int
            and authority["callback_event_count"] >= len(events),
            "CPU contact authority incomplete",
        )
        separations = authority.get("all_env_minimum_separation_m")
        require(
            isinstance(separations, list) and len(separations) == 8,
            "CPU all-env separation summary incomplete",
        )
        physics_steps: list[int] = []
        callback_indices: list[int] = []
        for event in events:
            event = require_dict(event, "CPU contact event must be an object")
            physics_step = event.get("physics_step")
            require(
                type(physics_step) is int
                and 1 <= physics_step <= expected_clock["callback_count"],
                "CPU contact event physics_step is outside 1..600",
            )
            assert isinstance(physics_step, int)
            physics_steps.append(physics_step)
            callback_index = event.get("callback_event_index")
            require(
                type(callback_index) is int
                and 1 <= callback_index <= authority["callback_event_count"]
                and event.get("complete") is True,
                "CPU contact event index/completeness mismatch",
            )
            assert isinstance(callback_index, int)
            callback_indices.append(callback_index)
        require(
            physics_steps == sorted(physics_steps),
            "CPU contact events must be sorted by physics_step",
        )
        require(
            callback_indices == sorted(set(callback_indices)),
            "CPU contact callback indices must be unique and increasing",
        )
    else:
        authority = require_dict(
            report.get("cpu_contact_authority"),
            "GPU fabricated CPU contact authority",
        )
        require(
            authority.get("passed") is None
            and authority.get("events") is None
            and authority.get("all_env_minimum_separation_m") is None
            and authority.get("physics_step_clock") == expected_clock,
            "GPU fabricated CPU contact authority",
        )
    execution_id = _validate_execution(report, evidence)
    commit, source_digest = _validate_source_bundle(report)
    predecessor, predecessor_input_reports = _validate_predecessor_binding(
        report, expected_arm, expected_device, commit, source_digest
    )
    historical_reproduced, runtime_candidate_passed = _validate_historical_summary(
        report, expected_arm, expected_device
    )
    force = _validate_physics_rows(report)
    base_force_series = force.pop("_base_force_series")
    traces, root_angular_by_step, joint_speed_by_step, control_series = (
        _validate_control_rows(report)
    )
    peak_step = int(force["peak_base_force_physics_step"])
    first_control = (peak_step - 8 + 3) // 4
    last_control = (peak_step + 8 + 3) // 4
    window_controls = range(first_control, last_control + 1)
    return {
        "evidence": evidence,
        "execution_id": execution_id,
        "source_commit": commit,
        "source_bundle_sha256": source_digest,
        "contract_sha256": report["contract_sha256"],
        "arm": expected_arm,
        "device": expected_device,
        "replicate_index": expected_replicate,
        "historical_reproduction_passed": historical_reproduced,
        "runtime_candidate_passed": runtime_candidate_passed,
        "predecessor_synthesis": predecessor,
        "predecessor_input_reports": predecessor_input_reports,
        **force,
        "peak_window_control_step_first": first_control,
        "peak_window_control_step_last": last_control,
        "peak_window_max_root_angular_speed_rad_s": max(
            root_angular_by_step[step] for step in window_controls
        ),
        "peak_window_max_joint_speed_rad_s": max(
            joint_speed_by_step[step] for step in window_controls
        ),
        "traces": traces,
        "physics_base_force_series": base_force_series,
        "control_series": control_series,
    }


def _trace_max_error(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in TRACE_FIELDS:
        a = left["traces"][field]
        b = right["traces"][field]
        require(len(a) == len(b), f"{field} trace length mismatch")
        result[field] = max(abs(x - y) for x, y in zip(a, b, strict=True))
    return result


def _max_abs_delta(left: list[float], right: list[float], label: str) -> float:
    require(len(left) == len(right), f"{label} series shape mismatch")
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def _first_divergence(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_physics = left["physics_base_force_series"]
    right_physics = right["physics_base_force_series"]
    require(
        len(left_physics) == len(right_physics) == 600, "physics series length mismatch"
    )
    physics_divergence = None
    physics_tolerance = DIVERGENCE_TOLERANCES["base_force_bodyweights"]
    for index, (left_value, right_value) in enumerate(
        zip(left_physics, right_physics, strict=True), 1
    ):
        delta = abs(left_value - right_value)
        if delta > physics_tolerance:
            physics_divergence = {
                "step": index,
                "time_s": index * 0.005,
                "variable": "base_force_bodyweights",
                "max_abs_delta": delta,
                "tolerance": physics_tolerance,
            }
            break

    left_control = left["control_series"]
    right_control = right["control_series"]
    require(
        len(left_control) == len(right_control) == 150, "control series length mismatch"
    )
    control_divergence = None
    for index, (left_step, right_step) in enumerate(
        zip(left_control, right_control, strict=True), 1
    ):
        candidates = []
        for variable, tolerance in DIVERGENCE_TOLERANCES.items():
            if variable == "base_force_bodyweights":
                continue
            delta = _max_abs_delta(left_step[variable], right_step[variable], variable)
            if delta > tolerance:
                candidates.append((delta, variable, tolerance))
        if candidates:
            delta, variable, tolerance = max(candidates, key=lambda item: item[0])
            control_divergence = {
                "step": index,
                "time_s": index * 0.02,
                "variable": variable,
                "max_abs_delta": delta,
                "tolerance": tolerance,
            }
            break

    overall = None
    if physics_divergence is not None or control_divergence is not None:
        if control_divergence is None or (
            physics_divergence is not None
            and physics_divergence["time_s"] <= control_divergence["time_s"]
        ):
            assert physics_divergence is not None
            overall = {"domain": "physics", **physics_divergence}
        else:
            overall = {"domain": "control", **control_divergence}
    return {
        "tolerances": dict(DIVERGENCE_TOLERANCES),
        "first_physics_divergence": physics_divergence,
        "first_control_divergence": control_divergence,
        "first_overall_divergence": overall,
    }


def _group_passed(group_index: int, runs: list[dict[str, Any]]) -> bool:
    expected = group_index < 3
    return all(
        run["historical_reproduction_passed"] is True
        and run["runtime_candidate_passed"] is expected
        for run in runs
    )


def synthesize_loaded(
    entries: list[tuple[dict[str, Any], dict[str, str]]],
) -> dict[str, Any]:
    require(
        len(entries) in ALLOWED_INPUT_COUNTS,
        "input count must be exactly 3, 6, 9, or 12",
    )
    validated: list[dict[str, Any]] = []
    for index, (report, evidence) in enumerate(entries):
        group_index = index // GROUP_SIZE
        replicate = index % GROUP_SIZE + 1
        arm, device = GROUP_ORDER[group_index]
        validated.append(validate_raw_report(report, evidence, arm, device, replicate))
    execution_ids = [run["execution_id"] for run in validated]
    require(
        len(set(execution_ids)) == len(execution_ids), "execution IDs must be unique"
    )
    evidences = [run["evidence"]["sha256"] for run in validated]
    require(len(set(evidences)) == len(evidences), "raw report hashes must be unique")
    source_bindings = {
        (run["source_commit"], run["source_bundle_sha256"]) for run in validated
    }
    require(len(source_bindings) == 1, "all raw reports must share one source binding")

    groups: list[dict[str, Any]] = []
    complete_groups = len(validated) // GROUP_SIZE
    prior_passed = True
    for group_index in range(complete_groups):
        arm, device = GROUP_ORDER[group_index]
        runs = validated[group_index * 3 : group_index * 3 + 3]
        group_passed = _group_passed(group_index, runs)
        historical_reproduction = all(
            run["historical_reproduction_passed"] is True for run in runs
        )
        expected_candidate = group_index < 3
        runtime_candidate_expected = all(
            run["runtime_candidate_passed"] is expected_candidate for run in runs
        )
        predecessor_bindings = [
            canonical_sha256(run["predecessor_synthesis"])
            for run in runs
            if run["predecessor_synthesis"] is not None
        ]
        require(
            not predecessor_bindings or len(set(predecessor_bindings)) == 1,
            "replicates in one group must share one predecessor synthesis binding",
        )
        expected_prefix = [
            run["evidence"] for run in validated[: group_index * GROUP_SIZE]
        ]
        for run in runs:
            require(
                run["predecessor_input_reports"] == expected_prefix,
                "predecessor input_reports do not exactly bind the validated prefix",
            )
        if group_index > 0:
            require(prior_passed, "a later group exists after an earlier group failed")
        groups.append(
            {
                "sequence_index": group_index + 1,
                "arm": arm,
                "device": device,
                "replicate_count": 3,
                "historical_reproduction_3_of_3": historical_reproduction,
                "runtime_candidate_expected_3_of_3": runtime_candidate_expected,
                "sequence_gate_passed": group_passed,
                "progression_allowed": group_passed and group_index < 3,
                "runs": [
                    {
                        key: value
                        for key, value in run.items()
                        if key
                        not in {
                            "traces",
                            "physics_base_force_series",
                            "control_series",
                        }
                    }
                    for run in runs
                ],
            }
        )
        prior_passed = group_passed

    hypothesis_replicates: list[dict[str, Any]] = []
    hypothesis_supported = None
    if complete_groups == 4:
        for replicate_index in range(3):
            a_cpu = validated[replicate_index]
            a_gpu = validated[3 + replicate_index]
            b_cpu = validated[6 + replicate_index]
            b_gpu = validated[9 + replicate_index]
            named_runs = {
                "a_cpu": a_cpu,
                "a_gpu": a_gpu,
                "b_cpu": b_cpu,
                "b_gpu": b_gpu,
            }
            trace_pairs = {
                f"{left_name}_vs_{right_name}": _trace_max_error(left, right)
                for (left_name, left), (right_name, right) in itertools.combinations(
                    named_runs.items(), 2
                )
            }
            divergence_pairs = {
                f"{left_name}_vs_{right_name}": _first_divergence(left, right)
                for (left_name, left), (right_name, right) in itertools.combinations(
                    named_runs.items(), 2
                )
            }
            trace_max = max(
                value for pair in trace_pairs.values() for value in pair.values()
            )
            checks = {
                "b_gpu_force_exceeds_15_bodyweights": b_gpu[
                    "peak_base_force_bodyweights"
                ]
                > FORCE_THRESHOLD_BODYWEIGHTS,
                "b_gpu_peak_at_least_one_substep_earlier_than_b_cpu": b_gpu[
                    "peak_base_force_physics_step"
                ]
                <= b_cpu["peak_base_force_physics_step"] - 1,
                "b_gpu_peak_at_least_one_substep_earlier_than_a_gpu": b_gpu[
                    "peak_base_force_physics_step"
                ]
                <= a_gpu["peak_base_force_physics_step"] - 1,
                "b_gpu_concentration_over_b_cpu_at_least_1_20": b_gpu[
                    "concentration_index"
                ]
                / b_cpu["concentration_index"]
                >= CONCENTRATION_RATIO_THRESHOLD,
                "b_gpu_concentration_exceeds_a_gpu": b_gpu["concentration_index"]
                > a_gpu["concentration_index"],
                "action_and_ema_trace_error_at_most_1e_6": trace_max <= TRACE_TOLERANCE,
                "b_gpu_peak_window_root_angular_speed_exceeds_b_cpu_and_a_gpu": b_gpu[
                    "peak_window_max_root_angular_speed_rad_s"
                ]
                > max(
                    b_cpu["peak_window_max_root_angular_speed_rad_s"],
                    a_gpu["peak_window_max_root_angular_speed_rad_s"],
                ),
                "b_gpu_peak_window_joint_speed_exceeds_b_cpu_and_a_gpu": b_gpu[
                    "peak_window_max_joint_speed_rad_s"
                ]
                > max(
                    b_cpu["peak_window_max_joint_speed_rad_s"],
                    a_gpu["peak_window_max_joint_speed_rad_s"],
                ),
                "safety_termination_zero": True,
            }
            hypothesis_replicates.append(
                {
                    "replicate_index": replicate_index + 1,
                    "checks": checks,
                    "passed": all(checks.values()),
                    "derived": {
                        "b_gpu_over_b_cpu_concentration_ratio": b_gpu[
                            "concentration_index"
                        ]
                        / b_cpu["concentration_index"],
                        "max_action_ema_trace_abs_error": trace_max,
                        "trace_pair_errors": trace_pairs,
                        "first_divergence_pairs": divergence_pairs,
                    },
                }
            )
        hypothesis_supported = all(item["passed"] for item in hypothesis_replicates)
        groups[-1]["hypothesis_supported"] = hypothesis_supported
        groups[-1]["progression_allowed"] = False

    decision = (
        "supported"
        if hypothesis_supported is True
        else "inconclusive"
        if hypothesis_supported is False
        else "pending_sequential_groups"
    )
    source_commit, source_bundle_sha256 = next(iter(source_bindings))
    next_group = (
        f"{GROUP_ORDER[complete_groups][0]}.{GROUP_ORDER[complete_groups][1]}"
        if complete_groups < len(GROUP_ORDER) and prior_passed
        else None
    )
    return {
        "schema_version": "g009.r0.rev16.backend_divergence_synthesis.v1",
        "goal_id": "g009",
        "stage_id": "R0",
        "revision": "rev16",
        "status": "complete",
        "evidence_synthesis_valid": True,
        "input_report_count": len(validated),
        "input_reports": [run["evidence"] for run in validated],
        "source_commit": source_commit,
        "source_bundle_sha256": source_bundle_sha256,
        "run_matrix": {
            "validated_run_count": len(validated),
            "validated_group_count": complete_groups,
        },
        "next_group": next_group,
        "required_sequence": [f"{arm}.{device}" for arm, device in GROUP_ORDER],
        "completed_group_count": complete_groups,
        "groups": groups,
        "hypothesis": {
            "decision": decision,
            "supported_3_of_3": hypothesis_supported,
            "replicates": hypothesis_replicates,
        },
        "governance": {
            "position16_accepted": False,
            "position16_status": "rejected_even_if_hypothesis_supported",
            "diagnostic_only": True,
            "learned": False,
            "ppo": {"allowed": False, "status": "not_run"},
            "gate01": {"allowed": False, "status": "forbidden"},
            "gate10": {"allowed": False, "status": "forbidden"},
            "qualification": {"eligible": False, "status": "not_run", "passed": None},
        },
    }


def synthesize(paths: list[Path]) -> dict[str, Any]:
    return synthesize_loaded([read_json(path) for path in paths])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    require(
        output.parent == RUNS_DIR.resolve(),
        "output must be a direct child of reports/runs",
    )
    require(not output.exists(), f"refusing to overwrite output: {output}")
    report = synthesize(args.reports)
    report["created_at_utc"] = (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    report["synthesis_execution_id"] = uuid.uuid4().hex
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    print(
        json.dumps(
            {"output": str(output), "hypothesis": report["hypothesis"]["decision"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
