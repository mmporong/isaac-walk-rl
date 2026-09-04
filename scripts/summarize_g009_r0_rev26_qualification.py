#!/usr/bin/env python3
"""Bind the preregistered rev26 training and held-out evaluation evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
import sys

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluate_g009_r0 import (  # noqa: E402
    DEFAULT_TASK,
    EXPECTED_ACTOR_OBSERVATION_DIM,
    EXPECTED_CRITIC_OBSERVATION_DIM,
    EXPECTED_EPISODES_PER_POSE,
    EXPECTED_TRAINING_CHECKPOINT,
    MAX_RAW_HARD_JOINT_LIMIT_VIOLATION_RAD,
    OFFICIAL_PROTOCOL,
    POSE_NAMES,
    _load_qualification_config,
    _read_json,
    file_sha256,
    git_source_state,
    portable_path,
    summarize_samples,
    validate_training_report,
)


SCHEMA_VERSION = "g009.r0.rev26.qualification_synthesis.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_evaluation_report(
    report: Mapping[str, Any], *, checkpoint_sha256: str, training_binding: Mapping[str, Any]
) -> dict[str, bool]:
    poses = report.get("poses")
    pose_list = poses if isinstance(poses, list) else []
    pose_order = tuple(
        item.get("pose_id") if isinstance(item, Mapping) else None for item in pose_list
    )
    pose_structure_valid = len(pose_list) == 4 and pose_order == POSE_NAMES and len(set(pose_order)) == 4
    pose_gates: dict[str, bool] = {}
    pose_report_consistency: dict[str, bool] = {}
    aggregate_episodes = 0
    aggregate_successes = 0
    aggregate_safety = 0
    aggregate_other = 0
    for index, pose in enumerate(POSE_NAMES):
        item = pose_list[index] if pose_structure_valid else {}
        recovery = item.get("recovery_time_s", {}) if isinstance(item, Mapping) else {}
        terminations = item.get("termination_counts", {}) if isinstance(item, Mapping) else {}
        raw_recovery = item.get("recovery_time_samples_s", []) if isinstance(item, Mapping) else []
        counts: list[int] = []
        for name in ("stable_success", "time_out", "numeric_invalid", "hard_joint_limit", "other"):
            value = terminations.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                break
            counts.append(value)
        counts_valid = len(counts) == 5
        episodes_value = item.get("episode_count")
        episodes: int
        if (
            isinstance(episodes_value, bool)
            or not isinstance(episodes_value, int)
            or episodes_value < 0
        ):
            episodes = -1
            episodes_valid = False
        else:
            episodes = episodes_value
            episodes_valid = True
        successes = counts[0] if counts_valid else -1
        recovery_values: list[float] = []
        recovery_values_valid = isinstance(raw_recovery, list)
        if recovery_values_valid:
            for value in raw_recovery:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    recovery_values_valid = False
                    break
                numeric_value = float(value)
                if not math.isfinite(numeric_value) or numeric_value < 0.0:
                    recovery_values_valid = False
                    break
                recovery_values.append(numeric_value)
        recomputed_recovery = summarize_samples(recovery_values) if recovery_values_valid else {}
        recomputed_median = recomputed_recovery.get("median")
        recomputed_rate = successes / episodes if episodes_valid and episodes > 0 else -1.0
        safety = counts[2] + counts[3] if counts_valid else -1
        other = counts[4] if counts_valid else -1
        raw_limit_value = item.get("max_raw_hard_joint_limit_violation_rad")
        raw_limit = (
            float(raw_limit_value)
            if not isinstance(raw_limit_value, bool) and isinstance(raw_limit_value, (int, float))
            else None
        )
        pose_gates[pose] = bool(
            pose_structure_valid
            and episodes_valid
            and episodes == EXPECTED_EPISODES_PER_POSE
            and counts_valid
            and sum(counts) == episodes
            and successes >= 205
            and recovery_values_valid
            and len(recovery_values) == successes
            and recomputed_median is not None
            and recomputed_median <= 4.0
            and safety == 0
            and other == 0
            and raw_limit is not None
            and math.isfinite(raw_limit)
            and raw_limit <= MAX_RAW_HARD_JOINT_LIMIT_VIOLATION_RAD
        )
        pose_report_consistency[pose] = bool(
            pose_gates[pose]
            and item.get("success_count") == successes
            and item.get("success_rate") == recomputed_rate
            and recovery == recomputed_recovery
            and item.get("gate_pass") is True
        )
        if pose_gates[pose]:
            aggregate_episodes += episodes
            aggregate_successes += successes
            aggregate_safety += safety
            aggregate_other += other
    aggregate = report.get("aggregate", {})
    recomputed_aggregate = {
        "episode_count": aggregate_episodes,
        "success_count": aggregate_successes,
        "success_rate": aggregate_successes / aggregate_episodes if aggregate_episodes else 0.0,
        "safety_termination_count": aggregate_safety,
        "other_termination_count": aggregate_other,
        "all_pose_gate_pass": all(pose_gates.values()),
    }
    physics = report.get("physics_readback", {})
    source_state = report.get("source_state", {})
    source_before = source_state.get("before", {}) if isinstance(source_state, Mapping) else {}
    source_after = source_state.get("after", {}) if isinstance(source_state, Mapping) else {}
    source_bindings = report.get("source_bindings", {})
    expected_source_bindings = {
        "evaluator": {
            "path": "scripts/evaluate_g009_r0.py",
            "sha256": file_sha256(REPO_ROOT / "scripts" / "evaluate_g009_r0.py"),
        },
        "config": {
            "path": "configs/g009_r0.json",
            "sha256": file_sha256(REPO_ROOT / "configs" / "g009_r0.json"),
        },
    }
    training_commit = training_binding.get("repository", {}).get("commit")
    checks = {
        "pose_structure": pose_structure_valid,
        "reported_status_consistent": report.get("status") == (
            "pass" if all(pose_gates.values()) else "fail"
        ),
        "official_protocol": (
            report.get("protocol_mode") == "official_qualification"
            and report.get("official_protocol") == OFFICIAL_PROTOCOL
        ),
        "task_seed_budget": (
            report.get("task") == DEFAULT_TASK
            and report.get("seed") == 1042
            and report.get("num_envs") == 1024
            and report.get("episodes_per_pose") == EXPECTED_EPISODES_PER_POSE
        ),
        "actor_corruption_enabled": report.get("observation_corruption") is True,
        "checkpoint_bound": report.get("checkpoint", {}).get("sha256") == checkpoint_sha256,
        "training_bound": report.get("training_binding") == training_binding,
        "evaluation_source_stable": (
            source_before == source_after
            and source_after.get("clean") is True
            and source_after.get("commit") == training_commit
        ),
        "evaluation_sources_bound": source_bindings == expected_source_bindings,
        "aggregate_recomputed": aggregate == recomputed_aggregate,
        "exact_aggregate_denominator": aggregate_episodes == 1024,
        "aggregate_safety_zero": aggregate_safety == 0,
        "aggregate_other_zero": aggregate_other == 0,
        "all_pose_gate": all(pose_gates.values()),
        "policy_observation_dimension": physics.get("policy_observation_dim") == EXPECTED_ACTOR_OBSERVATION_DIM,
        "action_dimension": physics.get("action_dim") == 12,
    }
    return {
        **checks,
        **{f"pose_{pose}": passed for pose, passed in pose_gates.items()},
        **{
            f"pose_{pose}_reported_consistent": passed
            for pose, passed in pose_report_consistency.items()
        },
    }


def build_summary(
    preregistration_path: Path,
    training_report_path: Path,
    checkpoint_path: Path,
    evaluation_report_path: Path,
) -> dict[str, Any]:
    preregistration = _load_qualification_config(preregistration_path)
    _require(checkpoint_path.name == EXPECTED_TRAINING_CHECKPOINT, "checkpoint must be model_299.pt")
    training_binding = validate_training_report(training_report_path, checkpoint_path)
    evaluation = _read_json(evaluation_report_path)
    evaluation_checks = validate_evaluation_report(
        evaluation,
        checkpoint_sha256=file_sha256(checkpoint_path),
        training_binding=training_binding,
    )
    source_state = git_source_state()
    gates = {
        "preregistration_valid": preregistration.get("task") == DEFAULT_TASK,
        "training_valid": True,
        "checkpoint_actor_dimension": training_binding["checkpoint_observation_dimensions"]["actor"]
        == EXPECTED_ACTOR_OBSERVATION_DIM,
        "checkpoint_critic_dimension": training_binding["checkpoint_observation_dimensions"]["critic"]
        == EXPECTED_CRITIC_OBSERVATION_DIM,
        "source_commit_stable": (
            source_state.get("clean") is True
            and source_state.get("commit") == training_binding["repository"]["commit"]
        ),
        **evaluation_checks,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": preregistration.get("evidence_id"),
        "revision": "rev26",
        "status": "pass" if all(gates.values()) else "fail",
        "task": DEFAULT_TASK,
        "gates": gates,
        "bindings": {
            "preregistration": {
                "path": portable_path(preregistration_path),
                "sha256": file_sha256(preregistration_path),
                "source_binding_path_manifest_sha256": preregistration[
                    "source_binding_path_manifest_sha256"
                ],
            },
            "training_report": {
                "path": portable_path(training_report_path),
                "sha256": file_sha256(training_report_path),
            },
            "checkpoint": {
                "path": portable_path(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
                "name": checkpoint_path.name,
                "actor_observation_dimension": EXPECTED_ACTOR_OBSERVATION_DIM,
                "critic_observation_dimension": EXPECTED_CRITIC_OBSERVATION_DIM,
            },
            "evaluation_report": {
                "path": portable_path(evaluation_report_path),
                "sha256": file_sha256(evaluation_report_path),
            },
            "repository": training_binding["repository"],
            "source_bundle": training_binding["source_bundle"],
        },
        "training": {
            "seed": 42,
            "num_envs": 1024,
            "num_steps_per_env": 24,
            "max_iterations": 300,
            "optimizer_mini_batch_updates": 6000,
            "scratch": True,
            "headless": True,
            "hydra_overrides": [],
        },
        "evaluation": {
            "seed": 1042,
            "num_envs": 1024,
            "environments_per_pose": EXPECTED_EPISODES_PER_POSE,
            "minimum_successes_per_pose": 205,
            "maximum_median_recovery_time_seconds": 4.0,
            "actor_corruption_enabled": True,
            "aggregate": evaluation.get("aggregate"),
            "poses": evaluation.get("poses"),
        },
        "claim_limits": {
            "scope": "flat nominal-friction simulation R0 seed42 training with held-out seed1042 evaluation",
            "training_alone_qualifies_recovery": False,
            "slope_recovery": False,
            "asymmetric_friction_recovery": False,
            "disturbance_recovery": False,
            "real_robot_or_sim_to_real": False,
        },
    }


def write_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(
        args.preregistration.resolve(),
        args.training_report.resolve(),
        args.checkpoint.resolve(),
        args.evaluation_report.resolve(),
    )
    write_json_no_overwrite(args.output.resolve(), summary)
    print(json.dumps({"status": summary["status"], "output": str(args.output.resolve())}))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
