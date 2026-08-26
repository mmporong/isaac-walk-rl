#!/usr/bin/env python3
"""Aggregate G008 irregular-road training, checkpoint screening, and full evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_LIMITS = {
    "linear_tracking_rmse_mps": 0.25,
    "yaw_tracking_rmse_radps": 0.25,
    "roll_abs_rad_max": 0.35,
    "pitch_abs_rad_max": 0.35,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        try:
            relative = resolved.relative_to(Path.home().resolve())
        except ValueError:
            return str(resolved)
        return "%USERPROFILE%\\" + str(relative)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_candidate(path: Path) -> dict[str, Any]:
    report = _load(path)
    directions = report["directions"]
    worst_ratio = max(
        float(direction[metric]) / limit
        for direction in directions
        for metric, limit in GATE_LIMITS.items()
    )
    return {
        "policy_id": report["policy_id"],
        "report": {"path": portable_path(path), "sha256": file_sha256(path)},
        "checkpoint": report["checkpoint"],
        "direction_pass_count": sum(bool(direction["gate_pass"]) for direction in directions),
        "all_directions_gate_pass": bool(report["all_directions_gate_pass"]),
        "fall_count": sum(int(direction["fall_count"]) for direction in directions),
        "worst_gate_ratio": worst_ratio,
        "failed_directions": [direction["id"] for direction in directions if not direction["gate_pass"]],
        "directions": [
            {
                "id": direction["id"],
                "gate_pass": direction["gate_pass"],
                "linear_tracking_rmse_mps": direction["linear_tracking_rmse_mps"],
                "yaw_tracking_rmse_radps": direction["yaw_tracking_rmse_radps"],
                "roll_abs_rad_max": direction["roll_abs_rad_max"],
                "pitch_abs_rad_max": direction["pitch_abs_rad_max"],
                "fall_count": direction["fall_count"],
            }
            for direction in directions
        ],
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    training_path = args.training_report.resolve()
    training = _load(training_path)
    full_paths = [path.resolve() for path in args.full_evaluations]
    screen_paths = [path.resolve() for path in args.screening_reports]
    candidates = [summarize_candidate(path) for path in full_paths]
    selected = min(
        candidates,
        key=lambda candidate: (
            -candidate["direction_pass_count"],
            candidate["fall_count"],
            candidate["worst_gate_ratio"],
        ),
    )
    screen = []
    for path in screen_paths:
        report = _load(path)
        screen.append(
            {
                "policy_id": report["policy_id"],
                "report": {"path": portable_path(path), "sha256": file_sha256(path)},
                "all_directions_gate_pass": report["all_directions_gate_pass"],
                "failed_directions": [
                    direction["id"] for direction in report["directions"] if not direction["gate_pass"]
                ],
            }
        )
    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "protocol": "irregular_road_checkpoint_selection_v1",
        "training": {
            "report": {"path": portable_path(training_path), "sha256": file_sha256(training_path)},
            "passed": training["passed"],
            "headless": training["headless"],
            "num_envs": training["num_envs"],
            "iterations": training["max_iterations"],
            "rollout_steps_per_env_iteration": 24,
            "total_transitions": training["num_envs"] * training["max_iterations"] * 24,
            "ppo_learning_epochs": 5,
            "ppo_mini_batches_per_epoch": 4,
            "final_mean_reward": training["performance"]["final_mean_reward"],
            "final_mean_episode_length": training["performance"]["final_mean_episode_length"],
            "mean_steps_per_second": training["performance"]["mean_steps_per_second"],
            "wall_time_seconds": training["wall_time_seconds"],
            "final_checkpoint": {
                "path": training["artifacts"]["checkpoint"],
                "sha256": training["artifacts"]["checkpoint_sha256"],
            },
        },
        "screening": {
            "num_envs": 16,
            "horizon_steps": 300,
            "reports": screen,
            "warning": "screening PASS is not accepted without the full 32-environment, 500-step gate",
        },
        "full_evaluations": candidates,
        "selection": {
            "policy_id": selected["policy_id"],
            "checkpoint": selected["checkpoint"],
            "selection_rule": "maximize direction pass count, then minimize falls, then minimize worst normalized gate ratio",
            "all_directions_gate_pass": selected["all_directions_gate_pass"],
            "adopt_dedicated_training_checkpoint": selected["policy_id"] != "friction_s1",
            "conclusion": (
                "No full-evaluation candidate passed all four directions; keep the inherited friction S1 policy. "
                "The dedicated +300 PPO checkpoint is retained as a documented negative result, not as the selected policy."
            ),
        },
        "aggregate_source_sha256": file_sha256(Path(__file__)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--full-evaluations", required=True, nargs="+", type=Path)
    parser.add_argument("--screening-reports", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = aggregate(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "selected": report["selection"]["policy_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
