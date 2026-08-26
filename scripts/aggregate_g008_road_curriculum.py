#!/usr/bin/env python3
"""Aggregate G008 G0 geometry and turn-aware reward checkpoint evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
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


def _report_ref(path: Path) -> dict[str, str]:
    return {"path": portable_path(path), "sha256": file_sha256(path)}


def _direction_worst_ratio(direction: dict[str, Any]) -> float:
    return max(float(direction[metric]) / limit for metric, limit in GATE_LIMITS.items())


def _policy_label(policy_id: str) -> str:
    return re.sub(r"_t20\d{6}$", "", policy_id)


def summarize_full_evaluations(paths: list[Path], expected_seeds: set[int]) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in paths:
        report = _load(path)
        if report.get("status") != "complete" or report.get("num_envs") != 32 or report.get("horizon_steps") != 500:
            raise ValueError(f"not a completed full evaluation: {path}")
        groups[report["checkpoint"]["sha256"]].append((path, report))

    candidates = []
    for checkpoint_sha, items in groups.items():
        items.sort(key=lambda item: int(item[1]["terrain_seed"]))
        seeds = {int(item[1]["terrain_seed"]) for item in items}
        if seeds != expected_seeds:
            raise ValueError(f"terrain seed coverage mismatch for {checkpoint_sha}: {sorted(seeds)}")
        directions = [direction for _, report in items for direction in report["directions"]]
        candidate = {
            "policy_id": _policy_label(items[0][1]["policy_id"]),
            "task": items[0][1]["task"],
            "checkpoint": items[0][1]["checkpoint"],
            "terrain_seeds": sorted(seeds),
            "terrain_seed_pass_count": sum(bool(report["all_directions_gate_pass"]) for _, report in items),
            "direction_pass_count": sum(bool(direction["gate_pass"]) for direction in directions),
            "direction_total": len(directions),
            "fall_count": sum(int(direction["fall_count"]) for direction in directions),
            "worst_gate_ratio": max(_direction_worst_ratio(direction) for direction in directions),
            "qualified_for_next_friction_stage": all(
                bool(report["all_directions_gate_pass"]) for _, report in items
            ),
            "reports": [
                {
                    "terrain_seed": int(report["terrain_seed"]),
                    "all_directions_gate_pass": bool(report["all_directions_gate_pass"]),
                    "failed_directions": [
                        direction["id"] for direction in report["directions"] if not direction["gate_pass"]
                    ],
                    "fall_count": sum(int(direction["fall_count"]) for direction in report["directions"]),
                    "report": _report_ref(path),
                }
                for path, report in items
            ],
        }
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            -item["terrain_seed_pass_count"],
            -item["direction_pass_count"],
            item["fall_count"],
            item["worst_gate_ratio"],
        )
    )
    return candidates


def summarize_screening(paths: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        report = _load(path)
        if report.get("status") != "complete" or report.get("num_envs") != 16 or report.get("horizon_steps") != 300:
            raise ValueError(f"not a completed screening report: {path}")
        summaries.append(
            {
                "policy_id": report["policy_id"],
                "task": report["task"],
                "checkpoint": report["checkpoint"],
                "terrain_seed": int(report["terrain_seed"]),
                "all_directions_gate_pass": bool(report["all_directions_gate_pass"]),
                "failed_directions": [
                    direction["id"] for direction in report["directions"] if not direction["gate_pass"]
                ],
                "report": _report_ref(path),
            }
        )
    return summaries


def summarize_training(paths: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        report = _load(path)
        if not report.get("passed"):
            raise ValueError(f"training harness did not pass: {path}")
        summaries.append(
            {
                "run_name": report["run_name"],
                "task": report["task"],
                "report": _report_ref(path),
                "headless": bool(report["headless"]),
                "num_envs": int(report["num_envs"]),
                "iterations": int(report["max_iterations"]),
                "total_transitions": int(report["num_envs"]) * int(report["max_iterations"]) * 24,
                "optimizer_mini_batch_updates": int(report["max_iterations"]) * 5 * 4,
                "wall_time_seconds": float(report["wall_time_seconds"]),
                "mean_steps_per_second": float(report["performance"]["mean_steps_per_second"]),
                "peak_vram_mib": float(report["gpu"]["peak_used_mib"]),
                "final_mean_reward": float(report["performance"]["final_mean_reward"]),
                "final_mean_episode_length": float(report["performance"]["final_mean_episode_length"]),
                "final_checkpoint": {
                    "path": report["artifacts"]["checkpoint"],
                    "sha256": report["artifacts"]["checkpoint_sha256"],
                },
            }
        )
    return summaries


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    expected_seeds = set(args.terrain_seeds)
    candidates = summarize_full_evaluations(
        [path.resolve() for path in args.full_evaluations], expected_seeds
    )
    selected = candidates[0]
    qualified = [candidate for candidate in candidates if candidate["qualified_for_next_friction_stage"]]
    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "protocol": "road_geometry_reward_curriculum_v1",
        "reward_contract": _report_ref(args.reward_contract.resolve()),
        "training_runs": summarize_training([path.resolve() for path in args.training_reports]),
        "screening": {
            "num_envs": 16,
            "horizon_steps": 300,
            "terrain_seed": args.screening_terrain_seed,
            "reports": summarize_screening([path.resolve() for path in args.screening_reports]),
            "acceptance_limit": "선별 PASS만으로 채택하지 않고 32환경·500-step·3 terrain seed를 다시 평가한다.",
        },
        "full_evaluation": {
            "num_envs": 32,
            "horizon_steps": 500,
            "warmup_steps": 50,
            "terrain_seeds": sorted(expected_seeds),
            "candidates": candidates,
        },
        "selection": {
            "policy_id": selected["policy_id"],
            "checkpoint": selected["checkpoint"],
            "selection_rule": (
                "통과한 terrain seed 수, 전체 방향 PASS 수를 차례로 최대화한 뒤 "
                "낙상 수와 최악 normalized gate ratio를 최소화한다."
            ),
            "qualified_policy_count": len(qualified),
            "proceed_to_friction_f1": bool(qualified),
            "next_action": (
                "G0를 통과한 정책으로 두 구간 마찰 F1을 시작한다."
                if qualified
                else "G0가 3개 terrain seed를 통과하지 못했으므로 F1은 열지 않고 회전 보상과 평가 정렬을 더 검증한다."
            ),
        },
        "aggregate_source_sha256": file_sha256(Path(__file__)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-contract", required=True, type=Path)
    parser.add_argument("--training-reports", required=True, nargs="+", type=Path)
    parser.add_argument("--screening-reports", required=True, nargs="+", type=Path)
    parser.add_argument("--full-evaluations", required=True, nargs="+", type=Path)
    parser.add_argument("--terrain-seeds", nargs="+", type=int, default=[20260826, 20260827, 20260828])
    parser.add_argument("--screening-terrain-seed", type=int, default=20260828)
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
    print(json.dumps({"output": str(output), "selection": report["selection"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
