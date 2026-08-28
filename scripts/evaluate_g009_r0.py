#!/usr/bin/env python3
"""Evaluate a G009-5 R0 checkpoint on four stratified flat recovery poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaac_walk_g009.recover_contracts import SOLVER_JOINT_LIMIT_TOLERANCE_RAD  # noqa: E402

GOAL_ID = "g009"
STAGE_NUMBER = "G009-5"
STAGE_ID = "R0"
REPORT_ID = "g009_r0_flat_quantitative_evaluation"
DEFAULT_TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
POSE_NAMES = ("prone", "supine", "left_side", "right_side")
OFFICIAL_PROTOCOL = {
    "task": DEFAULT_TASK,
    "seed": 42,
    "num_envs": 256,
    "horizon_steps": 400,
    "minimum_success_rate": 0.80,
    "maximum_median_recovery_time_s": 4.0,
}
MAX_RAW_HARD_JOINT_LIMIT_VIOLATION_RAD = SOLVER_JOINT_LIMIT_TOLERANCE_RAD


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        pass
    try:
        return "%USERPROFILE%\\" + str(resolved.relative_to(Path.home().resolve()))
    except ValueError:
        return str(resolved)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def git_source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty_paths = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    source_dirty_paths = [
        line for line in dirty_paths if not line[3:].replace("\\", "/").startswith("reports/runs/")
    ]
    return {
        "commit": commit,
        "clean": not source_dirty_paths,
        "dirty_paths": dirty_paths,
        "source_dirty_paths": source_dirty_paths,
        "allowed_dirty_scope": "reports/runs evidence only",
    }


def source_bundle_sha256(files: Mapping[str, str]) -> str:
    payload = "\n".join(f"{path}:{digest}" for path, digest in files.items())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_source_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    files = bundle.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("training source_bundle.files must be non-empty")
    normalized: dict[str, str] = {}
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError("training source bundle entries must be string pairs")
        path = (REPO_ROOT / relative).resolve()
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"source bundle path escapes repository: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"source bundle file hash mismatch: {relative}")
        normalized[relative.replace("\\", "/")] = actual
    actual_bundle = source_bundle_sha256(normalized)
    if bundle.get("sha256") != actual_bundle:
        raise ValueError("training source bundle aggregate hash mismatch")
    return {"sha256": actual_bundle, "files": normalized}


def validate_training_report(path: Path, checkpoint: Path) -> dict[str, Any]:
    report = _read_json(path)
    checks = {
        "task": report.get("task") == DEFAULT_TASK,
        "seed": report.get("seed") == 42,
        "num_envs": report.get("num_envs") == 1024,
        "max_iterations": report.get("max_iterations") == 300,
        "headless": report.get("headless") is True,
        "scratch": report.get("resume", {}).get("enabled") is False,
        "qualification_preflight": (
            report.get("qualification_mode", {}).get("enabled") is True
            and report.get("qualification_mode", {}).get("preflight_passed") is True
            and report.get("qualification_mode", {}).get("policy_qualification_status") == "not_run"
        ),
        "run_health": report.get("run_health_passed") is True,
        "repository_clean": report.get("repository", {}).get("dirty") is False,
        "checkpoint": report.get("artifacts", {}).get("checkpoint_sha256") == file_sha256(checkpoint),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("training qualification binding failed: " + ", ".join(failed))
    commit = report.get("repository", {}).get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("training repository commit is missing or invalid")
    source_bundle = validate_source_bundle(report.get("source_bundle", {}))
    state = git_source_state()
    if not state["clean"]:
        raise ValueError("evaluation requires a clean repository")
    if state["commit"] != commit:
        raise ValueError("training/current repository commit mismatch")
    return {
        "path": portable_path(path),
        "sha256": file_sha256(path),
        "repository": {"commit": commit, "clean": True},
        "source_bundle": source_bundle,
        "checkpoint_sha256": file_sha256(checkpoint),
    }


def protocol_mode(args: argparse.Namespace) -> str:
    matches = all(getattr(args, key) == value for key, value in OFFICIAL_PROTOCOL.items())
    if matches:
        return "official_qualification"
    if not getattr(args, "diagnostic", False):
        raise ValueError("official protocol values are fixed; use --diagnostic for non-qualifying runs")
    return "diagnostic_only"


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def summarize_samples(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = sorted(_finite(values))
    if not samples:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    p95_index = min(len(samples) - 1, math.ceil(0.95 * len(samples)) - 1)
    return {
        "count": len(samples),
        "min": samples[0],
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "p95": samples[p95_index],
        "max": samples[-1],
    }


def _new_accumulator() -> dict[str, Any]:
    return {
        "episode_count": 0,
        "success_count": 0,
        "timeout_count": 0,
        "numeric_invalid_count": 0,
        "hard_joint_limit_count": 0,
        "other_termination_count": 0,
        "recovery_times_s": [],
        "max_raw_hard_joint_limit_violation_rad": 0.0,
    }


def finalize_pose_metrics(
    accumulator: Mapping[str, Any],
    *,
    minimum_success_rate: float,
    maximum_median_recovery_time_s: float,
) -> dict[str, Any]:
    episodes = int(accumulator["episode_count"])
    successes = int(accumulator["success_count"])
    success_rate = 0.0 if episodes == 0 else successes / episodes
    recovery = summarize_samples(accumulator["recovery_times_s"])
    safety_count = int(accumulator["numeric_invalid_count"]) + int(
        accumulator["hard_joint_limit_count"]
    )
    median_recovery = recovery["median"]
    max_raw_limit_violation = float(accumulator["max_raw_hard_joint_limit_violation_rad"])
    gate_checks = {
        "episodes_present": episodes > 0,
        "success_rate": success_rate >= minimum_success_rate,
        "median_recovery_time": median_recovery is not None
        and float(median_recovery) <= maximum_median_recovery_time_s,
        "no_safety_termination": safety_count == 0,
        "joint_limit_violation_within_solver_tolerance": (
            math.isfinite(max_raw_limit_violation)
            and max_raw_limit_violation <= MAX_RAW_HARD_JOINT_LIMIT_VIOLATION_RAD
        ),
    }
    return {
        "episode_count": episodes,
        "success_count": successes,
        "success_rate": success_rate,
        "termination_counts": {
            "stable_success": successes,
            "time_out": int(accumulator["timeout_count"]),
            "numeric_invalid": int(accumulator["numeric_invalid_count"]),
            "hard_joint_limit": int(accumulator["hard_joint_limit_count"]),
            "other": int(accumulator["other_termination_count"]),
        },
        "safe_termination_rate": 0.0 if episodes == 0 else (episodes - safety_count) / episodes,
        "recovery_time_s": recovery,
        "max_raw_hard_joint_limit_violation_rad": max_raw_limit_violation,
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
    }


def build_report(
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    step_dt_s: float,
    pose_accumulators: Mapping[str, Mapping[str, Any]],
    physics_readback: Mapping[str, Any],
    training_binding: Mapping[str, Any],
    source_state_before: Mapping[str, Any],
    source_state_after: Mapping[str, Any],
) -> dict[str, Any]:
    poses = [
        {
            "pose_id": pose,
            **finalize_pose_metrics(
                pose_accumulators[pose],
                minimum_success_rate=args.minimum_success_rate,
                maximum_median_recovery_time_s=args.maximum_median_recovery_time_s,
            ),
        }
        for pose in POSE_NAMES
    ]
    total_episodes = sum(item["episode_count"] for item in poses)
    total_successes = sum(item["success_count"] for item in poses)
    safety_terminations = sum(
        item["termination_counts"]["numeric_invalid"]
        + item["termination_counts"]["hard_joint_limit"]
        for item in poses
    )
    gate_pass = all(item["gate_pass"] for item in poses)
    mode = protocol_mode(args)
    source_stable = source_state_before == source_state_after and source_state_after.get("clean") is True
    if not source_stable:
        raise RuntimeError("repository commit/source state changed during evaluation")
    return {
        "schema_version": 1,
        "goal_id": GOAL_ID,
        "stage_number": STAGE_NUMBER,
        "stage_id": STAGE_ID,
        "report_id": REPORT_ID,
        "status": "pass" if gate_pass and mode == "official_qualification" else ("fail" if mode == "official_qualification" else "diagnostic"),
        "protocol": "g009_5_r0_four_pose_stratified_checkpoint_evaluation_v1",
        "protocol_mode": mode,
        "official_protocol": dict(OFFICIAL_PROTOCOL),
        "task": args.task,
        "checkpoint": dict(checkpoint),
        "seed": args.seed,
        "device": args.device,
        "headless": bool(args.headless),
        "observation_corruption": False,
        "num_envs": args.num_envs,
        "episodes_per_pose": args.num_envs // len(POSE_NAMES),
        "horizon_steps": args.horizon_steps,
        "step_dt_s": step_dt_s,
        "gate": {
            "minimum_success_rate_per_pose": args.minimum_success_rate,
            "maximum_median_recovery_time_s_per_pose": args.maximum_median_recovery_time_s,
            "numeric_invalid_allowed": 0,
            "hard_joint_limit_allowed": 0,
            "maximum_raw_hard_joint_limit_violation_rad": (
                MAX_RAW_HARD_JOINT_LIMIT_VIOLATION_RAD
            ),
            "solver_tolerance_calibration": {
                "gpu_reset_hold_observed_max_rad": 0.007703065872192383,
                "cpu_reset_hold_observed_max_rad": 0.003261566162109375,
                "scope": "runtime reset-pose calibration; not a learned-policy result",
            },
            "blocking_rule": "all four pose gates must pass independently",
        },
        "aggregate": {
            "episode_count": total_episodes,
            "success_count": total_successes,
            "success_rate": 0.0 if total_episodes == 0 else total_successes / total_episodes,
            "safety_termination_count": safety_terminations,
            "all_pose_gate_pass": gate_pass,
        },
        "poses": poses,
        "physics_readback": dict(physics_readback),
        "training_binding": dict(training_binding),
        "source_state": {"before": dict(source_state_before), "after": dict(source_state_after)},
        "source_bindings": {
            "evaluator": {
                "path": "scripts/evaluate_g009_r0.py",
                "sha256": file_sha256(Path(__file__)),
            },
            "config": {
                "path": "configs/g009_r0.json",
                "sha256": file_sha256(REPO_ROOT / "configs" / "g009_r0.json"),
            },
        },
        "interpretation": {
            "decision_source": "this multi-environment quantitative report",
            "video_scope": "qualitative motion evidence only",
            "causal_scope": "flat nominal-friction simulation; slope and asymmetric-friction recovery are not claimed",
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from isaac_walk_g009 import register_tasks

    protocol_mode(args)
    training_binding = validate_training_report(args.training_report.resolve(), args.checkpoint.resolve())
    source_state_before = git_source_state()
    if not source_state_before["clean"]:
        raise ValueError("evaluation source tree is dirty outside reports/runs")
    if args.num_envs <= 0 or args.num_envs % len(POSE_NAMES) != 0:
        raise ValueError("num_envs must be a positive multiple of four")
    register_tasks()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.reset_base.params.update(
        {"assignment_mode": "stratified", "pose_xy_range": (0.0, 0.0), "yaw_range": (0.0, 0.0)}
    )
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint.resolve()))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]
    observations, _ = env.get_observations()
    class_ids = env.unwrapped._g009_recover_fall_class.detach().clone()
    expected_ids = torch.arange(args.num_envs, device=class_ids.device) % len(POSE_NAMES)
    if not torch.equal(class_ids, expected_ids):
        raise RuntimeError("stratified pose assignment readback mismatch")

    active = torch.ones(args.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    elapsed_steps = torch.zeros(args.num_envs, dtype=torch.long, device=env.unwrapped.device)
    accumulators = {pose: _new_accumulator() for pose in POSE_NAMES}
    step_dt_s = float(env_cfg.sim.dt * env_cfg.decimation)
    try:
        for _ in range(args.horizon_steps):
            joint_position = robot.data.joint_pos
            joint_limits = robot.data.joint_pos_limits
            raw_limit_violation = torch.maximum(
                (joint_limits[..., 0] - joint_position).clamp_min(0.0),
                (joint_position - joint_limits[..., 1]).clamp_min(0.0),
            ).amax(dim=-1)
            for pose_index, pose in enumerate(POSE_NAMES):
                active_pose = active & (class_ids == pose_index)
                if bool(active_pose.any().item()):
                    observed = float(raw_limit_violation[active_pose].max().item())
                    accumulators[pose]["max_raw_hard_joint_limit_violation_rad"] = max(
                        float(accumulators[pose]["max_raw_hard_joint_limit_violation_rad"]),
                        observed,
                    )
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, dones, _ = env.step(actions)
            elapsed_steps += active.long()

            terms = {
                name: env.unwrapped.termination_manager.get_term(name).clone()
                for name in env.unwrapped.termination_manager.active_terms
            }
            done_now = active & dones.bool()
            for env_index in torch.nonzero(done_now, as_tuple=False).flatten().cpu().tolist():
                pose = POSE_NAMES[int(class_ids[env_index].item())]
                accumulator = accumulators[pose]
                accumulator["episode_count"] += 1
                is_success = bool(terms.get("stable_success", torch.zeros_like(done_now))[env_index].item())
                is_timeout = bool(terms.get("time_out", torch.zeros_like(done_now))[env_index].item())
                is_numeric = bool(terms.get("numeric_invalid", torch.zeros_like(done_now))[env_index].item())
                is_limit = bool(terms.get("hard_joint_limit", torch.zeros_like(done_now))[env_index].item())
                accumulator["success_count"] += int(is_success)
                accumulator["timeout_count"] += int(is_timeout)
                accumulator["numeric_invalid_count"] += int(is_numeric)
                accumulator["hard_joint_limit_count"] += int(is_limit)
                accumulator["other_termination_count"] += int(
                    not (is_success or is_timeout or is_numeric or is_limit)
                )
                if is_success:
                    accumulator["recovery_times_s"].append(float(elapsed_steps[env_index].item()) * step_dt_s)
            active &= ~done_now
            if not bool(active.any().item()):
                break

        # Treat an active episode at the evaluation horizon as a timeout-like failure.
        for env_index in torch.nonzero(active, as_tuple=False).flatten().cpu().tolist():
            pose = POSE_NAMES[int(class_ids[env_index].item())]
            accumulator = accumulators[pose]
            accumulator["episode_count"] += 1
            accumulator["timeout_count"] += 1

        materials = getattr(env.unwrapped, "_g009_foot_material_readback", None)
        effective = getattr(env.unwrapped, "_g009_effective_foot_friction", None)
        friction_valid = getattr(env.unwrapped, "_g009_effective_foot_friction_valid", None)
        if materials is None or effective is None or friction_valid is None:
            raise RuntimeError("foot/effective friction readback provenance is unavailable")
        materials = materials.detach().cpu()
        effective = effective.detach().cpu()
        if materials.shape != (args.num_envs, 4, 2) or effective.shape != (args.num_envs, 4, 2):
            raise RuntimeError("foot/effective friction readback shape mismatch")
        if not bool(friction_valid.all().item()) or not torch.isfinite(materials).all() or not torch.isfinite(effective).all():
            raise RuntimeError("foot/effective friction readback is invalid")
        terrain_pair = torch.tensor(
            [env_cfg.scene.terrain.physics_material.static_friction, env_cfg.scene.terrain.physics_material.dynamic_friction]
        )
        if str(env_cfg.scene.terrain.physics_material.friction_combine_mode) != "multiply":
            raise RuntimeError("official R0 friction provenance requires multiply combine mode")
        if not torch.allclose(effective, materials * terrain_pair, rtol=0.0, atol=1.0e-6):
            raise RuntimeError("effective friction does not match foot x terrain multiply readback")
        masses = robot.root_physx_view.get_masses().detach().cpu()
        physics_readback = {
            "terrain": {
                "type": "plane",
                "static_friction": float(env_cfg.scene.terrain.physics_material.static_friction),
                "dynamic_friction": float(env_cfg.scene.terrain.physics_material.dynamic_friction),
                "combine_mode": str(env_cfg.scene.terrain.physics_material.friction_combine_mode),
            },
            "robot_total_mass_kg": {
                "min": float(masses.sum(dim=1).min().item()),
                "mean": float(masses.sum(dim=1).mean().item()),
                "max": float(masses.sum(dim=1).max().item()),
            },
            "foot_material_readback": {
                "static_friction_min": float(materials[..., 0].min().item()),
                "static_friction_max": float(materials[..., 0].max().item()),
                "dynamic_friction_min": float(materials[..., 1].min().item()),
                "dynamic_friction_max": float(materials[..., 1].max().item()),
            },
            "effective_foot_friction": {
                "static_friction_min": float(effective[..., 0].min().item()),
                "static_friction_max": float(effective[..., 0].max().item()),
                "dynamic_friction_min": float(effective[..., 1].min().item()),
                "dynamic_friction_max": float(effective[..., 1].max().item()),
                "valid_for_all_envs": True,
                "derivation": "foot material readback multiplied by terrain material readback",
            },
            "active_terminations": list(env.unwrapped.termination_manager.active_terms),
            "policy_observation_dim": int(observations.shape[1]),
            "action_dim": int(env.unwrapped.action_manager.total_action_dim),
        }
        checkpoint = {"path": portable_path(args.checkpoint), "sha256": file_sha256(args.checkpoint)}
        source_state_after = git_source_state()
        source_bundle_after = validate_source_bundle(training_binding["source_bundle"])
        if source_bundle_after != training_binding["source_bundle"]:
            raise RuntimeError("training source bundle changed during evaluation")
        return build_report(
            args=args,
            checkpoint=checkpoint,
            step_dt_s=step_dt_s,
            pose_accumulators=accumulators,
            physics_readback=physics_readback,
            training_binding=training_binding,
            source_state_before=source_state_before,
            source_state_after=source_state_after,
        )
    finally:
        env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon-steps", type=int, default=400)
    parser.add_argument("--minimum-success-rate", type=float, default=0.80)
    parser.add_argument("--maximum-median-recovery-time-s", type=float, default=4.0)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hard-exit-after-report", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.training_report.is_file():
        raise FileNotFoundError(args.training_report)
    if args.horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if not 0.0 <= args.minimum_success_rate <= 1.0:
        raise ValueError("minimum_success_rate must be in [0, 1]")
    if args.maximum_median_recovery_time_s <= 0.0:
        raise ValueError("maximum_median_recovery_time_s must be positive")
    protocol_mode(args)
    return args


def main(argv: list[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    args = parse_args(argv)
    started_at = time.time()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = evaluate(args)
        report["wall_time_seconds"] = round(time.time() - started_at, 3)
        _write_json_atomic(args.output.resolve(), report)
        print(json.dumps({"output": str(args.output.resolve()), "status": report["status"]}), flush=True)
        exit_code = 0 if report["status"] == "pass" else 1
        if args.hard_exit_after_report:
            sys.stdout.flush()
            os._exit(exit_code)
        return exit_code
    finally:
        simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
