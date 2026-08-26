#!/usr/bin/env python3
"""Evaluate fixed forward, backward, left-yaw, and right-yaw commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DIRECTION_COMMANDS = (
    {"id": "forward", "command": (0.60, 0.0, 0.0)},
    {"id": "backward", "command": (-0.40, 0.0, 0.0)},
    {"id": "left_turn", "command": (0.0, 0.0, 0.50)},
    {"id": "right_turn", "command": (0.0, 0.0, -0.50)},
)

GATE = {
    "survival_rate_min": 1.0,
    "roll_abs_rad_max": 0.35,
    "pitch_abs_rad_max": 0.35,
    "linear_tracking_rmse_mps_max": 0.25,
    "yaw_tracking_rmse_radps_max": 0.25,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(Path.home().resolve())
    except ValueError:
        return str(resolved)
    return "%USERPROFILE%\\" + str(relative)


def new_accumulator(trial_count: int) -> dict[str, float | int]:
    return {
        "trial_count": trial_count,
        "sample_count": 0,
        "linear_error_sq_sum": 0.0,
        "yaw_error_sq_sum": 0.0,
        "achieved_vx_sum": 0.0,
        "achieved_vy_sum": 0.0,
        "achieved_yaw_sum": 0.0,
        "torque_norm_sum": 0.0,
        "mechanical_power_sum": 0.0,
        "roll_abs_max": 0.0,
        "pitch_abs_max": 0.0,
        "fall_count": 0,
        "timeout_count": 0,
    }


def command_sign_pass(command: tuple[float, float, float], achieved: tuple[float, float, float]) -> bool:
    for target, actual in zip(command, achieved):
        if abs(target) > 1.0e-9 and target * actual <= 0.0:
            return False
    return True


def finalize_accumulator(
    accumulator: dict[str, float | int], command: tuple[float, float, float]
) -> dict[str, Any]:
    samples = int(accumulator["sample_count"])
    trials = int(accumulator["trial_count"])

    def mean(key: str) -> float | None:
        return None if samples == 0 else float(accumulator[key]) / samples

    achieved_values = (
        mean("achieved_vx_sum"),
        mean("achieved_vy_sum"),
        mean("achieved_yaw_sum"),
    )
    achieved = tuple(0.0 if value is None else value for value in achieved_values)
    survival_rate = None if trials == 0 else 1.0 - int(accumulator["fall_count"]) / trials
    linear_mse = mean("linear_error_sq_sum")
    yaw_mse = mean("yaw_error_sq_sum")
    linear_rmse = None if linear_mse is None else math.sqrt(linear_mse)
    yaw_rmse = None if yaw_mse is None else math.sqrt(yaw_mse)
    gate_pass = (
        samples > 0
        and survival_rate is not None
        and survival_rate >= GATE["survival_rate_min"]
        and float(accumulator["roll_abs_max"]) <= GATE["roll_abs_rad_max"]
        and float(accumulator["pitch_abs_max"]) <= GATE["pitch_abs_rad_max"]
        and linear_rmse is not None
        and linear_rmse <= GATE["linear_tracking_rmse_mps_max"]
        and yaw_rmse is not None
        and yaw_rmse <= GATE["yaw_tracking_rmse_radps_max"]
        and command_sign_pass(command, achieved)
    )
    return {
        "sample_count": samples,
        "trial_count": trials,
        "fall_count": int(accumulator["fall_count"]),
        "timeout_count": int(accumulator["timeout_count"]),
        "survival_rate": survival_rate,
        "linear_tracking_rmse_mps": linear_rmse,
        "yaw_tracking_rmse_radps": yaw_rmse,
        "achieved_mean": {
            "lin_vel_x_mps": achieved_values[0],
            "lin_vel_y_mps": achieved_values[1],
            "ang_vel_z_radps": achieved_values[2],
        },
        "torque_l2_norm_mean_nm": mean("torque_norm_sum"),
        "absolute_mechanical_power_mean_w": mean("mechanical_power_sum"),
        "roll_abs_rad_max": float(accumulator["roll_abs_max"]),
        "pitch_abs_rad_max": float(accumulator["pitch_abs_max"]),
        "command_sign_pass": command_sign_pass(command, achieved),
        "gate_pass": gate_pass,
    }


def validate_protocol(num_envs: int, horizon_steps: int, warmup_steps: int) -> None:
    if num_envs <= 0 or num_envs % len(DIRECTION_COMMANDS) != 0:
        raise ValueError(f"num_envs must be a positive multiple of {len(DIRECTION_COMMANDS)}")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if warmup_steps < 0 or warmup_steps >= horizon_steps:
        raise ValueError("warmup_steps must be in [0, horizon_steps)")


def _write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from isaac_walk_g008 import register_tasks

    validate_protocol(args.num_envs, args.horizon_steps, args.warmup_steps)
    register_tasks()
    device = args.device
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.eval_seed
    if args.terrain_mode == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        if hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None
    env_cfg.episode_length_s = max(
        float(env_cfg.episode_length_s),
        (args.horizon_steps + 2) * float(env_cfg.sim.dt * env_cfg.decimation),
    )
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass = None
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    if args.domain_mode == "nominal":
        env_cfg.events.physics_material.params.update(
            {
                "static_friction_range": (0.8, 0.8),
                "dynamic_friction_range": (0.6, 0.6),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
                "make_consistent": True,
            }
        )
        if hasattr(env_cfg.events, "add_leg_mass"):
            env_cfg.events.add_leg_mass = None

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.eval_seed
    agent_cfg.device = device
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(args.checkpoint.resolve()))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    command_buffer = env.unwrapped.command_manager.get_command("base_velocity")
    robot = env.unwrapped.scene["robot"]
    condition_index = torch.arange(args.num_envs, device=env.unwrapped.device) % len(DIRECTION_COMMANDS)
    fixed_commands = torch.tensor(
        [DIRECTION_COMMANDS[index]["command"] for index in condition_index.cpu().tolist()],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    accumulators = {
        item["id"]: new_accumulator(int((condition_index == index).sum().item()))
        for index, item in enumerate(DIRECTION_COMMANDS)
    }
    active = torch.ones(args.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    obs, _ = env.get_observations()

    try:
        for step in range(args.horizon_steps):
            with torch.inference_mode():
                command_buffer.copy_(fixed_commands)
                obs, _ = env.get_observations()
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

                root_lin = robot.data.root_lin_vel_b[:, :2]
                root_yaw = robot.data.root_ang_vel_b[:, 2]
                torque = robot.data.applied_torque
                joint_vel = robot.data.joint_vel
                quaternion = robot.data.root_quat_w
                w, x, y, z = quaternion.unbind(dim=1)
                roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
                fall = env.unwrapped.termination_manager.get_term("base_contact").clone()
                timeout = env.unwrapped.termination_manager.get_term("time_out").clone()

                if step >= args.warmup_steps:
                    for index, item in enumerate(DIRECTION_COMMANDS):
                        # env.step has already auto-reset terminal environments. Exclude
                        # those reset states from continuous metrics, but count their
                        # termination event below against the original active trial.
                        mask = active & ~(fall | timeout) & (condition_index == index)
                        count = int(mask.sum().item())
                        accumulator = accumulators[item["id"]]
                        accumulator["sample_count"] += count
                        if count == 0:
                            continue
                        error = root_lin[mask] - fixed_commands[mask, :2]
                        yaw_error = root_yaw[mask] - fixed_commands[mask, 2]
                        accumulator["linear_error_sq_sum"] += float(torch.square(error).sum().item())
                        accumulator["yaw_error_sq_sum"] += float(torch.square(yaw_error).sum().item())
                        accumulator["achieved_vx_sum"] += float(root_lin[mask, 0].sum().item())
                        accumulator["achieved_vy_sum"] += float(root_lin[mask, 1].sum().item())
                        accumulator["achieved_yaw_sum"] += float(root_yaw[mask].sum().item())
                        accumulator["torque_norm_sum"] += float(torch.linalg.vector_norm(torque[mask], dim=1).sum().item())
                        accumulator["mechanical_power_sum"] += float(
                            torch.sum(torch.abs(torque[mask] * joint_vel[mask]), dim=1).sum().item()
                        )
                        accumulator["roll_abs_max"] = max(
                            float(accumulator["roll_abs_max"]), float(torch.abs(roll[mask]).max().item())
                        )
                        accumulator["pitch_abs_max"] = max(
                            float(accumulator["pitch_abs_max"]), float(torch.abs(pitch[mask]).max().item())
                        )

                for index, item in enumerate(DIRECTION_COMMANDS):
                    mask = active & (condition_index == index)
                    accumulators[item["id"]]["fall_count"] += int((fall & mask).sum().item())
                    accumulators[item["id"]]["timeout_count"] += int((timeout & mask).sum().item())
                active &= ~(fall | timeout)
    finally:
        env.close()

    results = []
    for item in DIRECTION_COMMANDS:
        command = tuple(float(value) for value in item["command"])
        results.append(
            {
                "id": item["id"],
                "command": {
                    "lin_vel_x_mps": command[0],
                    "lin_vel_y_mps": command[1],
                    "ang_vel_z_radps": command[2],
                },
                **finalize_accumulator(accumulators[item["id"]], command),
            }
        )
    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "protocol": "directional_qualification_v1",
        "task": args.task,
        "domain_mode": args.domain_mode,
        "terrain_mode": args.terrain_mode,
        "headless": bool(args.headless),
        "device": device,
        "evaluation_seed": args.eval_seed,
        "num_envs": args.num_envs,
        "environments_per_direction": args.num_envs // len(DIRECTION_COMMANDS),
        "horizon_steps": args.horizon_steps,
        "warmup_steps": args.warmup_steps,
        "step_dt_s": float(env_cfg.sim.dt * env_cfg.decimation),
        "checkpoint": {
            "path": portable_path(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
        },
        "evaluation_source_sha256": file_sha256(Path(__file__)),
        "gate": GATE,
        "all_directions_gate_pass": all(item["gate_pass"] for item in results),
        "directions": results,
        "metric_notes": {
            "tracking_window": "warmup 이후, 첫 종료 전 active state만 집계",
            "torque": "관절 applied_torque 벡터의 L2 norm 평균",
            "mechanical_power": "sum(abs(applied_torque * joint_velocity))의 시뮬레이션 proxy",
            "causal_scope": "정책의 방향 명령 추종을 확인하며 실물 전이 성능을 입증하지 않음",
        },
    }


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", default="Isaac-G008-Velocity-Rough-Go2-CommandSuite-v0")
    parser.add_argument("--domain-mode", choices=("task", "nominal"), default="nominal")
    parser.add_argument("--terrain-mode", choices=("task", "plane"), default="plane")
    parser.add_argument("--eval-seed", type=int, default=20260826)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--horizon-steps", type=int, default=250)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--output", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    from isaaclab.app import AppLauncher

    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    started_at = time.time()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = evaluate(args)
        report["wall_time_seconds"] = round(time.time() - started_at, 3)
        _write_json_atomic(args.output.resolve(), report)
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "all_directions_gate_pass": report["all_directions_gate_pass"],
                }
            ),
            flush=True,
        )
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
