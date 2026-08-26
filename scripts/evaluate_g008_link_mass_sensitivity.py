#!/usr/bin/env python3
"""Evaluate command and leg-mass S1 policies under controlled link-group mass scales."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for import_root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluate_g008_directions import (  # noqa: E402
    DIRECTION_COMMANDS,
    GATE,
    file_sha256,
    finalize_accumulator,
    new_accumulator,
    portable_path,
)


LINK_GROUP_PATTERNS = {
    "hip": ".*_hip",
    "thigh": ".*_thigh",
    "calf": ".*_calf",
    "foot": ".*_foot",
}
MASS_FACTORS = (0.80, 0.90, 0.95, 1.05, 1.10, 1.20)


def build_mass_cases() -> tuple[dict[str, Any], ...]:
    cases: list[dict[str, Any]] = [{"id": "nominal", "group": None, "factor": 1.0}]
    for group in LINK_GROUP_PATTERNS:
        for factor in MASS_FACTORS:
            cases.append(
                {
                    "id": f"{group}_{int(round(factor * 100)):03d}",
                    "group": group,
                    "factor": factor,
                }
            )
    return tuple(cases)


MASS_CASES = build_mass_cases()


def validate_protocol(num_envs: int, horizon_steps: int, warmup_steps: int) -> None:
    assignment_width = len(MASS_CASES) * 2 * len(DIRECTION_COMMANDS)
    if num_envs <= 0 or num_envs % assignment_width:
        raise ValueError(f"num_envs must be a positive multiple of {assignment_width}")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if warmup_steps < 0 or warmup_steps >= horizon_steps:
        raise ValueError("warmup_steps must be in [0, horizon_steps)")


def summarize_group_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = {}
    for group in LINK_GROUP_PATTERNS:
        group_cases = [case for case in cases if case["group"] == group]
        summaries[group] = {
            "passing_factors": [case["factor"] for case in group_cases if case["all_directions_gate_pass"]],
            "failing_factors": [
                {
                    "factor": case["factor"],
                    "failed_directions": [
                        direction["id"] for direction in case["directions"] if not direction["gate_pass"]
                    ],
                }
                for case in group_cases
                if not case["all_directions_gate_pass"]
            ],
        }
    return summaries


def _statistics(values: Any) -> dict[str, float]:
    return {
        "min": float(values.min().item()),
        "mean": float(values.float().mean().item()),
        "max": float(values.max().item()),
    }


def _write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _configure_env(args: argparse.Namespace) -> Any:
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.eval_seed
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
    if hasattr(env_cfg.events, "add_leg_mass"):
        env_cfg.events.add_leg_mass = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None
    env_cfg.events.physics_material.params.update(
        {
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        }
    )
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    return env_cfg


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaac_walk_g008 import register_tasks

    validate_protocol(args.num_envs, args.horizon_steps, args.warmup_steps)
    register_tasks()
    env_cfg = _configure_env(args)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.eval_seed
    agent_cfg.device = args.device
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    checkpoints = {
        "command": args.command_checkpoint.resolve(),
        "leg_mass_s1": args.leg_mass_checkpoint.resolve(),
    }
    runners = {}
    policies = {}
    for policy_id, checkpoint in checkpoints.items():
        print(f"[link-mass] loading policy={policy_id}", flush=True)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(checkpoint))
        runners[policy_id] = runner
        policies[policy_id] = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    env_index = torch.arange(args.num_envs, device=device)
    direction_count = len(DIRECTION_COMMANDS)
    policy_ids = tuple(policies)
    direction_index = env_index % direction_count
    policy_index = torch.div(env_index, direction_count, rounding_mode="floor") % len(policy_ids)
    case_index = torch.div(
        env_index, direction_count * len(policy_ids), rounding_mode="floor"
    ) % len(MASS_CASES)
    fixed_commands = torch.tensor(
        [DIRECTION_COMMANDS[index]["command"] for index in direction_index.cpu().tolist()],
        dtype=torch.float32,
        device=device,
    )

    group_ids = {}
    group_names = {}
    for group, pattern in LINK_GROUP_PATTERNS.items():
        ids, names = robot.find_bodies(pattern)
        if len(ids) != 4:
            raise RuntimeError(f"expected four {group} bodies, got {names}")
        group_ids[group] = torch.tensor(ids, dtype=torch.long)
        group_names[group] = list(names)

    env_ids_cpu = torch.arange(args.num_envs, device="cpu")
    masses = robot.data.default_mass.detach().cpu().clone()
    inertias = robot.data.default_inertia.detach().cpu().clone()
    default_masses = masses.clone()
    default_inertias = inertias.clone()
    case_index_cpu = case_index.cpu()
    for case_number, case in enumerate(MASS_CASES):
        if case["group"] is None:
            continue
        rows = torch.where(case_index_cpu == case_number)[0]
        bodies = group_ids[case["group"]]
        masses[rows[:, None], bodies[None, :]] = (
            default_masses[rows[:, None], bodies[None, :]] * float(case["factor"])
        )
        inertias[rows[:, None], bodies[None, :], :] = (
            default_inertias[rows[:, None], bodies[None, :], :] * float(case["factor"])
        )
    robot.root_physx_view.set_masses(masses, env_ids_cpu)
    robot.root_physx_view.set_inertias(inertias, env_ids_cpu)

    case_physics = []
    all_leg_ids = torch.cat(tuple(group_ids.values()))
    for case_number, case in enumerate(MASS_CASES):
        rows = torch.where(case_index_cpu == case_number)[0]
        if case["group"] is None:
            scaled_ids = all_leg_ids
        else:
            scaled_ids = group_ids[case["group"]]
        ratios = masses[rows[:, None], scaled_ids[None, :]] / default_masses[
            rows[:, None], scaled_ids[None, :]
        ]
        inertia_error = torch.abs(
            inertias[rows[:, None], scaled_ids[None, :], :]
            - default_inertias[rows[:, None], scaled_ids[None, :], :] * ratios.unsqueeze(-1)
        )
        case_physics.append(
            {
                "case_id": case["id"],
                "mass_ratio": _statistics(ratios),
                "total_leg_mass_kg": _statistics(masses[rows][:, all_leg_ids].sum(dim=1)),
                "inertia_scale_absolute_error_max": float(inertia_error.max().item()),
            }
        )

    accumulators = {
        policy_id: {
            case["id"]: {
                direction["id"]: new_accumulator(
                    int(
                        (
                            (policy_index == policy_number)
                            & (case_index == case_number)
                            & (direction_index == direction_number)
                        )
                        .sum()
                        .item()
                    )
                )
                for direction_number, direction in enumerate(DIRECTION_COMMANDS)
            }
            for case_number, case in enumerate(MASS_CASES)
        }
        for policy_number, policy_id in enumerate(policy_ids)
    }
    command_buffer = env.unwrapped.command_manager.get_command("base_velocity")
    active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    env.reset()
    obs, _ = env.get_observations()

    for step in range(args.horizon_steps):
        with torch.inference_mode():
            command_buffer.copy_(fixed_commands)
            obs, _ = env.get_observations()
            actions = torch.zeros((args.num_envs, env.num_actions), device=device)
            for policy_number, policy_id in enumerate(policy_ids):
                mask = policy_index == policy_number
                actions[mask] = policies[policy_id](obs[mask])
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

            for policy_number, policy_id in enumerate(policy_ids):
                for case_number, case in enumerate(MASS_CASES):
                    for direction_number, direction in enumerate(DIRECTION_COMMANDS):
                        group_mask = (
                            (policy_index == policy_number)
                            & (case_index == case_number)
                            & (direction_index == direction_number)
                        )
                        metric_mask = active & ~(fall | timeout) & group_mask
                        accumulator = accumulators[policy_id][case["id"]][direction["id"]]
                        if step >= args.warmup_steps:
                            count = int(metric_mask.sum().item())
                            accumulator["sample_count"] += count
                            if count:
                                error = root_lin[metric_mask] - fixed_commands[metric_mask, :2]
                                yaw_error = root_yaw[metric_mask] - fixed_commands[metric_mask, 2]
                                accumulator["linear_error_sq_sum"] += float(torch.square(error).sum().item())
                                accumulator["yaw_error_sq_sum"] += float(torch.square(yaw_error).sum().item())
                                accumulator["achieved_vx_sum"] += float(root_lin[metric_mask, 0].sum().item())
                                accumulator["achieved_vy_sum"] += float(root_lin[metric_mask, 1].sum().item())
                                accumulator["achieved_yaw_sum"] += float(root_yaw[metric_mask].sum().item())
                                accumulator["torque_norm_sum"] += float(
                                    torch.linalg.vector_norm(torque[metric_mask], dim=1).sum().item()
                                )
                                accumulator["mechanical_power_sum"] += float(
                                    torch.sum(
                                        torch.abs(torque[metric_mask] * joint_vel[metric_mask]), dim=1
                                    )
                                    .sum()
                                    .item()
                                )
                                accumulator["roll_abs_max"] = max(
                                    float(accumulator["roll_abs_max"]),
                                    float(torch.abs(roll[metric_mask]).max().item()),
                                )
                                accumulator["pitch_abs_max"] = max(
                                    float(accumulator["pitch_abs_max"]),
                                    float(torch.abs(pitch[metric_mask]).max().item()),
                                )
                        accumulator["fall_count"] += int((fall & active & group_mask).sum().item())
                        accumulator["timeout_count"] += int((timeout & active & group_mask).sum().item())
            active &= ~(fall | timeout)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.horizon_steps:
            print(f"[link-mass] completed_steps={step + 1}", flush=True)

    policy_results = []
    for policy_id in policy_ids:
        cases = []
        for case in MASS_CASES:
            directions = []
            for direction in DIRECTION_COMMANDS:
                command = tuple(float(value) for value in direction["command"])
                result = finalize_accumulator(
                    accumulators[policy_id][case["id"]][direction["id"]], command
                )
                directions.append({"id": direction["id"], "command": list(command), **result})
            cases.append(
                {
                    **case,
                    "all_directions_gate_pass": all(direction["gate_pass"] for direction in directions),
                    "directions": directions,
                }
            )
        policy_results.append(
            {
                "policy_id": policy_id,
                "checkpoint": {
                    "path": portable_path(checkpoints[policy_id]),
                    "sha256": file_sha256(checkpoints[policy_id]),
                },
                "nominal_all_directions_gate_pass": cases[0]["all_directions_gate_pass"],
                "group_summary": summarize_group_cases(cases),
                "cases": cases,
            }
        )

    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "protocol": "controlled_link_group_mass_sensitivity_v1",
        "task": args.task,
        "terrain_mode": "plane",
        "headless": bool(args.headless),
        "device": args.device,
        "evaluation_seed": args.eval_seed,
        "num_envs": args.num_envs,
        "repetitions_per_policy_case_direction": args.num_envs
        // (len(MASS_CASES) * len(policy_ids) * len(DIRECTION_COMMANDS)),
        "horizon_steps": args.horizon_steps,
        "warmup_steps": args.warmup_steps,
        "step_dt_s": float(env_cfg.sim.dt * env_cfg.decimation),
        "observation_corruption": False,
        "gate": GATE,
        "link_groups": group_names,
        "mass_cases": [dict(case) for case in MASS_CASES],
        "case_physics": case_physics,
        "policies": policy_results,
        "evaluation_source_sha256": file_sha256(Path(__file__)),
        "direction_evaluator_source_sha256": file_sha256(REPO_ROOT / "scripts" / "evaluate_g008_directions.py"),
        "interpretation_contract": {
            "training_range": "leg-mass S1 trained with independent per-body mass factors in [0.95, 1.05]",
            "screening_range": "this controlled evaluation extends one link group at a time to [0.80, 1.20]",
            "inertia": "each selected link inertia tensor is scaled by the same factor as its mass",
            "claim_scope": "single-seed Isaac Sim sensitivity screen, not a payload certification",
        },
    }


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-checkpoint", required=True, type=Path)
    parser.add_argument("--leg-mass-checkpoint", required=True, type=Path)
    parser.add_argument("--task", default="Isaac-G008-Velocity-Rough-Go2-CommandSuite-v0")
    parser.add_argument("--eval-seed", type=int, default=20260826)
    parser.add_argument("--num-envs", type=int, default=800)
    parser.add_argument("--horizon-steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hard-exit-after-report", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    for checkpoint in (args.command_checkpoint, args.leg_mass_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    return args


def main() -> int:
    from isaaclab.app import AppLauncher

    args = parse_args()
    started_at = time.time()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = evaluate(args)
        report["wall_time_seconds"] = round(time.time() - started_at, 3)
        _write_json_atomic(args.output.resolve(), report)
        print(json.dumps({"output": str(args.output.resolve()), "status": report["status"]}), flush=True)
        if args.hard_exit_after_report:
            os._exit(0)
    finally:
        simulation_app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
