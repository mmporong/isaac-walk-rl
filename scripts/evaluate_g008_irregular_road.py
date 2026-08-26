#!/usr/bin/env python3
"""Evaluate one G008 policy on a non-periodic 2-D friction and road-height field."""

from __future__ import annotations

import argparse
import json
import math
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
from isaac_walk_g008.irregular_road import (  # noqa: E402
    IRREGULAR_ROAD_MESH_PRIM,
    IrregularRoadSpec,
    field_summary,
    generate_irregular_road,
)


GROUND_COLLISION_PRIM = "/World/ground/terrain/GroundPlane/CollisionPlane"


def _write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize_diversity_counts(counts: dict[int, int]) -> dict[str, Any]:
    total = sum(counts.values())
    return {
        "frame_counts": {str(key): int(counts.get(key, 0)) for key in range(1, 5)},
        "frame_ratios": {
            str(key): None if total == 0 else float(counts.get(key, 0) / total)
            for key in range(1, 5)
        },
        "all_same_frame_ratio": None if total == 0 else float(counts.get(1, 0) / total),
        "all_four_distinct_frame_ratio": None if total == 0 else float(counts.get(4, 0) / total),
        "maximum_simultaneous_bucket_count": max((key for key, value in counts.items() if value), default=0),
    }


def _new_road_accumulator(trial_count: int, bucket_count: int) -> dict[str, Any]:
    accumulator: dict[str, Any] = new_accumulator(trial_count)
    accumulator.update(
        {
            "friction_bucket_foot_sample_counts": [0] * bucket_count,
            "four_foot_diversity_counts": {index: 0 for index in range(1, 5)},
            "material_transition_count": 0,
            "footprint_height_span_sum_m": 0.0,
            "footprint_height_span_max_m": 0.0,
            "terrain_height_min_m": math.inf,
            "terrain_height_max_m": -math.inf,
            "local_slope_sum_deg": 0.0,
            "local_slope_max_deg": 0.0,
            "foot_surface_sample_count": 0,
            "out_of_field_foot_sample_count": 0,
            "contact_foot_sample_count": 0,
            "contact_slip_speed_sum": 0.0,
            "contact_slip_over_0_1_count": 0,
            "contact_termination_count": 0,
            "kinematic_fall_count": 0,
        }
    )
    return accumulator


def _verify_collision_surface(field: Any) -> dict[str, Any]:
    import isaacsim.core.utils.stage as stage_utils
    from pxr import UsdGeom, UsdPhysics

    stage = stage_utils.get_current_stage()
    scan_prim = stage.GetPrimAtPath(IRREGULAR_ROAD_MESH_PRIM)
    collision_meshes = []
    collision_face_total = 0
    material_targets = []
    for bucket in range(len(field.spec.static_friction)):
        mesh_path = f"/World/irregular_road_field/friction_{bucket}_surface"
        prim = stage.GetPrimAtPath(mesh_path)
        has_collision = bool(UsdPhysics.CollisionAPI(prim))
        if not prim.IsValid() or not has_collision:
            raise RuntimeError(f"missing collision material mesh: {mesh_path}")
        face_count = len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())
        targets = prim.GetRelationship("material:binding:physics").GetTargets()
        if len(targets) != 1:
            raise RuntimeError(f"physics material binding mismatch: {mesh_path} targets={targets}")
        collision_face_total += face_count
        material_targets.append(targets[0].pathString)
        collision_meshes.append(
            {"bucket": bucket, "prim_path": mesh_path, "face_count": face_count, "has_collision_api": True}
        )
    expected_faces = 2 * field.material_indices.size
    ground_exists = stage.GetPrimAtPath(GROUND_COLLISION_PRIM).IsValid()
    scan_has_collision = bool(UsdPhysics.CollisionAPI(scan_prim))
    if ground_exists or not scan_prim.IsValid() or scan_has_collision or collision_face_total != expected_faces:
        raise RuntimeError(
            "irregular-road surface isolation failed: "
            f"ground={ground_exists}, scan_valid={scan_prim.IsValid()}, "
            f"scan_collision={scan_has_collision}, faces={collision_face_total}/{expected_faces}"
        )
    return {
        "default_ground_collision_prim": GROUND_COLLISION_PRIM,
        "default_ground_collision_exists": ground_exists,
        "height_scan_prim": IRREGULAR_ROAD_MESH_PRIM,
        "height_scan_has_collision_api": scan_has_collision,
        "collision_meshes": collision_meshes,
        "collision_face_total": collision_face_total,
        "expected_face_total": expected_faces,
        "one_material_binding_per_collision_mesh": len(set(material_targets)) == len(material_targets),
        "physics_material_targets": material_targets,
    }


def _configure_env(args: argparse.Namespace) -> tuple[Any, Any]:
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.eval_seed
    env_cfg.scene.env_spacing = args.env_spacing_m
    road_cfg = env_cfg.scene.irregular_road_field.spawn
    road_cfg.seed = args.terrain_seed
    road_cfg.env_spacing_m = args.env_spacing_m
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
    spec = IrregularRoadSpec(
        x_min_m=road_cfg.x_min_m,
        x_max_m=road_cfg.x_max_m,
        y_min_m=road_cfg.y_min_m,
        y_max_m=road_cfg.y_max_m,
        cell_size_m=road_cfg.cell_size_m,
        seed=road_cfg.seed,
        env_spacing_m=road_cfg.env_spacing_m,
        static_friction=tuple(road_cfg.static_friction),
        dynamic_friction=tuple(road_cfg.dynamic_friction),
        colors_rgb=tuple(tuple(value) for value in road_cfg.colors_rgb),
        crown_height_m=road_cfg.crown_height_m,
        undulation_amplitude_m=road_cfg.undulation_amplitude_m,
        roughness_amplitude_m=road_cfg.roughness_amplitude_m,
        pothole_depth_m=road_cfg.pothole_depth_m,
    )
    return env_cfg, generate_irregular_road(spec)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaac_walk_g008 import register_tasks

    if args.num_envs <= 0 or args.num_envs % len(DIRECTION_COMMANDS) != 0:
        raise ValueError("num_envs must be a positive multiple of four")
    if args.warmup_steps < 0 or args.warmup_steps >= args.horizon_steps:
        raise ValueError("warmup_steps must be in [0, horizon_steps)")
    register_tasks()
    env_cfg, field = _configure_env(args)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.eval_seed
    agent_cfg.device = args.device
    print("[irregular-road] creating environment", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    surface_readback = _verify_collision_surface(field)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint.resolve()))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[irregular-road] loaded policy={args.policy_id}", flush=True)

    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]
    robot_foot_ids, robot_foot_names = robot.find_bodies(".*_foot")
    sensor_foot_ids, sensor_foot_names = contact_sensor.find_bodies(".*_foot")
    if robot_foot_names != sensor_foot_names:
        raise RuntimeError("robot and contact-sensor foot ordering differ")
    command_buffer = env.unwrapped.command_manager.get_command("base_velocity")
    condition_index = torch.arange(args.num_envs, device=env.unwrapped.device) % len(DIRECTION_COMMANDS)
    fixed_commands = torch.tensor(
        [DIRECTION_COMMANDS[index]["command"] for index in condition_index.cpu().tolist()],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    accumulators = {
        item["id"]: _new_road_accumulator(
            int((condition_index == index).sum().item()), len(field.spec.static_friction)
        )
        for index, item in enumerate(DIRECTION_COMMANDS)
    }
    device = env.unwrapped.device
    materials_t = torch.as_tensor(field.material_indices, dtype=torch.int64, device=device)
    heights_t = torch.as_tensor(field.heights_m, dtype=torch.float32, device=device)
    slopes_t = torch.as_tensor(field.local_slope_deg, dtype=torch.float32, device=device)
    active = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    previous_materials = None
    env.reset()
    obs, _ = env.get_observations()

    def sample_surface(xy: Any) -> tuple[Any, Any, Any, Any]:
        fx = (xy[..., 0] - field.spec.x_min_m) / field.spec.cell_size_m
        fy = (xy[..., 1] - field.spec.y_min_m) / field.spec.cell_size_m
        ix_raw = torch.floor(fx).to(torch.int64)
        iy_raw = torch.floor(fy).to(torch.int64)
        inside = (
            (ix_raw >= 0)
            & (iy_raw >= 0)
            & (ix_raw < materials_t.shape[0])
            & (iy_raw < materials_t.shape[1])
        )
        ix = torch.clamp(ix_raw, 0, materials_t.shape[0] - 1)
        iy = torch.clamp(iy_raw, 0, materials_t.shape[1] - 1)
        tx = torch.clamp(fx - ix, 0.0, 1.0)
        ty = torch.clamp(fy - iy, 0.0, 1.0)
        h00 = heights_t[ix, iy]
        h10 = heights_t[ix + 1, iy]
        h01 = heights_t[ix, iy + 1]
        h11 = heights_t[ix + 1, iy + 1]
        height = (
            (1.0 - tx) * (1.0 - ty) * h00
            + tx * (1.0 - ty) * h10
            + (1.0 - tx) * ty * h01
            + tx * ty * h11
        )
        return materials_t[ix, iy], height, slopes_t[ix, iy], inside

    for step in range(args.horizon_steps):
        with torch.inference_mode():
            command_buffer.copy_(fixed_commands)
            obs, _ = env.get_observations()
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            foot_xy = robot.data.body_pos_w[:, robot_foot_ids, :2]
            foot_materials, foot_heights, foot_slopes, foot_inside = sample_surface(foot_xy)
            root_material, root_height, _, root_inside = sample_surface(robot.data.root_pos_w[:, :2])
            del root_material
            sorted_materials, _ = torch.sort(foot_materials, dim=1)
            diversity = 1 + torch.sum(sorted_materials[:, 1:] != sorted_materials[:, :-1], dim=1)
            footprint_height_span = torch.max(foot_heights, dim=1).values - torch.min(foot_heights, dim=1).values
            root_lin = robot.data.root_lin_vel_b[:, :2]
            root_yaw = robot.data.root_ang_vel_b[:, 2]
            torque = robot.data.applied_torque
            joint_vel = robot.data.joint_vel
            quaternion = robot.data.root_quat_w
            w, x, y, z = quaternion.unbind(dim=1)
            roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
            pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
            contact_termination = env.unwrapped.termination_manager.get_term("base_contact").clone()
            body_up_world_z = 1.0 - 2.0 * (x * x + y * y)
            kinematic_fall = (
                ~root_inside
                | ((robot.data.root_pos_w[:, 2] - root_height) < args.kinematic_fall_height_m)
                | (body_up_world_z < args.kinematic_fall_up_axis_min)
            )
            fall = contact_termination | kinematic_fall
            timeout = env.unwrapped.termination_manager.get_term("time_out").clone()
            contact = (
                torch.linalg.vector_norm(contact_sensor.data.net_forces_w[:, sensor_foot_ids], dim=2)
                > args.contact_force_threshold_n
            )
            foot_slip_speed = torch.linalg.vector_norm(
                robot.data.body_lin_vel_w[:, robot_foot_ids, :2], dim=2
            )

            for direction_index, item in enumerate(DIRECTION_COMMANDS):
                group_mask = condition_index == direction_index
                metric_mask = active & ~(fall | timeout) & group_mask
                accumulator = accumulators[item["id"]]
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
                            torch.sum(torch.abs(torque[metric_mask] * joint_vel[metric_mask]), dim=1).sum().item()
                        )
                        accumulator["roll_abs_max"] = max(
                            accumulator["roll_abs_max"], float(torch.abs(roll[metric_mask]).max().item())
                        )
                        accumulator["pitch_abs_max"] = max(
                            accumulator["pitch_abs_max"], float(torch.abs(pitch[metric_mask]).max().item())
                        )
                        material_samples = foot_materials[metric_mask]
                        for bucket in range(len(field.spec.static_friction)):
                            accumulator["friction_bucket_foot_sample_counts"][bucket] += int(
                                (material_samples == bucket).sum().item()
                            )
                        for unique_count in range(1, 5):
                            accumulator["four_foot_diversity_counts"][unique_count] += int(
                                (diversity[metric_mask] == unique_count).sum().item()
                            )
                        if previous_materials is not None:
                            accumulator["material_transition_count"] += int(
                                (material_samples != previous_materials[metric_mask]).sum().item()
                            )
                        spans = footprint_height_span[metric_mask]
                        sampled_heights = foot_heights[metric_mask]
                        sampled_slopes = foot_slopes[metric_mask]
                        accumulator["footprint_height_span_sum_m"] += float(spans.sum().item())
                        accumulator["footprint_height_span_max_m"] = max(
                            accumulator["footprint_height_span_max_m"], float(spans.max().item())
                        )
                        accumulator["terrain_height_min_m"] = min(
                            accumulator["terrain_height_min_m"], float(sampled_heights.min().item())
                        )
                        accumulator["terrain_height_max_m"] = max(
                            accumulator["terrain_height_max_m"], float(sampled_heights.max().item())
                        )
                        accumulator["local_slope_sum_deg"] += float(sampled_slopes.sum().item())
                        accumulator["local_slope_max_deg"] = max(
                            accumulator["local_slope_max_deg"], float(sampled_slopes.max().item())
                        )
                        accumulator["foot_surface_sample_count"] += int(sampled_heights.numel())
                        accumulator["out_of_field_foot_sample_count"] += int(
                            (~foot_inside[metric_mask]).sum().item()
                        )
                        contact_mask = metric_mask[:, None] & contact
                        contact_count = int(contact_mask.sum().item())
                        accumulator["contact_foot_sample_count"] += contact_count
                        if contact_count:
                            speeds = foot_slip_speed[contact_mask]
                            accumulator["contact_slip_speed_sum"] += float(speeds.sum().item())
                            accumulator["contact_slip_over_0_1_count"] += int((speeds > 0.1).sum().item())
                accumulator["fall_count"] += int((fall & active & group_mask).sum().item())
                accumulator["contact_termination_count"] += int(
                    (contact_termination & active & group_mask).sum().item()
                )
                accumulator["kinematic_fall_count"] += int((kinematic_fall & active & group_mask).sum().item())
                accumulator["timeout_count"] += int((timeout & active & group_mask).sum().item())
            active &= ~(fall | timeout)
            previous_materials = foot_materials.clone()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.horizon_steps:
            print(f"[irregular-road] completed_steps={step + 1}", flush=True)

    env.close()
    directions = []
    for item in DIRECTION_COMMANDS:
        command = tuple(float(value) for value in item["command"])
        accumulator = accumulators[item["id"]]
        result = finalize_accumulator(accumulator, command)
        foot_count = int(accumulator["foot_surface_sample_count"])
        contact_count = int(accumulator["contact_foot_sample_count"])
        bucket_counts = accumulator["friction_bucket_foot_sample_counts"]
        result.update(
            {
                "friction_bucket_foot_sample_counts": bucket_counts,
                "friction_bucket_foot_sample_ratios": [
                    None if foot_count == 0 else value / foot_count for value in bucket_counts
                ],
                "four_foot_material_diversity": summarize_diversity_counts(
                    accumulator["four_foot_diversity_counts"]
                ),
                "material_transition_count": int(accumulator["material_transition_count"]),
                "footprint_height_span_mean_m": None
                if result["sample_count"] == 0
                else accumulator["footprint_height_span_sum_m"] / result["sample_count"],
                "footprint_height_span_max_m": float(accumulator["footprint_height_span_max_m"]),
                "terrain_height_min_m": None
                if foot_count == 0
                else float(accumulator["terrain_height_min_m"]),
                "terrain_height_max_m": None
                if foot_count == 0
                else float(accumulator["terrain_height_max_m"]),
                "local_slope_mean_deg": None
                if foot_count == 0
                else accumulator["local_slope_sum_deg"] / foot_count,
                "local_slope_max_deg": float(accumulator["local_slope_max_deg"]),
                "out_of_field_foot_sample_count": int(accumulator["out_of_field_foot_sample_count"]),
                "field_coverage_pass": int(accumulator["out_of_field_foot_sample_count"]) == 0,
                "contact_foot_sample_count": contact_count,
                "contact_observation_available": contact_count > 0,
                "contact_slip_speed_mean_mps": None
                if contact_count == 0
                else accumulator["contact_slip_speed_sum"] / contact_count,
                "contact_slip_over_0_1_ratio": None
                if contact_count == 0
                else accumulator["contact_slip_over_0_1_count"] / contact_count,
                "contact_termination_count": int(accumulator["contact_termination_count"]),
                "kinematic_fall_count": int(accumulator["kinematic_fall_count"]),
            }
        )
        directions.append({"id": item["id"], "command": list(command), **result})

    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "protocol": "irregular_road_spatial_friction_height_v1",
        "task": args.task,
        "policy_id": args.policy_id,
        "checkpoint": {
            "path": portable_path(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
        },
        "headless": bool(args.headless),
        "device": args.device,
        "evaluation_seed": args.eval_seed,
        "terrain_seed": args.terrain_seed,
        "num_envs": args.num_envs,
        "environments_per_direction": args.num_envs // len(DIRECTION_COMMANDS),
        "horizon_steps": args.horizon_steps,
        "warmup_steps": args.warmup_steps,
        "step_dt_s": float(env_cfg.sim.dt * env_cfg.decimation),
        "observation_corruption": False,
        "gate": GATE,
        "all_directions_gate_pass": all(item["gate_pass"] for item in directions),
        "all_directions_field_coverage_pass": all(item["field_coverage_pass"] for item in directions),
        "directions": directions,
        "road_field": field_summary(field),
        "contact_model": {
            "robot_foot_material": [1.0, 1.0, 0.0],
            "combine_mode": "multiply",
            "geometry": "four disjoint static triangle meshes, one per friction bucket; one non-colliding full mesh for height scan",
            "surface_readback": surface_readback,
        },
        "fall_detection": {
            "contact_termination_term": "base_contact",
            "kinematic_base_height_above_sampled_terrain_min_m": args.kinematic_fall_height_m,
            "kinematic_body_up_world_z_min": args.kinematic_fall_up_axis_min,
        },
        "evaluation_source_sha256": file_sha256(Path(__file__)),
        "road_generator_source_sha256": file_sha256(
            REPO_ROOT / "src" / "isaac_walk_g008" / "irregular_road.py"
        ),
        "metric_notes": {
            "four_foot_material_diversity": "각 active frame의 네 발 world XY가 가리키는 재질 버킷 수; 접촉 여부와 무관한 footprint 지표",
            "height": "같은 collision 지형의 bilinear 높이와 cell 국소 경사를 발 위치에서 표본화",
            "material_timing": "모든 재질과 높이는 시뮬레이션 시작 전에 고정되고 접촉 중에는 바뀌지 않음",
            "causal_scope": "단일 terrain/evaluation seed의 시뮬레이션 stress test이며 실물 도로 한계를 뜻하지 않음",
        },
    }


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--task", default="Isaac-G008-Velocity-IrregularRoad-Go2-S1-v0")
    parser.add_argument("--eval-seed", type=int, default=20260826)
    parser.add_argument("--terrain-seed", type=int, default=20260826)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--horizon-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--env-spacing-m", type=float, default=4.0)
    parser.add_argument("--contact-force-threshold-n", type=float, default=1.0)
    parser.add_argument("--kinematic-fall-height-m", type=float, default=0.18)
    parser.add_argument("--kinematic-fall-up-axis-min", type=float, default=0.5)
    parser.add_argument("--hard-exit-after-report", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.env_spacing_m <= 0.0:
        raise ValueError("env_spacing_m must be positive")
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
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "all_directions_gate_pass": report["all_directions_gate_pass"],
                }
            ),
            flush=True,
        )
        if args.hard_exit_after_report:
            os._exit(0)
    finally:
        simulation_app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
