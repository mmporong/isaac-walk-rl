#!/usr/bin/env python3
"""Sweep spatially periodic high/low friction stripes with two trained G008 policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable
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


DEFAULT_SWEEP = (
    {"id": "uniform_nominal", "low_static": 0.8, "low_dynamic": 0.6, "mixed": False},
    {"id": "mixed_070_050", "low_static": 0.7, "low_dynamic": 0.5, "mixed": True},
    {"id": "mixed_060_040", "low_static": 0.6, "low_dynamic": 0.4, "mixed": True},
    {"id": "mixed_050_030", "low_static": 0.5, "low_dynamic": 0.3, "mixed": True},
    {"id": "mixed_040_025", "low_static": 0.4, "low_dynamic": 0.25, "mixed": True},
    {"id": "mixed_030_020", "low_static": 0.3, "low_dynamic": 0.2, "mixed": True},
    {"id": "mixed_020_010", "low_static": 0.2, "low_dynamic": 0.1, "mixed": True},
    {"id": "mixed_010_005", "low_static": 0.1, "low_dynamic": 0.05, "mixed": True},
)

GROUND_COLLISION_PRIM = "/World/ground/terrain/GroundPlane/CollisionPlane"
HEIGHT_SCAN_PRIM = "/World/periodic_friction_field/height_scan_surface"


def _write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stripe_is_low(position_m: float, stripe_width_m: float, phase_m: float = 0.0) -> bool:
    """Return whether a one-dimensional position lies in an even low-friction stripe."""
    if stripe_width_m <= 0.0:
        raise ValueError("stripe_width_m must be positive")
    return math.floor((position_m + phase_m) / stripe_width_m) % 2 == 0


def validate_sweep(cases: tuple[dict[str, Any], ...]) -> None:
    if not cases or cases[0]["id"] != "uniform_nominal" or cases[0]["mixed"]:
        raise ValueError("the sweep must start with a uniform nominal baseline")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("friction case ids must be unique")
    for case in cases:
        static = float(case["low_static"])
        dynamic = float(case["low_dynamic"])
        if not (0.0 <= dynamic <= static):
            raise ValueError(f"invalid friction pair: {case}")


def summarize_threshold(cases: list[dict[str, Any]]) -> dict[str, Any]:
    mixed = [case for case in cases if case["mixed"]]
    contiguous_passes = []
    first_failure = None
    for case in mixed:
        if case["all_directions_gate_pass"] and first_failure is None:
            contiguous_passes.append(case)
        elif first_failure is None:
            first_failure = case
    all_passes = [case for case in mixed if case["all_directions_gate_pass"]]
    return {
        "contiguous_pass_floor": None
        if not contiguous_passes
        else {
            "case_id": contiguous_passes[-1]["case_id"],
            "low_static": contiguous_passes[-1]["low_static"],
            "low_dynamic": contiguous_passes[-1]["low_dynamic"],
        },
        "first_failure": None
        if first_failure is None
        else {
            "case_id": first_failure["case_id"],
            "low_static": first_failure["low_static"],
            "low_dynamic": first_failure["low_dynamic"],
            "failed_directions": [
                item["id"] for item in first_failure["directions"] if not item["gate_pass"]
            ],
        },
        "lowest_tested_passing": None
        if not all_passes
        else {
            "case_id": all_passes[-1]["case_id"],
            "low_static": all_passes[-1]["low_static"],
            "low_dynamic": all_passes[-1]["low_dynamic"],
        },
        "monotonic_gate_sequence": all(
            not later["all_directions_gate_pass"]
            for index, case in enumerate(mixed)
            if not case["all_directions_gate_pass"]
            for later in mixed[index + 1 :]
        ),
    }


def _selected_case(case_id: str) -> dict[str, Any]:
    for case in DEFAULT_SWEEP:
        if case["id"] == case_id:
            return dict(case)
    raise ValueError(f"unknown friction case: {case_id}")


def _spawn_periodic_friction_field(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Any:
    """Spawn one static triangle mesh with high- and low-friction face subsets."""
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.stage as stage_utils
    from pxr import UsdGeom, UsdPhysics, UsdShade

    del translation, orientation
    stage = stage_utils.get_current_stage()
    parent = UsdGeom.Xform.Define(stage, prim_path)
    material_specs = {
        "low": (cfg.low_static, cfg.low_dynamic),
        "high": (cfg.high_static, cfg.high_dynamic),
    }
    material_paths = {}
    for material_id, (static_friction, dynamic_friction) in material_specs.items():
        material_path = f"{prim_path}/{material_id}_material"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        material_cfg.func(material_path, material_cfg)
        material_paths[material_id] = material_path

    points: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    face_vertex_counts: list[int] = []
    face_vertex_indices: list[int] = []
    material_faces: dict[str, list[int]] = {"low": [], "high": []}
    for stripe_cell in range(cfg.first_stripe_cell, cfg.last_stripe_cell_exclusive):
        x_min = stripe_cell * cfg.stripe_width_m
        x_max = x_min + cfg.stripe_width_m
        base = len(points)
        points.extend(
            (
                (x_min, -cfg.global_half_width_y_m, cfg.stripe_surface_height_m),
                (x_max, -cfg.global_half_width_y_m, cfg.stripe_surface_height_m),
                (x_max, cfg.global_half_width_y_m, cfg.stripe_surface_height_m),
                (x_min, cfg.global_half_width_y_m, cfg.stripe_surface_height_m),
            )
        )
        normals.extend([(0.0, 0.0, 1.0)] * 4)
        face_vertex_counts.append(4)
        face_vertex_indices.extend((base, base + 1, base + 2, base + 3))
        material_id = "low" if cfg.mixed and stripe_cell % 2 == 0 else "high"
        material_faces[material_id].append(len(face_vertex_counts) - 1)

    mesh_path = f"{prim_path}/friction_stripes"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    sim_utils.define_collision_properties(
        mesh_path,
        sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        stage,
    )
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision_api.CreateApproximationAttr().Set("none")
    for material_id, face_indices in material_faces.items():
        if not face_indices:
            continue
        subset = UsdGeom.Subset.Define(stage, f"{mesh_path}/{material_id}_faces")
        subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
        subset.CreateFamilyNameAttr().Set(UsdShade.Tokens.materialBind)
        subset.CreateIndicesAttr().Set(face_indices)
        binding_api = UsdShade.MaterialBindingAPI.Apply(subset.GetPrim())
        binding_api.Bind(
            UsdShade.Material(stage.GetPrimAtPath(material_paths[material_id])),
            UsdShade.Tokens.weakerThanDescendants,
        )
        binding_targets = subset.GetPrim().GetRelationship("material:binding").GetTargets()
        if material_paths[material_id] not in {target.pathString for target in binding_targets}:
            raise RuntimeError(f"failed to bind material subset: {material_paths[material_id]}")
    height_scan_mesh = UsdGeom.Mesh.Define(stage, HEIGHT_SCAN_PRIM)
    field_x_min = cfg.first_stripe_cell * cfg.stripe_width_m
    field_x_max = cfg.last_stripe_cell_exclusive * cfg.stripe_width_m
    height_scan_mesh.CreatePointsAttr(
        [
            (field_x_min, -cfg.global_half_width_y_m, cfg.stripe_surface_height_m),
            (field_x_max, -cfg.global_half_width_y_m, cfg.stripe_surface_height_m),
            (field_x_max, cfg.global_half_width_y_m, cfg.stripe_surface_height_m),
            (field_x_min, cfg.global_half_width_y_m, cfg.stripe_surface_height_m),
        ]
    )
    height_scan_mesh.CreateFaceVertexCountsAttr([3, 3])
    height_scan_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    height_scan_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    height_scan_mesh.CreateDoubleSidedAttr(False)
    return parent.GetPrim()


def _configure_env(args: argparse.Namespace, case: dict[str, Any]) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils import configclass

    @configclass
    class PeriodicFrictionFieldCfg(SpawnerCfg):
        func: Callable = _spawn_periodic_friction_field
        stripe_width_m: float = args.stripe_width_m
        stripe_surface_height_m: float = args.stripe_surface_height_m
        global_half_width_y_m: float = 0.0
        first_stripe_cell: int = 0
        last_stripe_cell_exclusive: int = 0
        mixed: bool = bool(case["mixed"])
        low_static: float = float(case["low_static"])
        low_dynamic: float = float(case["low_dynamic"])
        high_static: float = args.high_static
        high_dynamic: float = args.high_dynamic

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.eval_seed
    env_cfg.scene.env_spacing = args.env_spacing_m
    env_cfg.scene.terrain = None
    env_cfg.scene.height_scanner.mesh_prim_paths = [HEIGHT_SCAN_PRIM]
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
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
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

    num_per_row = int(math.sqrt(args.num_envs))
    num_rows = math.ceil(args.num_envs / num_per_row)
    num_cols = math.ceil(args.num_envs / num_rows)
    max_origin_x = 0.5 * args.env_spacing_m * (num_rows - 1)
    max_origin_y = 0.5 * args.env_spacing_m * (num_cols - 1)
    local_half_length = 0.5 * args.stripe_count * args.stripe_width_m
    first_cell = math.floor((-max_origin_x - local_half_length) / args.stripe_width_m)
    last_cell_exclusive = math.ceil((max_origin_x + local_half_length) / args.stripe_width_m)
    field_cfg = PeriodicFrictionFieldCfg(
        global_half_width_y_m=max_origin_y + 0.5 * args.stripe_width_y_m,
        first_stripe_cell=first_cell,
        last_stripe_cell_exclusive=last_cell_exclusive,
    )
    env_cfg.scene.periodic_friction_field = AssetBaseCfg(
        prim_path="/World/periodic_friction_field",
        spawn=field_cfg,
        collision_group=-1,
    )
    return env_cfg


def _verify_collision_surface_isolation() -> dict[str, Any]:
    """Prove that no default plane collider exists and the height-scan mesh is non-colliding."""
    import isaacsim.core.utils.stage as stage_utils
    from pxr import UsdPhysics

    stage = stage_utils.get_current_stage()
    ground_collision_exists = stage.GetPrimAtPath(GROUND_COLLISION_PRIM).IsValid()
    height_scan_prim = stage.GetPrimAtPath(HEIGHT_SCAN_PRIM)
    height_scan_collision_enabled = bool(UsdPhysics.CollisionAPI(height_scan_prim))
    if ground_collision_exists or not height_scan_prim.IsValid() or height_scan_collision_enabled:
        raise RuntimeError(
            "collision-surface isolation failed: "
            f"ground_collision_exists={ground_collision_exists}, "
            f"height_scan_valid={height_scan_prim.IsValid()}, "
            f"height_scan_has_collision_api={height_scan_collision_enabled}"
        )
    return {
        "default_ground_collision_prim": GROUND_COLLISION_PRIM,
        "default_ground_collision_exists": ground_collision_exists,
        "height_scan_prim": HEIGHT_SCAN_PRIM,
        "height_scan_has_collision_api": height_scan_collision_enabled,
    }


def _validate_field_phase(env: Any, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    origin_cells = env.unwrapped.scene.env_origins[:, 0] / args.stripe_width_m
    rounded_cells = torch.round(origin_cells)
    integral_error = float(torch.max(torch.abs(origin_cells - rounded_cells)).item())
    even_cell_phase = bool(torch.all(torch.remainder(rounded_cells.to(torch.int64), 2) == 0).item())
    if integral_error > 1.0e-5 or not even_cell_phase:
        raise RuntimeError(
            "environment origins do not preserve the local periodic-friction phase: "
            f"integral_error={integral_error}, even_cell_phase={even_cell_phase}"
        )
    return {
        "environment_origin_cell_integral_error_max": integral_error,
        "all_environment_origins_on_even_period_cells": even_cell_phase,
    }


def _new_extended_accumulator(trial_count: int) -> dict[str, float | int]:
    accumulator = new_accumulator(trial_count)
    accumulator.update(
        {
            "contact_foot_sample_count": 0,
            "contact_slip_speed_sum": 0.0,
            "contact_slip_over_0_1_count": 0,
            "low_zone_foot_sample_count": 0,
            "active_foot_sample_count": 0,
            "stripe_transition_count": 0,
            "local_foot_abs_x_max_m": 0.0,
            "local_foot_abs_y_max_m": 0.0,
            "contact_termination_count": 0,
            "kinematic_fall_count": 0,
        }
    )
    return accumulator


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaac_walk_g008 import register_tasks

    validate_sweep(DEFAULT_SWEEP)
    policy_count = 2
    condition_count = len(DIRECTION_COMMANDS)
    if args.num_envs <= 0 or args.num_envs % (condition_count * policy_count) != 0:
        raise ValueError("num_envs must be a positive multiple of eight")
    if args.warmup_steps < 0 or args.warmup_steps >= args.horizon_steps:
        raise ValueError("warmup_steps must be in [0, horizon_steps)")

    register_tasks()
    case = _selected_case(args.case_id)
    env_cfg = _configure_env(args, case)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.eval_seed
    agent_cfg.device = args.device
    print("[periodic-friction] creating environment", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    underlay = _verify_collision_surface_isolation()
    field_phase = _validate_field_phase(env, args)
    print(
        "[periodic-friction] verified isolated collision field "
        f"ground_exists={underlay['default_ground_collision_exists']} "
        f"height_scan_collision={underlay['height_scan_has_collision_api']}",
        flush=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print("[periodic-friction] environment ready", flush=True)

    runners = {}
    policies = {}
    checkpoints = {
        "command": args.command_checkpoint.resolve(),
        "friction_s1": args.friction_checkpoint.resolve(),
    }
    for policy_id, checkpoint in checkpoints.items():
        print(f"[periodic-friction] loading policy={policy_id}", flush=True)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(checkpoint))
        runners[policy_id] = runner
        policies[policy_id] = runner.get_inference_policy(device=env.unwrapped.device)
        print(f"[periodic-friction] loaded policy={policy_id}", flush=True)

    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"]
    robot_foot_ids, robot_foot_names = robot.find_bodies(".*_foot")
    sensor_foot_ids, sensor_foot_names = contact_sensor.find_bodies(".*_foot")
    if robot_foot_names != sensor_foot_names:
        raise RuntimeError("robot and contact-sensor foot ordering differ")
    command_buffer = env.unwrapped.command_manager.get_command("base_velocity")
    env_index = torch.arange(args.num_envs, device=env.unwrapped.device)
    condition_index = env_index % condition_count
    policy_index = torch.div(env_index, condition_count, rounding_mode="floor") % policy_count
    policy_ids = tuple(policies)
    fixed_commands = torch.tensor(
        [DIRECTION_COMMANDS[index]["command"] for index in condition_index.cpu().tolist()],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    env_origin_x = env.unwrapped.scene.env_origins[:, 0]
    env_origin_y = env.unwrapped.scene.env_origins[:, 1]
    env_origin_z = env.unwrapped.scene.env_origins[:, 2]
    accumulators = {
        policy_id: {
            item["id"]: _new_extended_accumulator(
                int(((policy_index == policy_number) & (condition_index == direction_index)).sum().item())
            )
            for direction_index, item in enumerate(DIRECTION_COMMANDS)
        }
        for policy_number, policy_id in enumerate(policy_ids)
    }
    active = torch.ones(args.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    previous_low_mask = None
    env.reset()
    obs, _ = env.get_observations()
    print(f"[periodic-friction] running paired policies case={case['id']}", flush=True)

    for step in range(args.horizon_steps):
        with torch.inference_mode():
            command_buffer.copy_(fixed_commands)
            obs, _ = env.get_observations()
            actions = torch.zeros((args.num_envs, env.num_actions), device=env.unwrapped.device)
            for policy_number, policy_id in enumerate(policy_ids):
                mask = (policy_index == policy_number) & active
                actions[mask] = policies[policy_id](obs[mask])
            obs, _, _, _ = env.step(actions)

            local_foot_x = robot.data.body_pos_w[:, robot_foot_ids, 0] - env_origin_x[:, None]
            local_foot_y = robot.data.body_pos_w[:, robot_foot_ids, 1] - env_origin_y[:, None]
            if case["mixed"]:
                low_mask = torch.remainder(
                    torch.floor(local_foot_x / args.stripe_width_m).to(torch.int64), 2
                ) == 0
            else:
                low_mask = torch.zeros_like(local_foot_x, dtype=torch.bool)
            root_lin = robot.data.root_lin_vel_b[:, :2]
            root_yaw = robot.data.root_ang_vel_b[:, 2]
            torque = robot.data.applied_torque
            joint_vel = robot.data.joint_vel
            quaternion = robot.data.root_quat_w
            w, x, y, z = quaternion.unbind(dim=1)
            roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
            pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
            contact_termination = env.unwrapped.termination_manager.get_term("base_contact").clone()
            base_height_above_surface = (
                robot.data.root_pos_w[:, 2] - env_origin_z - args.stripe_surface_height_m
            )
            body_up_world_z = 1.0 - 2.0 * (x * x + y * y)
            kinematic_fall = (
                (base_height_above_surface < args.kinematic_fall_height_m)
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

            for policy_number, policy_id in enumerate(policy_ids):
                for direction_index, item in enumerate(DIRECTION_COMMANDS):
                    group_mask = (policy_index == policy_number) & (condition_index == direction_index)
                    metric_mask = active & ~(fall | timeout) & group_mask
                    accumulator = accumulators[policy_id][item["id"]]
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
                                torch.sum(torch.abs(torque[metric_mask] * joint_vel[metric_mask]), dim=1)
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
                            contact_mask = metric_mask[:, None] & contact
                            contact_count = int(contact_mask.sum().item())
                            accumulator["contact_foot_sample_count"] += contact_count
                            if contact_count:
                                speeds = foot_slip_speed[contact_mask]
                                accumulator["contact_slip_speed_sum"] += float(speeds.sum().item())
                                accumulator["contact_slip_over_0_1_count"] += int(
                                    (speeds > 0.1).sum().item()
                                )
                            accumulator["active_foot_sample_count"] += count * len(robot_foot_ids)
                            accumulator["low_zone_foot_sample_count"] += int(
                                low_mask[metric_mask].sum().item()
                            )
                            accumulator["local_foot_abs_x_max_m"] = max(
                                float(accumulator["local_foot_abs_x_max_m"]),
                                float(torch.abs(local_foot_x[metric_mask]).max().item()),
                            )
                            accumulator["local_foot_abs_y_max_m"] = max(
                                float(accumulator["local_foot_abs_y_max_m"]),
                                float(torch.abs(local_foot_y[metric_mask]).max().item()),
                            )
                            if previous_low_mask is not None:
                                accumulator["stripe_transition_count"] += int(
                                    (low_mask[metric_mask] != previous_low_mask[metric_mask]).sum().item()
                                )
                    accumulator["fall_count"] += int((fall & active & group_mask).sum().item())
                    accumulator["contact_termination_count"] += int(
                        (contact_termination & active & group_mask).sum().item()
                    )
                    accumulator["kinematic_fall_count"] += int(
                        (kinematic_fall & active & group_mask).sum().item()
                    )
                    accumulator["timeout_count"] += int((timeout & active & group_mask).sum().item())
            active &= ~(fall | timeout)
        previous_low_mask = low_mask.clone()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.horizon_steps:
            print(f"[periodic-friction] completed_steps={step + 1}", flush=True)

    policy_results = []
    for policy_id in policy_ids:
        directions = []
        for item in DIRECTION_COMMANDS:
            command = tuple(float(value) for value in item["command"])
            accumulator = accumulators[policy_id][item["id"]]
            result = finalize_accumulator(accumulator, command)
            contact_count = int(accumulator["contact_foot_sample_count"])
            active_foot_count = int(accumulator["active_foot_sample_count"])
            result.update(
                {
                    "contact_slip_speed_mean_mps": None
                    if contact_count == 0
                    else float(accumulator["contact_slip_speed_sum"]) / contact_count,
                    "contact_slip_over_0_1_ratio": None
                    if contact_count == 0
                    else int(accumulator["contact_slip_over_0_1_count"]) / contact_count,
                    "low_zone_foot_exposure_ratio": None
                    if active_foot_count == 0
                    else int(accumulator["low_zone_foot_sample_count"]) / active_foot_count,
                    "stripe_transition_count": int(accumulator["stripe_transition_count"]),
                    "local_foot_abs_x_max_m": float(accumulator["local_foot_abs_x_max_m"]),
                    "local_foot_abs_y_max_m": float(accumulator["local_foot_abs_y_max_m"]),
                    "minimum_local_field_coverage_pass": (
                        float(accumulator["local_foot_abs_x_max_m"])
                        <= 0.5 * args.stripe_count * args.stripe_width_m
                        and float(accumulator["local_foot_abs_y_max_m"])
                        <= 0.5 * args.stripe_width_y_m
                    ),
                    "contact_foot_sample_count": contact_count,
                    "contact_observation_available": contact_count > 0,
                    "contact_termination_count": int(accumulator["contact_termination_count"]),
                    "kinematic_fall_count": int(accumulator["kinematic_fall_count"]),
                }
            )
            directions.append({"id": item["id"], "command": list(command), **result})
        policy_results.append(
            {
                "policy_id": policy_id,
                "checkpoint": {
                    "path": portable_path(checkpoints[policy_id]),
                    "sha256": file_sha256(checkpoints[policy_id]),
                },
                "case": {
                    "case_id": case["id"],
                    "mixed": bool(case["mixed"]),
                    "high_static": args.high_static,
                    "high_dynamic": args.high_dynamic,
                    "low_static": float(case["low_static"]),
                    "low_dynamic": float(case["low_dynamic"]),
                    "all_directions_gate_pass": all(item["gate_pass"] for item in directions),
                    "all_directions_field_coverage_pass": all(
                        item["minimum_local_field_coverage_pass"] for item in directions
                    ),
                    "directions": directions,
                },
            }
        )

    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "protocol": "spatial_periodic_friction_stripes_paired_policy_case_v1",
        "task": args.task,
        "terrain_mode": "periodic_static_mesh_without_collision_underlay",
        "contact_model": {
            "field_axis": "environment_local_x",
            "stripe_width_m": args.stripe_width_m,
            "period_m": 2.0 * args.stripe_width_m,
            "minimum_local_stripe_count": args.stripe_count,
            "minimum_local_field_length_m": args.stripe_count * args.stripe_width_m,
            "stripe_width_y_m": args.stripe_width_y_m,
            "stripe_surface_height_m": args.stripe_surface_height_m,
            "environment_spacing_m": args.env_spacing_m,
            "global_stripe_count": (
                env_cfg.scene.periodic_friction_field.spawn.last_stripe_cell_exclusive
                - env_cfg.scene.periodic_friction_field.spawn.first_stripe_cell
            ),
            "global_field_x_bounds_m": [
                env_cfg.scene.periodic_friction_field.spawn.first_stripe_cell * args.stripe_width_m,
                env_cfg.scene.periodic_friction_field.spawn.last_stripe_cell_exclusive
                * args.stripe_width_m,
            ],
            "global_field_half_width_y_m": (
                env_cfg.scene.periodic_friction_field.spawn.global_half_width_y_m
            ),
            "robot_material": [1.0, 1.0, 0.0],
            "combine_mode": "multiply",
            "high_material": [args.high_static, args.high_dynamic, 0.0],
            "low_material": [float(case["low_static"]), float(case["low_dynamic"]), 0.0],
            "geometry": "one pre-spawned static triangle mesh with alternating coplanar face-material subsets and MeshCollisionAPI approximation none",
            "field_phase": field_phase,
            "underlay": underlay,
        },
        "headless": bool(args.headless),
        "device": args.device,
        "evaluation_seed": args.eval_seed,
        "num_envs": args.num_envs,
        "environments_per_policy": args.num_envs // policy_count,
        "environments_per_policy_direction": args.num_envs // (policy_count * len(DIRECTION_COMMANDS)),
        "horizon_steps": args.horizon_steps,
        "warmup_steps": args.warmup_steps,
        "step_dt_s": float(env_cfg.sim.dt * env_cfg.decimation),
        "observation_corruption": False,
        "gate": GATE,
        "fall_detection": {
            "contact_termination_term": "base_contact",
            "kinematic_base_height_min_m": args.kinematic_fall_height_m,
            "kinematic_body_up_world_z_min": args.kinematic_fall_up_axis_min,
            "combined_rule": "contact termination OR base height below threshold OR body up-axis z below threshold",
        },
        "case": case,
        "policies": policy_results,
        "evaluation_source_sha256": file_sha256(Path(__file__)),
        "direction_evaluator_source_sha256": file_sha256(REPO_ROOT / "scripts" / "evaluate_g008_directions.py"),
        "metric_notes": {
            "friction_equivalence": "unit robot material multiplied by each static stripe material",
            "material_timing": "stripe materials are fixed before simulation starts and never changed during contact",
            "underlay_isolation": "the default ground plane is not spawned; a separate non-colliding height-scan mesh is created beside the friction collision meshes before simulation starts",
            "field_coverage": "each direction records whether every active foot stayed inside the minimum 24 m by 4.0 m local coverage guaranteed around every environment origin",
            "slip": "foot horizontal world speed while net contact force exceeds the configured threshold",
            "contact_sensor_limitation": "slip metrics remain null when the robot contact sensor reports no foot-force samples against the multi-material triangle mesh",
            "survival": "fall_count combines the task contact termination with independent base-height and body-up-axis checks",
            "threshold": "contiguous pass floor is conservative and does not extrapolate below tested pairs",
            "causal_scope": "single seed simulation stress test; not a real-floor coefficient guarantee",
        },
    }


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-checkpoint", required=True, type=Path)
    parser.add_argument("--friction-checkpoint", required=True, type=Path)
    parser.add_argument("--task", default="Isaac-G008-Velocity-Rough-Go2-CommandSuite-v0")
    parser.add_argument("--eval-seed", type=int, default=20260826)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--horizon-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--case-id", choices=tuple(case["id"] for case in DEFAULT_SWEEP), required=True)
    parser.add_argument("--stripe-width-m", type=float, default=0.5)
    parser.add_argument("--stripe-count", type=int, default=48)
    parser.add_argument("--stripe-width-y-m", type=float, default=4.0)
    parser.add_argument("--stripe-surface-height-m", type=float, default=0.002)
    parser.add_argument("--env-spacing-m", type=float, default=16.0)
    parser.add_argument("--high-static", type=float, default=0.8)
    parser.add_argument("--high-dynamic", type=float, default=0.6)
    parser.add_argument("--contact-force-threshold-n", type=float, default=1.0)
    parser.add_argument("--kinematic-fall-height-m", type=float, default=0.18)
    parser.add_argument("--kinematic-fall-up-axis-min", type=float, default=0.5)
    parser.add_argument(
        "--hard-exit-after-report",
        action="store_true",
        help="Exit after the atomic report write when Isaac Sim headless teardown stalls on Windows.",
    )
    parser.add_argument("--output", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    for checkpoint in (args.command_checkpoint, args.friction_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if args.stripe_width_m <= 0.0 or args.stripe_width_y_m <= 0.0:
        raise ValueError("stripe dimensions must be positive")
    if args.stripe_surface_height_m < 0.0:
        raise ValueError("stripe_surface_height_m must be non-negative")
    if args.stripe_count <= 0 or args.stripe_count % 2:
        raise ValueError("stripe_count must be a positive even integer")
    if args.env_spacing_m <= 0.0:
        raise ValueError("env_spacing_m must be positive")
    if args.kinematic_fall_height_m <= 0.0:
        raise ValueError("kinematic_fall_height_m must be positive")
    if not (-1.0 <= args.kinematic_fall_up_axis_min <= 1.0):
        raise ValueError("kinematic_fall_up_axis_min must be in [-1, 1]")
    phase_multiple = args.env_spacing_m / (4.0 * args.stripe_width_m)
    if not math.isclose(phase_multiple, round(phase_multiple), abs_tol=1.0e-9):
        raise ValueError("env_spacing_m must preserve stripe phase at centered grid origins")
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
