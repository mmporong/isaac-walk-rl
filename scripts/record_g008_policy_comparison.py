#!/usr/bin/env python3
"""Record three local-only G008 policy videos under matched plane conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for import_root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from record_g008_directions import (  # noqa: E402
    SEQUENCE,
    WINDOWS_KIT_ARGS,
    command_at_step,
    file_sha256,
    portable_path,
)


@dataclass(frozen=True)
class CaptureProfile:
    profile_id: str
    label: str
    task: str
    output_name: str
    domain_mode: str


CAPTURE_PROFILES = (
    CaptureProfile(
        profile_id="command",
        label="Command baseline",
        task="Isaac-G008-Velocity-Rough-Go2-CommandSuite-v0",
        output_name="g008_policy_command_s42.mp4",
        domain_mode="nominal",
    ),
    CaptureProfile(
        profile_id="friction_s1",
        label="Friction S1",
        task="Isaac-G008-Velocity-Rough-Go2-Friction-S1-v0",
        output_name="g008_policy_friction_s1_s42.mp4",
        domain_mode="randomized",
    ),
    CaptureProfile(
        profile_id="leg_mass_s1",
        label="Leg mass S1",
        task="Isaac-G008-Velocity-Rough-Go2-LegMass-S1-v0",
        output_name="g008_policy_leg_mass_s1_s42.mp4",
        domain_mode="randomized",
    ),
)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tensor_statistics(values: Any) -> dict[str, float | int]:
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "mean": float(values.float().mean().item()),
        "max": float(values.max().item()),
    }


def _runtime_domain(environment: Any) -> dict[str, Any]:
    import torch

    robot = environment.scene["robot"]
    masses = robot.root_physx_view.get_masses().clone().cpu()
    default_masses = robot.data.default_mass.clone().cpu()
    leg_ids, leg_names = robot.find_bodies(".*_(hip|thigh|calf|foot)")
    foot_ids, foot_names = robot.find_bodies(".*_foot")
    leg_ids_cpu = torch.tensor(leg_ids, dtype=torch.long)
    mass_ratios = masses[:, leg_ids_cpu] / default_masses[:, leg_ids_cpu]

    materials = robot.root_physx_view.get_material_properties().clone().cpu()
    shapes_per_body = []
    for link_path in robot.root_physx_view.link_paths[0]:
        link_view = robot._physics_sim_view.create_rigid_body_view(link_path)
        shapes_per_body.append(link_view.max_shapes)
    foot_shape_ids = []
    for body_id in foot_ids:
        start = sum(shapes_per_body[:body_id])
        foot_shape_ids.extend(range(start, start + shapes_per_body[body_id]))
    foot_materials = materials[:, foot_shape_ids]

    return {
        "leg_body_names": list(leg_names),
        "foot_body_names": list(foot_names),
        "leg_mass_scale": _tensor_statistics(mass_ratios),
        "sampled_total_leg_mass_kg": _tensor_statistics(masses[:, leg_ids_cpu].sum(dim=1)),
        "foot_static_friction": _tensor_statistics(foot_materials[..., 0]),
        "foot_dynamic_friction": _tensor_statistics(foot_materials[..., 1]),
        "foot_restitution": _tensor_statistics(foot_materials[..., 2]),
    }


def _configure_environment(args: argparse.Namespace, profile: CaptureProfile) -> Any:
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(profile.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    if hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None
    env_cfg.episode_length_s = 20.0
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass = None
    if profile.profile_id != "leg_mass_s1" and hasattr(env_cfg.events, "add_leg_mass"):
        env_cfg.events.add_leg_mass = None
    if profile.profile_id != "friction_s1":
        env_cfg.events.physics_material.params.update(
            {
                "static_friction_range": (0.8, 0.8),
                "dynamic_friction_range": (0.6, 0.6),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
                "make_consistent": True,
            }
        )
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (4.0, 4.0, 2.6)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.35)
    return env_cfg


def _record_profile(args: argparse.Namespace, profile: CaptureProfile) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry

    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    destination = output_dir / profile.output_name
    if destination.exists():
        raise FileExistsError(destination)

    env_cfg = _configure_environment(args, profile)
    agent_cfg = load_cfg_from_registry(profile.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device

    raw_env = gym.make(profile.task, cfg=env_cfg, render_mode="rgb_array")
    controller = raw_env.unwrapped.viewport_camera_controller
    if controller is not None:
        controller.update_view_to_asset_root("robot")
        controller.update_view_location(eye=(4.0, 4.0, 2.6), lookat=(0.0, 0.0, 0.35))

    prefix = f"g008-{profile.profile_id}"
    total_steps = sum(length for _, length, _ in SEQUENCE)
    recorded_env = gym.wrappers.RecordVideo(
        raw_env,
        video_folder=str(output_dir),
        step_trigger=lambda step: step == 0,
        video_length=total_steps,
        disable_logger=True,
        name_prefix=prefix,
    )
    env = RslRlVecEnvWrapper(recorded_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    command_buffer = env.unwrapped.command_manager.get_command("base_velocity")
    obs, _ = env.get_observations()
    runtime_domain = _runtime_domain(env.unwrapped)

    try:
        for step in range(total_steps):
            _, command = command_at_step(step)
            fixed_command = torch.tensor(command, dtype=torch.float32, device=env.unwrapped.device)
            command_buffer[0].copy_(fixed_command)
            obs, _ = env.get_observations()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
    finally:
        env.close()

    candidates = sorted(
        output_dir.glob(f"{prefix}*.mp4"),
        key=lambda path: path.stat().st_mtime,
    )
    if len(candidates) != 1:
        raise RuntimeError(f"expected one recorded MP4 for {profile.profile_id}, found {candidates}")
    candidates[0].rename(destination)
    return {
        "profile_id": profile.profile_id,
        "label": profile.label,
        "task": profile.task,
        "domain_mode": profile.domain_mode,
        "checkpoint": {
            "path": portable_path(checkpoint),
            "sha256": file_sha256(checkpoint),
        },
        "runtime_domain": runtime_domain,
        "local_video": {
            "path": portable_path(destination),
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
            "git_policy": "local_only",
        },
    }


def record_capture(args: argparse.Namespace) -> dict[str, Any]:
    import isaaclab_tasks  # noqa: F401

    from isaac_walk_g008 import register_tasks

    register_tasks()
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    profile = next(item for item in CAPTURE_PROFILES if item.profile_id == args.profile)
    capture = _record_profile(args, profile)
    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "purpose": "one isolated process capture for the matched three-policy comparison",
        "seed": args.seed,
        "headless": bool(args.headless),
        "terrain_mode": "plane",
        "step_dt_s": 0.02,
        "total_steps_per_profile": sum(length for _, length, _ in SEQUENCE),
        "sequence": [
            {"name": name, "steps": length, "command": list(command)}
            for name, length, command in SEQUENCE
        ],
        "profile": capture,
        "record_source_sha256": file_sha256(Path(__file__)),
    }


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=tuple(item.profile_id for item in CAPTURE_PROFILES))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    if sys.platform == "win32" and args.headless and not args.kit_args:
        args.kit_args = WINDOWS_KIT_ARGS
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    return args


def main() -> int:
    from isaaclab.app import AppLauncher

    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = record_capture(args)
        _write_json_atomic(args.report.resolve(), report)
        print(json.dumps({"report": str(args.report.resolve()), "status": report["status"]}), flush=True)
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
