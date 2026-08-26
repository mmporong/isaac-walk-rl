#!/usr/bin/env python3
"""Record a local-only MP4 of the four fixed G008 direction commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

WINDOWS_KIT_ARGS = "--/app/vulkan=false --/app/window/hideUi=true"
SEQUENCE = (
    ("stand", 50, (0.0, 0.0, 0.0)),
    ("forward", 175, (0.60, 0.0, 0.0)),
    ("stand", 50, (0.0, 0.0, 0.0)),
    ("backward", 175, (-0.40, 0.0, 0.0)),
    ("stand", 50, (0.0, 0.0, 0.0)),
    ("left_turn", 175, (0.0, 0.0, 0.50)),
    ("stand", 50, (0.0, 0.0, 0.0)),
    ("right_turn", 175, (0.0, 0.0, -0.50)),
)


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


def command_at_step(step: int) -> tuple[str, tuple[float, float, float]]:
    if step < 0 or step >= sum(length for _, length, _ in SEQUENCE):
        raise IndexError(step)
    cursor = 0
    for name, length, command in SEQUENCE:
        if step < cursor + length:
            return name, command
        cursor += length
    raise AssertionError("unreachable command sequence state")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def record(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from isaac_walk_g008 import register_tasks

    register_tasks()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = 20.0
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass = None
    if hasattr(env_cfg.events, "add_leg_mass"):
        env_cfg.events.add_leg_mass = None
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

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / args.output_name
    if destination.exists():
        raise FileExistsError(destination)

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    controller = env.unwrapped.viewport_camera_controller
    if controller is not None:
        controller.update_view_to_asset_root("robot")
        controller.update_view_location(eye=(4.0, 4.0, 2.6), lookat=(0.0, 0.0, 0.35))
    total_steps = sum(length for _, length, _ in SEQUENCE)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(output_dir),
        step_trigger=lambda step: step == 0,
        video_length=total_steps,
        disable_logger=True,
        name_prefix="g008-directions",
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint.resolve()))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    command_buffer = env.unwrapped.command_manager.get_command("base_velocity")
    obs, _ = env.get_observations()

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
        (path for path in output_dir.glob("g008-directions*.mp4") if path != destination),
        key=lambda path: path.stat().st_mtime,
    )
    if len(candidates) != 1:
        raise RuntimeError(f"expected one recorded MP4, found {len(candidates)}: {candidates}")
    candidates[0].rename(destination)
    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "task": args.task,
        "seed": args.seed,
        "headless": bool(args.headless),
        "step_dt_s": float(env_cfg.sim.dt * env_cfg.decimation),
        "total_steps": total_steps,
        "sequence": [
            {"name": name, "steps": length, "command": list(command)} for name, length, command in SEQUENCE
        ],
        "checkpoint": {
            "path": portable_path(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
        },
        "local_video": {
            "path": portable_path(destination),
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
            "git_policy": "local_only",
        },
        "record_source_sha256": file_sha256(Path(__file__)),
    }


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task", default="Isaac-G008-Velocity-Rough-Go2-CommandSuite-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", default="g008_directions_s42.mp4")
    parser.add_argument("--report", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    if sys.platform == "win32" and args.headless and not args.kit_args:
        args.kit_args = WINDOWS_KIT_ARGS
    return args


def main() -> int:
    from isaaclab.app import AppLauncher

    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = record(args)
        _write_json_atomic(args.report.resolve(), report)
        print(json.dumps({"report": str(args.report.resolve()), "video": report["local_video"]}), flush=True)
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
