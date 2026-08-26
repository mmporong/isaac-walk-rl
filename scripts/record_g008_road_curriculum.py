#!/usr/bin/env python3
"""Record one G008 road/reward curriculum checkpoint to a local-only MP4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluate_g008_irregular_road import _configure_env, _verify_collision_surface  # noqa: E402
from isaac_walk_g008.irregular_road import field_summary  # noqa: E402
from record_g008_directions import (  # noqa: E402
    SEQUENCE,
    WINDOWS_KIT_ARGS,
    command_at_step,
    file_sha256,
    portable_path,
)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaac_walk_g008 import register_tasks

    register_tasks()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / args.output_name
    if destination.exists():
        raise FileExistsError(destination)

    total_steps = sum(length for _, length, _ in SEQUENCE)
    eval_args = SimpleNamespace(
        task=args.task,
        device=args.device,
        num_envs=1,
        eval_seed=args.seed,
        terrain_seed=args.terrain_seed,
        env_spacing_m=4.0,
        horizon_steps=total_steps,
    )
    env_cfg, field = _configure_env(eval_args)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (3.0, 3.0, 1.8)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.30)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device

    raw_env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    surface_readback = _verify_collision_surface(field)
    controller = raw_env.unwrapped.viewport_camera_controller
    if controller is not None:
        controller.update_view_to_asset_root("robot")
        controller.update_view_location(eye=(3.0, 3.0, 1.8), lookat=(0.0, 0.0, 0.30))
    prefix = f"g008-{args.profile_id}"
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
    try:
        for step in range(total_steps):
            _, command = command_at_step(step)
            command_buffer[0].copy_(
                torch.tensor(command, dtype=torch.float32, device=env.unwrapped.device)
            )
            obs, _ = env.get_observations()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
    finally:
        env.close()

    candidates = sorted(output_dir.glob(f"{prefix}*.mp4"), key=lambda path: path.stat().st_mtime)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one MP4 for {args.profile_id}, found {candidates}")
    candidates[0].rename(destination)

    quantitative_path = args.quantitative_report.resolve()
    reward_contract_path = args.reward_contract.resolve()
    for required in (quantitative_path, reward_contract_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "purpose": "road geometry and reward-ablation visual evidence",
        "profile": {
            "profile_id": args.profile_id,
            "label": args.label,
            "stage": "road_reward_curriculum",
            "task": args.task,
            "seed": args.seed,
            "terrain_seed": args.terrain_seed,
            "headless": bool(args.headless),
            "step_dt_s": float(env_cfg.sim.dt * env_cfg.decimation),
            "total_steps": total_steps,
            "sequence": [
                {"name": name, "steps": length, "command": list(command)}
                for name, length, command in SEQUENCE
            ],
            "checkpoint": {"path": portable_path(checkpoint), "sha256": file_sha256(checkpoint)},
            "stage_physics": {
                "field": field_summary(field),
                "surface_readback": surface_readback,
            },
            "quantitative_report": {
                "path": portable_path(quantitative_path),
                "sha256": file_sha256(quantitative_path),
            },
            "reward_contract": {
                "path": portable_path(reward_contract_path),
                "sha256": file_sha256(reward_contract_path),
            },
            "local_video": {
                "path": portable_path(destination),
                "sha256": file_sha256(destination),
                "bytes": destination.stat().st_size,
                "git_policy": "local_only",
            },
        },
        "record_source_sha256": file_sha256(Path(__file__)),
        "evaluator_source_sha256": file_sha256(REPO_ROOT / "scripts" / "evaluate_g008_irregular_road.py"),
        "road_generator_source_sha256": file_sha256(REPO_ROOT / "src" / "isaac_walk_g008" / "irregular_road.py"),
        "reward_variant_source_sha256": file_sha256(REPO_ROOT / "src" / "isaac_walk_g008" / "rewards.py"),
    }


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--quantitative-report", required=True, type=Path)
    parser.add_argument("--reward-contract", required=True, type=Path)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--terrain-seed", type=int, default=20260826)
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
    simulation_app = AppLauncher(args).app
    try:
        report = _record(args)
        _write_json_atomic(args.report.resolve(), report)
        print(json.dumps({"report": str(args.report.resolve()), "video": report["profile"]["local_video"]}))
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
