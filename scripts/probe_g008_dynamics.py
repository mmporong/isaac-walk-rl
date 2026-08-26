#!/usr/bin/env python3
"""Capture runtime material, mass, and inertia evidence for one G008 task."""

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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _statistics(values: Any) -> dict[str, float | int]:
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "mean": float(values.float().mean().item()),
        "max": float(values.max().item()),
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def probe(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaac_walk_g008 import register_tasks
    from isaac_walk_g008.contracts import COMMAND_PRIMITIVES, GO2_LEG_BODY_PATTERN

    register_tasks()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    try:
        env.reset()
        command_buffer = env.unwrapped.command_manager.get_command("base_velocity").detach().cpu()
        exact_command_counts = {}
        exact_any = torch.zeros(args.num_envs, dtype=torch.bool)
        for primitive in COMMAND_PRIMITIVES:
            target = torch.tensor(primitive.velocity_mps_radps, dtype=command_buffer.dtype)
            matches = torch.all(torch.isclose(command_buffer, target, atol=1.0e-6, rtol=0.0), dim=1)
            exact_command_counts[primitive.name] = int(matches.sum().item())
            exact_any |= matches

        masses = robot.root_physx_view.get_masses().clone().cpu()
        default_masses = robot.data.default_mass.clone().cpu()
        leg_ids, leg_names = robot.find_bodies(GO2_LEG_BODY_PATTERN)
        foot_ids, foot_names = robot.find_bodies(".*_foot")
        leg_ids_cpu = torch.tensor(leg_ids, dtype=torch.long)
        mass_ratios = masses[:, leg_ids_cpu] / default_masses[:, leg_ids_cpu]

        inertias = robot.root_physx_view.get_inertias().clone().cpu()
        default_inertias = robot.data.default_inertia.clone().cpu()
        expected_inertias = default_inertias[:, leg_ids_cpu] * mass_ratios.unsqueeze(-1)
        inertia_error = torch.abs(inertias[:, leg_ids_cpu] - expected_inertias)

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
            "schema_version": 1,
            "goal": "G008",
            "status": "complete",
            "task": args.task,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "device": args.device,
            "body_names": list(robot.body_names),
            "leg_body_names": list(leg_names),
            "foot_body_names": list(foot_names),
            "leg_body_count": len(leg_ids),
            "foot_shape_count_per_environment": len(foot_shape_ids),
            "command_sample": {
                "exact_primitive_counts": exact_command_counts,
                "continuous_or_non_exact_count": int((~exact_any).sum().item()),
                "total_count": args.num_envs,
            },
            "mass": {
                "leg_mass_scale": _statistics(mass_ratios),
                "nominal_total_leg_mass_kg": float(default_masses[0, leg_ids_cpu].sum().item()),
                "sampled_total_leg_mass_kg": _statistics(masses[:, leg_ids_cpu].sum(dim=1)),
            },
            "inertia": {
                "recomputed_from_nominal_by_mass_ratio": bool(float(inertia_error.max().item()) <= 1.0e-6),
                "absolute_error_max": float(inertia_error.max().item()),
            },
            "foot_material": {
                "static_friction": _statistics(foot_materials[..., 0]),
                "dynamic_friction": _statistics(foot_materials[..., 1]),
                "restitution": _statistics(foot_materials[..., 2]),
                "dynamic_not_greater_than_static": bool(
                    torch.all(foot_materials[..., 1] <= foot_materials[..., 0]).item()
                ),
            },
            "probe_source_sha256": file_sha256(Path(__file__)),
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--output", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    from isaaclab.app import AppLauncher

    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = probe(args)
        _write_json_atomic(args.output.resolve(), report)
        print(json.dumps({"output": str(args.output.resolve()), "status": report["status"]}), flush=True)
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
