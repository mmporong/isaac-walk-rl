#!/usr/bin/env python3
"""Record local-only G008 videos whenever a dynamics experiment stage changes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for import_root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluate_g008_link_mass_sensitivity import (  # noqa: E402
    LINK_GROUP_PATTERNS,
    _configure_env as configure_mass_environment,
)
from evaluate_g008_periodic_friction import (  # noqa: E402
    _configure_env as configure_periodic_environment,
    _selected_case,
    _validate_field_phase,
    _verify_collision_surface_isolation,
)
from record_g008_directions import (  # noqa: E402
    SEQUENCE,
    WINDOWS_KIT_ARGS,
    command_at_step,
    file_sha256,
    portable_path,
)


TASK = "Isaac-G008-Velocity-Rough-Go2-CommandSuite-v0"
PERIODIC_CASE_ID = "mixed_020_010"
MASS_FACTOR = 1.20


@dataclass(frozen=True)
class CaptureProfile:
    profile_id: str
    label: str
    stage: str
    output_name: str
    mass_group: str | None = None
    mass_factor: float | None = None


CAPTURE_PROFILES = (
    CaptureProfile(
        profile_id="periodic_friction_s1_mu020_010",
        label="Friction S1 | mixed 0.8/0.6 and 0.2/0.1",
        stage="periodic_friction",
        output_name="g008_stage_periodic_friction_s1_mu020_010_s20260826.mp4",
    ),
    *(
        CaptureProfile(
            profile_id=f"link_mass_{group}_120",
            label=f"Leg-mass S1 | {group} x1.20",
            stage="link_mass",
            output_name=f"g008_stage_link_mass_{group}_120_s20260826.mp4",
            mass_group=group,
            mass_factor=MASS_FACTOR,
        )
        for group in LINK_GROUP_PATTERNS
    ),
)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def profile_by_id(profile_id: str) -> CaptureProfile:
    matches = [profile for profile in CAPTURE_PROFILES if profile.profile_id == profile_id]
    if len(matches) != 1:
        raise ValueError(f"capture profile not found or duplicated: {profile_id}")
    return matches[0]


def _set_viewer(env_cfg: Any) -> None:
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (3.0, 3.0, 2.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.35)


def _periodic_env_cfg(args: argparse.Namespace) -> tuple[Any, SimpleNamespace, dict[str, Any]]:
    total_steps = sum(length for _, length, _ in SEQUENCE)
    periodic_args = SimpleNamespace(
        task=TASK,
        device=args.device,
        num_envs=1,
        eval_seed=args.seed,
        env_spacing_m=16.0,
        stripe_count=48,
        stripe_width_m=0.5,
        stripe_width_y_m=4.0,
        stripe_surface_height_m=0.002,
        high_static=0.8,
        high_dynamic=0.6,
        horizon_steps=total_steps,
    )
    case = _selected_case(PERIODIC_CASE_ID)
    env_cfg = configure_periodic_environment(periodic_args, case)
    _set_viewer(env_cfg)
    return env_cfg, periodic_args, case


def _mass_env_cfg(args: argparse.Namespace) -> Any:
    total_steps = sum(length for _, length, _ in SEQUENCE)
    mass_args = SimpleNamespace(
        task=TASK,
        device=args.device,
        num_envs=1,
        eval_seed=args.seed,
        horizon_steps=total_steps,
    )
    env_cfg = configure_mass_environment(mass_args)
    _set_viewer(env_cfg)
    return env_cfg


def _spawn_friction_visual_overlay(periodic_args: SimpleNamespace, case: dict[str, Any]) -> dict[str, Any]:
    """Add a collision-free colored overlay so physics-material stripes are visible on video."""
    import isaacsim.core.utils.stage as stage_utils
    from pxr import Gf, UsdGeom

    stage = stage_utils.get_current_stage()
    mesh_path = "/World/periodic_friction_capture_overlay"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    points = []
    face_vertex_counts = []
    face_vertex_indices = []
    colors = []
    half_length = 0.5 * periodic_args.stripe_count * periodic_args.stripe_width_m
    first_cell = round(-half_length / periodic_args.stripe_width_m)
    last_cell = first_cell + periodic_args.stripe_count
    low_color = Gf.Vec3f(0.18, 0.42, 0.75)
    high_color = Gf.Vec3f(0.72, 0.58, 0.30)
    for stripe_cell in range(first_cell, last_cell):
        x_min = stripe_cell * periodic_args.stripe_width_m
        x_max = x_min + periodic_args.stripe_width_m
        base = len(points)
        z = periodic_args.stripe_surface_height_m + 0.0005
        points.extend(
            (
                (x_min, -0.5 * periodic_args.stripe_width_y_m, z),
                (x_max, -0.5 * periodic_args.stripe_width_y_m, z),
                (x_max, 0.5 * periodic_args.stripe_width_y_m, z),
                (x_min, 0.5 * periodic_args.stripe_width_y_m, z),
            )
        )
        face_vertex_counts.append(4)
        face_vertex_indices.extend((base, base + 1, base + 2, base + 3))
        colors.append(low_color if stripe_cell % 2 == 0 else high_color)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(colors)
    return {
        "prim_path": mesh_path,
        "collision_api_applied": False,
        "stripe_count": periodic_args.stripe_count,
        "stripe_width_m": periodic_args.stripe_width_m,
        "low_color_rgb": [float(value) for value in low_color],
        "high_color_rgb": [float(value) for value in high_color],
        "physics_unchanged": True,
        "low_friction": [float(case["low_static"]), float(case["low_dynamic"])],
        "high_friction": [periodic_args.high_static, periodic_args.high_dynamic],
    }


def _statistics(values: Any) -> dict[str, float | int]:
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "mean": float(values.float().mean().item()),
        "max": float(values.max().item()),
    }


def _apply_controlled_mass(environment: Any, profile: CaptureProfile) -> dict[str, Any]:
    import torch

    if profile.mass_group is None or profile.mass_factor is None:
        raise ValueError("link-mass profile requires a group and factor")
    robot = environment.scene["robot"]
    body_ids, body_names = robot.find_bodies(LINK_GROUP_PATTERNS[profile.mass_group])
    if len(body_ids) != 4:
        raise RuntimeError(f"expected four {profile.mass_group} bodies, got {body_names}")
    all_leg_ids, _ = robot.find_bodies(".*_(hip|thigh|calf|foot)")
    selected = torch.tensor(body_ids, dtype=torch.long)
    all_legs = torch.tensor(all_leg_ids, dtype=torch.long)
    masses = robot.data.default_mass.detach().cpu().clone()
    inertias = robot.data.default_inertia.detach().cpu().clone()
    default_masses = masses.clone()
    default_inertias = inertias.clone()
    masses[:, selected] *= profile.mass_factor
    inertias[:, selected, :] *= profile.mass_factor
    env_ids_cpu = torch.arange(masses.shape[0], device="cpu")
    robot.root_physx_view.set_masses(masses, env_ids_cpu)
    robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
    readback_masses = robot.root_physx_view.get_masses().clone().cpu()
    readback_inertias = robot.root_physx_view.get_inertias().clone().cpu()
    ratios = readback_masses[:, selected] / default_masses[:, selected]
    expected_inertias = default_inertias[:, selected, :] * profile.mass_factor
    inertia_error = torch.abs(readback_inertias[:, selected, :] - expected_inertias)
    return {
        "group": profile.mass_group,
        "factor": profile.mass_factor,
        "body_names": list(body_names),
        "mass_ratio": _statistics(ratios),
        "selected_group_mass_kg": _statistics(readback_masses[:, selected].sum(dim=1)),
        "total_leg_mass_kg": _statistics(readback_masses[:, all_legs].sum(dim=1)),
        "inertia_scale_absolute_error_max": float(inertia_error.max().item()),
        "center_of_mass_changed": False,
    }


def _record_profile(args: argparse.Namespace, profile: CaptureProfile) -> dict[str, Any]:
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
    destination = output_dir / profile.output_name
    if destination.exists():
        raise FileExistsError(destination)

    if profile.stage == "periodic_friction":
        env_cfg, periodic_args, case = _periodic_env_cfg(args)
    else:
        env_cfg = _mass_env_cfg(args)
        periodic_args = None
        case = None
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    raw_env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")

    if profile.stage == "periodic_friction":
        assert periodic_args is not None and case is not None
        stage_physics = {
            "case": case,
            "underlay": _verify_collision_surface_isolation(),
            "field_phase": _validate_field_phase(raw_env, periodic_args),
            "visual_overlay": _spawn_friction_visual_overlay(periodic_args, case),
        }
    else:
        stage_physics = _apply_controlled_mass(raw_env.unwrapped, profile)

    controller = raw_env.unwrapped.viewport_camera_controller
    if controller is not None:
        controller.update_view_to_asset_root("robot")
        controller.update_view_location(eye=(3.0, 3.0, 2.0), lookat=(0.0, 0.0, 0.35))
    total_steps = sum(length for _, length, _ in SEQUENCE)
    prefix = f"g008-stage-{profile.profile_id}"
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
            fixed_command = torch.tensor(command, dtype=torch.float32, device=env.unwrapped.device)
            command_buffer[0].copy_(fixed_command)
            obs, _ = env.get_observations()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
    finally:
        env.close()

    candidates = sorted(output_dir.glob(f"{prefix}*.mp4"), key=lambda path: path.stat().st_mtime)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one MP4 for {profile.profile_id}, found {candidates}")
    candidates[0].rename(destination)
    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "purpose": "stage-change visual evidence; quantitative gates remain in evaluation reports",
        "profile": {
            "profile_id": profile.profile_id,
            "label": profile.label,
            "stage": profile.stage,
            "task": TASK,
            "seed": args.seed,
            "headless": bool(args.headless),
            "step_dt_s": float(env_cfg.sim.dt * env_cfg.decimation),
            "total_steps": total_steps,
            "sequence": [
                {"name": name, "steps": length, "command": list(command)}
                for name, length, command in SEQUENCE
            ],
            "checkpoint": {
                "path": portable_path(checkpoint),
                "sha256": file_sha256(checkpoint),
            },
            "stage_physics": stage_physics,
            "local_video": {
                "path": portable_path(destination),
                "sha256": file_sha256(destination),
                "bytes": destination.stat().st_size,
                "git_policy": "local_only",
            },
        },
        "record_source_sha256": file_sha256(Path(__file__)),
        "periodic_evaluator_source_sha256": file_sha256(
            REPO_ROOT / "scripts" / "evaluate_g008_periodic_friction.py"
        ),
        "mass_evaluator_source_sha256": file_sha256(
            REPO_ROOT / "scripts" / "evaluate_g008_link_mass_sensitivity.py"
        ),
    }


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        choices=tuple(profile.profile_id for profile in CAPTURE_PROFILES),
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260826)
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
    profile = profile_by_id(args.profile)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = _record_profile(args, profile)
        _write_json_atomic(args.report.resolve(), report)
        print(
            json.dumps(
                {"report": str(args.report.resolve()), "video": report["profile"]["local_video"]}
            ),
            flush=True,
        )
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
