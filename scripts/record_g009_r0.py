#!/usr/bin/env python3
"""Record one G009-5 R0 recovery pose as headless off-screen local-only MP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluate_g009_r0 import (  # noqa: E402
    OFFICIAL_PROTOCOL,
    _read_json,
    git_source_state,
    validate_source_bundle,
    validate_training_report,
)

GOAL_ID = "g009"
STAGE_NUMBER = "G009-5"
STAGE_ID = "R0"
DEFAULT_TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
POSE_NAMES = ("prone", "supine", "left_side", "right_side")
DEFAULT_OUTPUT_DIR = Path.home() / "IsaacLab" / "logs" / "visual_evidence" / "g009" / "R0"
WINDOWS_KIT_ARGS = (
    "--/app/window/enabled=false --/app/livestream/enabled=false "
    "--/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720"
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
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        pass
    try:
        return "%USERPROFILE%\\" + str(resolved.relative_to(Path.home().resolve()))
    except ValueError:
        return str(resolved)


def output_name(pose: str, seed: int) -> str:
    index = POSE_NAMES.index(pose) + 1
    return f"g009_5_r0_{index:02d}_{pose}_s{seed}.mp4"


def validate_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError(f"R0 MP4 output must be local-only directory {DEFAULT_OUTPUT_DIR}")
    return resolved


def _validate_quantitative_report(path: Path, checkpoint: Path, training_binding: Mapping[str, Any]) -> dict[str, Any]:
    report = _read_json(path)
    if report.get("status") != "pass" or report.get("protocol_mode") != "official_qualification":
        raise ValueError("capture requires a passing official quantitative report")
    if report.get("official_protocol") != OFFICIAL_PROTOCOL:
        raise ValueError("quantitative official protocol binding mismatch")
    if report.get("checkpoint", {}).get("sha256") != file_sha256(checkpoint):
        raise ValueError("quantitative/capture checkpoint mismatch")
    if report.get("training_binding") != training_binding:
        raise ValueError("quantitative/capture training binding mismatch")
    if report.get("aggregate", {}).get("all_pose_gate_pass") is not True:
        raise ValueError("quantitative all-pose gate must pass")
    return report


def _trim_terminal_reset_frame(
    source: Path, destination: Path, frame_count: int, ffmpeg: str, ffprobe: str
) -> int:
    if frame_count <= 0:
        raise ValueError("capture frame count must be positive")
    temporary = destination.with_suffix(".trim.mp4")
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-frames:v", str(frame_count), "-an", "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
        ],
        check=True,
    )
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("terminal reset frame trim produced no video")
    temporary.replace(destination)
    probe = subprocess.run(
        [
            ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "json", str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    value = json.loads(probe.stdout)
    actual_frames = int(value["streams"][0]["nb_read_frames"])
    if actual_frames != frame_count:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"trimmed capture frame count mismatch: expected {frame_count}, got {actual_frames}")
    return actual_frames


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _configure_environment(args: argparse.Namespace) -> Any:
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=len(POSE_NAMES))
    env_cfg.seed = args.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.reset_base.params.update(
        {"assignment_mode": "stratified", "pose_xy_range": (0.0, 0.0), "yaw_range": (0.0, 0.0)}
    )
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = POSE_NAMES.index(args.pose)
    env_cfg.viewer.eye = (3.2, 3.2, 1.8)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.30)
    return env_cfg


def _record(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaac_walk_g009 import register_tasks

    register_tasks()
    checkpoint = args.checkpoint.resolve()
    training_binding = validate_training_report(args.training_report.resolve(), checkpoint)
    quantitative = _validate_quantitative_report(
        args.quantitative_report.resolve(), checkpoint, training_binding
    )
    source_state_before = git_source_state()
    if not source_state_before["clean"]:
        raise ValueError("capture source tree is dirty outside reports/runs")
    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / output_name(args.pose, args.seed)
    if destination.exists():
        raise FileExistsError(destination)
    env_cfg = _configure_environment(args)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    raw_env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    controller = raw_env.unwrapped.viewport_camera_controller
    if controller is not None:
        controller.update_view_to_asset_root("robot")
        controller.update_view_location(eye=(3.2, 3.2, 1.8), lookat=(0.0, 0.0, 0.30))

    prefix = f"g009-5-r0-{POSE_NAMES.index(args.pose) + 1:02d}-{args.pose}"
    recorded_env = gym.wrappers.RecordVideo(
        raw_env,
        video_folder=str(output_dir),
        step_trigger=lambda step: step == 0,
        video_length=args.horizon_steps,
        disable_logger=True,
        name_prefix=prefix,
    )
    env = RslRlVecEnvWrapper(recorded_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]
    observations, _ = env.get_observations()
    class_ids = env.unwrapped._g009_recover_fall_class.detach().clone()
    selected_index = POSE_NAMES.index(args.pose)
    if int(class_ids[selected_index].item()) != selected_index:
        raise RuntimeError("selected pose does not match stratified reset readback")

    step_dt_s = float(env_cfg.sim.dt * env_cfg.decimation)
    success = False
    termination_reason = "capture_horizon"
    elapsed_steps = 0
    materials = None
    masses = None
    active_terminations: list[str] = []
    try:
        for step in range(args.horizon_steps):
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, dones, _ = env.step(actions)
            elapsed_steps = step + 1
            if bool(dones[selected_index].item()):
                active = {
                    name: bool(env.unwrapped.termination_manager.get_term(name)[selected_index].item())
                    for name in env.unwrapped.termination_manager.active_terms
                }
                success = active.get("stable_success", False)
                termination_reason = next((name for name, value in active.items() if value), "unknown")
                break
        materials = getattr(env.unwrapped, "_g009_foot_material_readback", None)
        effective = getattr(env.unwrapped, "_g009_effective_foot_friction", None)
        friction_valid = getattr(env.unwrapped, "_g009_effective_foot_friction_valid", None)
        if materials is None or effective is None or friction_valid is None:
            raise RuntimeError("foot/effective friction readback provenance is unavailable")
        materials = materials.detach().cpu()
        effective = effective.detach().cpu()
        if materials.shape != (len(POSE_NAMES), 4, 2) or effective.shape != (len(POSE_NAMES), 4, 2):
            raise RuntimeError("foot/effective friction readback shape mismatch")
        if not bool(friction_valid.all().item()) or not torch.isfinite(materials).all() or not torch.isfinite(effective).all():
            raise RuntimeError("foot/effective friction readback is invalid")
        terrain_pair = torch.tensor(
            [env_cfg.scene.terrain.physics_material.static_friction, env_cfg.scene.terrain.physics_material.dynamic_friction]
        )
        if str(env_cfg.scene.terrain.physics_material.friction_combine_mode) != "multiply":
            raise RuntimeError("capture friction provenance requires multiply combine mode")
        if not torch.allclose(effective, materials * terrain_pair, rtol=0.0, atol=1.0e-6):
            raise RuntimeError("effective friction does not match foot x terrain multiply readback")
        masses = robot.root_physx_view.get_masses().detach().cpu()
        active_terminations = list(env.unwrapped.termination_manager.active_terms)
    finally:
        env.close()

    candidates = sorted(output_dir.glob(f"{prefix}*.mp4"), key=lambda path: path.stat().st_mtime)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one captured MP4, found {candidates}")
    if not success:
        candidates[0].unlink(missing_ok=True)
        raise RuntimeError(f"capture pose did not reach stable_success: {args.pose} ({termination_reason})")
    recorded_frames = _trim_terminal_reset_frame(
        candidates[0], destination, elapsed_steps, args.ffmpeg, args.ffprobe
    )
    candidates[0].unlink(missing_ok=True)
    source_state_after = git_source_state()
    source_bundle_after = validate_source_bundle(training_binding["source_bundle"])
    if source_state_before != source_state_after or not source_state_after["clean"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError("repository commit/source state changed during capture")
    if source_bundle_after != training_binding["source_bundle"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError("training source bundle changed during capture")
    if materials is None or masses is None:
        raise RuntimeError("physics readback was not captured before environment close")
    return {
        "schema_version": 1,
        "goal_id": GOAL_ID,
        "stage_number": STAGE_NUMBER,
        "stage_id": STAGE_ID,
        "status": "complete",
        "purpose": "single-pose qualitative playback; quantitative decisions use evaluate_g009_r0.py",
        "task": args.task,
        "pose": {"index": selected_index + 1, "pose_id": args.pose, "source_class_id": selected_index},
        "seed": args.seed,
        "headless": bool(args.headless),
        "offscreen": True,
        "camera": {"resolution": [1280, 720], "eye": [3.2, 3.2, 1.8], "lookat": [0.0, 0.0, 0.30]},
        "step_dt_s": step_dt_s,
        "elapsed_steps": elapsed_steps,
        "recorded_frames": recorded_frames,
        "checkpoint": {"path": portable_path(checkpoint), "sha256": file_sha256(checkpoint)},
        "source_commit": source_state_after["commit"],
        "source_state": {"before": source_state_before, "after": source_state_after},
        "training_binding": training_binding,
        "quantitative_report": {
            "path": portable_path(args.quantitative_report),
            "sha256": file_sha256(args.quantitative_report),
            "report_id": quantitative["report_id"],
        },
        "physics_readback": {
            "terrain_static_friction": float(env_cfg.scene.terrain.physics_material.static_friction),
            "terrain_dynamic_friction": float(env_cfg.scene.terrain.physics_material.dynamic_friction),
            "friction_combine_mode": str(env_cfg.scene.terrain.physics_material.friction_combine_mode),
            "robot_total_mass_kg": float(masses[selected_index].sum().item()),
            "foot_material_static_friction_range": [
                float(materials[selected_index, :, 0].min().item()),
                float(materials[selected_index, :, 0].max().item()),
            ],
            "foot_material_dynamic_friction_range": [
                float(materials[selected_index, :, 1].min().item()),
                float(materials[selected_index, :, 1].max().item()),
            ],
            "effective_foot_static_friction_range": [
                float(effective[selected_index, :, 0].min().item()),
                float(effective[selected_index, :, 0].max().item()),
            ],
            "effective_foot_dynamic_friction_range": [
                float(effective[selected_index, :, 1].min().item()),
                float(effective[selected_index, :, 1].max().item()),
            ],
            "effective_friction_valid": True,
            "effective_friction_derivation": "foot material readback multiplied by terrain material readback",
            "active_terminations": active_terminations,
        },
        "metrics": {
            "stable_success": success,
            "termination_reason": termination_reason,
            "recovery_time_s": elapsed_steps * step_dt_s if success else None,
            "terminal_reset_frame_policy": "initial through last pre-terminal frame; terminal auto-reset frame excluded",
        },
        "local_video": {
            "path": portable_path(destination),
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
            "git_policy": "local_only",
        },
        "source_bindings": {
            "record_source": {"path": "scripts/record_g009_r0.py", "sha256": file_sha256(Path(__file__))},
            "evaluator": quantitative["source_bindings"]["evaluator"],
            "config": {
                "path": "configs/g009_r0.json",
                "sha256": file_sha256(REPO_ROOT / "configs" / "g009_r0.json"),
            },
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", required=True, choices=POSE_NAMES)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--quantitative-report", required=True, type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-steps", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.enable_cameras = True
    if not args.headless:
        raise ValueError("G009 R0 capture requires --headless off-screen mode")
    if sys.platform == "win32" and not args.kit_args:
        args.kit_args = WINDOWS_KIT_ARGS
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.training_report.is_file():
        raise FileNotFoundError(args.training_report)
    if not args.quantitative_report.is_file():
        raise FileNotFoundError(args.quantitative_report)
    if args.seed != 42 or args.horizon_steps != 400 or args.task != DEFAULT_TASK:
        raise ValueError("capture protocol is fixed to task/seed42/400 steps")
    if args.horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    validate_output_dir(args.output_dir)
    return args


def main(argv: list[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    args = parse_args(argv)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = _record(args)
        _write_json_atomic(args.report.resolve(), report)
        print(json.dumps({"report": str(args.report.resolve()), "status": "complete"}), flush=True)
        return 0
    finally:
        simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
