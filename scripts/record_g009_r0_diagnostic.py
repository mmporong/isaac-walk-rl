#!/usr/bin/env python3
"""Record the rejected rev9 prone pilot as a headless, local-only diagnostic MP4."""

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

GOAL_ID = "g009"
STAGE_NUMBER = "G009-5"
STAGE_ID = "R0"
DEFAULT_TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
POSE = "prone"
POSE_INDEX = 1
SOURCE_CLASS_ID = 0
CAPTURE_NUM_ENVS = 1
CAMERA_EYE = (1.4, 1.4, 0.85)
CAMERA_LOOKAT = (0.0, 0.0, 0.24)
DEFAULT_OUTPUT_DIR = Path.home() / "IsaacLab" / "logs" / "visual_evidence" / "g009" / "R0" / "diagnostic"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "runs" / "g009_r0_diag_rev9_01_prone_capture_s42.json"
OUTPUT_FILENAME = "g009_5_r0_diag_rev9_01_prone_s42.mp4"
EXPECTED_RUN_NAME = "go2_flat_recover_rev9_prone_pilot_s42_20260828-1421"
EXPECTED_TRAINING_REPORT_SHA256 = "4728a712a763e3f4857f54b97519c8cf7e350ac21f1d9ec8268b70e1694911e5"
EXPECTED_TRAINING_COMMIT = "030d6b4471848f538a28a8649e2d5b4e615df568"
EXPECTED_SOURCE_BUNDLE_SHA256 = "45a1b4cc9ccf73b8dedd63d69ab8e8163addb5b6cb0297daa89861a9a72abd55"
EXPECTED_CHECKPOINT_SHA256 = "18e87baf43351d5e36aae5cabc608666099e7460a20d2606610607bfc35b3bf1"
WINDOWS_KIT_ARGS = (
    "--/app/vulkan=false --/app/window/hideUi=true "
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
        return str(resolved.relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        pass
    try:
        return "%USERPROFILE%\\" + str(resolved.relative_to(Path.home().resolve()))
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty_paths = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    source_dirty_paths = [
        line for line in dirty_paths if not line[3:].replace("\\", "/").startswith("reports/runs/")
    ]
    return {
        "commit": commit,
        "clean": not source_dirty_paths,
        "dirty_paths": dirty_paths,
        "source_dirty_paths": source_dirty_paths,
        "allowed_dirty_scope": "reports/runs evidence only",
    }


def source_bundle_sha256(files: Mapping[str, str]) -> str:
    payload = "\n".join(f"{path}:{digest}" for path, digest in files.items())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_source_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    files = bundle.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("diagnostic training source_bundle.files must be non-empty")
    normalized: dict[str, str] = {}
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError("diagnostic training source bundle entries must be string pairs")
        normalized_relative = relative.replace("\\", "/")
        path = (REPO_ROOT / normalized_relative).resolve()
        try:
            path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"source bundle path escapes repository: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"source bundle file hash mismatch: {normalized_relative}")
        normalized[normalized_relative] = actual
    aggregate = source_bundle_sha256(normalized)
    if bundle.get("sha256") != aggregate:
        raise ValueError("diagnostic training source bundle aggregate hash mismatch")
    return {"sha256": aggregate, "files": normalized}


def validate_diagnostic_training_report(path: Path, checkpoint: Path) -> dict[str, Any]:
    report = _read_json(path)
    actual_report_sha = file_sha256(path)
    actual_checkpoint_sha = file_sha256(checkpoint)
    safety = report.get("training_safety_aggregate", {})
    series = report.get("tensorboard", {}).get("series_summary", {})
    checks = {
        "report_sha256": actual_report_sha == EXPECTED_TRAINING_REPORT_SHA256,
        "run_name": report.get("run_name") == EXPECTED_RUN_NAME,
        "task": report.get("task") == DEFAULT_TASK,
        "seed": report.get("seed") == 42,
        "num_envs": report.get("num_envs") == 1024,
        "max_iterations": report.get("max_iterations") == 50,
        "last_iteration": report.get("last_iteration") == 49,
        "iteration_target": report.get("iteration_target") == 50,
        "headless": report.get("headless") is True,
        "scratch": report.get("resume", {}).get("enabled") is False,
        "no_hydra_overrides": report.get("effective_hydra_overrides") == [],
        "diagnostic_training": (
            report.get("qualification_mode", {}).get("enabled") is False
            and report.get("qualification_mode", {}).get("preflight_passed") is None
            and report.get("qualification_mode", {}).get("policy_qualification_status") == "not_run"
        ),
        "run_health": report.get("run_health_passed") is True and report.get("passed") is True,
        "repository_clean_at_training": report.get("repository", {}).get("dirty") is False,
        "training_commit": report.get("repository", {}).get("commit") == EXPECTED_TRAINING_COMMIT,
        "source_bundle_commit_match": report.get("source_bundle", {}).get("matches_repository_commit") is True,
        "source_bundle_identity": report.get("source_bundle", {}).get("sha256") == EXPECTED_SOURCE_BUNDLE_SHA256,
        "checkpoint_identity": (
            report.get("artifacts", {}).get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256
            and actual_checkpoint_sha == EXPECTED_CHECKPOINT_SHA256
            and Path(str(report.get("artifacts", {}).get("checkpoint", "")).replace("\\", "/")).name
            == "model_49.pt"
        ),
        "numeric_safety_series_present": safety.get("numeric_invalid", {}).get("maximum") == 0.0,
        "rejected_hard_limit_evidence_present": (
            isinstance(safety.get("hard_joint_limit", {}).get("maximum"), (int, float))
            and safety["hard_joint_limit"]["maximum"] > 0.0
        ),
        "strict_success_absent": series.get("Episode_Reward/stable_success_once", {}).get("maximum") == 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("rev9 diagnostic training binding failed: " + ", ".join(failed))
    source_bundle = validate_source_bundle(report.get("source_bundle", {}))
    if source_bundle["sha256"] != EXPECTED_SOURCE_BUNDLE_SHA256:
        raise ValueError("rev9 diagnostic source bundle identity mismatch")
    return {
        "path": portable_path(path),
        "sha256": actual_report_sha,
        "run_name": EXPECTED_RUN_NAME,
        "repository": {"commit": EXPECTED_TRAINING_COMMIT, "clean": True},
        "source_bundle": source_bundle,
        "checkpoint_sha256": actual_checkpoint_sha,
        "protocol": {
            "num_envs": 1024,
            "max_iterations": 50,
            "seed": 42,
            "headless": True,
            "scratch": True,
            "qualification_enabled": False,
        },
    }


def validate_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError(f"diagnostic MP4 output must be local-only directory {DEFAULT_OUTPUT_DIR}")
    return resolved


def validate_report_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != DEFAULT_REPORT_PATH.resolve():
        raise ValueError(f"diagnostic capture report path is fixed to {DEFAULT_REPORT_PATH}")
    return resolved


def output_name() -> str:
    return OUTPUT_FILENAME


def diagnostic_recorded_frame_count(elapsed_steps: int, terminated: bool) -> int:
    if elapsed_steps <= 0:
        raise ValueError("diagnostic elapsed_steps must be positive")
    return elapsed_steps if terminated else elapsed_steps + 1


def _cleanup_capture_artifacts(output_dir: Path, prefix: str, destination: Path) -> None:
    for path in output_dir.glob(f"{prefix}*.mp4"):
        path.unlink(missing_ok=True)
    destination.unlink(missing_ok=True)
    destination.with_suffix(".trim.mp4").unlink(missing_ok=True)


def _trim_capture(source: Path, destination: Path, frame_count: int, ffmpeg: str, ffprobe: str) -> int:
    if frame_count <= 0:
        raise ValueError("diagnostic capture frame count must be positive")
    temporary = destination.with_suffix(".trim.mp4")
    try:
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                "-frames:v", str(frame_count), "-an", "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
            ],
            check=True,
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("diagnostic frame trim produced no video")
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
        actual_frames = int(json.loads(probe.stdout)["streams"][0]["nb_read_frames"])
        if actual_frames != frame_count:
            raise RuntimeError(f"trimmed diagnostic frame count mismatch: expected {frame_count}, got {actual_frames}")
    except Exception:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
    return actual_frames


def _configure_environment(args: argparse.Namespace) -> Any:
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg: Any = parse_env_cfg(args.task, device=args.device, num_envs=CAPTURE_NUM_ENVS)
    env_cfg.seed = args.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.reset_base.params.update(
        {"assignment_mode": "stratified", "pose_xy_range": (0.0, 0.0), "yaw_range": (0.0, 0.0)}
    )
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = SOURCE_CLASS_ID
    env_cfg.viewer.eye = CAMERA_EYE
    env_cfg.viewer.lookat = CAMERA_LOOKAT
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
    training_binding = validate_diagnostic_training_report(args.training_report.resolve(), checkpoint)
    source_state_before = git_source_state()
    if not source_state_before["clean"]:
        raise ValueError("diagnostic capture source tree is dirty outside reports/runs")
    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / output_name()
    if destination.exists():
        raise FileExistsError(destination)

    env_cfg: Any = _configure_environment(args)
    agent_cfg: Any = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    raw_env: Any = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    controller = raw_env.unwrapped.viewport_camera_controller
    if controller is not None:
        controller.update_view_to_asset_root("robot")
        controller.update_view_location(eye=CAMERA_EYE, lookat=CAMERA_LOOKAT)
    prefix = "g009-5-r0-diag-rev9-pilot-01-prone"
    recorded_env: Any = gym.wrappers.RecordVideo(
        raw_env,
        video_folder=str(output_dir),
        episode_trigger=lambda episode: False,
        video_length=args.horizon_steps + 1,
        disable_logger=True,
        name_prefix=prefix,
    )
    env: Any = RslRlVecEnvWrapper(recorded_env, clip_actions=agent_cfg.clip_actions)
    runner: Any = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(checkpoint))
    policy: Any = runner.get_inference_policy(device=env.unwrapped.device)
    robot: Any = env.unwrapped.scene["robot"]
    observations, _ = env.get_observations()
    class_ids = env.unwrapped._g009_recover_fall_class.detach().clone()
    if int(class_ids[SOURCE_CLASS_ID].item()) != SOURCE_CLASS_ID:
        raise RuntimeError("diagnostic selected pose does not match stratified prone reset readback")
    recorded_env.start_recording(prefix)
    recorded_env._capture_frame()

    step_dt_s = float(env_cfg.sim.dt * env_cfg.decimation)
    success = False
    terminated = False
    termination_reason = "capture_horizon"
    elapsed_steps = 0
    active_at_termination: dict[str, bool] = {}
    materials = None
    effective = None
    masses = None
    active_terminations: list[str] = []
    try:
        try:
            for step in range(args.horizon_steps):
                with torch.inference_mode():
                    actions = policy(observations)
                    observations, _, dones, _ = env.step(actions)
                elapsed_steps = step + 1
                if bool(dones[SOURCE_CLASS_ID].item()):
                    terminated = True
                    active_at_termination = {
                        name: bool(env.unwrapped.termination_manager.get_term(name)[SOURCE_CLASS_ID].item())
                        for name in env.unwrapped.termination_manager.active_terms
                    }
                    success = active_at_termination.get("stable_success", False)
                    termination_reason = next(
                        (name for name, value in active_at_termination.items() if value), "unknown"
                    )
                    break
            materials = getattr(env.unwrapped, "_g009_foot_material_readback", None)
            effective = getattr(env.unwrapped, "_g009_effective_foot_friction", None)
            friction_valid = getattr(env.unwrapped, "_g009_effective_foot_friction_valid", None)
            if materials is None or effective is None or friction_valid is None:
                raise RuntimeError("diagnostic foot/effective friction readback provenance is unavailable")
            materials = materials.detach().cpu()
            effective = effective.detach().cpu()
            if materials.shape != (CAPTURE_NUM_ENVS, 4, 2) or effective.shape != (CAPTURE_NUM_ENVS, 4, 2):
                raise RuntimeError("diagnostic foot/effective friction readback shape mismatch")
            if not bool(friction_valid.all().item()) or not torch.isfinite(materials).all() or not torch.isfinite(effective).all():
                raise RuntimeError("diagnostic foot/effective friction readback is invalid")
            terrain_pair = torch.tensor(
                [env_cfg.scene.terrain.physics_material.static_friction, env_cfg.scene.terrain.physics_material.dynamic_friction]
            )
            if str(env_cfg.scene.terrain.physics_material.friction_combine_mode) != "multiply":
                raise RuntimeError("diagnostic friction provenance requires multiply combine mode")
            if not torch.allclose(effective, materials * terrain_pair, rtol=0.0, atol=1.0e-6):
                raise RuntimeError("diagnostic effective friction does not match foot x terrain readback")
            masses = robot.root_physx_view.get_masses().detach().cpu()
            active_terminations = list(env.unwrapped.termination_manager.active_terms)
        finally:
            env.close()
    except Exception:
        _cleanup_capture_artifacts(output_dir, prefix, destination)
        raise

    try:
        candidates = sorted(output_dir.glob(f"{prefix}*.mp4"), key=lambda path: path.stat().st_mtime)
        if len(candidates) != 1:
            raise RuntimeError(f"expected one diagnostic MP4, found {candidates}")
        target_frames = diagnostic_recorded_frame_count(elapsed_steps, terminated)
        recorded_frames = _trim_capture(candidates[0], destination, target_frames, args.ffmpeg, args.ffprobe)
        candidates[0].unlink(missing_ok=True)
    except Exception:
        _cleanup_capture_artifacts(output_dir, prefix, destination)
        raise

    source_state_after = git_source_state()
    source_bundle_after = validate_source_bundle(training_binding["source_bundle"])
    if source_state_before != source_state_after or not source_state_after["clean"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError("repository commit/source state changed during diagnostic capture")
    if source_bundle_after != training_binding["source_bundle"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError("training source bundle changed during diagnostic capture")
    if materials is None or effective is None or masses is None:
        destination.unlink(missing_ok=True)
        raise RuntimeError("diagnostic physics readback was not captured")

    safety_termination = bool(
        active_at_termination.get("numeric_invalid", False)
        or active_at_termination.get("hard_joint_limit", False)
    )
    return {
        "schema_version": 1,
        "goal_id": GOAL_ID,
        "stage_number": STAGE_NUMBER,
        "stage_id": STAGE_ID,
        "status": "diagnostic_complete",
        "diagnostic_only": True,
        "public_claim_eligible": False,
        "qualification_status": "not_run",
        "policy_result": "single_playback_success" if success else "failure",
        "strict_success": int(success),
        "purpose": "rejected rev9 prone pilot playback; not policy qualification evidence",
        "task": args.task,
        "pose": {"index": POSE_INDEX, "pose_id": POSE, "source_class_id": SOURCE_CLASS_ID},
        "seed": args.seed,
        "headless": bool(args.headless),
        "offscreen": True,
        "camera": {"resolution": [1280, 720], "eye": list(CAMERA_EYE), "lookat": list(CAMERA_LOOKAT)},
        "step_dt_s": step_dt_s,
        "elapsed_steps": elapsed_steps,
        "recorded_frames": recorded_frames,
        "checkpoint": {"path": portable_path(checkpoint), "sha256": file_sha256(checkpoint)},
        "source_commit": source_state_after["commit"],
        "capture_commit": source_state_after["commit"],
        "source_state": {"before": source_state_before, "after": source_state_after},
        "training_binding": training_binding,
        "physics_readback": {
            "terrain_static_friction": float(env_cfg.scene.terrain.physics_material.static_friction),
            "terrain_dynamic_friction": float(env_cfg.scene.terrain.physics_material.dynamic_friction),
            "friction_combine_mode": str(env_cfg.scene.terrain.physics_material.friction_combine_mode),
            "robot_total_mass_kg": float(masses[SOURCE_CLASS_ID].sum().item()),
            "foot_material_static_friction_range": [
                float(materials[SOURCE_CLASS_ID, :, 0].min().item()),
                float(materials[SOURCE_CLASS_ID, :, 0].max().item()),
            ],
            "foot_material_dynamic_friction_range": [
                float(materials[SOURCE_CLASS_ID, :, 1].min().item()),
                float(materials[SOURCE_CLASS_ID, :, 1].max().item()),
            ],
            "effective_foot_static_friction_range": [
                float(effective[SOURCE_CLASS_ID, :, 0].min().item()),
                float(effective[SOURCE_CLASS_ID, :, 0].max().item()),
            ],
            "effective_foot_dynamic_friction_range": [
                float(effective[SOURCE_CLASS_ID, :, 1].min().item()),
                float(effective[SOURCE_CLASS_ID, :, 1].max().item()),
            ],
            "effective_friction_valid": True,
            "active_terminations": active_terminations,
        },
        "metrics": {
            "stable_success": success,
            "terminated": terminated,
            "termination_reason": termination_reason,
            "active_at_termination": active_at_termination,
            "safety_termination": safety_termination,
            "recovery_time_s": elapsed_steps * step_dt_s if success else None,
            "terminal_reset_frame_excluded": terminated,
            "recorded_frame_policy": "initial frame plus post-step frames; terminal auto-reset frame removed when termination occurs",
        },
        "local_video": {
            "path": portable_path(destination),
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
            "git_policy": "local_only",
        },
        "source_bindings": {
            "record_source": {
                "path": "scripts/record_g009_r0_diagnostic.py",
                "sha256": file_sha256(Path(__file__)),
            },
            "config": {
                "path": "configs/g009_r0.json",
                "sha256": file_sha256(REPO_ROOT / "configs" / "g009_r0.json"),
            },
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon-steps", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.enable_cameras = True
    if not args.headless:
        raise ValueError("rev9 diagnostic capture requires --headless off-screen mode")
    if sys.platform == "win32" and not args.kit_args:
        args.kit_args = WINDOWS_KIT_ARGS
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.training_report.is_file():
        raise FileNotFoundError(args.training_report)
    if args.seed != 42 or args.horizon_steps != 400 or args.task != DEFAULT_TASK:
        raise ValueError("rev9 diagnostic capture is fixed to task/seed42/400 steps")
    validate_output_dir(args.output_dir)
    validate_report_path(args.report)
    return args


def main(argv: list[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    args = parse_args(argv)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = _record(args)
        _write_json_atomic(args.report.resolve(), report)
        print(json.dumps({"report": str(args.report.resolve()), "status": report["status"]}), flush=True)
        return 0
    finally:
        simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
