#!/usr/bin/env python3
"""Record rev13 right-side/reset-pose-hold as local-only Isaac Sim camera footage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from probe_g009_recover_runtime import (  # noqa: E402
    articulation_solver_iteration_checks,
    articulation_solver_iteration_readback,
    reset_pose_hold_action_diagnostics,
    source_bundle_provenance,
)
from isaac_walk_g009.recover_contracts import canonical_sha256, recover_contract  # noqa: E402

TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
SEED = 42
NUM_ENVS = 8
ROLLOUT_STEPS = 150
POSE_COUNT = 4
SOURCE_ENV_INDEX = 7
POSE_ID = "right_side"
ACTION_MODE = "reset_pose_hold"
EXPECTED_COMMIT = "e3734b728fcf546fea4ee05b9c8733800d6ab536"
EXPECTED_CONTRACT_SHA256 = "ebee855c503c77bce93c0884535d4fdf66ee5a01538fa59eef0e1b7aabba7558"
EXPECTED_SOURCE_BUNDLE_SHA256 = "df6c6aa46181ca033791fb11ccfa76d9eab8643822da1c6cdc2e288409cabe3d"
EXPECTED_REPORT_SHA256 = "c4ffa272f3974f059e7b281452c40d4685ade6ef2dcfd626496f9f4e67ff38a3"
EXPECTED_EXECUTION_ID = "9e66cca532f64a7eaba06b615f38f37d"
EXPECTED_SOLVER_POSITION_ITERATIONS = 8
EXPECTED_SOLVER_VELOCITY_ITERATIONS = 1
EXPECTED_CONTROL_DT_S = 0.02
EXPECTED_PHYSICS_DT_S = 0.005
EXPECTED_FAILURE_PEAK_BW = 15.97161865234375
EXPECTED_FAILURE_STEP = 129
EXPECTED_FAILURE_BODY = "base"
OUTPUT_STEM = "g009_5_r0_diag_rev13_04_right_side_runtime"
DEFAULT_RUNTIME_REPORT = REPO_ROOT / "reports/runs/g009_r0_runtime_probe_rev13_cpu_rep01_s42.json"
DEFAULT_CAPTURE_REPORT = REPO_ROOT / "reports/runs" / f"{OUTPUT_STEM}_capture_s42.json"
DEFAULT_OUTPUT_DIR = Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic"
DEFAULT_VIDEO = DEFAULT_OUTPUT_DIR / f"{OUTPUT_STEM}_s42.mp4"
CAMERA_OFFSET_EYE = (1.55, 1.55, 0.95)
CAMERA_OFFSET_LOOKAT = (0.0, 0.0, 0.22)
WINDOWS_KIT_ARGS = (
    "--/app/vulkan=false --/app/window/hideUi=true "
    "--/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
        return "%USERPROFILE%\\" + str(resolved.relative_to(Path.home().resolve()))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def git_source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "clean": not dirty, "dirty_paths": dirty}


def contains_rev13_source_commit(commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_COMMIT, commit],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode == 0


def validate_current_capture_binding() -> dict[str, Any]:
    bundle = source_bundle_provenance()
    require(bundle.get("source_bundle_sha256") == EXPECTED_SOURCE_BUNDLE_SHA256, "current capture source bundle drifted from rev13")
    require(bundle.get("all_files_present") is True, "current capture source bundle has missing files")
    require(bundle.get("git_commit_valid") is True, "current capture source commit is invalid")
    require(bundle.get("clean") is True, "current capture source binding is dirty")
    contract_sha256 = canonical_sha256(recover_contract())
    require(contract_sha256 == EXPECTED_CONTRACT_SHA256, "current capture contract drifted from rev13")
    return {"source_bundle": bundle, "contract_sha256": contract_sha256}


def validate_output_paths(output_dir: Path, video: Path, report: Path) -> None:
    require(output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve(), "output directory is fixed to local visual_evidence")
    require(video.resolve() == DEFAULT_VIDEO.resolve(), "local MP4 numbered path is fixed")
    require(report.resolve() == DEFAULT_CAPTURE_REPORT.resolve(), "capture report path is fixed")
    require(not video.exists() and not report.exists(), "capture refuses to overwrite evidence")


def failure_cell(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = report.get("pose_mode_metrics")
    if not isinstance(rows, list):
        raise ValueError("runtime report pose_mode_metrics is required")
    matches = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("env_index") == SOURCE_ENV_INDEX
        and row.get("pose_id") == POSE_ID
        and row.get("action_mode") == ACTION_MODE
    ]
    require(len(matches) == 1, "runtime report must contain exactly one bound failure cell")
    return matches[0]


def validate_runtime_report(path: Path) -> dict[str, Any]:
    require(path.resolve() == DEFAULT_RUNTIME_REPORT.resolve(), "runtime report path is fixed")
    require(path.is_file(), "bound rev13 runtime report is missing")
    require(file_sha256(path) == EXPECTED_REPORT_SHA256, "bound rev13 runtime report hash mismatch")
    report = read_json(path)
    require(report.get("contract_sha256") == EXPECTED_CONTRACT_SHA256, "rev13 contract mismatch")
    source = report.get("source_bundle", {})
    require(source.get("git_commit") == EXPECTED_COMMIT, "rev13 source commit mismatch")
    require(source.get("source_bundle_sha256") == EXPECTED_SOURCE_BUNDLE_SHA256, "rev13 source bundle mismatch")
    require(source.get("clean") is True, "original rev13 runtime source was not clean")
    require(report.get("execution", {}).get("execution_id") == EXPECTED_EXECUTION_ID, "execution binding mismatch")
    require(report.get("seed") == SEED and report.get("num_envs") == NUM_ENVS, "seed/env count mismatch")
    require(report.get("rollout_steps") == ROLLOUT_STEPS, "rollout length mismatch")
    require(report.get("headless") is True and report.get("device") == "cpu", "runtime mode mismatch")
    cell = failure_cell(report)
    require(cell.get("max_nonfoot_force_bodyweights") == EXPECTED_FAILURE_PEAK_BW, "failure peak mismatch")
    require(cell.get("max_nonfoot_force_physics_step") == EXPECTED_FAILURE_STEP, "failure step mismatch")
    require(cell.get("max_nonfoot_force_body_name") == EXPECTED_FAILURE_BODY, "failure body mismatch")
    require(cell.get("termination_counts", {}).get("numeric_invalid") == 0, "numeric safety mismatch")
    require(cell.get("termination_counts", {}).get("hard_joint_limit") == 0, "joint-limit safety mismatch")
    return {"report": report, "cell": cell, "sha256": file_sha256(path)}


def ffprobe_summary(path: Path, executable: str) -> dict[str, Any]:
    value = json.loads(subprocess.run(
        [executable, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout)
    video = next(stream for stream in value["streams"] if stream.get("codec_type") == "video")
    duration = value.get("format", {}).get("duration") or video.get("duration")
    return {
        "codec": video.get("codec_name"),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frame_rate": video.get("avg_frame_rate"),
        "duration_s": float(duration),
    }


def transcode_h264(source: Path, destination: Path, ffmpeg: str) -> None:
    subprocess.run(
        [ffmpeg, "-y", "-i", str(source), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)],
        check=True,
    )


def capture(args: argparse.Namespace) -> dict[str, Any]:
    source_before = git_source_state()
    require(source_before["clean"] is True, "capture requires a clean committed source tree")
    require(contains_rev13_source_commit(source_before["commit"]), "capture commit must contain the rev13 source commit")
    current_binding_before = validate_current_capture_binding()
    binding = validate_runtime_report(args.runtime_report)
    validate_output_paths(args.output_dir, args.video, args.capture_report)

    import gymnasium as gym  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]
    import omni.usd  # pyright: ignore[reportMissingImports]
    from pxr import PhysxSchema  # pyright: ignore[reportMissingImports]
    from isaaclab_tasks.utils import parse_env_cfg  # pyright: ignore[reportMissingImports]
    from isaac_walk_g009 import register_tasks
    from isaac_walk_g009.recover_contracts import (
        ACTION_SCALE,
        ARTICULATION_SOLVER_POSITION_ITERATION_COUNT,
        ARTICULATION_SOLVER_VELOCITY_ITERATION_COUNT,
    )

    register_tasks()
    env_cfg = parse_env_cfg(TASK, device="cpu", num_envs=NUM_ENVS)
    env_cfg.seed = SEED
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.contact_forces.history_length = env_cfg.decimation
    env_cfg.events.reset_base.params.update(
        {"assignment_mode": "stratified", "pose_xy_range": (0.0, 0.0), "yaw_range": (0.0, 0.0)}
    )
    env_cfg.validate()
    require(float(env_cfg.sim.dt) == EXPECTED_PHYSICS_DT_S, "physics dt drifted")
    require(float(env_cfg.sim.dt * env_cfg.decimation) == EXPECTED_CONTROL_DT_S, "control dt drifted")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_env: Any = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
    controller = raw_env.unwrapped.viewport_camera_controller
    env_origin = raw_env.unwrapped.scene.env_origins[SOURCE_ENV_INDEX].detach().cpu().tolist()
    eye = tuple(env_origin[i] + CAMERA_OFFSET_EYE[i] for i in range(3))
    lookat = tuple(env_origin[i] + CAMERA_OFFSET_LOOKAT[i] for i in range(3))
    if controller is None:
        raw_env.close()
        raise RuntimeError("viewport camera controller is unavailable")
    controller.update_view_location(eye=eye, lookat=lookat)
    prefix = OUTPUT_STEM.replace("_", "-")
    recorded: Any = gym.wrappers.RecordVideo(
        raw_env, video_folder=str(args.output_dir), episode_trigger=lambda _: False,
        video_length=ROLLOUT_STEPS + 1, disable_logger=True, name_prefix=prefix,
    )
    env = recorded
    try:
        try:
            observations, _ = env.reset()
            del observations
            class_ids = env.unwrapped._g009_recover_fall_class.detach().clone()
            require(int(class_ids[SOURCE_ENV_INDEX].item()) == 3, "env7 is not right_side")
            robot = env.unwrapped.scene["robot"]
            action_term = env.unwrapped.action_manager.get_term("joint_pos")
            soft_limits = robot.data.soft_joint_pos_limits.detach().clone()
            diagnostics = reset_pose_hold_action_diagnostics(
                robot.data.joint_pos[POSE_COUNT:].detach(), soft_limits[POSE_COUNT:],
                list(robot.joint_names), action_scale=float(action_term.cfg.scale),
            )
            actions = torch.zeros((NUM_ENVS, env.unwrapped.action_manager.total_action_dim), device=env.unwrapped.device)
            actions[POSE_COUNT:] = diagnostics["normalized_action"]
            require(not bool(diagnostics["saturated_mask"].any().item()), "reset-pose-hold action saturated")
            solver = articulation_solver_iteration_readback(
                omni.usd.get_context().get_stage(), list(robot.root_physx_view.prim_paths), PhysxSchema,
            )
            solver_checks = articulation_solver_iteration_checks(
                solver,
                expected_position_count=ARTICULATION_SOLVER_POSITION_ITERATION_COUNT,
                expected_velocity_count=ARTICULATION_SOLVER_VELOCITY_ITERATION_COUNT,
                expected_articulations=NUM_ENVS,
            )
            require(all(solver_checks.values()), "live articulation solver readback mismatch")
            require(ARTICULATION_SOLVER_POSITION_ITERATION_COUNT == EXPECTED_SOLVER_POSITION_ITERATIONS, "position solver constant drifted")
            require(ARTICULATION_SOLVER_VELOCITY_ITERATION_COUNT == EXPECTED_SOLVER_VELOCITY_ITERATIONS, "velocity solver constant drifted")
            require(float(action_term.cfg.scale) == ACTION_SCALE, "action scale drifted")
            stable_cfg = env.unwrapped.termination_manager.get_term_cfg("stable_success")
            stable_cfg.params["required_consecutive_steps"] = ROLLOUT_STEPS + 1
            recorded.start_recording(prefix)
            recorded._capture_frame()
            for _ in range(ROLLOUT_STEPS):
                env.step(actions)
        finally:
            env.close()
    except Exception:
        for partial in args.output_dir.glob(f"{prefix}*.mp4"):
            partial.unlink(missing_ok=True)
        args.video.unlink(missing_ok=True)
        raise

    candidates = sorted(args.output_dir.glob(f"{prefix}*.mp4"))
    require(len(candidates) == 1, f"expected one raw camera recording, found {candidates}")
    temporary_video = candidates[0]
    transcode_h264(temporary_video, args.video, args.ffmpeg)
    if temporary_video.resolve() != args.video.resolve():
        temporary_video.unlink(missing_ok=True)
    probe = ffprobe_summary(args.video, args.ffprobe)
    require(probe["codec"] == "h264" and probe["width"] == 1280 and probe["height"] == 720, "H264 1280x720 output required")
    source_after = git_source_state()
    current_binding_after = validate_current_capture_binding()
    if source_after != source_before:
        args.video.unlink(missing_ok=True)
        raise RuntimeError("repository state changed during capture")
    if current_binding_after != current_binding_before:
        args.video.unlink(missing_ok=True)
        raise RuntimeError("capture source bundle or contract changed during capture")

    cell = binding["cell"]
    result = {
        "schema_version": "g009.r0.rev13.camera_capture.v1",
        "capture_id": uuid.uuid4().hex,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "goal_id": "g009", "stage_id": "R0", "revision": "rev13",
        "status": "diagnostic_complete", "diagnostic_only": True,
        "camera_footage": True, "telemetry_animation": False,
        "qualification_status": "not_run", "public_claim_eligible": False,
        "learned": False, "ppo_checkpoint_used": False,
        "labels": ["DIAGNOSTIC", "NOT QUALIFIED", "NO PPO", "RIGHT_SIDE", "RESET_POSE_HOLD", "REV13 REJECTED"],
        "task": TASK, "seed": SEED, "device": "cpu", "num_envs": NUM_ENVS,
        "source_env_index": SOURCE_ENV_INDEX, "pose_id": POSE_ID, "action_mode": ACTION_MODE,
        "headless": True, "offscreen": True,
        "camera": {"resolution": [1280, 720], "eye_world": list(eye), "lookat_world": list(lookat)},
        "timing": {"physics_dt_s": EXPECTED_PHYSICS_DT_S, "control_dt_s": EXPECTED_CONTROL_DT_S, "rollout_steps": ROLLOUT_STEPS},
        "source": {
            "original_runtime_binding": {
                "commit": EXPECTED_COMMIT,
                "bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
                "contract_sha256": EXPECTED_CONTRACT_SHA256,
            },
            "current_capture_binding": {
                "capture_commit": source_after["commit"],
                **current_binding_after,
            },
        },
        "solver_live_readback": solver,
        "action_generation": {
            "implementation": "probe_g009_recover_runtime.reset_pose_hold_action_diagnostics",
            "same_eight_env_stratified_path_as_runtime_probe": True,
            "normalized_action_saturated": False,
            "target_max_error_rad": float(diagnostics["max_target_error"][-1].item()),
        },
        "original_rev13_report_binding": {
            "path": portable_path(args.runtime_report), "sha256": binding["sha256"],
            "execution_id": EXPECTED_EXECUTION_ID, "failure_cell": {
                "env_index": SOURCE_ENV_INDEX, "pose_id": POSE_ID, "action_mode": ACTION_MODE,
                "max_nonfoot_force_bodyweights": cell["max_nonfoot_force_bodyweights"],
                "max_nonfoot_force_physics_step": cell["max_nonfoot_force_physics_step"],
                "max_nonfoot_force_body_name": cell["max_nonfoot_force_body_name"],
            },
        },
        "evidence_scope": "condition-matched visual playback; it does not claim direct reproduction of the report peak",
        "blocked_stages": {"gpu_runtime": True, "gate01": True, "gate10": True, "ppo_training": True},
        "local_video": {"path": portable_path(args.video), "sha256": file_sha256(args.video), "bytes": args.video.stat().st_size, "git_policy": "local_only", **probe},
    }
    write_json_new(args.capture_report, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-report", type=Path, default=DEFAULT_RUNTIME_REPORT)
    parser.add_argument("--capture-report", type=Path, default=DEFAULT_CAPTURE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.enable_cameras = True
    require(bool(args.headless), "camera capture requires --headless")
    require(getattr(args, "device", "cpu") == "cpu", "rev13 evidence is fixed to CPU")
    if sys.platform == "win32" and not args.kit_args:
        args.kit_args = WINDOWS_KIT_ARGS
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prelaunch = git_source_state()
    require(prelaunch["clean"] is True, "capture requires a clean committed source tree")
    require(contains_rev13_source_commit(prelaunch["commit"]), "capture commit must contain the rev13 source commit")
    validate_current_capture_binding()
    validate_runtime_report(args.runtime_report)
    validate_output_paths(args.output_dir, args.video, args.capture_report)
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    launcher = AppLauncher(args)
    try:
        result = capture(args)
    finally:
        launcher.app.close()
    print(json.dumps({"capture_report": str(args.capture_report), "local_video": result["local_video"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
