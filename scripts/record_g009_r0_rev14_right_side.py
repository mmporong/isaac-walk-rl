#!/usr/bin/env python3
"""Record the rejected rev14 right-side/zero-action CPU cell as local-only footage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from probe_g009_recover_runtime import (
    rigid_body_max_depenetration_velocity_checks,
    rigid_body_max_depenetration_velocity_readback,
    source_bundle_provenance,
)

from isaac_walk_g009.recover_contracts import (
    canonical_sha256,
    recover_contract,
)

TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
SEED = 42
NUM_ENVS = 8
ROLLOUT_STEPS = 150
EXPECTED_PHYSICS_DT_S = 0.005
EXPECTED_CONTROL_DT_S = 0.02
SOURCE_ENV_INDEX = 3
POSE_ID = "right_side"
ACTION_MODE = "zero_normalized"
EXPECTED_COMMIT = "e9c1eff15bb2679c67e325546a749dbe7f98b07c"
EXPECTED_CONTRACT_SHA256 = (
    "744c53d3c8d1e608f849af405c7d0fad314b01234fc4cb9a4ab1000c69140506"
)
EXPECTED_SOURCE_BUNDLE_SHA256 = (
    "5c3cfa41a9c6b61a5579ed48ed17eb4f0f363eeebb9f970b61eada09fca8bacc"
)
EXPECTED_REPORT_SHA256 = (
    "3cc04ef58582b8a9dc7c77b46ce8747965db66561c949f70f4ff40c3241faa1f"
)
EXPECTED_EXECUTION_ID = "a0fe0250b7cb4f51bb691deaa4d986af"
EXPECTED_MIN_SEPARATION_M = -0.010990187525749207
EXPECTED_SEPARATION_STEP = 2
EXPECTED_SEPARATION_BODY = "RL_foot"
EXPECTED_NONFOOT_PEAK_BW = 4.611482620239258
EXPECTED_NONFOOT_STEP = 11
EXPECTED_NONFOOT_BODY = "FL_thigh"
OUTPUT_STEM = "g009_5_r0_diag_rev14_04_right_side_tradeoff"
DEFAULT_RUNTIME_REPORT = (
    REPO_ROOT
    / "reports/runs/g009_r0_runtime_probe_rev14_actualtopology_cpu_rep01_s42.json"
)
DEFAULT_CAPTURE_REPORT = REPO_ROOT / "reports/runs" / f"{OUTPUT_STEM}_capture_s42.json"
DEFAULT_OUTPUT_DIR = Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic"
DEFAULT_VIDEO = DEFAULT_OUTPUT_DIR / f"{OUTPUT_STEM}_s42.mp4"
CAMERA_OFFSET_EYE = (1.55, 1.55, 0.95)
CAMERA_OFFSET_LOOKAT = (0.0, 0.0, 0.22)
WINDOWS_KIT_ARGS = (
    "--/app/vulkan=false --/app/window/hideUi=true "
    "--/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720"
)
EXPECTED_BODY_NAMES = (
    "base",
    "FL_hip",
    "FR_hip",
    "Head_upper",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "Head_lower",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
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
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_failure_manifest(
    output_dir: Path, phase: str, error: BaseException, evidence_paths: list[Path]
) -> Path:
    """Preserve failed local capture artifacts instead of deleting evidence."""

    manifest = output_dir / f"{OUTPUT_STEM}_failed_{uuid.uuid4().hex}.json"
    write_json_new(
        manifest,
        {
            "schema_version": "g009.r0.rev14.camera_capture_failure.v1",
            "status": "failed",
            "phase": phase,
            "captured_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "error_type": type(error).__name__,
            "error": str(error),
            "git_policy": "local_only",
            "preserved_artifacts": [
                {
                    "path": portable_path(path),
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in evidence_paths
                if path.is_file()
            ],
        },
    )
    return manifest


def git_source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "clean": not dirty, "dirty_paths": dirty}


def contains_expected_source_commit(commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTED_COMMIT, commit],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0
    )


def validate_current_capture_binding() -> dict[str, Any]:
    bundle = source_bundle_provenance()
    require(
        bundle.get("source_bundle_sha256") == EXPECTED_SOURCE_BUNDLE_SHA256,
        "current capture source bundle drifted from rev14",
    )
    require(
        bundle.get("all_files_present") is True
        and bundle.get("git_commit_valid") is True,
        "current capture source binding is invalid",
    )
    require(bundle.get("clean") is True, "current capture source binding is dirty")
    contract_sha256 = canonical_sha256(recover_contract())
    require(
        contract_sha256 == EXPECTED_CONTRACT_SHA256,
        "current capture contract drifted from rev14",
    )
    return {"source_bundle": bundle, "contract_sha256": contract_sha256}


def validate_output_paths(output_dir: Path, video: Path, report: Path) -> None:
    require(
        output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve(),
        "output directory is fixed to local visual_evidence",
    )
    require(
        video.resolve() == DEFAULT_VIDEO.resolve(), "local MP4 numbered path is fixed"
    )
    require(
        report.resolve() == DEFAULT_CAPTURE_REPORT.resolve(),
        "capture report path is fixed",
    )
    require(
        not video.exists() and not report.exists(),
        "capture refuses to overwrite evidence",
    )
    raw_prefix = OUTPUT_STEM.replace("_", "-")
    existing_raw = (
        sorted(output_dir.glob(f"{raw_prefix}*.mp4")) if output_dir.is_dir() else []
    )
    require(not existing_raw, f"capture raw prefix already exists: {existing_raw}")


def failure_cell(report: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        row
        for row in report.get("pose_mode_metrics", [])
        if isinstance(row, dict)
        and row.get("env_index") == SOURCE_ENV_INDEX
        and row.get("pose_id") == POSE_ID
        and row.get("action_mode") == ACTION_MODE
    ]
    require(
        len(matches) == 1, "runtime report must contain exactly one bound failure cell"
    )
    return matches[0]


def validate_runtime_report(path: Path) -> dict[str, Any]:
    require(
        path.resolve() == DEFAULT_RUNTIME_REPORT.resolve(),
        "runtime report path is fixed",
    )
    require(
        path.is_file() and file_sha256(path) == EXPECTED_REPORT_SHA256,
        "bound rev14 runtime report hash mismatch",
    )
    report = read_json(path)
    source = report.get("source_bundle", {})
    require(
        report.get("schema_version") == 3 and report.get("passed") is True,
        "runtime probe status mismatch",
    )
    require(
        report.get("run_health", {}).get("passed") is True
        and report.get("runtime_contract", {}).get("passed") is True,
        "runtime health/contract mismatch",
    )
    require(
        isinstance(report.get("checks"), dict)
        and report["checks"]
        and all(value is True for value in report["checks"].values()),
        "runtime checks mismatch",
    )
    require(
        report.get("qualification", {}).get("status") == "not_run"
        and report.get("qualification", {}).get("passed") is None,
        "qualification mismatch",
    )
    require(
        report.get("contract_sha256") == EXPECTED_CONTRACT_SHA256,
        "rev14 contract mismatch",
    )
    require(
        source.get("git_commit") == EXPECTED_COMMIT
        and source.get("source_bundle_sha256") == EXPECTED_SOURCE_BUNDLE_SHA256,
        "rev14 source binding mismatch",
    )
    require(
        source.get("clean") is True
        and source.get("all_files_present") is True
        and source.get("missing_files") == [],
        "original rev14 runtime source was not clean and complete",
    )
    require(
        report.get("execution", {}).get("execution_id") == EXPECTED_EXECUTION_ID,
        "execution binding mismatch",
    )
    require(
        report.get("seed") == SEED
        and report.get("num_envs") == NUM_ENVS
        and report.get("rollout_steps") == ROLLOUT_STEPS,
        "runtime shape mismatch",
    )
    require(
        report.get("headless") is True and report.get("device") == "cpu",
        "runtime mode mismatch",
    )
    crosscheck = report.get("required_crosschecks", {}).get(
        "cpu_contact_separation", {}
    )
    require(
        crosscheck.get("data_available") is True
        and crosscheck.get("passed") is False
        and crosscheck.get("threshold_m") == -0.01,
        "CPU separation failure mismatch",
    )
    cell = failure_cell(report)
    provenance = cell.get("min_contact_separation_provenance", {})
    require(
        cell.get("min_contact_separation_m") == EXPECTED_MIN_SEPARATION_M,
        "minimum separation mismatch",
    )
    require(
        provenance.get("physics_step") == EXPECTED_SEPARATION_STEP
        and str(provenance.get("actor0_path", "")).endswith(
            f"/{EXPECTED_SEPARATION_BODY}"
        ),
        "separation provenance mismatch",
    )
    require(
        cell.get("max_nonfoot_force_bodyweights") == EXPECTED_NONFOOT_PEAK_BW,
        "nonfoot force peak mismatch",
    )
    require(
        cell.get("max_nonfoot_force_physics_step") == EXPECTED_NONFOOT_STEP
        and cell.get("max_nonfoot_force_body_name") == EXPECTED_NONFOOT_BODY,
        "nonfoot force provenance mismatch",
    )
    require(
        cell.get("termination_counts", {}).get("numeric_invalid") == 0
        and cell.get("termination_counts", {}).get("hard_joint_limit") == 0,
        "safety termination mismatch",
    )
    return {"report": report, "cell": cell, "sha256": file_sha256(path)}


def ffprobe_summary(path: Path, executable: str) -> dict[str, Any]:
    value = json.loads(
        subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    video = next(
        stream for stream in value["streams"] if stream.get("codec_type") == "video"
    )
    duration = value.get("format", {}).get("duration") or video.get("duration")
    return {
        "codec": video.get("codec_name"),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frame_rate": video.get("avg_frame_rate"),
        "duration_s": float(duration),
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    source_before = git_source_state()
    require(
        source_before["clean"] is True
        and contains_expected_source_commit(source_before["commit"]),
        "capture requires a clean descendant of the bound rev14 commit",
    )
    binding_before = validate_current_capture_binding()
    runtime = validate_runtime_report(args.runtime_report)
    validate_output_paths(args.output_dir, args.video, args.capture_report)

    import gymnasium as gym  # pyright: ignore[reportMissingImports]
    import omni.usd  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]
    from isaaclab import sim as sim_utils  # pyright: ignore[reportMissingImports]
    from isaaclab_tasks.utils import (  # pyright: ignore[reportMissingImports]
        parse_env_cfg,  # pyright: ignore[reportMissingImports]
    )
    from pxr import PhysxSchema, UsdPhysics  # pyright: ignore[reportMissingImports]

    from isaac_walk_g009 import register_tasks

    register_tasks()
    env_cfg = parse_env_cfg(TASK, device="cpu", num_envs=NUM_ENVS)
    env_cfg.seed = SEED
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.contact_forces.history_length = env_cfg.decimation
    env_cfg.events.reset_base.params.update(
        {
            "assignment_mode": "stratified",
            "pose_xy_range": (0.0, 0.0),
            "yaw_range": (0.0, 0.0),
        }
    )
    env_cfg.validate()
    require(
        float(env_cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity) == 0.75,
        "rev14 max depenetration velocity drifted",
    )
    require(
        math.isclose(float(env_cfg.sim.dt), EXPECTED_PHYSICS_DT_S, abs_tol=1.0e-12)
        and math.isclose(
            float(env_cfg.sim.dt) * int(env_cfg.decimation),
            EXPECTED_CONTROL_DT_S,
            abs_tol=1.0e-12,
        ),
        "rev14 physics/control timing drifted",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_env: Any = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
    unwrapped = raw_env.unwrapped
    controller = unwrapped.viewport_camera_controller
    robot = unwrapped.scene["robot"]
    stage = omni.usd.get_context().get_stage()
    live_readback = rigid_body_max_depenetration_velocity_readback(
        stage,
        sim_utils.find_matching_prim_paths(robot.cfg.prim_path, stage),
        list(robot.root_physx_view.prim_paths),
        [list(group) for group in robot.root_physx_view.link_paths],
        list(robot.body_names),
        PhysxSchema,
        UsdPhysics,
    )
    live_readback_checks = rigid_body_max_depenetration_velocity_checks(
        live_readback,
        expected_velocity_m_s=0.75,
        expected_articulation_count=NUM_ENVS,
        expected_body_names=list(EXPECTED_BODY_NAMES),
    )
    if not all(live_readback_checks.values()):
        raw_env.close()
        raise RuntimeError("capture live 8x19 rigid-body readback failed")
    require(
        math.isclose(unwrapped.physics_dt, EXPECTED_PHYSICS_DT_S, abs_tol=1.0e-12)
        and math.isclose(unwrapped.step_dt, EXPECTED_CONTROL_DT_S, abs_tol=1.0e-12),
        "live rev14 physics/control timing drifted",
    )
    origin = unwrapped.scene.env_origins[SOURCE_ENV_INDEX].detach().cpu().tolist()
    eye = tuple(origin[i] + CAMERA_OFFSET_EYE[i] for i in range(3))
    lookat = tuple(origin[i] + CAMERA_OFFSET_LOOKAT[i] for i in range(3))
    if controller is None:
        raw_env.close()
        raise RuntimeError("viewport camera controller is unavailable")
    controller.update_view_location(eye=eye, lookat=lookat)
    prefix = OUTPUT_STEM.replace("_", "-")
    recorded: Any = gym.wrappers.RecordVideo(
        raw_env,
        video_folder=str(args.output_dir),
        episode_trigger=lambda _: False,
        video_length=ROLLOUT_STEPS + 1,
        disable_logger=True,
        name_prefix=prefix,
    )
    try:
        recorded.reset()
        class_ids = recorded.unwrapped._g009_recover_fall_class.detach().clone()
        require(int(class_ids[SOURCE_ENV_INDEX].item()) == 3, "env3 is not right_side")
        actions = torch.zeros(
            (NUM_ENVS, recorded.unwrapped.action_manager.total_action_dim),
            device=recorded.unwrapped.device,
        )
        stable_cfg = recorded.unwrapped.termination_manager.get_term_cfg(
            "stable_success"
        )
        stable_cfg.params["required_consecutive_steps"] = ROLLOUT_STEPS + 1
        recorded.start_recording(prefix)
        recorded._capture_frame()
        for _ in range(ROLLOUT_STEPS):
            recorded.step(actions)
    except Exception as error:
        recorded.close()
        preserved = sorted(args.output_dir.glob(f"{prefix}*.mp4"))
        manifest = write_failure_manifest(
            args.output_dir, "record_video", error, preserved
        )
        raise RuntimeError(
            f"capture failed; evidence preserved in {manifest}"
        ) from error
    recorded.close()
    candidates = sorted(args.output_dir.glob(f"{prefix}*.mp4"))
    require(len(candidates) == 1, f"expected one raw recording, found {candidates}")
    temporary = candidates[0]
    try:
        subprocess.run(
            [
                args.ffmpeg,
                "-n",
                "-i",
                str(temporary),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.video),
            ],
            check=True,
        )
    except Exception as error:
        preserved = [temporary, args.video]
        manifest = write_failure_manifest(
            args.output_dir, "transcode", error, preserved
        )
        raise RuntimeError(
            f"transcode failed; evidence preserved in {manifest}"
        ) from error
    probe = ffprobe_summary(args.video, args.ffprobe)
    require(
        probe["codec"] == "h264" and probe["width"] == 1280 and probe["height"] == 720,
        "H264 1280x720 output required",
    )
    source_after = git_source_state()
    binding_after = validate_current_capture_binding()
    if source_after != source_before or binding_after != binding_before:
        error = RuntimeError("repository source binding changed during capture")
        manifest = write_failure_manifest(
            args.output_dir, "post_capture_binding", error, [temporary, args.video]
        )
        raise RuntimeError(
            f"repository source binding changed; evidence preserved in {manifest}"
        ) from error
    cell = runtime["cell"]
    result = {
        "schema_version": "g009.r0.rev14.camera_capture.v1",
        "capture_id": uuid.uuid4().hex,
        "captured_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "goal_id": "g009",
        "stage_id": "R0",
        "revision": "rev14",
        "status": "rejected",
        "diagnostic_only": True,
        "camera_footage": True,
        "telemetry_animation": False,
        "qualification_status": "not_run",
        "public_claim_eligible": False,
        "learned": False,
        "ppo_checkpoint_used": False,
        "labels": [
            "DIAGNOSTIC",
            "REJECTED",
            "NO PPO",
            "RIGHT_SIDE",
            "ZERO_NORMALIZED",
            "CPU SEPARATION FAIL",
        ],
        "task": TASK,
        "seed": SEED,
        "device": "cpu",
        "num_envs": NUM_ENVS,
        "source_env_index": SOURCE_ENV_INDEX,
        "pose_id": POSE_ID,
        "action_mode": ACTION_MODE,
        "headless": True,
        "offscreen": True,
        "camera": {
            "resolution": [1280, 720],
            "eye_world": list(eye),
            "lookat_world": list(lookat),
        },
        "timing": {
            "physics_dt_s": unwrapped.physics_dt,
            "control_dt_s": unwrapped.step_dt,
            "decimation": int(env_cfg.decimation),
            "rollout_steps": ROLLOUT_STEPS,
            "rollout_duration_s": ROLLOUT_STEPS * unwrapped.step_dt,
        },
        "live_physics_readback": {
            "checks": live_readback_checks,
            "readback": live_readback,
        },
        "source": {
            "original_runtime_binding": {
                "commit": EXPECTED_COMMIT,
                "bundle_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
                "contract_sha256": EXPECTED_CONTRACT_SHA256,
            },
            "current_capture_binding": {
                "capture_commit": source_after["commit"],
                **binding_after,
            },
        },
        "action_generation": {
            "mode": "zero_normalized",
            "all_actions_exact_zero": True,
        },
        "original_rev14_report_binding": {
            "path": portable_path(args.runtime_report),
            "sha256": runtime["sha256"],
            "execution_id": EXPECTED_EXECUTION_ID,
            "tradeoff_cell": {
                "env_index": SOURCE_ENV_INDEX,
                "pose_id": POSE_ID,
                "action_mode": ACTION_MODE,
                "min_contact_separation_m": cell["min_contact_separation_m"],
                "separation_body": EXPECTED_SEPARATION_BODY,
                "separation_physics_step": EXPECTED_SEPARATION_STEP,
                "max_nonfoot_force_bodyweights": cell["max_nonfoot_force_bodyweights"],
                "max_nonfoot_force_body_name": cell["max_nonfoot_force_body_name"],
                "max_nonfoot_force_physics_step": cell[
                    "max_nonfoot_force_physics_step"
                ],
            },
        },
        "evidence_scope": "condition-matched visual playback; it does not claim direct visual measurement of contact separation",
        "completed_stages": {
            "cpu_runtime_3x": True,
            "gpu_runtime_3x": True,
            "strict_tradeoff_synthesis": True,
        },
        "blocked_stages": {"gate01": True, "gate10": True, "ppo_training": True},
        "local_video": {
            "path": portable_path(args.video),
            "sha256": file_sha256(args.video),
            "bytes": args.video.stat().st_size,
            "git_policy": "local_only",
            **probe,
        },
        "local_raw_video": {
            "path": portable_path(temporary),
            "sha256": file_sha256(temporary),
            "bytes": temporary.stat().st_size,
            "git_policy": "local_only",
        },
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
    require(
        bool(args.headless) and getattr(args, "device", "cpu") == "cpu",
        "capture requires --headless --device cpu",
    )
    if sys.platform == "win32" and not args.kit_args:
        args.kit_args = WINDOWS_KIT_ARGS
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prelaunch = git_source_state()
    require(
        prelaunch["clean"] is True
        and contains_expected_source_commit(prelaunch["commit"]),
        "capture requires a clean descendant of the bound rev14 commit",
    )
    validate_current_capture_binding()
    validate_runtime_report(args.runtime_report)
    validate_output_paths(args.output_dir, args.video, args.capture_report)
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    launcher = AppLauncher(args)
    try:
        result = capture(args)
    finally:
        launcher.app.close()
    print(
        json.dumps(
            {
                "capture_report": str(args.capture_report),
                "local_video": result["local_video"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
