"""1080p diagnostic playback of the bound rev32 checkpoint, never qualification."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from g009_r0_rev32 import REPO_ROOT, actuator_readback, configure_damping
from record_g009_r0_diagnostic import (
    file_sha256, git_source_state, portable_path, resolve_portable_path, validate_source_bundle,
)

TASK = "Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0"
POSES = ("prone", "supine", "left_side", "right_side")
OUTPUT = Path.home() / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic"
RESOLUTION = (1920, 1080)
FPS = 30
BASE_SOURCE_PATHS = {
    "configs/g009_r0.json", "scripts/bootstrap_train_g009.py",
    "scripts/bootstrap_train_g009_rev31_attribution.py", "scripts/run_training.ps1",
    "src/isaac_walk_g009/agent_cfg.py", "src/isaac_walk_g009/recover_env_cfg.py",
    "src/isaac_walk_g009/recover_contracts.py", "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/mdp/recover.py", "src/isaac_walk_g009/matrix_gate01.py",
    "src/isaac_walk_g009/matrix_observation_adapter.py", "src/isaac_walk_g009/registry.py",
}


def sample_steps(horizon: int, control_hz: int = 50, fps: int = FPS) -> tuple[int, ...]:
    """Sample actual control states, without duplicates or altered playback speed."""
    if horizon < 1 or not 1 <= fps <= control_hz:
        raise ValueError("positive horizon and fps <= control rate required")
    return tuple(round(i * control_hz / fps) for i in range(math.ceil(horizon * fps / control_hz)))


def validate_binding(path: Path, revision: str = "rev32") -> tuple[dict, Path]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if revision not in ("rev32", "rev33"):
        raise ValueError("unknown damping revision")
    group = "rear" if revision == "rev32" else "front"
    run = report.get("run_name", "")
    if not re.fullmatch(rf"go2_flat_g009_r0_{revision}_{group}_damping_s42_[A-Za-z0-9._-]+", run):
        raise ValueError("canonical damping run identity mismatch")
    if path.resolve() != (REPO_ROOT / "reports/runs" / (run + ".json")).resolve():
        raise ValueError("canonical report path mismatch")
    mode = report.get("qualification_mode", {})
    if mode.get("enabled") is not False or mode.get("policy_qualification_status") != "not_run":
        raise ValueError("diagnostic cannot be qualification-enabled")
    checks = report["success_checks"]
    required = (
        "process_exit_zero", "no_traceback_or_error", "requested_iteration_reached",
        "checkpoint_exists", "requested_training_gpu_safety",
    )
    if not all(checks.get(name) is True for name in required):
        raise ValueError("training operational checks failed")
    if (report["task"], report["num_envs"], report["max_iterations"], report["seed"]) != (TASK, 1024, 50, 42):
        raise ValueError("rev32 training protocol mismatch")
    gate = report.get("training_safety_gate", {})
    if (report.get("headless") is not True or report.get("resume", {}).get("enabled") is not False
        or report.get("effective_hydra_overrides") != [] or report.get("last_iteration") != 49
        or report.get("iteration_target") != 50
        or not all(gate.get(key) is True for key in
                   ("requested", "required", "scratch_required", "requires_both_maximum_counts_zero"))):
        raise ValueError("scratch/headless/safety-gated smoke contract mismatch")
    bundle = report["source_bundle"]
    expected_paths = BASE_SOURCE_PATHS | {f"configs/g009_r0_{revision}_damping.json",
                                        f"scripts/bootstrap_train_g009_{revision}_damping.py",
                                        f"scripts/g009_r0_{revision}.py"}
    if revision == "rev33":
        expected_paths.add("scripts/g009_r0_rev32.py")
    if set(bundle["files"]) != expected_paths:
        raise ValueError("damping source manifest mismatch")
    if report["repository"]["dirty"] or not bundle["postrun"]["stable"]:
        raise ValueError("training source is not stable and clean")
    if (bundle.get("matches_repository_commit") is not True
        or bundle["prelaunch"]["repository_commit"] != report["repository"]["commit"]
        or bundle["postrun"]["repository_commit"] != report["repository"]["commit"]
        or bundle["prelaunch"]["files"] != bundle["files"] or bundle["postrun"]["files"] != bundle["files"]
        or bundle["prelaunch"]["sha256"] != bundle["sha256"] or bundle["postrun"]["sha256"] != bundle["sha256"]):
        raise ValueError("training source snapshot binding mismatch")
    validate_source_bundle(bundle)
    checkpoint = resolve_portable_path(report["artifacts"]["checkpoint"])
    if checkpoint.name != "model_49.pt" or file_sha256(checkpoint) != report["artifacts"]["checkpoint_sha256"]:
        raise ValueError("checkpoint identity mismatch")
    intervention_path = path.with_name(path.stem + "_intervention.json")
    intervention = json.loads(intervention_path.read_text(encoding="utf-8"))
    if (intervention.get("protocol") != f"g009_r0_{revision}_{group}_calf_damping_smoke_v1"
        or intervention.get("actuator_readback_stable") is not True
        or intervention.get("candidate_damping_n_m_s_rad") != 1.0
        or intervention.get("diagnostic_only") is not True or intervention.get("qualification_eligible") is not False
        or intervention.get("runtime") != {"task": TASK, "num_envs": 1024, "max_iterations": 50,
                                           "seed": 42, "device": "cuda:0", "run_name": run, "headless": True}):
        raise ValueError("damping intervention binding mismatch")
    before = intervention["actuator_before"]
    names = [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
    expected_gains = {name: 1.0 if name in ("RL_calf_joint", "RR_calf_joint") or
                      (revision == "rev33" and name.endswith("calf_joint")) else 0.5 for name in names}
    if (before != intervention["actuator_after"] or before.get("joint_names") != names
        or before.get("environments_checked") != 1024 or before.get("damping_by_joint") != expected_gains
        or before.get("stiffness_by_joint") != dict.fromkeys(names, 25.0)
        or before.get("effort_limit_by_joint") != dict.fromkeys(names, 23.5)):
        raise ValueError("intervention joint gain matrix mismatch")
    report["verified_intervention"] = {"path": portable_path(intervention_path),
                                       "sha256": file_sha256(intervention_path), "protocol": intervention["protocol"]}
    return report, checkpoint


def record(args: argparse.Namespace) -> dict:
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from rsl_rl.runners import OnPolicyRunner
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from isaac_walk_g009 import register_tasks

    training, checkpoint = validate_binding(args.training_report, args.revision)
    configure = configure_damping
    readback = actuator_readback
    if args.revision == "rev33":
        from g009_r0_rev33 import configure_damping as configure, actuator_readback as readback
    source_before = git_source_state()
    if not source_before["clean"]:
        raise ValueError("capture source must be clean outside reports/runs")
    checkpoint_hash = file_sha256(checkpoint)
    register_tasks()
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=4)
    cfg.seed = 42
    cfg.observations.policy.enable_corruption = False
    cfg.events.reset_base.params.update(assignment_mode="stratified", pose_xy_range=(0.0, 0.0), yaw_range=(0.0, 0.0))
    configure(cfg)
    selected = POSES.index(args.pose)
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = selected
    cfg.viewer.eye = (1.4, 1.4, 0.85)
    cfg.viewer.lookat = (0.0, 0.0, 0.24)
    cfg.viewer.resolution = RESOLUTION
    raw = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
    env = None
    encoder = None
    stem = f"g009_5_r0_diag_{args.revision}_{selected + 1:02d}_{args.pose}_hd_s42"
    video = OUTPUT / (stem + ".mp4")
    report_path = REPO_ROOT / "reports/runs" / (stem + ".json")
    if video.exists() or report_path.exists():
        raw.close()
        raise FileExistsError(stem)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    schedule = set(sample_steps(400))
    captured = []
    success = False
    reason = "capture_horizon"
    elapsed = 0
    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 30)
    try:
        live_before = readback(raw.unwrapped)
        agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        agent.device = args.device
        env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        observations, _ = env.get_observations()
        if int(raw.unwrapped._g009_recover_fall_class[selected].item()) != selected:
            raise RuntimeError("stratified pose readback mismatch")
        robot = raw.unwrapped.scene["robot"]
        physics = {
            "total_mass_kg": float(robot.root_physx_view.get_masses()[selected].sum().item()),
            "joint_position_limits_rad": robot.data.joint_pos_limits[selected].detach().cpu().tolist(),
            "foot_material_static_dynamic_friction": raw.unwrapped._g009_foot_material_readback[selected].detach().cpu().tolist(),
            "effective_foot_static_dynamic_friction": raw.unwrapped._g009_effective_foot_friction[selected].detach().cpu().tolist(),
            "effective_friction_valid": bool(raw.unwrapped._g009_effective_foot_friction_valid[selected].all().item()),
            "terrain_static_friction": cfg.scene.terrain.physics_material.static_friction,
            "terrain_dynamic_friction": cfg.scene.terrain.physics_material.dynamic_friction,
            "friction_combine_mode": cfg.scene.terrain.physics_material.friction_combine_mode,
        }
        controller = raw.unwrapped.viewport_camera_controller
        controller.update_view_to_asset_root("robot")
        controller.update_view_location(eye=cfg.viewer.eye, lookat=cfg.viewer.lookat)
        raw.render()  # Create the RGB annotator before warming it, without a physics step.
        for _ in range(4):
            raw.unwrapped.sim.render()
        encoder = subprocess.Popen([
            args.ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", "1920x1080", "-framerate", str(FPS),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
        ], stdin=subprocess.PIPE)

        def frame(step: int) -> None:
            pixels = raw.render()
            if pixels.shape != (1080, 1920, 3) or pixels.dtype != np.uint8:
                raise RuntimeError(f"unexpected native render {pixels.shape}/{pixels.dtype}")
            picture = Image.fromarray(pixels)
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, 1920, 96), fill=(15, 20, 30))
            draw.text((24, 8), f"DIAGNOSTIC / NOT QUALIFIED - G009 R0 {args.revision}", font=font, fill=(255, 200, 90))
            gain_label = "rear calf Kd=1.0" if args.revision == "rev32" else "all calf Kd=1.0"
            draw.text((24, 50), f"{args.pose} | {gain_label} | sim t={step * 0.02:.2f}s | checkpoint {checkpoint_hash[:12]}", font=font, fill="white")
            assert encoder is not None and encoder.stdin is not None
            encoder.stdin.write(picture.tobytes())
            captured.append(step)

        frame(0)
        for step in range(1, 401):
            with torch.inference_mode():
                observations, _, dones, _ = env.step(policy(observations))
            elapsed = step
            if bool(dones[selected].item()):
                active = [name for name in raw.unwrapped.termination_manager.active_terms
                          if bool(raw.unwrapped.termination_manager.get_term(name)[selected].item())]
                success = "stable_success" in active
                reason = "+".join(active) or "unknown"
                break  # Never show the automatic-reset state as recovery.
            if step in schedule:
                frame(step)
        live_after = readback(raw.unwrapped)
        validate_source_bundle(training["source_bundle"])
        source_after = git_source_state()
        if source_before["commit"] != source_after["commit"] or not source_after["clean"]:
            raise RuntimeError("capture source changed during playback")
        assert encoder.stdin is not None
        encoder.stdin.close()
        if encoder.wait(timeout=60) != 0:
            raise RuntimeError("video encoder failed")
        probe = json.loads(subprocess.check_output([
            args.ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
            "-of", "json", str(video),
        ], text=True))
        stream = probe["streams"][0]
        if (stream["width"], stream["height"], stream["avg_frame_rate"], int(stream["nb_read_frames"])) != (1920, 1080, "30/1", len(captured)):
            raise RuntimeError("encoded video does not match native frames")
        result = {
            "schema_version": "g009.r0.damping.hd_diagnostic.v1", "revision": args.revision,
            "status": "diagnostic_complete", "diagnostic_only": True,
            "qualification_eligible": False, "task": TASK, "pose": args.pose,
            "seed": 42, "headless": True, "actor_corruption": False,
            "elapsed_control_steps": elapsed, "stable_success": success, "termination_reason": reason,
            "initial_pose_xy_and_yaw": "zero", "training_pose_curriculum": "prone only in first 50 iterations",
            "training_report": {"path": portable_path(args.training_report), "sha256": file_sha256(args.training_report)},
            "training_intervention": training["verified_intervention"],
            "checkpoint": {"path": portable_path(checkpoint), "sha256": file_sha256(checkpoint)},
            "actuator_before": live_before, "actuator_after": live_after,
            "physics_readback": physics,
            "source_commit": source_before["commit"],
            "camera": {"resolution": list(RESOLUTION), "eye": list(cfg.viewer.eye), "lookat": list(cfg.viewer.lookat)},
            "recording": {"fps": FPS, "physics_dt_s": 0.005, "control_dt_s": 0.02,
                          "captured_control_steps": captured, "no_repeated_control_states": len(set(captured)) == len(captured),
                          "terminal_auto_reset_frame_excluded": True, "probe": probe},
            "local_video": {"path": portable_path(video), "sha256": file_sha256(video), "bytes": video.stat().st_size},
            "recorder_source_sha256": file_sha256(Path(__file__)),
        }
        report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(report_path), "video": str(video), "result": reason}), flush=True)
        return result
    finally:
        if encoder is not None and encoder.poll() is None:
            if encoder.stdin is not None and not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.wait(timeout=60)
        (env if env is not None else raw).close()


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--pose", choices=POSES, default="prone")
    parser.add_argument("--revision", choices=("rev32", "rev33"), default="rev32")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.headless or args.device != "cuda:0":
        raise ValueError("capture is headless CUDA only")
    validate_binding(args.training_report, args.revision)
    args.enable_cameras = True
    args.kit_args = "--/app/vulkan=false --/app/window/hideUi=true --/app/renderer/resolution/width=1920 --/app/renderer/resolution/height=1080"
    app = AppLauncher(args).app
    try:
        record(args)
        return 0
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
