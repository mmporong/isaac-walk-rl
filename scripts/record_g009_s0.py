#!/usr/bin/env python3
"""Record one G009 S0 slope profile as local-only off-screen video evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "configs" / "g009_s0.json"
DEFAULT_OUTPUT_DIR = Path.home() / "IsaacLab" / "logs" / "visual_evidence" / "g009" / "S0"
WINDOWS_KIT_ARGS = (
    "--/app/vulkan=false --/app/window/hideUi=true "
    "--/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720"
)
G009_CAPTURE_REPORT_DIRTY_ALLOWLIST = frozenset(
    {
        "reports/runs/g009_s0_slope_05_capture.json",
        "reports/runs/g009_s0_slope_15_capture.json",
        "reports/runs/g009_s0_slope_25_stress_capture.json",
    }
)
GEOMETRY_TOLERANCE_DEG = 1.0e-3
RESET_ALIGNMENT_TOLERANCE_DEG = 0.5


@dataclass(frozen=True)
class CaptureProfile:
    profile_id: str
    slope_deg: float
    terrain_azimuth_deg: float


@dataclass(frozen=True)
class SequenceSegment:
    name: str
    steps: int
    command: tuple[float, float, float]


@dataclass(frozen=True)
class CaptureContract:
    path: Path
    raw: dict[str, Any]
    task_id: str
    seed: int
    profiles: tuple[CaptureProfile, ...]
    sequence: tuple[SequenceSegment, ...]


@dataclass(frozen=True)
class SourceSnapshot:
    commit: str
    dirty_paths: tuple[str, ...]
    recorder_sha256: str
    config_sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(Path.home().resolve())
    except ValueError:
        return str(resolved)
    return "%USERPROFILE%\\" + str(relative)


def portable_repo_path(path: Path) -> str:
    try:
        return path.expanduser().resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside repository: {path}") from exc


def resolve_portable_path(value: str) -> Path:
    prefix = "%USERPROFILE%\\"
    if value.upper().startswith(prefix):
        suffix = value[len(prefix) :].replace("\\", os.sep).replace("/", os.sep)
        return (Path.home() / suffix).resolve()
    return Path(os.path.expandvars(value)).expanduser().resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_capture_contract(path: Path) -> CaptureContract:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "config schema_version must be 1")
    _require(payload.get("goal_id") == "g009" and payload.get("stage_id") == "S0", "config must be G009 S0")
    task_id = payload.get("task_id")
    _require(isinstance(task_id, str) and task_id, "task_id must be a non-empty string")
    visual = payload.get("visual_protocol")
    _require(isinstance(visual, dict), "visual_protocol must be an object")
    _require(visual.get("headless_offscreen") is True, "visual protocol must require headless off-screen capture")
    seed = visual.get("seed")
    _require(isinstance(seed, int) and not isinstance(seed, bool), "visual_protocol.seed must be an integer")

    raw_profiles = visual.get("profiles")
    _require(isinstance(raw_profiles, list) and len(raw_profiles) == 3, "visual protocol must define exactly three profiles")
    profiles = tuple(
        CaptureProfile(
            profile_id=str(item["profile_id"]),
            slope_deg=float(item["slope_deg"]),
            terrain_azimuth_deg=float(item["terrain_azimuth_deg"]),
        )
        for item in raw_profiles
    )
    _require(len({profile.profile_id for profile in profiles}) == len(profiles), "profile_id values must be unique")
    _require(all(0.0 <= profile.slope_deg < 90.0 for profile in profiles), "profile slopes must be in [0, 90)")

    raw_sequence = visual.get("sequence")
    _require(isinstance(raw_sequence, list) and raw_sequence, "visual protocol sequence must be non-empty")
    sequence = tuple(
        SequenceSegment(
            name=str(item["name"]),
            steps=int(item["steps"]),
            command=tuple(float(value) for value in item["command"]),
        )
        for item in raw_sequence
    )
    _require(all(segment.steps > 0 for segment in sequence), "sequence steps must be positive")
    _require(all(len(segment.command) == 3 for segment in sequence), "sequence commands must contain three values")
    moving = [segment for segment in sequence if segment.name in {"contour_left", "contour_right"}]
    _require(
        len(moving) == 2
        and moving[0].command[0] > 0.0
        and moving[1].command[0] < 0.0
        and all(value == 0.0 for segment in moving for value in segment.command[1:]),
        "sequence must encode contour-left/right as positive/negative body-x commands",
    )
    return CaptureContract(resolved, payload, task_id, seed, profiles, sequence)


def profile_by_id(contract: CaptureContract, profile_id: str) -> CaptureProfile:
    matches = [profile for profile in contract.profiles if profile.profile_id == profile_id]
    if len(matches) != 1:
        raise ValueError(f"capture profile not found or duplicated: {profile_id}")
    return matches[0]


def command_at_step(sequence: Iterable[SequenceSegment], step: int) -> tuple[int, SequenceSegment]:
    if step < 0:
        raise IndexError(step)
    cursor = 0
    for index, segment in enumerate(sequence):
        if step < cursor + segment.steps:
            return index, segment
        cursor += segment.steps
    raise IndexError(step)


def validate_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    canonical = DEFAULT_OUTPUT_DIR.resolve()
    try:
        resolved.relative_to(canonical)
    except ValueError as exc:
        raise ValueError(f"MP4 output must remain under {canonical}") from exc
    return resolved


def expected_video_path(output_dir: Path, profile: CaptureProfile, seed: int) -> Path:
    return validate_output_dir(output_dir) / f"g009_s0_{profile.profile_id}_s{seed}.mp4"


def validate_prelaunch_outputs(
    output_dir: Path,
    report_path: Path,
    profile: CaptureProfile,
    seed: int,
) -> Path:
    destination = expected_video_path(output_dir, profile, seed)
    prefix = f"g009-s0-{profile.profile_id}"
    stale_candidates = list(destination.parent.glob(f"{prefix}*.mp4")) if destination.parent.exists() else []
    if report_path.expanduser().resolve().exists():
        raise FileExistsError(report_path.expanduser().resolve())
    if destination.exists() or stale_candidates:
        raise FileExistsError(destination if destination.exists() else stale_candidates[0])
    return destination


def _write_json_atomic_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_source_state() -> tuple[str, list[str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    dirty_paths = filter_source_dirty_paths(
        line[3:].split(" -> ")[-1] for line in status if len(line) >= 4
    )
    _require(len(commit) == 40, "git source commit must be a full 40-character hash")
    return commit, dirty_paths


def capture_source_snapshot(config_path: Path) -> SourceSnapshot:
    commit, dirty_paths = _git_source_state()
    return SourceSnapshot(
        commit=commit,
        dirty_paths=tuple(dirty_paths),
        recorder_sha256=file_sha256(Path(__file__)),
        config_sha256=file_sha256(config_path.expanduser().resolve()),
    )


def require_clean_source_snapshot(snapshot: SourceSnapshot) -> None:
    if snapshot.dirty_paths:
        raise RuntimeError(
            "G009 capture requires a clean source tree apart from allowlisted prior capture reports: "
            + ", ".join(snapshot.dirty_paths)
        )


def verify_source_snapshot_unchanged(start: SourceSnapshot, config_path: Path) -> None:
    current = capture_source_snapshot(config_path)
    if current != start:
        raise RuntimeError(f"G009 capture source changed during recording: start={start!r}, end={current!r}")


def filter_source_dirty_paths(paths: Iterable[str]) -> list[str]:
    """Ignore only prior G009 capture JSONs during a sequential three-profile run."""
    normalized = {value.replace("\\", "/") for value in paths}
    return sorted(normalized - G009_CAPTURE_REPORT_DIRTY_ALLOWLIST)


def require_base_contact_termination(get_term: Any) -> None:
    """Fail before playback when the environment lacks the required fall signal."""
    try:
        term = get_term("base_contact")
    except (KeyError, AttributeError) as exc:
        raise RuntimeError("G009 S0 requires the base_contact termination term") from exc
    if term is None:
        raise RuntimeError("G009 S0 requires the base_contact termination term")


def _unit_vector(values: Iterable[float], label: str) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise RuntimeError(f"{label} must be a finite 3-vector")
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1.0e-12:
        raise RuntimeError(f"{label} must be non-zero")
    return tuple(value / length for value in vector)


def _vector_angle_error_deg(first: Iterable[float], second: Iterable[float]) -> float:
    first_unit = _unit_vector(first, "first vector")
    second_unit = _unit_vector(second, "second vector")
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(first_unit, second_unit))))
    return math.degrees(math.acos(dot))


def validate_terrain_readback(
    readback: dict[str, Any],
    profile: CaptureProfile,
    *,
    tolerance_deg: float = GEOMETRY_TOLERANCE_DEG,
) -> dict[str, float]:
    measured_slope = float(readback.get("measured_slope_deg", math.nan))
    if not math.isfinite(measured_slope):
        raise RuntimeError("terrain measured_slope_deg is missing or non-finite")
    requested_normal = _terrain_axes_3d(profile.slope_deg, profile.terrain_azimuth_deg)["normal"]
    measured_normal = _unit_vector(readback.get("first_triangle_normal_w", ()), "first_triangle_normal_w")
    slope_error = abs(measured_slope - profile.slope_deg)
    normal_error = _vector_angle_error_deg(measured_normal, requested_normal)
    measured_azimuth = math.degrees(math.atan2(-measured_normal[1], -measured_normal[0])) % 360.0
    requested_azimuth = profile.terrain_azimuth_deg % 360.0
    azimuth_error = abs((measured_azimuth - requested_azimuth + 180.0) % 360.0 - 180.0)
    errors = {
        "slope_error_deg": slope_error,
        "azimuth_error_deg": azimuth_error,
        "normal_error_deg": normal_error,
    }
    failed = {name: value for name, value in errors.items() if value > tolerance_deg}
    if failed:
        raise RuntimeError(f"terrain geometry readback is outside {tolerance_deg} deg tolerance: {failed}")
    return errors


def validate_reset_alignment(
    *,
    support_normal_error_deg: float,
    body_up_error_deg: float,
    body_x_contour_left_error_deg: float,
    tolerance_deg: float = RESET_ALIGNMENT_TOLERANCE_DEG,
) -> None:
    errors = {
        "support_normal_error_deg": float(support_normal_error_deg),
        "body_up_error_deg": float(body_up_error_deg),
        "body_x_contour_left_error_deg": float(body_x_contour_left_error_deg),
    }
    if not all(math.isfinite(value) for value in errors.values()):
        raise RuntimeError(f"reset alignment contains non-finite values: {errors}")
    failed = {name: value for name, value in errors.items() if value > tolerance_deg}
    if failed:
        raise RuntimeError(f"reset alignment is outside {tolerance_deg} deg tolerance: {failed}")


def build_capture_report(
    *,
    contract: CaptureContract,
    profile: CaptureProfile,
    source_commit: str,
    dirty_paths: list[str],
    headless: bool,
    step_dt_s: float,
    camera: dict[str, Any],
    checkpoint: dict[str, Any],
    physics_readback: dict[str, Any],
    metrics: dict[str, Any],
    local_video: dict[str, Any],
    record_source_sha256: str,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    """Assemble the exact fail-closed capture surface consumed by the S0 builder."""
    config_path = portable_repo_path(contract.path)
    if config_path != "configs/g009_s0.json":
        raise ValueError("capture config must be configs/g009_s0.json")
    return {
        "schema_version": 1,
        "goal_id": "g009",
        "stage_id": "S0",
        "status": "complete",
        "source_commit": source_commit,
        "dirty_paths": list(dirty_paths),
        "scope": {
            "claim": "S0 wiring and qualitative visual playback evidence only",
            "walk_qualification": False,
            "policy_success_claimed": False,
        },
        "profile": {
            "profile_id": profile.profile_id,
            "slope_deg": profile.slope_deg,
            "terrain_azimuth_deg": profile.terrain_azimuth_deg,
            "seed": contract.seed,
            "headless": headless,
            "step_dt_s": step_dt_s,
            "total_steps": sum(item.steps for item in contract.sequence),
            "sequence": [
                {"name": item.name, "steps": item.steps, "command": list(item.command)}
                for item in contract.sequence
            ],
            "camera": camera,
        },
        "config": {"path": config_path, "sha256": config_sha256 or file_sha256(contract.path)},
        "checkpoint": checkpoint,
        "physics_readback": physics_readback,
        "metrics": metrics,
        "local_video": local_video,
        "record_source_sha256": record_source_sha256,
    }


def _terrain_axes_3d(slope_deg: float, azimuth_deg: float) -> dict[str, tuple[float, float, float]]:
    slope = math.radians(slope_deg)
    azimuth = math.radians(azimuth_deg)
    uphill = (math.cos(slope) * math.cos(azimuth), math.cos(slope) * math.sin(azimuth), math.sin(slope))
    downhill = tuple(-value for value in uphill)
    contour_left = (-math.sin(azimuth), math.cos(azimuth), 0.0)
    normal = (-math.sin(slope) * math.cos(azimuth), -math.sin(slope) * math.sin(azimuth), math.cos(slope))
    return {"uphill": uphill, "downhill": downhill, "contour_left": contour_left, "normal": normal}


def _physics_readback(surface_path: str) -> dict[str, Any]:
    import isaacsim.core.utils.stage as stage_utils
    from pxr import UsdGeom, UsdPhysics, UsdShade

    stage = stage_utils.get_current_stage()
    surface = stage.GetPrimAtPath(surface_path)
    if not surface.IsValid() or not surface.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"missing G009 USD mesh: {surface_path}")
    field_root = str(Path(surface_path).parent).replace("\\", "/")
    meshes = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()).startswith(field_root + "/")]
    if meshes != [surface_path]:
        raise RuntimeError(f"expected one G009 terrain mesh, found {meshes}")
    if not surface.HasAPI(UsdPhysics.CollisionAPI) or not surface.HasAPI(UsdPhysics.MeshCollisionAPI):
        raise RuntimeError("G009 terrain mesh is missing collision APIs")
    material_path = UsdShade.MaterialBindingAPI(surface).GetDirectBinding("physics").GetMaterialPath()
    material = stage.GetPrimAtPath(material_path)
    if not material.IsValid():
        raise RuntimeError("G009 terrain physics material binding is missing")
    mesh = UsdGeom.Mesh(surface)
    points = mesh.GetPointsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    first, second, third = (points[indices[index]] for index in range(3))
    edge_a = tuple(float(second[index] - first[index]) for index in range(3))
    edge_b = tuple(float(third[index] - first[index]) for index in range(3))
    normal = (
        edge_a[1] * edge_b[2] - edge_a[2] * edge_b[1],
        edge_a[2] * edge_b[0] - edge_a[0] * edge_b[2],
        edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0],
    )
    normal_length = math.sqrt(sum(value * value for value in normal))
    normal = tuple(value / normal_length for value in normal)
    if normal[2] < 0.0:
        normal = tuple(-value for value in normal)
    return {
        "surface_prim": surface_path,
        "single_mesh": True,
        "mesh_count": len(meshes),
        "face_count": len(mesh.GetFaceVertexCountsAttr().Get()),
        "point_count": len(points),
        "first_triangle_normal_w": list(normal),
        "measured_slope_deg": math.degrees(math.acos(max(-1.0, min(1.0, normal[2])))),
        "collision_api": True,
        "mesh_collision_api": True,
        "material_path": str(material_path),
        "static_friction": float(material.GetAttribute("physics:staticFriction").Get()),
        "dynamic_friction": float(material.GetAttribute("physics:dynamicFriction").Get()),
        "friction_combine_mode": str(material.GetAttribute("physxMaterial:frictionCombineMode").Get()),
    }


def _record(
    args: argparse.Namespace,
    contract: CaptureContract,
    profile: CaptureProfile,
    source_snapshot: SourceSnapshot,
) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils import math as math_utils
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
    from isaac_walk_g009 import register_tasks

    register_tasks()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_checkpoint = contract.raw["parent_checkpoint"]
    expected_checkpoint_path = resolve_portable_path(expected_checkpoint["path"])
    if checkpoint != expected_checkpoint_path:
        raise ValueError("checkpoint path does not match configs/g009_s0.json")
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != expected_checkpoint["sha256"]:
        raise ValueError("checkpoint SHA-256 does not match configs/g009_s0.json")

    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = expected_video_path(output_dir, profile, contract.seed)
    prefix = f"g009-s0-{profile.profile_id}"
    stale_candidates = list(output_dir.glob(f"{prefix}*.mp4"))
    if destination.exists() or stale_candidates:
        raise FileExistsError(destination if destination.exists() else stale_candidates[0])

    total_steps = sum(segment.steps for segment in contract.sequence)
    env_cfg = parse_env_cfg(contract.task_id, device=args.device, num_envs=1)
    env_cfg.seed = contract.seed
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), (total_steps + 2) * float(env_cfg.sim.dt * env_cfg.decimation))
    terrain_cfg = env_cfg.scene.slope_field.spawn
    terrain = contract.raw["terrain"]
    terrain_cfg.x_min_m, terrain_cfg.x_max_m = map(float, terrain["bounds_m"]["x"])
    terrain_cfg.y_min_m, terrain_cfg.y_max_m = map(float, terrain["bounds_m"]["y"])
    terrain_cfg.cell_size_m = float(terrain["cell_size_m"])
    terrain_cfg.slope_deg = profile.slope_deg
    terrain_cfg.azimuth_deg = profile.terrain_azimuth_deg
    terrain_cfg.seed = int(contract.raw["analytic_gate"]["terrain_seed"])
    terrain_cfg.residual_amplitude_m = float(contract.raw["analytic_gate"]["residual_amplitude_m"])
    ground_material = terrain["ground_material"]
    terrain_cfg.static_friction = (float(ground_material["static_friction"]),)
    terrain_cfg.dynamic_friction = (float(ground_material["dynamic_friction"]),)
    reset_yaw = math.radians(float(contract.raw["reset"]["visual_yaw_about_support_normal_deg"]))
    env_cfg.events.reset_base.params["pose_range"]["yaw"] = (reset_yaw, reset_yaw)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (4.0, 4.0, 2.8)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.35)
    if hasattr(env_cfg.viewer, "resolution"):
        env_cfg.viewer.resolution = (1280, 720)

    agent_cfg = load_cfg_from_registry(contract.task_id, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = contract.seed
    agent_cfg.device = args.device
    raw_env = gym.make(contract.task_id, cfg=env_cfg, render_mode="rgb_array")
    require_base_contact_termination(raw_env.unwrapped.termination_manager.get_term)
    readback = _physics_readback(terrain["surface_prim"])
    geometry_errors = validate_terrain_readback(readback, profile)
    controller = raw_env.unwrapped.viewport_camera_controller
    if controller is not None:
        controller.update_view_location(eye=(4.0, 4.0, 2.8), lookat=(0.0, 0.0, 0.35))
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
    robot = env.unwrapped.scene["robot"]
    command_buffer = env.unwrapped.command_manager.get_command("base_velocity")
    axes = _terrain_axes_3d(profile.slope_deg, profile.terrain_azimuth_deg)
    downhill_axis = torch.tensor(axes["downhill"], dtype=torch.float32, device=env.unwrapped.device)
    contour_axis = torch.tensor(axes["contour_left"], dtype=torch.float32, device=env.unwrapped.device)
    requested_normal = torch.tensor(axes["normal"], dtype=torch.float32, device=env.unwrapped.device)
    reset_log = env.unwrapped.extras.get("g009_reset", {})
    reset_normal = reset_log.get("support_normal_w")
    if reset_normal is None:
        raise RuntimeError("G009 reset support-normal readback is missing")
    reset_normal = reset_normal[0]
    reset_body_up = math_utils.quat_apply(robot.data.root_quat_w[:1], torch.tensor([[0.0, 0.0, 1.0]], device=env.unwrapped.device))[0]
    reset_body_x = math_utils.quat_apply(robot.data.root_quat_w[:1], torch.tensor([[1.0, 0.0, 0.0]], device=env.unwrapped.device))[0]

    def angle_error(first, second) -> float:
        return float(torch.rad2deg(torch.acos(torch.clamp(torch.dot(first, second), -1.0, 1.0))).item())

    reset_errors = {
        "support_normal_error_deg": angle_error(reset_normal, requested_normal),
        "body_up_error_deg": angle_error(reset_body_up, reset_normal),
        "body_x_contour_left_error_deg": angle_error(reset_body_x, contour_axis),
    }
    validate_reset_alignment(**reset_errors)

    segment_values = [{"contour": [], "downhill": []} for _ in contract.sequence]
    initial_position = robot.data.root_pos_w[0].detach().clone()
    downhill_displacements: list[float] = []
    tilt_max = 0.0
    fall = False
    terminated = False
    first_fall_step: int | None = None
    obs, _ = env.get_observations()
    try:
        for step in range(total_steps):
            segment_index, segment = command_at_step(contract.sequence, step)
            command_buffer[0].copy_(torch.tensor(segment.command, dtype=torch.float32, device=env.unwrapped.device))
            obs, _ = env.get_observations()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            velocity_w = robot.data.root_lin_vel_w[0]
            segment_values[segment_index]["contour"].append(float(torch.dot(velocity_w, contour_axis).item()))
            segment_values[segment_index]["downhill"].append(float(torch.dot(velocity_w, downhill_axis).item()))
            displacement = robot.data.root_pos_w[0] - initial_position
            downhill_displacements.append(float(torch.dot(displacement, downhill_axis).item()))
            body_up = math_utils.quat_apply(robot.data.root_quat_w[:1], torch.tensor([[0.0, 0.0, 1.0]], device=env.unwrapped.device))[0]
            tilt = torch.rad2deg(torch.acos(torch.clamp(torch.dot(body_up, requested_normal), -1.0, 1.0)))
            tilt_max = max(tilt_max, float(tilt.item()))
            base_contact = bool(env.unwrapped.termination_manager.get_term("base_contact")[0].item())
            fall = fall or base_contact
            terminated = terminated or bool(dones[0].item())
            if base_contact and first_fall_step is None:
                first_fall_step = step
    finally:
        env.close()

    candidates = sorted(output_dir.glob(f"{prefix}*.mp4"), key=lambda path: path.stat().st_mtime)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one MP4 for {profile.profile_id}, found {candidates}")
    candidates[0].rename(destination)

    segment_metrics = []
    for segment, values in zip(contract.sequence, segment_values):
        segment_metrics.append(
            {
                "name": segment.name,
                "steps": segment.steps,
                "root_velocity_world_projection_mean_mps": {
                    "contour_left": sum(values["contour"]) / len(values["contour"]),
                    "downhill": sum(values["downhill"]) / len(values["downhill"]),
                },
            }
        )
    physics_readback = {
        **readback,
        "geometry_validation": {**geometry_errors, "tolerance_deg": GEOMETRY_TOLERANCE_DEG},
        "slope_deg": profile.slope_deg,
        "terrain_azimuth_deg": profile.terrain_azimuth_deg,
        "ground_material": {
            "static_friction": readback["static_friction"],
            "dynamic_friction": readback["dynamic_friction"],
            "friction_combine_mode": readback["friction_combine_mode"],
        },
        "requested_geometry": {
            "slope_deg": profile.slope_deg,
            "terrain_azimuth_deg": profile.terrain_azimuth_deg,
            "support_normal_w": list(axes["normal"]),
        },
        "reset": {
            "yaw_about_support_normal_deg": math.degrees(reset_yaw),
            "support_normal_w": [float(value) for value in reset_normal.tolist()],
            **reset_errors,
            "tolerance_deg": RESET_ALIGNMENT_TOLERANCE_DEG,
        },
    }
    return build_capture_report(
        contract=contract,
        profile=profile,
        source_commit=source_snapshot.commit,
        dirty_paths=list(source_snapshot.dirty_paths),
        headless=bool(args.headless),
        step_dt_s=float(env_cfg.sim.dt * env_cfg.decimation),
        camera={
            "resolution": [1280, 720],
            "origin_type": "env",
            "env_index": 0,
            "eye": [4.0, 4.0, 2.8],
            "lookat": [0.0, 0.0, 0.35],
        },
        checkpoint={"path": expected_checkpoint["path"], "sha256": checkpoint_hash},
        physics_readback=physics_readback,
        metrics={
            "segments": segment_metrics,
            "downhill_drift_m": {"final": downhill_displacements[-1], "max": max(downhill_displacements)},
            "support_normal_relative_body_tilt_max_deg": tilt_max,
            "termination": {"terminated": terminated, "fall": fall, "first_fall_step": first_fall_step},
        },
        local_video={
            "path": portable_path(destination),
            "sha256": file_sha256(destination),
            "bytes": destination.stat().st_size,
            "git_policy": "local_only",
        },
        record_source_sha256=source_snapshot.recorder_sha256,
        config_sha256=source_snapshot.config_sha256,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.enable_cameras = True
    if sys.platform == "win32" and args.headless and not args.kit_args:
        args.kit_args = WINDOWS_KIT_ARGS
    return args


def main(argv: list[str] | None = None) -> int:
    from isaaclab.app import AppLauncher

    args = _parse_args(argv)
    contract = load_capture_contract(args.config)
    profile = profile_by_id(contract, args.profile)
    if not args.headless:
        raise ValueError("G009 S0 visual protocol requires --headless off-screen capture")
    validate_prelaunch_outputs(args.output_dir, args.report, profile, contract.seed)
    source_snapshot = capture_source_snapshot(contract.path)
    require_clean_source_snapshot(source_snapshot)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = _record(args, contract, profile, source_snapshot)
        verify_source_snapshot_unchanged(source_snapshot, contract.path)
        _write_json_atomic_new(args.report.expanduser().resolve(), report)
        print(json.dumps({"report": str(args.report.resolve()), "video": report["local_video"]}), flush=True)
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
