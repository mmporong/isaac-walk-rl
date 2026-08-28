#!/usr/bin/env python3
"""Strict Isaac Sim runtime calibration for G009 R0 recovery.

``passed`` means only that the runtime contract passed.  This probe never
qualifies a learned checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaac_walk_g009.recover_contracts import PPO_GAMMA

DEFAULT_TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
POSE_COUNT = 4
ACTION_MODES = ("zero_normalized", "reset_pose_hold")
EXPECTED_NUM_ENVS = 8
EXPECTED_ROLLOUT_STEPS = 150
TAIL_STEPS = 25
ACTOR_FOOT_LOAD_SLICE = slice(49, 53)
ACTOR_RANGE_SLICE = slice(53, 68)
ACTOR_RANGE_MASK_SLICE = slice(68, 83)
CRITIC_TERRAIN_NORMAL_SLICE = slice(83, 86)
CRITIC_BASE_HEIGHT_INDEX = 86

MIN_ROOT_HEIGHT_M = 0.02
MIN_CONTACT_SEPARATION_M = -0.01
MAX_NON_FOOT_FORCE_BODYWEIGHTS = 15.0
MAX_EXCESS_CONTACT_DELTA_V_M_S = 3.0
MAX_TAIL_HORIZONTAL_SPEED_M_S = 0.50
MAX_TAIL_VERTICAL_SPEED_M_S = 0.25
MAX_TAIL_ANGULAR_SPEED_RAD_S = 2.0
CONTACT_EXERCISE_THRESHOLD_N = 1.0
RESET_HOLD_TARGET_TOLERANCE_RAD = 1.0e-6

SOURCE_BINDING_PATHS = (
    "configs/g009_r0.json",
    "scripts/bootstrap_train_g009.py",
    "scripts/probe_g009_recover_runtime.py",
    "scripts/run_training.ps1",
    "scripts/sync_g009_r0_contract.py",
    "scripts/synthesize_g009_r0_probe.py",
    "src/isaac_walk_g009/agent_cfg.py",
    "src/isaac_walk_g009/mdp/__init__.py",
    "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/mdp/recover.py",
    "src/isaac_walk_g009/recover_contracts.py",
    "src/isaac_walk_g009/recover_env_cfg.py",
    "src/isaac_walk_g009/registry.py",
)


def _progress(stage: str) -> None:
    print(json.dumps({"stage": stage}, ensure_ascii=False), flush=True)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {path}")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite existing temporary report: {temporary}")

    created_temporary = False
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            created_temporary = True
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if created_temporary and temporary.exists():
            temporary.unlink()


def canonical_report_output(output: Path) -> tuple[Path, str]:
    """Resolve one new direct-child JSON report under the canonical run folder."""

    reports_root = (REPO_ROOT / "reports" / "runs").resolve()
    resolved = output.expanduser().resolve()
    if resolved.parent != reports_root:
        raise ValueError("output must be a direct child of the canonical reports/runs directory")
    if resolved.suffix != ".json" or resolved.name == ".json":
        raise ValueError("output must use a non-empty .json filename")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {resolved}")
    return resolved, resolved.relative_to(REPO_ROOT.resolve()).as_posix()


def validate_execution_metadata(
    execution: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Validate fresh-run identity and its exact canonical output binding."""

    expected_keys = {
        "execution_id",
        "started_at_utc",
        "output_path_repo_relative",
        "no_overwrite",
    }
    if set(execution) != expected_keys:
        raise ValueError("execution metadata keys do not match the provenance contract")
    execution_id = execution["execution_id"]
    if not isinstance(execution_id, str):
        raise ValueError("execution_id must be a UUID4 hex string")
    try:
        parsed_uuid = uuid.UUID(hex=execution_id)
    except (ValueError, AttributeError) as error:
        raise ValueError("execution_id must be a UUID4 hex string") from error
    if parsed_uuid.version != 4 or parsed_uuid.hex != execution_id:
        raise ValueError("execution_id must be a lowercase UUID4 hex string")

    started_at_utc = execution["started_at_utc"]
    if not isinstance(started_at_utc, str) or not started_at_utc.endswith("Z"):
        raise ValueError("started_at_utc must be an ISO-8601 UTC timestamp")
    try:
        parsed_time = datetime.fromisoformat(started_at_utc.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError("started_at_utc must be an ISO-8601 UTC timestamp") from error
    if parsed_time.utcoffset() != timezone.utc.utcoffset(parsed_time):
        raise ValueError("started_at_utc must use UTC")

    resolved, expected_relative = canonical_report_output(output)
    if execution["output_path_repo_relative"] != expected_relative:
        raise ValueError("execution output binding does not match the requested report path")
    if execution["no_overwrite"] is not True:
        raise ValueError("execution must declare no_overwrite=true")
    return execution


def prepare_execution(output: Path) -> tuple[Path, dict[str, Any]]:
    """Create fresh provenance before Isaac Sim or AppLauncher is initialized."""

    resolved, relative = canonical_report_output(output)
    execution = {
        "execution_id": uuid.uuid4().hex,
        "started_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "output_path_repo_relative": relative,
        "no_overwrite": True,
    }
    return resolved, validate_execution_metadata(execution, resolved)


def parse_prelaunch_output(argv: list[str] | None = None) -> Path:
    """Read only --output so provenance rejection precedes AppLauncher creation."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True, type=Path)
    args, _ = parser.parse_known_args(argv)
    return args.output


def source_bundle_provenance() -> dict[str, Any]:
    """Bind a runtime probe to committed files, not only to the contract payload."""

    files: dict[str, str] = {}
    missing: list[str] = []
    for relative_path in SOURCE_BINDING_PATHS:
        path = REPO_ROOT / relative_path
        if not path.is_file():
            missing.append(relative_path)
            continue
        files[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()

    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    bundle_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest() if files else None
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *SOURCE_BINDING_PATHS,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "schema_version": 1,
        "git_commit": commit,
        "git_commit_valid": bool(re.fullmatch(r"[0-9a-f]{40}", commit)),
        "source_binding_paths": list(SOURCE_BINDING_PATHS),
        "source_binding_files": files,
        "source_bundle_sha256": bundle_sha256,
        "all_files_present": not missing and len(files) == len(SOURCE_BINDING_PATHS),
        "missing_files": missing,
        "clean": not status,
        "dirty_source_paths": status,
    }


def _tensor_list(value) -> list[Any]:
    return value.detach().cpu().tolist()


def validate_calibration_budget(num_envs: int, rollout_steps: int) -> None:
    """Reject cheaper probes that cannot cover two modes for three seconds."""

    if num_envs != EXPECTED_NUM_ENVS:
        raise ValueError(f"G009 R0 calibration requires exactly {EXPECTED_NUM_ENVS} environments")
    if rollout_steps != EXPECTED_ROLLOUT_STEPS:
        raise ValueError(
            f"G009 R0 calibration requires exactly {EXPECTED_ROLLOUT_STEPS} control steps"
        )


def reward_temporal_expectations(step_dt_s: float) -> dict[str, float]:
    """Return exact contributions after Isaac RewardManager applies ``dt``."""

    if step_dt_s <= 0.0 or not math.isfinite(step_dt_s):
        raise ValueError("step_dt_s must be finite and positive")
    return {
        "step_dt_s": step_dt_s,
        "terminal_raw_pulse": 1.0 / step_dt_s,
        "terminal_weight": 10.0,
        "terminal_contribution": (1.0 / step_dt_s) * 10.0 * step_dt_s,
        "potential_delta": PPO_GAMMA * 0.25,
        "potential_raw_rate": PPO_GAMMA * 0.25 / step_dt_s,
        "potential_weight": 2.0,
        "potential_contribution": (PPO_GAMMA * 0.25 / step_dt_s) * 2.0 * step_dt_s,
    }


def _inverse_map_position_targets(target, soft_limits, *, action_scale: float = 1.0):
    """Map joint targets back to normalized limit-rescaled actions."""

    if not math.isfinite(action_scale) or not 0.0 < action_scale <= 1.0:
        raise ValueError("action_scale must be finite and in (0, 1]")
    span = soft_limits[..., 1] - soft_limits[..., 0]
    if bool((span <= 0.0).any().item()):
        raise RuntimeError("soft joint position limits must have positive span")
    scaled_action = 2.0 * (target - soft_limits[..., 0]) / span - 1.0
    return (scaled_action / action_scale).clamp(-1.0, 1.0)


def reset_pose_hold_action_diagnostics(
    target,
    soft_limits,
    joint_names: list[str],
    *,
    action_scale: float,
):
    """Describe inverse-map clipping and the target that the action term can reach."""

    if target.ndim != 2 or soft_limits.shape != (*target.shape, 2):
        raise ValueError("target and soft_limits must have shapes (N, J) and (N, J, 2)")
    if len(joint_names) != target.shape[1]:
        raise ValueError("joint_names must match the target joint dimension")
    if not math.isfinite(action_scale) or not 0.0 < action_scale <= 1.0:
        raise ValueError("action_scale must be finite and in (0, 1]")
    span = soft_limits[..., 1] - soft_limits[..., 0]
    if bool((span <= 0.0).any().item()):
        raise RuntimeError("soft joint position limits must have positive span")

    scaled_action = 2.0 * (target - soft_limits[..., 0]) / span - 1.0
    unclamped_normalized_action = scaled_action / action_scale
    normalized_action = unclamped_normalized_action.clamp(-1.0, 1.0)
    saturated_mask = unclamped_normalized_action.abs() > 1.0
    reachable_scaled_action = (normalized_action * action_scale).clamp(-1.0, 1.0)
    reachable_target = soft_limits[..., 0] + (reachable_scaled_action + 1.0) * 0.5 * span
    target_error = (reachable_target - target).abs()
    max_target_error, max_target_error_joint_index = target_error.max(dim=1)

    return {
        "unclamped_normalized_action": unclamped_normalized_action,
        "normalized_action": normalized_action,
        "saturated_mask": saturated_mask,
        "reachable_target": reachable_target,
        "target_error": target_error,
        "max_target_error": max_target_error,
        "max_target_error_joint_index": max_target_error_joint_index,
        "max_target_error_joint_name": [
            joint_names[index]
            for index in max_target_error_joint_index.detach().cpu().tolist()
        ],
        "saturated_joint_names": [
            [joint_names[index] for index in row.nonzero(as_tuple=False).flatten().cpu().tolist()]
            for row in saturated_mask
        ],
    }


def reset_pose_hold_checks(diagnostics: dict[str, Any]) -> dict[str, bool]:
    """Return blocking checks for hold-action reachability diagnostics."""

    required_tensors = (
        "unclamped_normalized_action",
        "normalized_action",
        "reachable_target",
        "target_error",
        "max_target_error",
    )
    finite = all(
        bool(diagnostics[name].isfinite().all().item()) for name in required_tensors
    )
    return {
        "reset_pose_hold_action_diagnostics_finite": finite,
        "reset_pose_hold_actions_unsaturated": finite
        and bool((~diagnostics["saturated_mask"]).all().item()),
        "reset_pose_hold_reachable_targets_match_reset_positions": finite
        and bool(
            (
                diagnostics["max_target_error"]
                <= RESET_HOLD_TARGET_TOLERANCE_RAD
            ).all().item()
        ),
    }


def peak_body_attribution_complete(
    max_force,
    physics_step,
    body_index,
    *,
    nonfoot_ids: list[int],
    body_count: int,
) -> bool:
    """Fail closed unless every observed peak names a valid non-foot sensor body."""

    if max_force.shape != physics_step.shape or max_force.shape != body_index.shape:
        raise ValueError("peak attribution tensors must have identical shapes")
    valid_nonfoot_ids = set(nonfoot_ids)
    for force, step, index in zip(
        max_force.detach().cpu().tolist(),
        physics_step.detach().cpu().tolist(),
        body_index.detach().cpu().tolist(),
    ):
        if not math.isfinite(force) or force < 0.0:
            return False
        if force == 0.0:
            if step != -1 or index != -1:
                return False
            continue
        if step < 1 or index < 0 or index >= body_count or index not in valid_nonfoot_ids:
            return False
    return True


def nonfoot_body_indices_from_flat(flat_indices, nonfoot_ids: list[int]):
    """Decode flattened ``(history, non-foot body)`` maxima to sensor body ids."""

    if not nonfoot_ids:
        raise ValueError("nonfoot_ids must not be empty")
    lookup = flat_indices.new_tensor(nonfoot_ids)
    return lookup[flat_indices % len(nonfoot_ids)]


def _actuator_joint_limits(robot, attribute: str, torch):
    """Assemble explicit actuator limits in articulation joint order."""

    limits = torch.full(
        (robot.num_instances, robot.num_joints),
        float("nan"),
        device=robot.device,
    )
    for actuator in robot.actuators.values():
        joint_ids = actuator.joint_indices
        value = getattr(actuator, attribute)
        if not isinstance(value, torch.Tensor):
            value = torch.full(
                (robot.num_instances, actuator.num_joints),
                float(value),
                device=robot.device,
            )
        elif value.ndim == 1:
            value = value.unsqueeze(0).expand(robot.num_instances, -1)
        limits[:, joint_ids] = value
    if not bool(torch.isfinite(limits).all().item()) or bool((limits <= 0.0).any().item()):
        raise RuntimeError(f"invalid actuator {attribute} readback")
    return limits


def pose_mode_rows(pose_names: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "env_index": index,
            "pose_id": pose_names[index % POSE_COUNT],
            "action_mode": ACTION_MODES[index // POSE_COUNT],
        }
        for index in range(EXPECTED_NUM_ENVS)
    ]


def summarize_status(checks: dict[str, bool], health_names: tuple[str, ...]) -> dict[str, Any]:
    """Separate process health, runtime-contract verdict, and policy qualification."""

    missing = [name for name in health_names if name not in checks]
    if missing:
        raise KeyError(f"missing run-health checks: {missing}")
    run_health_passed = all(checks[name] for name in health_names)
    runtime_passed = all(checks.values())
    return {
        "run_health": {"passed": run_health_passed, "check_names": list(health_names)},
        "runtime_contract": {"passed": runtime_passed, "blocking_checks": list(checks)},
        "qualification": {
            "status": "not_run",
            "passed": None,
            "reason": "runtime calibration does not evaluate a learned checkpoint",
        },
        "passed_semantics": "runtime_contract_only_not_policy_qualification",
        "passed": runtime_passed,
    }


def contact_exercise_checks(foot_exercised, nonfoot_exercised) -> dict[str, bool]:
    """Require contact per environment without prescribing the supporting body."""

    if foot_exercised.shape != nonfoot_exercised.shape:
        raise ValueError("foot and non-foot contact exercise shapes must match")
    return {
        "at_least_one_contact_type_exercised_per_pose_mode": bool(
            (foot_exercised | nonfoot_exercised).all().item()
        ),
        "foot_contact_exercised_globally": bool(foot_exercised.any().item()),
        "nonfoot_contact_exercised_globally": bool(
            nonfoot_exercised.any().item()
        ),
    }


def camera_observation_checks(foot_load, body_range, hit_mask) -> dict[str, Any]:
    """Validate the fixed P-RECOVER-83 camera/load schema at reset."""

    if foot_load.shape != (EXPECTED_NUM_ENVS, 4):
        raise ValueError("foot load must have shape (8, 4)")
    if body_range.shape != (EXPECTED_NUM_ENVS, 15):
        raise ValueError("body range must have shape (8, 15)")
    if hit_mask.shape != (EXPECTED_NUM_ENVS, 15):
        raise ValueError("body range hit mask must have shape (8, 15)")
    mask_binary = (hit_mask == 0.0) | (hit_mask == 1.0)
    no_hit = hit_mask == 0.0
    hit_counts = hit_mask.sum(dim=1)
    return {
        "checks": {
            "actor_foot_load_finite_nonnegative": bool(
                (foot_load >= 0.0).all().item()
                and foot_load.isfinite().all().item()
            ),
            "body_range_finite_unit_interval": bool(
                body_range.isfinite().all().item()
                and (body_range >= 0.0).all().item()
                and (body_range <= 1.0).all().item()
            ),
            "body_range_mask_binary": bool(mask_binary.all().item()),
            "body_range_no_hit_is_one": bool(
                (body_range[no_hit] == 1.0).all().item()
            ),
            "prone_has_at_least_one_camera_hit_both_modes": bool(
                (hit_counts[[0, 4]] >= 1.0).all().item()
            ),
            "supine_has_zero_camera_hits_both_modes": bool(
                (hit_counts[[1, 5]] == 0.0).all().item()
            ),
        },
        "hit_count_per_env": [int(value) for value in hit_counts.tolist()],
    }


def camera_config_readback(camera_cfg) -> dict[str, Any]:
    """Read and validate the fixed rev3 body-mounted range camera contract."""

    position = tuple(float(value) for value in camera_cfg.offset.pos)
    rotation = tuple(float(value) for value in camera_cfg.offset.rot)
    expected_rotation = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
    pattern = camera_cfg.pattern_cfg
    checks = {
        "body_range_camera_attached_to_base": str(camera_cfg.prim_path).endswith(
            "/Robot/base"
        ),
        "body_range_camera_targets_ground": list(camera_cfg.mesh_prim_paths)
        == ["/World/ground"],
        "body_range_camera_distance_output": list(camera_cfg.data_types)
        == ["distance_to_camera"],
        "body_range_camera_pattern_5x3": int(pattern.width) == 5
        and int(pattern.height) == 3,
        "body_range_camera_max_distance_1m": math.isclose(
            float(camera_cfg.max_distance), 1.0, abs_tol=1.0e-9
        ),
        "body_range_camera_offset_matches_contract": all(
            math.isclose(actual, expected, abs_tol=1.0e-9)
            for actual, expected in zip(position, (0.0, 0.0, -0.05))
        ),
        "body_range_camera_rotation_matches_contract": all(
            math.isclose(actual, expected, abs_tol=1.0e-7)
            for actual, expected in zip(rotation, expected_rotation)
        )
        and str(camera_cfg.offset.convention) == "world",
    }
    return {
        "prim_path": str(camera_cfg.prim_path),
        "mesh_prim_paths": list(camera_cfg.mesh_prim_paths),
        "data_types": list(camera_cfg.data_types),
        "max_distance_m": float(camera_cfg.max_distance),
        "offset_position_m": list(position),
        "offset_rotation_wxyz": list(rotation),
        "offset_convention": str(camera_cfg.offset.convention),
        "pattern": {
            "width": int(pattern.width),
            "height": int(pattern.height),
            "focal_length": float(pattern.focal_length),
            "horizontal_aperture": float(pattern.horizontal_aperture),
        },
        "checks": checks,
    }


_ENV_ROBOT_PATH = re.compile(r"/World/envs/env_(\d+)/Robot(?:/|$)")


def contact_report_separations(
    contact_headers,
    contact_data,
    *,
    num_envs: int,
    int_to_path,
) -> dict[str, Any]:
    """Extract robot-ground separation without unsupported GPU contact filters."""

    minima = [float("inf")] * num_envs
    provenance = [None] * num_envs
    counts = [0] * num_envs
    robot_ground_headers = 0
    for header in contact_headers:
        paths = [
            str(int_to_path(header.actor0)),
            str(int_to_path(header.actor1)),
            str(int_to_path(header.collider0)),
            str(int_to_path(header.collider1)),
        ]
        if not any(path.startswith("/World/ground") for path in paths):
            continue
        match = next((_ENV_ROBOT_PATH.search(path) for path in paths if _ENV_ROBOT_PATH.search(path)), None)
        if match is None:
            continue
        env_index = int(match.group(1))
        if env_index < 0 or env_index >= num_envs:
            continue
        robot_ground_headers += 1
        start = int(header.contact_data_offset)
        end = start + int(header.num_contact_data)
        for datum in contact_data[start:end]:
            separation = float(datum.separation)
            if math.isfinite(separation):
                if separation < minima[env_index]:
                    minima[env_index] = separation
                    provenance[env_index] = {
                        "separation_m": separation,
                        "actor0_path": paths[0],
                        "actor1_path": paths[1],
                        "collider0_path": paths[2],
                        "collider1_path": paths[3],
                    }
                counts[env_index] += 1
    return {
        "minimum_separation_m": minima,
        "minimum_separation_provenance": provenance,
        "contact_point_count": counts,
        "header_count": len(contact_headers),
        "robot_ground_header_count": robot_ground_headers,
    }


class ContactReportAccumulator:
    """Copy per-step contact data while the PhysX callback buffer is valid."""

    def __init__(
        self, num_envs: int, int_to_path, physics_dt_s: float = 1.0
    ) -> None:
        if physics_dt_s <= 0.0 or not math.isfinite(physics_dt_s):
            raise ValueError("physics_dt_s must be finite and positive")
        self._num_envs = num_envs
        self._int_to_path = int_to_path
        self._physics_dt_s = physics_dt_s
        self.reset()

    def reset(self) -> None:
        self._minimum_separation_m = [float("inf")] * self._num_envs
        self._minimum_separation_provenance = [None] * self._num_envs
        self._contact_point_count = [0] * self._num_envs
        self._event_count = 0
        self._header_count = 0
        self._robot_ground_header_count = 0
        self._available = True
        self._error = None

    def mark_unavailable(self, error: BaseException) -> None:
        self._available = False
        self._error = f"{type(error).__name__}: {error}"

    def __call__(self, contact_headers, contact_data) -> None:
        self._event_count += 1
        try:
            report = contact_report_separations(
                contact_headers,
                contact_data,
                num_envs=self._num_envs,
                int_to_path=self._int_to_path,
            )
        except Exception as exc:
            self.mark_unavailable(exc)
            return
        self._header_count += report["header_count"]
        self._robot_ground_header_count += report["robot_ground_header_count"]
        for env_index in range(self._num_envs):
            event_minimum = report["minimum_separation_m"][env_index]
            if event_minimum < self._minimum_separation_m[env_index]:
                self._minimum_separation_m[env_index] = event_minimum
                event_provenance = report["minimum_separation_provenance"][
                    env_index
                ]
                self._minimum_separation_provenance[env_index] = {
                    **event_provenance,
                    "physics_step": self._event_count,
                    "time_s": self._event_count * self._physics_dt_s,
                }
            self._contact_point_count[env_index] += report["contact_point_count"][
                env_index
            ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "available": self._available,
            "error": self._error,
            "event_count": self._event_count,
            "minimum_separation_m": list(self._minimum_separation_m),
            "minimum_separation_provenance": [
                dict(value) if value is not None else None
                for value in self._minimum_separation_provenance
            ],
            "contact_point_count": list(self._contact_point_count),
            "header_count": self._header_count,
            "robot_ground_header_count": self._robot_ground_header_count,
        }


def within_hard_joint_limit_margin(max_violation, margin_rad: float):
    """Apply the same raw hard-limit margin used by the runtime contract."""

    if margin_rad < 0.0 or not math.isfinite(margin_rad):
        raise ValueError("hard joint limit margin must be finite and non-negative")
    return max_violation <= margin_rad


def separation_crosscheck_status(
    *, device: str, data_available: bool, threshold_passed: bool
) -> dict[str, Any]:
    """Describe CPU separation authority without blocking GPU dynamics."""

    this_run_is_authority = device.strip().lower() == "cpu"
    if not this_run_is_authority:
        status = "requires_cpu_crosscheck"
        passed = None
    elif not data_available:
        status = "authority_unavailable"
        passed = False
    else:
        status = "observed"
        passed = bool(threshold_passed)
    return {
        "name": "cpu_contact_separation",
        "authority_device": "cpu",
        "required_for_final_synthesis": True,
        "this_run_is_authority": this_run_is_authority,
        "data_available": bool(data_available),
        "threshold_m": MIN_CONTACT_SEPARATION_M,
        "threshold_passed": bool(threshold_passed) if data_available else None,
        "status": status,
        "passed": passed,
    }


class _MassReadForbiddenView:
    def __init__(self, delegate) -> None:
        self._delegate = delegate

    def get_masses(self):
        raise RuntimeError("actor observation attempted root_physx_view.get_masses")

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


def _probe_actor_privilege_isolation(raw_env, robot, torch) -> dict[str, Any]:
    manager = raw_env.observation_manager
    baseline_policy = manager.compute_group("policy").detach().clone()
    baseline_critic = manager.compute_group("critic").detach().clone()
    original_friction = raw_env._g009_effective_foot_friction.detach().clone()
    original_class = raw_env._g009_recover_fall_class.detach().clone()
    original_one_hot = raw_env._g009_recover_fall_class_one_hot.detach().clone()
    original_view = robot._root_physx_view
    mass_error = None
    try:
        raw_env._g009_effective_foot_friction = original_friction + 0.123
        raw_env._g009_recover_fall_class = (original_class + 1) % POSE_COUNT
        raw_env._g009_recover_fall_class_one_hot = torch.roll(original_one_hot, 1, 1)
        perturbed_policy = manager.compute_group("policy").detach().clone()
        perturbed_critic = manager.compute_group("critic").detach().clone()
        robot._root_physx_view = _MassReadForbiddenView(original_view)
        try:
            manager.compute_group("policy")
            avoids_mass_read = True
        except Exception as exc:
            avoids_mass_read = False
            mass_error = f"{type(exc).__name__}: {exc}"
    finally:
        robot._root_physx_view = original_view
        raw_env._g009_effective_foot_friction = original_friction
        raw_env._g009_recover_fall_class = original_class
        raw_env._g009_recover_fall_class_one_hot = original_one_hot
    actor_error = float(torch.max(torch.abs(perturbed_policy - baseline_policy)).item())
    critic_change = float(torch.max(torch.abs(perturbed_critic - baseline_critic)).item())
    return {
        "actor_max_abs_change": actor_error,
        "critic_max_abs_change": critic_change,
        "actor_unchanged": actor_error <= 1.0e-6,
        "critic_perturbation_exercised": critic_change > 1.0e-6,
        "actor_call_succeeds_without_live_mass_read": avoids_mass_read,
        "actor_mass_read_error": mass_error,
    }


def _probe_actor_load_range_independence(raw_env, torch) -> dict[str, Any]:
    """Force valid no-hit camera data and prove foot load is unchanged."""

    manager = raw_env.observation_manager
    camera = raw_env.scene.sensors["body_range_camera"]
    distances = camera.data.output["distance_to_camera"]
    original_distances = distances.detach().clone()
    baseline = manager.compute_group("policy").detach().clone()
    invalid_buffer = getattr(raw_env, "_g009_actor_signal_invalid", None)
    invalid_before = (
        invalid_buffer.detach().clone()
        if invalid_buffer is not None
        else torch.zeros(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)
    )
    try:
        distances.fill_(float("inf"))
        forced_no_hit = manager.compute_group("policy").detach().clone()
        invalid_after = getattr(
            raw_env,
            "_g009_actor_signal_invalid",
            torch.zeros_like(invalid_before),
        ).detach().clone()
    finally:
        distances.copy_(original_distances)
    baseline_load = baseline[:, ACTOR_FOOT_LOAD_SLICE]
    no_hit_load = forced_no_hit[:, ACTOR_FOOT_LOAD_SLICE]
    no_hit_range = forced_no_hit[:, ACTOR_RANGE_SLICE]
    no_hit_mask = forced_no_hit[:, ACTOR_RANGE_MASK_SLICE]
    return {
        "foot_load_max_abs_change_under_forced_no_hit": float(
            torch.max(torch.abs(no_hit_load - baseline_load)).item()
        ),
        "foot_load_unchanged_under_forced_no_hit": bool(
            torch.allclose(no_hit_load, baseline_load, atol=1.0e-6, rtol=0.0)
        ),
        "forced_no_hit_range_is_one": bool(
            torch.equal(no_hit_range, torch.ones_like(no_hit_range))
        ),
        "forced_no_hit_mask_is_zero": bool(
            torch.equal(no_hit_mask, torch.zeros_like(no_hit_mask))
        ),
        "forced_no_hit_does_not_mark_actor_invalid": bool(
            torch.equal(invalid_after, invalid_before)
        ),
    }


def _probe_reward_and_latch_managers(raw_env, recover_mdp, torch) -> dict[str, Any]:
    termination_manager = raw_env.termination_manager
    reward_manager = raw_env.reward_manager
    stable_cfg = termination_manager.get_term_cfg("stable_success")
    stable_term = stable_cfg.func
    required_steps = int(stable_cfg.params["required_consecutive_steps"])
    original_predicate = recover_mdp._stable_predicate
    device = raw_env.device
    alignment = torch.ones(raw_env.num_envs, device=device)
    base_height = torch.full(
        (raw_env.num_envs,),
        0.5
        * (
            float(stable_cfg.params["min_base_height"])
            + float(stable_cfg.params["max_base_height"])
        ),
        device=device,
    )
    contact_count = torch.zeros(raw_env.num_envs, dtype=torch.long, device=device)
    contact_count[[0, 1, 2, 3]] = int(stable_cfg.params["min_contacts"])
    required_load = (
        float(stable_cfg.params["nominal_total_mass_kg"])
        * float(stable_cfg.params["gravity_magnitude"])
        * float(stable_cfg.params["min_total_foot_support_ratio"])
    )
    total_foot_normal_load = torch.zeros(raw_env.num_envs, device=device)
    total_foot_normal_load[[0, 1, 3]] = required_load
    total_foot_normal_load[2] = 0.59 * (
        float(stable_cfg.params["nominal_total_mass_kg"])
        * float(stable_cfg.params["gravity_magnitude"])
    )
    non_foot_contact_count = torch.zeros(
        raw_env.num_envs, dtype=torch.long, device=device
    )
    non_foot_contact_count[1] = 1
    linear_speed = torch.zeros(raw_env.num_envs, device=device)
    angular_speed = torch.zeros(raw_env.num_envs, device=device)

    def physical_value_predicate(*args, **kwargs):
        return recover_mdp._stable_mask_from_values(
            alignment,
            base_height,
            contact_count,
            total_foot_normal_load,
            non_foot_contact_count,
            linear_speed,
            angular_speed,
            upright_threshold=float(kwargs["upright_threshold"]),
            min_base_height=float(kwargs["min_base_height"]),
            max_base_height=float(kwargs["max_base_height"]),
            min_contacts=int(kwargs["min_contacts"]),
            min_total_foot_normal_load=(
                float(kwargs["nominal_total_mass_kg"])
                * float(kwargs["gravity_magnitude"])
                * float(kwargs["min_total_foot_support_ratio"])
            ),
            max_linear_speed=float(kwargs["max_linear_speed"]),
            max_angular_speed=float(kwargs["max_angular_speed"]),
        )

    recover_mdp._stable_predicate = physical_value_predicate
    pulses = []
    try:
        termination_manager.reset()
        for _ in range(required_steps - 1):
            termination_manager.compute()
            pulses.append(termination_manager.get_term("stable_success").detach().clone())
        termination_manager.compute()
        trigger = termination_manager.get_term("stable_success").detach().clone()
        negative_counter = stable_term._counter[[1, 2]].detach().clone()
        negative_latched = stable_term._latched[[1, 2]].detach().clone()
        reward_manager.compute(dt=raw_env.step_dt)
        success_index = reward_manager.active_terms.index("stable_success_once")
        contribution = reward_manager._step_reward[:, success_index].detach() * raw_env.step_dt
        termination_manager.compute()
        repeated = termination_manager.get_term("stable_success").detach().clone()
        termination_manager.reset(env_ids=torch.tensor([0], device=raw_env.device))
        partial_counter = stable_term._counter.detach().clone()
        partial_latched = stable_term._latched.detach().clone()
        non_foot_contact_count[3] = 1
        for _ in range(required_steps):
            termination_manager.compute()
        retrigger = termination_manager.get_term("stable_success").detach().clone()
    finally:
        recover_mdp._stable_predicate = original_predicate
        termination_manager.reset()
        reward_manager.reset()

    progress_cfg = reward_manager.get_term_cfg("upright_progress")
    progress_term = progress_cfg.func
    progress_term.reset()
    discount_factor = float(progress_cfg.params["discount_factor"])
    progress_term._difference(
        torch.zeros(raw_env.num_envs, device=raw_env.device),
        1.0,
        raw_env.step_dt,
        discount_factor,
    )
    progress_raw = progress_term._difference(
        torch.full((raw_env.num_envs,), 0.25, device=raw_env.device),
        1.0,
        raw_env.step_dt,
        discount_factor,
    )
    progress_contribution = progress_raw * float(progress_cfg.weight) * raw_env.step_dt
    progress_term.reset()

    expected_trigger = torch.zeros_like(trigger)
    expected_trigger[[0, 3]] = True
    expected_retrigger = torch.zeros_like(trigger)
    expected_retrigger[0] = True
    return {
        "required_consecutive_steps": required_steps,
        "pretrigger_steps_clear": all(not bool(item.any().item()) for item in pulses),
        "trigger_on_required_step": bool(torch.equal(trigger, expected_trigger)),
        "repeated_step_has_no_pulse": not bool(repeated.any().item()),
        "success_contribution": _tensor_list(contribution),
        "success_contribution_exact_10": bool(
            torch.allclose(contribution[expected_trigger], torch.full_like(contribution[expected_trigger], 10.0))
            and torch.equal(contribution[~expected_trigger], torch.zeros_like(contribution[~expected_trigger]))
        ),
        "partial_reset_clears_only_selected": bool(
            partial_counter[0].item() == 0
            and not partial_latched[0].item()
            and partial_latched[3].item()
        ),
        "partial_reset_retriggers_only_selected": bool(torch.equal(retrigger, expected_retrigger)),
        "configured_min_foot_support_ratio": float(
            stable_cfg.params["min_total_foot_support_ratio"]
        ),
        "body_supported_nonfoot_contact_did_not_latch": bool(
            not trigger[1].item()
            and negative_counter[0].item() == 0
            and not negative_latched[0].item()
        ),
        "low_foot_load_did_not_latch": bool(
            not trigger[2].item()
            and negative_counter[1].item() == 0
            and not negative_latched[1].item()
        ),
        "potential_contribution": _tensor_list(progress_contribution),
        "potential_contribution_exact_0_495": bool(
            torch.allclose(progress_contribution, torch.full_like(progress_contribution, 0.495))
        ),
    }


def _startup_readback(raw_env, torch, constants: dict[str, float]) -> tuple[dict[str, Any], dict[str, bool]]:
    foot = raw_env._g009_foot_material_readback.detach().clone()
    effective = raw_env._g009_effective_foot_friction.detach().clone()
    masses = raw_env._g009_r0_body_mass.detach().clone()
    total = masses.sum(dim=1)
    expected_foot = foot.new_tensor((constants["foot_static"], constants["foot_dynamic"]))
    expected_effective = effective.new_tensor(
        (
            constants["ground_static"] * constants["foot_static"],
            constants["ground_dynamic"] * constants["foot_dynamic"],
        )
    )
    checks = {
        "foot_material_readback_matches_startup": bool(
            getattr(raw_env, "_g009_effective_foot_friction_valid", torch.zeros(1, dtype=torch.bool)).all().item()
            and torch.allclose(foot, expected_foot.expand_as(foot), atol=1.0e-6, rtol=0.0)
        ),
        "effective_friction_matches_multiply_combine": bool(
            torch.allclose(effective, expected_effective.expand_as(effective), atol=1.0e-6, rtol=0.0)
        ),
        "nominal_mass_readback_matches_contract": bool(
            getattr(raw_env, "_g009_r0_body_mass_valid", False)
            and torch.allclose(
                total,
                torch.full_like(total, constants["nominal_mass"]),
                atol=1.0e-3,
                rtol=0.0,
            )
        ),
    }
    return {
        "nominal_total_mass_kg": _tensor_list(total),
        "foot_material_static_dynamic": _tensor_list(foot),
        "effective_friction_static_dynamic": _tensor_list(effective),
    }, checks


def articulation_solver_iteration_readback(
    stage, articulation_prim_paths: list[str], PhysxSchema
) -> dict[str, Any]:
    """Read authored PhysX solver counts from every live articulation root."""

    rows: list[dict[str, Any]] = []
    for prim_path in articulation_prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        api = PhysxSchema.PhysxArticulationAPI(prim) if prim.IsValid() else None
        position_value = api.GetSolverPositionIterationCountAttr().Get() if api else None
        velocity_value = api.GetSolverVelocityIterationCountAttr().Get() if api else None
        rows.append(
            {
                "prim_path": prim_path,
                "solver_position_iteration_count": (
                    int(position_value) if position_value is not None else None
                ),
                "solver_velocity_iteration_count": (
                    int(velocity_value) if velocity_value is not None else None
                ),
            }
        )
    return {
        "source": "USD PhysxArticulationAPI live-stage readback",
        "articulations": rows,
    }


def articulation_solver_iteration_checks(
    readback: dict[str, Any],
    *,
    expected_position_count: int,
    expected_velocity_count: int,
    expected_articulations: int,
) -> dict[str, bool]:
    """Fail closed unless every live articulation reports both contract values."""

    rows = readback.get("articulations")
    if not isinstance(rows, list) or len(rows) != expected_articulations:
        return {
            "articulation_solver_iteration_counts_match_contract": False
        }
    return {
        "articulation_solver_iteration_counts_match_contract": bool(
            all(
                isinstance(row, dict)
                and row.get("solver_position_iteration_count")
                == expected_position_count
                and row.get("solver_velocity_iteration_count")
                == expected_velocity_count
                for row in rows
            )
        )
    }


def rigid_body_max_depenetration_velocity_readback(
    stage,
    robot_container_paths: list[str],
    articulation_prim_paths: list[str],
    link_path_groups: list[list[str]],
    body_names: list[str],
    PhysxSchema,
    UsdPhysics,
) -> dict[str, Any]:
    """Read each authoritative PhysX articulation link without subtree traversal."""

    articulations: list[dict[str, Any]] = []
    all_link_paths: list[str] = []
    group_count = max(
        len(robot_container_paths),
        len(articulation_prim_paths),
        len(link_path_groups),
    )
    for articulation_index in range(group_count):
        container_path = (
            robot_container_paths[articulation_index]
            if articulation_index < len(robot_container_paths)
            else None
        )
        articulation_path = (
            articulation_prim_paths[articulation_index]
            if articulation_index < len(articulation_prim_paths)
            else None
        )
        link_paths = (
            list(link_path_groups[articulation_index])
            if articulation_index < len(link_path_groups)
            else []
        )
        links: list[dict[str, Any]] = []
        for body_index, link_path in enumerate(link_paths):
            all_link_paths.append(link_path)
            prim = stage.GetPrimAtPath(link_path)
            prim_valid = bool(prim.IsValid())
            has_usd_api = bool(
                prim_valid and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            )
            has_physx_api = bool(
                prim_valid and prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
            )
            value: float | None = None
            error: str | None = None
            if not prim_valid:
                error = "invalid_link_prim"
            elif not has_usd_api:
                error = "missing_usd_rigid_body_api"
            elif not has_physx_api:
                error = "missing_physx_rigid_body_api"
            else:
                api = PhysxSchema.PhysxRigidBodyAPI(prim)
                attribute_getter = getattr(
                    api, "GetMaxDepenetrationVelocityAttr", None
                )
                attribute: Any = (
                    attribute_getter() if callable(attribute_getter) else None
                )
                raw_value = attribute.Get() if attribute else None
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    error = "missing_or_non_numeric_max_depenetration_velocity"
                else:
                    value = float(raw_value)
                    if not math.isfinite(value):
                        value = None
                        error = "non_finite_max_depenetration_velocity"
            links.append(
                {
                    "body_index": body_index,
                    "body_name": (
                        body_names[body_index]
                        if body_index < len(body_names)
                        else None
                    ),
                    "prim_path": link_path,
                    "prim_valid": prim_valid,
                    "usd_rigid_body_api": has_usd_api,
                    "physx_rigid_body_api": has_physx_api,
                    "max_depenetration_velocity_m_s": value,
                    "error": error,
                }
            )
        articulations.append(
            {
                "articulation_index": articulation_index,
                "robot_container_prim_path": container_path,
                "articulation_prim_path": articulation_path,
                "root_link_prim_path": link_paths[0] if link_paths else None,
                "authoritative_body_names": list(body_names),
                "authoritative_link_paths": link_paths,
                "links": links,
            }
        )

    prim_path_counts = Counter(all_link_paths)
    return {
        "source": "root_physx_view.link_paths direct USD/PhysX live-stage readback",
        "robot_container_prim_paths": list(robot_container_paths),
        "articulation_prim_paths": list(articulation_prim_paths),
        "authoritative_body_names": list(body_names),
        "authoritative_link_path_groups": [list(group) for group in link_path_groups],
        "articulation_group_count": len(articulations),
        "rigid_body_count": len(all_link_paths),
        "duplicate_link_prim_paths": sorted(
            path for path, count in prim_path_counts.items() if count > 1
        ),
        "articulations": articulations,
    }


def rigid_body_max_depenetration_velocity_checks(
    readback: dict[str, Any],
    *,
    expected_velocity_m_s: float,
    expected_articulation_count: int,
    expected_body_names: list[str],
) -> dict[str, bool]:
    """Fail closed unless authoritative link groups exactly match live readback."""

    articulations = readback.get("articulations")
    container_paths = readback.get("robot_container_prim_paths")
    articulation_paths = readback.get("articulation_prim_paths")
    link_path_groups = readback.get("authoritative_link_path_groups")
    reported_body_names = readback.get("authoritative_body_names")
    expected_total = expected_articulation_count * len(expected_body_names)
    all_link_paths: list[str] = []
    if isinstance(articulations, list):
        for articulation in articulations:
            if not isinstance(articulation, dict):
                continue
            links = articulation.get("links")
            if isinstance(links, list):
                for link in links:
                    if not isinstance(link, dict):
                        continue
                    prim_path = link.get("prim_path")
                    if isinstance(prim_path, str):
                        all_link_paths.append(prim_path)
    duplicate_paths = {
        path for path, count in Counter(all_link_paths).items() if count > 1
    }
    valid = (
        expected_articulation_count > 0
        and bool(expected_body_names)
        and all(isinstance(name, str) and name for name in expected_body_names)
        and len(set(expected_body_names)) == len(expected_body_names)
        and isinstance(articulations, list)
        and isinstance(container_paths, list)
        and isinstance(articulation_paths, list)
        and isinstance(link_path_groups, list)
        and reported_body_names == expected_body_names
        and len(articulations) == expected_articulation_count
        and len(container_paths) == expected_articulation_count
        and len(articulation_paths) == expected_articulation_count
        and len(link_path_groups) == expected_articulation_count
        and all(isinstance(path, str) and path for path in container_paths)
        and all(isinstance(path, str) and path for path in articulation_paths)
        and all(
            isinstance(group, list)
            and all(isinstance(path, str) and path for path in group)
            for group in link_path_groups
        )
        and len(set(container_paths)) == expected_articulation_count
        and len(set(articulation_paths)) == expected_articulation_count
        and not duplicate_paths
        and readback.get("articulation_group_count") == expected_articulation_count
        and readback.get("rigid_body_count") == expected_total
        and all(
            isinstance(articulation, dict)
            and articulation.get("articulation_index") == articulation_index
            and articulation.get("robot_container_prim_path")
            == container_paths[articulation_index]
            and articulation.get("articulation_prim_path")
            == articulation_paths[articulation_index]
            and articulation.get("root_link_prim_path")
            == articulation_paths[articulation_index]
            and articulation_paths[articulation_index].rsplit("/", 1)[0]
            == container_paths[articulation_index]
            and articulation.get("authoritative_body_names") == expected_body_names
            and articulation.get("authoritative_link_paths")
            == link_path_groups[articulation_index]
            and isinstance(articulation.get("links"), list)
            and len(articulation["links"]) == len(expected_body_names)
            and len(link_path_groups[articulation_index]) == len(expected_body_names)
            and all(
                isinstance(link, dict)
                and link.get("body_index") == body_index
                and link.get("body_name") == expected_body_names[body_index]
                and link.get("prim_path")
                == link_path_groups[articulation_index][body_index]
                and link["prim_path"].rsplit("/", 1)[-1] == expected_body_names[body_index]
                and link["prim_path"].startswith(
                    container_paths[articulation_index].rstrip("/") + "/"
                )
                and link.get("prim_valid") is True
                and link.get("usd_rigid_body_api") is True
                and link.get("physx_rigid_body_api") is True
                and link.get("error") is None
                and isinstance(link.get("max_depenetration_velocity_m_s"), float)
                and math.isfinite(link["max_depenetration_velocity_m_s"])
                and math.isclose(
                    link["max_depenetration_velocity_m_s"],
                    expected_velocity_m_s,
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                )
                for body_index, link in enumerate(articulation["links"])
            )
            for articulation_index, articulation in enumerate(articulations)
        )
    )
    return {"rigid_body_max_depenetration_velocity_matches_contract": bool(valid)}


def probe(args: argparse.Namespace, execution: dict[str, Any]) -> dict[str, Any]:
    execution = validate_execution_metadata(execution, args.output.resolve())
    source_bundle = source_bundle_provenance()
    _progress("import_runtime_dependencies")
    import gymnasium as gym
    import torch

    import omni.usd
    import isaaclab_tasks  # noqa: F401
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics
    from isaaclab import sim as sim_utils
    from isaaclab.utils import math as math_utils
    from isaaclab_tasks.utils import parse_env_cfg
    from isaac_walk_g009 import register_tasks
    from isaac_walk_g009.mdp import recover as recover_mdp
    from isaac_walk_g009.recover_contracts import (
        ACTION_EMA_ALPHA,
        ACTION_SCALE,
        ARTICULATION_SOLVER_POSITION_ITERATION_COUNT,
        ARTICULATION_SOLVER_VELOCITY_ITERATION_COUNT,
        ACTOR_OBSERVATION_DIM,
        CRITIC_OBSERVATION_DIM,
        FOOT_DYNAMIC_FRICTION,
        FOOT_STATIC_FRICTION,
        GROUND_DYNAMIC_FRICTION,
        GROUND_STATIC_FRICTION,
        GO2_SOFT_JOINT_LIMIT_FACTOR,
        MAX_DEPENETRATION_VELOCITY_M_S,
        NOMINAL_TOTAL_MASS_KG,
        RECOVER_POSES,
        SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
        STABLE_DWELL_STEPS,
        canonical_sha256,
        recover_contract,
    )

    validate_calibration_budget(args.num_envs, args.rollout_steps)
    register_tasks()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.contact_forces.history_length = env_cfg.decimation
    env_cfg.events.reset_base.params.update(
        {"assignment_mode": "stratified", "pose_xy_range": (0.0, 0.0), "yaw_range": (0.0, 0.0)}
    )
    env_cfg.validate()
    env = gym.make(args.task, cfg=env_cfg)
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    solver_iteration_readback = articulation_solver_iteration_readback(
        omni.usd.get_context().get_stage(),
        list(robot.root_physx_view.prim_paths),
        PhysxSchema,
    )
    max_depenetration_velocity_readback = rigid_body_max_depenetration_velocity_readback(
        omni.usd.get_context().get_stage(),
        sim_utils.find_matching_prim_paths(
            robot.cfg.prim_path, omni.usd.get_context().get_stage()
        ),
        list(robot.root_physx_view.prim_paths),
        [list(group) for group in robot.root_physx_view.link_paths],
        list(robot.body_names),
        PhysxSchema,
        UsdPhysics,
    )
    action_term = raw_env.action_manager.get_term("joint_pos")
    runtime_action_scale = float(action_term.cfg.scale)
    runtime_soft_limit_factor = float(robot.cfg.soft_joint_pos_limit_factor)
    sensor = raw_env.scene.sensors["contact_forces"]
    range_camera = raw_env.scene.sensors["body_range_camera"]
    range_camera_config = camera_config_readback(range_camera.cfg)
    physx_simulation = get_physx_simulation_interface()
    separation_accumulator = ContactReportAccumulator(
        args.num_envs,
        PhysicsSchemaTools.intToSdfPath,
        physics_dt_s=raw_env.physics_dt,
    )
    contact_report_subscription = None
    try:
        observations, _ = env.reset()
        class_ids = raw_env._g009_recover_fall_class.detach().clone()
        class_one_hot = raw_env._g009_recover_fall_class_one_hot.detach().clone()
        expected_ids = torch.arange(args.num_envs, device=raw_env.device) % POSE_COUNT
        expected_one_hot = torch.nn.functional.one_hot(expected_ids, POSE_COUNT).float()
        root_pose = robot.data.root_pose_w.detach().clone()
        root_local = root_pose[:, :3] - raw_env.scene.env_origins
        joint_position = robot.data.joint_pos.detach().clone()
        joint_limits = robot.data.joint_pos_limits.detach().clone()
        soft_limits = robot.data.soft_joint_pos_limits.detach().clone()
        effort_limits = _actuator_joint_limits(robot, "effort_limit", torch).max(dim=1).values
        velocity_limits = _actuator_joint_limits(robot, "velocity_limit", torch).max(dim=1).values
        startup, startup_checks = _startup_readback(
            raw_env,
            torch,
            {
                "foot_static": FOOT_STATIC_FRICTION,
                "foot_dynamic": FOOT_DYNAMIC_FRICTION,
                "ground_static": GROUND_STATIC_FRICTION,
                "ground_dynamic": GROUND_DYNAMIC_FRICTION,
                "nominal_mass": NOMINAL_TOTAL_MASS_KG,
            },
        )
        total_mass = raw_env._g009_r0_body_mass.sum(dim=1)
        world_up = torch.tensor((0.0, 0.0, 1.0), device=raw_env.device).expand(args.num_envs, -1)
        body_up = math_utils.quat_apply(root_pose[:, 3:7], world_up)
        expected_up = torch.tensor(
            [RECOVER_POSES[name].expected_body_up for name in RECOVER_POSES] * 2,
            device=raw_env.device,
            dtype=body_up.dtype,
        )
        body_up_error = torch.linalg.vector_norm(body_up - expected_up, dim=-1)
        reset_camera_position = range_camera.data.pos_w.detach().clone()
        reset_camera_quaternion = range_camera.data.quat_w_world.detach().clone()
        camera_offset_pos = torch.tensor(
            [range_camera.cfg.offset.pos],
            device=raw_env.device,
            dtype=root_pose.dtype,
        ).expand(args.num_envs, -1)
        camera_offset_quat = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor(
                [range_camera.cfg.offset.rot],
                device=raw_env.device,
                dtype=root_pose.dtype,
            ),
            origin=range_camera.cfg.offset.convention,
            target="world",
        ).expand(args.num_envs, -1)
        expected_camera_pos = root_pose[:, :3] + math_utils.quat_apply(
            root_pose[:, 3:7], camera_offset_pos
        )
        expected_camera_quat = math_utils.quat_mul(
            root_pose[:, 3:7], camera_offset_quat
        )
        camera_position_error = torch.linalg.vector_norm(
            reset_camera_position - expected_camera_pos, dim=1
        )
        camera_quaternion_error = torch.minimum(
            torch.linalg.vector_norm(
                reset_camera_quaternion - expected_camera_quat, dim=1
            ),
            torch.linalg.vector_norm(
                reset_camera_quaternion + expected_camera_quat, dim=1
            ),
        )
        shapes = {name: list(value.shape) for name, value in observations.items()}
        finite_reset_obs = {name: bool(torch.isfinite(value).all().item()) for name, value in observations.items()}
        reset_policy = observations["policy"].detach().clone()
        reset_critic = observations["critic"].detach().clone()
        critic_height = reset_critic[:, CRITIC_BASE_HEIGHT_INDEX]
        critic_normal = reset_critic[:, CRITIC_TERRAIN_NORMAL_SLICE]
        height_error = torch.abs(critic_height - root_local[:, 2])
        reset_camera_observations = camera_observation_checks(
            reset_policy[:, ACTOR_FOOT_LOAD_SLICE],
            reset_policy[:, ACTOR_RANGE_SLICE],
            reset_policy[:, ACTOR_RANGE_MASK_SLICE],
        )
        load_range_independence = _probe_actor_load_range_independence(
            raw_env, torch
        )
        privilege = _probe_actor_privilege_isolation(raw_env, robot, torch)
        manager_probe = _probe_reward_and_latch_managers(raw_env, recover_mdp, torch)

        observations, _ = env.reset()
        separation_accumulator.reset()
        try:
            contact_report_subscription = (
                physx_simulation.subscribe_contact_report_events(
                    separation_accumulator
                )
            )
            if contact_report_subscription is None:
                raise RuntimeError(
                    "subscribe_contact_report_events returned no subscription holder"
                )
        except Exception as exc:
            separation_accumulator.mark_unavailable(exc)
        stable_cfg = raw_env.termination_manager.get_term_cfg("stable_success")
        original_dwell = stable_cfg.params["required_consecutive_steps"]
        stable_cfg.params["required_consecutive_steps"] = args.rollout_steps + 1
        action_dim = int(raw_env.action_manager.total_action_dim)
        ema_previous_targets = getattr(action_term, "_prev_applied_actions", None)
        if ema_previous_targets is None:
            raise RuntimeError("EMA action term did not expose previous applied targets")
        ema_reset_target_error = torch.abs(
            ema_previous_targets.detach() - robot.data.joint_pos.detach()
        ).amax(dim=1)
        reset_pose_hold_diagnostics = reset_pose_hold_action_diagnostics(
            robot.data.joint_pos[POSE_COUNT:].detach(),
            soft_limits[POSE_COUNT:],
            list(robot.joint_names),
            action_scale=runtime_action_scale,
        )
        actions = torch.zeros((args.num_envs, action_dim), device=raw_env.device)
        actions[POSE_COUNT:] = reset_pose_hold_diagnostics["normalized_action"]
        terms = list(raw_env.termination_manager.active_terms)
        term_counts = {name: torch.zeros(args.num_envs, dtype=torch.long, device=raw_env.device) for name in terms}
        finite = torch.ones(args.num_envs, dtype=torch.bool, device=raw_env.device)
        hard_safe = torch.ones(args.num_envs, dtype=torch.bool, device=raw_env.device)
        max_lin = torch.zeros(args.num_envs, device=raw_env.device)
        max_ang = torch.zeros_like(max_lin)
        min_height = torch.full_like(max_lin, float("inf"))
        max_height = torch.full_like(max_lin, float("-inf"))
        max_joint_speed = torch.zeros_like(max_lin)
        max_torque = torch.zeros_like(max_lin)
        max_hard_violation = torch.zeros_like(max_lin)
        max_hard_violation_step = torch.full(
            (args.num_envs,), -1, dtype=torch.long, device=raw_env.device
        )
        max_hard_violation_joint = torch.full_like(max_hard_violation_step, -1)
        contact_exercised = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
        nonfoot_contact_exercised = torch.zeros(
            args.num_envs, dtype=torch.bool, device=raw_env.device
        )
        max_nonfoot_force = torch.zeros_like(max_lin)
        max_nonfoot_force_step = torch.full_like(max_hard_violation_step, -1)
        max_nonfoot_force_body_index = torch.full_like(max_hard_violation_step, -1)
        excess_delta_v = torch.zeros_like(max_lin)
        max_step_excess_delta_v = torch.zeros_like(max_lin)
        max_step_excess_delta_v_step = torch.full_like(max_hard_violation_step, -1)
        first_contact_step = torch.full_like(max_hard_violation_step, -1)
        first_contact_force_bodyweights = torch.zeros_like(max_lin)
        first_contact_excess_delta_v = torch.zeros_like(max_lin)
        rollout_foot_load_valid = True
        rollout_range_valid = True
        rollout_mask_valid = True
        rollout_no_hit_range_valid = True
        tail_horizontal, tail_vertical, tail_angular = [], [], []
        foot_ids = [index for index, name in enumerate(sensor.body_names) if name.endswith("_foot")]
        nonfoot_ids = [index for index, name in enumerate(sensor.body_names) if not name.endswith("_foot")]
        if len(foot_ids) != 4 or not nonfoot_ids:
            raise RuntimeError("contact sensor must resolve four feet and non-foot bodies")

        _progress("three_second_physics_calibration")
        for step in range(args.rollout_steps):
            observations, _, _, _, _ = env.step(actions)
            actor_observation = observations["policy"]
            actor_load = actor_observation[:, ACTOR_FOOT_LOAD_SLICE]
            actor_range = actor_observation[:, ACTOR_RANGE_SLICE]
            actor_mask = actor_observation[:, ACTOR_RANGE_MASK_SLICE]
            rollout_foot_load_valid &= bool(
                torch.isfinite(actor_load).all().item()
                and (actor_load >= 0.0).all().item()
            )
            rollout_range_valid &= bool(
                torch.isfinite(actor_range).all().item()
                and (actor_range >= 0.0).all().item()
                and (actor_range <= 1.0).all().item()
            )
            rollout_mask_valid &= bool(
                ((actor_mask == 0.0) | (actor_mask == 1.0)).all().item()
            )
            rollout_no_hit_range_valid &= bool(
                (actor_range[actor_mask == 0.0] == 1.0).all().item()
            )
            for name in terms:
                term_counts[name] += raw_env.termination_manager.get_term(name).long()
            root_lin = robot.data.root_lin_vel_w
            root_ang = robot.data.root_ang_vel_b
            lin = torch.linalg.vector_norm(root_lin, dim=-1)
            ang = torch.linalg.vector_norm(root_ang, dim=-1)
            height = robot.data.root_pos_w[:, 2] - raw_env.scene.env_origins[:, 2]
            joint_speed = torch.abs(robot.data.joint_vel).max(dim=1).values
            torque = torch.abs(robot.data.applied_torque).max(dim=1).values
            forces = sensor.data.net_forces_w
            force_history = sensor.data.net_forces_w_history
            if force_history is None or force_history.shape[1] != raw_env.cfg.decimation:
                raise RuntimeError("contact force history must cover every physics substep")
            foot_force_history = torch.linalg.vector_norm(
                force_history[:, :, foot_ids], dim=-1
            )
            nonfoot_force_history = torch.linalg.vector_norm(
                force_history[:, :, nonfoot_ids], dim=-1
            )
            nonfoot_total_history = nonfoot_force_history.sum(dim=2)
            body_weight = total_mass * 9.81
            total_contact_force_history = (
                foot_force_history.sum(dim=2) + nonfoot_total_history
            )
            lower_violation = torch.clamp(
                joint_limits[..., 0] - robot.data.joint_pos, min=0.0
            )
            upper_violation = torch.clamp(
                robot.data.joint_pos - joint_limits[..., 1], min=0.0
            )
            joint_violation = torch.maximum(lower_violation, upper_violation)
            step_violation, step_violation_joint = joint_violation.max(dim=1)
            new_max_violation = step_violation > max_hard_violation
            max_hard_violation = torch.maximum(max_hard_violation, step_violation)
            max_hard_violation_step = torch.where(
                new_max_violation,
                torch.full_like(max_hard_violation_step, step + 1),
                max_hard_violation_step,
            )
            max_hard_violation_joint = torch.where(
                new_max_violation,
                step_violation_joint,
                max_hard_violation_joint,
            )
            step_excess_delta_v = (
                torch.clamp(
                    nonfoot_total_history - 1.5 * body_weight.unsqueeze(1),
                    min=0.0,
                ).sum(dim=1)
                * raw_env.physics_dt
                / total_mass
            )
            for env_index in range(args.num_envs):
                if first_contact_step[env_index].item() >= 0:
                    continue
                active_history = torch.nonzero(
                    total_contact_force_history[env_index]
                    >= CONTACT_EXERCISE_THRESHOLD_N,
                    as_tuple=False,
                ).flatten()
                if len(active_history) == 0:
                    continue
                history_index = int(active_history.max().item())
                physics_offset = raw_env.cfg.decimation - history_index
                first_contact_step[env_index] = (
                    step * raw_env.cfg.decimation + physics_offset
                )
                first_contact_force_bodyweights[env_index] = (
                    total_contact_force_history[env_index, history_index]
                    / body_weight[env_index]
                )
                first_contact_excess_delta_v[env_index] = (
                    torch.clamp(
                        nonfoot_total_history[env_index, history_index]
                        - 1.5 * body_weight[env_index],
                        min=0.0,
                    )
                    * raw_env.physics_dt
                    / total_mass[env_index]
                )
            finite &= torch.isfinite(robot.data.root_state_w).all(dim=1)
            finite &= torch.isfinite(robot.data.joint_pos).all(dim=1)
            finite &= torch.isfinite(robot.data.joint_vel).all(dim=1)
            finite &= torch.isfinite(robot.data.applied_torque).all(dim=1)
            finite &= torch.isfinite(forces).reshape(args.num_envs, -1).all(dim=1)
            finite &= torch.isfinite(force_history).reshape(args.num_envs, -1).all(dim=1)
            finite &= torch.stack(
                [torch.isfinite(value).reshape(args.num_envs, -1).all(dim=1) for value in observations.values()]
            ).all(dim=0)
            hard_safe &= within_hard_joint_limit_margin(
                step_violation, SOLVER_JOINT_LIMIT_TOLERANCE_RAD
            )
            max_lin = torch.maximum(max_lin, lin)
            max_ang = torch.maximum(max_ang, ang)
            min_height = torch.minimum(min_height, height)
            max_height = torch.maximum(max_height, height)
            max_joint_speed = torch.maximum(max_joint_speed, joint_speed)
            max_torque = torch.maximum(max_torque, torque)
            contact_exercised |= (
                foot_force_history >= CONTACT_EXERCISE_THRESHOLD_N
            ).reshape(args.num_envs, -1).any(dim=1)
            nonfoot_contact_exercised |= (
                nonfoot_force_history >= CONTACT_EXERCISE_THRESHOLD_N
            ).reshape(args.num_envs, -1).any(dim=1)
            step_nonfoot_force, step_nonfoot_flat_index = (
                nonfoot_force_history.reshape(args.num_envs, -1).max(dim=1)
            )
            new_max_force = step_nonfoot_force > max_nonfoot_force
            max_nonfoot_force = torch.maximum(max_nonfoot_force, step_nonfoot_force)
            force_history_index = (
                step_nonfoot_flat_index // len(nonfoot_ids)
            )
            step_nonfoot_body_index = nonfoot_body_indices_from_flat(
                step_nonfoot_flat_index, nonfoot_ids
            )
            force_physics_step = (
                step * raw_env.cfg.decimation
                + raw_env.cfg.decimation
                - force_history_index
            )
            max_nonfoot_force_step = torch.where(
                new_max_force,
                force_physics_step,
                max_nonfoot_force_step,
            )
            max_nonfoot_force_body_index = torch.where(
                new_max_force,
                step_nonfoot_body_index,
                max_nonfoot_force_body_index,
            )
            excess_delta_v += step_excess_delta_v
            new_max_step_impulse = step_excess_delta_v > max_step_excess_delta_v
            max_step_excess_delta_v = torch.maximum(
                max_step_excess_delta_v, step_excess_delta_v
            )
            max_step_excess_delta_v_step = torch.where(
                new_max_step_impulse,
                torch.full_like(max_step_excess_delta_v_step, step + 1),
                max_step_excess_delta_v_step,
            )
            if step >= args.rollout_steps - TAIL_STEPS:
                tail_horizontal.append(torch.linalg.vector_norm(root_lin[:, :2], dim=-1).detach())
                tail_vertical.append(torch.abs(root_lin[:, 2]).detach())
                tail_angular.append(ang.detach())
        stable_cfg.params["required_consecutive_steps"] = original_dwell

        tail_h = torch.quantile(torch.stack(tail_horizontal), 0.95, dim=0)
        tail_v = torch.quantile(torch.stack(tail_vertical), 0.95, dim=0)
        tail_a = torch.quantile(torch.stack(tail_angular), 0.95, dim=0)
        separation_report = separation_accumulator.snapshot()
        min_separation = torch.tensor(
            separation_report["minimum_separation_m"], device=raw_env.device
        )
        separation_point_counts = torch.tensor(
            separation_report["contact_point_count"],
            dtype=torch.long,
            device=raw_env.device,
        )
        separation_available = separation_report["available"]
        separation_error = separation_report["error"]
        contact_report_headers_seen = separation_report["header_count"]
        robot_ground_headers_seen = separation_report[
            "robot_ground_header_count"
        ]
        separation_valid = (
            separation_available
            and bool(torch.isfinite(min_separation).all().item())
            and bool((separation_point_counts > 0).all().item())
        )
        separation_threshold_passed = separation_valid and bool(
            (min_separation >= MIN_CONTACT_SEPARATION_M).all().item()
        )
        separation_crosscheck = separation_crosscheck_status(
            device=args.device,
            data_available=separation_valid,
            threshold_passed=separation_threshold_passed,
        )
        rows = pose_mode_rows(list(RECOVER_POSES))
        for row in rows:
            i = row["env_index"]
            row.update(
                {
                    "finite": bool(finite[i].item()),
                    "reset_actor_foot_load_bodyweights": _tensor_list(
                        reset_policy[i, ACTOR_FOOT_LOAD_SLICE]
                    ),
                    "reset_body_range_normalized": _tensor_list(
                        reset_policy[i, ACTOR_RANGE_SLICE]
                    ),
                    "reset_body_range_hit_mask": _tensor_list(
                        reset_policy[i, ACTOR_RANGE_MASK_SLICE]
                    ),
                    "reset_body_range_hit_count": reset_camera_observations[
                        "hit_count_per_env"
                    ][i],
                    "hard_limit_margin_safe": bool(hard_safe[i].item()),
                    "max_hard_joint_limit_violation_rad": float(
                        max_hard_violation[i].item()
                    ),
                    "max_hard_joint_limit_violation_control_step": int(
                        max_hard_violation_step[i].item()
                    ),
                    "max_hard_joint_limit_violation_time_s": (
                        float(max_hard_violation_step[i].item() * raw_env.step_dt)
                        if max_hard_violation_step[i].item() >= 0
                        else None
                    ),
                    "max_hard_joint_limit_violation_joint": (
                        robot.joint_names[max_hard_violation_joint[i].item()]
                        if max_hard_violation_joint[i].item() >= 0
                        else None
                    ),
                    "max_root_linear_speed_m_s": float(max_lin[i].item()),
                    "max_root_angular_speed_rad_s": float(max_ang[i].item()),
                    "min_root_height_m": float(min_height[i].item()),
                    "max_root_height_m": float(max_height[i].item()),
                    "max_joint_speed_rad_s": float(max_joint_speed[i].item()),
                    "joint_speed_limit_rad_s": float(velocity_limits[i].item()),
                    "max_abs_torque_nm": float(max_torque[i].item()),
                    "torque_limit_nm": float(effort_limits[i].item()),
                    "foot_contact_exercised": bool(contact_exercised[i].item()),
                    "nonfoot_contact_exercised": bool(
                        nonfoot_contact_exercised[i].item()
                    ),
                    "max_nonfoot_force_bodyweights": float((max_nonfoot_force[i] / (total_mass[i] * 9.81)).item()),
                    "max_nonfoot_force_physics_step": int(
                        max_nonfoot_force_step[i].item()
                    ),
                    "max_nonfoot_force_body_index": int(
                        max_nonfoot_force_body_index[i].item()
                    ),
                    "max_nonfoot_force_body_name": (
                        sensor.body_names[max_nonfoot_force_body_index[i].item()]
                        if max_nonfoot_force_body_index[i].item() >= 0
                        else None
                    ),
                    "max_nonfoot_force_time_s": (
                        float(
                            max_nonfoot_force_step[i].item()
                            * raw_env.physics_dt
                        )
                        if max_nonfoot_force_step[i].item() >= 0
                        else None
                    ),
                    "excess_contact_delta_v_m_s": float(excess_delta_v[i].item()),
                    "peak_step_excess_contact_delta_v_m_s": float(
                        max_step_excess_delta_v[i].item()
                    ),
                    "peak_step_excess_contact_impulse_control_step": int(
                        max_step_excess_delta_v_step[i].item()
                    ),
                    "peak_step_excess_contact_impulse_time_s": (
                        float(
                            max_step_excess_delta_v_step[i].item()
                            * raw_env.step_dt
                        )
                        if max_step_excess_delta_v_step[i].item() >= 0
                        else None
                    ),
                    "first_contact_physics_step": int(first_contact_step[i].item()),
                    "first_contact_time_s": (
                        float(first_contact_step[i].item() * raw_env.physics_dt)
                        if first_contact_step[i].item() >= 0
                        else None
                    ),
                    "first_contact_force_bodyweights": float(
                        first_contact_force_bodyweights[i].item()
                    ),
                    "first_contact_excess_delta_v_m_s": float(
                        first_contact_excess_delta_v[i].item()
                    ),
                    "reset_pose_hold_normalized_action_saturated": (
                        bool(
                            reset_pose_hold_diagnostics["saturated_mask"][
                                i - POSE_COUNT
                            ].any().item()
                        )
                        if i >= POSE_COUNT
                        else None
                    ),
                    "reset_pose_hold_saturated_joint_count": (
                        int(
                            reset_pose_hold_diagnostics["saturated_mask"][
                                i - POSE_COUNT
                            ].sum().item()
                        )
                        if i >= POSE_COUNT
                        else None
                    ),
                    "reset_pose_hold_saturated_joint_names": (
                        reset_pose_hold_diagnostics["saturated_joint_names"][i - POSE_COUNT]
                        if i >= POSE_COUNT
                        else None
                    ),
                    "reset_pose_hold_reachable_processed_joint_target_rad": (
                        _tensor_list(
                            reset_pose_hold_diagnostics["reachable_target"][i - POSE_COUNT]
                        )
                        if i >= POSE_COUNT
                        else None
                    ),
                    "reset_pose_hold_reachable_target_max_error_rad": (
                        float(
                            reset_pose_hold_diagnostics["max_target_error"][
                                i - POSE_COUNT
                            ].item()
                        )
                        if i >= POSE_COUNT
                        else None
                    ),
                    "reset_pose_hold_reachable_target_max_error_joint_index": (
                        int(
                            reset_pose_hold_diagnostics["max_target_error_joint_index"][
                                i - POSE_COUNT
                            ].item()
                        )
                        if i >= POSE_COUNT
                        else None
                    ),
                    "reset_pose_hold_reachable_target_max_error_joint_name": (
                        reset_pose_hold_diagnostics["max_target_error_joint_name"][i - POSE_COUNT]
                        if i >= POSE_COUNT
                        else None
                    ),
                    "min_contact_separation_m": float(min_separation[i].item()) if torch.isfinite(min_separation[i]) else None,
                    "min_contact_separation_provenance": separation_report[
                        "minimum_separation_provenance"
                    ][i],
                    "tail_horizontal_speed_p95_m_s": float(tail_h[i].item()),
                    "tail_vertical_speed_p95_m_s": float(tail_v[i].item()),
                    "tail_angular_speed_p95_rad_s": float(tail_a[i].item()),
                    "termination_counts": {name: int(value[i].item()) for name, value in term_counts.items()},
                }
            )

        checks = {
            **articulation_solver_iteration_checks(
                solver_iteration_readback,
                expected_position_count=ARTICULATION_SOLVER_POSITION_ITERATION_COUNT,
                expected_velocity_count=ARTICULATION_SOLVER_VELOCITY_ITERATION_COUNT,
                expected_articulations=args.num_envs,
            ),
            **rigid_body_max_depenetration_velocity_checks(
                max_depenetration_velocity_readback,
                expected_velocity_m_s=MAX_DEPENETRATION_VELOCITY_M_S,
                expected_articulation_count=args.num_envs,
                expected_body_names=list(robot.body_names),
            ),
            "joint_action_type_matches_contract": (
                type(action_term).__name__ == "EMAJointPositionToLimitsAction"
            ),
            "joint_action_scale_matches_contract": math.isclose(
                runtime_action_scale, ACTION_SCALE, abs_tol=1.0e-12
            ),
            "joint_action_ema_alpha_matches_contract": math.isclose(
                float(action_term.cfg.alpha), ACTION_EMA_ALPHA, abs_tol=1.0e-12
            ),
            "joint_action_ema_history_matches_reset_joint_positions": bool(
                (ema_reset_target_error <= 1.0e-6).all().item()
            ),
            "soft_joint_limit_factor_matches_contract": math.isclose(
                runtime_soft_limit_factor,
                GO2_SOFT_JOINT_LIMIT_FACTOR,
                abs_tol=1.0e-12,
            ),
            "policy_observation_dim_83": shapes.get("policy")
            == [args.num_envs, ACTOR_OBSERVATION_DIM],
            "critic_observation_dim_107": shapes.get("critic")
            == [args.num_envs, CRITIC_OBSERVATION_DIM],
            "action_dim_12": action_dim == 12,
            "eight_env_pose_mode_stratification": bool(torch.equal(class_ids, expected_ids)),
            "eight_env_pose_one_hot": bool(torch.equal(class_one_hot, expected_one_hot)),
            "pose_orientation_matches_contract": bool((body_up_error <= 1.0e-5).all().item()),
            "observations_finite_at_reset": all(finite_reset_obs.values()),
            "critic_base_height_matches_root_height": bool(
                (height_error <= 1.0e-5).all().item()
            ),
            "critic_terrain_normal_finite_unit_length": bool(
                torch.isfinite(critic_normal).all().item()
                and torch.allclose(
                    torch.linalg.vector_norm(critic_normal, dim=1),
                    torch.ones(args.num_envs, device=raw_env.device),
                    atol=1.0e-5,
                    rtol=0.0,
                )
            ),
            "body_range_camera_world_pose_matches_base_offset": bool(
                (camera_position_error <= 1.0e-5).all().item()
                and (camera_quaternion_error <= 1.0e-5).all().item()
            ),
            **range_camera_config["checks"],
            **reset_camera_observations["checks"],
            "actor_foot_load_valid_through_rollout": rollout_foot_load_valid,
            "body_range_valid_through_rollout": rollout_range_valid,
            "body_range_mask_valid_through_rollout": rollout_mask_valid,
            "body_range_no_hit_is_one_through_rollout": rollout_no_hit_range_valid,
            "actor_foot_load_independent_of_ray_no_hit": load_range_independence[
                "foot_load_unchanged_under_forced_no_hit"
            ],
            "forced_ray_no_hit_maps_to_valid_actor_observation": (
                load_range_independence["forced_no_hit_range_is_one"]
                and load_range_independence["forced_no_hit_mask_is_zero"]
                and load_range_independence[
                    "forced_no_hit_does_not_mark_actor_invalid"
                ]
            ),
            **startup_checks,
            "actor_privileged_perturbation_invariant": privilege["actor_unchanged"] and privilege["critic_perturbation_exercised"],
            "actor_call_path_avoids_live_mass_read": privilege["actor_call_succeeds_without_live_mass_read"],
            "reward_terminal_contribution_exact_10": manager_probe["success_contribution_exact_10"],
            "reward_potential_contribution_exact_0_495": manager_probe[
                "potential_contribution_exact_0_495"
            ],
            "stable_success_25_step_one_shot": manager_probe["required_consecutive_steps"] == STABLE_DWELL_STEPS and manager_probe["pretrigger_steps_clear"] and manager_probe["trigger_on_required_step"] and manager_probe["repeated_step_has_no_pulse"],
            "stable_success_partial_reset_isolated": manager_probe["partial_reset_clears_only_selected"] and manager_probe["partial_reset_retriggers_only_selected"],
            "stable_success_uses_60pct_foot_load_gate": math.isclose(
                manager_probe["configured_min_foot_support_ratio"],
                0.60,
                abs_tol=1.0e-9,
            ),
            "body_supported_state_does_not_latch_success": manager_probe[
                "body_supported_nonfoot_contact_did_not_latch"
            ],
            "low_foot_load_state_does_not_latch_success": manager_probe[
                "low_foot_load_did_not_latch"
            ],
            "rollout_state_finite_all_pose_modes": bool(finite.all().item()),
            "no_numeric_invalid_termination": bool((term_counts["numeric_invalid"] == 0).all().item()),
            "no_hard_joint_limit_termination": bool((term_counts["hard_joint_limit"] == 0).all().item()),
            "joint_positions_within_hard_limit_margin": bool(
                hard_safe.all().item()
            ),
            "torque_within_runtime_limit": bool((max_torque <= effort_limits * 1.01).all().item()),
            "joint_speed_within_runtime_limit": bool((max_joint_speed <= velocity_limits * 1.01).all().item()),
            "root_height_above_2cm": bool((min_height >= MIN_ROOT_HEIGHT_M).all().item()),
            **contact_exercise_checks(
                contact_exercised, nonfoot_contact_exercised
            ),
            "nonfoot_peak_force_bounded": bool((max_nonfoot_force / (total_mass * 9.81) <= MAX_NON_FOOT_FORCE_BODYWEIGHTS).all().item()),
            "nonfoot_peak_force_body_attribution_complete": peak_body_attribution_complete(
                max_nonfoot_force,
                max_nonfoot_force_step,
                max_nonfoot_force_body_index,
                nonfoot_ids=nonfoot_ids,
                body_count=len(sensor.body_names),
            ),
            "nonfoot_excess_impulse_bounded": bool((excess_delta_v <= MAX_EXCESS_CONTACT_DELTA_V_M_S).all().item()),
            "tail_horizontal_speed_settled": bool((tail_h <= MAX_TAIL_HORIZONTAL_SPEED_M_S).all().item()),
            "tail_vertical_speed_settled": bool((tail_v <= MAX_TAIL_VERTICAL_SPEED_M_S).all().item()),
            "tail_angular_speed_settled": bool((tail_a <= MAX_TAIL_ANGULAR_SPEED_RAD_S).all().item()),
            "reset_pose_hold_actions_bounded": bool((torch.abs(actions[POSE_COUNT:]) <= 1.0).all().item()),
            **reset_pose_hold_checks(reset_pose_hold_diagnostics),
            "zero_normalized_actions_are_zero": bool(
                torch.equal(actions[:POSE_COUNT], torch.zeros_like(actions[:POSE_COUNT]))
            ),
            "base_contact_not_a_termination": "base_contact" not in terms,
            "source_binding_files_present": source_bundle["all_files_present"],
            "source_binding_git_commit_valid": source_bundle["git_commit_valid"],
            "source_binding_clean": source_bundle["clean"],
        }
        health_names = (
            "observations_finite_at_reset",
            "rollout_state_finite_all_pose_modes",
            "no_numeric_invalid_termination",
        )
        status = summarize_status(checks, health_names)
        startup.update(
            {
                "articulation_solver_iterations": (
                    solver_iteration_readback
                ),
                "rigid_body_max_depenetration_velocity": (
                    max_depenetration_velocity_readback
                ),
                "contact_separation_method": "PhysX per-physics-step contact report event without GPU filter",
                "contact_separation_api_available": separation_available,
                "contact_separation_error": separation_error,
                "contact_report_event_count": separation_report["event_count"],
                "contact_report_expected_physics_steps": (
                    args.rollout_steps * raw_env.cfg.decimation
                ),
                "contact_report_headers_seen": contact_report_headers_seen,
                "robot_ground_headers_seen": robot_ground_headers_seen,
                "contact_force_history_steps": raw_env.cfg.decimation,
                "contact_force_sampling_dt_s": raw_env.physics_dt,
                "separation_point_count_per_env": _tensor_list(
                    separation_point_counts
                ),
                "minimum_separation_provenance_per_env": separation_report[
                    "minimum_separation_provenance"
                ],
            }
        )
        return {
            "schema_version": 3,
            "goal_id": "g009",
            "stage_id": "R0",
            "probe": "flat_recover_runtime_calibration",
            "execution": execution,
            "task": args.task,
            "contract_sha256": canonical_sha256(recover_contract()),
            "source_bundle": source_bundle,
            "seed": args.seed,
            "device": args.device,
            "num_envs": args.num_envs,
            "rollout_steps": args.rollout_steps,
            "rollout_duration_s": args.rollout_steps * raw_env.step_dt,
            "headless": bool(args.headless),
            "pose_name_order": list(RECOVER_POSES),
            "action_modes": list(ACTION_MODES),
            "observation_shapes": shapes,
            "observation_calibration": {
                "critic_base_height_m": _tensor_list(critic_height),
                "flat_ground_height_error_m": _tensor_list(height_error),
                "critic_terrain_normal_base": _tensor_list(critic_normal),
                "reset_actor_foot_load_bodyweights": _tensor_list(
                    reset_policy[:, ACTOR_FOOT_LOAD_SLICE]
                ),
                "reset_body_range_normalized": _tensor_list(
                    reset_policy[:, ACTOR_RANGE_SLICE]
                ),
                "reset_body_range_hit_mask": _tensor_list(
                    reset_policy[:, ACTOR_RANGE_MASK_SLICE]
                ),
                "reset_body_range_hit_count": reset_camera_observations[
                    "hit_count_per_env"
                ],
                "body_range_camera_config": range_camera_config,
                "body_range_camera_world_position_m": _tensor_list(
                    reset_camera_position
                ),
                "body_range_camera_world_quaternion_wxyz": _tensor_list(
                    reset_camera_quaternion
                ),
                "body_range_camera_position_error_m": _tensor_list(
                    camera_position_error
                ),
                "body_range_camera_quaternion_error_l2": _tensor_list(
                    camera_quaternion_error
                ),
                "foot_load_range_independence": load_range_independence,
                "privilege_isolation": privilege,
            },
            "physics_readback": startup,
            "device_capability_diagnostics": {
                "ground_contact_separation_available": separation_valid,
                "ground_contact_separation_above_minus_1cm": (
                    separation_threshold_passed
                ),
            },
            "required_crosschecks": {
                "cpu_contact_separation": separation_crosscheck,
            },
            "final_synthesis": {
                "status": "not_run",
                "passed": None,
                "reason": (
                    "combine GPU blocking dynamics with the authoritative CPU "
                    "contact-separation crosscheck"
                ),
            },
            "reward_manager_calibration": {
                "expected": reward_temporal_expectations(raw_env.step_dt),
                "observed": manager_probe,
            },
            "action_readback": {
                "type": type(action_term).__name__,
                "scale": runtime_action_scale,
                "ema_alpha": float(action_term.cfg.alpha),
                "ema_reset_target_max_error_rad_per_env": _tensor_list(
                    ema_reset_target_error
                ),
                "asset_soft_joint_limit_factor": runtime_soft_limit_factor,
                "effective_target_hard_limit_range_fraction": (
                    runtime_action_scale * runtime_soft_limit_factor
                ),
            },
            "action_dim": action_dim,
            "active_terminations": terms,
            "reset": {
                "source_class_ids": _tensor_list(class_ids),
                "source_class_one_hot": _tensor_list(class_one_hot),
                "root_position_local_m": _tensor_list(root_local),
                "root_quaternion_wxyz": _tensor_list(root_pose[:, 3:7]),
                "body_up_error_l2": _tensor_list(body_up_error),
                "joint_position_rad": _tensor_list(joint_position),
                "joint_hard_limits_rad": _tensor_list(joint_limits[0]),
                "hold_normalized_action": _tensor_list(actions[POSE_COUNT:]),
                "hold_unclamped_normalized_action": _tensor_list(
                    reset_pose_hold_diagnostics["unclamped_normalized_action"]
                ),
                "hold_reachable_processed_joint_target_rad": _tensor_list(
                    reset_pose_hold_diagnostics["reachable_target"]
                ),
                "hold_reachable_target_error_rad": _tensor_list(
                    reset_pose_hold_diagnostics["target_error"]
                ),
            },
            "calibration_thresholds": {
                "solver_joint_limit_tolerance_rad": SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
                "min_root_height_m": MIN_ROOT_HEIGHT_M,
                "min_contact_separation_m": MIN_CONTACT_SEPARATION_M,
                "max_nonfoot_force_bodyweights": MAX_NON_FOOT_FORCE_BODYWEIGHTS,
                "reset_hold_target_tolerance_rad": RESET_HOLD_TARGET_TOLERANCE_RAD,
                "max_excess_contact_delta_v_m_s": MAX_EXCESS_CONTACT_DELTA_V_M_S,
                "tail_window_steps": TAIL_STEPS,
                "max_tail_horizontal_speed_m_s": MAX_TAIL_HORIZONTAL_SPEED_M_S,
                "max_tail_vertical_speed_m_s": MAX_TAIL_VERTICAL_SPEED_M_S,
                "max_tail_angular_speed_rad_s": MAX_TAIL_ANGULAR_SPEED_RAD_S,
            },
            "pose_mode_metrics": rows,
            "checks": checks,
            **status,
        }
    finally:
        contact_report_subscription = None
        env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=EXPECTED_NUM_ENVS)
    parser.add_argument("--rollout-steps", type=int, default=EXPECTED_ROLLOUT_STEPS)
    parser.add_argument("--output", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    output, execution = prepare_execution(parse_prelaunch_output(argv))
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = probe(args, execution)
        _write_json_atomic(output, report)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "run_health_passed": report["run_health"]["passed"],
                    "runtime_contract_passed": report["runtime_contract"]["passed"],
                    "qualification_status": report["qualification"]["status"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if report["runtime_contract"]["passed"] else 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
