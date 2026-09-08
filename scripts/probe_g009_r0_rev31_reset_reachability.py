#!/usr/bin/env python3
"""GPU diagnostic for G009 R0 reset reachability; no policy or PPO is loaded."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for _root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import probe_g009_recover_runtime as runtime_probe  # noqa: E402
from isaac_walk_g009.recover_contracts import SOLVER_JOINT_LIMIT_TOLERANCE_RAD  # noqa: E402


SCHEMA_VERSION = "g009.r0.rev31.reset_reachability.v1"
DEFAULT_TASK = "Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0"
POSE_NAMES = ("prone", "supine", "left_side", "right_side")
ACTION_MODES = ("zero_normalized", "reset_pose_hold")
ACTION_SCALES = (0.70, 0.65, 0.60)
CALF_RESET_CHOICES = (-2.37, -2.28)
NUM_ENVS = 8
ROLLOUT_STEPS = 150
SEED = 42
DEVICE = "cuda:0"
MAX_NONFOOT_FORCE_BW = 15.0
SOURCE_PATHS = (
    "configs/g009_r0.json",
    "configs/g009_r0_rev31_diagnostic.json",
    "scripts/probe_g009_recover_runtime.py",
    "scripts/probe_g009_r0_rev31_reset_reachability.py",
    "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/recover_contracts.py",
    "src/isaac_walk_g009/recover_env_cfg.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot() -> dict[str, Any]:
    files = {relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_PATHS}
    payload = "\n".join(f"{name}:{files[name]}" for name in sorted(files))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_PATHS],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {
        "git_commit": commit,
        "dirty": bool(dirty),
        "dirty_source_paths": dirty,
        "source_files_sha256": files,
        "source_bundle_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def write_json_no_overwrite(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary report: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = (REPO_ROOT / "reports" / "runs").resolve()
    if resolved.parent != root or resolved.suffix.lower() != ".json" or resolved.name.startswith("."):
        raise ValueError("output must be a visible JSON direct child of reports/runs")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite report: {resolved}")
    return resolved


def core_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--calf-reset", type=float, choices=CALF_RESET_CHOICES, default=-2.37)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.task != DEFAULT_TASK:
        raise ValueError(f"task is fixed to {DEFAULT_TASK}")
    if args.seed != SEED or args.num_envs != NUM_ENVS or args.rollout_steps != ROLLOUT_STEPS:
        raise ValueError("diagnostic budget is fixed to seed=42, num_envs=8, rollout_steps=150")
    if args.calf_reset not in CALF_RESET_CHOICES:
        raise ValueError("calf reset must be exactly -2.37 or -2.28")
    if getattr(args, "device", DEVICE) != DEVICE or getattr(args, "headless", True) is not True:
        raise ValueError("runtime is fixed to headless cuda:0")
    canonical_output(args.output)


def parse_prelaunch(argv: list[str] | None = None) -> tuple[Path, float]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calf-reset", type=float, choices=CALF_RESET_CHOICES, default=-2.37)
    values, _ = parser.parse_known_args(argv)
    return canonical_output(values.output), values.calf_reset


def build_assignment() -> list[dict[str, Any]]:
    return [
        {"env_index": index, "pose": POSE_NAMES[index % 4], "action_mode": ACTION_MODES[index // 4]}
        for index in range(NUM_ENVS)
    ]


def diagnostic_checks(
    *, runtime_scale: float, source_stable: bool, finite_all: bool,
    max_nonfoot_force_bw: float | None, reset_selection_reported: bool,
    runtime_config_stable: bool = True, hard_limit_count: int = 0,
    numeric_invalid_count: int = 0, reset_target_error_rad: float = 0.0,
) -> dict[str, bool]:
    return {
        "runtime_action_scale_is_current_0_60": math.isclose(runtime_scale, 0.60, abs_tol=1.0e-9),
        "source_files_unchanged_during_run": source_stable,
        "runtime_configuration_unchanged_during_rollout": runtime_config_stable,
        "all_recorded_runtime_values_finite": finite_all,
        "max_nonfoot_contact_force_within_15_bodyweights": (
            max_nonfoot_force_bw is not None
            and math.isfinite(max_nonfoot_force_bw)
            and max_nonfoot_force_bw <= MAX_NONFOOT_FORCE_BW
        ),
        "reset_selection_explicitly_reported": reset_selection_reported,
        "hard_joint_limit_termination_count_is_zero": hard_limit_count == 0,
        "numeric_invalid_termination_count_is_zero": numeric_invalid_count == 0,
        "reset_hold_target_max_error_within_1e_6_rad": (
            math.isfinite(reset_target_error_rad) and reset_target_error_rad <= 1.0e-6
        ),
    }


def _tensor_summary(value: Any) -> dict[str, Any]:
    finite_values = value[value.isfinite()]
    return {
        "maximum": float(finite_values.max().item()) if finite_values.numel() else None,
        "mean": float(finite_values.mean().item()) if finite_values.numel() else None,
        "nonfinite_count": int((~value.isfinite()).sum().item()),
    }


def _joint_excess(position: Any, limits: Any, torch: Any) -> Any:
    return torch.maximum(
        (limits[..., 0] - position).clamp_min(0.0),
        (position - limits[..., 1]).clamp_min(0.0),
    )


def _state_metrics(
    raw_env: Any, robot: Any, sensor: Any, nonfoot_ids: list[int], torch: Any,
    env_ids: Any = slice(None),
) -> dict[str, Any]:
    position = robot.data.joint_pos[env_ids]
    hard_excess = _joint_excess(position, robot.data.joint_pos_limits[env_ids], torch)
    hard_threshold_limits = robot.data.joint_pos_limits[env_ids].clone()
    hard_threshold_limits[..., 0] -= SOLVER_JOINT_LIMIT_TOLERANCE_RAD
    hard_threshold_limits[..., 1] += SOLVER_JOINT_LIMIT_TOLERANCE_RAD
    hard_threshold_excess = _joint_excess(position, hard_threshold_limits, torch)
    soft_excess = _joint_excess(position, robot.data.soft_joint_pos_limits[env_ids], torch)
    torque = robot.data.applied_torque[env_ids].abs()
    velocity = robot.data.joint_vel[env_ids].abs()
    forces = sensor.data.net_forces_w_history[env_ids, :, nonfoot_ids, :]
    nonfoot_force = torch.linalg.vector_norm(forces, dim=-1).amax(dim=(1, 2))
    masses = raw_env._g009_r0_body_mass[env_ids].sum(dim=1)
    nonfoot_bw = nonfoot_force / (masses * 9.81)
    root_quat = robot.data.root_quat_w[env_ids]
    world_up = torch.tensor((0.0, 0.0, 1.0), device=root_quat.device, dtype=root_quat.dtype).expand(len(root_quat), -1)
    from isaaclab.utils import math as math_utils  # pyright: ignore[reportMissingImports]
    body_up = math_utils.quat_apply(root_quat, world_up)
    tilt = torch.acos(body_up[:, 2].clamp(-1.0, 1.0))
    root_pos = robot.data.root_pos_w[env_ids]
    origins = raw_env.scene.env_origins[env_ids]
    values = (position, hard_excess, hard_threshold_excess, soft_excess, torque, velocity, nonfoot_bw, tilt, root_pos)
    return {
        "finite": all(bool(torch.isfinite(value).all().item()) for value in values),
        "hard_joint_excess_rad": _tensor_summary(hard_excess),
        "hard_termination_threshold_excess_rad": _tensor_summary(hard_threshold_excess),
        "soft_joint_excess_rad": _tensor_summary(soft_excess),
        "applied_torque_nm_abs": _tensor_summary(torque),
        "joint_velocity_rad_s_abs": _tensor_summary(velocity),
        "max_nonfoot_contact_force_bodyweights": _tensor_summary(nonfoot_bw)["maximum"],
        "root_height_m": _tensor_summary(root_pos[:, 2] - origins[:, 2]),
        "tilt_rad": _tensor_summary(tilt),
    }


def _reachability(target: Any, soft_limits: Any, joint_names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scale in ACTION_SCALES:
        item = runtime_probe.reset_pose_hold_action_diagnostics(
            target, soft_limits, joint_names, action_scale=scale
        )
        checks = runtime_probe.reset_pose_hold_checks(item)
        result[f"{scale:.2f}"] = {
            "action_scale": scale,
            "reachable": bool(checks["reset_pose_hold_reachable_targets_match_reset_positions"]),
            "unsaturated": bool(checks["reset_pose_hold_actions_unsaturated"]),
            "finite": bool(checks["reset_pose_hold_action_diagnostics_finite"]),
            "max_target_error_rad": float(item["max_target_error"].max().item()),
            "saturated_joint_names_by_env": item["saturated_joint_names"],
        }
    return result


def _runtime_config_readback(action_term: Any, reset_params: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_scale": float(action_term.cfg.scale),
        "action_rescale_to_limits": bool(action_term.cfg.rescale_to_limits),
        "action_ema_alpha": float(action_term.cfg.alpha),
        "reset_assignment_mode": reset_params["assignment_mode"],
        "reset_pose_xy_range_m": list(reset_params["pose_xy_range"]),
        "reset_yaw_range_rad": list(reset_params["yaw_range"]),
        "reset_calf_angle_rad": float(reset_params["calf_angle"]),
    }


class PreResetObserver:
    """Capture terminal physics state before ManagerBasedRLEnv auto-reset overwrites it."""

    def __init__(self, raw_env: Any, robot: Any, sensor: Any, nonfoot_ids: list[int], torch: Any):
        self.raw_env = raw_env
        self.robot = robot
        self.sensor = sensor
        self.nonfoot_ids = nonfoot_ids
        self.torch = torch
        self.control_step = 0
        self.rows: list[dict[str, Any]] = []
        self.termination_counts = {name: 0 for name in raw_env.termination_manager.active_terms}

    def capture(self, env_ids: Any) -> None:
        ids = env_ids.to(device=self.raw_env.device, dtype=self.torch.long)
        metrics = _state_metrics(
            self.raw_env, self.robot, self.sensor, self.nonfoot_ids, self.torch, ids
        )
        terminations = {}
        for name in self.termination_counts:
            count = int(self.raw_env.termination_manager.get_term(name)[ids].sum().item())
            self.termination_counts[name] += count
            terminations[name] = count
        self.rows.append({
            "control_step": self.control_step,
            "env_indices": ids.detach().cpu().tolist(),
            "sampling_boundary": "RecorderManager.record_pre_reset before auto-reset",
            "terminations": terminations,
            **metrics,
        })


def install_pre_reset_observer(recorder_manager: Any, observer: PreResetObserver) -> Any:
    if list(recorder_manager.active_terms):
        raise RuntimeError("diagnostic requires zero active recorder terms")
    original = recorder_manager.record_pre_reset

    def observed(env_ids: Any, force_export_or_skip: Any = None) -> Any:
        observer.capture(env_ids)
        return original(env_ids, force_export_or_skip)

    recorder_manager.record_pre_reset = observed
    return original


def run_diagnostic(args: argparse.Namespace, execution: dict[str, Any]) -> dict[str, Any]:
    import gymnasium as gym  # pyright: ignore[reportMissingImports]
    import torch
    import isaaclab_tasks  # noqa: F401  # pyright: ignore[reportMissingImports]
    from isaaclab_tasks.utils import parse_env_cfg  # pyright: ignore[reportMissingImports]
    from isaac_walk_g009 import register_tasks

    before = source_snapshot()
    register_tasks()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.contact_forces.history_length = env_cfg.decimation
    reset_params = env_cfg.events.reset_base.params
    reset_params.update({
        "assignment_mode": "stratified", "pose_xy_range": (0.0, 0.0), "yaw_range": (0.0, 0.0),
        "calf_angle": args.calf_reset,
    })
    env_cfg.validate()
    env = gym.make(args.task, cfg=env_cfg)
    try:
        raw_env = env.unwrapped
        robot = raw_env.scene["robot"]
        sensor = raw_env.scene.sensors["contact_forces"]
        action_term = raw_env.action_manager.get_term("joint_pos")
        runtime_scale = float(action_term.cfg.scale)
        runtime_config_before = _runtime_config_readback(action_term, reset_params)
        observations, _ = env.reset()
        del observations
        class_ids = raw_env._g009_recover_fall_class.detach().cpu().tolist()
        expected_class_ids = [index % 4 for index in range(NUM_ENVS)]
        if class_ids != expected_class_ids:
            raise RuntimeError(f"stratified pose assignment mismatch: {class_ids}")
        target = robot.data.joint_pos.detach().clone()
        hard_limits = robot.data.joint_pos_limits.detach().clone()
        soft_limits = robot.data.soft_joint_pos_limits.detach().clone()
        reachability = _reachability(target, soft_limits, list(robot.joint_names))
        hold = runtime_probe.reset_pose_hold_action_diagnostics(
            target[4:], soft_limits[4:], list(robot.joint_names), action_scale=runtime_scale
        )
        actions = torch.zeros((NUM_ENVS, raw_env.action_manager.total_action_dim), device=raw_env.device)
        actions[4:] = hold["normalized_action"]
        nonfoot_ids = [i for i, name in enumerate(sensor.body_names) if not name.endswith("_foot")]
        if not nonfoot_ids:
            raise RuntimeError("contact sensor resolved no non-foot bodies")
        terms = list(raw_env.termination_manager.active_terms)
        observer = PreResetObserver(raw_env, robot, sensor, nonfoot_ids, torch)
        recorder_manager = raw_env.recorder_manager
        original_pre_reset = install_pre_reset_observer(recorder_manager, observer)
        reset_metrics = _state_metrics(raw_env, robot, sensor, nonfoot_ids, torch)
        timeline: list[dict[str, Any]] = []
        for step in range(1, ROLLOUT_STEPS + 1):
            observer.control_step = step
            env.step(actions)
            row = _state_metrics(raw_env, robot, sensor, nonfoot_ids, torch)
            row["control_step"] = step
            row["terminations"] = {}
            for name in terms:
                count = int(raw_env.termination_manager.get_term(name).sum().item())
                row["terminations"][name] = count
            timeline.append(row)
        recorder_manager.record_pre_reset = original_pre_reset
        if recorder_manager.record_pre_reset is not original_pre_reset:
            raise RuntimeError("pre-reset observer was not restored")
        final_metrics = timeline[-1]
        finite_all = bool(
            reset_metrics["finite"]
            and all(row["finite"] for row in timeline)
            and all(row["finite"] for row in observer.rows)
        )
        force_samples = [reset_metrics["max_nonfoot_contact_force_bodyweights"]] + [
            row["max_nonfoot_contact_force_bodyweights"] for row in timeline
        ] + [row["max_nonfoot_contact_force_bodyweights"] for row in observer.rows]
        max_force = max(force_samples) if all(value is not None for value in force_samples) else None
        after = source_snapshot()
        stable = before["source_files_sha256"] == after["source_files_sha256"]
        runtime_config_after = _runtime_config_readback(action_term, reset_params)
        runtime_config_stable = runtime_config_before == runtime_config_after
        candidate_applied = (args.calf_reset == -2.28)
        hard_count = observer.termination_counts.get("hard_joint_limit", 0)
        numeric_count = observer.termination_counts.get("numeric_invalid", 0)
        reset_target_error = float(hold["max_target_error"].max().item())
        checks = diagnostic_checks(
            runtime_scale=runtime_scale, source_stable=stable, finite_all=finite_all,
            max_nonfoot_force_bw=max_force,
            reset_selection_reported=(args.calf_reset in CALF_RESET_CHOICES),
            runtime_config_stable=runtime_config_stable,
            hard_limit_count=hard_count,
            numeric_invalid_count=numeric_count,
            reset_target_error_rad=reset_target_error,
        )
        diagnostic_passed = all(checks.values()) and reachability["0.60"]["reachable"]
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "diagnostic_complete",
            "diagnostic_result": "PASS" if diagnostic_passed else "FAIL",
            "execution": execution,
            "task": args.task,
            "seed": args.seed,
            "budget": {"num_envs": NUM_ENVS, "rollout_steps": ROLLOUT_STEPS, "policy_updates": 0, "optimizer_updates": 0},
            "runtime": {"device": args.device, "headless": bool(args.headless), "action_scale": runtime_scale},
            "configuration": {
                "requested": {
                    "task": args.task, "seed": args.seed, "num_envs": args.num_envs,
                    "rollout_steps": args.rollout_steps, "calf_reset_rad": args.calf_reset,
                    "device": args.device, "headless": bool(args.headless),
                },
                "runtime_before": runtime_config_before,
                "runtime_after": runtime_config_after,
                "runtime_stable": runtime_config_stable,
            },
            "checkpoint": None,
            "preregistration": {
                "path": "configs/g009_r0_rev31_diagnostic.json",
                "sha256": before["source_files_sha256"]["configs/g009_r0_rev31_diagnostic.json"],
            },
            "assignment": build_assignment(),
            "reset_override": {
                "parameter": "env_cfg.events.reset_base.params.calf_angle",
                "canonical_default_rad": -2.37,
                "requested_rad": args.calf_reset,
                "candidate_applied": candidate_applied,
                "source_config_mutated": False,
                "routing_scope": "diagnostic env_cfg only; training pose curriculum and source remain unchanged",
            },
            "limits": {
                "joint_names": list(robot.joint_names),
                "hard_rad_by_joint": dict(zip(list(robot.joint_names), hard_limits[0].detach().cpu().tolist())),
                "soft_rad_by_joint": dict(zip(list(robot.joint_names), soft_limits[0].detach().cpu().tolist())),
                "identical_across_environments": bool(
                    torch.equal(hard_limits, hard_limits[0:1].expand_as(hard_limits))
                    and torch.equal(soft_limits, soft_limits[0:1].expand_as(soft_limits))
                ),
                "runtime_soft_joint_pos_limit_factor": float(robot.cfg.soft_joint_pos_limit_factor),
                "hard_joint_limit_margin_rad": SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
            },
            "reset_reachability_by_action_scale": reachability,
            "runtime_observations": {
                "initial_reset": reset_metrics,
                "step_1": timeline[0],
                "per_step_aggregate": timeline,
                "final": final_metrics,
                "terminal_pre_reset_samples": observer.rows,
                "termination_counts_pre_reset": observer.termination_counts,
                "max_nonfoot_contact_force_bodyweights": max_force,
                "sampling_limits": {
                    "per_step_state": "post env.step and may be an auto-reset state for terminated environments",
                    "terminal_state": "captured separately at RecorderManager.record_pre_reset before auto-reset",
                },
            },
            "checks": checks,
            "source_state": {"before": before, "after": after, "hashes_stable": stable},
            "claim_limits": {
                "diagnostic_only": True,
                "official_evaluation": False,
                "checkpoint_qualification": False,
                "recovery_success_measured": False,
                "action_scale_or_calf_reset_causal_proof": False,
                "initial_unreachable_stops_rollout": False,
                "diagnostic_fail_is_operational_failure": False,
                "original_isaaclab_source_modified": False,
                "source_configuration_mutated": False,
            },
        }
    finally:
        if "recorder_manager" in locals() and "original_pre_reset" in locals():
            recorder_manager.record_pre_reset = original_pre_reset
        env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]
    parser = core_parser()
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, device=DEVICE)
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def main(argv: list[str] | None = None) -> int:
    output, calf_reset = parse_prelaunch(argv)
    execution = {
        "execution_id": uuid.uuid4().hex,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "output_path_repo_relative": output.relative_to(REPO_ROOT).as_posix(),
        "no_overwrite": True,
        "requested_calf_reset_rad": calf_reset,
    }
    args = parse_args(argv)
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]
    started = time.monotonic()
    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        report = run_diagnostic(args, execution)
        report["wall_time_seconds"] = round(time.monotonic() - started, 3)
        write_json_no_overwrite(output, report)
        print(json.dumps({"output": str(output), "diagnostic_result": report["diagnostic_result"]}), flush=True)
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
