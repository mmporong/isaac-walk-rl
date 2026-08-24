#!/usr/bin/env python3
"""Evaluate one G006 rough checkpoint with fixed push or no-push guardrail trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaac_walk_g006.evaluation.protocol import (  # noqa: E402
    PUSH_INJECTION_COMPLETED_STEPS,
    TOTAL_STEPS,
    PushRecoveryStateMachine,
    build_guardrail_trials,
    build_push_trials,
    compute_evaluation_source_bundle,
    tile_boundary_violation,
    validate_success_criteria,
    wilson_interval,
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(Path.home().resolve())
    except ValueError:
        return str(resolved)
    return "%USERPROFILE%\\" + str(relative)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("protocol root must be an object")
    return value, canonical_sha256(protocol_section(value))


def protocol_section(manifest: dict[str, Any]) -> dict[str, Any]:
    section = manifest.get("evaluation_protocol", manifest.get("protocol"))
    if not isinstance(section, dict):
        raise ValueError("manifest evaluation_protocol/protocol object is required")
    return section


def initial_states(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    section = protocol_section(manifest)
    states = section.get("initial_states", manifest.get("initial_states"))
    if not isinstance(states, list) or len(states) != 10 or not all(isinstance(item, dict) for item in states):
        raise ValueError("manifest must contain exactly 10 literal initial_states")
    return states


def make_trial_plan(manifest: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    states = initial_states(manifest)
    section = protocol_section(manifest)
    commands = section.get("commands")
    if mode == "push":
        return build_push_trials(states, commands, section.get("push_directions"), section.get("push_magnitudes_mps"))
    return build_guardrail_trials(states, commands)


def validate_terrain_evidence(entries: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [entry for entry in entries if entry.get("row") in (1, 4, 8) and entry.get("col") in range(10)]
    if len(selected) != 30:
        raise RuntimeError(f"expected 30 held-out evidence tiles, found {len(selected)}")
    for hash_name in ("raw_sha256", "mesh_sha256"):
        values = [entry.get(hash_name) for entry in selected]
        if any(not isinstance(value, str) or len(value) != 64 for value in values) or len(set(values)) != 30:
            raise RuntimeError(f"30 unique {hash_name} values are required")
    metric_names = (
        "height_rms_m",
        "height_p90_abs_m",
        "face_normal_slope_rms_rad",
        "face_normal_slope_p90_rad",
    )
    aggregates: dict[str, dict[str, float]] = {}
    for row, label in ((1, "low"), (4, "mid"), (8, "high")):
        row_entries = [entry for entry in selected if entry["row"] == row]
        aggregates[label] = {
            metric: sum(float(entry["metrics"][metric]) for entry in row_entries) / len(row_entries)
            for metric in metric_names
        }
    for metric in metric_names:
        if not aggregates["low"][metric] < aggregates["mid"][metric] < aggregates["high"][metric]:
            raise RuntimeError(f"terrain metric is not strictly monotonic: {metric}")
        for col in range(10):
            by_row = {entry["row"]: entry for entry in selected if entry["col"] == col}
            if not (
                float(by_row[1]["metrics"][metric])
                < float(by_row[4]["metrics"][metric])
                < float(by_row[8]["metrics"][metric])
            ):
                raise RuntimeError(f"terrain metric is not paired-monotonic: col={col} metric={metric}")
    return {"selected_tiles": selected, "difficulty_aggregates": aggregates}


def _tensor_state(torch: Any, state_literals: list[dict[str, Any]], common: dict[str, Any], robot: Any, device: str) -> tuple[dict[str, Any], str]:
    if any(set(state) != {"id", "root_relative_pos_m"} for state in state_literals):
        raise ValueError("each literal initial state requires only id and root_relative_pos_m")
    root_quat = common.get("root_quaternion_wxyz")
    root_linear = common.get("root_linear_velocity_mps")
    root_angular = common.get("root_angular_velocity_radps")
    if not (isinstance(root_quat, list) and len(root_quat) == 4 and isinstance(root_linear, list) and len(root_linear) == 3 and isinstance(root_angular, list) and len(root_angular) == 3):
        raise ValueError("invalid initial_state_common vectors")
    root_pose = torch.tensor(
        [list(state["root_relative_pos_m"]) + root_quat for state in state_literals], dtype=torch.float32, device=device
    )
    root_velocity = torch.tensor([root_linear + root_angular for _ in state_literals], dtype=torch.float32, device=device)
    joint_position = robot.data.default_joint_pos[:1].repeat(len(state_literals), 1).clone()
    joint_velocity = torch.zeros_like(joint_position)
    values = {
        "root_pose": root_pose,
        "root_velocity": root_velocity,
        "joint_position": joint_position,
        "joint_velocity": joint_velocity,
    }
    vector_hash = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in values.values())).hexdigest()
    return {"articulation": {"robot": values}, "deformable_object": {}, "rigid_object": {}}, vector_hash


def _finalize_trial(trial: dict[str, Any], state: PushRecoveryStateMachine, sums: dict[str, float], samples: int) -> dict[str, Any]:
    result = {key: value for key, value in trial.items() if key != "initial_state"}
    result.update(state.finalize())
    result["sample_count"] = samples
    for key in ("tracking_error_sq", "yaw_error_sq", "torque_l2", "absolute_mechanical_power", "action_rate_l2"):
        result[key + "_mean"] = None if samples == 0 else sums[key] / samples
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaac_walk_g006.evaluation.terrain_cfg import UnitreeGo2EvidenceRoughEnvCfg
    from isaac_walk_g006.evaluation.terrain_generator import latest_terrain_evidence

    manifest_path = args.protocol.resolve()
    manifest, protocol_hash = load_protocol(manifest_path)
    protocol = protocol_section(manifest)
    success_criteria = validate_success_criteria(protocol.get("success_criteria"))
    trials = make_trial_plan(manifest, args.mode)
    checkpoint_pair_id = f"{args.variant}-s{args.training_seed}"
    for trial in trials:
        trial["paired_trial_key"] = trial.pop("pair_id", trial["trial_id"])
        trial["trial_id"] = f"{checkpoint_pair_id}:{trial['trial_id']}"
        trial["pair_id"] = checkpoint_pair_id
    expected_count = 1080 if args.mode == "push" else 90
    if len(trials) != expected_count:
        raise ValueError("trial-plan count mismatch")
    if args.variant not in {"baseline", "push_curriculum"}:
        raise ValueError("variant must be baseline or push_curriculum")

    env_cfg = UnitreeGo2EvidenceRoughEnvCfg()
    env_cfg.scene.num_envs = expected_count
    env_cfg.seed = int(protocol.get("evaluation_seed", 20260824))
    env_cfg.curriculum.terrain_levels = None
    env_cfg.observations.policy.enable_corruption = False
    for event_name in tuple(env_cfg.events.to_dict()):
        if not event_name.startswith("_"):
            setattr(env_cfg.events, event_name, None)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.terminations.base_contact.params["threshold"] = 1.0
    threshold = float(env_cfg.terminations.base_contact.params["threshold"])
    if threshold != 1.0:
        raise ValueError("base_contact threshold must be exactly 1 N")

    task = str(protocol.get("task", "Isaac-Velocity-Rough-Unitree-Go2-v0"))
    if "Play" in task:
        raise ValueError("full rough task is required; Play config is prohibited")
    agent_cfg = load_cfg_from_registry("Isaac-Velocity-Rough-Unitree-Go2-v0", "rsl_rl_cfg_entry_point")
    agent_cfg.device = getattr(args, "device", "cuda:0")
    env = gym.make(task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint.resolve()))
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
    base = wrapped.unwrapped
    robot = base.scene["robot"]

    # Pin every environment to its declared held-out row/column before reset_to.
    terrain_rows = torch.tensor([trial["terrain_row"] for trial in trials], dtype=torch.long, device=base.device)
    terrain_cols = torch.tensor([trial["terrain_col"] for trial in trials], dtype=torch.long, device=base.device)
    terrain = base.scene.terrain
    terrain.terrain_levels.copy_(terrain_rows)
    terrain.terrain_types.copy_(terrain_cols)
    terrain.env_origins.copy_(terrain.terrain_origins[terrain_rows, terrain_cols])
    base.scene.env_origins.copy_(terrain.env_origins)

    literal_by_trial = [trial["initial_state"] for trial in trials]
    common = protocol.get("initial_state_common")
    if not isinstance(common, dict):
        raise ValueError("initial_state_common object is required")
    state, initial_state_vector_hash = _tensor_state(torch, literal_by_trial, common, robot, base.device)
    env_ids = torch.arange(expected_count, device=base.device)
    base.reset_to(state, env_ids, seed=int(protocol.get("evaluation_seed", 20260824)), is_relative=True)

    fixed_commands = torch.tensor([trial["command"] for trial in trials], dtype=torch.float32, device=base.device)
    command_buffer = base.command_manager.get_command("base_velocity")
    command_buffer.copy_(fixed_commands)
    observations, _ = wrapped.get_observations()  # fresh observation after origin→reset_to→command

    tracking_limit = float(success_criteria["lin_vel_error_mps_max"])
    angular_limit = float(success_criteria["yaw_rate_error_radps_max"])
    roll_limit = float(success_criteria["roll_abs_rad_max"])
    pitch_limit = float(success_criteria["pitch_abs_rad_max"])
    machines = [
        PushRecoveryStateMachine(
            tracking_limit,
            angular_limit,
            roll_limit,
            pitch_limit,
            dwell_required=int(success_criteria["consecutive_post_push_samples"]),
            push_enabled=args.mode == "push",
        )
        for _ in trials
    ]
    metric_keys = ("tracking_error_sq", "yaw_error_sq", "torque_l2", "absolute_mechanical_power", "action_rate_l2")
    sums = [{key: 0.0 for key in metric_keys} for _ in trials]
    samples = [0] * expected_count
    ever_fall = [False] * expected_count
    previous_action = base.action_manager.action.clone()
    push_body_delta = torch.zeros((expected_count, 2), dtype=torch.float32, device=base.device)
    if args.mode == "push":
        push_body_delta = torch.tensor(
            [
                [
                    float(trial["push_magnitude_mps"]) * float(trial["push_direction_body_xy"][0]),
                    float(trial["push_magnitude_mps"]) * float(trial["push_direction_body_xy"][1]),
                ]
                for trial in trials
            ],
            dtype=torch.float32,
            device=base.device,
        )

    try:
        for completed_steps in range(TOTAL_STEPS):
            command_buffer.copy_(fixed_commands)
            observations, _ = wrapped.get_observations()
            with torch.inference_mode():
                actions = policy(observations)
            effective = actions if agent_cfg.clip_actions is None else torch.clamp(actions, -agent_cfg.clip_actions, agent_cfg.clip_actions)

            root_lin_b = robot.data.root_lin_vel_b.clone()
            root_ang_b = robot.data.root_ang_vel_b.clone()
            torques = robot.data.applied_torque.clone()
            joint_vel = robot.data.joint_vel.clone()
            action_delta = effective - previous_action
            previous_action = effective.clone()
            quaternion_before = robot.data.root_quat_w.clone()

            active_indices = [index for index, machine in enumerate(machines) if machine.active]
            metric_matrix = torch.stack(
                (
                    torch.sum((root_lin_b[:, :2] - fixed_commands[:, :2]) ** 2, dim=1),
                    (root_ang_b[:, 2] - fixed_commands[:, 2]).square(),
                    torch.sum(torques**2, dim=1),
                    torch.sum(torch.abs(torques * joint_vel), dim=1),
                    torch.sum(action_delta**2, dim=1),
                ),
                dim=1,
            ).detach().cpu().numpy()
            for index in active_indices:
                sums[index]["tracking_error_sq"] += float(metric_matrix[index, 0])
                sums[index]["yaw_error_sq"] += float(metric_matrix[index, 1])
                sums[index]["torque_l2"] += float(metric_matrix[index, 2])
                sums[index]["absolute_mechanical_power"] += float(metric_matrix[index, 3])
                sums[index]["action_rate_l2"] += float(metric_matrix[index, 4])
                samples[index] += 1

            if args.mode == "push" and completed_steps == PUSH_INJECTION_COMPLETED_STEPS:
                for machine in machines:
                    machine.mark_push(completed_steps)
                root_velocity = robot.data.root_vel_w.clone()
                w, x, y, z = quaternion_before.unbind(dim=1)
                yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                cosine, sine = torch.cos(yaw), torch.sin(yaw)
                root_velocity[:, 0] += cosine * push_body_delta[:, 0] - sine * push_body_delta[:, 1]
                root_velocity[:, 1] += sine * push_body_delta[:, 0] + cosine * push_body_delta[:, 1]
                robot.write_root_velocity_to_sim(root_velocity)

            observations, _, dones, _ = wrapped.step(actions)
            fall = base.termination_manager.get_term("base_contact").clone()
            timeout = base.termination_manager.get_term("time_out").clone()
            unknown_auto_reset = dones.clone() & ~(fall | timeout)
            root_after = robot.data.root_pos_w[:, :2].clone()
            lin_after = robot.data.root_lin_vel_b[:, :2].clone()
            yaw_after = robot.data.root_ang_vel_b[:, 2].clone()
            quaternion_after = robot.data.root_quat_w.clone()
            w, x, y, z = quaternion_after.unbind(dim=1)
            roll_after = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
            pitch_after = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
            origins = terrain.env_origins[:, :2]
            after_matrix = torch.cat(
                (
                    lin_after,
                    yaw_after.unsqueeze(1),
                    roll_after.unsqueeze(1),
                    pitch_after.unsqueeze(1),
                    root_after,
                    origins,
                    fall.unsqueeze(1),
                    unknown_auto_reset.unsqueeze(1),
                ),
                dim=1,
            ).detach().cpu().numpy()
            for index, machine in enumerate(machines):
                if bool(after_matrix[index, 9]):
                    ever_fall[index] = True
                if not machine.active:
                    continue
                command = trials[index]["command"]
                tracking_error = float(((after_matrix[index, 0] - float(command[0])) ** 2 + (after_matrix[index, 1] - float(command[1])) ** 2) ** 0.5)
                angular_error = abs(float(after_matrix[index, 2]) - float(command[2]))
                terminated_now = bool(after_matrix[index, 9])
                auto_reset_now = bool(after_matrix[index, 10])
                boundary = False if (terminated_now or auto_reset_now) else tile_boundary_violation(after_matrix[index, 5:7], after_matrix[index, 7:9])
                machine.observe(
                    completed_steps + 1,
                    tracking_error=tracking_error,
                    angular_error=angular_error,
                    roll=float(after_matrix[index, 3]),
                    pitch=float(after_matrix[index, 4]),
                    terminated=terminated_now,
                    auto_reset_detected=auto_reset_now,
                    boundary_violation=boundary,
                )
    finally:
        wrapped.close()

    trial_reports = [_finalize_trial(trial, machines[index], sums[index], samples[index]) for index, trial in enumerate(trials)]
    for index, trial in enumerate(trial_reports):
        trial["guardrail_eligible"] = trial["excluded_reason"] != "auto_reset_poison"
        trial["guardrail_survived"] = trial["guardrail_eligible"] and bool(trial["survived_to_horizon"])
    eligible = [trial for trial in trial_reports if trial["eligible"]]
    recovered = sum(bool(trial["recovered"]) for trial in eligible)
    survived = sum(bool(trial["survived_to_horizon"]) for trial in (trial_reports if args.mode == "guardrail" else eligible))
    recovery_ci = wilson_interval(recovered, len(eligible))
    cells: list[dict[str, Any]] = []
    for cell_id in sorted({trial["stratum_id"] for trial in trial_reports}):
        members = [trial for trial in trial_reports if trial["stratum_id"] == cell_id]
        cell_eligible = [trial for trial in members if trial["eligible"]]
        cell_recovered = sum(bool(trial["recovered"]) for trial in cell_eligible)
        raw_metrics = {}
        for metric in (
            "tracking_error_sq_mean",
            "yaw_error_sq_mean",
            "torque_l2_mean",
            "absolute_mechanical_power_mean",
            "action_rate_l2_mean",
        ):
            values = [float(trial[metric]) for trial in members if trial[metric] is not None]
            raw_metrics[metric] = None if not values else sum(values) / len(values)
        cells.append({
            "cell_id": cell_id,
            "trial_count": len(members),
            "eligible_count": len(cell_eligible),
            "recovered_count": cell_recovered,
            "recovery_rate": None if not cell_eligible else cell_recovered / len(cell_eligible),
            "wilson95": list(wilson_interval(cell_recovered, len(cell_eligible))),
            "survival_rate": (
                None
                if not any(trial["guardrail_eligible"] for trial in members)
                else sum(bool(trial["guardrail_survived"]) for trial in members)
                / sum(bool(trial["guardrail_eligible"]) for trial in members)
            ),
            "prepush_failure_count": sum(bool(trial["prepush_failure"]) for trial in members),
            "raw_metrics": raw_metrics,
        })

    terrain_evidence = validate_terrain_evidence(latest_terrain_evidence())
    checkpoint_hash = file_sha256(args.checkpoint.resolve())
    evaluation_bundle = compute_evaluation_source_bundle(REPO_ROOT)
    report = {
        "schema_version": 1,
        "goal": "G006",
        "status": "complete",
        "protocol_compliant": not args.smoke_only,
        "experimental_use": "smoke_only" if args.smoke_only else "g006_production_evaluation",
        "mode": args.mode,
        "variant": args.variant,
        "training_seed": args.training_seed,
        "protocol": {"path": portable_path(manifest_path), "sha256": protocol_hash},
        "checkpoint": {"path": portable_path(args.checkpoint.resolve()), "sha256": checkpoint_hash},
        "evaluation_source_bundle_sha256": evaluation_bundle["sha256"],
        "evaluation_source_bundle_files": evaluation_bundle["files"],
        "success_criteria": success_criteria,
        "runtime": {
            "task": task,
            "headless": bool(args.headless),
            "isaaclab_commit": manifest.get("isaaclab", {}).get("commit"),
            "sim_version": manifest.get("isaaclab", {}).get("sim_version"),
            "device": str(base.device),
            "step_dt": float(base.step_dt),
            "num_envs": expected_count,
            "horizon_steps": TOTAL_STEPS,
            "horizon_completed_step": TOTAL_STEPS,
            "push_injection_completed_steps": PUSH_INJECTION_COMPLETED_STEPS,
            "terrain_levels_runtime": None,
            "observation_corruption": False,
            "events_enabled": [],
            "base_contact_threshold_n": threshold,
            "preliminary": True,
        },
        "terrain_evidence": terrain_evidence,
        "initial_state_vector_sha256": initial_state_vector_hash,
        "trials": trial_reports,
        "cells": cells,
        "aggregate": {
            "trial_count": len(trial_reports),
            "eligible_count": len(eligible),
            "criterion_met_count": sum(bool(trial["criterion_met"]) for trial in eligible),
            "recovered_count": recovered,
            "recovery_rate": None if not eligible else recovered / len(eligible),
            "recovery_wilson95": list(recovery_ci),
            "survival_count": survived,
            "survived_to_horizon_count": survived,
            "survival_rate": (
                None
                if (sum(bool(trial["guardrail_eligible"]) for trial in trial_reports) if args.mode == "guardrail" else len(eligible)) == 0
                else survived
                / (sum(bool(trial["guardrail_eligible"]) for trial in trial_reports) if args.mode == "guardrail" else len(eligible))
            ),
            "prepush_failure_count": sum(bool(trial["prepush_failure"]) for trial in trial_reports),
            "auto_reset_excluded_count": sum(trial["excluded_reason"] == "auto_reset_poison" for trial in trial_reports),
            "boundary_violation_count": sum(trial["excluded_reason"] == "tile_boundary" for trial in trial_reports),
        },
        "warnings": [
            "absolute_mechanical_power_mean is a simulated mechanical-power proxy, not electrical energy.",
            "No observation noise or evaluation-time domain randomization is enabled.",
        ],
    }
    if report["aggregate"]["boundary_violation_count"] or report["aggregate"]["auto_reset_excluded_count"]:
        report["status"] = "protocol_blocked"
    return report


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("baseline", "push_curriculum"))
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=("push", "guardrail"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke-only", action="store_true", help="Run the full tensor/state-machine path without claiming production evidence.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    from isaaclab.app import AppLauncher

    args = parse_args()
    started_at = time.time()
    preliminary: dict[str, Any] | None = None
    try:
        if not args.checkpoint.is_file() or not args.protocol.is_file():
            raise FileNotFoundError("checkpoint and protocol must exist")
        launcher = AppLauncher(args)
        app = launcher.app
        try:
            preliminary = evaluate(args)
            preliminary["runtime"]["started_at_epoch"] = started_at
            write_json_atomic(args.output.resolve(), preliminary)
        finally:
            app.close()
        print(json.dumps({"status": preliminary["status"], "output": str(args.output.resolve())}), flush=True)
        return 0 if preliminary["status"] == "complete" else 1
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "goal": "G006",
            "status": "failed",
            "mode": getattr(args, "mode", None),
            "variant": getattr(args, "variant", None),
            "training_seed": getattr(args, "training_seed", None),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        write_json_atomic(args.output.resolve(), failure)
        print(json.dumps(failure["error"]), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
