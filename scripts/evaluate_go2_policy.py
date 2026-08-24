#!/usr/bin/env python3
"""Evaluate one Go2 checkpoint with the fixed G005 command-grid protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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


def resolve_portable_path(value: str) -> Path:
    if value.upper().startswith("%USERPROFILE%"):
        value = str(Path.home()) + value[len("%USERPROFILE%") :]
    return Path(os.path.expandvars(value)).resolve()


def _gpu_snapshot() -> dict[str, Any]:
    try:
        memory = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "measurement_complete": True,
            "memory_used_mib": [int(value.strip()) for value in memory.stdout.splitlines() if value.strip()],
            "compute_pids": [int(value.strip()) for value in processes.stdout.splitlines() if value.strip().isdigit()],
            "error": None,
        }
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"measurement_complete": False, "memory_used_mib": None, "compute_pids": None, "error": str(exc)}


def _find_kit_log(checkpoint_path: Path, started_at: float) -> Path | None:
    for ancestor in checkpoint_path.resolve().parents:
        log_root = ancestor / "_isaac_sim" / "kit" / "logs" / "Kit" / "Isaac-Sim"
        if not log_root.is_dir():
            continue
        candidates = [path for path in log_root.rglob("kit_*.log") if path.stat().st_mtime >= started_at - 2.0]
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    return None


def _fatal_scan(checkpoint_path: Path, started_at: float) -> dict[str, Any]:
    log_path = _find_kit_log(checkpoint_path, started_at)
    if log_path is None:
        return {"measurement_complete": False, "log_reference": None, "patterns": [], "count": None}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = [pattern for pattern in ("[Error]", "Traceback", "RuntimeError") if pattern in text]
    return {
        "measurement_complete": True,
        "log_reference": portable_path(log_path),
        "patterns": patterns,
        "count": len(patterns),
    }


def command_grid(spec: dict[str, Any]) -> list[dict[str, float | str]]:
    expected_axes = {
        "vx_mps": [-1.0, 0.0, 1.0],
        "vy_mps": [-0.5, 0.0, 0.5],
        "yaw_rate_radps": [-0.5, 0.0, 0.5],
    }
    for key, expected in expected_axes.items():
        values = spec.get(key)
        if values != expected or len(set(values)) != len(values):
            raise ValueError(f"command_grid.{key} must be exactly {expected}")
    excluded = spec.get("exclude")
    if excluded != [[0.0, 0.0, 0.0]]:
        raise ValueError("command_grid.exclude must be exactly [[0.0, 0.0, 0.0]]")
    excluded_tuples = {tuple(float(value) for value in item) for item in excluded}
    commands: list[dict[str, float | str]] = []
    for vx in spec["vx_mps"]:
        for vy in spec["vy_mps"]:
            for yaw in spec["yaw_rate_radps"]:
                if (float(vx), float(vy), float(yaw)) in excluded_tuples:
                    continue
                commands.append({
                    "id": f"vx{vx:+.1f}_vy{vy:+.1f}_yaw{yaw:+.1f}",
                    "vx_mps": vx,
                    "vy_mps": vy,
                    "yaw_rate_radps": yaw,
                })
    return commands


def _new_accumulator(initial_trials: int) -> dict[str, float | int]:
    return {
        "sample_count": 0,
        "lin_vel_error_sq_sum": 0.0,
        "yaw_rate_error_sq_sum": 0.0,
        "torque_l2_sum": 0.0,
        "absolute_mechanical_power_sum": 0.0,
        "action_rate_l2_sum": 0.0,
        "feet_air_time_sum": 0.0,
        "first_contact_air_time_sum": 0.0,
        "first_contact_count": 0,
        "fall_count": 0,
        "timeout_count": 0,
        "reset_count": 0,
        "trials_started": initial_trials,
    }


def finalize_accumulator(acc: dict[str, float | int]) -> dict[str, float | int | None]:
    samples = int(acc["sample_count"])
    contacts = int(acc["first_contact_count"])
    trials = int(acc["trials_started"])
    fall_count = int(acc["fall_count"])

    def mean(key: str, denominator: int) -> float | None:
        return None if denominator == 0 else float(acc[key]) / denominator

    lin_mse = mean("lin_vel_error_sq_sum", samples)
    yaw_mse = mean("yaw_rate_error_sq_sum", samples)
    fall_rate = None if trials == 0 else fall_count / trials
    return {
        "sample_count": samples,
        "lin_vel_rmse_mps": None if lin_mse is None else math.sqrt(lin_mse),
        "yaw_rate_rmse_radps": None if yaw_mse is None else math.sqrt(yaw_mse),
        "torque_l2_mean": mean("torque_l2_sum", samples),
        "absolute_mechanical_power_w": mean("absolute_mechanical_power_sum", samples),
        "action_rate_l2_mean": mean("action_rate_l2_sum", samples),
        "feet_air_time_raw_mean": mean("feet_air_time_sum", samples),
        "mean_air_time_at_first_contact_s": mean("first_contact_air_time_sum", contacts),
        "first_contact_count": contacts,
        "fall_count": fall_count,
        "timeout_count": int(acc["timeout_count"]),
        "reset_count": int(acc["reset_count"]),
        "fall_timeout_overlap_count": fall_count + int(acc["timeout_count"]) - int(acc["reset_count"]),
        "trials_started": trials,
        "fall_trial_rate": fall_rate,
        "survival_rate": None if fall_rate is None else 1.0 - fall_rate,
    }


def _load_manifest(path: Path) -> tuple[dict[str, Any], str, str]:
    raw_bytes = path.read_bytes()
    raw = raw_bytes.decode("utf-8")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("protocol manifest root must be an object")
    return manifest, canonical_sha256(manifest), hashlib.sha256(raw_bytes).hexdigest()


def _variant(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [variant for variant in manifest.get("variants", []) if variant.get("name", variant.get("id")) == name]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one variant named {name!r}")
    return matches[0]


def _update_accumulator(acc: dict[str, float | int], values: dict[str, Any], mask: Any) -> None:
    count = int(mask.sum().item())
    acc["sample_count"] += count
    if count == 0:
        return
    acc["lin_vel_error_sq_sum"] += float(values["lin_error_sq"][mask].sum().item())
    acc["yaw_rate_error_sq_sum"] += float(values["yaw_error_sq"][mask].sum().item())
    acc["torque_l2_sum"] += float(values["torque_l2"][mask].sum().item())
    acc["absolute_mechanical_power_sum"] += float(values["power"][mask].sum().item())
    acc["action_rate_l2_sum"] += float(values["action_rate_l2"][mask].sum().item())
    acc["feet_air_time_sum"] += float(values["feet_air_time_raw"][mask].sum().item())
    first = values["first_contact"][mask]
    last_air = values["last_air_time"][mask]
    acc["first_contact_air_time_sum"] += float(last_air[first].sum().item())
    acc["first_contact_count"] += int(first.sum().item())


def _record_terminations(acc: dict[str, float | int], fall: Any, timeout: Any, mask: Any) -> None:
    fall_mask = fall & mask
    timeout_mask = timeout & mask
    reset_mask = (fall | timeout) & mask
    acc["fall_count"] += int(fall_mask.sum().item())
    acc["timeout_count"] += int(timeout_mask.sum().item())
    acc["reset_count"] += int(reset_mask.sum().item())


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate(args: argparse.Namespace, simulation_app: Any) -> dict[str, Any]:
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    manifest_path = args.protocol.resolve()
    checkpoint_path = args.checkpoint.resolve()
    manifest, config_sha, config_file_sha = _load_manifest(manifest_path)
    protocol = manifest.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("evaluation_protocol is missing")
    protocol_sha = canonical_sha256(protocol)
    variant = _variant(manifest, args.variant)
    variant_config_sha = canonical_sha256(variant)
    declared_variant_sha = manifest.get("variant_sha256", {}).get(args.variant)
    if declared_variant_sha is not None and declared_variant_sha != variant_config_sha:
        raise ValueError(f"manifest variant SHA mismatch for {args.variant}")
    grid_spec = protocol.get("command_grid")
    if not isinstance(grid_spec, dict):
        raise ValueError("evaluation_protocol.command_grid is required")
    grid = command_grid(grid_spec)
    expected_conditions = int(protocol.get("command_grid_conditions", len(grid)))
    if expected_conditions != len(grid):
        raise ValueError(f"command grid count mismatch: manifest={expected_conditions}, generated={len(grid)}")
    protocol_num_envs = int(protocol["num_envs"])
    per_condition = int(protocol["environments_per_condition"])
    if int(grid_spec.get("environments_per_condition", -1)) != per_condition:
        raise ValueError("command_grid environments_per_condition mismatch")
    horizon = int(protocol["horizon_steps"])
    smoke_only = args.smoke_steps is not None
    if not smoke_only and args.num_envs != protocol_num_envs:
        raise ValueError(f"production evaluation requires num_envs={protocol_num_envs}")
    if not smoke_only and args.num_envs != len(grid) * per_condition:
        raise ValueError("num_envs must equal command conditions times environments_per_condition")
    if not smoke_only and args.eval_seed != int(protocol["seed"]):
        raise ValueError(f"production evaluation requires eval_seed={protocol['seed']}")
    if smoke_only:
        horizon = int(args.smoke_steps)
        if horizon <= 0:
            raise ValueError("smoke_steps must be positive")

    task = str(protocol["task"])
    device = getattr(args, "device", "cuda:0")
    env_cfg = parse_env_cfg(task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.eval_seed
    env_cfg.events.add_base_mass = None
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
    runtime_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    if not smoke_only and not math.isclose(runtime_step_dt, float(protocol["step_dt"]), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"runtime step_dt={runtime_step_dt} differs from protocol step_dt={protocol['step_dt']}")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.eval_seed
    agent_cfg.device = device
    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    command_buffer = env.unwrapped.command_manager.get_command("base_velocity")
    robot = env.unwrapped.scene["robot"]
    contact = env.unwrapped.scene.sensors["contact_forces"]
    foot_ids = contact.find_bodies(".*_foot")[0]
    if len(foot_ids) != 4:
        raise RuntimeError(f"expected four Go2 feet, found {len(foot_ids)}")
    condition_index = torch.arange(args.num_envs, device=env.unwrapped.device) % len(grid)
    fixed_commands = torch.tensor(
        [[grid[index]["vx_mps"], grid[index]["vy_mps"], grid[index]["yaw_rate_radps"]] for index in condition_index.cpu().tolist()],
        dtype=torch.float32,
        device=env.unwrapped.device,
    )
    overall = _new_accumulator(args.num_envs)
    by_condition = {
        command["id"]: _new_accumulator(int((condition_index == index).sum().item()))
        for index, command in enumerate(grid)
    }
    active = torch.ones(args.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    feet_air_time_threshold = float(env.unwrapped.reward_manager.get_term_cfg("feet_air_time").params["threshold"])
    obs, _ = env.get_observations()

    try:
        for step in range(horizon):
            with torch.inference_mode():
                # The command manager may resample during env.step; overwrite it before
                # recomputing observations so the policy always sees the fixed grid.
                command_buffer.copy_(fixed_commands)
                obs, _ = env.get_observations()
                actions = policy(obs)
                effective_actions = actions
                if agent_cfg.clip_actions is not None:
                    effective_actions = torch.clamp(actions, -agent_cfg.clip_actions, agent_cfg.clip_actions)
                action_delta = effective_actions - env.unwrapped.action_manager.action
                torque = robot.data.applied_torque
                joint_vel = robot.data.joint_vel
                root_lin = robot.data.root_lin_vel_b[:, :2]
                root_yaw = robot.data.root_ang_vel_b[:, 2]
                last_air_time = contact.data.last_air_time[:, foot_ids]
                first_contact = contact.compute_first_contact(env.unwrapped.step_dt)[:, foot_ids]
                feet_air_time_raw = torch.sum(
                    (last_air_time - feet_air_time_threshold) * first_contact, dim=1
                ) * (torch.linalg.norm(fixed_commands[:, :2], dim=1) > 0.1)
                values = {
                    "lin_error_sq": torch.sum(torch.square(root_lin - fixed_commands[:, :2]), dim=1),
                    "yaw_error_sq": torch.square(root_yaw - fixed_commands[:, 2]),
                    "torque_l2": torch.sum(torch.square(torque), dim=1),
                    "power": torch.sum(torch.abs(torque * joint_vel), dim=1),
                    "action_rate_l2": torch.sum(torch.square(action_delta), dim=1),
                    "feet_air_time_raw": feet_air_time_raw,
                    "last_air_time": last_air_time,
                    "first_contact": first_contact,
                }
                _update_accumulator(overall, values, active)
                for index, command in enumerate(grid):
                    _update_accumulator(by_condition[command["id"]], values, active & (condition_index == index))

                obs, _, _, _ = env.step(actions)
                # These term buffers retain the just-computed event even though env.step
                # has already reset the corresponding simulation state.
                fall = env.unwrapped.termination_manager.get_term("base_contact").clone()
                timeout = env.unwrapped.termination_manager.get_term("time_out").clone()
                _record_terminations(overall, fall, timeout, active)
                for index, command in enumerate(grid):
                    _record_terminations(by_condition[command["id"]], fall, timeout, active & (condition_index == index))
                active &= ~(fall | timeout)
    finally:
        env.close()

    by_command = []
    for command in grid:
        by_command.append({"command": command, **finalize_accumulator(by_condition[command["id"]])})
    checkpoint_sha = file_sha256(checkpoint_path)
    return {
        "schema_version": 1,
        "variant": args.variant,
        "training_seed": args.training_seed,
        "evaluation_seed": args.eval_seed,
        "config_sha256": config_sha,
        "config_file_sha256": config_file_sha,
        "variant_config_sha256": variant_config_sha,
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint": {"reference": portable_path(checkpoint_path), "sha256": checkpoint_sha},
        "protocol_compliant": not smoke_only,
        "experimental_use": "tensor_smoke_only" if smoke_only else "g005_reward_ablation_evaluation",
        "task": task,
        "num_envs": args.num_envs,
        "horizon_steps": horizon,
        "step_dt": runtime_step_dt,
        "effective_weights": variant["weights"],
        "denominators": {
            "sample_count": "active pre-step environment states",
            "fall_trial_rate": "base_contact events / trials_started; timeout overlap is retained separately",
            "survival_rate": "1 - fall_trial_rate; timeouts are not classified as falls",
            "trials_started": "fixed initial environments; each environment contributes only its first episode",
            "fall_timeout_overlap_count": "fall_count + timeout_count - reset_count; overlap is counted in both event columns but once in reset_count",
        },
        "warnings": [
            "absolute_mechanical_power_w is a simulation proxy: mean(sum(abs(applied_torque * joint_velocity))).",
            "Reward weights and TensorBoard reward terms are not used as raw physical metrics.",
        ],
        "metrics": {"overall": finalize_accumulator(overall), "by_command": by_command},
    }


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--eval-seed", required=True, type=int)
    parser.add_argument("--num-envs", required=True, type=int)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke-steps", type=int, default=None, help="Non-protocol tensor-path smoke horizon.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> int:
    from isaaclab.app import AppLauncher

    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    started_at = time.time()
    gpu_before = _gpu_snapshot()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = evaluate(args, simulation_app)
        report["runtime_evidence"] = {
            "exit_code": 0,
            "app_close_completed": False,
            "finalized_after_process_exit": False,
            "started_at_epoch": started_at,
            "evaluation_pid": os.getpid(),
            "fatal_scan": _fatal_scan(args.checkpoint.resolve(), started_at),
            "gpu_before": gpu_before,
            "gpu_after": None,
            "gpu_recovered_to_baseline": None,
            "process_recovered": None,
        }
        _write_json_atomic(args.output.resolve(), report)
        print(
            json.dumps({"output": str(args.output.resolve()), "protocol_compliant": report["protocol_compliant"]}),
            flush=True,
        )
    finally:
        simulation_app.close()
    return 0


def finalize_runtime_main() -> int:
    parser = argparse.ArgumentParser(description="Finalize runtime evidence after the Isaac Sim process exits.")
    parser.add_argument("--finalize-runtime", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    report = json.loads(output.read_text(encoding="utf-8"))
    runtime = report.get("runtime_evidence")
    if not isinstance(runtime, dict) or runtime.get("exit_code") != 0:
        raise ValueError("successful preliminary runtime_evidence is required")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("reference"):
        raise ValueError("checkpoint reference is required")
    gpu_before = runtime.get("gpu_before") or {}
    gpu_after = _gpu_snapshot()
    before_memory = gpu_before.get("memory_used_mib")
    after_memory = gpu_after.get("memory_used_mib")
    process_recovered = int(runtime["evaluation_pid"]) not in (gpu_after.get("compute_pids") or [])
    recovered = (
        gpu_before.get("measurement_complete") is True
        and gpu_after["measurement_complete"]
        and before_memory is not None
        and after_memory is not None
        and len(before_memory) == len(after_memory)
        and all(after <= before + 64 for before, after in zip(before_memory, after_memory))
        and process_recovered
    )
    checkpoint_path = resolve_portable_path(checkpoint["reference"])
    runtime.update({
        "app_close_completed": True,
        "finalized_after_process_exit": True,
        "fatal_scan": _fatal_scan(checkpoint_path, float(runtime["started_at_epoch"])),
        "gpu_after": gpu_after,
        "gpu_recovered_to_baseline": recovered,
        "process_recovered": process_recovered,
    })
    _write_json_atomic(output, report)
    print(json.dumps({"output": str(output), "runtime_finalized": True}), flush=True)
    return 0


if __name__ == "__main__":
    if "--finalize-runtime" in sys.argv:
        raise SystemExit(finalize_runtime_main())
    raise SystemExit(main())
