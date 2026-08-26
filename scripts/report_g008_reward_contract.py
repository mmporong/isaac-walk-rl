#!/usr/bin/env python3
"""Resolve the effective G008 reward and PPO contract from the pinned Isaac Lab configuration."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path.home() / "IsaacLab"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

EXPECTED_WEIGHTS = {
    "track_lin_vel_xy_exp": 1.5,
    "track_ang_vel_z_exp": 0.75,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,
    "dof_torques_l2": -0.0002,
    "dof_acc_l2": -2.5e-7,
    "action_rate_l2": -0.01,
    "feet_air_time": 0.01,
    "flat_orientation_l2": 0.0,
    "dof_pos_limits": 0.0,
}

RAW_FORMULAS = {
    "track_lin_vel_xy_exp": "exp(-||v_cmd_xy - v_base_xy||^2 / std^2), std=0.5 m/s",
    "track_ang_vel_z_exp": "exp(-(w_cmd_z - w_base_z)^2 / std^2), std=0.5 rad/s",
    "lin_vel_z_l2": "v_base_z^2",
    "ang_vel_xy_l2": "w_base_x^2 + w_base_y^2",
    "dof_torques_l2": "sum_j(tau_j^2)",
    "dof_acc_l2": "sum_j(qddot_j^2)",
    "action_rate_l2": "sum_j((a_t_j - a_(t-1)_j)^2)",
    "feet_air_time": (
        "sum_feet((last_air_time - 0.5 s) * first_contact) * "
        "I(||v_cmd_xy|| > 0.1 m/s)"
    ),
    "flat_orientation_l2": "projected_gravity_body_x^2 + projected_gravity_body_y^2",
    "dof_pos_limits": "sum_j(distance outside soft joint position limits)",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    for root, token in ((REPO_ROOT, "%REPO_ROOT%"), (ISAACLAB_ROOT, "%ISAACLAB_ROOT%"), (Path.home(), "%USERPROFILE%")):
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        suffix = str(relative)
        return token if not suffix else token + "\\" + suffix
    return str(resolved)


def callable_name(value: Any) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if callable(value):
        return callable_name(value)
    attributes = {}
    for name in ("name", "joint_names", "body_names", "preserve_order"):
        if hasattr(value, name):
            attributes[name] = jsonable(getattr(value, name))
    if attributes:
        return attributes
    raise TypeError(f"unsupported report value: {type(value)!r}")


def source_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": portable_path(path),
        "sha256": file_sha256(path),
    }


def _raw_formula(name: str, function: Any) -> str:
    if name == "feet_air_time" and callable_name(function).endswith("feet_air_time_turn_aware"):
        return (
            "sum_feet((last_air_time - 0.5 s) * first_contact) * "
            "I(||v_cmd_xy|| > 0.1 m/s OR |w_cmd_z| > 0.1 rad/s)"
        )
    return RAW_FORMULAS[name]


def reward_terms(env_cfg: Any) -> list[dict[str, Any]]:
    from isaaclab.managers import RewardTermCfg

    terms = []
    for name in EXPECTED_WEIGHTS:
        term = getattr(env_cfg.rewards, name)
        if not isinstance(term, RewardTermCfg):
            raise TypeError(f"reward term is not RewardTermCfg: {name}={term!r}")
        weight = float(term.weight)
        if weight != EXPECTED_WEIGHTS[name]:
            raise RuntimeError(f"reward weight drift: {name} expected={EXPECTED_WEIGHTS[name]} actual={weight}")
        function_path = Path(inspect.getsourcefile(term.func) or "")
        terms.append(
            {
                "name": name,
                "active": weight != 0.0,
                "weight": weight,
                "function": callable_name(term.func),
                "params": jsonable(term.params),
                "raw_formula": _raw_formula(name, term.func),
                "weighted_step_formula": f"step_dt * ({weight}) * raw_term",
                "function_source": source_metadata(function_path),
            }
        )
    if getattr(env_cfg.rewards, "undesired_contacts") is not None:
        raise RuntimeError("Go2 undesired_contacts must remain disabled for this contract")
    return terms


def termination_terms(env_cfg: Any) -> list[dict[str, Any]]:
    terms = []
    for name in ("time_out", "base_contact"):
        term = getattr(env_cfg.terminations, name)
        terms.append(
            {
                "name": name,
                "function": callable_name(term.func),
                "time_out": bool(term.time_out),
                "params": jsonable(term.params),
            }
        )
    return terms


def ppo_contract(agent_cfg: Any) -> dict[str, Any]:
    return {
        "num_steps_per_env": int(agent_cfg.num_steps_per_env),
        "actor_hidden_dims": list(agent_cfg.policy.actor_hidden_dims),
        "critic_hidden_dims": list(agent_cfg.policy.critic_hidden_dims),
        "activation": agent_cfg.policy.activation,
        "init_noise_std": float(agent_cfg.policy.init_noise_std),
        "value_loss_coef": float(agent_cfg.algorithm.value_loss_coef),
        "use_clipped_value_loss": bool(agent_cfg.algorithm.use_clipped_value_loss),
        "clip_param": float(agent_cfg.algorithm.clip_param),
        "entropy_coef": float(agent_cfg.algorithm.entropy_coef),
        "num_learning_epochs": int(agent_cfg.algorithm.num_learning_epochs),
        "num_mini_batches": int(agent_cfg.algorithm.num_mini_batches),
        "initial_learning_rate": float(agent_cfg.algorithm.learning_rate),
        "schedule": agent_cfg.algorithm.schedule,
        "gamma": float(agent_cfg.algorithm.gamma),
        "lam": float(agent_cfg.algorithm.lam),
        "desired_kl": float(agent_cfg.algorithm.desired_kl),
        "max_grad_norm": float(agent_cfg.algorithm.max_grad_norm),
        "empirical_normalization": bool(agent_cfg.empirical_normalization),
    }


def contract_for_task(task: str, device: str) -> dict[str, Any]:
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    terms = reward_terms(env_cfg)
    feet_air_time = next(item for item in terms if item["name"] == "feet_air_time")
    pure_yaw_air_time_active = feet_air_time["function"].endswith("feet_air_time_turn_aware")
    terminations = termination_terms(env_cfg)
    ppo = ppo_contract(agent_cfg)
    return {
        "task": task,
        "physics_dt_s": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.decimation),
        "step_dt_s": step_dt,
        "episode_length_s": float(env_cfg.episode_length_s),
        "episode_length_control_steps": int(round(env_cfg.episode_length_s / step_dt)),
        "reward_aggregation": "sum_i(step_dt * weight_i * raw_term_i)",
        "reward_terms": terms,
        "disabled_reward_terms": [item["name"] for item in terms if not item["active"]]
        + ["undesired_contacts"],
        "terminations": terminations,
        "ppo": ppo,
        "turn_specific_readback": {
            "pure_yaw_command": [0.0, 0.0, 0.5],
            "feet_air_time_active_for_pure_yaw": pure_yaw_air_time_active,
            "reason": (
                "project variant gates on ||v_cmd_xy|| > 0.1 m/s OR |w_cmd_z| > 0.1 rad/s"
                if pure_yaw_air_time_active
                else "Isaac Lab baseline gates on ||v_cmd_xy|| > 0.1 m/s and ignores yaw command magnitude"
            ),
            "explicit_roll_pitch_angle_penalty_active": False,
            "explicit_contact_conditioned_foot_slip_penalty_active": False,
        },
    }


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "Isaac-G008-Velocity-Rough-Go2-CommandSuite-v0",
            "Isaac-G008-Velocity-IrregularRoad-Go2-G0-v0",
            "Isaac-G008-Velocity-IrregularRoad-Go2-S1-v0",
        ],
    )
    parser.add_argument(
        "--variant-tasks",
        nargs="+",
        default=["Isaac-G008-Velocity-IrregularRoad-Go2-G0-TurnAir-v0"],
    )
    parser.add_argument("--output", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    import isaaclab
    import isaaclab_tasks  # noqa: F401
    import rsl_rl
    from isaac_walk_g008 import register_tasks

    register_tasks()
    contracts = [contract_for_task(task, args.device) for task in args.tasks]
    variant_contracts = [contract_for_task(task, args.device) for task in args.variant_tasks]
    reference = contracts[0]
    shared_keys = (
        "physics_dt_s",
        "decimation",
        "step_dt_s",
        "episode_length_s",
        "episode_length_control_steps",
        "reward_aggregation",
        "reward_terms",
        "disabled_reward_terms",
        "terminations",
        "ppo",
        "turn_specific_readback",
    )
    if any(any(contract[key] != reference[key] for key in shared_keys) for contract in contracts[1:]):
        raise RuntimeError("G008 task reward/PPO contracts diverged")
    invariant_keys = tuple(key for key in shared_keys if key not in ("reward_terms", "turn_specific_readback"))
    for variant in variant_contracts:
        if any(variant[key] != reference[key] for key in invariant_keys):
            raise RuntimeError(f"reward variant changed a non-reward contract: {variant['task']}")
        changed_terms = [
            (baseline_term, variant_term)
            for baseline_term, variant_term in zip(reference["reward_terms"], variant["reward_terms"])
            if baseline_term != variant_term
        ]
        if len(changed_terms) != 1 or changed_terms[0][0]["name"] != "feet_air_time":
            raise RuntimeError(f"reward variant must change only feet_air_time: {variant['task']}")

    lab_commit = subprocess.check_output(
        ["git", "-C", str(ISAACLAB_ROOT), "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()
    from isaaclab.managers import RewardManager

    reward_manager_path = Path(inspect.getsourcefile(RewardManager) or "")
    return {
        "schema_version": 2,
        "goal": "G008",
        "status": "complete",
        "purpose": "runtime-resolved reward, termination, and PPO contract",
        "tasks": list(args.tasks),
        "all_task_contracts_identical": True,
        "versions": {
            "isaac_lab_commit": lab_commit,
            "isaaclab_package": getattr(isaaclab, "__version__", None),
            "rsl_rl": getattr(rsl_rl, "__version__", "2.3.3"),
        },
        "contract": {key: reference[key] for key in shared_keys},
        "reward_variants": [
            {
                "task": variant["task"],
                "changed_term": next(
                    term
                    for term in variant["reward_terms"]
                    if term != next(base for base in reference["reward_terms"] if base["name"] == term["name"])
                ),
                "turn_specific_readback": variant["turn_specific_readback"],
                "all_non_reward_contract_fields_identical": True,
            }
            for variant in variant_contracts
        ],
        "source_files": {
            "reward_manager": source_metadata(reward_manager_path),
            "report_script": source_metadata(Path(__file__)),
            "project_reward_variant": source_metadata(REPO_ROOT / "src" / "isaac_walk_g008" / "rewards.py"),
        },
        "interpretation_limits": [
            "This report fixes the objective actually configured for the listed tasks; it does not claim that the reward is optimal.",
            "A zero-weight configured term is reported as disabled because RewardManager skips its computation.",
            "Base contact ends an episode but has no separate scalar termination penalty in this task.",
        ],
    }


def main() -> int:
    from isaaclab.app import AppLauncher

    args = parse_args()
    app = AppLauncher(args).app
    try:
        report = build_report(args)
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)
        print(json.dumps({"output": str(output), "terms": len(report["contract"]["reward_terms"])}))
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
