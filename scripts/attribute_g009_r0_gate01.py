#!/usr/bin/env python3
"""Attribute rev11 gate01 hard-joint-limit terminations before reset.

The diagnostic reproduces only the first stochastic 24-step PPO rollout.  It
observes Isaac Lab's ``RecorderManager.record_pre_reset`` lifecycle boundary,
which runs after termination computation and before ``_reset_idx`` mutates the
state.  The manager remains term-free so the observation-corruption RNG stream
keeps the same observation-call structure as training; the observer also
verifies that it does not consume Torch RNG.  The historical rollout did not
retain an action trace, so bitwise identity with that past event is not claimed.
An attributed event is evidence about the failure, not evidence that recovery
training or the failure itself has been fixed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
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

DEFAULT_TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
EXPECTED_SEED = 42
EXPECTED_NUM_ENVS = 1024
EXPECTED_ROLLOUT_STEPS = 24
EXPECTED_DEVICE = "cuda:0"
EXPECTED_ISAACLAB_COMMIT = "90b79bb2d44feb8d833f260f2bf37da3487180ba"
EXPECTED_ISAACLAB_TAG = "v2.1.1"
EXPECTED_RSL_RL_VERSION = "2.3.3"
EXPECTED_OFFICIAL_RUNTIME_SOURCE_SHA256 = {
    "isaaclab_manager_based_rl_env": "d699f2d0467e851cdfb16b5c8e7aab3eb0d00ba7f1f13ba54d52f5e459743485",
    "isaaclab_recorder_manager": "06d1d419e4415a742d9ad714fe7406d115c7ddab2da35b724d63b05f8aed167a",
    "isaaclab_rsl_rl_vecenv_wrapper": "df54debea2947b1905a0b25db80eb5344d0cf27b136d98378c09af8a87e34869",
    "rsl_rl_on_policy_runner": "5ae9e87aabd62ecc9cf1ffdced0866875e0613f396af69877daaf517f7d2e33d",
    "rsl_rl_ppo": "efc1acf07745562d610c17ed28297e188c68101f3c38c68a0b029f68f5467cdc",
    "rsl_rl_actor_critic": "51ce8c6504a7434f86e5188ccb11e9fc981e46f764c00e32be3e8febe7112ae2",
    "rsl_rl_rollout_storage": "456450bbba03f9f5818f9ec43d2f3bd5d14d2d27f4bf2308838dd580f32ffcf0",
    "isaaclab_upstream_train": "8b995f75ac57ce7403973ff1f3f2715fbff9563ef2cdcdc321a7edc5dd15f5df",
    "isaaclab_ema_action": "fa560606fcb53d796dfc4bd4c45b0379381af5828bd58433a17cadc5d7d979be",
    "isaaclab_seed_source": "236f808e7792dcdba31ec37dd1c9ce18be14a9531a84a4648d34341d8ce8cdec",
    "isaacsim_torch_set_seed": "686be6e545b36cf4c51a9b6e7f9b05f177b6c93af7fdc3f7d710c7a24fde75e6",
}
ISAACLAB_TRACKED_RUNTIME_PATHS = (
    "scripts/reinforcement_learning/rsl_rl/train.py",
    "source/isaaclab/isaaclab/envs/manager_based_env.py",
    "source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
    "source/isaaclab/isaaclab/envs/mdp/actions/joint_actions_to_limits.py",
    "source/isaaclab/isaaclab/managers/recorder_manager.py",
    "source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py",
)
POSE_NAMES = ("prone", "supine", "left_side", "right_side")
ACTION_MODE = "stochastic_ppo_train"
PROTOCOL_ID = "g009_r0_rev11_gate01_pre_reset_attribution_v1"
SOURCE_BINDING_PATHS = (
    "configs/g009_r0.json",
    "scripts/attribute_g009_r0_gate01.py",
    "scripts/bootstrap_train_g009.py",
    "scripts/run_training.ps1",
    "src/isaac_walk_g009/agent_cfg.py",
    "src/isaac_walk_g009/mdp/__init__.py",
    "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/mdp/recover.py",
    "src/isaac_walk_g009/recover_contracts.py",
    "src/isaac_walk_g009/recover_env_cfg.py",
    "src/isaac_walk_g009/registry.py",
)
TRAINING_CORE_PATHS = tuple(path for path in SOURCE_BINDING_PATHS if path != "scripts/attribute_g009_r0_gate01.py")


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
    reports_root = (REPO_ROOT / "reports" / "runs").resolve()
    resolved = output.expanduser().resolve()
    if resolved.parent != reports_root:
        raise ValueError("output must be a direct child of the canonical reports/runs directory")
    if resolved.suffix != ".json" or resolved.name == ".json":
        raise ValueError("output must use a non-empty .json filename")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {resolved}")
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite existing temporary report: {temporary}")
    return resolved, resolved.relative_to(REPO_ROOT.resolve()).as_posix()


def prepare_execution(output: Path) -> tuple[Path, dict[str, Any]]:
    resolved, relative = canonical_report_output(output)
    return resolved, {
        "execution_id": uuid.uuid4().hex,
        "started_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "output_path_repo_relative": relative,
        "no_overwrite": True,
    }


def parse_prelaunch_output(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True, type=Path)
    args, _ = parser.parse_known_args(argv)
    return args.output


def source_bundle_provenance() -> dict[str, Any]:
    files: dict[str, str] = {}
    missing: list[str] = []
    for relative_path in SOURCE_BINDING_PATHS:
        path = REPO_ROOT / relative_path
        if path.is_file():
            files[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            missing.append(relative_path)
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_BINDING_PATHS],
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
        "source_bundle_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest() if files else None,
        "all_files_present": not missing and len(files) == len(SOURCE_BINDING_PATHS),
        "missing_files": missing,
        "clean": not status,
        "dirty_source_paths": status,
    }


def _expand_user_path(value: str) -> Path:
    expanded = value.replace("%USERPROFILE%", str(Path.home()))
    return Path(expanded).expanduser().resolve()


def _tensor_bytes(tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def torch_rng_state_sha256(torch_module) -> str:
    digest = hashlib.sha256()
    digest.update(_tensor_bytes(torch_module.get_rng_state()))
    if torch_module.cuda.is_available():
        for state in torch_module.cuda.get_rng_state_all():
            digest.update(_tensor_bytes(state))
    return digest.hexdigest()


def policy_state_sha256(policy) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def official_runtime_source_provenance() -> dict[str, str]:
    paths = {
        "isaaclab_manager_based_rl_env": Path.home()
        / "IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
        "isaaclab_recorder_manager": Path.home()
        / "IsaacLab/source/isaaclab/isaaclab/managers/recorder_manager.py",
        "isaaclab_rsl_rl_vecenv_wrapper": Path.home()
        / "IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py",
        "rsl_rl_on_policy_runner": Path.home()
        / "IsaacLab/_isaac_sim/kit/python/Lib/site-packages/rsl_rl/runners/on_policy_runner.py",
        "rsl_rl_ppo": Path.home()
        / "IsaacLab/_isaac_sim/kit/python/Lib/site-packages/rsl_rl/algorithms/ppo.py",
        "rsl_rl_actor_critic": Path.home()
        / "IsaacLab/_isaac_sim/kit/python/Lib/site-packages/rsl_rl/modules/actor_critic.py",
        "rsl_rl_rollout_storage": Path.home()
        / "IsaacLab/_isaac_sim/kit/python/Lib/site-packages/rsl_rl/storage/rollout_storage.py",
        "isaaclab_upstream_train": Path.home()
        / "IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py",
        "isaaclab_ema_action": Path.home()
        / "IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions_to_limits.py",
        "isaaclab_seed_source": Path.home()
        / "IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py",
        "isaacsim_torch_set_seed": Path.home()
        / "IsaacLab/_isaac_sim/exts/isaacsim.core.utils/isaacsim/core/utils/torch/maths.py",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"official runtime source missing: {missing}")
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def official_runtime_source_hashes_pinned(actual: dict[str, str]) -> bool:
    return actual == EXPECTED_OFFICIAL_RUNTIME_SOURCE_SHA256


def isaaclab_tracked_runtime_source_status() -> dict[str, Any]:
    isaaclab_root = Path.home() / "IsaacLab"
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *ISAACLAB_TRACKED_RUNTIME_PATHS,
        ],
        cwd=isaaclab_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "tracked_paths": list(ISAACLAB_TRACKED_RUNTIME_PATHS),
        "clean": not status,
        "dirty_paths": status,
    }


def installed_runtime_versions() -> dict[str, Any]:
    isaaclab_root = Path.home() / "IsaacLab"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=isaaclab_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tag = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=isaaclab_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "isaaclab_commit": commit,
        "isaaclab_exact_tag": tag,
        "rsl_rl_lib_version": importlib.metadata.version("rsl-rl-lib"),
    }


def expected_training_hard_limit_event_count(hard: dict[str, Any]) -> tuple[float, int]:
    hard_scalar = hard.get("maximum")
    if (
        isinstance(hard_scalar, bool)
        or not isinstance(hard_scalar, (int, float))
        or not math.isfinite(hard_scalar)
        or hard.get("sample_count") != 1
        or hard.get("nonzero_sample_count") != 1
    ):
        raise ValueError("training hard-joint-limit scalar is unavailable")
    expected_event_count_float = float(hard_scalar) * EXPECTED_ROLLOUT_STEPS
    if not math.isclose(expected_event_count_float, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("training hard-limit scalar does not imply exactly one gate01 event")
    return expected_event_count_float, 1


def validate_training_report(path: Path) -> dict[str, Any]:
    """Bind attribution to the exact clean scratch gate01 run and checkpoint."""

    reports_root = (REPO_ROOT / "reports" / "runs").resolve()
    resolved = path.expanduser().resolve()
    if resolved.parent != reports_root or resolved.suffix != ".json" or not resolved.is_file():
        raise ValueError("training report must be an existing direct-child reports/runs JSON")
    report = json.loads(resolved.read_text(encoding="utf-8"))
    run_name = report.get("run_name")
    if not isinstance(run_name, str) or not re.fullmatch(
        r"go2_flat_recover_rev11_prone_gate01_s42_\d{8}-\d{4}", run_name
    ):
        raise ValueError("training report is not the canonical rev11 prone gate01 run")
    expected = {
        "task": DEFAULT_TASK,
        "num_envs": EXPECTED_NUM_ENVS,
        "max_iterations": 1,
        "seed": EXPECTED_SEED,
        "headless": True,
        "exit_code": 0,
        "last_iteration": 0,
        "iteration_target": 1,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("training report task/budget/seed/headless/completion contract mismatch")
    if report.get("resume") != {"enabled": False, "load_run": None, "checkpoint": None}:
        raise ValueError("training report must be a scratch run without resume")
    if report.get("effective_hydra_overrides") != []:
        raise ValueError("training report must not contain Hydra overrides")
    qualification = report.get("qualification_mode", {})
    if qualification.get("enabled") is not False or qualification.get("policy_qualification_status") != "not_run":
        raise ValueError("gate01 must not be reported as qualification")
    repository = report.get("repository", {})
    training_commit = repository.get("commit")
    if repository.get("dirty") is not False or not isinstance(training_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", training_commit
    ):
        raise ValueError("training report must bind a clean valid source commit")
    source_bundle = report.get("source_bundle", {})
    training_files = source_bundle.get("files")
    if (
        not isinstance(training_files, dict)
        or set(training_files) != set(TRAINING_CORE_PATHS)
        or source_bundle.get("matches_repository_commit") is not True
    ):
        raise ValueError("training report core source bundle is incomplete")
    current_files = {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in TRAINING_CORE_PATHS
    }
    payload = "\n".join(f"{name}:{current_files[name]}" for name in sorted(current_files))
    current_bundle_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if training_files != current_files or source_bundle.get("sha256") != current_bundle_sha:
        raise ValueError("current core source hashes do not match the gate01 training bundle")
    changed_since_training = subprocess.run(
        ["git", "diff", "--name-only", f"{training_commit}..HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    core_changes = sorted(set(changed_since_training) & set(TRAINING_CORE_PATHS))
    non_core_changes = sorted(set(changed_since_training) - set(TRAINING_CORE_PATHS))
    artifacts = report.get("artifacts", {})
    checkpoint_portable = artifacts.get("checkpoint", "")
    if not isinstance(checkpoint_portable, str) or not checkpoint_portable.startswith("%USERPROFILE%\\"):
        raise ValueError("checkpoint path must retain portable %USERPROFILE% form")
    checkpoint = _expand_user_path(checkpoint_portable)
    checkpoint_sha = artifacts.get("checkpoint_sha256")
    if not checkpoint.is_file() or not isinstance(checkpoint_sha, str):
        raise ValueError("bound gate01 checkpoint is missing")
    actual_checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_checkpoint_sha != checkpoint_sha:
        raise ValueError("bound gate01 checkpoint SHA-256 mismatch")
    hard = report.get("training_safety_aggregate", {}).get("hard_joint_limit", {})
    expected_event_count_float, expected_event_count = expected_training_hard_limit_event_count(hard)
    return {
        "report_path_repo_relative": resolved.relative_to(REPO_ROOT).as_posix(),
        "report_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "run_name": run_name,
        "training_commit": training_commit,
        "core_source_bundle_sha256": current_bundle_sha,
        "changed_since_training": changed_since_training,
        "core_changes_since_training": core_changes,
        "non_core_changes_since_training": non_core_changes,
        "checkpoint_path": checkpoint_portable,
        "checkpoint_sha256": actual_checkpoint_sha,
        "checkpoint_exists_but_was_not_loaded": True,
        "hard_joint_limit_scalar": float(hard["maximum"]),
        "hard_joint_limit_scalar_times_rollout_steps": expected_event_count_float,
        "expected_termination_event_count": expected_event_count,
    }


def validate_protocol_args(args: argparse.Namespace) -> None:
    actual = (args.task, args.seed, args.num_envs, args.rollout_steps, args.headless, args.device)
    expected = (
        DEFAULT_TASK,
        EXPECTED_SEED,
        EXPECTED_NUM_ENVS,
        EXPECTED_ROLLOUT_STEPS,
        True,
        EXPECTED_DEVICE,
    )
    if actual != expected:
        raise ValueError(
            "protocol requires the canonical task, seed42, 1024 envs, 24 steps, --headless, and cuda:0"
        )


def joint_limit_attributions(
    *,
    position: list[float],
    lower: list[float],
    upper: list[float],
    joint_names: list[str],
    margin_rad: float,
) -> list[dict[str, Any]]:
    """Recompute the exact termination predicate and return every violating joint."""

    count = len(joint_names)
    if not (len(position) == len(lower) == len(upper) == count and count > 0):
        raise ValueError("joint vectors must have one non-empty, identical dimension")
    if not math.isfinite(margin_rad) or margin_rad < 0.0:
        raise ValueError("margin_rad must be finite and non-negative")
    records: list[dict[str, Any]] = []
    for joint_index, (actual, lo, hi, name) in enumerate(zip(position, lower, upper, joint_names)):
        values = (actual, lo, hi)
        if not name or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError("joint names and limits must be finite and valid")
        if lo >= hi:
            raise ValueError("joint lower limit must be smaller than upper limit")
        lower_violated = actual < lo - margin_rad
        upper_violated = actual > hi + margin_rad
        if lower_violated or upper_violated:
            side = "lower" if lower_violated else "upper"
            raw_excess = lo - actual if lower_violated else actual - hi
            margin_excess = raw_excess - margin_rad
            records.append(
                {
                    "joint_index": joint_index,
                    "joint_name": name,
                    "actual_position_rad": float(actual),
                    "lower_limit_rad": float(lo),
                    "upper_limit_rad": float(hi),
                    "violated_side": side,
                    "raw_excess_rad": float(raw_excess),
                    "margin_excess_rad": float(margin_excess),
                    "predicate_recomputed": True,
                }
            )
    return records


def validate_attribution_result(
    *,
    events: list[dict[str, Any]],
    observed_termination_keys,
    margin_rad: float,
) -> dict[str, bool]:
    """Fail closed on missing, malformed, non-finite, or inconsistent attribution."""

    observed_key_counts = Counter(observed_termination_keys)
    attributed_keys: list[tuple[int, int]] = []
    records_valid = True
    for event in events:
        try:
            key = (int(event["rollout_control_step"]), int(event["env_index"]))
            pose_id = int(event["pose_id"])
            records = event["joint_attributions"]
            expected = joint_limit_attributions(
                position=event["joint_position_rad"],
                lower=event["joint_lower_limit_rad"],
                upper=event["joint_upper_limit_rad"],
                joint_names=event["joint_names"],
                margin_rad=margin_rad,
            )
            comparable = [
                {
                    key: record[key]
                    for key in (
                        "joint_index",
                        "joint_name",
                        "actual_position_rad",
                        "lower_limit_rad",
                        "upper_limit_rad",
                        "violated_side",
                        "raw_excess_rad",
                        "margin_excess_rad",
                        "predicate_recomputed",
                    )
                }
                for record in records
            ]
            finite_vectors = all(
                all(isinstance(value, (int, float)) and math.isfinite(value) for value in event[name])
                for name in (
                    "ppo_sample_pre_wrapper_clip",
                    "action_term_raw_post_wrapper_clip",
                    "processed_ema_target_rad",
                    "joint_velocity_rad_s",
                    "applied_torque_nm",
                )
            )
            vector_dimensions_match = all(
                len(event[name]) == len(event["joint_names"])
                for name in (
                    "ppo_sample_pre_wrapper_clip",
                    "action_term_raw_post_wrapper_clip",
                    "processed_ema_target_rad",
                    "joint_velocity_rad_s",
                    "applied_torque_nm",
                )
            )
            action_clip_matches = all(
                math.isclose(
                    float(post),
                    max(-1.0, min(1.0, float(pre))),
                    rel_tol=0.0,
                    abs_tol=1.0e-7,
                )
                for pre, post in zip(
                    event["ppo_sample_pre_wrapper_clip"],
                    event["action_term_raw_post_wrapper_clip"],
                )
            )
            records_valid &= (
                key in observed_key_counts
                and 0 <= key[1] < EXPECTED_NUM_ENVS
                and 1 <= key[0] <= EXPECTED_ROLLOUT_STEPS
                and 0 <= pose_id < len(POSE_NAMES)
                and event.get("pose_name") == POSE_NAMES[pose_id]
                and event.get("action_mode") == ACTION_MODE
                and bool(records)
                and comparable == expected
                and finite_vectors
                and vector_dimensions_match
                and action_clip_matches
                and isinstance(event.get("episode_control_step"), int)
                and event["episode_control_step"] >= 1
                and isinstance(event.get("sim_step_counter"), int)
                and event["sim_step_counter"] >= 1
            )
            attributed_keys.append(key)
        except (KeyError, TypeError, ValueError):
            records_valid = False
    return {
        "hard_joint_limit_reproduced": bool(observed_key_counts),
        "termination_attribution_present": bool(observed_key_counts)
        and Counter(attributed_keys) == observed_key_counts,
        "termination_and_attribution_counts_match": len(events)
        == len(attributed_keys)
        == sum(observed_key_counts.values()),
        "attribution_records_valid_and_predicate_recomputed": records_valid
        and len(events) == len(attributed_keys)
        and Counter(attributed_keys) == observed_key_counts,
        "post_wrapper_action_matches_clamped_ppo_sample": records_valid
        and bool(attributed_keys),
    }


class Gate01PreResetObserver:
    """Capture reset-preceding tensors without activating a recorder term."""

    def __init__(self, env, collector_state: dict[str, Any]):
        self.env = env
        self.collector_state = collector_state
        self.rollout_control_step = 0

    def capture(self, env_ids) -> None:
        if self.rollout_control_step <= 0:
            return
        from isaac_walk_g009.recover_contracts import SOLVER_JOINT_LIMIT_TOLERANCE_RAD

        hard = self.env.termination_manager.get_term("hard_joint_limit")
        robot = self.env.scene["robot"]
        action_term = self.env.action_manager.get_term("joint_pos")
        joint_ids = action_term._joint_ids
        joint_names = list(action_term._joint_names)
        positions = robot.data.joint_pos[:, joint_ids]
        limits = robot.data.joint_pos_limits[:, joint_ids]
        velocities = robot.data.joint_vel[:, joint_ids]
        torques = robot.data.applied_torque[:, joint_ids]
        pose_ids = getattr(self.env, "_g009_recover_fall_class", None)
        if pose_ids is None:
            self.collector_state["errors"].append("recovery pose ids are unavailable at pre-reset")
            return
        for env_index in env_ids:
            env_index = int(env_index)
            if not bool(hard[env_index].item()):
                continue
            try:
                pose_id = int(pose_ids[env_index].item())
                position = positions[env_index].detach().cpu().tolist()
                lower = limits[env_index, :, 0].detach().cpu().tolist()
                upper = limits[env_index, :, 1].detach().cpu().tolist()
                records = joint_limit_attributions(
                    position=position,
                    lower=lower,
                    upper=upper,
                    joint_names=joint_names,
                    margin_rad=SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
                )
                event = {
                    "rollout_control_step": int(self.rollout_control_step),
                    "episode_control_step": int(self.env.episode_length_buf[env_index].item()),
                    "sim_step_counter": int(self.env._sim_step_counter),
                    "env_index": env_index,
                    "pose_id": pose_id,
                    "pose_name": POSE_NAMES[pose_id] if 0 <= pose_id < len(POSE_NAMES) else None,
                    "action_mode": ACTION_MODE,
                    "joint_names": joint_names,
                    "joint_position_rad": position,
                    "joint_lower_limit_rad": lower,
                    "joint_upper_limit_rad": upper,
                    "ppo_sample_pre_wrapper_clip": self.collector_state["current_ppo_actions"][env_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "action_term_raw_post_wrapper_clip": action_term.raw_actions[env_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "processed_ema_target_rad": action_term.processed_actions[env_index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "joint_velocity_rad_s": velocities[env_index].detach().cpu().tolist(),
                    "applied_torque_nm": torques[env_index].detach().cpu().tolist(),
                    "joint_attributions": records,
                }
                if not records:
                    self.collector_state["errors"].append(
                        f"hard termination without recomputed violation: step={self.rollout_control_step} env={env_index}"
                    )
                self.collector_state["events"].append(event)
            except (IndexError, RuntimeError, TypeError, ValueError) as error:
                self.collector_state["errors"].append(
                    f"pre-reset attribution failed: step={self.rollout_control_step} env={env_index}: {error}"
                )


def install_pre_reset_observer(recorder_manager, observer: Gate01PreResetObserver):
    """Wrap the manager boundary while preserving ``active_terms == []``."""

    if len(recorder_manager.active_terms) != 0:
        raise RuntimeError("exact-RNG attribution requires zero active recorder terms")
    original = recorder_manager.record_pre_reset

    def observed_pre_reset(env_ids, force_export_or_skip=None):
        import torch

        cpu_before = torch.get_rng_state().clone()
        cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
        observer.capture(env_ids)
        cpu_after = torch.get_rng_state()
        cuda_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        observer.collector_state["observer_rng_neutral"] &= bool(torch.equal(cpu_before, cpu_after))
        observer.collector_state["observer_rng_neutral"] &= len(cuda_before) == len(cuda_after) and all(
            bool(torch.equal(before, after)) for before, after in zip(cuda_before, cuda_after)
        )
        return original(env_ids, force_export_or_skip)

    recorder_manager.record_pre_reset = observed_pre_reset
    if len(recorder_manager.active_terms) != 0:
        raise RuntimeError("pre-reset observer changed active recorder terms")
    return original


def run(args: argparse.Namespace, execution: dict[str, Any]) -> dict[str, Any]:
    import gymnasium as gym  # pyright: ignore[reportMissingImports]
    import torch
    from rsl_rl.runners import OnPolicyRunner  # pyright: ignore[reportMissingImports]

    import isaaclab_tasks  # noqa: F401  # pyright: ignore[reportMissingImports]
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # pyright: ignore[reportMissingImports]
    from isaaclab_tasks.utils import (  # pyright: ignore[reportMissingImports]
        load_cfg_from_registry,
        parse_env_cfg,
    )
    from isaac_walk_g009 import register_tasks
    from isaac_walk_g009.recover_contracts import (
        ACTION_EMA_ALPHA,
        ACTION_SCALE,
        POSE_CURRICULUM_PROBABILITIES,
        PPO_INIT_NOISE_STD,
        SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
        canonical_sha256,
        recover_contract,
    )

    validate_protocol_args(args)
    training_binding = validate_training_report(args.training_report)
    source_bundle = source_bundle_provenance()
    collector_state: dict[str, Any] = {
        "events": [],
        "errors": [],
        "current_ppo_actions": None,
        "observer_rng_neutral": True,
    }
    register_tasks()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = agent_cfg.seed = args.seed
    env_cfg.sim.device = agent_cfg.device = args.device
    agent_cfg.max_iterations = 1
    agent_cfg.run_name = training_binding["run_name"]
    agent_cfg.resume = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    import tempfile

    diagnostic_log_dir = tempfile.TemporaryDirectory(prefix="g009_gate01_attribution_")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=diagnostic_log_dir.name, device=args.device)
    if runner.num_steps_per_env != args.rollout_steps:
        raise RuntimeError("RSL-RL runner rollout length does not match the 24-step training report")
    observer = Gate01PreResetObserver(env.unwrapped, collector_state)
    install_pre_reset_observer(env.unwrapped.recorder_manager, observer)
    observed_termination_keys: list[tuple[int, int]] = []
    all_pose_assignments_prone = True
    action_stream_samples = []
    runtime_sources = official_runtime_source_provenance()
    isaaclab_source_status = isaaclab_tracked_runtime_source_status()
    runtime_versions = installed_runtime_versions()
    rng_before_rollout = torch_rng_state_sha256(torch)
    policy_before_rollout = policy_state_sha256(runner.alg.policy)
    policy_after_rollout: str | None = None
    rng_before_update: str | None = None
    storage_step_before_update: int | None = None
    sentinel_reached = False

    class StopBeforePpoUpdate(RuntimeError):
        pass

    original_act = runner.alg.act
    original_step = env.step

    def observed_act(obs, privileged_obs):
        actions = original_act(obs, privileged_obs)
        observer.rollout_control_step += 1
        collector_state["current_ppo_actions"] = actions.detach().clone()
        action_stream_samples.append(actions.detach().clone())
        return actions

    def observed_step(actions):
        result = original_step(actions)
        hard = env.unwrapped.termination_manager.get_term("hard_joint_limit")
        for env_index in torch.nonzero(hard, as_tuple=False).flatten().cpu().tolist():
            observed_termination_keys.append((observer.rollout_control_step, int(env_index)))
        pose_ids = getattr(env.unwrapped, "_g009_recover_fall_class", None)
        nonlocal_pose[0] &= pose_ids is not None and bool((pose_ids == 0).all().item())
        return result

    def stop_before_update():
        nonlocal rng_before_update, sentinel_reached, storage_step_before_update
        sentinel_reached = True
        rng_before_update = torch_rng_state_sha256(torch)
        storage_step_before_update = int(runner.alg.storage.step)
        raise StopBeforePpoUpdate("expected stop before PPO update")

    runner.alg.act = observed_act
    env.step = observed_step
    runner.alg.update = stop_before_update
    nonlocal_pose = [all_pose_assignments_prone]
    active_recorder_terms_zero_before = len(env.unwrapped.recorder_manager.active_terms) == 0
    active_recorder_terms_zero_after = False
    checkpoint_files_before_cleanup: list[str] = []
    try:
        initial_pose_ids = getattr(env.unwrapped, "_g009_recover_fall_class", None)
        nonlocal_pose[0] &= initial_pose_ids is not None and bool(
            (initial_pose_ids == 0).all().item()
        )
        try:
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
        except StopBeforePpoUpdate:
            pass
        else:
            collector_state["errors"].append("PPO update sentinel was not reached")
        if observer.rollout_control_step != args.rollout_steps:
            collector_state["errors"].append(
                f"official rollout collected {observer.rollout_control_step} steps, expected {args.rollout_steps}"
            )
        active_recorder_terms_zero_after = len(env.unwrapped.recorder_manager.active_terms) == 0
        policy_after_rollout = policy_state_sha256(runner.alg.policy)
        checkpoint_files_before_cleanup = [
            str(path.relative_to(diagnostic_log_dir.name)).replace("\\", "/")
            for path in Path(diagnostic_log_dir.name).rglob("model_*.pt")
        ]
    finally:
        if runner.writer is not None:
            runner.writer.close()
        env.close()
        diagnostic_log_dir.cleanup()

    all_pose_assignments_prone = nonlocal_pose[0]
    action_stream_digest = hashlib.sha256()
    for sample in action_stream_samples:
        action_stream_digest.update(_tensor_bytes(sample))

    events = collector_state["events"]
    checks = validate_attribution_result(
        events=events,
        observed_termination_keys=observed_termination_keys,
        margin_rad=SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
    )
    checks.update(
        {
            "recorder_hook_errors_absent": not collector_state["errors"],
            "source_commit_valid": bool(source_bundle["git_commit_valid"]),
            "source_bundle_complete": bool(source_bundle["all_files_present"]),
            "source_bundle_clean": bool(source_bundle["clean"]),
            "prone_only_curriculum_contract": tuple(POSE_CURRICULUM_PROBABILITIES[0])
            == (1.0, 0.0, 0.0, 0.0),
            "all_runtime_pose_assignments_prone": all_pose_assignments_prone,
            "all_attributed_poses_prone": bool(events)
            and all(event.get("pose_id") == 0 and event.get("pose_name") == "prone" for event in events),
            "action_contract_unchanged": ACTION_SCALE == 0.70 and ACTION_EMA_ALPHA == 0.2,
            "ppo_noise_contract_unchanged": PPO_INIT_NOISE_STD == 0.5,
            "ppo_update_sentinel_reached": sentinel_reached,
            "no_ppo_update_executed": sentinel_reached
            and policy_before_rollout == policy_after_rollout,
            "training_report_expected_event_count_reproduced": len(observed_termination_keys)
            == training_binding["expected_termination_event_count"],
            "active_recorder_terms_remained_zero": active_recorder_terms_zero_before
            and active_recorder_terms_zero_after,
            "pre_reset_observer_rng_neutral": bool(collector_state["observer_rng_neutral"]),
            "policy_state_unchanged_before_update": policy_before_rollout == policy_after_rollout,
            "rng_state_before_update_captured": isinstance(rng_before_update, str),
            "official_runner_act_count_exact": observer.rollout_control_step == EXPECTED_ROLLOUT_STEPS,
            "rollout_storage_step_count_exact": storage_step_before_update == EXPECTED_ROLLOUT_STEPS,
            "diagnostic_checkpoint_file_absent": not checkpoint_files_before_cleanup,
            "observed_key_multiset_matches_attribution": Counter(observed_termination_keys)
            == Counter(
                (int(event["rollout_control_step"]), int(event["env_index"])) for event in events
            ),
            "installed_isaaclab_version_pinned": (
                runtime_versions["isaaclab_commit"] == EXPECTED_ISAACLAB_COMMIT
                and runtime_versions["isaaclab_exact_tag"] == EXPECTED_ISAACLAB_TAG
            ),
            "installed_rsl_rl_version_pinned": (
                runtime_versions["rsl_rl_lib_version"] == EXPECTED_RSL_RL_VERSION
            ),
            "official_runtime_source_set_complete": len(runtime_sources) == 11,
            "official_runtime_source_hashes_pinned": (
                official_runtime_source_hashes_pinned(runtime_sources)
            ),
            "isaaclab_tracked_runtime_sources_clean": bool(isaaclab_source_status["clean"]),
        }
    )
    if not observed_termination_keys:
        outcome = "not_reproduced"
    elif all(checks.values()):
        outcome = "attributed"
    else:
        outcome = "invalid"
    return {
        "schema_version": 1,
        "probe_id": "g009_r0_gate01_hard_joint_limit_attribution",
        "execution": execution,
        "protocol": {
            "id": PROTOCOL_ID,
            "task": args.task,
            "seed": args.seed,
            "headless": bool(args.headless),
            "device": args.device,
            "num_envs": args.num_envs,
            "rollout_control_steps": args.rollout_steps,
            "init_at_random_ep_len": True,
            "pose_distribution": list(POSE_CURRICULUM_PROBABILITIES[0]),
            "action_mode": ACTION_MODE,
            "ppo_update_executed": False,
            "checkpoint_loaded": False,
            "pre_reset_hook": "observer wrapper around Isaac Lab RecorderManager.record_pre_reset",
            "active_recorder_term_count": 0,
            "hard_joint_limit_margin_rad": SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
        },
        "contract": {
            "id": "g009_r0_recover_rev11",
            "sha256": canonical_sha256(recover_contract()),
            "action_scale": ACTION_SCALE,
            "action_ema_alpha": ACTION_EMA_ALPHA,
            "ppo_initial_noise_std": PPO_INIT_NOISE_STD,
        },
        "training_binding": training_binding,
        "runtime_reproduction": {
            "official_source_sha256": runtime_sources,
            "expected_official_source_sha256": EXPECTED_OFFICIAL_RUNTIME_SOURCE_SHA256,
            "isaaclab_tracked_source_status": isaaclab_source_status,
            "installed_versions": runtime_versions,
            "torch_rng_state_before_rollout_sha256": rng_before_rollout,
            "torch_rng_state_before_update_sha256": rng_before_update,
            "ppo_sample_pre_wrapper_clip_stream_sha256": action_stream_digest.hexdigest(),
            "act_count": observer.rollout_control_step,
            "rollout_storage_step_before_update": storage_step_before_update,
            "update_sentinel_reached": sentinel_reached,
            "diagnostic_checkpoint_files_before_cleanup": checkpoint_files_before_cleanup,
            "observed_termination_key_multiset": [
                {"rollout_control_step": step, "env_index": env_index, "count": count}
                for (step, env_index), count in sorted(Counter(observed_termination_keys).items())
            ],
            "policy_state_before_rollout_sha256": policy_before_rollout,
            "policy_state_before_update_sha256": policy_after_rollout,
        },
        "source_bundle": source_bundle,
        "counts": {
            "hard_joint_limit_termination_env_events": len(observed_termination_keys),
            "attributed_env_events": len(events),
            "attributed_joint_records": sum(len(event.get("joint_attributions", ())) for event in events),
        },
        "events": events,
        "hook_errors": collector_state["errors"],
        "checks": checks,
        "outcome": outcome,
        "attribution_contract_passed": outcome == "attributed",
        "safety_gate_passed": False,
        "learned_policy_qualified": False,
        "historical_event_identity_confirmed": False,
        "historical_event_identity_reason": (
            "The original gate01 report retained only an aggregate hard-limit scalar; it did not retain "
            "the action stream, environment index, joint identity, or reset-preceding state."
        ),
        "interpretation": {
            "attributed": (
                "A new event with the same seed and protocol reproduced the same aggregate count and was "
                "exactly attributed at the reset-preceding boundary. Historical event identity is not proven."
            ),
            "not_reproduced": (
                "No event occurred in this source/seed/protocol-matched fresh rollout; this is not PASS "
                "and does not resolve gate01."
            ),
            "invalid": "The event occurred but fail-closed attribution or provenance checks failed.",
            "qualification_scope": "Attribution only; this report never qualifies a learned policy or a fix.",
            "safety_semantics": "safety_gate_passed remains false because the bound gate01 run contained a hard-limit termination.",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument("--num-envs", type=int, default=EXPECTED_NUM_ENVS)
    parser.add_argument("--rollout-steps", type=int, default=EXPECTED_ROLLOUT_STEPS)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    output, execution = prepare_execution(parse_prelaunch_output())
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    args = build_parser().parse_args()
    app_launcher = AppLauncher(args)
    try:
        report = run(args, execution)
        _write_json_atomic(output, report)
        print(json.dumps({"output": str(output), "outcome": report["outcome"]}, ensure_ascii=False))
        return 0 if report["attribution_contract_passed"] else 2
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    raise SystemExit(main())
