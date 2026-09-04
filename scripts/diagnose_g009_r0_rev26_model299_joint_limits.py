#!/usr/bin/env python3
"""Attribute joint-limit behavior from the rejected rev26 R0 model_299 checkpoint.

This is a diagnostic-only replay.  It cannot qualify the checkpoint, estimate
historical training-event attribution, or stand in for the held-out evaluator.
"""

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
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for search_root in (SCRIPTS_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import evaluate_g009_r0 as qualification  # noqa: E402


SCHEMA_VERSION = "g009.r0.rev27.model299_limit_diagnostic.v1"
REPORT_ID = "g009_r0_rejected_checkpoint_hard_limit_diagnostic"
PREREG_PATH = REPO_ROOT / "configs" / "g009_r0_rev27_model299_limit_diagnostic.json"
POSE_NAMES = ("prone", "supine", "left_side", "right_side")
DEFAULT_TASK = qualification.DEFAULT_TASK
EXPECTED_CHECKPOINT_NAME = qualification.EXPECTED_TRAINING_CHECKPOINT
EXPECTED_ACTOR_DIM = qualification.EXPECTED_ACTOR_OBSERVATION_DIM
EXPECTED_CRITIC_DIM = qualification.EXPECTED_CRITIC_OBSERVATION_DIM
EXPECTED_SOURCE_PATHS = qualification.QUALIFICATION_SOURCE_PATHS
EXPECTED_TRAINING_CHECKS = {
    "process_exit_zero": True,
    "no_traceback_or_error": True,
    "requested_iteration_reached": True,
    "log_directory_exists": True,
    "tensorboard_exists": True,
    "checkpoint_exists": True,
    "gpu_measurement_complete": True,
    "gpu_recovered_to_baseline": True,
    "qualification_training_safety_zero": False,
    "qualification_gpu_safety": True,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_preregistration(path: Path) -> dict[str, Any]:
    value = qualification._read_json(path)
    _require(
        value.get("schema_version")
        == "g009.r0.rev27.model299_limit_diagnostic_preregistration.v1"
        and value.get("evidence_id") == "G009-5-E020"
        and value.get("revision") == "rev27",
        "rev27 diagnostic preregistration identity mismatch",
    )
    runtime = value.get("runtime")
    checkpoint = value.get("checkpoint")
    training_report = value.get("training_report")
    instrumentation = value.get("instrumentation")
    claim_limits = value.get("claim_limits")
    for name, item in (
        ("runtime", runtime),
        ("checkpoint", checkpoint),
        ("training_report", training_report),
        ("instrumentation", instrumentation),
        ("claim_limits", claim_limits),
    ):
        _require(isinstance(item, Mapping), f"rev27 {name} contract must be an object")
    assert isinstance(runtime, Mapping)
    assert isinstance(checkpoint, Mapping)
    assert isinstance(training_report, Mapping)
    assert isinstance(instrumentation, Mapping)
    assert isinstance(claim_limits, Mapping)
    expected_runtime = {
        "task": DEFAULT_TASK,
        "seed": 42,
        "forbidden_held_out_seed": 1042,
        "num_envs": 1024,
        "environments_per_pose": 256,
        "horizon_steps": 400,
        "action_mode": "stochastic",
        "device": "cuda:0",
        "headless": True,
        "policy_updates": 0,
        "optimizer_updates": 0,
    }
    for key, expected in expected_runtime.items():
        _require(runtime.get(key) == expected, f"rev27 runtime contract mismatch: {key}")
    _require(tuple(runtime.get("poses", ())) == POSE_NAMES, "rev27 pose order mismatch")
    _require(
        training_report.get("sha256")
        == "71a3b45129b79f2beaed14fab486423aafdd0f92e860022cf22c2c6242391234"
        and training_report.get("passed") is False
        and training_report.get("qualification_passed") is False
        and training_report.get("qualification_passed_scope") == "training_safety_aggregate"
        and training_report.get("held_out_qualification_status") == "not_run"
        and training_report.get("only_failed_required_check")
        == "qualification_training_safety_zero",
        "rev27 rejected training report binding mismatch",
    )
    _require(
        checkpoint.get("name") == EXPECTED_CHECKPOINT_NAME
        and checkpoint.get("sha256")
        == "75b38ac4c8f8b2ed17d73893350c1ca484ca3ef3c0d633273553e13efdf44c95"
        and checkpoint.get("iteration") == 299
        and checkpoint.get("actor_observation_dimension") == EXPECTED_ACTOR_DIM
        and checkpoint.get("critic_observation_dimension") == EXPECTED_CRITIC_DIM,
        "rev27 checkpoint binding mismatch",
    )
    _require(
        instrumentation.get("boundary") == "RecorderManager.record_pre_reset instance wrapper"
        and instrumentation.get("preserve_original_hard_termination_callable") is True
        and instrumentation.get("active_recorder_terms") == 0
        and instrumentation.get("active_first_episode_only") is True
        and instrumentation.get("exclude_auto_reset_samples") is True
        and instrumentation.get("sample_initial_reset_state_before_first_policy_step") is True
        and instrumentation.get("rng_neutral") is True,
        "rev27 instrumentation contract mismatch",
    )
    _require(
        instrumentation.get("checkpoint_sha256_recheck_immediately_before_runner_load") is True
        and instrumentation.get("checkpoint_sha256_stable_after_rollout") is True,
        "rev27 checkpoint TOCTOU contract mismatch",
    )
    _require(
        instrumentation.get("action_fields")
        == {
            "sampled_policy_action": "policy sample before RslRlVecEnvWrapper clamp",
            "clipped_normalized_action": "ActionManager raw_actions after wrapper clamp",
            "processed_target_rad": "post-action-term joint target in radians",
        },
        "rev27 action-field semantics mismatch",
    )
    _require(
        all(
            claim_limits.get(key) is False
            for key in (
                "qualification_eligible",
                "official_evaluation",
                "success_rate_metrics",
                "historical_training_event_attribution",
                "completion_is_safety_pass",
            )
        ),
        "rev27 claim limits must all fail closed",
    )
    paths = value.get("diagnostic_source_binding_paths")
    _require(
        isinstance(paths, list)
        and paths == sorted(set(paths))
        and paths
        == [
            "configs/g009_r0_rev27_model299_limit_diagnostic.json",
            "scripts/diagnose_g009_r0_rev26_model299_joint_limits.py",
            "tests/test_g009_r0_rev27_model299_limit_diagnostic.py",
        ],
        "rev27 diagnostic source binding paths mismatch",
    )
    manifest = hashlib.sha256(
        json.dumps(paths, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _require(
        value.get("diagnostic_source_binding_path_manifest_sha256") == manifest
        == "35ef5f9201560da6e294039a6d88c60b39653e6ccb6d9e4418a89e69cc496505",
        "rev27 diagnostic source binding path manifest mismatch",
    )
    _require(
        tuple(value.get("outcomes", ()))
        == ("hard_limit_reproduced", "soft_only_reproduced", "not_reproduced"),
        "rev27 diagnostic outcomes mismatch",
    )
    return value


PREREGISTRATION = _load_preregistration(PREREG_PATH)


def validate_rejected_training_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the pure-data portion of the one-failure rev26 contract."""
    checks = report.get("success_checks")
    _require(isinstance(checks, Mapping), "training success_checks must be an object")
    assert isinstance(checks, Mapping)
    mismatched_checks = [
        name for name, expected in EXPECTED_TRAINING_CHECKS.items() if checks.get(name) is not expected
    ]
    _require(
        not mismatched_checks,
        "only qualification_training_safety_zero may fail: " + ", ".join(mismatched_checks),
    )
    extra_false = [
        str(name)
        for name, value in checks.items()
        if name not in EXPECTED_TRAINING_CHECKS and value is False
    ]
    _require(not extra_false, "unexpected false success checks: " + ", ".join(extra_false))

    training_safety = report.get("training_safety_aggregate")
    _require(isinstance(training_safety, Mapping), "training_safety_aggregate must be an object")
    assert isinstance(training_safety, Mapping)
    hard = training_safety.get("hard_joint_limit")
    numeric = training_safety.get("numeric_invalid")
    _require(isinstance(hard, Mapping), "hard_joint_limit summary must be an object")
    _require(isinstance(numeric, Mapping), "numeric_invalid summary must be an object")
    assert isinstance(hard, Mapping) and isinstance(numeric, Mapping)
    hard_maximum = hard.get("maximum")
    hard_nonzero = hard.get("nonzero_sample_count")
    numeric_maximum = numeric.get("maximum")
    _require(
        isinstance(hard_maximum, (int, float))
        and math.isfinite(float(hard_maximum))
        and float(hard_maximum) > 0.0,
        "rejected report must contain a positive hard-joint-limit maximum",
    )
    _require(
        isinstance(hard_nonzero, int) and not isinstance(hard_nonzero, bool) and hard_nonzero > 0,
        "rejected report must contain positive hard-joint-limit nonzero samples",
    )
    _require(
        isinstance(numeric_maximum, (int, float))
        and math.isfinite(float(numeric_maximum))
        and float(numeric_maximum) == 0.0,
        "rejected report numeric-invalid maximum must be zero",
    )
    assert isinstance(hard_maximum, (int, float))
    assert isinstance(hard_nonzero, int)
    assert isinstance(numeric_maximum, (int, float))
    _require(training_safety.get("qualification_passed") is False, "training safety must reject qualification")

    qualification_mode = report.get("qualification_mode")
    _require(isinstance(qualification_mode, Mapping), "qualification_mode must be an object")
    assert isinstance(qualification_mode, Mapping)
    expected_identity = {
        "task": DEFAULT_TASK,
        "seed": 42,
        "num_envs": 1024,
        "max_iterations": 300,
        "headless": True,
        "last_iteration": 299,
        "iteration_target": 300,
        "passed": False,
        "run_health_passed": False,
    }
    for name, expected in expected_identity.items():
        _require(report.get(name) == expected, f"rejected rev26 report mismatch: {name}")
    _require(report.get("qualification_passed") is None, "held-out qualification must remain not run")
    _require(report.get("effective_hydra_overrides") == [], "training must have no Hydra overrides")
    resume = report.get("resume")
    _require(isinstance(resume, Mapping) and resume.get("enabled") is False, "training must be scratch")
    _require(
        qualification_mode.get("enabled") is True
        and qualification_mode.get("preflight_passed") is True
        and qualification_mode.get("policy_qualification_status") == "not_run",
        "qualification preflight/not-run binding mismatch",
    )
    return {
        "hard_joint_limit_maximum": float(hard_maximum),
        "hard_joint_limit_nonzero_sample_count": hard_nonzero,
        "numeric_invalid_maximum": float(numeric_maximum),
        "only_failed_required_check": "qualification_training_safety_zero",
        "training_qualification_passed": False,
        "held_out_qualification_status": "not_run",
    }


def _validate_checkpoint(report: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    import torch

    artifacts = report.get("artifacts")
    _require(isinstance(artifacts, Mapping), "training artifacts must be an object")
    assert isinstance(artifacts, Mapping)
    artifact_path = artifacts.get("checkpoint")
    _require(isinstance(artifact_path, str), "training checkpoint path is missing")
    assert isinstance(artifact_path, str)
    expected_path = qualification.resolve_portable_path(artifact_path)
    _require(checkpoint.resolve() == expected_path, "checkpoint path does not match rejected report")
    _require(checkpoint.name == EXPECTED_CHECKPOINT_NAME, "checkpoint name must be model_299.pt")
    checkpoint_sha256 = qualification.file_sha256(checkpoint)
    _require(artifacts.get("checkpoint_sha256") == checkpoint_sha256, "checkpoint SHA-256 mismatch")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping), "checkpoint root must be an object")
    state = payload.get("model_state_dict")
    _require(isinstance(state, Mapping), "checkpoint model_state_dict is missing")
    assert isinstance(state, Mapping)
    actor_shape = tuple(getattr(state.get("actor.0.weight"), "shape", ()))
    critic_shape = tuple(getattr(state.get("critic.0.weight"), "shape", ()))
    _require(actor_shape == (512, EXPECTED_ACTOR_DIM), f"actor observation dimension mismatch: {actor_shape}")
    _require(critic_shape == (512, EXPECTED_CRITIC_DIM), f"critic observation dimension mismatch: {critic_shape}")
    _require(payload.get("iter") == 299, "checkpoint iteration must be 299")
    _require(
        all(
            bool(torch.isfinite(value).all().item())
            for value in state.values()
            if isinstance(value, torch.Tensor)
        ),
        "checkpoint model tensors must be finite",
    )
    return {
        "path": qualification.portable_path(checkpoint),
        "sha256": checkpoint_sha256,
        "name": checkpoint.name,
        "iteration": 299,
        "actor_observation_dimension": EXPECTED_ACTOR_DIM,
        "critic_observation_dimension": EXPECTED_CRITIC_DIM,
        "finite_model_tensors": True,
    }


def validate_rejected_training_binding(report_path: Path, checkpoint: Path) -> dict[str, Any]:
    prereg_report = PREREGISTRATION["training_report"]
    prereg_checkpoint = PREREGISTRATION["checkpoint"]
    expected_report_path = (REPO_ROOT / prereg_report["path"]).resolve()
    _require(report_path.resolve() == expected_report_path, "training report path differs from rev27 preregistration")
    _require(
        qualification.file_sha256(report_path) == prereg_report["sha256"],
        "training report SHA-256 differs from rev27 preregistration",
    )
    report = qualification._read_json(report_path)
    rejection = validate_rejected_training_contract(report)
    checkpoint_binding = _validate_checkpoint(report, checkpoint)
    _require(
        checkpoint_binding["sha256"] == prereg_checkpoint["sha256"],
        "checkpoint SHA-256 differs from rev27 preregistration",
    )

    entrypoint = report.get("training_entrypoint")
    _require(isinstance(entrypoint, Mapping), "training entrypoint binding is missing")
    assert isinstance(entrypoint, Mapping)
    entrypoint_path = entrypoint.get("path")
    _require(isinstance(entrypoint_path, str), "training entrypoint path is missing")
    assert isinstance(entrypoint_path, str)
    expected_entrypoint = qualification.TRAINING_ENTRYPOINT_PATH.resolve()
    _require(
        qualification.resolve_portable_path(entrypoint_path) == expected_entrypoint
        and entrypoint.get("sha256") == qualification.file_sha256(expected_entrypoint)
        and entrypoint.get("repository_internal") is True,
        "training entrypoint binding mismatch",
    )

    upstream = qualification._validate_upstream_binding(report.get("upstream", {}))
    qualification_contract = report.get("qualification_contract")
    _require(isinstance(qualification_contract, Mapping), "qualification contract binding is missing")
    assert isinstance(qualification_contract, Mapping)
    contract_path = qualification_contract.get("path")
    _require(isinstance(contract_path, str), "qualification contract path is missing")
    assert isinstance(contract_path, str)
    _require(
        qualification.resolve_portable_path(contract_path) == qualification.QUALIFICATION_CONFIG_PATH.resolve()
        and qualification_contract.get("sha256")
        == qualification.file_sha256(qualification.QUALIFICATION_CONFIG_PATH)
        and qualification_contract.get("source_binding_path_manifest_sha256")
        == qualification.EXPECTED_QUALIFICATION_SOURCE_MANIFEST_SHA256,
        "qualification contract binding mismatch",
    )

    repository = report.get("repository")
    _require(isinstance(repository, Mapping), "training repository binding is missing")
    assert isinstance(repository, Mapping)
    training_commit = repository.get("commit")
    _require(
        isinstance(training_commit, str) and len(training_commit) == 40 and repository.get("dirty") is False,
        "training repository commit/clean binding mismatch",
    )
    source_bundle = qualification.validate_source_bundle(report.get("source_bundle", {}))
    _require(tuple(source_bundle["files"]) == EXPECTED_SOURCE_PATHS, "source bundle path contract mismatch")
    _require(
        report.get("source_bundle", {}).get("matches_repository_commit") is True,
        "training source bundle was not commit-bound",
    )
    current = qualification.git_source_state()
    _require(current["clean"] is True, "diagnostic repository is dirty outside reports/runs")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *PREREGISTRATION["diagnostic_source_binding_paths"]],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    _require(tracked.returncode == 0, "diagnostic source binding files must be committed and tracked")
    diagnostic_files = {
        relative: qualification.file_sha256(REPO_ROOT / relative)
        for relative in PREREGISTRATION["diagnostic_source_binding_paths"]
    }
    diagnostic_bundle = {
        "sha256": qualification.source_bundle_sha256(diagnostic_files),
        "files": diagnostic_files,
    }
    return {
        "training_report": {
            "path": qualification.portable_path(report_path),
            "sha256": qualification.file_sha256(report_path),
        },
        "rejection": rejection,
        "checkpoint": checkpoint_binding,
        "training_repository": {"commit": training_commit, "clean": True},
        "diagnostic_repository": {"commit": current["commit"], "clean": True},
        "commit_difference_allowed": training_commit != current["commit"],
        "commit_difference_basis": "exact current source bundle equality",
        "source_bundle": source_bundle,
        "diagnostic_source_bundle": diagnostic_bundle,
        "diagnostic_preregistration": {
            "path": qualification.portable_path(PREREG_PATH),
            "sha256": qualification.file_sha256(PREREG_PATH),
            "evidence_id": PREREGISTRATION["evidence_id"],
            "source_binding_path_manifest_sha256": PREREGISTRATION[
                "diagnostic_source_binding_path_manifest_sha256"
            ],
        },
        "training_entrypoint": {
            "path": "scripts/bootstrap_train_g009.py",
            "sha256": qualification.file_sha256(expected_entrypoint),
        },
        "upstream": upstream,
        "qualification_contract": {
            "path": "configs/g009_r0_rev26_qualification.json",
            "sha256": qualification.file_sha256(qualification.QUALIFICATION_CONFIG_PATH),
            "source_binding_path_manifest_sha256": qualification.EXPECTED_QUALIFICATION_SOURCE_MANIFEST_SHA256,
        },
    }


def aggregate_attribution_rows(
    rows: Sequence[Mapping[str, Any]], joint_order: Sequence[str], event_cap: int
) -> dict[str, Any]:
    """Aggregate uncapped rows while returning a deterministic capped sample."""
    if event_cap < 0:
        raise ValueError("event_cap must be non-negative")
    if tuple(joint_order) != tuple(dict.fromkeys(joint_order)) or not joint_order:
        raise ValueError("joint_order must be non-empty and unique")
    joint_rank = {name: index for index, name in enumerate(joint_order)}
    counts = {pose: {joint: 0 for joint in joint_order} for pose in POSE_NAMES}
    maxima = {pose: {joint: 0.0 for joint in joint_order} for pose in POSE_NAMES}
    episode_keys: set[tuple[str, int]] = set()
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        pose = raw.get("pose_id")
        joint = raw.get("joint_name")
        if pose not in POSE_NAMES or joint not in joint_rank:
            raise ValueError("attribution row pose/joint is outside the bound order")
        env_index = raw.get("env_index")
        excess = raw.get("threshold_excess_rad")
        if not isinstance(env_index, int) or isinstance(env_index, bool) or env_index < 0:
            raise ValueError("attribution env_index must be a non-negative integer")
        if not isinstance(excess, (int, float)) or not math.isfinite(float(excess)) or excess <= 0.0:
            raise ValueError("threshold excess must be positive and finite")
        counts[str(pose)][str(joint)] += 1
        maxima[str(pose)][str(joint)] = max(maxima[str(pose)][str(joint)], float(excess))
        episode_keys.add((str(pose), env_index))
        normalized.append(dict(raw))
    normalized.sort(
        key=lambda row: (
            int(row["rollout_step"]),
            int(row["env_index"]),
            joint_rank[str(row["joint_name"])],
        )
    )
    sample = normalized[:event_cap]
    return {
        "joint_order": list(joint_order),
        "pose_joint_event_counts": counts,
        "pose_joint_max_threshold_excess_rad": maxima,
        "hard_limit_episode_count": len(episode_keys),
        "hard_limit_joint_event_count": len(normalized),
        "event_sample_cap": event_cap,
        "event_sample_count": len(sample),
        "event_sample_dropped_count": len(normalized) - len(sample),
        "event_samples": sample,
    }


def _tensor_bytes(tensor: Any) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _hash_state(value: Any, digest: Any) -> None:
    if hasattr(value, "detach") and hasattr(value, "dtype"):
        digest.update(b"tensor")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=repr):
            digest.update(repr(key).encode("utf-8"))
            _hash_state(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _hash_state(item, digest)
    else:
        digest.update(repr(value).encode("utf-8"))


def state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_state(value, digest)
    return digest.hexdigest()


def callable_name(value: Any) -> str:
    return f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"


def load_prevalidated_checkpoint(
    runner: Any, checkpoint: Path, training_binding: Mapping[str, Any]
) -> str:
    """Recheck the pre-App checkpoint binding immediately before runner.load()."""
    checkpoint_binding = training_binding.get("checkpoint")
    _require(isinstance(checkpoint_binding, Mapping), "pre-App checkpoint binding is missing")
    assert isinstance(checkpoint_binding, Mapping)
    checkpoint_sha256 = qualification.file_sha256(checkpoint)
    _require(
        checkpoint_sha256 == checkpoint_binding.get("sha256"),
        "checkpoint changed after CPU binding and before runner.load",
    )
    runner.load(str(checkpoint.resolve()))
    return checkpoint_sha256


class PreResetLimitObserver:
    """Read terminal state before auto-reset without changing termination logic."""

    def __init__(self, env: Any, *, event_cap: int) -> None:
        self.env = env
        self.event_cap = event_cap
        self.active_before_step = None
        self.sampled_policy_actions = None
        self.rollout_step = 0
        self.rows: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.rng_neutral = True
        action_term = env.action_manager.get_term("joint_pos")
        raw_joint_ids = action_term._joint_ids
        if isinstance(raw_joint_ids, slice):
            self.joint_ids = list(range(len(env.scene["robot"].joint_names)))[raw_joint_ids]
        else:
            self.joint_ids = list(raw_joint_ids)
        self.joint_order = tuple(action_term._joint_names)
        if not self.joint_order or len(self.joint_ids) != len(self.joint_order):
            raise RuntimeError("resolved action joint ids/names are inconsistent")
        self._soft_counts = None
        self._soft_maxima = None
        self._soft_seen_env = None
        self.soft_state_steps: list[int] = []

    def _record_soft_state(self, env_indices: Any, active_mask: Any) -> None:
        import torch

        robot = self.env.scene["robot"]
        position = robot.data.joint_pos[:, self.joint_ids]
        limits = robot.data.soft_joint_pos_limits[:, self.joint_ids]
        excess = torch.maximum(
            (limits[..., 0] - position).clamp_min(0.0),
            (position - limits[..., 1]).clamp_min(0.0),
        )
        pose_ids = self.env._g009_recover_fall_class
        eligible = torch.zeros(self.env.num_envs, dtype=torch.bool, device=excess.device)
        eligible[env_indices] = True
        eligible &= active_mask.to(device=eligible.device, dtype=torch.bool)
        mask = (excess > 0.0) & eligible.unsqueeze(-1)
        joint_count = len(self.joint_order)
        flat_indices = (
            pose_ids.to(dtype=torch.long).unsqueeze(-1) * joint_count
            + torch.arange(joint_count, device=excess.device).unsqueeze(0)
        ).reshape(-1)
        flat_mask = mask.reshape(-1)
        counts = torch.zeros(len(POSE_NAMES) * joint_count, dtype=torch.long, device=excess.device)
        counts.scatter_add_(0, flat_indices, flat_mask.to(dtype=torch.long))
        maxima = torch.zeros(len(POSE_NAMES) * joint_count, dtype=excess.dtype, device=excess.device)
        maxima.scatter_reduce_(
            0,
            flat_indices,
            torch.where(flat_mask, excess.reshape(-1), torch.zeros_like(excess).reshape(-1)),
            reduce="amax",
            include_self=True,
        )
        counts = counts.reshape(len(POSE_NAMES), joint_count)
        maxima = maxima.reshape(len(POSE_NAMES), joint_count)
        if self._soft_counts is None:
            self._soft_counts = counts
            self._soft_maxima = maxima
            self._soft_seen_env = mask.any(dim=1)
        else:
            assert self._soft_maxima is not None and self._soft_seen_env is not None
            self._soft_counts += counts
            self._soft_maxima = torch.maximum(self._soft_maxima, maxima)
            self._soft_seen_env |= mask.any(dim=1)

    def set_step_context(self, *, active: Any, actions: Any, rollout_step: int) -> None:
        self.active_before_step = active.detach().clone()
        self.sampled_policy_actions = actions.detach().clone()
        self.rollout_step = rollout_step

    def sample_active_soft_limits(self, *, active: Any, state_step: int) -> None:
        import torch

        if state_step < 0:
            raise ValueError("soft-limit state_step must be non-negative")
        active_snapshot = active.detach().clone().to(dtype=torch.bool)
        active_ids = torch.nonzero(active_snapshot, as_tuple=False).flatten()
        self._record_soft_state(active_ids, active_snapshot)
        self.soft_state_steps.append(state_step)

    def capture(self, env_ids: Any) -> None:
        from isaac_walk_g009.recover_contracts import SOLVER_JOINT_LIMIT_TOLERANCE_RAD

        if (
            self.active_before_step is None
            or self.sampled_policy_actions is None
            or self.rollout_step <= 0
        ):
            return
        hard = self.env.termination_manager.get_term("hard_joint_limit")
        robot = self.env.scene["robot"]
        action_term = self.env.action_manager.get_term("joint_pos")
        pose_ids = self.env._g009_recover_fall_class
        positions = robot.data.joint_pos[:, self.joint_ids]
        limits = robot.data.joint_pos_limits[:, self.joint_ids]
        self._record_soft_state(env_ids, self.active_before_step)
        for raw_env_index in env_ids:
            env_index = int(raw_env_index)
            if not bool(self.active_before_step[env_index].item()) or not bool(hard[env_index].item()):
                continue
            pose = POSE_NAMES[int(pose_ids[env_index].item())]
            found = False
            for joint_local_index, joint_name in enumerate(self.joint_order):
                position = float(positions[env_index, joint_local_index].item())
                lower = float(limits[env_index, joint_local_index, 0].item())
                upper = float(limits[env_index, joint_local_index, 1].item())
                raw_excess = max(lower - position, position - upper, 0.0)
                threshold_excess = max(
                    lower - SOLVER_JOINT_LIMIT_TOLERANCE_RAD - position,
                    position - upper - SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
                    0.0,
                )
                if threshold_excess <= 0.0:
                    continue
                found = True
                self.rows.append(
                    {
                        "pose_id": pose,
                        "env_index": env_index,
                        "rollout_step": self.rollout_step,
                        "episode_step": int(self.env.episode_length_buf[env_index].item()),
                        "joint_name": joint_name,
                        "joint_index": self.joint_ids[joint_local_index],
                        "joint_position_rad": position,
                        "hard_lower_rad": lower,
                        "hard_upper_rad": upper,
                        "raw_excess_rad": raw_excess,
                        "margin_rad": float(SOLVER_JOINT_LIMIT_TOLERANCE_RAD),
                        "threshold_excess_rad": threshold_excess,
                        "root_position_w": robot.data.root_pos_w[env_index].detach().cpu().tolist(),
                        "root_quaternion_wxyz": robot.data.root_quat_w[env_index].detach().cpu().tolist(),
                        "root_linear_velocity_w": robot.data.root_lin_vel_w[env_index].detach().cpu().tolist(),
                        "root_angular_velocity_w": robot.data.root_ang_vel_w[env_index].detach().cpu().tolist(),
                        "sampled_policy_action": float(
                            self.sampled_policy_actions[env_index, joint_local_index].item()
                        ),
                        "sampled_policy_action_stage": "before RslRlVecEnvWrapper clamp",
                        "clipped_normalized_action": float(
                            action_term.raw_actions[env_index, joint_local_index].item()
                        ),
                        "clipped_normalized_action_stage": "ActionManager raw_actions after wrapper clamp",
                        "processed_target_rad": float(
                            action_term.processed_actions[env_index, joint_local_index].item()
                        ),
                    }
                )
            if not found:
                self.errors.append(
                    f"hard termination without recomputed threshold crossing: step={self.rollout_step} env={env_index}"
                )

    def aggregate(self) -> dict[str, Any]:
        hard = aggregate_attribution_rows(self.rows, self.joint_order, self.event_cap)
        if self._soft_counts is None or self._soft_maxima is None or self._soft_seen_env is None:
            raise RuntimeError("soft-limit instrumentation was never sampled")
        counts = self._soft_counts.detach().cpu().tolist()
        maxima = self._soft_maxima.detach().cpu().tolist()
        hard["soft_joint_limit"] = {
            "pose_joint_sample_counts": {
                pose: {joint: int(counts[pose_index][joint_index]) for joint_index, joint in enumerate(self.joint_order)}
                for pose_index, pose in enumerate(POSE_NAMES)
            },
            "pose_joint_max_excess_rad": {
                pose: {joint: float(maxima[pose_index][joint_index]) for joint_index, joint in enumerate(self.joint_order)}
                for pose_index, pose in enumerate(POSE_NAMES)
            },
            "episode_count": int(self._soft_seen_env.sum().item()),
            "sampling": "active first-episode pre-policy and terminal pre-reset states",
            "sampled_state_steps": list(self.soft_state_steps),
            "initial_reset_state_sampled": bool(self.soft_state_steps and self.soft_state_steps[0] == 0),
        }
        return hard


def install_pre_reset_observer(recorder_manager: Any, observer: PreResetLimitObserver) -> Any:
    """Wrap the reset boundary and prove that observation consumes no RNG."""
    if len(recorder_manager.active_terms) != 0:
        raise RuntimeError("diagnostic requires zero active recorder terms")
    original = recorder_manager.record_pre_reset

    def observed_pre_reset(env_ids: Any, force_export_or_skip: Any = None) -> Any:
        import torch

        cpu_before = torch.get_rng_state().clone()
        cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
        observer.capture(env_ids)
        cpu_after = torch.get_rng_state()
        cuda_after = torch.cuda.get_rng_state_all()
        observer.rng_neutral &= bool(torch.equal(cpu_before, cpu_after))
        observer.rng_neutral &= len(cuda_before) == len(cuda_after) and all(
            bool(torch.equal(before, after)) for before, after in zip(cuda_before, cuda_after)
        )
        return original(env_ids, force_export_or_skip)

    recorder_manager.record_pre_reset = observed_pre_reset
    if len(recorder_manager.active_terms) != 0:
        raise RuntimeError("pre-reset observer changed active recorder terms")
    return original


def validate_diagnostic_report_claims(report: Mapping[str, Any]) -> None:
    required = {
        "status": "diagnostic_complete",
        "protocol_mode": "diagnostic_only",
        "qualification_eligible": False,
        "historical_training_event_attribution": False,
        "completion_is_safety_pass": False,
    }
    for key, expected in required.items():
        _require(report.get(key) == expected, f"diagnostic claim invariant mismatch: {key}")
    _require(
        report.get("result")
        in {"hard_limit_reproduced", "soft_only_reproduced", "not_reproduced"},
        "diagnostic result is invalid",
    )
    claims = report.get("claim_limits")
    _require(isinstance(claims, Mapping), "diagnostic claim_limits must be an object")
    assert isinstance(claims, Mapping)
    _require(
        claims.get("official_evaluation") is False
        and claims.get("success_rate_estimate") is False
        and claims.get("checkpoint_qualification") is False,
        "diagnostic claim limits must deny qualification interpretations",
    )
    _require(claims.get("completion_is_safety_pass") is False, "diagnostic completion is not a safety pass")


def evaluate(args: argparse.Namespace, training_binding: Mapping[str, Any]) -> dict[str, Any]:
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

    source_before = qualification.git_source_state()
    _require(source_before["clean"] is True, "diagnostic source tree is dirty outside reports/runs")
    _require(
        source_before["commit"] == training_binding["diagnostic_repository"]["commit"],
        "repository commit changed after CPU binding and before simulation",
    )
    register_tasks()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.events.reset_base.params.update(
        {"assignment_mode": "stratified", "pose_xy_range": (0.0, 0.0), "yaw_range": (0.0, 0.0)}
    )
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device
    raw_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    observer = PreResetLimitObserver(env.unwrapped, event_cap=args.event_sample_cap)
    recorder_manager = env.unwrapped.recorder_manager
    recorder_terms_before = list(recorder_manager.active_terms)
    hard_callable_before = env.unwrapped.cfg.terminations.hard_joint_limit.func
    original_pre_reset = install_pre_reset_observer(recorder_manager, observer)
    hard_callable_after_install = env.unwrapped.cfg.terminations.hard_joint_limit.func
    try:
        checkpoint_sha256_before = load_prevalidated_checkpoint(
            runner, args.checkpoint, training_binding
        )
        observations, _ = env.get_observations()
        class_ids = env.unwrapped._g009_recover_fall_class.detach().clone()
        expected_ids = torch.arange(args.num_envs, device=class_ids.device) % len(POSE_NAMES)
        if not torch.equal(class_ids, expected_ids):
            raise RuntimeError("stratified pose assignment readback mismatch")
        active = torch.ones(args.num_envs, dtype=torch.bool, device=env.unwrapped.device)
        policy_before = state_sha256(runner.alg.policy.state_dict())
        optimizer_before = state_sha256(runner.alg.optimizer.state_dict())
        executed_steps = 0
        for rollout_step in range(1, args.horizon_steps + 1):
            observer.sample_active_soft_limits(active=active, state_step=rollout_step - 1)
            with torch.inference_mode():
                actions = runner.alg.policy.act(observations)
            observer.set_step_context(
                active=active,
                actions=actions,
                rollout_step=rollout_step,
            )
            observations, _, dones, _ = env.step(actions)
            active &= ~dones.bool()
            executed_steps = rollout_step
            if not bool(active.any().item()):
                break
        observer.sample_active_soft_limits(active=active, state_step=executed_steps)
        recorder_manager.record_pre_reset = original_pre_reset
        pre_reset_original_restored = recorder_manager.record_pre_reset is original_pre_reset
        if not pre_reset_original_restored:
            raise RuntimeError("RecorderManager.record_pre_reset wrapper did not restore original callable")
        policy_after = state_sha256(runner.alg.policy.state_dict())
        optimizer_after = state_sha256(runner.alg.optimizer.state_dict())
        if policy_before != policy_after or optimizer_before != optimizer_after:
            raise RuntimeError("diagnostic replay mutated policy or optimizer state")
        if observer.errors:
            raise RuntimeError("pre-reset attribution failed: " + "; ".join(observer.errors))
        if not observer.rng_neutral:
            raise RuntimeError("pre-reset attribution observer changed torch RNG state")
        if hard_callable_before is not hard_callable_after_install:
            raise RuntimeError("pre-reset observer changed the hard termination callable")
        if recorder_terms_before != [] or list(recorder_manager.active_terms) != []:
            raise RuntimeError("pre-reset observer changed recorder active terms")
        aggregate = observer.aggregate()
        hard_count = int(aggregate["hard_limit_episode_count"])
        soft_count = int(aggregate["soft_joint_limit"]["episode_count"])
        if hard_count > 0:
            result = "hard_limit_reproduced"
        elif soft_count > 0:
            result = "soft_only_reproduced"
        else:
            result = "not_reproduced"
        checkpoint_sha256_after = qualification.file_sha256(args.checkpoint)
        if checkpoint_sha256_before != checkpoint_sha256_after:
            raise RuntimeError("checkpoint changed during diagnostic replay")
        source_after = qualification.git_source_state()
        if source_before != source_after or source_after.get("clean") is not True:
            raise RuntimeError("repository commit/source state changed during diagnostic")
        source_bundle_after = qualification.validate_source_bundle(training_binding["source_bundle"])
        if source_bundle_after != training_binding["source_bundle"]:
            raise RuntimeError("bound training source bundle changed during diagnostic")
        diagnostic_files_after = {
            relative: qualification.file_sha256(REPO_ROOT / relative)
            for relative in PREREGISTRATION["diagnostic_source_binding_paths"]
        }
        diagnostic_bundle_after = {
            "sha256": qualification.source_bundle_sha256(diagnostic_files_after),
            "files": diagnostic_files_after,
        }
        if diagnostic_bundle_after != training_binding["diagnostic_source_bundle"]:
            raise RuntimeError("diagnostic source bundle changed during replay")
        report = {
            "schema_version": SCHEMA_VERSION,
            "report_id": REPORT_ID,
            "status": "diagnostic_complete",
            "protocol_mode": "diagnostic_only",
            "qualification_eligible": False,
            "historical_training_event_attribution": False,
            "completion_is_safety_pass": False,
            "result": result,
            "task": args.task,
            "seed": args.seed,
            "device": args.device,
            "headless": bool(args.headless),
            "num_envs": args.num_envs,
            "environments_per_pose": args.num_envs // len(POSE_NAMES),
            "pose_assignment": "stratified_equal_four_pose",
            "poses": list(POSE_NAMES),
            "horizon_steps": args.horizon_steps,
            "executed_steps": executed_steps,
            "action_mode": args.action_mode,
            "policy_updates": 0,
            "optimizer_updates": 0,
            "active_first_episode_only": True,
            "auto_reset_samples_excluded": True,
            "termination_callable_semantics": "original hard-limit predicate unchanged",
            "attribution": aggregate,
            "training_binding": training_binding,
            "source_state": {
                "pre_app_binding": training_binding["diagnostic_repository"],
                "before_rollout": source_before,
                "after_rollout": source_after,
            },
            "no_update_evidence": {
                "policy_before_sha256": policy_before,
                "policy_after_sha256": policy_after,
                "policy_unchanged": True,
                "optimizer_before_sha256": optimizer_before,
                "optimizer_after_sha256": optimizer_after,
                "optimizer_unchanged": True,
                "checkpoint_before_sha256": checkpoint_sha256_before,
                "checkpoint_after_sha256": checkpoint_sha256_after,
                "checkpoint_unchanged": True,
                "observer_rng_neutral": True,
            },
            "instrumentation_invariants": {
                "boundary": "RecorderManager.record_pre_reset instance wrapper",
                "hard_termination_callable_before": callable_name(hard_callable_before),
                "hard_termination_callable_after_install": callable_name(hard_callable_after_install),
                "hard_termination_callable_identity_preserved": hard_callable_before is hard_callable_after_install,
                "active_recorder_terms_before": recorder_terms_before,
                "active_recorder_terms_after_install": list(recorder_manager.active_terms),
                "active_recorder_term_count_preserved_at_zero": (
                    recorder_terms_before == [] and list(recorder_manager.active_terms) == []
                ),
                "pre_reset_original_restored": pre_reset_original_restored,
            },
            "claim_limits": {
                "official_evaluation": False,
                "success_rate_estimate": False,
                "checkpoint_qualification": False,
                "historical_training_event_attribution": False,
                "completion_is_safety_pass": False,
                "statement": (
                    "This replay only attributes whether the rejected checkpoint reproduces "
                    "hard-limit events under this diagnostic seed and rollout."
                ),
            },
            "instrumentation_source": {
                "path": "scripts/diagnose_g009_r0_rev26_model299_joint_limits.py",
                "sha256": qualification.file_sha256(Path(__file__)),
            },
        }
        validate_diagnostic_report_claims(report)
        return report
    finally:
        recorder_manager.record_pre_reset = original_pre_reset
        env.close()


def _core_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--horizon-steps", type=int, default=400)
    parser.add_argument("--action-mode", choices=("stochastic",), default="stochastic")
    parser.add_argument("--event-sample-cap", type=int, default=512)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hard-exit-after-report", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.training_report.is_file():
        raise FileNotFoundError(args.training_report)
    if args.task != DEFAULT_TASK:
        raise ValueError("diagnostic task is fixed to the rev26 Matrix task")
    if args.seed == 1042:
        raise ValueError("held-out qualification seed 1042 is forbidden for this diagnostic")
    if args.seed != PREREGISTRATION["runtime"]["seed"]:
        raise ValueError("diagnostic seed is fixed to preregistered seed 42")
    if args.num_envs != PREREGISTRATION["runtime"]["num_envs"]:
        raise ValueError("diagnostic num_envs is fixed to 1024")
    if args.horizon_steps != PREREGISTRATION["runtime"]["horizon_steps"]:
        raise ValueError("diagnostic horizon_steps is fixed to 400")
    if args.action_mode != "stochastic":
        raise ValueError("diagnostic action mode is fixed to stochastic")
    if args.device != "cuda:0":
        raise ValueError("diagnostic device is fixed to cuda:0")
    if args.event_sample_cap != PREREGISTRATION["instrumentation"]["event_sample_cap"]:
        raise ValueError("diagnostic event_sample_cap is fixed to 512")
    if args.headless is not True:
        raise ValueError("diagnostic must run headless")
    if args.output.exists():
        raise FileExistsError(args.output)
    output = args.output.resolve()
    reports_root = (REPO_ROOT / "reports" / "runs").resolve()
    if output.parent != reports_root or output.suffix.lower() != ".json" or output.name.startswith("."):
        raise ValueError("output must be a visible JSON direct child of reports/runs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    parser = _core_parser()
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, device="cuda:0")
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def main(argv: list[str] | None = None) -> int:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    args = parse_args(argv)
    training_binding = validate_rejected_training_binding(
        args.training_report.resolve(), args.checkpoint.resolve()
    )
    started_at = time.time()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        report = evaluate(args, training_binding)
        report["wall_time_seconds"] = round(time.time() - started_at, 3)
        qualification._write_json_atomic(args.output.resolve(), report)
        print(
            json.dumps(
                {"output": str(args.output.resolve()), "status": report["status"], "result": report["result"]}
            ),
            flush=True,
        )
        if args.hard_exit_after_report:
            sys.stdout.flush()
            os._exit(0)
        return 0
    finally:
        simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
