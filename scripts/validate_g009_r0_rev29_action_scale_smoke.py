#!/usr/bin/env python3
"""Fail-closed prelaunch validation for the G009 R0 rev29 action-scale smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaac_walk_g009.recover_contracts import (  # noqa: E402
    ACTION_EMA_ALPHA,
    ACTION_SCALE,
    ARTICULATION_SOLVER_POSITION_ITERATION_COUNT,
    ARTICULATION_SOLVER_VELOCITY_ITERATION_COUNT,
    CRITIC_OBSERVATION_DIM,
    GO2_SOFT_JOINT_LIMIT_FACTOR,
    MAX_DEPENETRATION_VELOCITY_M_S,
    PPO_ENTROPY_COEF,
    PPO_INIT_NOISE_STD,
    canonical_sha256,
    recover_contract,
)


SCHEMA_VERSION = "g009.r0.rev29.action_scale_smoke_preregistration.v1"
EVIDENCE_ID = "G009-5-E022"
DEFAULT_PREREGISTRATION = REPO_ROOT / "configs" / "g009_r0_rev29_action_scale_smoke.json"
EXPECTED_SOURCE_MANIFEST_SHA256 = "cbd542ae61179ee47be8f313d083ae9dc75d27864af103693affa662d1b5892d"
EXPECTED_ISAAC_LAB_COMMIT = "90b79bb2d44feb8d833f260f2bf37da3487180ba"
EXPECTED_OFFICIAL_TRAIN_SHA256 = "8b995f75ac57ce7403973ff1f3f2715fbff9563ef2cdcdc321a7edc5dd15f5df"
EXPECTED_HISTORICAL_SHA256 = {
    "rev27_diagnostic_report": "ffd373d3937558aa71b5afccc70468aff068a28443f728a94343e488046bc315",
    "rev28_training_report": "a3178b3e35969b94a234e98a97106c2be9c4a7fa182d0f8a1fee166201e4e49d",
    "rev28_rejection_report": "712d8f1e788c14d03e86211dd7d44813c3467a577f26a9ed8268b9c1b379577f",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=not binary,
        encoding=None if binary else "utf-8",
        errors=None if binary else "replace",
    )
    if completed.returncode != 0:
        stderr = completed.stderr if isinstance(completed.stderr, str) else completed.stderr.decode(errors="replace")
        raise ValueError(
            f"git command failed: root={root} args={arguments!r} "
            f"exit={completed.returncode} stderr={stderr.strip()!r}"
        )
    return completed.stdout


def load_preregistration(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "preregistration must be an object")
    require(payload.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    require(payload.get("evidence_id") == EVIDENCE_ID, "evidence_id mismatch")
    require(payload.get("revision") == "rev29", "revision mismatch")
    return payload


def validate_semantics(preregistration: dict[str, Any]) -> dict[str, Any]:
    variable = preregistration["single_experimental_variable"]
    training = preregistration["training"]
    frozen = preregistration["frozen_contract"]
    readback = preregistration["runtime_readback"]
    gate = preregistration["acceptance_gate"]
    execution = preregistration["execution_order"]
    claims = preregistration["claim_limits"]

    require(
        variable
        == {
            "name": "normalized_joint_position_action_scale",
            "rejected_rev28_value": 0.7,
            "candidate_value": 0.65,
        },
        "single experimental variable mismatch",
    )
    require(math.isclose(ACTION_SCALE, 0.65, abs_tol=1e-12), "canonical action scale mismatch")
    require(training["task"] == "Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0", "task mismatch")
    require(training["device"] == "cuda:0" and training["headless"] is True, "GPU/headless mismatch")
    require(training["seed"] == 42 and training["scratch"] is True, "seed/scratch mismatch")
    require(training["resume"] is False, "resume must be false")
    require(training["num_envs"] == 1024, "num_envs mismatch")
    require(training["num_steps_per_env"] == 24, "rollout horizon mismatch")
    require(training["max_iterations"] == 50, "iteration budget mismatch")
    require(training["transitions"] == 1024 * 24 * 50 == 1_228_800, "transition budget mismatch")
    require(training["ppo_num_learning_epochs"] == 5, "PPO epoch mismatch")
    require(training["ppo_num_mini_batches"] == 4, "PPO minibatch mismatch")
    require(training["optimizer_updates_per_iteration"] == 20, "updates/iteration mismatch")
    require(training["optimizer_mini_batch_updates"] == 1000, "optimizer update budget mismatch")
    require(training["expected_checkpoint_name"] == "model_49.pt", "checkpoint name mismatch")
    require(training["pose_curriculum_phase"] == 0, "curriculum phase mismatch")
    require(
        training["pose_distribution"]
        == {"prone": 1.0, "supine": 0.0, "left_side": 0.0, "right_side": 0.0},
        "phase-0 pose distribution mismatch",
    )

    expected_frozen = {
        "ppo_entropy_coefficient": PPO_ENTROPY_COEF,
        "ppo_init_noise_std": PPO_INIT_NOISE_STD,
        "action_ema_alpha": ACTION_EMA_ALPHA,
        "articulation_solver_position_iteration_count": ARTICULATION_SOLVER_POSITION_ITERATION_COUNT,
        "articulation_solver_velocity_iteration_count": ARTICULATION_SOLVER_VELOCITY_ITERATION_COUNT,
        "max_depenetration_velocity_m_s": MAX_DEPENETRATION_VELOCITY_M_S,
        "asset_soft_joint_limit_factor": GO2_SOFT_JOINT_LIMIT_FACTOR,
        "policy_observation_dimension": 140,
        "critic_observation_dimension": CRITIC_OBSERVATION_DIM + 57,
        "reset_reward_observation_contract": "canonical configs/g009_r0.json",
    }
    require(frozen == expected_frozen, "frozen contract mismatch")
    require(
        readback
        == {
            "agent_yaml": {
                "entropy_coef": 0.0,
                "init_noise_std": 0.5,
                "num_steps_per_env": 24,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "max_iterations": 50,
                "device": "cuda:0",
            },
            "env_yaml": {
                "action_scale": 0.65,
                "rescale_to_limits": True,
                "action_ema_alpha": 0.2,
                "asset_soft_joint_limit_factor": 0.9,
                "articulation_solver_position_iteration_count": 8,
                "articulation_solver_velocity_iteration_count": 0,
                "max_depenetration_velocity_m_s": 1.0,
            },
        },
        "runtime readback contract mismatch",
    )
    require(
        gate["required_series"]
        == [
            "Episode_Termination/hard_joint_limit",
            "Episode_Termination/numeric_invalid",
            "Policy/mean_noise_std",
        ],
        "required TensorBoard series mismatch",
    )
    require(gate["tensorboard_exact_sample_count"] == 50, "sample count gate mismatch")
    require(gate["hard_joint_limit_maximum"] == 0.0, "hard-limit maximum gate mismatch")
    require(gate["hard_joint_limit_nonzero_sample_count"] == 0, "hard-limit nonzero gate mismatch")
    require(gate["numeric_invalid_maximum"] == 0.0, "numeric-invalid maximum gate mismatch")
    require(gate["numeric_invalid_nonzero_sample_count"] == 0, "numeric-invalid nonzero gate mismatch")
    require(gate["mean_noise_std_all_finite"] is True, "noise finite gate mismatch")
    require(gate["checkpoint_std_vector_dimension"] == 12, "checkpoint std dimension mismatch")
    require(gate["checkpoint_std_vector_all_finite"] is True, "checkpoint std finite gate mismatch")
    require(gate["gpu_temperature_threshold_c"] == 90.0, "GPU temperature gate mismatch")
    require(gate["gpu_sustained_hot_sample_count"] == 3, "GPU hot-sample gate mismatch")
    require(gate["gpu_fatal_events_maximum"] == 0, "GPU fatal-event gate mismatch")
    require(gate["gpu_descendants_must_exit"] is True, "GPU descendant gate mismatch")
    require(
        execution
        == {
            "prelaunch_validator_required": True,
            "smoke_must_pass_before_full_300_iteration_training": True,
            "held_out_seed_1042_forbidden_until_full_300_training_safety_zero": True,
        },
        "execution order mismatch",
    )
    require(claims["smoke_acceptance_is_policy_qualification"] is False, "qualification claim opened")
    require(claims["recovery_success_measured"] is False, "success claim opened")
    require(claims["action_scale_change_proved_causal"] is False, "causal claim opened")
    require(claims["held_out_evaluation_status"] == "forbidden_until_full_300_training_safety_zero", "held-out gate opened")
    require(claims["full_300_iteration_training_status"] == "forbidden_until_smoke_accepted", "full-run gate opened")
    return {"action_scale": ACTION_SCALE, **readback}


def validate_canonical_manifest() -> dict[str, Any]:
    path = REPO_ROOT / "configs" / "g009_r0.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    contract = recover_contract()
    require(manifest.get("contract") == contract, "configs/g009_r0.json is not canonical")
    require(manifest.get("contract_sha256") == canonical_sha256(contract), "canonical contract hash mismatch")
    require(contract.get("contract_id") == "g009_r0_recover_rev29", "canonical revision mismatch")
    return {"path": "configs/g009_r0.json", "sha256": file_sha256(path)}


def validate_historical_evidence(preregistration: dict[str, Any]) -> dict[str, Any]:
    evidence = preregistration["historical_evidence"]
    result: dict[str, Any] = {}
    reports_root = (REPO_ROOT / "reports" / "runs").resolve()
    for name, expected_sha256 in EXPECTED_HISTORICAL_SHA256.items():
        item = evidence[name]
        require(item["sha256"] == expected_sha256, f"{name} preregistered SHA-256 mismatch")
        path = (REPO_ROOT / item["path"]).resolve()
        require(path.is_relative_to(reports_root), f"{name} must remain under reports/runs")
        require(path.is_file(), f"{name} missing: {path}")
        actual = file_sha256(path)
        require(actual == expected_sha256, f"{name} file SHA-256 mismatch")
        result[name] = {"path": item["path"], "sha256": actual}
    require(
        math.isclose(evidence["rev28_step49_mean_noise_std"], 0.47734370827674866, abs_tol=1e-15),
        "rev28 step49 mean_noise_std mismatch",
    )
    return result


def validate_source_state(preregistration: dict[str, Any]) -> dict[str, Any]:
    paths = preregistration["source_binding_paths"]
    require(paths == sorted(paths), "source binding paths must use ordinal order")
    require(len(paths) == len(set(paths)) and paths, "source binding paths must be unique/nonempty")
    manifest_sha256 = canonical_json_sha256(paths)
    require(manifest_sha256 == EXPECTED_SOURCE_MANIFEST_SHA256, "compiled source manifest mismatch")
    require(manifest_sha256 == preregistration["source_binding_path_manifest_sha256"], "preregistered source manifest mismatch")
    tracked = set(str(_git(REPO_ROOT, "ls-files", "--full-name")).splitlines())
    require(all(path in tracked for path in paths), "source binding contains untracked file")
    dirty = str(_git(REPO_ROOT, "status", "--porcelain=v1", "--untracked-files=all")).strip()
    require(not dirty, f"repository must be clean before smoke: {dirty}")
    commit = str(_git(REPO_ROOT, "rev-parse", "HEAD")).strip()
    require(len(commit) == 40, "repository commit is invalid")
    worktree_files = {path: file_sha256(REPO_ROOT / path) for path in paths}
    worktree_payload = "\n".join(f"{path}:{worktree_files[path]}" for path in paths)
    commit_files: dict[str, str] = {}
    for path in paths:
        blob = _git(REPO_ROOT, "show", f"{commit}:{path}", binary=True)
        require(isinstance(blob, bytes), f"commit blob unavailable: {path}")
        commit_files[path] = hashlib.sha256(blob).hexdigest()
    commit_payload = "\n".join(f"{path}:{commit_files[path]}" for path in paths)
    return {
        "repository_commit": commit,
        "repository_clean": True,
        "source_paths_logically_equal_to_head": True,
        "path_manifest_sha256": manifest_sha256,
        "executed_worktree_sha256": {
            "bundle": hashlib.sha256(worktree_payload.encode("utf-8")).hexdigest(),
            "files": worktree_files,
        },
        "commit_blob_sha256": {
            "bundle": hashlib.sha256(commit_payload.encode("utf-8")).hexdigest(),
            "files": commit_files,
        },
    }


def validate_upstream(isaac_lab_path: Path) -> dict[str, Any]:
    root = isaac_lab_path.resolve()
    commit = str(_git(root, "rev-parse", "HEAD")).strip()
    require(commit == EXPECTED_ISAAC_LAB_COMMIT, "Isaac Lab commit mismatch")
    tracked_dirty = str(_git(root, "status", "--porcelain=v1", "--untracked-files=no")).strip()
    require(not tracked_dirty, f"Isaac Lab tracked files must be clean: {tracked_dirty}")
    train = root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    require(train.is_file(), "official train.py missing")
    train_hash = file_sha256(train)
    require(train_hash == EXPECTED_OFFICIAL_TRAIN_SHA256, "official train.py SHA-256 mismatch")
    return {
        "isaac_lab_path": str(root),
        "isaac_lab_commit": commit,
        "tracked_clean": True,
        "official_train_path": str(train),
        "official_train_sha256": train_hash,
    }


def validate(
    path: Path = DEFAULT_PREREGISTRATION,
    isaac_lab_path: Path = Path.home() / "IsaacLab",
) -> dict[str, Any]:
    preregistration = load_preregistration(path)
    return {
        "schema_version": "g009.r0.rev29.action_scale_smoke_prelaunch_validation.v1",
        "status": "pass",
        "evidence_id": EVIDENCE_ID,
        "preregistration": {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(path),
        },
        "canonical_manifest": validate_canonical_manifest(),
        "canonical_static_readback": validate_semantics(preregistration),
        "historical_evidence": validate_historical_evidence(preregistration),
        "source_state": validate_source_state(preregistration),
        "upstream": validate_upstream(isaac_lab_path),
        "claim_limits": preregistration["claim_limits"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--isaac-lab-path", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                validate(args.preregistration.resolve(), args.isaac_lab_path.resolve()),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except Exception as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
