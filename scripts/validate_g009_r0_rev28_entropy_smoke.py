#!/usr/bin/env python3
"""Fail-closed prelaunch validation for the G009 R0 rev28 entropy smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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


SCHEMA_VERSION = "g009.r0.rev28.entropy_smoke_preregistration.v1"
EVIDENCE_ID = "G009-5-E021"
DEFAULT_PREREGISTRATION = REPO_ROOT / "configs" / "g009_r0_rev28_entropy_smoke.json"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "230473fa68c7121656a50beb01e8a013c7230de5762b03230c806b3988dc3b07"
)
EXPECTED_REV27_SHA256 = "ffd373d3937558aa71b5afccc70468aff068a28443f728a94343e488046bc315"
EXPECTED_REV26_REPORT_SHA256 = (
    "71a3b45129b79f2beaed14fab486423aafdd0f92e860022cf22c2c6242391234"
)
EXPECTED_REV26_CHECKPOINT_SHA256 = (
    "75b38ac4c8f8b2ed17d73893350c1ca484ca3ef3c0d633273553e13efdf44c95"
)


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


def resolve_portable_path(value: str) -> Path:
    prefix = "%USERPROFILE%"
    if value.upper().startswith(prefix):
        home = Path(os.environ.get("USERPROFILE", Path.home()))
        value = str(home) + value[len(prefix) :]
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def load_preregistration(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "preregistration must be an object")
    require(payload.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    require(payload.get("evidence_id") == EVIDENCE_ID, "evidence_id mismatch")
    require(payload.get("revision") == "rev28", "revision mismatch")
    return payload


def validate_semantics(preregistration: dict[str, Any]) -> dict[str, Any]:
    variable = preregistration["single_experimental_variable"]
    training = preregistration["training"]
    frozen = preregistration["frozen_contract"]
    gate = preregistration["acceptance_gate"]
    execution = preregistration["execution_order"]
    claims = preregistration["claim_limits"]

    require(variable == {
        "name": "ppo_entropy_coefficient",
        "rejected_rev26_value": 0.01,
        "candidate_value": 0.0,
    }, "single experimental variable mismatch")
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
    require(training["pose_distribution"] == {
        "prone": 1.0, "supine": 0.0, "left_side": 0.0, "right_side": 0.0
    }, "phase-0 pose distribution mismatch")

    expected_frozen = {
        "ppo_init_noise_std": PPO_INIT_NOISE_STD,
        "action_scale": ACTION_SCALE,
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
    require(PPO_ENTROPY_COEF == 0.0, "canonical entropy coefficient mismatch")

    static_readback = {
        "entropy_coef": PPO_ENTROPY_COEF,
        "init_noise_std": PPO_INIT_NOISE_STD,
        "num_steps_per_env": training["num_steps_per_env"],
        "num_learning_epochs": training["ppo_num_learning_epochs"],
        "num_mini_batches": training["ppo_num_mini_batches"],
    }

    required_series = gate["required_series"]
    require(required_series == [
        "Episode_Termination/hard_joint_limit",
        "Episode_Termination/numeric_invalid",
        "Policy/mean_noise_std",
    ], "required TensorBoard series mismatch")
    require(gate["tensorboard_exact_sample_count"] == 50, "sample count gate mismatch")
    require(gate["hard_joint_limit_maximum"] == 0.0, "hard-limit maximum gate mismatch")
    require(gate["hard_joint_limit_nonzero_sample_count"] == 0, "hard-limit nonzero gate mismatch")
    require(gate["numeric_invalid_maximum"] == 0.0, "numeric-invalid maximum gate mismatch")
    require(gate["numeric_invalid_nonzero_sample_count"] == 0, "numeric-invalid nonzero gate mismatch")
    require(gate["checkpoint_std_vector_dimension"] == 12, "checkpoint std dimension mismatch")
    require(gate["checkpoint_std_vector_all_finite"] is True, "checkpoint std finite gate mismatch")
    require(gate["iteration49_mean_noise_std_maximum"] == 0.5513023734, "noise direction gate mismatch")
    require(gate["iteration49_mean_noise_std_comparison"] == "less_than_or_equal_to_rev26_step49", "noise comparison mismatch")
    require(gate["gpu_temperature_threshold_c"] == 90.0, "GPU temperature gate mismatch")
    require(gate["gpu_sustained_hot_sample_count"] == 3, "GPU hot-sample gate mismatch")
    require(gate["gpu_fatal_events_maximum"] == 0, "GPU fatal-event gate mismatch")
    require(gate["gpu_descendants_must_exit"] is True, "GPU descendant gate mismatch")
    require(execution == {
        "prelaunch_validator_required": True,
        "smoke_must_pass_before_full_300_iteration_training": True,
        "held_out_seed_1042_forbidden_until_full_300_training_safety_zero": True,
    }, "execution order mismatch")
    require(claims["smoke_acceptance_is_policy_qualification"] is False, "qualification claim opened")
    require(claims["recovery_success_measured"] is False, "success claim opened")
    require(claims["held_out_evaluation_status"] == "forbidden_until_full_300_training_safety_zero", "held-out gate opened")
    require(claims["full_300_iteration_training_status"] == "forbidden_until_smoke_accepted", "full-run gate opened")
    return static_readback


def validate_canonical_manifest() -> dict[str, Any]:
    manifest_path = REPO_ROOT / "configs" / "g009_r0.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = recover_contract()
    require(manifest.get("contract") == contract, "configs/g009_r0.json is not canonical")
    require(manifest.get("contract_sha256") == canonical_sha256(contract), "canonical contract hash mismatch")
    return {"path": "configs/g009_r0.json", "sha256": file_sha256(manifest_path)}


def validate_historical_evidence(preregistration: dict[str, Any]) -> dict[str, Any]:
    evidence = preregistration["historical_evidence"]
    expected = (
        ("rev27_diagnostic_report", EXPECTED_REV27_SHA256),
        ("rev26_training_report", EXPECTED_REV26_REPORT_SHA256),
        ("rev26_checkpoint", EXPECTED_REV26_CHECKPOINT_SHA256),
    )
    result: dict[str, Any] = {}
    for name, expected_sha256 in expected:
        item = evidence[name]
        require(item["sha256"] == expected_sha256, f"{name} preregistered SHA-256 mismatch")
        path = resolve_portable_path(item["path"])
        if name != "rev26_checkpoint":
            reports_root = (REPO_ROOT / "reports" / "runs").resolve()
            require(path.is_relative_to(reports_root), f"{name} must remain under reports/runs")
        else:
            require(Path(item["path"]).is_absolute() or item["path"].upper().startswith("%USERPROFILE%"), "rev26 checkpoint must use a portable absolute path")
        require(path.is_file(), f"{name} missing: {path}")
        actual = file_sha256(path)
        require(actual == expected_sha256, f"{name} file SHA-256 mismatch")
        result[name] = {"path": item["path"], "sha256": actual}
    require(
        math.isclose(evidence["rev26_step49_mean_noise_std"], 0.5513023734, abs_tol=1e-12),
        "rev26 step49 mean_noise_std mismatch",
    )
    require(
        evidence["comparison_semantics"]
        == "non-worsening smoke acceptance gate only; equality is allowed and neither strict improvement nor causality is claimed",
        "historical comparison semantics mismatch",
    )
    return result


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git command failed: args={arguments!r} exit={completed.returncode} "
            f"stderr={completed.stderr.strip()!r}"
        )
    return completed.stdout


def validate_source_state(preregistration: dict[str, Any]) -> dict[str, Any]:
    paths = preregistration["source_binding_paths"]
    require(paths == sorted(paths), "source binding paths must use ordinal order")
    require(len(paths) == len(set(paths)) and paths, "source binding paths must be unique/nonempty")
    manifest_sha256 = canonical_json_sha256(paths)
    require(manifest_sha256 == EXPECTED_SOURCE_MANIFEST_SHA256, "compiled source manifest mismatch")
    require(manifest_sha256 == preregistration["source_binding_path_manifest_sha256"], "preregistered source manifest mismatch")
    tracked = set(_git("ls-files", "--full-name").splitlines())
    require(all(path in tracked for path in paths), "source binding contains untracked file")
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all").strip()
    require(not dirty, f"repository must be clean before smoke: {dirty}")
    commit = _git("rev-parse", "HEAD").strip()
    require(len(commit) == 40, "repository commit is invalid")
    diff = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--quiet", "HEAD", "--", *paths],
        check=False,
    )
    require(diff.returncode == 0, "source paths differ logically from HEAD")
    worktree_files = {path: file_sha256(REPO_ROOT / path) for path in paths}
    worktree_bundle = "\n".join(f"{path}:{worktree_files[path]}" for path in paths)
    commit_files: dict[str, str] = {}
    for path in paths:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{commit}:{path}"],
            check=False,
            capture_output=True,
        )
        require(completed.returncode == 0, f"cannot read commit blob: {path}")
        commit_files[path] = hashlib.sha256(completed.stdout).hexdigest()
    commit_bundle = "\n".join(f"{path}:{commit_files[path]}" for path in paths)
    return {
        "repository_commit": commit,
        "repository_clean": True,
        "source_paths_logically_equal_to_head": True,
        "path_manifest_sha256": manifest_sha256,
        "executed_worktree_sha256": {
            "bundle": hashlib.sha256(worktree_bundle.encode("utf-8")).hexdigest(),
            "files": worktree_files,
        },
        "commit_blob_sha256": {
            "bundle": hashlib.sha256(commit_bundle.encode("utf-8")).hexdigest(),
            "files": commit_files,
        },
    }


def validate(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    preregistration = load_preregistration(path)
    return {
        "schema_version": "g009.r0.rev28.entropy_smoke_prelaunch_validation.v1",
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
        "claim_limits": preregistration["claim_limits"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.preregistration.resolve()), ensure_ascii=False, allow_nan=False))
    except Exception as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
