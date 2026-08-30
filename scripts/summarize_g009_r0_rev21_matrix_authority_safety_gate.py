#!/usr/bin/env python3
"""Pure rev21 evidence evaluator and read-only artifact revalidation adapter.

This module deliberately does not import Isaac Lab or the mutable rev20 runtime
module.  Historical contracts are read from caller-supplied Git blob bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = REPO_ROOT / "configs/g009_r0_rev21_matrix_authority_safety_gate.json"
CANONICAL_ARTIFACT_PATH = REPO_ROOT / "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json"
PREREGISTRATION_SCHEMA = "g009.r0.rev21.matrix_authority_safety_gate_preregistration.v1"
ARTIFACT_SCHEMA = "g009.r0.rev21.matrix_authority_safety_gate.v1"
PASS_REASON = "matrix_authority_safety_gate_passed_for_diagnostic_preregistration"
REQUIRED_SOURCE_PATHS = (
    "configs/g009_r0_rev21_matrix_authority_safety_gate.json",
    "scripts/summarize_g009_r0_rev21_matrix_authority_safety_gate.py",
    "scripts/run_g009_r0_rev21_matrix_authority_safety_gate.py",
)
HISTORICAL_SOURCE_PATHS = (
    "configs/g009_r0.json",
    "configs/g009_r0_rev20_terrain_contact_matrix.json",
    "reports/runs/g009_r0_rev19_contact_offset_intervention_synthesis_2x2x2_s42.json",
    "scripts/probe_g009_recover_runtime.py",
    "scripts/probe_g009_r0_rev18_gpu_raw_contact.py",
    "scripts/probe_g009_r0_rev19_contact_offset_intervention.py",
    "scripts/probe_g009_r0_rev20_terrain_contact_matrix.py",
    "scripts/summarize_g009_r0_rev20_terrain_contact_matrix.py",
    "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/recover_contracts.py",
    "src/isaac_walk_g009/recover_env_cfg.py",
    "src/isaac_walk_g009/registry.py",
)
RUNTIME_SOURCE_PATHS = (
    "configs/g009_r0.json",
    "configs/g009_r0_rev20_terrain_contact_matrix.json",
    "scripts/probe_g009_recover_runtime.py",
    "scripts/probe_g009_r0_rev18_gpu_raw_contact.py",
    "scripts/probe_g009_r0_rev19_contact_offset_intervention.py",
    "scripts/probe_g009_r0_rev20_terrain_contact_matrix.py",
    "reports/runs/g009_r0_rev19_contact_offset_intervention_synthesis_2x2x2_s42.json",
    "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/recover_contracts.py",
    "src/isaac_walk_g009/recover_env_cfg.py",
    "src/isaac_walk_g009/registry.py",
)
SYNTHESIS_SOURCE_PATHS = (
    "configs/g009_r0_rev20_terrain_contact_matrix.json",
    "scripts/probe_g009_r0_rev18_gpu_raw_contact.py",
    "scripts/probe_g009_r0_rev19_contact_offset_intervention.py",
    "scripts/probe_g009_r0_rev20_terrain_contact_matrix.py",
    "scripts/summarize_g009_r0_rev20_terrain_contact_matrix.py",
    "scripts/probe_g009_recover_runtime.py",
    "src/isaac_walk_g009/recover_contracts.py",
)
RAW_REPORT_PATHS = (
    "reports/runs/g009_r0_rev20_terrain_contact_matrix_cpu_rep01_s42.json",
    "reports/runs/g009_r0_rev20_terrain_contact_matrix_cpu_rep02_s42.json",
    "reports/runs/g009_r0_rev20_terrain_contact_matrix_gpu_rep01_s42.json",
    "reports/runs/g009_r0_rev20_terrain_contact_matrix_gpu_rep02_s42.json",
)
EXPECTED_SLOTS = ("cpu.rep1", "cpu.rep2", "cuda:0.rep1", "cuda:0.rep2")
REASON_PRIORITY = (
    "rev21_preregistration_invalid",
    "canonical_output_path_invalid",
    "canonical_output_already_exists",
    "rev21_source_provenance_invalid",
    "rev20_synthesis_missing_or_path_invalid",
    "rev20_synthesis_sha256_mismatch",
    "rev20_synthesis_json_or_schema_invalid",
    "rev20_synthesis_identity_or_decision_mismatch",
    "rev20_governance_or_claim_limit_mismatch",
    "rev20_evidence_chain_binding_invalid",
    "rev20_execution_identity_invalid",
    "rev20_source_provenance_invalid",
    "rev20_baseline_or_runtime_drift",
    "rev20_matrix_shape_or_order_invalid",
    "rev20_matrix_numeric_invalid",
    "rev20_matrix_physics_limit_exceeded",
    PASS_REASON,
)
REV20_GOVERNANCE = {
    "diagnostic_only": True,
    "learned": False,
    "reward_computed": False,
    "ppo_updates": 0,
    "gate_execution_allowed": False,
    "qualification_eligible": False,
    "qualification_status": "not_run",
    "physics_ground_truth_authority": False,
}
REV20_CLAIM_LIMITS = {
    "terrain_pair_aggregated_normal_force_authority_candidate_only": True,
    "gpu_contact_absence_claimed": False,
    "physics_failure_claimed": False,
    "callback_count_used": False,
}
UUID4_HEX = re.compile(r"[0-9a-f]{32}")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class GateValidationError(ValueError):
    """Expected fail-closed rejection with a stable public reason."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_bytes(raw: bytes, reason: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GateValidationError(reason, f"{label}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateValidationError(reason, f"{label} root must be an object")
    return value


def _fail(reason: str, detail: str) -> None:
    raise GateValidationError(reason, detail)


def _require(condition: object, reason: str, detail: str) -> None:
    if not condition:
        _fail(reason, detail)


def _mapping(value: Any, reason: str, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), reason, f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _is_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _number(value: Any, reason: str, label: str, *, nonnegative: bool = False) -> float:
    _require(_is_number(value), reason, f"{label} must be a finite non-bool number")
    result = float(value)
    if nonnegative:
        _require(result >= 0.0, reason, f"{label} must be nonnegative")
    return result


def _close(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    return _is_number(left) and _is_number(right) and abs(float(left) - float(right)) <= tolerance


def _uuid4_hex(value: Any, reason: str, label: str) -> str:
    _require(isinstance(value, str) and UUID4_HEX.fullmatch(value) is not None, reason, f"{label} malformed")
    try:
        parsed = uuid.UUID(hex=cast(str, value))
    except ValueError as exc:
        raise GateValidationError(reason, f"{label} malformed") from exc
    _require(parsed.version == 4 and parsed.hex == value, reason, f"{label} is not lowercase UUID4 hex")
    return cast(str, value)


def _sha(value: Any, reason: str, label: str) -> str:
    _require(isinstance(value, str) and SHA256_HEX.fullmatch(value) is not None, reason, f"{label} malformed")
    return cast(str, value)


def _bundle_digest(paths: Sequence[str], blobs: Mapping[str, bytes]) -> tuple[dict[str, str], str]:
    files = {path: sha256_bytes(blobs[path]) for path in paths}
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return files, sha256_bytes(payload.encode("utf-8"))


def validate_preregistration_bytes(raw: bytes) -> dict[str, Any]:
    value = strict_json_bytes(raw, "rev21_preregistration_invalid", "rev21 preregistration")
    expected_keys = {
        "schema_version", "evidence_id", "goal_id", "stage_id", "revision", "seed",
        "single_changed_axis", "predecessor", "rev21_source_binding", "historical_source_binding",
        "evidence_chain", "runtime_contract", "diagnostic_guards", "decision", "assurance_tiers",
        "claim_limits", "governance", "output_contract", "forbidden_changes", "stop_rules",
    }
    _require(set(value) == expected_keys, "rev21_preregistration_invalid", "rev21 top-level key set mismatch")
    _require(
        value.get("schema_version") == PREREGISTRATION_SCHEMA
        and value.get("evidence_id") == "G009-5-E014"
        and value.get("goal_id") == "g009"
        and value.get("stage_id") == "R0"
        and value.get("revision") == "rev21"
        and value.get("seed") == 42,
        "rev21_preregistration_invalid",
        "rev21 identity mismatch",
    )
    predecessor = value.get("predecessor")
    _require(isinstance(predecessor, Mapping), "rev21_preregistration_invalid", "predecessor missing")
    expected_predecessor = {
        "path": "reports/runs/g009_r0_rev20_terrain_contact_matrix_synthesis_2x2_s42.json",
        "sha256": "dcb8f446a212390f94f9ae5ccad97d9e770f9b8f5961f5ffb0c920f8d62580b3",
        "required_schema_version": "g009.r0.rev20.terrain_contact_matrix_synthesis.v1",
        "required_evidence_id": "G009-5-E013",
        "required_outcome": "terrain_pair_matrix_authority_candidate_validated",
        "required_next_step": "preregister_matrix_authority_safety_gate",
        "historical_source_commit": "fb2992965fcfb502a679065eac253a6bdcdf7086",
        "probe_source_bundle_sha256": "21353b2d90e43260e8446df094ef178264227a1040901d99230fb2c40b99b83c",
        "synthesis_source_bundle_sha256": "bf00813f9051969377d0bf7dee8092eff2df4643de3e29f2eff40d31d3be62e8",
        "cpu_preflight_path": "reports/runs/g009_r0_rev20_terrain_contact_matrix_cpu_preflight_2x_s42.json",
        "cpu_preflight_sha256": "2c4996f837d0c6003d653761c53ac399e98fb80db25c7d00ee647b871ac4968c",
    }
    _require(dict(predecessor) == expected_predecessor, "rev21_preregistration_invalid", "predecessor exact binding mismatch")
    source = value.get("rev21_source_binding", {})
    _require(source == {
        "required_committed_path_scoped_clean": True,
        "ordered_paths": list(REQUIRED_SOURCE_PATHS),
        "per_path_digest": "sha256 of exact git blob/worktree bytes",
        "aggregate_serialization": "ordered path:sha256 rows joined with LF, UTF-8, no trailing LF",
        "expected_aggregate_sha256": None,
    }, "rev21_preregistration_invalid", "rev21 source binding contract mismatch")
    _require("sha256" not in source or source.get("expected_aggregate_sha256") is None, "rev21_preregistration_invalid", "self aggregate digest is forbidden")
    historical = value.get("historical_source_binding", {})
    _require(historical == {
        "commit": expected_predecessor["historical_source_commit"],
        "ordered_unique_paths": list(HISTORICAL_SOURCE_PATHS),
        "runtime_bundle_paths": list(RUNTIME_SOURCE_PATHS),
        "synthesis_bundle_paths": list(SYNTHESIS_SOURCE_PATHS),
        "bundle_serialization": "path:sha256 rows sorted lexicographically by path, joined with LF, UTF-8, no trailing LF",
        "missing_commit_or_blob_policy": "reject_without_fetch",
    }, "rev21_preregistration_invalid", "historical source binding contract mismatch")
    chain = value.get("evidence_chain", {})
    _require(chain == {
        "ordered_raw_report_paths": list(RAW_REPORT_PATHS), "ordered_slots": list(EXPECTED_SLOTS),
        "raw_report_count": 4, "cpu_preflight_count": 1,
        "unique_path_sha256_and_execution_id_required": True,
        "parse_stop_rule": "synthesis plus exact four raw reports plus one CPU preflight",
        "rev19_stop_rule": "historical git blob hash only",
        "external_isaaclab_stop_rule": "compare recorded metadata only; do not read current external files",
    }, "rev21_preregistration_invalid", "evidence chain contract mismatch")
    runtime = value.get("runtime_contract", {})
    _require(runtime == {
        "num_envs": 8, "physics_steps": 150, "physics_dt_s": 0.005, "control_decimation": 4,
        "control_dt_s": 0.02, "simulated_duration_s": 0.75, "headless": True, "render": False,
        "raw_matrix_shape": [152, 1, 3], "reshaped_matrix_shape": [8, 19, 1, 3],
        "sensor_buffer_matrix_shape": [8, 19, 1, 3], "net_force_shape": [8, 19, 3],
        "source_env_index": 7, "filter_count": 1,
        "filter_paths": ["/World/ground/terrain/GroundPlane/CollisionPlane"],
    }, "rev21_preregistration_invalid", "runtime contract mismatch")
    _require(value.get("single_changed_axis") == "rev20 observation authority candidate pre-consumption safety decision only", "rev21_preregistration_invalid", "single changed axis mismatch")
    _require(value.get("diagnostic_guards") == {
        "gravity_m_s2": 9.81, "non_foot_peak_force_max_body_weight_inclusive": 15.0,
        "joint_lower_margin_min_rad_inclusive": -0.01, "joint_upper_margin_min_rad_inclusive": -0.01,
        "force_positive_threshold_n": 0.000001,
        "authority": "E013 diagnostic guard only; not a manufacturer or real-robot safety limit",
    }, "rev21_preregistration_invalid", "diagnostic guard contract mismatch")
    _require(value.get("decision") == {
        "reason_priority": list(REASON_PRIORITY), "check_states": ["pass", "fail", "not_evaluated"],
        "pass_outcome": PASS_REASON, "pass_next_step": "preregister_read_only_matrix_observation_adapter",
        "single_primary_reason": True, "dependency_failure_marks_downstream_not_evaluated": True,
    }, "rev21_preregistration_invalid", "decision contract mismatch")
    _require(value.get("assurance_tiers") == {
        "byte_and_provenance_verified": "fixed paths and SHA-256 plus historical Git blobs match",
        "persisted_evidence_internally_recomputed": "only values reproducible from persisted structures, magnitudes, and ledgers are recomputed",
        "physics_execution_not_independently_reobserved": "PhysX tensor generation, CPU/GPU physics reproducibility, and current external IsaacLab bytes are not independently reobserved",
    }, "rev21_preregistration_invalid", "assurance tier contract mismatch")
    _require(value.get("claim_limits") == {
        "policy_observation_connected": False, "reward_computed": False, "ppo_training_executed": False,
        "walking_turning_slope_or_self_recovery_qualified": False, "physics_ground_truth_authority": False,
        "simulator_launched": False, "rollout_steps": 0, "optimizer_updates": 0,
    }, "rev21_preregistration_invalid", "claim limits mismatch")
    _require(value.get("governance") == REV20_GOVERNANCE, "rev21_preregistration_invalid", "governance mismatch")
    output = value.get("output_contract", {})
    _require(output == {
        "canonical_path": "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json",
        "schema_version": ARTIFACT_SCHEMA, "immutable_no_overwrite": True, "pass_only_canonical_write": True,
        "check_only_is_read_only": True, "verify_artifact_is_read_only": True,
        "self_aggregate_digest_forbidden": True,
    }, "rev21_preregistration_invalid", "output contract mismatch")
    _require(value.get("forbidden_changes") == [
        "friction", "mass_or_inertia", "terrain", "reset_distribution", "action", "solver",
        "contact_or_rest_offset", "observation_adapter", "reward", "policy", "ppo", "checkpoint", "curriculum",
    ], "rev21_preregistration_invalid", "forbidden change contract mismatch")
    _require(value.get("stop_rules") == {
        "fail_closed_on_missing_or_mismatch": True, "network_fetch_allowed": False,
        "glob_or_directory_discovery_allowed": False, "dependency_traversal_allowed": False,
        "isaac_import_allowed": False, "physics_execution_allowed": False,
        "canonical_reject_write_allowed": False,
    }, "rev21_preregistration_invalid", "stop rule contract mismatch")
    return value


def _historical_preregistration(blobs: Mapping[str, bytes]) -> dict[str, Any]:
    path = "configs/g009_r0_rev20_terrain_contact_matrix.json"
    _require(set(blobs) == set(HISTORICAL_SOURCE_PATHS), "rev20_source_provenance_invalid", "historical blob path set mismatch")
    prereg = strict_json_bytes(blobs[path], "rev20_source_provenance_invalid", "historical rev20 preregistration")
    _require(prereg.get("schema_version") == "g009.r0.rev20.terrain_contact_matrix_preregistration.v1" and prereg.get("evidence_id") == "G009-5-E013" and prereg.get("revision") == "rev20" and prereg.get("seed") == 42, "rev20_source_provenance_invalid", "historical rev20 preregistration identity mismatch")
    _require(prereg.get("source_binding_contract", {}).get("runtime_source_binding_paths") == list(RUNTIME_SOURCE_PATHS) and prereg.get("source_binding_contract", {}).get("synthesis_source_binding_paths") == list(SYNTHESIS_SOURCE_PATHS), "rev20_source_provenance_invalid", "historical rev20 source path contract mismatch")
    return prereg


def _validate_source_bundle(bundle: Any, paths: Sequence[str], files: Mapping[str, str], digest: str, commit: str, *, runtime: bool) -> None:
    reason = "rev20_source_provenance_invalid"
    _require(isinstance(bundle, Mapping), reason, "source bundle missing")
    expected_keys = {"schema_version", "git_commit", "git_commit_valid", "source_binding_paths", "source_binding_files", "source_bundle_sha256", "clean"}
    if runtime:
        expected_keys |= {"all_files_present", "missing_files", "dirty_source_paths"}
    _require(set(bundle) == expected_keys, reason, "source bundle schema mismatch")
    _require(bundle.get("schema_version") == 1 and bundle.get("git_commit") == commit and bundle.get("git_commit_valid") is True and bundle.get("source_binding_paths") == list(paths) and bundle.get("source_binding_files") == {path: files[path] for path in paths} and bundle.get("source_bundle_sha256") == digest and bundle.get("clean") is True, reason, "source bundle bytes/provenance mismatch")
    if runtime:
        _require(bundle.get("all_files_present") is True and bundle.get("missing_files") == [] and bundle.get("dirty_source_paths") == [], reason, "runtime source bundle was not committed and clean")


def _validate_execution_mapping(
    execution: Any,
    *,
    expected_path: str,
    reason: str,
    label: str,
) -> Mapping[str, Any]:
    value = _mapping(execution, reason, label)
    _require(
        set(value)
        == {
            "execution_id",
            "started_at_utc",
            "output_path_repo_relative",
            "no_overwrite",
        },
        reason,
        f"{label} schema mismatch",
    )
    _uuid4_hex(value.get("execution_id"), reason, f"{label} execution_id")
    _require(
        isinstance(value.get("started_at_utc"), str)
        and value.get("output_path_repo_relative") == expected_path
        and value.get("no_overwrite") is True,
        reason,
        f"{label} canonical execution mismatch",
    )
    return value


def _validate_external_source_binding(
    report: Mapping[str, Any], historical_prereg: Mapping[str, Any]
) -> None:
    reason = "rev20_source_provenance_invalid"
    baseline_physics = _mapping(
        historical_prereg.get("baseline_physics"), reason, "historical baseline physics"
    )
    expected_contract = _mapping(
        baseline_physics.get("isaaclab_external_source_binding"),
        reason,
        "historical external source binding",
    )
    expected_files = _mapping(expected_contract.get("files"), reason, "external source files")
    external = _mapping(
        report.get("external_source_binding"), reason, "raw external source binding"
    )
    _require(
        set(external) == {"root", "files", "all_hashes_match"}
        and isinstance(external.get("root"), str)
        and bool(cast(str, external.get("root")).strip())
        and external.get("files") == dict(expected_files)
        and external.get("all_hashes_match") is True,
        reason,
        "external IsaacLab recorded metadata mismatch",
    )


def _validate_live_readback(report: Mapping[str, Any]) -> None:
    reason = "rev20_baseline_or_runtime_drift"
    baseline = _mapping(report.get("baseline_snapshot"), reason, "baseline snapshot")
    invariants = _mapping(baseline.get("invariants"), reason, "baseline invariants")
    body_names = invariants.get("mass_body_names")
    _require(
        isinstance(body_names, list)
        and len(body_names) == 19
        and len(set(body_names)) == 19
        and all(isinstance(name, str) and name for name in body_names),
        reason,
        "mass body-name inventory malformed",
    )
    namespaces = [f"/World/envs/env_{index}/Robot" for index in range(8)]
    roots = [f"{namespace}/base" for namespace in namespaces]
    live = _mapping(report.get("live_physics_readback"), reason, "live physics readback")
    _require(
        set(live) == {"solver", "max_depenetration_velocity"},
        reason,
        "live physics readback schema mismatch",
    )
    solver = _mapping(live.get("solver"), reason, "solver readback")
    expected_solver_rows = [
        {
            "prim_path": root,
            "solver_position_iteration_count": 8,
            "solver_velocity_iteration_count": 0,
        }
        for root in roots
    ]
    _require(
        solver
        == {
            "source": "USD PhysxArticulationAPI live-stage readback",
            "articulations": expected_solver_rows,
        },
        reason,
        "solver live readback mismatch",
    )
    depen = _mapping(
        live.get("max_depenetration_velocity"), reason, "depenetration readback"
    )
    link_groups = [
        [f"{namespace}/{name}" for name in cast(list[str], body_names)]
        for namespace in namespaces
    ]
    expected_articulations = []
    for env_index, (namespace, root, link_paths) in enumerate(
        zip(namespaces, roots, link_groups, strict=True)
    ):
        links = [
            {
                "body_index": body_index,
                "body_name": body_name,
                "prim_path": link_paths[body_index],
                "prim_valid": True,
                "usd_rigid_body_api": True,
                "physx_rigid_body_api": True,
                "max_depenetration_velocity_m_s": 1.0,
                "error": None,
            }
            for body_index, body_name in enumerate(cast(list[str], body_names))
        ]
        expected_articulations.append(
            {
                "articulation_index": env_index,
                "robot_container_prim_path": namespace,
                "articulation_prim_path": root,
                "root_link_prim_path": root,
                "authoritative_body_names": body_names,
                "authoritative_link_paths": link_paths,
                "links": links,
            }
        )
    _require(
        depen
        == {
            "source": "root_physx_view.link_paths direct USD/PhysX live-stage readback",
            "robot_container_prim_paths": namespaces,
            "articulation_prim_paths": roots,
            "authoritative_body_names": body_names,
            "authoritative_link_path_groups": link_groups,
            "articulation_group_count": 8,
            "rigid_body_count": 152,
            "duplicate_link_prim_paths": [],
            "articulations": expected_articulations,
        },
        reason,
        "max-depenetration live readback mismatch",
    )
    clock = _mapping(report.get("physics_step_clock"), reason, "physics step clock")
    observed_dt = clock.get("observed_dt_s")
    _require(
        set(clock)
        == {"source", "callback_count", "expected_callback_count", "observed_dt_s", "passed"}
        and clock.get("callback_count") == 150
        and clock.get("expected_callback_count") == 150
        and isinstance(observed_dt, list)
        and len(observed_dt) == 150
        and all(_close(value, 0.005, 1e-9) for value in observed_dt)
        and clock.get("passed") is True,
        reason,
        "physics step clock mismatch",
    )


def _validate_runtime_and_baseline(
    report: Mapping[str, Any], historical_prereg: Mapping[str, Any]
) -> None:
    reason = "rev20_baseline_or_runtime_drift"
    device = report.get("device")
    replicate = report.get("replicate_index")
    expected_index = {
        ("cpu", 1): 0,
        ("cpu", 2): 1,
        ("cuda:0", 1): 2,
        ("cuda:0", 2): 3,
    }.get((device, replicate))
    _require(expected_index is not None, reason, "raw report slot invalid")
    expected_runtime = {
        "num_envs": 8,
        "physics_steps": 150,
        "physics_dt_s": 0.005,
        "headless": True,
        "render": False,
    }
    baseline_physics = _mapping(
        historical_prereg.get("baseline_physics"), reason, "historical baseline physics"
    )
    expected_contract = {
        "schema_version": 1,
        "experiment_id": "G009-5-E013",
        "revision": "rev20",
        "slot": f"{device}.rep{replicate}",
        "baseline_source_cell": baseline_physics.get("source_cell"),
        "single_changed_axis": "contact observation path only",
        "terrain_filter_paths": ["/World/ground/terrain/GroundPlane/CollisionPlane"],
        "solver_iterations": [8, 0],
        "contact_offset_scale": 1.0,
        "runtime": expected_runtime,
        "governance": REV20_GOVERNANCE,
        "canonical_output": RAW_REPORT_PATHS[cast(int, expected_index)],
    }
    contract = _mapping(report.get("contract"), reason, "raw runtime contract")
    _require(
        dict(contract) == expected_contract
        and report.get("contract_sha256") == canonical_sha256(expected_contract)
        and report.get("goal_id") == "g009"
        and report.get("stage_id") == "R0"
        and report.get("revision") == "rev20"
        and report.get("seed") == 42
        and report.get("num_envs") == 8
        and report.get("source_env_index") == 7
        and report.get("physics_substeps") == 150
        and report.get("physics_dt_s") == 0.005
        and report.get("headless") is True,
        reason,
        "raw report runtime or contract drift",
    )
    manual = _mapping(report.get("manual_inner_loop"), reason, "manual inner loop")
    _require(
        manual
        == {
            "control_decimation": 4,
            "action_process_steps": list(range(1, 150, 4)),
            "action_process_count": 38,
            "manager_post_step_executed": False,
            "reward_computed": False,
            "termination_computed": False,
            "trajectory_equivalence_claimed": False,
            "scope": "capability_only",
        }
        and report.get("governance") == REV20_GOVERNANCE,
        reason,
        "manual loop or governance runtime drift",
    )
    baseline = _mapping(report.get("baseline_snapshot"), reason, "baseline snapshot")
    _require(baseline.get("all_match") is True, reason, "baseline aggregate did not pass")
    expected = _mapping(
        baseline_physics.get("expected_snapshot_contracts"), reason, "baseline contracts"
    )
    for name in ("material", "action", "motor", "reset", "timing"):
        item = _mapping(baseline.get(name), reason, f"baseline {name}")
        expected_item = _mapping(expected.get(name), reason, f"expected baseline {name}")
        digest = canonical_sha256(item.get("value"))
        _require(
            item.get("value") == expected_item.get("value")
            and item.get("sha256") == digest == expected_item.get("sha256")
            and item.get("expected_sha256") == expected_item.get("sha256")
            and item.get("matches") is True,
            reason,
            f"baseline {name} drift",
        )
    invariants = _mapping(baseline.get("invariants"), reason, "baseline invariants")
    for key, expected_key in (
        ("contact_offsets", "expected_contact_offset_tensor_sha256"),
        ("rest_offsets", "expected_rest_offset_tensor_sha256"),
        ("mass", "expected_mass_tensor_sha256"),
    ):
        record = _mapping(invariants.get(key), reason, f"baseline {key}")
        observed = canonical_sha256(
            {"shape": record.get("shape"), "values": record.get("values")}
        )
        _require(
            record.get("sha256") == observed == baseline_physics.get(expected_key),
            reason,
            f"baseline {key} hash drift",
        )
    raw_checks = _mapping(
        baseline.get("raw_runtime_checks"), reason, "baseline raw runtime checks"
    )
    invariant_checks = _mapping(
        invariants.get("checks"), reason, "baseline invariant checks"
    )
    _require(
        len(raw_checks) == 22
        and all(type(value) is bool and value is True for value in raw_checks.values())
        and set(invariant_checks)
        == {
            "contact_offset_tensor_hash",
            "rest_offset_tensor_hash",
            "mass_tensor_hash",
            "mass_body_order_hash",
            "force_body_order_hash",
        }
        and all(type(value) is bool and value is True for value in invariant_checks.values())
        and canonical_sha256(invariants.get("mass_body_names"))
        == baseline_physics.get("expected_mass_body_names_sha256")
        and canonical_sha256(invariants.get("force_body_names"))
        == baseline_physics.get("expected_force_body_names_sha256"),
        reason,
        "baseline raw checks or body inventories drift",
    )
    _validate_live_readback(report)


def _validate_cpu_preflight_binding(
    report: Mapping[str, Any],
    prereg: Mapping[str, Any],
    expected_cpu_inputs: Sequence[Mapping[str, str]] | None = None,
) -> None:
    reason = "rev20_evidence_chain_binding_invalid"
    device = report.get("device")
    binding = _mapping(
        report.get("cpu_preflight_binding"), reason, "raw CPU preflight binding"
    )
    cpu_not_required = {
        "status": "not_required_for_cpu",
        "path": None,
        "sha256": None,
        "git_commit": None,
        "probe_source_bundle_sha256": None,
        "input_reports": [],
    }
    if device == "cpu":
        _require(
            dict(binding) == cpu_not_required,
            reason,
            "CPU report preflight binding mismatch",
        )
        return
    _require(device == "cuda:0", reason, "preflight binding device invalid")
    inputs = binding.get("input_reports")
    _require(
        isinstance(inputs, list)
        and len(inputs) == 2
        and all(
            isinstance(item, Mapping)
            and set(item) == {"path", "sha256"}
            and item.get("path") == RAW_REPORT_PATHS[index]
            and isinstance(item.get("sha256"), str)
            and SHA256_HEX.fullmatch(cast(str, item.get("sha256"))) is not None
            for index, item in enumerate(inputs)
        )
        and len({cast(str, item["sha256"]) for item in inputs}) == 2,
        reason,
        "GPU report CPU input binding malformed",
    )
    if expected_cpu_inputs is not None:
        _require(
            inputs == [dict(item) for item in expected_cpu_inputs],
            reason,
            "GPU report CPU input bytes do not match canonical reports",
        )
    expected = {
        "status": "validated_for_gpu",
        "path": prereg["predecessor"]["cpu_preflight_path"],
        "sha256": prereg["predecessor"]["cpu_preflight_sha256"],
        "git_commit": prereg["predecessor"]["historical_source_commit"],
        "probe_source_bundle_sha256": prereg["predecessor"][
            "probe_source_bundle_sha256"
        ],
        "input_reports": inputs,
    }
    _require(
        dict(binding) == expected,
        reason,
        "GPU report preflight binding mismatch",
    )


def _validate_path_order_and_filter(
    report: Mapping[str, Any],
    matrix: Mapping[str, Any],
    prereg: Mapping[str, Any],
    historical_prereg: Mapping[str, Any],
) -> None:
    reason = "rev20_matrix_shape_or_order_invalid"
    runtime = prereg["runtime_contract"]
    terrain_contract = _mapping(
        historical_prereg.get("terrain_filter"), reason, "historical terrain filter"
    )
    logical_filters = runtime["filter_paths"]
    _require(
        terrain_contract.get("filter_prim_paths_expr") == logical_filters
        and terrain_contract.get("expected_filter_paths_sha256")
        == canonical_sha256(logical_filters)
        and terrain_contract.get("view_filter_paths_property_shape") == [152, 1]
        and terrain_contract.get("view_filter_names_property_shape") == [152, 1]
        and terrain_contract.get("filter_path_fallback_or_wildcard_allowed") is False,
        reason,
        "historical terrain-filter contract mismatch",
    )
    terrain_filter = _mapping(
        report.get("terrain_filter"), reason, "raw terrain filter"
    )
    _require(
        terrain_filter
        == {
            "filter_prim_paths_expr": logical_filters,
            "filter_paths_sha256": canonical_sha256(logical_filters),
            "injected_before_view_initialization": True,
            "fallback_used": False,
        },
        reason,
        "raw terrain-filter injection mismatch",
    )
    path_order = _mapping(matrix.get("path_order"), reason, "matrix path order")
    expected_keys = {
        "articulation_root_body_paths",
        "body_namespace_paths",
        "view_metadata",
        "sensor_paths",
        "sensor_paths_sha256",
        "filter_paths",
        "raw_filter_paths_sha256",
        "logical_filter_paths_sha256",
        "force_body_names",
        "force_body_names_sha256",
    }
    _require(set(path_order) == expected_keys, reason, "matrix path-order schema mismatch")
    body_names = path_order.get("force_body_names")
    _require(
        isinstance(body_names, list)
        and len(body_names) == 19
        and len(set(body_names)) == 19
        and all(isinstance(name, str) and name for name in body_names),
        reason,
        "force body-name inventory malformed",
    )
    namespaces = [f"/World/envs/env_{index}/Robot" for index in range(8)]
    roots = [f"{namespace}/base" for namespace in namespaces]
    sensor_paths = [
        f"{namespace}/{body_name}"
        for namespace in namespaces
        for body_name in cast(list[str], body_names)
    ]
    filter_rows = [list(logical_filters) for _ in sensor_paths]
    filter_name_row = terrain_contract.get("view_filter_names_row_expected")
    _require(
        isinstance(filter_name_row, list) and filter_name_row == ["CollisionPlane"],
        reason,
        "historical filter-name row mismatch",
    )
    expected_metadata = {
        "sensor_count": 152,
        "filter_count": 1,
        "sensor_names": cast(list[str], body_names) * 8,
        "filter_names": [list(filter_name_row) for _ in sensor_paths],
    }
    _require(
        path_order.get("articulation_root_body_paths") == roots
        and path_order.get("body_namespace_paths") == namespaces
        and path_order.get("view_metadata") == expected_metadata
        and path_order.get("sensor_paths") == sensor_paths
        and path_order.get("sensor_paths_sha256") == canonical_sha256(sensor_paths)
        and path_order.get("filter_paths") == filter_rows
        and path_order.get("raw_filter_paths_sha256")
        == canonical_sha256(filter_rows)
        and path_order.get("logical_filter_paths_sha256")
        == canonical_sha256(logical_filters)
        and path_order.get("force_body_names_sha256")
        == canonical_sha256(body_names),
        reason,
        "matrix body/filter/env mapping or digest mismatch",
    )


def _matrix_numeric_and_physics(matrix: Mapping[str, Any], prereg: Mapping[str, Any], expected_device: str) -> None:
    numeric_reason = "rev20_matrix_numeric_invalid"
    physics_reason = "rev20_matrix_physics_limit_exceeded"
    runtime = prereg["runtime_contract"]
    guards = prereg["diagnostic_guards"]
    expected_steps = list(range(1, int(runtime["physics_steps"]) + 1))
    expected_check_keys = {
        "exact_150_samples", "view_filter_shape_order_finite",
        "direct_matrix_sensor_buffer_parity_150_of_150",
        "direct_and_buffer_storage_independent_150_of_150",
        "same_body_positive_force_overlap_8_of_8", "source_env_7_overlap",
        "finite_joint_position_and_contact_force", "hard_joint_limit_with_margin",
        "all_env_non_foot_peak_force_within_15_bw",
        "source_env_non_foot_peak_force_within_15_bw",
        "force_and_mass_body_name_inventories_match",
        "default_mass_8x19_finite_positive_unchanged", "collection_error_absent",
    }
    checks = matrix.get("checks")
    _require(
        matrix.get("sample_count") == runtime["physics_steps"]
        and matrix.get("parity_step_indices") == expected_steps
        and matrix.get("storage_independent_step_indices") == expected_steps,
        "rev20_matrix_shape_or_order_invalid",
        "matrix sample/parity/storage step contract mismatch",
    )
    _require(
        isinstance(checks, Mapping) and set(checks) == expected_check_keys
        and all(type(value) is bool and value is True for value in checks.values()),
        "rev20_matrix_shape_or_order_invalid",
        "matrix serialized check set/status mismatch",
    )
    _require(
        matrix.get("structural_probe_valid") is True
        and matrix.get("safety_valid") is True
        and matrix.get("overlap_available") is True
        and matrix.get("contract_valid") is True
        and matrix.get("passed") is True
        and matrix.get("error") is None
        and matrix.get("availability_state") == "observed_valid",
        "rev20_matrix_shape_or_order_invalid",
        "matrix aggregate status mismatch",
    )
    _require(matrix.get("requested_device") == expected_device, "rev20_matrix_shape_or_order_invalid", "matrix requested device mismatch")
    ledger = matrix.get("step_ledger")
    _require(isinstance(ledger, list) and len(ledger) == runtime["physics_steps"], "rev20_matrix_shape_or_order_invalid", "matrix ledger length mismatch")
    safety = _mapping(matrix.get("safety"), "rev20_matrix_shape_or_order_invalid", "matrix safety")
    path_order = _mapping(
        matrix.get("path_order"), "rev20_matrix_shape_or_order_invalid", "matrix path order"
    )
    body_names = path_order.get("force_body_names")
    mass_body_names = safety.get("mass_body_names")
    _require(isinstance(body_names, list) and isinstance(mass_body_names, list) and len(body_names) == len(mass_body_names) == 19 and set(body_names) == set(mass_body_names), "rev20_matrix_shape_or_order_invalid", "force/mass body inventory mismatch")
    foot_indices = {index for index, name in enumerate(body_names) if isinstance(name, str) and name.endswith("_foot")}
    _require(len(foot_indices) == 4, "rev20_matrix_shape_or_order_invalid", "exactly four foot links required")
    mass_record = _mapping(
        safety.get("mass_tensor"), "rev20_matrix_shape_or_order_invalid", "mass tensor"
    )
    mass_values = mass_record.get("values")
    _require(mass_record.get("shape") == [8, 19] and isinstance(mass_values, list) and len(mass_values) == 8 and all(isinstance(row, list) and len(row) == 19 for row in mass_values), "rev20_matrix_shape_or_order_invalid", "mass tensor shape mismatch")
    for env_index, row in enumerate(mass_values):
        for body_index, value in enumerate(row):
            _number(value, numeric_reason, f"mass[{env_index}][{body_index}]", nonnegative=True)
            _require(float(value) > 0.0, numeric_reason, "mass must be positive")
    mass_digest = canonical_sha256({"shape": mass_record.get("shape"), "values": mass_values})
    _require(mass_record.get("sha256") == mass_digest, "rev20_matrix_shape_or_order_invalid", "safety mass tensor digest mismatch")
    weights = [sum(float(value) for value in row) * float(guards["gravity_m_s2"]) for row in mass_values]
    serialized_weights = safety.get("per_env_body_weight_n")
    _require(isinstance(serialized_weights, list) and len(serialized_weights) == 8, "rev20_matrix_shape_or_order_invalid", "body-weight vector shape mismatch")
    for index, (actual, expected) in enumerate(zip(serialized_weights, weights)):
        _number(actual, numeric_reason, f"body weight {index}", nonnegative=True)
        _require(_close(actual, expected, 1e-9), physics_reason, f"body weight {index} does not equal mass*9.81")
    nonfoot = safety.get("non_foot_peak_force_n_per_env")
    ratios = safety.get("non_foot_peak_force_body_weight_per_env")
    _require(isinstance(nonfoot, list) and isinstance(ratios, list) and len(nonfoot) == len(ratios) == 8, "rev20_matrix_shape_or_order_invalid", "non-foot safety vector shape mismatch")
    for index, (force, ratio, weight) in enumerate(zip(nonfoot, ratios, weights)):
        force_value = _number(force, numeric_reason, f"non-foot force {index}", nonnegative=True)
        ratio_value = _number(ratio, numeric_reason, f"non-foot BW {index}", nonnegative=True)
        _require(_close(ratio_value, force_value / weight, 1e-9), physics_reason, f"non-foot BW ratio {index} mismatch")
        _require(ratio_value <= float(guards["non_foot_peak_force_max_body_weight_inclusive"]), physics_reason, f"non-foot BW ratio {index} exceeds diagnostic guard")
    overlap_steps: list[list[int]] = [[] for _ in range(8)]
    all_peaks: list[float] = []
    source_peaks: list[float] = []
    observed_nonfoot = [0.0] * 8
    for expected_step, item in enumerate(ledger, start=1):
        _require(isinstance(item, Mapping) and item.get("step") == expected_step, "rev20_matrix_shape_or_order_invalid", "ledger step order mismatch")
        _require(item.get("direct_matrix_sha256") == item.get("sensor_buffer_sha256"), "rev20_matrix_shape_or_order_invalid", "direct/buffer digest parity mismatch")
        _require(item.get("storage_independent_before_clone") is True, "rev20_matrix_shape_or_order_invalid", "direct/buffer storage alias detected")
        _require(item.get("finite") is True and item.get("joint_position_finite") is True, numeric_reason, "ledger contains non-finite force or joint state")
        _require(item.get("mass_tensor_sha256") == mass_digest, "rev20_matrix_shape_or_order_invalid", "ledger mass tensor digest drift")
        _require(item.get("tensor_devices") == {"net_force": expected_device, "sensor_buffer": expected_device, "direct_matrix": expected_device}, "rev20_matrix_shape_or_order_invalid", "ledger tensor device mismatch")
        matrix_indices = item.get("matrix_positive_body_indices_by_env")
        net_indices = item.get("net_positive_body_indices_by_env")
        magnitudes = item.get("matrix_body_magnitude_n_by_env")
        net_magnitudes = item.get("net_body_magnitude_n_by_env")
        _require(all(isinstance(value, list) and len(value) == 8 for value in (matrix_indices, net_indices, magnitudes, net_magnitudes)), "rev20_matrix_shape_or_order_invalid", "ledger environment mapping mismatch")
        _require(item.get("matrix_body_magnitude_sha256") == canonical_sha256({"shape": [8, 19], "values": magnitudes}), "rev20_matrix_shape_or_order_invalid", "matrix magnitude digest mismatch")
        _require(item.get("net_body_magnitude_sha256") == canonical_sha256({"shape": [8, 19], "values": net_magnitudes}), "rev20_matrix_shape_or_order_invalid", "net magnitude digest mismatch")
        step_nonfoot = item.get("non_foot_peak_force_n_per_env")
        _require(isinstance(step_nonfoot, list) and len(step_nonfoot) == 8, "rev20_matrix_shape_or_order_invalid", "step non-foot peak vector shape mismatch")
        for env_index in range(8):
            _require(isinstance(magnitudes[env_index], list) and isinstance(net_magnitudes[env_index], list) and len(magnitudes[env_index]) == len(net_magnitudes[env_index]) == 19, "rev20_matrix_shape_or_order_invalid", "ledger body mapping mismatch")
            for body_index, value in enumerate(magnitudes[env_index]):
                _number(value, numeric_reason, f"matrix magnitude {expected_step}/{env_index}/{body_index}", nonnegative=True)
            for body_index, value in enumerate(net_magnitudes[env_index]):
                _number(value, numeric_reason, f"net magnitude {expected_step}/{env_index}/{body_index}", nonnegative=True)
            expected_step_nonfoot = max(
                (float(value) for body_index, value in enumerate(net_magnitudes[env_index]) if body_index not in foot_indices),
                default=0.0,
            )
            serialized_step_nonfoot = _number(step_nonfoot[env_index], numeric_reason, f"step non-foot peak {expected_step}/{env_index}", nonnegative=True)
            _require(_close(serialized_step_nonfoot, expected_step_nonfoot, 1e-9), physics_reason, "step non-foot peak does not match persisted net magnitudes")
            expected_matrix_indices = [index for index, value in enumerate(magnitudes[env_index]) if float(value) > float(guards["force_positive_threshold_n"])]
            expected_net_indices = [index for index, value in enumerate(net_magnitudes[env_index]) if float(value) > float(guards["force_positive_threshold_n"])]
            _require(matrix_indices[env_index] == expected_matrix_indices and net_indices[env_index] == expected_net_indices, "rev20_matrix_shape_or_order_invalid", "positive body index ledger mismatch")
            if set(expected_matrix_indices) & set(expected_net_indices):
                overlap_steps[env_index].append(expected_step)
            observed_nonfoot[env_index] = max(observed_nonfoot[env_index], expected_step_nonfoot)
        all_peak = max(float(value) for row in magnitudes for value in row)
        source_peak = max(float(value) for value in magnitudes[7])
        _require(_close(item.get("all_env_matrix_peak_force_n"), all_peak, 1e-9) and _close(item.get("source_env_matrix_peak_force_n"), source_peak, 1e-9), physics_reason, "serialized ledger peak mismatch")
        all_peaks.append(all_peak)
        source_peaks.append(source_peak)
        lower = item.get("joint_lower_margin_rad_by_env")
        upper = item.get("joint_upper_margin_rad_by_env")
        _require(isinstance(lower, list) and isinstance(upper, list) and len(lower) == len(upper) == 8, "rev20_matrix_shape_or_order_invalid", "joint margin mapping mismatch")
        margins_safe = True
        for rows, minimum, label in ((lower, guards["joint_lower_margin_min_rad_inclusive"], "lower"), (upper, guards["joint_upper_margin_min_rad_inclusive"], "upper")):
            for env_index, row in enumerate(rows):
                _require(isinstance(row, list) and len(row) == 12, "rev20_matrix_shape_or_order_invalid", f"{label} joint margin shape mismatch")
                for joint_index, value in enumerate(row):
                    margin = _number(value, numeric_reason, f"{label} margin {env_index}/{joint_index}")
                    margins_safe = margins_safe and margin >= float(minimum)
        _require(margins_safe and item.get("hard_joint_limit_with_margin") is True, physics_reason, "joint margin exceeded E013 diagnostic tolerance")
    _require(all(_close(observed_nonfoot[index], nonfoot[index], 1e-9) for index in range(8)), physics_reason, "non-foot peak does not match persisted ledger")
    summary = _mapping(
        matrix.get("same_step_overlap"),
        "rev20_matrix_shape_or_order_invalid",
        "same-step overlap summary",
    )
    _require(summary.get("per_env_overlap_step_indices") == overlap_steps and summary.get("source_env_overlap_step_indices") == overlap_steps[7], "rev20_matrix_shape_or_order_invalid", "overlap ledger mismatch")
    _require(_close(summary.get("all_env_matrix_peak_force_n"), max(all_peaks), 1e-9) and _close(summary.get("source_env_matrix_peak_force_n"), max(source_peaks), 1e-9), physics_reason, "summary peak mismatch")
    _require(_close(summary.get("all_env_matrix_force_integral_n_s"), sum(all_peaks) * 0.005, 1e-9) and _close(summary.get("source_env_matrix_force_integral_n_s"), sum(source_peaks) * 0.005, 1e-9), physics_reason, "summary integral mismatch")


def _validate_raw_report_or_raise(
    raw: bytes,
    prereg: Mapping[str, Any],
    historical_prereg: Mapping[str, Any],
    *,
    expected_cpu_inputs: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    report = strict_json_bytes(raw, "rev20_matrix_numeric_invalid", "raw report")
    expected_top_keys = {
        "schema_version", "goal_id", "stage_id", "experiment_id", "revision",
        "status", "headless", "device", "seed", "replicate_index", "num_envs",
        "source_env_index", "physics_substeps", "physics_dt_s", "manual_inner_loop",
        "finished_at_utc", "execution", "contract", "contract_sha256", "predecessor",
        "source_bundle", "governance", "pose_action_assignment",
        "live_physics_readback", "device_readback", "residual_capability",
        "physics_step_clock", "raw_contact_observation", "supporting_telemetry",
        "feasibility", "external_source_binding", "cpu_preflight_binding",
        "terrain_filter", "terrain_contact_matrix", "baseline_snapshot",
    }
    _require(
        set(report) == expected_top_keys
        and report.get("schema_version") == "g009.r0.rev20.terrain_contact_matrix.v1"
        and report.get("goal_id") == "g009"
        and report.get("stage_id") == "R0"
        and report.get("experiment_id") == "G009-5-E013"
        and report.get("revision") == "rev20"
        and report.get("status") == "complete",
        "rev20_evidence_chain_binding_invalid",
        "raw report schema or identity mismatch",
    )
    device = report.get("device")
    replicate = report.get("replicate_index")
    _require(device in {"cpu", "cuda:0"} and replicate in {1, 2}, "rev20_evidence_chain_binding_invalid", "raw report slot mismatch")
    expected_index = {("cpu", 1): 0, ("cpu", 2): 1, ("cuda:0", 1): 2, ("cuda:0", 2): 3}[(cast(str, device), cast(int, replicate))]
    _validate_cpu_preflight_binding(report, prereg, expected_cpu_inputs)
    _validate_execution_mapping(
        report.get("execution"),
        expected_path=RAW_REPORT_PATHS[expected_index],
        reason="rev20_execution_identity_invalid",
        label="raw report execution",
    )
    _validate_external_source_binding(report, historical_prereg)
    _validate_runtime_and_baseline(report, historical_prereg)
    readback = _mapping(
        report.get("device_readback"),
        "rev20_matrix_shape_or_order_invalid",
        "device readback",
    )
    _require(
        set(readback)
        == {
            "requested_device",
            "runtime_device",
            "physics_scene_prim_path",
            "gpu_dynamics_enabled",
            "gpu_dynamics_matches_device",
            "error",
        }
        and
        readback.get("requested_device") == device
        and readback.get("runtime_device") == device
        and readback.get("gpu_dynamics_enabled") is (device == "cuda:0")
        and readback.get("gpu_dynamics_matches_device") is True
        and readback.get("error") is None,
        "rev20_matrix_shape_or_order_invalid",
        "report device readback mismatch",
    )
    matrix = _mapping(
        report.get("terrain_contact_matrix"),
        "rev20_matrix_shape_or_order_invalid",
        "terrain matrix",
    )
    _validate_path_order_and_filter(report, matrix, prereg, historical_prereg)
    shapes = matrix.get("shapes", {})
    runtime = prereg["runtime_contract"]
    _require(shapes == {"raw": runtime["raw_matrix_shape"], "reshaped": runtime["reshaped_matrix_shape"], "sensor_buffer": runtime["sensor_buffer_matrix_shape"], "net_force": runtime["net_force_shape"]}, "rev20_matrix_shape_or_order_invalid", "matrix shape mismatch")
    _matrix_numeric_and_physics(matrix, prereg, cast(str, device))
    feasibility = _mapping(
        report.get("feasibility"), "rev20_baseline_or_runtime_drift", "feasibility"
    )
    _require(
        feasibility
        == {
            "probe_valid": True,
            "availability_state": matrix.get("availability_state"),
            "run_interpretable": True,
            "matrix_authority_candidate": matrix.get("availability_state")
            == "observed_valid",
        },
        "rev20_baseline_or_runtime_drift",
        "raw feasibility mirror mismatch",
    )
    return report


def validate_raw_report(raw_report_bytes: bytes, preregistration: Mapping[str, Any] | bytes, historical_preregistration: Mapping[str, Any] | bytes) -> dict[str, Any]:
    """Validate one lower-level raw fixture without outer immutable SHA masking."""
    try:
        prereg = validate_preregistration_bytes(preregistration) if isinstance(preregistration, bytes) else dict(preregistration)
        historical = strict_json_bytes(historical_preregistration, "rev20_source_provenance_invalid", "historical preregistration") if isinstance(historical_preregistration, bytes) else dict(historical_preregistration)
        _validate_raw_report_or_raise(raw_report_bytes, prereg, historical)
    except GateValidationError as exc:
        return {"passed": False, "primary_reason": exc.reason, "checks": [{"reason": exc.reason, "status": "fail", "detail": exc.detail}]}
    return {"passed": True, "primary_reason": PASS_REASON, "checks": [{"reason": reason, "status": "pass"} for reason in REASON_PRIORITY[12:17]]}


def _synthesis_row(
    report: Mapping[str, Any], item_binding: Mapping[str, str]
) -> dict[str, Any]:
    reason = "rev20_matrix_shape_or_order_invalid"
    matrix = _mapping(report.get("terrain_contact_matrix"), reason, "terrain matrix")
    path_order = _mapping(matrix.get("path_order"), reason, "matrix path order")
    overlap = _mapping(matrix.get("same_step_overlap"), reason, "same-step overlap")
    source_bundle = _mapping(
        report.get("source_bundle"), "rev20_source_provenance_invalid", "source bundle"
    )
    execution = _mapping(
        report.get("execution"), "rev20_execution_identity_invalid", "raw execution"
    )
    feasibility = _mapping(
        report.get("feasibility"), "rev20_baseline_or_runtime_drift", "feasibility"
    )
    baseline = _mapping(
        report.get("baseline_snapshot"), "rev20_baseline_or_runtime_drift", "baseline"
    )
    readback = _mapping(report.get("device_readback"), reason, "device readback")
    external = _mapping(
        report.get("external_source_binding"),
        "rev20_source_provenance_invalid",
        "external source binding",
    )
    return {
        "slot": f"{report['device']}.rep{report['replicate_index']}",
        "device": report["device"],
        "replicate_index": report["replicate_index"],
        "binding": dict(item_binding),
        "execution_id": execution["execution_id"],
        "availability_state": matrix["availability_state"],
        "sensor_paths_sha256": path_order["sensor_paths_sha256"],
        "raw_filter_paths_sha256": path_order["raw_filter_paths_sha256"],
        "logical_filter_paths_sha256": path_order["logical_filter_paths_sha256"],
        "force_body_names_sha256": path_order["force_body_names_sha256"],
        "raw_and_reshaped_tensor_shapes": [matrix["shapes"]["raw"], matrix["shapes"]["reshaped"]],
        "per_env_overlap_step_indices": overlap["per_env_overlap_step_indices"],
        "source_env_overlap_step_indices": overlap["source_env_overlap_step_indices"],
        "safety_checks": matrix["checks"],
        "all_env_matrix_peak_force_n": overlap["all_env_matrix_peak_force_n"],
        "source_env_matrix_peak_force_n": overlap["source_env_matrix_peak_force_n"],
        "all_env_matrix_force_integral_n_s": overlap["all_env_matrix_force_integral_n_s"],
        "source_env_matrix_force_integral_n_s": overlap["source_env_matrix_force_integral_n_s"],
        "source_bundle_sha256": source_bundle["source_bundle_sha256"],
        "cpu_preflight_binding": report["cpu_preflight_binding"],
        "git_commit": source_bundle["git_commit"],
        "probe_valid": feasibility["probe_valid"],
        "structural_probe_valid": matrix["structural_probe_valid"],
        "overlap_available": matrix["overlap_available"],
        "baseline_passed": baseline["all_match"],
        "device_passed": readback["gpu_dynamics_matches_device"],
        "live_readback_passed": True,
        "external_passed": external["all_hashes_match"],
        "safety_passed": matrix["safety_valid"],
    }


def validate_synthesis_row_mirrors(
    rows: Any,
    reports: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Validate every rev20 synthesis row as an exact mirror of its raw report."""

    try:
        _require(
            isinstance(rows, list)
            and len(rows) == len(reports) == len(bindings) == 4,
            "rev20_matrix_shape_or_order_invalid",
            "synthesis row mirror count mismatch",
        )
        for index, row in enumerate(rows):
            _require(
                isinstance(row, Mapping)
                and dict(row) == _synthesis_row(reports[index], bindings[index]),
                "rev20_matrix_shape_or_order_invalid",
                f"synthesis row {index} does not exactly mirror raw report",
            )
    except GateValidationError as exc:
        return {
            "passed": False,
            "primary_reason": exc.reason,
            "checks": [{"reason": exc.reason, "status": "fail", "detail": exc.detail}],
        }
    return {
        "passed": True,
        "primary_reason": PASS_REASON,
        "checks": [{"reason": "rev20_matrix_shape_or_order_invalid", "status": "pass"}],
    }


def _validate_cpu_preflight_value(
    preflight: Mapping[str, Any],
    prereg: Mapping[str, Any],
    observed_cpu_bindings: Sequence[Mapping[str, str]],
) -> None:
    reason = "rev20_evidence_chain_binding_invalid"
    _require(
        set(preflight)
        == {
            "schema_version",
            "evidence_id",
            "status",
            "mode",
            "input_report_count",
            "input_reports",
            "integrity",
            "cpu_preflight",
            "decision",
            "governance",
            "synthesis_source_bundle",
            "execution",
        }
        and preflight.get("schema_version")
        == "g009.r0.rev20.terrain_contact_matrix_cpu_preflight.v1"
        and preflight.get("evidence_id") == "G009-5-E013"
        and preflight.get("status") == "complete"
        and preflight.get("mode") == "cpu_preflight_2x"
        and preflight.get("input_report_count") == 2
        and preflight.get("input_reports")
        == [dict(item) for item in observed_cpu_bindings],
        reason,
        "CPU preflight schema, identity, or input binding mismatch",
    )
    integrity = _mapping(preflight.get("integrity"), reason, "CPU preflight integrity")
    predecessor = prereg["predecessor"]
    _require(
        integrity
        == {
            "passed": True,
            "hash_bound": True,
            "unique_report_paths": True,
            "unique_report_sha256": True,
            "unique_execution_ids": True,
            "exact_slots": ["cpu.rep1", "cpu.rep2"],
            "git_commit": predecessor["historical_source_commit"],
            "probe_source_bundle_sha256": predecessor["probe_source_bundle_sha256"],
            "synthesis_source_bundle_sha256": predecessor[
                "synthesis_source_bundle_sha256"
            ],
        },
        reason,
        "CPU preflight integrity mismatch",
    )
    cpu_gate = _mapping(preflight.get("cpu_preflight"), reason, "CPU preflight gate")
    _require(
        cpu_gate
        == {
            "passed": True,
            "required_checks_passed": True,
            "within_cpu_repeatability_passed": True,
            "gpu_stage_allowed": True,
        },
        reason,
        "CPU preflight decision flags mismatch",
    )
    decision = _mapping(preflight.get("decision"), reason, "CPU preflight decision")
    _require(
        decision
        == {
            "outcome": "gpu_stage_authorized",
            "third_run_allowed": False,
            "repeatability": {
                "exact_fields_match": True,
                "numeric_fields_within_tolerance": True,
                "repeatable": True,
                "absolute_tolerance": 1e-5,
                "relative_tolerance": 1e-6,
            },
        }
        and preflight.get("governance") == REV20_GOVERNANCE,
        reason,
        "CPU preflight outcome, repeatability, or governance mismatch",
    )
    _validate_execution_mapping(
        preflight.get("execution"),
        expected_path=predecessor["cpu_preflight_path"],
        reason="rev20_execution_identity_invalid",
        label="CPU preflight execution",
    )


def _check_record(reason: str, status: str, detail: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"reason": reason, "status": status}
    if detail is not None:
        value["detail"] = detail
    return value


def complete_reason_ledger(
    checks: Sequence[Mapping[str, Any]], primary_reason: str
) -> list[dict[str, Any]]:
    """Return the exact 17-row priority ledger for a pass or deterministic reject."""

    _require(
        primary_reason in REASON_PRIORITY,
        "rev21_preregistration_invalid",
        "unknown primary reason",
    )
    failure_index = (
        None if primary_reason == PASS_REASON else REASON_PRIORITY.index(primary_reason)
    )
    supplied: dict[str, dict[str, Any]] = {}
    previous_index = -1
    for item in checks:
        _require(
            isinstance(item, Mapping)
            and set(item).issubset({"reason", "status", "detail"})
            and {"reason", "status"}.issubset(item),
            "rev21_preregistration_invalid",
            "reason ledger row schema mismatch",
        )
        reason = item.get("reason")
        status = item.get("status")
        _require(
            isinstance(reason, str)
            and reason in REASON_PRIORITY
            and status in {"pass", "fail", "not_evaluated"},
            "rev21_preregistration_invalid",
            "reason ledger row value mismatch",
        )
        index = REASON_PRIORITY.index(cast(str, reason))
        _require(
            index > previous_index and reason not in supplied,
            "rev21_preregistration_invalid",
            "reason ledger contains a duplicate or is out of order",
        )
        expected_status = (
            "pass"
            if failure_index is None or index < failure_index
            else "fail"
            if index == failure_index
            else "not_evaluated"
        )
        _require(
            status == expected_status,
            "rev21_preregistration_invalid",
            f"reason ledger status contradicts primary reason: {reason}",
        )
        if "detail" in item:
            _require(
                isinstance(item.get("detail"), str),
                "rev21_preregistration_invalid",
                "reason ledger detail must be text",
            )
        supplied[cast(str, reason)] = dict(item)
        previous_index = index
    completed: list[dict[str, Any]] = []
    for index, reason in enumerate(REASON_PRIORITY):
        if failure_index is None:
            status = "pass"
        elif index < failure_index:
            status = "pass"
        elif index == failure_index:
            status = "fail"
        else:
            status = "not_evaluated"
        existing = supplied.get(reason, {})
        detail = existing.get("detail") if existing.get("status") == status else None
        if status == "not_evaluated" and detail is None and failure_index is not None:
            detail = f"depends on {primary_reason}"
        completed.append(
            _check_record(reason, status, cast(str | None, detail))
        )
    return completed


def validate_complete_reason_ledger(
    checks: Any, primary_reason: str
) -> list[dict[str, Any]]:
    _require(
        isinstance(checks, list) and len(checks) == len(REASON_PRIORITY),
        "rev21_preregistration_invalid",
        "complete reason ledger must contain exactly 17 rows",
    )
    completed = complete_reason_ledger(cast(list[Mapping[str, Any]], checks), primary_reason)
    _require(
        checks == completed,
        "rev21_preregistration_invalid",
        "complete reason ledger does not match the canonical ledger",
    )
    return completed


def _rejected_projection(prereg: Mapping[str, Any], checks: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "evidence_id": "G009-5-E014",
        "status": "rejected",
        "mode": "static_bounded_recursive_evidence_gate",
        "decision": {"passed": False, "outcome": reason, "primary_reason": reason, "next_step": "stop_and_repair_evidence_chain"},
        "checks": complete_reason_ledger(checks, reason),
        "assurance_tiers": prereg.get("assurance_tiers", {}),
        "claim_limits": prereg.get("claim_limits", {}),
        "governance": prereg.get("governance", REV20_GOVERNANCE),
        "execution_metrics": {"simulator_launched": False, "rollout_steps": 0, "optimizer_updates": 0},
    }


def evaluate_evidence(
    preregistration_bytes: bytes,
    synthesis_bytes: bytes,
    raw_report_bytes_by_path: Mapping[str, bytes],
    cpu_preflight_bytes: bytes,
    historical_blobs_by_path: Mapping[str, bytes],
) -> dict[str, Any]:
    """Return the canonical artifact deterministic projection; never performs I/O."""
    try:
        prereg = validate_preregistration_bytes(preregistration_bytes)
    except GateValidationError as exc:
        fallback = {"assurance_tiers": {}, "claim_limits": {}, "governance": REV20_GOVERNANCE}
        return _rejected_projection(fallback, [_check_record(exc.reason, "fail", exc.detail)], exc.reason)
    checks: list[dict[str, Any]] = []
    pure_reasons = REASON_PRIORITY[6:16]
    if sha256_bytes(synthesis_bytes) != prereg["predecessor"]["sha256"]:
        reason = "rev20_synthesis_sha256_mismatch"
        checks.append(_check_record(reason, "fail", "rev20 synthesis SHA-256 mismatch before JSON parse"))
        checks.extend(_check_record(item, "not_evaluated", f"depends on {reason}") for item in pure_reasons)
        return _rejected_projection(prereg, checks, reason)
    synthesis: dict[str, Any] | None = None
    reports: list[dict[str, Any]] = []
    preflight: dict[str, Any] | None = None
    historical_files: dict[str, str] = {}
    runtime_digest = synthesis_digest = ""
    try:
        synthesis = strict_json_bytes(synthesis_bytes, "rev20_synthesis_json_or_schema_invalid", "rev20 synthesis")
        _require(set(synthesis) == {"schema_version", "evidence_id", "status", "mode", "input_report_count", "input_reports", "integrity", "repeatability", "rows", "decision", "claim_limits", "governance", "synthesis_source_bundle", "execution"}, "rev20_synthesis_json_or_schema_invalid", "synthesis top-level schema mismatch")
        checks.append(_check_record(pure_reasons[0], "pass"))
        predecessor = prereg["predecessor"]
        synthesis_decision = _mapping(
            synthesis.get("decision"),
            "rev20_synthesis_identity_or_decision_mismatch",
            "synthesis decision",
        )
        _require(synthesis.get("schema_version") == predecessor["required_schema_version"] and synthesis.get("evidence_id") == predecessor["required_evidence_id"] and synthesis.get("status") == "complete" and synthesis.get("mode") == "final_2x2" and synthesis_decision.get("outcome") == predecessor["required_outcome"] and synthesis_decision.get("next_step") == predecessor["required_next_step"], "rev20_synthesis_identity_or_decision_mismatch", "synthesis identity/decision mismatch")
        checks.append(_check_record(pure_reasons[1], "pass"))
        _require(synthesis.get("governance") == REV20_GOVERNANCE and synthesis.get("claim_limits") == REV20_CLAIM_LIMITS, "rev20_governance_or_claim_limit_mismatch", "synthesis governance/claim limit mismatch")
        checks.append(_check_record(pure_reasons[2], "pass"))
        _require(set(raw_report_bytes_by_path) == set(RAW_REPORT_PATHS), "rev20_evidence_chain_binding_invalid", "raw report byte map path set mismatch")
        bindings = synthesis.get("input_reports")
        _require(isinstance(bindings, list) and len(bindings) == 4 and all(isinstance(item, Mapping) for item in bindings) and [item.get("path") for item in bindings] == list(RAW_REPORT_PATHS), "rev20_evidence_chain_binding_invalid", "synthesis raw report order mismatch")
        observed_bindings = [{"path": path, "sha256": sha256_bytes(raw_report_bytes_by_path[path])} for path in RAW_REPORT_PATHS]
        _require(bindings == observed_bindings and len({item["sha256"] for item in observed_bindings}) == 4, "rev20_evidence_chain_binding_invalid", "raw report binding mismatch")
        synthesis_integrity = _mapping(
            synthesis.get("integrity"),
            "rev20_evidence_chain_binding_invalid",
            "synthesis integrity",
        )
        _require(synthesis_integrity.get("preflight") == {"path": predecessor["cpu_preflight_path"], "sha256": sha256_bytes(cpu_preflight_bytes)} and sha256_bytes(cpu_preflight_bytes) == predecessor["cpu_preflight_sha256"], "rev20_evidence_chain_binding_invalid", "CPU preflight binding mismatch")
        preflight = strict_json_bytes(cpu_preflight_bytes, "rev20_evidence_chain_binding_invalid", "CPU preflight")
        _validate_cpu_preflight_value(preflight, prereg, observed_bindings[:2])
        checks.append(_check_record(pure_reasons[3], "pass"))
        synthesis_execution = _mapping(
            synthesis.get("execution"),
            "rev20_execution_identity_invalid",
            "synthesis execution",
        )
        preflight_execution = _mapping(
            preflight.get("execution"),
            "rev20_execution_identity_invalid",
            "preflight execution",
        )
        execution_ids = [_uuid4_hex(synthesis_execution.get("execution_id"), "rev20_execution_identity_invalid", "synthesis execution_id"), _uuid4_hex(preflight_execution.get("execution_id"), "rev20_execution_identity_invalid", "preflight execution_id")]
        for path in RAW_REPORT_PATHS:
            raw_value = strict_json_bytes(raw_report_bytes_by_path[path], "rev20_matrix_numeric_invalid", path)
            raw_execution = _mapping(
                raw_value.get("execution"),
                "rev20_execution_identity_invalid",
                f"{path} execution",
            )
            execution_ids.append(_uuid4_hex(raw_execution.get("execution_id"), "rev20_execution_identity_invalid", f"{path} execution_id"))
        _require(len(set(execution_ids)) == 6, "rev20_execution_identity_invalid", "execution IDs are not unique")
        checks.append(_check_record(pure_reasons[4], "pass"))
        historical_prereg = _historical_preregistration(historical_blobs_by_path)
        historical_files = {path: sha256_bytes(historical_blobs_by_path[path]) for path in HISTORICAL_SOURCE_PATHS}
        runtime_files, runtime_digest = _bundle_digest(RUNTIME_SOURCE_PATHS, historical_blobs_by_path)
        synthesis_files, synthesis_digest = _bundle_digest(SYNTHESIS_SOURCE_PATHS, historical_blobs_by_path)
        _require(runtime_digest == predecessor["probe_source_bundle_sha256"] and synthesis_digest == predecessor["synthesis_source_bundle_sha256"], "rev20_source_provenance_invalid", "historical aggregate digest mismatch")
        _validate_source_bundle(synthesis.get("synthesis_source_bundle"), SYNTHESIS_SOURCE_PATHS, synthesis_files, synthesis_digest, predecessor["historical_source_commit"], runtime=False)
        _validate_source_bundle(preflight.get("synthesis_source_bundle"), SYNTHESIS_SOURCE_PATHS, synthesis_files, synthesis_digest, predecessor["historical_source_commit"], runtime=False)
        for path in RAW_REPORT_PATHS:
            report = strict_json_bytes(raw_report_bytes_by_path[path], "rev20_matrix_numeric_invalid", path)
            _validate_source_bundle(report.get("source_bundle"), RUNTIME_SOURCE_PATHS, runtime_files, runtime_digest, predecessor["historical_source_commit"], runtime=True)
        checks.append(_check_record(pure_reasons[5], "pass"))
        reports = [
            _validate_raw_report_or_raise(
                raw_report_bytes_by_path[path],
                prereg,
                historical_prereg,
                expected_cpu_inputs=observed_bindings[:2],
            )
            for path in RAW_REPORT_PATHS
        ]
        checks.append(_check_record(pure_reasons[6], "pass"))
        rows = synthesis.get("rows")
        _require(isinstance(rows, list) and len(rows) == 4 and all(isinstance(row, Mapping) for row in rows) and [row.get("slot") for row in rows] == list(EXPECTED_SLOTS), "rev20_matrix_shape_or_order_invalid", "synthesis row order mismatch")
        for index, report in enumerate(reports):
            matrix = report.get("terrain_contact_matrix", {})
            runtime = prereg["runtime_contract"]
            _require(matrix.get("shapes") == {"raw": runtime["raw_matrix_shape"], "reshaped": runtime["reshaped_matrix_shape"], "sensor_buffer": runtime["sensor_buffer_matrix_shape"], "net_force": runtime["net_force_shape"]}, "rev20_matrix_shape_or_order_invalid", "matrix shape mismatch")
            _require(rows[index].get("binding") == observed_bindings[index] and rows[index].get("execution_id") == report.get("execution", {}).get("execution_id"), "rev20_matrix_shape_or_order_invalid", "synthesis row/raw report mismatch")
        row_result = validate_synthesis_row_mirrors(rows, reports, observed_bindings)
        _require(
            row_result.get("passed") is True,
            cast(str, row_result.get("primary_reason")),
            "synthesis row mirror mismatch",
        )
        checks.append(_check_record(pure_reasons[7], "pass"))
        checks.append(_check_record(pure_reasons[8], "pass"))
        checks.append(_check_record(pure_reasons[9], "pass"))
    except GateValidationError as exc:
        if not any(item["reason"] == exc.reason for item in checks):
            checks.append(_check_record(exc.reason, "fail", exc.detail))
        else:
            checks = [(_check_record(exc.reason, "fail", exc.detail) if item["reason"] == exc.reason else item) for item in checks]
        failed_index = REASON_PRIORITY.index(exc.reason)
        recorded = {item["reason"] for item in checks}
        for reason in pure_reasons:
            if REASON_PRIORITY.index(reason) > failed_index and reason not in recorded:
                checks.append(_check_record(reason, "not_evaluated", f"depends on {exc.reason}"))
        checks.sort(key=lambda item: REASON_PRIORITY.index(item["reason"]))
        return _rejected_projection(prereg, checks, exc.reason)
    assert synthesis is not None and preflight is not None
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "evidence_id": "G009-5-E014",
        "status": "complete",
        "mode": "static_bounded_recursive_evidence_gate",
        "predecessor": {"path": prereg["predecessor"]["path"], "sha256": sha256_bytes(synthesis_bytes), "outcome": synthesis["decision"]["outcome"], "next_step": synthesis["decision"]["next_step"]},
        "evidence_chain": {
            "raw_reports": [{"path": path, "sha256": sha256_bytes(raw_report_bytes_by_path[path])} for path in RAW_REPORT_PATHS],
            "cpu_preflight": {"path": prereg["predecessor"]["cpu_preflight_path"], "sha256": sha256_bytes(cpu_preflight_bytes)},
            "bounded_recursive_parse_complete": True,
        },
        "historical_source_binding": {
            "commit": prereg["historical_source_binding"]["commit"],
            "files": historical_files,
            "runtime_bundle_sha256": runtime_digest,
            "synthesis_bundle_sha256": synthesis_digest,
            "blob_count": len(historical_files),
        },
        "decision": {"passed": True, "outcome": PASS_REASON, "primary_reason": PASS_REASON, "next_step": "preregister_read_only_matrix_observation_adapter"},
        "checks": complete_reason_ledger(
            checks + [_check_record(PASS_REASON, "pass")], PASS_REASON
        ),
        "assurance_tiers": prereg["assurance_tiers"],
        "claim_limits": prereg["claim_limits"],
        "governance": prereg["governance"],
        "execution_metrics": {"simulator_launched": False, "rollout_steps": 0, "optimizer_updates": 0},
        "limitations": {
            "persisted_data_recalculation_only": True,
            "direct_matrix_and_sensor_buffer_sha256_equality_cross_checked_only": True,
            "raw_filtered_tensor_sha256_not_independently_regenerated": True,
            "physics_execution_independently_reobserved": False,
            "current_external_isaaclab_bytes_read": False,
            "diagnostic_force_and_joint_guards_are_real_robot_limits": False,
        },
    }


def artifact_deterministic_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the only volatile and runner-owned top-level fields."""
    return {key: item for key, item in value.items() if key not in {"execution", "rev21_source_binding"}}


def validate_artifact_value(value: Mapping[str, Any], expected_projection: Mapping[str, Any], rev21_source_binding: Mapping[str, Any]) -> dict[str, Any]:
    _require(value.get("schema_version") == ARTIFACT_SCHEMA, "rev21_preregistration_invalid", "artifact schema mismatch")
    _require(artifact_deterministic_projection(value) == dict(expected_projection), "rev21_source_provenance_invalid", "artifact deterministic projection mismatch")
    _require(value.get("rev21_source_binding") == dict(rev21_source_binding), "rev21_source_provenance_invalid", "artifact rev21 source binding mismatch")
    execution = value.get("execution")
    _require(
        isinstance(execution, Mapping)
        and set(execution)
        == {
            "execution_id",
            "started_at_utc",
            "output_path_repo_relative",
            "no_overwrite",
        }
        and execution.get("output_path_repo_relative")
        == "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json"
        and execution.get("no_overwrite") is True,
        "rev20_execution_identity_invalid",
        "artifact execution schema mismatch",
    )
    timestamp = execution.get("started_at_utc")
    _require(
        isinstance(timestamp, str)
        and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", timestamp
        )
        is not None,
        "rev20_execution_identity_invalid",
        "artifact timestamp must be RFC3339 UTC with Z",
    )
    try:
        parsed_timestamp = datetime.fromisoformat(cast(str, timestamp).replace("Z", "+00:00"))
    except ValueError as exc:
        raise GateValidationError(
            "rev20_execution_identity_invalid", "artifact timestamp is invalid"
        ) from exc
    _require(
        parsed_timestamp.tzinfo == timezone.utc,
        "rev20_execution_identity_invalid",
        "artifact timestamp is not UTC",
    )
    _uuid4_hex(execution.get("execution_id"), "rev20_execution_identity_invalid", "artifact execution_id")
    return dict(value)


def _git_blob(commit: str, path: str, reason: str = "rev20_source_provenance_invalid") -> bytes:
    try:
        return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO_ROOT, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as exc:
        raise GateValidationError(reason, f"Git blob unavailable without fetch: {commit}:{path}") from exc


def _fresh_rev21_source_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    commit = value.get("git_commit")
    _require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "rev21_source_provenance_invalid", "artifact rev21 commit malformed")
    files = {path: sha256_bytes(_git_blob(commit, path, "rev21_source_provenance_invalid")) for path in REQUIRED_SOURCE_PATHS}
    payload = "\n".join(f"{path}:{files[path]}" for path in REQUIRED_SOURCE_PATHS)
    expected = {
        "schema_version": 1,
        "git_commit": commit,
        "source_binding_paths": list(REQUIRED_SOURCE_PATHS),
        "source_binding_files": files,
        "source_bundle_sha256": sha256_bytes(payload.encode("utf-8")),
        "path_scoped_clean": True,
    }
    _require(dict(value) == expected, "rev21_source_provenance_invalid", "fresh rev21 blob binding mismatch")
    return expected


def verify_artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved == CANONICAL_ARTIFACT_PATH.resolve(), "canonical_output_path_invalid", "artifact must use canonical path")
    artifact = strict_json_bytes(resolved.read_bytes(), "rev21_preregistration_invalid", "rev21 artifact")
    prereg_raw = PREREGISTRATION_PATH.read_bytes()
    prereg = validate_preregistration_bytes(prereg_raw)
    synthesis_path = REPO_ROOT / prereg["predecessor"]["path"]
    raw_map = {relative: (REPO_ROOT / relative).read_bytes() for relative in RAW_REPORT_PATHS}
    preflight_raw = (REPO_ROOT / prereg["predecessor"]["cpu_preflight_path"]).read_bytes()
    historical_commit = prereg["historical_source_binding"]["commit"]
    historical_blobs = {relative: _git_blob(historical_commit, relative) for relative in HISTORICAL_SOURCE_PATHS}
    expected = evaluate_evidence(prereg_raw, synthesis_path.read_bytes(), raw_map, preflight_raw, historical_blobs)
    _require(expected.get("decision", {}).get("passed") is True, expected.get("decision", {}).get("primary_reason", "rev21_preregistration_invalid"), "fresh evidence no longer passes")
    source = artifact.get("rev21_source_binding")
    _require(isinstance(source, Mapping), "rev21_source_provenance_invalid", "artifact rev21 source binding missing")
    fresh_source = _fresh_rev21_source_binding(source)
    return validate_artifact_value(artifact, expected, fresh_source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-artifact", type=Path, metavar="PATH")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_artifact is None:
        print("--verify-artifact is required", file=sys.stderr)
        return 2
    try:
        value = verify_artifact(args.verify_artifact)
    except GateValidationError as exc:
        print(json.dumps({"verified": False, "primary_reason": exc.reason, "detail": exc.detail}, ensure_ascii=False), file=sys.stderr)
        return 2
    except (OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"verified": False, "primary_reason": "runner_input_io_error", "detail": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 3
    print(json.dumps({"verified": True, "path": str(args.verify_artifact), "outcome": value["decision"]["outcome"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
