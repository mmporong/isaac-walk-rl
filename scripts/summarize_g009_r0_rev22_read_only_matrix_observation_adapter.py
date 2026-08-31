#!/usr/bin/env python3
"""Pure E015 preregistration evaluator and read-only artifact verifier."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = REPO_ROOT / "configs/g009_r0_rev22_read_only_matrix_observation_adapter.json"
PREDECESSOR_PATH = REPO_ROOT / "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json"
CANONICAL_ARTIFACT_PATH = REPO_ROOT / "reports/runs/g009_r0_rev22_read_only_matrix_observation_adapter_preregistration_s42.json"
REV21_VERIFIER_PATH = REPO_ROOT / "scripts/summarize_g009_r0_rev21_matrix_authority_safety_gate.py"
PREREGISTRATION_SCHEMA = "g009.r0.rev22.read_only_matrix_observation_adapter_preregistration.v1"
ARTIFACT_SCHEMA = "g009.r0.rev22.read_only_matrix_observation_adapter_preregistration_artifact.v1"
PASS_REASON = "read_only_matrix_observation_adapter_preregistration_passed"
NEXT_STEP = "implement_and_run_read_only_matrix_observation_adapter_runtime_probe"
REQUIRED_SOURCE_PATHS = (
    "configs/g009_r0_rev22_read_only_matrix_observation_adapter.json",
    "scripts/summarize_g009_r0_rev22_read_only_matrix_observation_adapter.py",
    "scripts/run_g009_r0_rev22_read_only_matrix_observation_adapter.py",
    "scripts/summarize_g009_r0_rev21_matrix_authority_safety_gate.py",
)
REASON_PRIORITY = (
    "rev22_preregistration_invalid",
    "canonical_output_path_invalid",
    "canonical_output_already_exists",
    "rev22_source_provenance_invalid",
    "rev21_predecessor_missing_or_path_invalid",
    "rev21_predecessor_sha256_mismatch",
    "rev21_predecessor_json_or_schema_invalid",
    "rev21_predecessor_decision_or_governance_mismatch",
    "rev21_predecessor_full_verification_failed",
    "adapter_representation_contract_invalid",
    "adapter_coordinate_or_axis_order_invalid",
    "adapter_filter_reduction_invalid",
    "adapter_dtype_device_or_shape_invalid",
    "adapter_numeric_or_missing_contact_contract_invalid",
    "adapter_normalization_or_clipping_contract_invalid",
    "adapter_source_immutability_contract_invalid",
    "rev22_governance_or_claim_limit_mismatch",
    PASS_REASON,
)
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
UUID4_HEX = re.compile(r"[0-9a-f]{32}")
RFC3339_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z")
GOVERNANCE = {
    "diagnostic_only": True,
    "learned": False,
    "reward_computed": False,
    "ppo_updates": 0,
    "gate_execution_allowed": False,
    "qualification_eligible": False,
    "qualification_status": "not_run",
    "physics_ground_truth_authority": False,
}
CLAIM_LIMITS = {
    "adapter_implemented": False,
    "adapter_runtime_observed": False,
    "policy_observation_connected": False,
    "reward_computed": False,
    "ppo_training_executed": False,
    "walking_turning_slope_or_self_recovery_qualified": False,
    "physics_ground_truth_authority": False,
    "simulator_launched": False,
    "rollout_steps": 0,
    "optimizer_updates": 0,
}
REV21_GOVERNANCE = GOVERNANCE
REV21_CLAIM_LIMITS = {
    "policy_observation_connected": False,
    "reward_computed": False,
    "ppo_training_executed": False,
    "walking_turning_slope_or_self_recovery_qualified": False,
    "physics_ground_truth_authority": False,
    "simulator_launched": False,
    "rollout_steps": 0,
    "optimizer_updates": 0,
}


class GateValidationError(ValueError):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class OperationalVerificationError(RuntimeError):
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


def _fail(reason: str, detail: str) -> None:
    raise GateValidationError(reason, detail)


def _require(condition: object, reason: str, detail: str) -> None:
    if not condition:
        _fail(reason, detail)


def _mapping(value: Any, reason: str, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), reason, f"{label} must be an object")
    return cast(Mapping[str, Any], value)


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
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GateValidationError(reason, f"{label}: {exc}") from exc
    _require(isinstance(value, dict), reason, f"{label} root must be an object")
    return cast(dict[str, Any], value)


def _is_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _source_contract() -> dict[str, Any]:
    return {
        "name": "terrain_pair_force_matrix_w", "status_required": "available",
        "quantity_semantics": "filtered_normal_contact_force_vector",
        "total_contact_force_included": False,
        "tangential_friction_force_included": False,
        "friction_effect_directly_observed": False,
        "upstream_semantics_source": "Isaac Lab ContactSensorData.force_matrix_w",
        "coordinate_frame": "world", "axis_order": ["environment", "body", "filter", "xyz"],
        "component_order": ["x", "y", "z"], "shape": ["N", 19, 1, 3],
        "dtype": "torch.float32", "device_rule": "preserve_source_device_without_fallback_or_transfer",
        "body_order_authority": "rev20 validated sensor body order",
        "filter_order_authority": "rev20 validated terrain filter order", "units": "N",
    }


def validate_adapter_contract(value: Any) -> dict[str, Any]:
    contract = _mapping(value, "adapter_representation_contract_invalid", "adapter contract")
    _require(set(contract) == {"authority_role", "policy_observation_connected", "source", "world_xyz_output", "magnitude_projection", "contact_mask", "missing_contact", "normalization_and_clipping", "source_immutability", "claim_limits"}, "adapter_representation_contract_invalid", "adapter contract key set mismatch")
    _require(contract.get("authority_role") == "world_xyz_authority_preserving_output" and contract.get("policy_observation_connected") is False, "adapter_representation_contract_invalid", "adapter authority role mismatch")
    source = _mapping(contract.get("source"), "adapter_representation_contract_invalid", "adapter source")
    expected_source = _source_contract()
    _require(set(source) == set(expected_source), "adapter_representation_contract_invalid", "adapter source key set mismatch")
    xyz = _mapping(contract.get("world_xyz_output"), "adapter_representation_contract_invalid", "world XYZ output")
    expected_xyz = {
        "name": "terrain_pair_force_w_xyz_n", "formula": "source.sum(dim=2)",
        "reduction_order": "sum_filter_before_any_vector_norm", "coordinate_frame": "world",
        "component_order": ["x", "y", "z"], "shape": ["N", 19, 3],
        "dtype_rule": "exact_source_dtype", "device_rule": "exact_source_device", "units": "N", "authoritative": True,
    }
    _require(set(xyz) == set(expected_xyz), "adapter_representation_contract_invalid", "world XYZ output key set mismatch")
    magnitude = _mapping(contract.get("magnitude_projection"), "adapter_representation_contract_invalid", "magnitude projection")
    expected_magnitude = {
        "name": "terrain_pair_force_magnitude_n", "formula": "linalg_vector_norm(world_xyz_output,dim=-1)",
        "shape": ["N", 19], "dtype_rule": "exact_source_dtype", "device_rule": "exact_source_device",
        "units": "N", "authoritative": False, "diagnostic_only": True,
        "policy_input_allowed": False, "direction_and_sign_preserved": False,
    }
    _require(set(magnitude) == set(expected_magnitude), "adapter_representation_contract_invalid", "magnitude projection key set mismatch")
    contact_mask = _mapping(contract.get("contact_mask"), "adapter_representation_contract_invalid", "contact mask")
    expected_contact_mask = {
        "name": "terrain_pair_contact_mask", "formula": "magnitude_projection > 0.000001",
        "threshold_n": 0.000001, "threshold_comparison": "strict_greater_than",
        "shape": ["N", 19], "dtype": "torch.bool", "device_rule": "exact_source_device",
    }
    _require(set(contact_mask) == set(expected_contact_mask), "adapter_representation_contract_invalid", "contact mask key set mismatch")
    semantic_keys = {
        "name",
        "status_required",
        "quantity_semantics",
        "total_contact_force_included",
        "tangential_friction_force_included",
        "friction_effect_directly_observed",
        "upstream_semantics_source",
        "units",
    }
    _require(
        all(source.get(key) == expected_source[key] for key in semantic_keys),
        "adapter_representation_contract_invalid",
        "source force-quantity semantics mismatch",
    )
    _require(
        all(xyz.get(key) == expected_xyz[key] for key in {"name", "units", "authoritative"})
        and all(
            magnitude.get(key) == expected_magnitude[key]
            for key in {
                "name",
                "units",
                "authoritative",
                "diagnostic_only",
                "policy_input_allowed",
                "direction_and_sign_preserved",
            }
        ),
        "adapter_representation_contract_invalid",
        "adapter output authority roles mismatch",
    )
    coordinate_keys = {
        "coordinate_frame",
        "axis_order",
        "component_order",
        "body_order_authority",
        "filter_order_authority",
    }
    _require(
        all(source.get(key) == expected_source[key] for key in coordinate_keys),
        "adapter_coordinate_or_axis_order_invalid",
        "source coordinate/axis contract mismatch",
    )
    _require(
        all(xyz.get(key) == expected_xyz[key] for key in {"coordinate_frame", "component_order"}),
        "adapter_coordinate_or_axis_order_invalid",
        "world XYZ coordinate/component order mismatch",
    )
    _require(
        xyz.get("formula") == expected_xyz["formula"]
        and xyz.get("reduction_order") == expected_xyz["reduction_order"]
        and magnitude.get("formula") == expected_magnitude["formula"],
        "adapter_filter_reduction_invalid",
        "filter-sum-before-norm contract mismatch",
    )
    dtype_device_shape_keys = {"shape", "dtype", "device_rule"}
    _require(
        all(source.get(key) == expected_source[key] for key in dtype_device_shape_keys),
        "adapter_dtype_device_or_shape_invalid",
        "source dtype/device/shape contract mismatch",
    )
    _require(
        all(xyz.get(key) == expected_xyz[key] for key in {"shape", "dtype_rule", "device_rule"})
        and all(magnitude.get(key) == expected_magnitude[key] for key in {"shape", "dtype_rule", "device_rule"})
        and all(contact_mask.get(key) == expected_contact_mask[key] for key in {"shape", "dtype", "device_rule"}),
        "adapter_dtype_device_or_shape_invalid",
        "adapter output dtype/device/shape mismatch",
    )
    _require(
        all(contact_mask.get(key) == expected_contact_mask[key] for key in {"name", "formula", "threshold_n", "threshold_comparison"}),
        "adapter_numeric_or_missing_contact_contract_invalid",
        "contact mask numeric contract mismatch",
    )
    _require(contract.get("missing_contact") == {
        "valid_zero_source_vector": "emit exact zero XYZ, zero magnitude, and false mask",
        "threshold_equal_magnitude": "false mask without changing XYZ or magnitude",
        "missing_or_unavailable_source": "fail_closed", "none_source": "fail_closed",
        "wrong_shape_dtype_device_or_nonfinite": "fail_closed", "zero_fill_for_invalid_source_allowed": False,
    }, "adapter_numeric_or_missing_contact_contract_invalid", "missing-contact contract mismatch")
    _require(contract.get("normalization_and_clipping") == {
        "normalization": "none", "empirical_normalization": False,
        "body_weight_normalization": False, "clipping": "none", "saturation": "none",
        "policy_load_projection_deferred": True,
    }, "adapter_normalization_or_clipping_contract_invalid", "normalization/clipping contract mismatch")
    _require(contract.get("source_immutability") == {
        "in_place_operation_allowed": False,
        "source_shape_dtype_device_stride_must_match_before_after": True,
        "source_storage_identity_must_match_before_after": True,
        "source_version_must_match_before_after": True,
        "source_exact_values_and_sha256_must_match_before_after": True,
        "outputs_must_not_alias_source_storage": True, "operations_must_be_out_of_place": True,
        "correctness_hash_sampling_excluded_from_throughput_timing": True,
    }, "adapter_source_immutability_contract_invalid", "source immutability contract mismatch")
    _require(contract.get("claim_limits") == {
        "normal_contact_force_vector_claim_allowed": True,
        "total_contact_force_claim_allowed": False,
        "tangential_friction_force_claim_allowed": False,
        "friction_effect_directly_observed_claim_allowed": False,
        "surface_normal_or_tangential_decomposition_claim_allowed": False,
        "contact_point_or_separation_claim_allowed": False,
        "policy_input_claim_allowed": False, "physics_ground_truth_authority": False,
    }, "rev22_governance_or_claim_limit_mismatch", "adapter claim limits mismatch")
    return dict(contract)


def validate_preregistration_bytes(raw: bytes) -> dict[str, Any]:
    value = strict_json_bytes(raw, "rev22_preregistration_invalid", "rev22 preregistration")
    expected_keys = {"schema_version", "evidence_id", "goal_id", "stage_id", "revision", "seed", "single_changed_axis", "predecessor", "rev22_source_binding", "adapter_contract", "decision", "assurance_tiers", "claim_limits", "governance", "output_contract", "runtime_and_throughput_deferred", "forbidden_changes", "stop_rules"}
    _require(set(value) == expected_keys, "rev22_preregistration_invalid", "rev22 top-level key set mismatch")
    _require(value.get("schema_version") == PREREGISTRATION_SCHEMA and value.get("evidence_id") == "G009-5-E015" and value.get("goal_id") == "g009" and value.get("stage_id") == "R0" and value.get("revision") == "rev22" and value.get("seed") == 42 and value.get("single_changed_axis") == "read-only terrain-pair contact-matrix observation adapter contract only", "rev22_preregistration_invalid", "rev22 identity mismatch")
    _require(value.get("predecessor") == {
        "path": "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json",
        "sha256": "68a60383d9d49bf009d189498f94e6fe1c03155259932dcff8965caa7d9aa250",
        "required_schema_version": "g009.r0.rev21.matrix_authority_safety_gate.v1",
        "required_evidence_id": "G009-5-E014",
        "required_outcome": "matrix_authority_safety_gate_passed_for_diagnostic_preregistration",
        "required_next_step": "preregister_read_only_matrix_observation_adapter",
        "required_source_commit": "e202ae1d514c7abfe05ce0da130c2db47e9e05f3",
        "required_source_bundle_sha256": "3589f3852c4365afadb8dbdc871ffb2fe81888155b7e66c274034b5506eaf40e",
        "full_read_only_verification_required": True,
    }, "rev22_preregistration_invalid", "predecessor exact binding mismatch")
    _require(value.get("rev22_source_binding") == {
        "required_committed_path_scoped_clean": True, "ordered_paths": list(REQUIRED_SOURCE_PATHS),
        "per_path_digest": "sha256 of exact git blob/worktree bytes",
        "aggregate_serialization": "ordered path:sha256 rows joined with LF, UTF-8, no trailing LF",
        "expected_aggregate_sha256": None,
    }, "rev22_preregistration_invalid", "rev22 source binding contract mismatch")
    _require(value.get("decision") == {
        "reason_priority": list(REASON_PRIORITY), "check_states": ["pass", "fail", "not_evaluated"],
        "pass_outcome": PASS_REASON, "pass_next_step": NEXT_STEP,
        "single_primary_reason": True, "dependency_failure_marks_downstream_not_evaluated": True,
    }, "rev22_preregistration_invalid", "decision contract mismatch")
    _require(value.get("assurance_tiers") == {
        "predecessor_byte_and_full_verification_required": True,
        "adapter_semantics_preregistered_not_runtime_observed": True,
        "runtime_and_throughput_deferred": True,
    }, "rev22_preregistration_invalid", "assurance tiers mismatch")
    _require(value.get("output_contract") == {
        "canonical_path": "reports/runs/g009_r0_rev22_read_only_matrix_observation_adapter_preregistration_s42.json",
        "schema_version": ARTIFACT_SCHEMA, "immutable_no_overwrite": True,
        "pass_only_canonical_write": True, "check_only_is_read_only": True,
        "verify_artifact_is_read_only": True, "self_aggregate_digest_forbidden": True,
    }, "rev22_preregistration_invalid", "output contract mismatch")
    _require(value.get("runtime_and_throughput_deferred") == {
        "runtime_probe_required_after_pass": True,
        "initial_correctness_probe": "8 env, 150 physics steps, CPU 2x and cuda:0 2x",
        "throughput_measurement_before_runtime_correctness_pass_allowed": False,
        "throughput_contract_requires_separate_preregistration": True,
        "stable_maximum_definition": "maximum median env-control-steps-per-second among stable rungs, not maximum environment count or 100 percent GPU utilization",
    }, "rev22_preregistration_invalid", "runtime/throughput deferral mismatch")
    _require(value.get("forbidden_changes") == ["friction", "mass_or_inertia", "terrain", "reset_distribution", "action", "solver", "contact_or_rest_offset", "reward", "policy", "ppo", "checkpoint", "curriculum", "simulator_runtime"], "rev22_preregistration_invalid", "forbidden changes mismatch")
    _require(value.get("stop_rules") == {
        "fail_closed_on_missing_or_mismatch": True, "network_fetch_allowed": False,
        "glob_or_directory_discovery_allowed": False, "isaac_import_allowed": False,
        "simulator_launch_allowed": False, "runtime_adapter_implementation_allowed": False,
        "canonical_reject_write_allowed": False,
    }, "rev22_preregistration_invalid", "stop rules mismatch")
    validate_adapter_contract(value.get("adapter_contract"))
    _require(value.get("claim_limits") == CLAIM_LIMITS and value.get("governance") == GOVERNANCE, "rev22_governance_or_claim_limit_mismatch", "governance/claim limits mismatch")
    return value


def validate_predecessor_value(value: Any, preregistration: Mapping[str, Any]) -> dict[str, Any]:
    reason_schema = "rev21_predecessor_json_or_schema_invalid"
    predecessor = _mapping(value, reason_schema, "rev21 predecessor")
    expected_keys = {"schema_version", "evidence_id", "status", "mode", "predecessor", "evidence_chain", "historical_source_binding", "decision", "checks", "assurance_tiers", "claim_limits", "governance", "execution_metrics", "limitations", "rev21_source_binding", "execution"}
    _require(set(predecessor) == expected_keys, reason_schema, "rev21 predecessor top-level schema mismatch")
    contract = preregistration["predecessor"]
    decision = _mapping(predecessor.get("decision"), "rev21_predecessor_decision_or_governance_mismatch", "rev21 decision")
    _require(predecessor.get("schema_version") == contract["required_schema_version"] and predecessor.get("evidence_id") == contract["required_evidence_id"] and predecessor.get("status") == "complete" and predecessor.get("mode") == "static_bounded_recursive_evidence_gate", reason_schema, "rev21 predecessor identity mismatch")
    _require(decision == {"passed": True, "outcome": contract["required_outcome"], "primary_reason": contract["required_outcome"], "next_step": contract["required_next_step"]} and predecessor.get("governance") == REV21_GOVERNANCE and predecessor.get("claim_limits") == REV21_CLAIM_LIMITS and predecessor.get("execution_metrics") == {"simulator_launched": False, "rollout_steps": 0, "optimizer_updates": 0}, "rev21_predecessor_decision_or_governance_mismatch", "rev21 decision/governance mismatch")
    source = _mapping(predecessor.get("rev21_source_binding"), "rev21_predecessor_decision_or_governance_mismatch", "rev21 source binding")
    _require(source.get("git_commit") == contract["required_source_commit"] and source.get("source_bundle_sha256") == contract["required_source_bundle_sha256"] and source.get("path_scoped_clean") is True, "rev21_predecessor_decision_or_governance_mismatch", "rev21 source binding mismatch")
    checks = predecessor.get("checks")
    _require(isinstance(checks, list) and len(checks) == 17 and all(isinstance(item, Mapping) and item.get("status") == "pass" for item in checks), "rev21_predecessor_decision_or_governance_mismatch", "rev21 check ledger mismatch")
    return dict(predecessor)


def _check(reason: str, status: str, detail: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"reason": reason, "status": status}
    if detail is not None:
        result["detail"] = detail
    return result


def complete_reason_ledger(checks: Sequence[Mapping[str, Any]] | None, primary_reason: str) -> list[dict[str, Any]]:
    _require(primary_reason in REASON_PRIORITY, "rev22_preregistration_invalid", "unknown reason")
    supplied = {cast(str, item.get("reason")): dict(item) for item in (checks or []) if isinstance(item, Mapping) and item.get("reason") in REASON_PRIORITY}
    failure_index = None if primary_reason == PASS_REASON else REASON_PRIORITY.index(primary_reason)
    result: list[dict[str, Any]] = []
    for index, reason in enumerate(REASON_PRIORITY):
        status = "pass" if failure_index is None or index < failure_index else ("fail" if index == failure_index else "not_evaluated")
        detail = supplied.get(reason, {}).get("detail")
        if status == "not_evaluated" and detail is None:
            detail = f"depends on {primary_reason}"
        result.append(_check(reason, status, cast(str | None, detail)))
    return result


def validate_complete_reason_ledger(checks: Any, primary_reason: str) -> list[dict[str, Any]]:
    _require(isinstance(checks, list) and len(checks) == len(REASON_PRIORITY), "rev22_preregistration_invalid", "reason ledger length mismatch")
    expected = complete_reason_ledger(cast(list[Mapping[str, Any]], checks), primary_reason)
    _require(checks == expected, "rev22_preregistration_invalid", "reason ledger order/status mismatch")
    return expected


def _source_binding(value: Any) -> dict[str, Any]:
    source = _mapping(value, "rev22_source_provenance_invalid", "rev22 source binding")
    _require(set(source) == {"schema_version", "git_commit", "source_binding_paths", "source_binding_files", "source_bundle_sha256", "path_scoped_clean"}, "rev22_source_provenance_invalid", "rev22 source binding schema mismatch")
    files = _mapping(source.get("source_binding_files"), "rev22_source_provenance_invalid", "rev22 source files")
    _require(source.get("schema_version") == 1 and isinstance(source.get("git_commit"), str) and re.fullmatch(r"[0-9a-f]{40}", cast(str, source.get("git_commit"))) is not None and source.get("source_binding_paths") == list(REQUIRED_SOURCE_PATHS) and set(files) == set(REQUIRED_SOURCE_PATHS) and all(isinstance(files[path], str) and SHA256_HEX.fullmatch(cast(str, files[path])) is not None for path in REQUIRED_SOURCE_PATHS) and source.get("path_scoped_clean") is True, "rev22_source_provenance_invalid", "rev22 source binding fields mismatch")
    payload = "\n".join(f"{path}:{files[path]}" for path in REQUIRED_SOURCE_PATHS)
    _require(source.get("source_bundle_sha256") == sha256_bytes(payload.encode("utf-8")), "rev22_source_provenance_invalid", "rev22 source bundle digest mismatch")
    return dict(source)


def _rejected(prereg: Mapping[str, Any], reason: str, detail: str) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA, "evidence_id": "G009-5-E015", "status": "rejected",
        "mode": "static_read_only_adapter_preregistration",
        "decision": {"passed": False, "outcome": reason, "primary_reason": reason, "next_step": "stop_and_repair_preregistration"},
        "checks": complete_reason_ledger([_check(reason, "fail", detail)], reason),
        "assurance_tiers": prereg.get("assurance_tiers", {}),
        "claim_limits": prereg.get("claim_limits", {}), "governance": prereg.get("governance", GOVERNANCE),
        "execution_metrics": {"simulator_launched": False, "rollout_steps": 0, "optimizer_updates": 0},
    }


def evaluate_evidence(preregistration_bytes: bytes, predecessor_bytes: bytes, rev22_source_binding: Mapping[str, Any]) -> dict[str, Any]:
    try:
        prereg = validate_preregistration_bytes(preregistration_bytes)
    except GateValidationError as exc:
        return _rejected({}, exc.reason, exc.detail)
    if sha256_bytes(predecessor_bytes) != prereg["predecessor"]["sha256"]:
        return _rejected(prereg, "rev21_predecessor_sha256_mismatch", "rev21 predecessor SHA-256 mismatch before JSON parse")
    try:
        predecessor = strict_json_bytes(predecessor_bytes, "rev21_predecessor_json_or_schema_invalid", "rev21 predecessor")
        validate_predecessor_value(predecessor, prereg)
        source = _source_binding(rev22_source_binding)
    except GateValidationError as exc:
        return _rejected(prereg, exc.reason, exc.detail)
    return {
        "schema_version": ARTIFACT_SCHEMA, "evidence_id": "G009-5-E015", "status": "complete",
        "mode": "static_read_only_adapter_preregistration",
        "predecessor": {"path": prereg["predecessor"]["path"], "sha256": sha256_bytes(predecessor_bytes), "outcome": predecessor["decision"]["outcome"], "next_step": predecessor["decision"]["next_step"], "source_commit": predecessor["rev21_source_binding"]["git_commit"], "source_bundle_sha256": predecessor["rev21_source_binding"]["source_bundle_sha256"], "full_read_only_verification_required": True},
        "adapter_contract_sha256": canonical_sha256(prereg["adapter_contract"]),
        "decision": {"passed": True, "outcome": PASS_REASON, "primary_reason": PASS_REASON, "next_step": NEXT_STEP},
        "checks": complete_reason_ledger([], PASS_REASON),
        "assurance_tiers": prereg["assurance_tiers"], "claim_limits": prereg["claim_limits"], "governance": prereg["governance"],
        "execution_metrics": {"simulator_launched": False, "rollout_steps": 0, "optimizer_updates": 0},
        "runtime_and_throughput_deferred": prereg["runtime_and_throughput_deferred"],
        "source_binding_projection": {"source_bundle_sha256": source["source_bundle_sha256"], "path_scoped_clean": True},
    }


def reference_adapter_projection(source_values: Any, *, source_status: str = "available") -> dict[str, Any]:
    reason_missing = "adapter_numeric_or_missing_contact_contract_invalid"
    _require(source_status == "available" and source_values is not None, reason_missing, "source is missing or unavailable")
    _require(isinstance(source_values, list) and len(source_values) > 0, "adapter_dtype_device_or_shape_invalid", "source environment axis invalid")
    xyz: list[list[list[float]]] = []
    for env in source_values:
        _require(isinstance(env, list) and len(env) == 19, "adapter_dtype_device_or_shape_invalid", "source body axis invalid")
        env_xyz: list[list[float]] = []
        for body in env:
            _require(isinstance(body, list) and len(body) >= 1 and all(isinstance(filter_vector, list) and len(filter_vector) == 3 for filter_vector in body), "adapter_dtype_device_or_shape_invalid", "source filter/xyz shape invalid")
            vector = [0.0, 0.0, 0.0]
            for filter_vector in body:
                for component_index, component in enumerate(filter_vector):
                    _require(_is_number(component), reason_missing, "source component must be finite non-bool")
                    vector[component_index] += float(component)
            env_xyz.append(vector)
        xyz.append(env_xyz)
    magnitude = [[math.sqrt(math.fsum(component * component for component in vector)) for vector in env] for env in xyz]
    mask = [[value > 0.000001 for value in env] for env in magnitude]
    return {"world_xyz": xyz, "magnitude": magnitude, "mask": mask}


def validate_adapter_fixture(value: Any) -> dict[str, Any]:
    fixture = _mapping(value, "adapter_representation_contract_invalid", "adapter fixture")
    _require(set(fixture) == {"source_status", "source_values", "source_dtype", "source_device", "source_shape", "source_before", "source_after", "world_xyz", "magnitude", "mask", "output_dtype", "mask_dtype", "output_device", "world_xyz_shape", "magnitude_shape", "mask_shape", "output_storage_ids", "outputs_alias_source", "normalization_applied", "clipping_applied"}, "adapter_representation_contract_invalid", "adapter fixture schema mismatch")
    _require(
        fixture.get("source_dtype") == fixture.get("output_dtype") == "torch.float32"
        and fixture.get("mask_dtype") == "torch.bool"
        and fixture.get("source_device") in {"cpu", "cuda:0"}
        and fixture.get("source_device") == fixture.get("output_device"),
        "adapter_dtype_device_or_shape_invalid",
        "dtype/device preservation mismatch",
    )
    source = fixture.get("source_values")
    reference = reference_adapter_projection(source, source_status=cast(str, fixture.get("source_status")))
    source_envs = cast(list[Any], source)
    env_count = len(source_envs)
    _require(
        all(len(cast(list[Any], body)) == 1 for env in source_envs for body in cast(list[Any], env)),
        "adapter_dtype_device_or_shape_invalid",
        "runtime fixture source filter axis must contain exactly one filter",
    )
    _require(fixture.get("source_shape") == [env_count, 19, 1, 3] and fixture.get("world_xyz_shape") == [env_count, 19, 3] and fixture.get("magnitude_shape") == [env_count, 19] and fixture.get("mask_shape") == [env_count, 19], "adapter_dtype_device_or_shape_invalid", "adapter fixture shape mismatch")
    _require(fixture.get("world_xyz") == reference["world_xyz"] and fixture.get("magnitude") == reference["magnitude"] and fixture.get("mask") == reference["mask"], "adapter_filter_reduction_invalid", "adapter output differs from sum-before-norm oracle")
    before = _mapping(fixture.get("source_before"), "adapter_source_immutability_contract_invalid", "source before")
    after = _mapping(fixture.get("source_after"), "adapter_source_immutability_contract_invalid", "source after")
    snapshot_keys = {"shape", "dtype", "device", "stride", "storage_id", "version", "sha256"}
    _require(set(before) == snapshot_keys and set(after) == snapshot_keys, "adapter_source_immutability_contract_invalid", "source snapshot schema mismatch")
    _require(
        before.get("shape") == fixture.get("source_shape")
        and before.get("dtype") == fixture.get("source_dtype")
        and before.get("device") == fixture.get("source_device")
        and isinstance(before.get("stride"), list)
        and len(cast(list[Any], before.get("stride"))) == 4
        and all(type(item) is int and item >= 0 for item in cast(list[Any], before.get("stride")))
        and isinstance(before.get("storage_id"), str)
        and bool(before.get("storage_id"))
        and type(before.get("version")) is int
        and cast(int, before.get("version")) >= 0
        and isinstance(before.get("sha256"), str)
        and SHA256_HEX.fullmatch(cast(str, before.get("sha256"))) is not None,
        "adapter_source_immutability_contract_invalid",
        "source snapshot metadata mismatch",
    )
    output_storage_ids = _mapping(fixture.get("output_storage_ids"), "adapter_source_immutability_contract_invalid", "output storage IDs")
    _require(
        set(output_storage_ids) == {"world_xyz", "magnitude", "mask"}
        and all(isinstance(output_storage_ids.get(key), str) and bool(output_storage_ids.get(key)) for key in output_storage_ids)
        and len(set(output_storage_ids.values())) == 3
        and before.get("storage_id") not in set(output_storage_ids.values()),
        "adapter_source_immutability_contract_invalid",
        "output storage identity aliases source or another output",
    )
    _require(before == after and before.get("sha256") == canonical_sha256(source) and fixture.get("outputs_alias_source") is False, "adapter_source_immutability_contract_invalid", "source mutation or output alias detected")
    _require(fixture.get("normalization_applied") is False and fixture.get("clipping_applied") is False, "adapter_normalization_or_clipping_contract_invalid", "normalization or clipping was applied")
    return dict(fixture)


def artifact_deterministic_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {"execution", "rev22_source_binding"}}


def _validate_execution(value: Any) -> None:
    execution = _mapping(value, "rev22_preregistration_invalid", "artifact execution")
    _require(set(execution) == {"execution_id", "started_at_utc", "output_path_repo_relative", "no_overwrite"} and execution.get("output_path_repo_relative") == "reports/runs/g009_r0_rev22_read_only_matrix_observation_adapter_preregistration_s42.json" and execution.get("no_overwrite") is True, "rev22_preregistration_invalid", "artifact execution schema mismatch")
    execution_id = execution.get("execution_id")
    _require(isinstance(execution_id, str) and UUID4_HEX.fullmatch(execution_id) is not None and uuid.UUID(hex=execution_id).version == 4, "rev22_preregistration_invalid", "artifact execution UUID mismatch")
    timestamp = execution.get("started_at_utc")
    _require(isinstance(timestamp, str) and RFC3339_UTC.fullmatch(timestamp) is not None, "rev22_preregistration_invalid", "artifact timestamp must be RFC3339 UTC Z")
    _require(datetime.fromisoformat(cast(str, timestamp).replace("Z", "+00:00")).tzinfo == timezone.utc, "rev22_preregistration_invalid", "artifact timestamp is not UTC")


def validate_artifact_value(value: Mapping[str, Any], expected_projection: Mapping[str, Any], rev22_source_binding: Mapping[str, Any]) -> dict[str, Any]:
    _require(artifact_deterministic_projection(value) == dict(expected_projection), "rev22_source_provenance_invalid", "artifact deterministic projection mismatch")
    expected_source = _source_binding(rev22_source_binding)
    _require(value.get("rev22_source_binding") == expected_source, "rev22_source_provenance_invalid", "artifact source binding mismatch")
    _validate_execution(value.get("execution"))
    validate_complete_reason_ledger(value.get("checks"), PASS_REASON)
    return dict(value)


def _load_rev21_verifier() -> Any:
    spec = importlib.util.spec_from_file_location("g009_rev21_verifier", REV21_VERIFIER_PATH)
    _require(spec is not None and spec.loader is not None, "rev21_predecessor_full_verification_failed", "cannot load rev21 verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_predecessor_fresh(path: Path = PREDECESSOR_PATH) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], _load_rev21_verifier().verify_artifact(path))
    except Exception as exc:
        raise GateValidationError("rev21_predecessor_full_verification_failed", str(exc)) from exc


def _git_bytes(args: list[str]) -> bytes:
    git_environment = dict(os.environ)
    git_environment.update({"LC_ALL": "C", "LANG": "C"})
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            env=git_environment,
        ).stdout
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        lowered = detail.lower()
        missing_markers = (
            "does not exist",
            "not in the working tree",
            "invalid object name",
            "bad object",
            "unknown revision",
            "ambiguous argument",
        )
        if any(marker in lowered for marker in missing_markers):
            raise GateValidationError("rev22_source_provenance_invalid", detail or f"Git object is unavailable: {' '.join(args)}") from exc
        raise OperationalVerificationError("runner_input_io_error", detail or f"Git command failed: {' '.join(args)}") from exc
    except OSError as exc:
        raise OperationalVerificationError("runner_input_io_error", f"cannot execute Git: {exc}") from exc


def _git_blob(commit: str, path: str) -> bytes:
    return _git_bytes(["show", f"{commit}:{path}"])


def _fresh_source_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    commit = value.get("git_commit")
    _require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "rev22_source_provenance_invalid", "rev22 commit malformed")
    dirty = _git_bytes(["status", "--porcelain=v1", "--untracked-files=all", "--", *REQUIRED_SOURCE_PATHS]).decode("utf-8", errors="replace").splitlines()
    _require(not dirty, "rev22_source_provenance_invalid", "bound source paths are not path-scoped clean: " + "; ".join(dirty))
    files: dict[str, str] = {}
    for path in REQUIRED_SOURCE_PATHS:
        blob = _git_blob(cast(str, commit), path)
        try:
            worktree = (REPO_ROOT / path).read_bytes()
        except FileNotFoundError as exc:
            raise GateValidationError("rev22_source_provenance_invalid", f"bound source path is missing: {path}") from exc
        except OSError as exc:
            raise OperationalVerificationError("runner_input_io_error", f"cannot read bound source path {path}: {exc}") from exc
        _require(worktree == blob, "rev22_source_provenance_invalid", f"worktree bytes differ from bound commit blob: {path}")
        files[path] = sha256_bytes(blob)
    payload = "\n".join(f"{path}:{files[path]}" for path in REQUIRED_SOURCE_PATHS)
    fresh = {"schema_version": 1, "git_commit": commit, "source_binding_paths": list(REQUIRED_SOURCE_PATHS), "source_binding_files": files, "source_bundle_sha256": sha256_bytes(payload.encode("utf-8")), "path_scoped_clean": True}
    _require(dict(value) == fresh, "rev22_source_provenance_invalid", "fresh rev22 source binding mismatch")
    return fresh


def verify_artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved == CANONICAL_ARTIFACT_PATH.resolve(), "canonical_output_path_invalid", "artifact path is not canonical")
    artifact = strict_json_bytes(resolved.read_bytes(), "rev22_preregistration_invalid", "rev22 artifact")
    prereg_raw = PREREGISTRATION_PATH.read_bytes()
    predecessor_raw = PREDECESSOR_PATH.read_bytes()
    source = _mapping(artifact.get("rev22_source_binding"), "rev22_source_provenance_invalid", "artifact source binding")
    fresh_source = _fresh_source_binding(source)
    verify_predecessor_fresh(PREDECESSOR_PATH)
    expected = evaluate_evidence(prereg_raw, predecessor_raw, fresh_source)
    _require(expected.get("decision", {}).get("passed") is True, cast(str, expected.get("decision", {}).get("primary_reason")), "fresh evidence rejected")
    return validate_artifact_value(artifact, expected, fresh_source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = verify_artifact(args.verify_artifact)
    except GateValidationError as exc:
        print(json.dumps({"verified": False, "primary_reason": exc.reason, "detail": exc.detail}, ensure_ascii=False), file=sys.stderr)
        return 2
    except OperationalVerificationError as exc:
        print(json.dumps({"verified": False, "primary_reason": exc.reason, "detail": exc.detail}, ensure_ascii=False), file=sys.stderr)
        return 3
    except OSError as exc:
        print(json.dumps({"verified": False, "primary_reason": "runner_input_io_error", "detail": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 3
    print(json.dumps({"verified": True, "outcome": value["decision"]["outcome"], "path": str(args.verify_artifact)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
