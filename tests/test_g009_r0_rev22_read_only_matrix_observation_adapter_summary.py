from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "scripts/summarize_g009_r0_rev22_read_only_matrix_observation_adapter.py"
CONFIG_PATH = ROOT / "configs/g009_r0_rev22_read_only_matrix_observation_adapter.json"
PREDECESSOR_PATH = ROOT / "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json"
SPEC = importlib.util.spec_from_file_location("rev22_summary", SUMMARY_PATH)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def strict_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def fake_source_binding() -> dict[str, object]:
    files = {path: SUMMARY.sha256_bytes(f"blob:{path}".encode()) for path in SUMMARY.REQUIRED_SOURCE_PATHS}
    payload = "\n".join(f"{path}:{files[path]}" for path in SUMMARY.REQUIRED_SOURCE_PATHS)
    return {"schema_version": 1, "git_commit": "b" * 40, "source_binding_paths": list(SUMMARY.REQUIRED_SOURCE_PATHS), "source_binding_files": files, "source_bundle_sha256": SUMMARY.sha256_bytes(payload.encode()), "path_scoped_clean": True}


@pytest.fixture(scope="module")
def canonical() -> dict[str, object]:
    prereg_raw = CONFIG_PATH.read_bytes()
    return {
        "prereg_raw": prereg_raw,
        "prereg": SUMMARY.validate_preregistration_bytes(prereg_raw),
        "predecessor_raw": PREDECESSOR_PATH.read_bytes(),
        "source": fake_source_binding(),
    }


def evaluate(values: dict[str, object], **overrides: object) -> dict:
    data = {**values, **overrides}
    return SUMMARY.evaluate_evidence(data["prereg_raw"], data["predecessor_raw"], data["source"])


def test_canonical_evidence_passes_with_exact_eighteen_reason_ledger(canonical: dict[str, object]) -> None:
    result = evaluate(canonical)
    assert result["decision"] == {"passed": True, "outcome": SUMMARY.PASS_REASON, "primary_reason": SUMMARY.PASS_REASON, "next_step": SUMMARY.NEXT_STEP}
    assert [item["reason"] for item in result["checks"]] == list(SUMMARY.REASON_PRIORITY)
    assert [item["status"] for item in result["checks"]] == ["pass"] * 18
    assert result["claim_limits"]["simulator_launched"] is False
    assert result["claim_limits"]["rollout_steps"] == result["claim_limits"]["optimizer_updates"] == 0


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(extra=True),
    lambda value: value.update(goal_id="g008"),
    lambda value: value["decision"]["reason_priority"].reverse(),
    lambda value: value["claim_limits"].update(rollout_steps=1),
])
def test_preregistration_exact_schema_rejects_mutations(mutation) -> None:
    value = json.loads(CONFIG_PATH.read_bytes())
    mutation(value)
    with pytest.raises(SUMMARY.GateValidationError):
        SUMMARY.validate_preregistration_bytes(strict_bytes(value))


def test_duplicate_key_and_nonfinite_json_rejected() -> None:
    with pytest.raises(SUMMARY.GateValidationError) as duplicate:
        SUMMARY.validate_preregistration_bytes(b'{"schema_version":"a","schema_version":"b"}')
    assert duplicate.value.reason == "rev22_preregistration_invalid"
    value = CONFIG_PATH.read_bytes().replace(b'"seed": 42', b'"seed": NaN')
    with pytest.raises(SUMMARY.GateValidationError) as nonfinite:
        SUMMARY.validate_preregistration_bytes(value)
    assert nonfinite.value.reason == "rev22_preregistration_invalid"


def test_predecessor_sha_is_checked_before_json(canonical: dict[str, object]) -> None:
    result = evaluate(canonical, predecessor_raw=canonical["predecessor_raw"] + b"\nnot-json")
    assert result["decision"]["primary_reason"] == "rev21_predecessor_sha256_mismatch"
    SUMMARY.validate_complete_reason_ledger(result["checks"], "rev21_predecessor_sha256_mismatch")


@pytest.mark.parametrize(("mutation", "reason"), [
    (lambda value: value["decision"].update(next_step="wrong"), "rev21_predecessor_decision_or_governance_mismatch"),
    (lambda value: value["governance"].update(learned=True), "rev21_predecessor_decision_or_governance_mismatch"),
    (lambda value: value["rev21_source_binding"].update(source_bundle_sha256="0" * 64), "rev21_predecessor_decision_or_governance_mismatch"),
])
def test_lower_predecessor_semantics_reject_mutations(canonical: dict[str, object], mutation, reason: str) -> None:
    value = json.loads(canonical["predecessor_raw"])
    mutation(value)
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.validate_predecessor_value(value, canonical["prereg"])
    assert error.value.reason == reason


@pytest.mark.parametrize(("mutation", "reason"), [
    (lambda value: value.update(authority_role="magnitude_policy_input"), "adapter_representation_contract_invalid"),
    (lambda value: value["source"].update(quantity_semantics="total_contact_force_vector"), "adapter_representation_contract_invalid"),
    (lambda value: value["source"].update(tangential_friction_force_included=True), "adapter_representation_contract_invalid"),
    (lambda value: value["source"].update(coordinate_frame="body"), "adapter_coordinate_or_axis_order_invalid"),
    (lambda value: value["source"].update(dtype="torch.float64"), "adapter_dtype_device_or_shape_invalid"),
    (lambda value: value["world_xyz_output"].update(formula="norm_then_sum"), "adapter_filter_reduction_invalid"),
    (lambda value: value["normalization_and_clipping"].update(clipping="[-1,1]"), "adapter_normalization_or_clipping_contract_invalid"),
    (lambda value: value["source_immutability"].update(in_place_operation_allowed=True), "adapter_source_immutability_contract_invalid"),
])
def test_adapter_contract_reason_classification(canonical: dict[str, object], mutation, reason: str) -> None:
    contract = copy.deepcopy(canonical["prereg"]["adapter_contract"])
    mutation(contract)
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.validate_adapter_contract(contract)
    assert error.value.reason == reason


def test_force_quantity_semantics_exclude_tangential_friction_measurement(canonical: dict[str, object]) -> None:
    contract = canonical["prereg"]["adapter_contract"]
    source = contract["source"]
    claims = contract["claim_limits"]
    assert source["quantity_semantics"] == "filtered_normal_contact_force_vector"
    assert source["total_contact_force_included"] is False
    assert source["tangential_friction_force_included"] is False
    assert source["friction_effect_directly_observed"] is False
    assert claims["normal_contact_force_vector_claim_allowed"] is True
    assert claims["total_contact_force_claim_allowed"] is False
    assert claims["tangential_friction_force_claim_allowed"] is False
    assert claims["friction_effect_directly_observed_claim_allowed"] is False


def source_with_first_body(filters: list[list[float]]) -> list:
    zero = [[[0.0, 0.0, 0.0]] for _ in range(19)]
    zero[0] = filters
    return [zero]


def test_reference_oracle_sums_filters_before_norm() -> None:
    result = SUMMARY.reference_adapter_projection(source_with_first_body([[3.0, 0.0, 0.0], [-3.0, 4.0, 0.0]]))
    assert result["world_xyz"][0][0] == [0.0, 4.0, 0.0]
    assert result["magnitude"][0][0] == 4.0
    assert result["magnitude"][0][0] != 8.0


def test_reference_oracle_zero_threshold_and_signed_xyz() -> None:
    values = source_with_first_body([[-0.000001, 0.0, 0.0]])
    result = SUMMARY.reference_adapter_projection(values)
    assert result["world_xyz"][0][0] == [-0.000001, 0.0, 0.0]
    assert result["magnitude"][0][0] == 0.000001
    assert result["mask"][0][0] is False
    assert all(item is False for item in result["mask"][0][1:])


@pytest.mark.parametrize(("source", "status", "reason"), [
    (None, "available", "adapter_numeric_or_missing_contact_contract_invalid"),
    (source_with_first_body([[0.0, 0.0, 0.0]]), "unavailable", "adapter_numeric_or_missing_contact_contract_invalid"),
    (source_with_first_body([[True, 0.0, 0.0]]), "available", "adapter_numeric_or_missing_contact_contract_invalid"),
])
def test_missing_or_invalid_source_fails_closed(source, status: str, reason: str) -> None:
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.reference_adapter_projection(source, source_status=status)
    assert error.value.reason == reason


def canonical_fixture() -> dict[str, object]:
    source = source_with_first_body([[3.0, 4.0, 0.0]])
    projection = SUMMARY.reference_adapter_projection(source)
    snapshot = {"shape": [1, 19, 1, 3], "dtype": "torch.float32", "device": "cuda:0", "stride": [57, 3, 3, 1], "storage_id": "source-1", "version": 0, "sha256": SUMMARY.canonical_sha256(source)}
    return {"source_status": "available", "source_values": source, "source_dtype": "torch.float32", "source_device": "cuda:0", "source_shape": [1, 19, 1, 3], "source_before": snapshot, "source_after": copy.deepcopy(snapshot), "world_xyz": projection["world_xyz"], "magnitude": projection["magnitude"], "mask": projection["mask"], "output_dtype": "torch.float32", "mask_dtype": "torch.bool", "output_device": "cuda:0", "world_xyz_shape": [1, 19, 3], "magnitude_shape": [1, 19], "mask_shape": [1, 19], "output_storage_ids": {"world_xyz": "output-xyz", "magnitude": "output-magnitude", "mask": "output-mask"}, "outputs_alias_source": False, "normalization_applied": False, "clipping_applied": False}


def test_runtime_fixture_contract_passes() -> None:
    assert SUMMARY.validate_adapter_fixture(canonical_fixture())["outputs_alias_source"] is False


@pytest.mark.parametrize(("mutation", "reason"), [
    (lambda value: value.update(magnitude=[[8.0] + [0.0] * 18]), "adapter_filter_reduction_invalid"),
    (lambda value: value.update(output_dtype="torch.float64"), "adapter_dtype_device_or_shape_invalid"),
    (lambda value: value.update(mask_dtype="torch.float32"), "adapter_dtype_device_or_shape_invalid"),
    (lambda value: value.update(outputs_alias_source=True), "adapter_source_immutability_contract_invalid"),
    (lambda value: value["output_storage_ids"].update(world_xyz="source-1"), "adapter_source_immutability_contract_invalid"),
    (lambda value: value["source_after"].update(version=1), "adapter_source_immutability_contract_invalid"),
    (lambda value: value.update(normalization_applied=True), "adapter_normalization_or_clipping_contract_invalid"),
])
def test_runtime_fixture_mutations_reject(mutation, reason: str) -> None:
    value = canonical_fixture()
    mutation(value)
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.validate_adapter_fixture(value)
    assert error.value.reason == reason


def test_runtime_fixture_rejects_nested_filter_count_disagreeing_with_declared_shape() -> None:
    value = canonical_fixture()
    value["source_values"][0][0] = [[3.0, 0.0, 0.0], [-3.0, 4.0, 0.0]]
    value["source_before"]["sha256"] = SUMMARY.canonical_sha256(value["source_values"])
    value["source_after"]["sha256"] = value["source_before"]["sha256"]
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.validate_adapter_fixture(value)
    assert error.value.reason == "adapter_dtype_device_or_shape_invalid"


def test_preregistration_structural_failure_precedes_adapter_semantic_failure() -> None:
    value = json.loads(CONFIG_PATH.read_bytes())
    value["decision"]["reason_priority"].reverse()
    value["adapter_contract"]["world_xyz_output"]["formula"] = "norm_then_sum"
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.validate_preregistration_bytes(strict_bytes(value))
    assert error.value.reason == "rev22_preregistration_invalid"


@pytest.mark.parametrize(
    ("first_mutation", "second_mutation", "expected_reason"),
    [
        (
            lambda value: value.update(authority_role="magnitude_policy_input"),
            lambda value: value["source"].update(coordinate_frame="body"),
            "adapter_representation_contract_invalid",
        ),
        (
            lambda value: value["source"].update(coordinate_frame="body"),
            lambda value: value["world_xyz_output"].update(formula="norm_then_sum"),
            "adapter_coordinate_or_axis_order_invalid",
        ),
        (
            lambda value: value["world_xyz_output"].update(formula="norm_then_sum"),
            lambda value: value["source"].update(dtype="torch.float64"),
            "adapter_filter_reduction_invalid",
        ),
        (
            lambda value: value["source"].update(dtype="torch.float64"),
            lambda value: value["missing_contact"].update(none_source="zero_fill"),
            "adapter_dtype_device_or_shape_invalid",
        ),
        (
            lambda value: value["missing_contact"].update(none_source="zero_fill"),
            lambda value: value["normalization_and_clipping"].update(clipping="[-1,1]"),
            "adapter_numeric_or_missing_contact_contract_invalid",
        ),
        (
            lambda value: value["normalization_and_clipping"].update(clipping="[-1,1]"),
            lambda value: value["source_immutability"].update(in_place_operation_allowed=True),
            "adapter_normalization_or_clipping_contract_invalid",
        ),
        (
            lambda value: value["source_immutability"].update(in_place_operation_allowed=True),
            lambda value: value["claim_limits"].update(policy_input_claim_allowed=True),
            "adapter_source_immutability_contract_invalid",
        ),
    ],
)
def test_adapter_contract_combined_faults_follow_reason_priority(
    canonical: dict[str, object],
    first_mutation,
    second_mutation,
    expected_reason: str,
) -> None:
    contract = copy.deepcopy(canonical["prereg"]["adapter_contract"])
    first_mutation(contract)
    second_mutation(contract)
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.validate_adapter_contract(contract)
    assert error.value.reason == expected_reason


def test_artifact_projection_and_execution_validation(canonical: dict[str, object]) -> None:
    projection = evaluate(canonical)
    execution = {"execution_id": "1234567890ab4def81234567890abcde", "started_at_utc": "2026-08-31T00:00:00Z", "output_path_repo_relative": "reports/runs/g009_r0_rev22_read_only_matrix_observation_adapter_preregistration_s42.json", "no_overwrite": True}
    artifact = {**projection, "rev22_source_binding": canonical["source"], "execution": execution}
    assert SUMMARY.validate_artifact_value(artifact, projection, canonical["source"])["status"] == "complete"
    artifact["execution"] = {**execution, "extra": True}
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.validate_artifact_value(artifact, projection, canonical["source"])
    assert error.value.reason == "rev22_preregistration_invalid"


def test_fresh_source_binding_rejects_dirty_bound_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    source = fake_source_binding()

    def dirty_git(args: list[str]) -> bytes:
        if args and args[0] == "status":
            return b" M scripts/summarize_g009_r0_rev21_matrix_authority_safety_gate.py\n"
        raise AssertionError("blob reads must not happen after dirty status")

    monkeypatch.setattr(SUMMARY, "_git_bytes", dirty_git)
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY._fresh_source_binding(source)
    assert error.value.reason == "rev22_source_provenance_invalid"


def test_git_missing_object_is_contract_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    error = SUMMARY.subprocess.CalledProcessError(
        128,
        ["git", "show"],
        stderr=b"fatal: path 'missing.py' does not exist in 'deadbeef'",
    )
    monkeypatch.setattr(SUMMARY.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(SUMMARY.GateValidationError) as raised:
        SUMMARY._git_bytes(["show", "deadbeef:missing.py"])
    assert raised.value.reason == "rev22_source_provenance_invalid"


@pytest.mark.parametrize(
    "failure",
    [
        OSError("git executable unavailable"),
        None,
    ],
)
def test_git_operational_failures_are_exit_three_class(
    monkeypatch: pytest.MonkeyPatch,
    failure: OSError | None,
) -> None:
    injected = failure or SUMMARY.subprocess.CalledProcessError(
        128,
        ["git", "show"],
        stderr=b"fatal: permission denied",
    )
    monkeypatch.setattr(SUMMARY.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(injected))
    with pytest.raises(SUMMARY.OperationalVerificationError) as raised:
        SUMMARY._git_bytes(["show", "deadbeef:path.py"])
    assert raised.value.reason == "runner_input_io_error"
