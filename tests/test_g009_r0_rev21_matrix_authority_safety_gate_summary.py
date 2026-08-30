from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "scripts/summarize_g009_r0_rev21_matrix_authority_safety_gate.py"
SPEC = importlib.util.spec_from_file_location("rev21_summary", SUMMARY_PATH)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)

PREREGISTRATION_PATH = ROOT / "configs/g009_r0_rev21_matrix_authority_safety_gate.json"
SYNTHESIS_PATH = ROOT / "reports/runs/g009_r0_rev20_terrain_contact_matrix_synthesis_2x2_s42.json"


def strict_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def git_blob(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


@pytest.fixture(scope="module")
def canonical_inputs() -> dict[str, object]:
    preregistration_bytes = PREREGISTRATION_PATH.read_bytes()
    preregistration = SUMMARY.validate_preregistration_bytes(preregistration_bytes)
    synthesis_bytes = SYNTHESIS_PATH.read_bytes()
    synthesis = json.loads(synthesis_bytes)
    raw_report_bytes = {
        binding["path"]: (ROOT / binding["path"]).read_bytes()
        for binding in synthesis["input_reports"]
    }
    preflight_path = preregistration["predecessor"]["cpu_preflight_path"]
    historical_commit = preregistration["historical_source_binding"]["commit"]
    historical_blobs = {
        path: git_blob(historical_commit, path)
        for path in preregistration["historical_source_binding"]["ordered_unique_paths"]
    }
    return {
        "preregistration_bytes": preregistration_bytes,
        "preregistration": preregistration,
        "synthesis_bytes": synthesis_bytes,
        "raw_report_bytes": raw_report_bytes,
        "cpu_preflight_bytes": (ROOT / preflight_path).read_bytes(),
        "historical_blobs": historical_blobs,
    }


def evaluate(inputs: dict[str, object], **overrides: object) -> dict:
    values = {**inputs, **overrides}
    return SUMMARY.evaluate_evidence(
        values["preregistration_bytes"],
        values["synthesis_bytes"],
        values["raw_report_bytes"],
        values["cpu_preflight_bytes"],
        values["historical_blobs"],
    )


def reason(result: dict) -> str:
    decision = result["decision"]
    return decision.get("primary_reason", decision.get("outcome"))


def raw_reason(result: object) -> str:
    if isinstance(result, dict):
        return result.get("primary_reason", result.get("reason", result.get("outcome", "")))
    return str(result)


def test_preregistration_locks_ordered_source_provenance_without_self_digest() -> None:
    preregistration = SUMMARY.validate_preregistration_bytes(PREREGISTRATION_PATH.read_bytes())
    binding = preregistration["rev21_source_binding"]
    assert binding["ordered_paths"] == [
        "configs/g009_r0_rev21_matrix_authority_safety_gate.json",
        "scripts/summarize_g009_r0_rev21_matrix_authority_safety_gate.py",
        "scripts/run_g009_r0_rev21_matrix_authority_safety_gate.py",
    ]
    assert binding["expected_aggregate_sha256"] is None
    assert len(preregistration["historical_source_binding"]["ordered_unique_paths"]) == 12


def test_preregistration_rejects_an_extra_top_level_key() -> None:
    value = json.loads(PREREGISTRATION_PATH.read_bytes())
    value["unexpected"] = True
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.validate_preregistration_bytes(strict_bytes(value))
    assert error.value.reason == "rev21_preregistration_invalid"


@pytest.mark.parametrize(("field", "replacement"), [("goal_id", "g008"), ("stage_id", "R1")])
def test_preregistration_rejects_goal_or_stage_identity_drift(field: str, replacement: str) -> None:
    value = json.loads(PREREGISTRATION_PATH.read_bytes())
    value[field] = replacement
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.validate_preregistration_bytes(strict_bytes(value))
    assert error.value.reason == "rev21_preregistration_invalid"


def test_canonical_evidence_passes_with_diagnostic_only_claims(canonical_inputs: dict[str, object]) -> None:
    result = evaluate(canonical_inputs)
    assert reason(result) == "matrix_authority_safety_gate_passed_for_diagnostic_preregistration"
    assert result["decision"]["next_step"] == "preregister_read_only_matrix_observation_adapter"
    assert result["governance"] == canonical_inputs["preregistration"]["governance"]
    assert result["claim_limits"]["simulator_launched"] is False
    assert result["claim_limits"]["rollout_steps"] == 0
    assert result["claim_limits"]["optimizer_updates"] == 0
    assert set(result["assurance_tiers"]) == {
        "byte_and_provenance_verified",
        "persisted_evidence_internally_recomputed",
        "physics_execution_not_independently_reobserved",
    }


def test_pass_projection_records_all_seventeen_checks_in_priority_order(
    canonical_inputs: dict[str, object]
) -> None:
    result = evaluate(canonical_inputs)
    assert [item["reason"] for item in result["checks"]] == list(SUMMARY.REASON_PRIORITY)
    assert [item["status"] for item in result["checks"]] == ["pass"] * 17


def test_outer_anchor_rejects_changed_synthesis_before_json_semantics(canonical_inputs: dict[str, object]) -> None:
    malformed = canonical_inputs["synthesis_bytes"] + b"\nnot-json"
    result = evaluate(canonical_inputs, synthesis_bytes=malformed)
    assert reason(result) == "rev20_synthesis_sha256_mismatch"


def test_rejected_projection_preserves_ordered_list_check_ledger(canonical_inputs: dict[str, object]) -> None:
    result = evaluate(canonical_inputs, synthesis_bytes=canonical_inputs["synthesis_bytes"] + b"\n")
    assert isinstance(result["checks"], list)
    assert [item["reason"] for item in result["checks"]] == list(SUMMARY.REASON_PRIORITY)
    assert result["checks"][5]["status"] == "fail"
    assert all(item["status"] == "not_evaluated" for item in result["checks"][6:])


def test_outer_anchor_rejects_missing_historical_blob_before_runtime_semantics(canonical_inputs: dict[str, object]) -> None:
    blobs = dict(canonical_inputs["historical_blobs"])
    blobs.pop("configs/g009_r0_rev20_terrain_contact_matrix.json")
    reports = dict(canonical_inputs["raw_report_bytes"])
    first_path = next(iter(reports))
    changed = json.loads(reports[first_path])
    changed["headless"] = False
    reports[first_path] = strict_bytes(changed)
    result = evaluate(canonical_inputs, historical_blobs=blobs, raw_report_bytes=reports)
    assert reason(result) == "rev20_evidence_chain_binding_invalid"


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (lambda value: value.update(headless=False), "rev20_baseline_or_runtime_drift"),
        (
            lambda value: value["terrain_contact_matrix"]["shapes"].update(raw=[151, 1, 3]),
            "rev20_matrix_shape_or_order_invalid",
        ),
        (
            lambda value: value["terrain_contact_matrix"]["safety"]["mass_tensor"]["values"][0].__setitem__(0, -1.0),
            "rev20_matrix_numeric_invalid",
        ),
        (
            lambda value: value["terrain_contact_matrix"]["safety"]["non_foot_peak_force_body_weight_per_env"].__setitem__(0, 15.000001),
            "rev20_matrix_physics_limit_exceeded",
        ),
    ],
)
def test_lower_level_raw_report_validator_classifies_semantic_mutations(
    canonical_inputs: dict[str, object], mutation, expected_reason: str
) -> None:
    preregistration = canonical_inputs["preregistration"]
    historical_preregistration = json.loads(
        canonical_inputs["historical_blobs"]["configs/g009_r0_rev20_terrain_contact_matrix.json"]
    )
    raw = next(iter(canonical_inputs["raw_report_bytes"].values()))
    value = json.loads(raw)
    mutation(value)
    result = SUMMARY.validate_raw_report(strict_bytes(value), preregistration, historical_preregistration)
    assert raw_reason(result) == expected_reason


def test_lower_level_raw_report_rejects_nonfinite_json_token(canonical_inputs: dict[str, object]) -> None:
    preregistration = canonical_inputs["preregistration"]
    historical_preregistration = json.loads(
        canonical_inputs["historical_blobs"]["configs/g009_r0_rev20_terrain_contact_matrix.json"]
    )
    raw = next(iter(canonical_inputs["raw_report_bytes"].values()))
    value = json.loads(raw)
    encoded = strict_bytes(value).replace(b'"physics_dt_s":0.005', b'"physics_dt_s":NaN', 1)
    result = SUMMARY.validate_raw_report(encoded, preregistration, historical_preregistration)
    assert raw_reason(result) == "rev20_matrix_numeric_invalid"


@pytest.mark.parametrize(
    ("report_index", "mutation", "expected_reason"),
    [
        (0, lambda report: report["cpu_preflight_binding"].update(status="validated_for_gpu"), "rev20_evidence_chain_binding_invalid"),
        (2, lambda report: report["cpu_preflight_binding"].update(sha256="0" * 64), "rev20_evidence_chain_binding_invalid"),
        (0, lambda report: report["terrain_filter"].update(fallback_used=True), "rev20_matrix_shape_or_order_invalid"),
        (0, lambda report: report["terrain_filter"].update(filter_paths_sha256="0" * 64), "rev20_matrix_shape_or_order_invalid"),
        (0, lambda report: report["terrain_contact_matrix"]["path_order"].update(sensor_paths_sha256="0" * 64), "rev20_matrix_shape_or_order_invalid"),
        (0, lambda report: report["terrain_contact_matrix"]["path_order"].update(raw_filter_paths_sha256="0" * 64), "rev20_matrix_shape_or_order_invalid"),
        (0, lambda report: report["terrain_contact_matrix"]["path_order"].update(force_body_names_sha256="0" * 64), "rev20_matrix_shape_or_order_invalid"),
        (0, lambda report: report["terrain_contact_matrix"]["path_order"]["sensor_paths"].reverse(), "rev20_matrix_shape_or_order_invalid"),
        (0, lambda report: report["external_source_binding"].update(all_hashes_match=False), "rev20_source_provenance_invalid"),
    ],
)
def test_lower_level_raw_report_rejects_binding_filter_hash_mapping_and_external_drift(
    canonical_inputs: dict[str, object], report_index: int, mutation, expected_reason: str
) -> None:
    preregistration = canonical_inputs["preregistration"]
    historical_preregistration = canonical_inputs["historical_blobs"][
        "configs/g009_r0_rev20_terrain_contact_matrix.json"
    ]
    raw_values = list(canonical_inputs["raw_report_bytes"].values())
    value = json.loads(raw_values[report_index])
    mutation(value)
    result = SUMMARY.validate_raw_report(strict_bytes(value), preregistration, historical_preregistration)
    assert result["passed"] is False
    assert result["primary_reason"] == expected_reason


@pytest.mark.parametrize(
    ("field", "replacement", "expected_reason"),
    [
        ("execution", [], "rev20_execution_identity_invalid"),
        ("baseline_snapshot", [], "rev20_baseline_or_runtime_drift"),
        ("device_readback", [], "rev20_matrix_shape_or_order_invalid"),
        ("terrain_contact_matrix.path_order", [], "rev20_matrix_shape_or_order_invalid"),
    ],
)
def test_nested_wrong_types_return_stable_gate_result_instead_of_raising(
    canonical_inputs: dict[str, object], field: str, replacement: object, expected_reason: str
) -> None:
    preregistration = canonical_inputs["preregistration"]
    historical_preregistration = canonical_inputs["historical_blobs"][
        "configs/g009_r0_rev20_terrain_contact_matrix.json"
    ]
    value = json.loads(next(iter(canonical_inputs["raw_report_bytes"].values())))
    if field == "terrain_contact_matrix.path_order":
        value["terrain_contact_matrix"]["path_order"] = replacement
    else:
        value[field] = replacement
    result = SUMMARY.validate_raw_report(strict_bytes(value), preregistration, historical_preregistration)
    assert result == {
        "passed": False,
        "primary_reason": expected_reason,
        "checks": [
            {
                "reason": expected_reason,
                "status": "fail",
                "detail": result["checks"][0]["detail"],
            }
        ],
    }


@pytest.mark.parametrize(
    ("mutations", "expected_reason"),
    [
        (
            (
                lambda value: value["cpu_preflight_binding"].update(status="validated_for_gpu"),
                lambda value: value.update(headless=False),
            ),
            "rev20_evidence_chain_binding_invalid",
        ),
        (
            (
                lambda value: value.update(execution=[]),
                lambda value: value.update(headless=False),
            ),
            "rev20_execution_identity_invalid",
        ),
        (
            (
                lambda value: value["external_source_binding"].update(all_hashes_match=False),
                lambda value: value.update(headless=False),
            ),
            "rev20_source_provenance_invalid",
        ),
        (
            (
                lambda value: value.update(headless=False),
                lambda value: value["terrain_contact_matrix"]["path_order"].update(
                    sensor_paths_sha256="0" * 64
                ),
            ),
            "rev20_baseline_or_runtime_drift",
        ),
        (
            (
                lambda value: value["terrain_contact_matrix"]["shapes"].update(
                    raw=[151, 1, 3]
                ),
                lambda value: value["terrain_contact_matrix"]["safety"]["mass_tensor"][
                    "values"
                ][0].__setitem__(0, True),
            ),
            "rev20_matrix_shape_or_order_invalid",
        ),
        (
            (
                lambda value: value["terrain_contact_matrix"]["safety"]["mass_tensor"][
                    "values"
                ][0].__setitem__(0, True),
                lambda value: value["terrain_contact_matrix"]["safety"][
                    "non_foot_peak_force_body_weight_per_env"
                ].__setitem__(0, 15.000001),
            ),
            "rev20_matrix_numeric_invalid",
        ),
    ],
)
def test_lower_level_raw_report_uses_fixed_primary_reason_priority_for_combined_faults(
    canonical_inputs: dict[str, object], mutations, expected_reason: str
) -> None:
    preregistration = canonical_inputs["preregistration"]
    historical_preregistration = canonical_inputs["historical_blobs"][
        "configs/g009_r0_rev20_terrain_contact_matrix.json"
    ]
    value = json.loads(next(iter(canonical_inputs["raw_report_bytes"].values())))
    for mutation in mutations:
        mutation(value)
    result = SUMMARY.validate_raw_report(
        strict_bytes(value), preregistration, historical_preregistration
    )
    assert result["passed"] is False
    assert result["primary_reason"] == expected_reason


def test_synthesis_row_full_mirror_rejects_nonbinding_field_drift(canonical_inputs: dict[str, object]) -> None:
    synthesis = json.loads(canonical_inputs["synthesis_bytes"])
    rows = copy.deepcopy(synthesis["rows"])
    reports = [json.loads(canonical_inputs["raw_report_bytes"][path]) for path in SUMMARY.RAW_REPORT_PATHS]
    bindings = [dict(item) for item in synthesis["input_reports"]]
    rows[0]["availability_state"] = "unavailable"
    result = SUMMARY.validate_synthesis_row_mirrors(rows, reports, bindings)
    assert result["passed"] is False
    assert result["primary_reason"] == "rev20_matrix_shape_or_order_invalid"


def test_duplicate_json_key_is_rejected_deterministically() -> None:
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.validate_preregistration_bytes(b'{"schema_version":"a","schema_version":"b"}')
    assert error.value.reason == "rev21_preregistration_invalid"


def test_bool_mass_is_not_accepted_as_a_number(canonical_inputs: dict[str, object]) -> None:
    preregistration = canonical_inputs["preregistration"]
    historical_preregistration = canonical_inputs["historical_blobs"][
        "configs/g009_r0_rev20_terrain_contact_matrix.json"
    ]
    value = json.loads(next(iter(canonical_inputs["raw_report_bytes"].values())))
    value["terrain_contact_matrix"]["safety"]["mass_tensor"]["values"][0][0] = True
    result = SUMMARY.validate_raw_report(strict_bytes(value), preregistration, historical_preregistration)
    assert result["primary_reason"] == "rev20_matrix_numeric_invalid"


@pytest.mark.parametrize(
    ("margin", "passed"),
    [(-0.01, True), (-0.010001, False)],
)
def test_joint_margin_diagnostic_guard_is_inclusive(
    canonical_inputs: dict[str, object], margin: float, passed: bool
) -> None:
    preregistration = canonical_inputs["preregistration"]
    historical_preregistration = canonical_inputs["historical_blobs"][
        "configs/g009_r0_rev20_terrain_contact_matrix.json"
    ]
    value = json.loads(next(iter(canonical_inputs["raw_report_bytes"].values())))
    value["terrain_contact_matrix"]["step_ledger"][0]["joint_lower_margin_rad_by_env"][0][0] = margin
    result = SUMMARY.validate_raw_report(strict_bytes(value), preregistration, historical_preregistration)
    assert result["passed"] is passed
    if not passed:
        assert result["primary_reason"] == "rev20_matrix_physics_limit_exceeded"


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (
            lambda matrix: matrix["step_ledger"][0].update(storage_independent_before_clone=False),
            "rev20_matrix_shape_or_order_invalid",
        ),
        (
            lambda matrix: matrix["parity_step_indices"].pop(),
            "rev20_matrix_shape_or_order_invalid",
        ),
        (
            lambda matrix: matrix["storage_independent_step_indices"].pop(),
            "rev20_matrix_shape_or_order_invalid",
        ),
        (
            lambda matrix: matrix["step_ledger"][0].update(matrix_body_magnitude_sha256="0" * 64),
            "rev20_matrix_shape_or_order_invalid",
        ),
        (
            lambda matrix: matrix["step_ledger"][0]["non_foot_peak_force_n_per_env"].__setitem__(0, 1.0),
            "rev20_matrix_physics_limit_exceeded",
        ),
        (
            lambda matrix: matrix["checks"].update(exact_150_samples=False),
            "rev20_matrix_shape_or_order_invalid",
        ),
        (
            lambda matrix: matrix.update(structural_probe_valid=False),
            "rev20_matrix_shape_or_order_invalid",
        ),
        (
            lambda matrix: matrix.update(safety_valid=False),
            "rev20_matrix_shape_or_order_invalid",
        ),
        (
            lambda matrix: matrix.update(contract_valid=False),
            "rev20_matrix_shape_or_order_invalid",
        ),
        (
            lambda matrix: matrix.update(passed=False),
            "rev20_matrix_shape_or_order_invalid",
        ),
    ],
)
def test_lower_level_matrix_semantic_contract_rejects_persisted_flag_and_ledger_drift(
    canonical_inputs: dict[str, object], mutation, expected_reason: str
) -> None:
    preregistration = canonical_inputs["preregistration"]
    historical_preregistration = canonical_inputs["historical_blobs"][
        "configs/g009_r0_rev20_terrain_contact_matrix.json"
    ]
    value = json.loads(next(iter(canonical_inputs["raw_report_bytes"].values())))
    mutation(value["terrain_contact_matrix"])
    result = SUMMARY.validate_raw_report(strict_bytes(value), preregistration, historical_preregistration)
    assert result["passed"] is False
    assert result["primary_reason"] == expected_reason


def test_source_provenance_rejects_one_changed_historical_blob(canonical_inputs: dict[str, object]) -> None:
    blobs = dict(canonical_inputs["historical_blobs"])
    path = "scripts/probe_g009_r0_rev20_terrain_contact_matrix.py"
    blobs[path] += b"\n# tampered"
    result = evaluate(canonical_inputs, historical_blobs=blobs)
    assert reason(result) == "rev20_source_provenance_invalid"


def test_full_artifact_verification_detects_fresh_raw_tamper_without_writing(
    canonical_inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    projection = evaluate(canonical_inputs)
    assert reason(projection) == "matrix_authority_safety_gate_passed_for_diagnostic_preregistration"
    preregistration = canonical_inputs["preregistration"]
    prereg_path = tmp_path / "configs/g009_r0_rev21_matrix_authority_safety_gate.json"
    prereg_path.parent.mkdir(parents=True)
    prereg_path.write_bytes(canonical_inputs["preregistration_bytes"])
    synthesis_path = tmp_path / preregistration["predecessor"]["path"]
    synthesis_path.parent.mkdir(parents=True)
    synthesis_path.write_bytes(canonical_inputs["synthesis_bytes"])
    for relative, raw in canonical_inputs["raw_report_bytes"].items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    preflight_path = tmp_path / preregistration["predecessor"]["cpu_preflight_path"]
    preflight_path.write_bytes(canonical_inputs["cpu_preflight_bytes"])

    rev21_commit = "b" * 40
    rev21_blobs = {path: f"blob:{path}".encode() for path in SUMMARY.REQUIRED_SOURCE_PATHS}
    source_files = {path: SUMMARY.sha256_bytes(raw) for path, raw in rev21_blobs.items()}
    source_payload = "\n".join(f"{path}:{source_files[path]}" for path in SUMMARY.REQUIRED_SOURCE_PATHS)
    source_binding = {
        "schema_version": 1,
        "git_commit": rev21_commit,
        "source_binding_paths": list(SUMMARY.REQUIRED_SOURCE_PATHS),
        "source_binding_files": source_files,
        "source_bundle_sha256": SUMMARY.sha256_bytes(source_payload.encode()),
        "path_scoped_clean": True,
    }
    artifact = {
        **projection,
        "rev21_source_binding": source_binding,
        "execution": {
            "execution_id": "1234567890ab4def81234567890abcde",
            "started_at_utc": "2026-08-30T00:00:00Z",
            "output_path_repo_relative": "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json",
            "no_overwrite": True,
        },
    }
    artifact_path = tmp_path / "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json"
    artifact_path.write_bytes(strict_bytes(artifact))
    original_artifact = artifact_path.read_bytes()

    def fresh_blob(commit: str, path: str) -> bytes:
        if commit == rev21_commit:
            return rev21_blobs[path]
        return canonical_inputs["historical_blobs"][path]

    monkeypatch.setattr(SUMMARY, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(SUMMARY, "PREREGISTRATION_PATH", prereg_path)
    monkeypatch.setattr(SUMMARY, "CANONICAL_ARTIFACT_PATH", artifact_path)
    monkeypatch.setattr(SUMMARY, "_git_blob", fresh_blob)
    first = tmp_path / next(iter(canonical_inputs["raw_report_bytes"]))
    first.write_bytes(first.read_bytes() + b"\n")
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.verify_artifact(artifact_path)
    assert error.value.reason == "rev20_evidence_chain_binding_invalid"
    assert artifact_path.read_bytes() == original_artifact
    assert not any("tmp" in path.name or path.name.endswith(".part") for path in tmp_path.rglob("*"))


@pytest.mark.parametrize(
    "execution",
    [
        {
            "execution_id": "1234567890ab4def81234567890abcde",
            "started_at_utc": "2026-08-30T00:00:00Z",
            "output_path_repo_relative": "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json",
            "no_overwrite": True,
            "extra": True,
        },
        {
            "execution_id": "1234567890ab4def81234567890abcde",
            "started_at_utc": "not-rfc3339",
            "output_path_repo_relative": "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json",
            "no_overwrite": True,
        },
        {
            "execution_id": "1234567890ab4def81234567890abcde",
            "started_at_utc": "2026-08-30T00:00:00+09:00",
            "output_path_repo_relative": "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json",
            "no_overwrite": True,
        },
    ],
)
def test_artifact_execution_requires_exact_keys_and_rfc3339_utc(
    canonical_inputs: dict[str, object], execution: dict[str, object]
) -> None:
    projection = evaluate(canonical_inputs)
    value = {**projection, "rev21_source_binding": {}, "execution": execution}
    with pytest.raises(SUMMARY.GateValidationError) as error:
        SUMMARY.validate_artifact_value(value, projection, {})
    assert error.value.reason == "rev20_execution_identity_invalid"
