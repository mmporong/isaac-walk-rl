from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g009_rev17_mechanism_split_test",
    ROOT / "scripts" / "summarize_g009_r0_rev17_mechanism_split.py",
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


@pytest.fixture(scope="module")
def canonical_entries():
    return SUMMARY._canonical_inputs()


@pytest.fixture(scope="module")
def synthesis():
    return SUMMARY.synthesize()


def test_canonical_offline_synthesis_is_diagnostic_only(synthesis) -> None:
    assert synthesis["schema_version"] == "g009.r0.rev17.mechanism_split.v1"
    assert synthesis["evidence_id"] == "G009-5-E010"
    assert synthesis["status"] == "pass"
    assert synthesis["integrity"]["passed"] is True
    assert synthesis["integrity"]["hash_bound"] is True
    assert synthesis["input_report_count"] == 12
    assert synthesis["mode"] == "offline_reanalysis_of_immutable_rev16_reports"
    split = synthesis["mechanism_split"]
    assert split["temporal_signatures"]["all_rev16_concentration_values_reproduced"] is True
    assert split["causal_inferences"]["decision"]["outcome"] == "inconclusive"
    assert split["causal_inferences"]["decision"]["selected_lever"] is None
    assert synthesis["ppo"] == {"allowed": False, "status": "not_run"}
    runs = split["direct_observations"]["runs"]
    assert len(runs) == 12
    assert all(run["physics_row_count"] == 600 for run in runs)
    assert all(run["control_context"]["control_row_count"] == 150 for run in runs)
    assert all(len(run["focus_steps"]) == 3 for run in runs)
    assert all(run["peak_window"]["step_count"] == 17 for run in runs)


def test_concentration_reproduces_canonical_rev16_exactly(synthesis) -> None:
    ratios = synthesis["mechanism_split"]["temporal_signatures"][
        "b_gpu_over_b_cpu_concentration_ratio_by_replicate"
    ]
    assert ratios == pytest.approx([1.183556126964255] * 3, rel=1e-12)


def test_b_cpu_gpu_audit_metrics_are_explicit(synthesis) -> None:
    comparisons = synthesis["mechanism_split"]["temporal_signatures"][
        "b_cpu_gpu_mechanism_comparison"
    ]
    assert len(comparisons) == 3
    for comparison in comparisons:
        assert comparison["gpu_over_cpu_peak_base_force_percent_change"] == pytest.approx(
            26.7204222332
        )
        assert comparison[
            "gpu_over_cpu_all_body_impulse_magnitude_window_percent_change"
        ] == pytest.approx(1.42027016703)
        assert comparison["gpu_over_cpu_base_window_impulse_percent_change"] == pytest.approx(
            7.06752248261
        )
        assert comparison["cpu_base_share_of_all_body_impulse_magnitude"] == pytest.approx(
            0.642132922736
        )
        assert comparison["gpu_base_share_of_all_body_impulse_magnitude"] == pytest.approx(
            0.677887970803
        )
        assert comparison[
            "gpu_over_cpu_fr_rr_hip_impulse_magnitude_percent_change"
        ] == pytest.approx(-10.8337844094)


def test_cpu_contact_authority_is_not_synthesized_for_gpu(synthesis) -> None:
    runs = synthesis["mechanism_split"]["direct_observations"]["runs"]
    for run in runs:
        authority = run["contact_authority"]
        if run["device"] == "cpu":
            assert authority["topology_available"] is True
            assert authority["body_pair_counts"] is not None
            assert set(authority["per_physics_step"]) == {"128", "129", "130"}
            for step in authority["per_physics_step"].values():
                assert set(step) == {
                    "event_count",
                    "header_count",
                    "contact_point_count",
                    "reported_impulse_vector_sum_n_s",
                    "body_pair_counts",
                    "minimum_separation_m",
                }
        else:
            assert authority == {
                "authority": "cpu_only",
                "availability": "unavailable_on_gpu",
                "topology_available": False,
                "event_count": None,
                "contact_point_count": None,
                "reported_impulse_vector_sum_n_s": None,
                "body_pair_counts": None,
                "minimum_separation_m": None,
                "per_physics_step": None,
                "per_physics_step_status": "unavailable_on_gpu",
            }


def test_b_cpu_contact_topology_sequence_is_exact_and_gpu_stays_unavailable(
    synthesis,
) -> None:
    runs = synthesis["mechanism_split"]["direct_observations"]["runs"]
    b_cpu = next(
        run
        for run in runs
        if run["arm"] == "B"
        and run["device"] == "cpu"
        and run["replicate_index"] == 1
    )
    per_step = b_cpu["contact_authority"]["per_physics_step"]
    assert [set(per_step[str(step)]["body_pair_counts"]) for step in (128, 129, 130)] == [
        {
            "FL_hip<->/World/ground/terrain/GroundPlane/CollisionPlane",
            "RL_hip<->/World/ground/terrain/GroundPlane/CollisionPlane",
        },
        {
            "FL_hip<->/World/ground/terrain/GroundPlane/CollisionPlane",
            "base<->/World/ground/terrain/GroundPlane/CollisionPlane",
        },
        {"base<->/World/ground/terrain/GroundPlane/CollisionPlane"},
    ]
    assert all(
        run["contact_authority"]["per_physics_step"] is None
        and run["contact_authority"]["per_physics_step_status"]
        == "unavailable_on_gpu"
        for run in runs
        if run["device"] == "cuda:0"
    )


def test_control_bucket_mapping_and_context_are_explicit(synthesis) -> None:
    runs = synthesis["mechanism_split"]["direct_observations"]["runs"]
    expected_mapping = [
        {"physics_step": 128, "control_step": 32, "contact_force_history_slot": 0},
        {"physics_step": 129, "control_step": 33, "contact_force_history_slot": 3},
        {"physics_step": 130, "control_step": 33, "contact_force_history_slot": 2},
    ]
    for run in runs:
        context = run["control_context"]
        assert context["physics_to_control_bucket_mapping"] == expected_mapping
        assert "must not be interpreted as an instantaneous state" in context[
            "mapping_interpretation_label"
        ]
        assert set(context["selected_control_buckets"]) == {"32", "33"}
        for bucket in context["selected_control_buckets"].values():
            assert bucket["root_linear_speed_m_s"] >= 0
            assert bucket["root_angular_speed_rad_s"] >= 0
            assert bucket["max_link_linear_speed_m_s"] >= 0
            assert bucket["max_link_angular_speed_rad_s"] >= 0
            assert bucket["max_joint_speed_rad_s"] >= 0
            assert bucket["max_abs_applied_torque_nm"] >= 0
            assert set(bucket["action_and_ema_trace"]) == set(SUMMARY.CONTROL_FIELDS)


def test_replicates_are_semantically_identical_3_of_3(synthesis) -> None:
    identity = synthesis["mechanism_split"]["temporal_signatures"][
        "replicate_semantic_identity"
    ]
    assert [item["group"] for item in identity] == [
        "A.cpu",
        "A.cuda:0",
        "B.cpu",
        "B.cuda:0",
    ]
    assert all(item["identical_3_of_3"] is True for item in identity)
    assert all(
        all(len(value) == 64 for value in item["canonical_json_sha256"].values())
        for item in identity
    )


def _measure_mutated(canonical_entries, run_index: int, mutation) -> None:
    canonical, _, entries = canonical_entries
    report = copy.deepcopy(entries[run_index][0])
    mutation(report)
    canonical_run = [run for group in canonical["groups"] for run in group["runs"]][run_index]
    group = canonical["groups"][run_index // 3]
    SUMMARY._measure_run(
        report, entries[run_index][1], canonical_run, group["arm"], group["device"]
    )


def test_rejects_missing_physics_row(canonical_entries) -> None:
    with pytest.raises(ValueError, match="exactly 600 physics rows"):
        _measure_mutated(canonical_entries, 0, lambda report: report["physics_substep_telemetry"].pop())


def test_rejects_step_or_history_drift(canonical_entries) -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _measure_mutated(canonical_entries, 0, lambda report: report["physics_substep_telemetry"][127].__setitem__("physics_step", 129))
    with pytest.raises(ValueError, match="history mapping"):
        _measure_mutated(canonical_entries, 0, lambda report: report["physics_substep_telemetry"][127].__setitem__("contact_force_history_slot", 3))


def test_rejects_body_index_name_or_float_drift(canonical_entries) -> None:
    with pytest.raises(ValueError, match="body index/name alignment"):
        _measure_mutated(canonical_entries, 0, lambda report: report["physics_substep_telemetry"][127]["body_names"].reverse())
    with pytest.raises(ValueError, match="magnitude/vector mismatch"):
        _measure_mutated(canonical_entries, 0, lambda report: report["physics_substep_telemetry"][127]["per_body_force_magnitude_n"].__setitem__(0, 123.0))


def test_rejects_nan_and_zero_concentration_denominator(canonical_entries) -> None:
    with pytest.raises(ValueError, match="finite"):
        _measure_mutated(canonical_entries, 0, lambda report: report["physics_substep_telemetry"][127]["per_body_force_vector_n"][0].__setitem__(0, float("nan")))

    with pytest.raises(ValueError, match="denominator"):
        SUMMARY.concentration_index(0.0, 0.0)


def test_rejects_gpu_contact_topology_fabrication(canonical_entries) -> None:
    def fabricate(report):
        report["cpu_contact_authority"]["events"] = []

    with pytest.raises(ValueError, match="explicitly unavailable"):
        _measure_mutated(canonical_entries, 3, fabricate)


def test_rejects_control_row_shape_name_and_nan_drift(canonical_entries) -> None:
    with pytest.raises(ValueError, match="exactly 150 control rows"):
        _measure_mutated(
            canonical_entries,
            0,
            lambda report: report["control_step_telemetry"].pop(),
        )
    with pytest.raises(ValueError, match="link state contract"):
        _measure_mutated(
            canonical_entries,
            0,
            lambda report: report["control_step_telemetry"][31]["link_names"].reverse(),
        )
    with pytest.raises(ValueError, match="finite"):
        _measure_mutated(
            canonical_entries,
            0,
            lambda report: report["control_step_telemetry"][31]["raw_action"].__setitem__(
                0, float("nan")
            ),
        )


def test_rejects_replicate_semantic_payload_mismatch() -> None:
    key = "physics_substep_telemetry_sha256"
    rows = [
        {
            "semantic_payload_hashes": {
                key: "a" * 64,
                "control_step_telemetry_sha256": "b" * 64,
                "cpu_contact_authority_sha256": "c" * 64,
            }
        }
        for _ in range(3)
    ]
    rows[2]["semantic_payload_hashes"][key] = "d" * 64
    with pytest.raises(ValueError, match="replicate semantic payload mismatch"):
        SUMMARY._validate_group_semantic_identity("A.cpu", rows)


def test_causal_next_action_keeps_lever_unselected(synthesis) -> None:
    causal = synthesis["mechanism_split"]["causal_inferences"]
    assert causal["decision"]["selected_lever"] is None
    assert "authoritative constraint/contact instrumentation" in causal["next_action"]
    assert "preregistered single-variable intervention probe" in causal["next_action"]


def test_force_divergence_is_observed_but_contact_divergence_is_unavailable(
    synthesis,
) -> None:
    temporal = synthesis["mechanism_split"]["temporal_signatures"]
    assert temporal["b_cpu_gpu_first_physics_force_divergence"] == {
        "status": "observed_in_force_aggregation",
        "replicate_count": 3,
        "identical_3_of_3": True,
        "step": 128,
        "time_s": 0.64,
        "variable": "base_force_bodyweights",
        "max_abs_delta": pytest.approx(3.1033276173468844),
        "tolerance": 1e-06,
    }
    assert temporal["b_cpu_gpu_contact_topology_divergence"]["status"] == (
        "unavailable_on_gpu"
    )
    assert temporal["b_cpu_gpu_contact_topology_divergence"]["step"] is None


def test_all_diagnostic_governance_contracts_remain_closed(synthesis) -> None:
    assert synthesis["diagnostic_only"] is True
    assert synthesis["ppo"] == {"allowed": False, "status": "not_run"}
    assert synthesis["qualification"] == {
        "eligible": False,
        "status": "not_run",
        "passed": None,
    }
    assert synthesis["governance"] == {
        "diagnostic_only": True,
        "learned": False,
        "ppo": {"allowed": False, "status": "not_run"},
        "gate01": {"allowed": False, "status": "forbidden"},
        "qualification": {"eligible": False, "status": "not_run", "passed": None},
    }
    root_decision = synthesis["mechanism_split"]["decision"]
    causal_decision = synthesis["mechanism_split"]["causal_inferences"]["decision"]
    assert root_decision == {
        "outcome": causal_decision["outcome"],
        "selected_lever": causal_decision["selected_lever"],
    }


def test_main_refuses_existing_output_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = SUMMARY.RUNS_DIR / "g009_r0_rev17_existing_output_test.json"
    output.unlink(missing_ok=True)
    output.write_bytes(b"user-owned")
    monkeypatch.setattr(SUMMARY, "synthesize", lambda _path: {})
    try:
        with pytest.raises(ValueError, match="refusing to overwrite"):
            SUMMARY.main(["--synthesis", str(SUMMARY.CANONICAL_SYNTHESIS), "--output", str(output)])
        assert output.read_bytes() == b"user-owned"
    finally:
        output.unlink(missing_ok=True)


def test_canonical_hash_is_pinned(tmp_path: Path) -> None:
    source = SUMMARY.CANONICAL_SYNTHESIS
    changed = json.loads(source.read_text(encoding="utf-8"))
    changed["status"] = "tampered"
    target = SUMMARY.RUNS_DIR / "g009_r0_rev17_hash_tamper_test.json"
    try:
        target.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="synthesis hash mismatch"):
            SUMMARY._canonical_inputs(target)
    finally:
        target.unlink(missing_ok=True)
