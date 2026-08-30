from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARY = load_module("g009_rev19_summary", ROOT / "scripts/summarize_g009_r0_rev19_contact_offset_intervention.py")
PROBE_TEST = load_module("g009_rev19_probe_test_helpers", ROOT / "tests/test_g009_r0_rev19_contact_offset_intervention.py")
PROBE = SUMMARY.probe


def synthesis_bundle_fixture() -> dict:
    files = {path: hashlib.sha256(path.encode()).hexdigest() for path in SUMMARY.SYNTHESIS_SOURCE_BINDING_PATHS}
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return {"schema_version": 1, "role": "offline_synthesis_implementation", "git_commit": "2" * 40, "git_commit_valid": True, "source_binding_paths": list(SUMMARY.SYNTHESIS_SOURCE_BINDING_PATHS), "source_binding_files": files, "source_bundle_sha256": hashlib.sha256(payload.encode()).hexdigest(), "all_files_present": True, "missing_files": [], "clean": True, "dirty_source_paths": []}


def entries_fixture(slots=SUMMARY.EXPECTED_SLOTS) -> list[tuple[dict, dict[str, str]]]:
    entries = []
    for arm, device, replicate in slots:
        report = PROBE_TEST.report_fixture(arm, device, replicate)
        report["execution"]["execution_id"] = uuid.uuid4().hex
        path = PROBE.expected_output_relative(arm, device, replicate)
        entries.append((report, {"path": path, "sha256": hashlib.sha256(path.encode()).hexdigest()}))
    return entries


@pytest.fixture
def synthesis_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    bundle = synthesis_bundle_fixture()
    monkeypatch.setattr(SUMMARY, "synthesis_source_bundle_provenance", lambda: bundle)
    monkeypatch.setattr(PROBE, "validate_source_bundle", lambda value: value)
    monkeypatch.setattr(PROBE, "validate_predecessor", lambda: {"path": PROBE.PREDECESSOR_PATH.relative_to(ROOT).as_posix(), "sha256": PROBE.PREDECESSOR_SHA256})

    def prepare(entries):
        path = tmp_path / "g009_r0_rev19_contact_offset_cpu_preflight_2x2_s42.json"
        path.write_text(json.dumps({"input_reports": [binding for _, binding in entries[:4]]}), encoding="utf-8")
        binding = {"status": "validated_for_gpu", "path": "reports/runs/g009_r0_rev19_contact_offset_cpu_preflight_2x2_s42.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "git_commit": "1" * 40, "probe_source_bundle_sha256": entries[0][0]["source_bundle"]["source_bundle_sha256"]}
        for report, _ in entries[4:]:
            report["cpu_preflight_binding"] = copy.deepcopy(binding)
        monkeypatch.setattr(PROBE, "validate_cpu_preflight_artifact", lambda _path, _source: copy.deepcopy(binding))
        return path

    return prepare


def _make_gpu_raw_observed(report: dict) -> None:
    cpu = PROBE_TEST.report_fixture(report["arm"], "cpu", report["replicate_index"])
    report["raw_contact_observation"] = copy.deepcopy(cpu["raw_contact_observation"])
    report["supporting_telemetry"] = copy.deepcopy(cpu["supporting_telemetry"])
    report["device_readback"] = {"requested_device": "cuda:0", "runtime_device": "cuda:0", "physics_scene_prim_path": "/physicsScene", "gpu_dynamics_enabled": True, "gpu_dynamics_matches_device": True, "error": None}
    report["feasibility"] = PROBE.derive_feasibility(report)


def test_exact_cpu_first_slots_and_paths_are_preregistered() -> None:
    assert SUMMARY.EXPECTED_SLOTS == (("A", "cpu", 1), ("A", "cpu", 2), ("B", "cpu", 1), ("B", "cpu", 2), ("A", "cuda:0", 1), ("A", "cuda:0", 2), ("B", "cuda:0", 1), ("B", "cuda:0", 2))
    assert SUMMARY.EXPECTED_PATHS[0].endswith("armA_cpu_rep01_s42.json")
    assert SUMMARY.EXPECTED_PATHS[-1].endswith("armB_gpu_rep02_s42.json")


def test_cpu_preflight_synthesis_requires_four_ready_repeatable_reports(synthesis_harness) -> None:
    value = SUMMARY.synthesize_cpu_preflight_loaded(entries_fixture(SUMMARY.CPU_SLOTS))
    assert value["status"] == "complete"
    assert value["cpu_preflight"]["gpu_stage_allowed"] is True
    assert value["integrity"]["exact_slots"] == ["A.cpu.rep1", "A.cpu.rep2", "B.cpu.rep1", "B.cpu.rep2"]


def test_unavailable_both_gpu_arms_keeps_all_authority_closed(synthesis_harness) -> None:
    entries = entries_fixture()
    value = SUMMARY.synthesize_loaded(entries, synthesis_harness(entries))
    assert value["status"] == "complete"
    assert value["decision"]["outcome"] == "gpu_raw_unavailable_both_arms"
    assert value["decision"]["selected_lever"] is None
    assert value["raw_callback_observation"]["physics_ground_truth_authority"] is False
    assert value["governance"]["ppo"]["updates"] == 0


def test_gpu_b_only_available_is_observation_not_selected_lever(synthesis_harness) -> None:
    entries = entries_fixture()
    for report, _ in entries:
        if report["arm"] == "B" and report["device"] == "cuda:0":
            _make_gpu_raw_observed(report)
    value = SUMMARY.synthesize_loaded(entries, synthesis_harness(entries))
    assert value["decision"]["outcome"] == "gpu_raw_enabled_by_contact_offset"
    assert value["governance"]["selected_lever"] is None


def test_one_of_two_split_stops_without_third_run(synthesis_harness) -> None:
    entries = entries_fixture()
    _make_gpu_raw_observed(entries[-1][0])
    value = SUMMARY.synthesize_loaded(entries, synthesis_harness(entries))
    assert value["decision"]["outcome"] == "inconclusive_nondeterministic"
    assert value["raw_callback_observation"]["third_run_majority_vote_allowed"] is False


@pytest.mark.parametrize(("values", "expected"), [
    ({"probe_integrity": False, "safety_available": True, "cpu_preflight_passed": True, "safety_passed": True, "any_split": True}, "probe_invalid"),
    ({"probe_integrity": True, "safety_available": False, "cpu_preflight_passed": True, "safety_passed": True, "any_split": True}, "safety_unavailable"),
    ({"probe_integrity": True, "safety_available": True, "cpu_preflight_passed": False, "safety_passed": True, "any_split": True}, "cpu_preflight_failed_gpu_results_not_interpretable"),
])
def test_decision_priority_precedes_replicate_split(values: dict, expected: str) -> None:
    outcome, _ = SUMMARY._decision(**values, gpu_a="split_or_nonrepeatable", gpu_b="split_or_nonrepeatable")
    assert outcome == expected


def test_duplicate_execution_slot_order_and_mass_drift_fail_closed(synthesis_harness) -> None:
    entries = entries_fixture()
    entries[1][0]["execution"]["execution_id"] = entries[0][0]["execution"]["execution_id"]
    with pytest.raises(ValueError, match="unique"):
        SUMMARY.synthesize_loaded(entries, synthesis_harness(entries))
    entries = entries_fixture()
    entries[0], entries[1] = entries[1], entries[0]
    with pytest.raises(ValueError, match="canonical slot order"):
        SUMMARY.synthesize_loaded(entries, synthesis_harness(entries))
    entries = entries_fixture()
    entries[-1][0]["manual_probe_safety"]["mass_evidence"]["tensor"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mass tensor hash"):
        SUMMARY.synthesize_loaded(entries, synthesis_harness(entries))
    entries = entries_fixture()
    mass = entries[-1][0]["manual_probe_safety"]["mass_evidence"]
    force_names = list(reversed(mass["contact_force_body_names"]))
    mass["contact_force_body_names"] = force_names
    mass["contact_force_body_names_sha256"] = hashlib.sha256(json.dumps(force_names, separators=(",", ":")).encode()).hexdigest()
    mass["ordered_body_names_equal"] = force_names == mass["body_names"]
    with pytest.raises(ValueError, match="contact force body ordering changed"):
        SUMMARY.synthesize_loaded(entries, synthesis_harness(entries))


def test_gpu_binding_and_preflight_cpu_inputs_are_immutable(synthesis_harness) -> None:
    entries = entries_fixture()
    path = synthesis_harness(entries)
    entries[-1][0]["cpu_preflight_binding"]["sha256"] = "9" * 64
    with pytest.raises(ValueError, match="preflight binding drift"):
        SUMMARY.synthesize_loaded(entries, path)


def test_synthesis_source_bundle_requires_exact_committed_blobs(monkeypatch: pytest.MonkeyPatch) -> None:
    value = synthesis_bundle_fixture()
    monkeypatch.setattr(PROBE, "committed_synthesis_blob_sha256", lambda path, _commit: value["source_binding_files"][path])
    assert SUMMARY.validate_synthesis_source_bundle(value) == value
    changed = copy.deepcopy(value)
    changed["source_binding_paths"] = list(reversed(changed["source_binding_paths"]))
    with pytest.raises(ValueError, match="path order"):
        SUMMARY.validate_synthesis_source_bundle(changed)


def test_cli_modes_require_explicit_preflight_contract() -> None:
    args = SUMMARY.parse_args(["--mode", "cpu-preflight", "--report", "a.json", "--output", "out.json"])
    assert args.mode == "cpu-preflight" and args.cpu_preflight is None
    args = SUMMARY.parse_args(["--mode", "final", "--report", "a.json", "--cpu-preflight", "pre.json", "--output", "out.json"])
    assert args.mode == "final" and args.cpu_preflight == Path("pre.json")
    assert SUMMARY.FINAL_SYNTHESIS_PATH.name == "g009_r0_rev19_contact_offset_intervention_synthesis_2x2x2_s42.json"
