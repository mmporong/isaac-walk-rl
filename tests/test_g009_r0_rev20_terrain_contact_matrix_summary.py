from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/summarize_g009_r0_rev20_terrain_contact_matrix.py"
SPEC = importlib.util.spec_from_file_location("rev20_summary", PATH)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(SUMMARY)
REAL_VALIDATE_REPORT = SUMMARY.probe.validate_report


def synthesis_bundle() -> dict:
    files = {relative: f"{index + 1:064x}" for index, relative in enumerate(SUMMARY.probe.SYNTHESIS_SOURCE_BINDING_PATHS)}
    payload = "\n".join(f"{relative}:{files[relative]}" for relative in sorted(files))
    return {"schema_version": 1, "git_commit": "a" * 40, "git_commit_valid": True, "source_binding_paths": list(SUMMARY.probe.SYNTHESIS_SOURCE_BINDING_PATHS), "source_binding_files": files, "source_bundle_sha256": SUMMARY.sha256_bytes(payload.encode()), "clean": True}


def report(device: str, replicate: int, *, state: str = "observed_valid", execution_id: str | None = None, callback_count: int = 0) -> dict:
    steps = [[1] for _ in range(8)]
    matrix = {
        "availability_state": state,
        "path_order": {"sensor_paths_sha256": "1" * 64, "raw_filter_paths_sha256": "2" * 64, "logical_filter_paths_sha256": "4" * 64, "force_body_names_sha256": "3" * 64},
        "shapes": {"raw": [152, 1, 3], "reshaped": [8, 19, 1, 3]},
        "same_step_overlap": {"per_env_overlap_step_indices": steps, "source_env_overlap_step_indices": [1], "all_env_matrix_peak_force_n": 10.0, "source_env_matrix_peak_force_n": 9.0, "all_env_matrix_force_integral_n_s": 1.0, "source_env_matrix_force_integral_n_s": 0.9},
        "checks": {"all": True}, "structural_probe_valid": state != "invalid", "safety_valid": True,
        "overlap_available": state == "observed_valid", "contract_valid": state != "invalid", "passed": state != "invalid",
    }
    return {
        "device": device, "replicate_index": replicate, "execution": {"execution_id": execution_id or uuid.uuid4().hex},
        "terrain_contact_matrix": matrix, "source_bundle": {"source_bundle_sha256": "4" * 64, "git_commit": "a" * 40},
        "cpu_preflight_binding": SUMMARY.probe.cpu_preflight_not_required_binding(),
        "raw_contact_observation": {"callback_count": callback_count}, "feasibility": {"probe_valid": True},
        "baseline_snapshot": {"all_match": True}, "device_readback": {"gpu_dynamics_matches_device": True},
        "external_source_binding": {"all_hashes_match": True},
    }


def entries(reports: list[dict], paths: tuple[str, ...]) -> list[tuple[dict, dict[str, str]]]:
    return [(value, {"path": path, "sha256": f"{index + 5:064x}"}) for index, (value, path) in enumerate(zip(reports, paths, strict=True))]


def final_entries(reports: list[dict], preflight: dict) -> list[tuple[dict, dict[str, str]]]:
    result = entries(reports, SUMMARY.FINAL_PATHS)
    result[0] = (reports[0], preflight["input_reports"][0]); result[1] = (reports[1], preflight["input_reports"][1])
    return result


@pytest.fixture(autouse=True)
def isolate_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SUMMARY.probe, "validate_report", lambda value: {"run_interpretable": True})
    monkeypatch.setattr(SUMMARY.probe, "live_readback_valid", lambda value: True)
    monkeypatch.setattr(SUMMARY, "synthesis_source_bundle_provenance", synthesis_bundle)


def test_repeatability_tolerance_boundary_is_inclusive() -> None:
    left = SUMMARY.row(report("cpu", 1), {"path": SUMMARY.CPU_PATHS[0], "sha256": "5" * 64})
    right = SUMMARY.row(report("cpu", 2), {"path": SUMMARY.CPU_PATHS[1], "sha256": "6" * 64})
    right["all_env_matrix_peak_force_n"] = left["all_env_matrix_peak_force_n"] + SUMMARY.ABS_TOL
    assert SUMMARY.repeatability([left, right])["repeatable"] is True
    right["all_env_matrix_peak_force_n"] += 1e-10
    assert SUMMARY.repeatability([left, right])["repeatable"] is False


def test_row_uses_real_matrix_validator_for_representative_report(monkeypatch: pytest.MonkeyPatch) -> None:
    body_names = ["base"] + [f"body_{index}" for index in range(1, 19)]
    roots = [f"/World/envs/env_{index}/Robot/base" for index in range(8)]; namespaces = [root.rsplit("/", 1)[0] for root in roots]
    direct = torch.ones((152, 1, 3)); buffer = direct.reshape(8, 19, 1, 3).clone(); net = torch.ones((8, 19, 3))
    view = type("RigidContactView", (), {"sensor_paths": [f"{namespace}/{body}" for namespace in namespaces for body in body_names], "filter_paths": [[SUMMARY.probe.FILTER_PATHS[0]] for _ in range(152)], "sensor_count": 152, "filter_count": 1, "sensor_names": body_names * 8, "filter_names": [["CollisionPlane"] for _ in range(152)], "get_contact_force_matrix": lambda self, dt: direct})()
    sensor = type("Sensor", (), {"data": type("Data", (), {"net_forces_w": net, "force_matrix_w": buffer})(), "body_names": body_names, "contact_physx_view": view})()
    data = type("RobotData", (), {"joint_pos": torch.zeros((8, 12)), "joint_pos_limits": torch.stack((torch.full((8, 12), -1.0), torch.full((8, 12), 1.0)), dim=-1), "default_mass": torch.ones((8, 19))})()
    robot = type("Robot", (), {"data": data, "body_names": body_names, "root_physx_view": type("RootView", (), {"prim_paths": roots})()})()
    accumulator = SUMMARY.probe.MatrixSafetyAccumulator(requested_device="cpu")
    for step in range(1, 151): accumulator.observe(step, sensor, robot, 0.005, torch)
    prereg = SUMMARY.probe.load_preregistration(); contract = SUMMARY.probe.probe_contract("cpu", 1)
    matrix = accumulator.snapshot(); assert matrix["structural_probe_valid"] is True, matrix["error"]
    value = report("cpu", 1); value.update({"schema_version": SUMMARY.probe.SCHEMA_VERSION, "experiment_id": "G009-5-E013", "status": "complete", "revision": "rev20", "contract": contract, "contract_sha256": SUMMARY.probe.canonical_sha256(contract), "predecessor": {"path": "p", "sha256": "s"}, "governance": SUMMARY.probe.governance(), "terrain_contact_matrix": matrix, "external_source_binding": {"root": "R", "files": prereg["baseline_physics"]["isaaclab_external_source_binding"]["files"], "all_hashes_match": True}, "device_readback": {"requested_device": "cpu", "runtime_device": "cpu", "gpu_dynamics_enabled": False, "error": None, "gpu_dynamics_matches_device": True}})
    value["execution"] = {"execution_id": uuid.uuid4().hex, "started_at_utc": "2026-01-01T00:00:00Z", "output_path_repo_relative": SUMMARY.probe.EXPECTED_PATHS[("cpu", 1)], "no_overwrite": True}
    monkeypatch.setattr(SUMMARY.probe, "validate_report", REAL_VALIDATE_REPORT); monkeypatch.setattr(SUMMARY.probe, "validate_source_bundle", lambda item: item); monkeypatch.setattr(SUMMARY.probe, "validate_predecessor", lambda: value["predecessor"]); monkeypatch.setattr(SUMMARY.probe, "validate_baseline_payload", lambda *_args: True)
    value["feasibility"] = SUMMARY.probe.derive_feasibility(value)
    assert SUMMARY.row(value, {"path": SUMMARY.CPU_PATHS[0], "sha256": "5" * 64})["structural_probe_valid"] is True


def test_cpu_preflight_binds_two_unique_reports_and_execution_ids() -> None:
    value = SUMMARY.cpu_preflight(entries([report("cpu", 1), report("cpu", 2)], SUMMARY.CPU_PATHS))
    assert value["decision"]["outcome"] == "gpu_stage_authorized"
    assert value["integrity"]["unique_report_paths"] is True
    assert value["integrity"]["unique_report_sha256"] is True
    assert value["integrity"]["unique_execution_ids"] is True
    assert value["execution"]["execution_id"] not in {item["execution_id"] for item in [SUMMARY.row(report("cpu", 1), {"path": "x", "sha256": "1"}), SUMMARY.row(report("cpu", 2), {"path": "y", "sha256": "2"})]}


def test_cpu_preflight_rejects_duplicate_execution_id() -> None:
    shared = uuid.uuid4().hex
    with pytest.raises(ValueError, match="duplicate execution"):
        SUMMARY.cpu_preflight(entries([report("cpu", 1, execution_id=shared), report("cpu", 2, execution_id=shared)], SUMMARY.CPU_PATHS))


@pytest.mark.parametrize(("mutation", "outcome"), [
    (lambda rows: rows[0]["baseline_snapshot"].update(all_match=False), "probe_invalid"),
    (lambda rows: rows[0]["terrain_contact_matrix"].update(structural_probe_valid=False, safety_valid=False, availability_state="invalid"), "terrain_matrix_probe_invalid"),
    (lambda rows: rows[0]["terrain_contact_matrix"].update(safety_valid=False), "safety_limit_exceeded"),
    (lambda rows: [row["terrain_contact_matrix"].update(availability_state="unavailable", overlap_available=False) for row in rows], "cpu_terrain_matrix_unavailable_gpu_forbidden"),
    (lambda rows: rows[0]["terrain_contact_matrix"].update(availability_state="unavailable", overlap_available=False), "inconclusive_nondeterministic_gpu_forbidden"),
])
def test_cpu_decision_matrix_priority_and_reachable_outcomes(mutation, outcome: str) -> None:
    rows = [report("cpu", 1), report("cpu", 2)]; mutation(rows)
    assert SUMMARY.cpu_preflight(entries(rows, SUMMARY.CPU_PATHS))["decision"]["outcome"] == outcome


def test_load_inputs_rejects_canonical_path_and_duplicate_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(SUMMARY, "RUNS_DIR", tmp_path); monkeypatch.setattr(SUMMARY, "REPO_ROOT", tmp_path)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text('{"x":1}', encoding="utf-8"); second.write_text('{"x":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical input path"):
        SUMMARY.load_inputs([first, second], ["wrong.json", "b.json"])
    with pytest.raises(ValueError, match="duplicate report SHA"):
        SUMMARY.load_inputs([first, second], ["a.json", "b.json"])


def build_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    path = tmp_path / "cpu_preflight.json"; monkeypatch.setattr(SUMMARY, "CPU_OUTPUT", path); monkeypatch.setattr(SUMMARY, "REPO_ROOT", tmp_path)
    cpu_reports = [report("cpu", 1), report("cpu", 2)]
    bound = []
    for value, relative in zip(cpu_reports, SUMMARY.CPU_PATHS, strict=True):
        report_path = tmp_path / relative; report_path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, allow_nan=False).encode(); report_path.write_bytes(raw)
        bound.append((value, {"path": relative, "sha256": SUMMARY.sha256_bytes(raw)}))
    value = SUMMARY.cpu_preflight(bound, output=path)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def test_final_synthesis_requires_exact_preflight_and_four_unique_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preflight_path, preflight = build_preflight(tmp_path, monkeypatch)
    raw = preflight_path.read_bytes()
    binding = {"status": "validated_for_gpu", "path": preflight_path.relative_to(tmp_path).as_posix(), "sha256": SUMMARY.sha256_bytes(raw), "git_commit": preflight["integrity"]["git_commit"], "probe_source_bundle_sha256": preflight["integrity"]["probe_source_bundle_sha256"], "input_reports": preflight["input_reports"]}
    values = [report("cpu", 1), report("cpu", 2), report("cuda:0", 1), report("cuda:0", 2)]
    values[2]["cpu_preflight_binding"] = binding; values[3]["cpu_preflight_binding"] = binding
    result = SUMMARY.final_synthesis(final_entries(values, preflight), preflight_path, tmp_path / "final.json")
    assert result["decision"]["outcome"] == "terrain_pair_matrix_authority_candidate_validated"
    assert result["claim_limits"]["callback_count_used"] is False
    assert result["claim_limits"]["gpu_contact_absence_claimed"] is False


def test_final_synthesis_rejects_gpu_preflight_binding_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preflight_path, preflight = build_preflight(tmp_path, monkeypatch)
    raw = preflight_path.read_bytes()
    binding = {"status": "validated_for_gpu", "path": preflight_path.relative_to(tmp_path).as_posix(), "sha256": SUMMARY.sha256_bytes(raw), "git_commit": preflight["integrity"]["git_commit"], "probe_source_bundle_sha256": preflight["integrity"]["probe_source_bundle_sha256"], "input_reports": preflight["input_reports"]}
    values = [report("cpu", 1), report("cpu", 2), report("cuda:0", 1), report("cuda:0", 2)]
    values[2]["cpu_preflight_binding"] = binding; values[3]["cpu_preflight_binding"] = {**binding, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="exact-bind"):
        SUMMARY.final_synthesis(final_entries(values, preflight), preflight_path, tmp_path / "final.json")


def test_callback_count_cannot_change_final_outcome(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preflight_path, preflight = build_preflight(tmp_path, monkeypatch); raw = preflight_path.read_bytes()
    binding = {"status": "validated_for_gpu", "path": preflight_path.relative_to(tmp_path).as_posix(), "sha256": SUMMARY.sha256_bytes(raw), "git_commit": preflight["integrity"]["git_commit"], "probe_source_bundle_sha256": preflight["integrity"]["probe_source_bundle_sha256"], "input_reports": preflight["input_reports"]}
    values = [report("cpu", 1, callback_count=0), report("cpu", 2, callback_count=999), report("cuda:0", 1, callback_count=0), report("cuda:0", 2, callback_count=999999)]
    for value in values[2:]: value["cpu_preflight_binding"] = binding
    assert SUMMARY.final_synthesis(final_entries(values, preflight), preflight_path, tmp_path / "final.json")["decision"]["outcome"] == "terrain_pair_matrix_authority_candidate_validated"


def test_final_gpu_unavailable_two_of_two_is_reachable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preflight_path, preflight = build_preflight(tmp_path, monkeypatch); raw = preflight_path.read_bytes()
    binding = {"status": "validated_for_gpu", "path": preflight_path.relative_to(tmp_path).as_posix(), "sha256": SUMMARY.sha256_bytes(raw), "git_commit": preflight["integrity"]["git_commit"], "probe_source_bundle_sha256": preflight["integrity"]["probe_source_bundle_sha256"], "input_reports": preflight["input_reports"]}
    values = [report("cpu", 1), report("cpu", 2), report("cuda:0", 1, state="unavailable"), report("cuda:0", 2, state="unavailable")]
    for value in values[2:]: value["cpu_preflight_binding"] = binding
    result = SUMMARY.final_synthesis(final_entries(values, preflight), preflight_path, tmp_path / "final.json")
    assert result["decision"]["outcome"] == "gpu_terrain_matrix_unavailable"


def test_write_json_exclusive_rejects_nan_overwrite_and_rolls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(SUMMARY, "RUNS_DIR", tmp_path)
    output = tmp_path / "out.json"
    with pytest.raises(ValueError): SUMMARY.write_json_exclusive(output, {"bad": float("nan")})
    assert not output.exists()
    output.write_text("owned", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"): SUMMARY.write_json_exclusive(output, {"ok": True})
    assert output.read_text() == "owned"
    output.unlink()
    original_open = Path.open
    def fail_destination(path: Path, *args, **kwargs):
        if path == output and args and args[0] == "xb": raise OSError("injected")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", fail_destination)
    with pytest.raises(OSError, match="injected"): SUMMARY.write_json_exclusive(output, {"ok": True})
    assert not output.exists()


def test_cli_requires_explicit_mode_inputs_and_output() -> None:
    help_text = SUMMARY.build_parser().format_help()
    for option in ("--mode", "--inputs", "--cpu-preflight", "--output"): assert option in help_text
