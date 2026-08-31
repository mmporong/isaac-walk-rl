from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/summarize_g009_r0_rev23_matrix_observation_adapter_runtime.py"
SPEC = importlib.util.spec_from_file_location("rev23_runtime_summary", PATH)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


EXACT_FIELDS = (
    "adapter_contract_sha256",
    "sample_count",
    "source_shape",
    "world_xyz_shape",
    "magnitude_shape",
    "contact_mask_shape",
    "source_dtype",
    "world_xyz_dtype",
    "magnitude_dtype",
    "contact_mask_dtype",
    "source_device",
    "world_xyz_device",
    "magnitude_device",
    "contact_mask_device",
    "checks",
    "source_mutation_steps",
    "oracle_mismatch_steps",
    "alias_violation_steps",
)


def source_bundle() -> dict:
    paths = list(SUMMARY.probe.SYNTHESIS_SOURCE_BINDING_PATHS)
    files = {path: f"{index + 1:064x}" for index, path in enumerate(paths)}
    payload = "\n".join(f"{path}:{files[path]}" for path in paths)
    return {
        "schema_version": 1,
        "git_commit": "a" * 40,
        "source_binding_paths": paths,
        "source_binding_files": files,
        "source_bundle_sha256": SUMMARY.sha256_bytes(payload.encode()),
        "path_scoped_clean": True,
    }


def report(
    device: str,
    replicate: int,
    *,
    execution_id: str | None = None,
    maximum: float = 10.0,
    passed: bool = True,
) -> dict:
    adapter = {
        "adapter_contract_sha256": "0" * 64,
        "sample_count": 150,
        "source_shape": [8, 19, 1, 3],
        "world_xyz_shape": [8, 19, 3],
        "magnitude_shape": [8, 19],
        "contact_mask_shape": [8, 19],
        "source_dtype": "torch.float32",
        "world_xyz_dtype": "torch.float32",
        "magnitude_dtype": "torch.float32",
        "contact_mask_dtype": "torch.bool",
        "source_device": device,
        "world_xyz_device": device,
        "magnitude_device": device,
        "contact_mask_device": device,
        "checks": {"exact_150_samples": passed, "source_unchanged_150_of_150": passed},
        "source_mutation_steps": [] if passed else [1],
        "oracle_mismatch_steps": [],
        "alias_violation_steps": [],
        "max_magnitude_n": maximum,
        "magnitude_integral_n_s": 1.25,
        "passed": passed,
    }
    return {
        "device": device,
        "replicate_index": replicate,
        "execution": {"execution_id": execution_id or uuid.uuid4().hex},
        "rev23_source_bundle": source_bundle(),
        "cpu_preflight_binding": SUMMARY.probe.cpu_preflight_not_required_binding(),
        "feasibility": {"run_interpretable": True},
        "adapter_runtime": adapter,
        "adapter_decision": {
            "passed": passed,
            "outcome": "read_only_matrix_observation_adapter_runtime_run_passed" if passed else "read_only_matrix_observation_adapter_runtime_run_failed",
        },
    }


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SUMMARY.probe, "validate_report", lambda _value: {"passed": True})
    monkeypatch.setattr(SUMMARY, "synthesis_source_bundle_provenance", source_bundle)
    monkeypatch.setattr(
        SUMMARY,
        "_repeatability_contract",
        lambda: (EXACT_FIELDS, ("max_magnitude_n", "magnitude_integral_n_s"), 1e-5, 1e-6),
    )


def entries(reports: list[dict], paths: tuple[str, ...]) -> list[tuple[dict, dict[str, str]]]:
    return [
        (value, {"path": path, "sha256": f"{index + 20:064x}"})
        for index, (value, path) in enumerate(zip(reports, paths, strict=True))
    ]


def test_cpu_preflight_authorizes_gpu_only_after_two_passes() -> None:
    value = SUMMARY.cpu_preflight(entries([report("cpu", 1), report("cpu", 2)], SUMMARY.CPU_PATHS))
    assert value["decision"]["outcome"] == "gpu_stage_authorized"
    assert value["cpu_preflight"]["adapter_150_of_150_passed"] is True
    assert value["cpu_preflight"]["within_cpu_repeatability_passed"] is True


def test_cpu_preflight_rejects_wrong_order_and_duplicate_execution() -> None:
    first, second = report("cpu", 1), report("cpu", 2)
    with pytest.raises(ValueError, match="slot order"):
        SUMMARY.cpu_preflight(entries([second, first], SUMMARY.CPU_PATHS))
    shared = uuid.uuid4().hex
    with pytest.raises(ValueError, match="duplicate execution"):
        SUMMARY.cpu_preflight(entries([report("cpu", 1, execution_id=shared), report("cpu", 2, execution_id=shared)], SUMMARY.CPU_PATHS))


def test_cpu_preflight_rejects_adapter_failure_and_source_drift() -> None:
    with pytest.raises(ValueError, match="adapter runtime"):
        SUMMARY.cpu_preflight(entries([report("cpu", 1, passed=False), report("cpu", 2)], SUMMARY.CPU_PATHS))
    drift = report("cpu", 2)
    drift["rev23_source_bundle"] = {**source_bundle(), "source_bundle_sha256": "f" * 64}
    with pytest.raises(ValueError, match="source bundle drift"):
        SUMMARY.cpu_preflight(entries([report("cpu", 1), drift], SUMMARY.CPU_PATHS))


def test_repeatability_uses_inclusive_config_tolerance() -> None:
    left = SUMMARY.row(report("cpu", 1), {"path": "a", "sha256": "1" * 64})
    right = SUMMARY.row(report("cpu", 2, maximum=10.0 + 1e-5), {"path": "b", "sha256": "2" * 64})
    assert SUMMARY.repeatability([left, right])["repeatable"] is True
    right["max_magnitude_n"] += 1e-10
    assert SUMMARY.repeatability([left, right])["repeatable"] is False


def test_load_inputs_rejects_path_order_duplicate_hash_and_calls_probe_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(SUMMARY, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(SUMMARY, "RUNS_DIR", tmp_path / "reports/runs")
    SUMMARY.RUNS_DIR.mkdir(parents=True)
    calls: list[str] = []
    monkeypatch.setattr(SUMMARY.probe, "validate_report", lambda value: calls.append(value["id"]))
    first, second = SUMMARY.RUNS_DIR / "a.json", SUMMARY.RUNS_DIR / "b.json"
    first.write_text('{"id":"a"}', encoding="utf-8")
    second.write_text('{"id":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="path/order"):
        SUMMARY.load_inputs([first, second], ["reports/runs/b.json", "reports/runs/a.json"])
    assert calls == ["a", "b"]
    second.write_text('{"id":"a"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate report SHA"):
        SUMMARY.load_inputs([first, second], ["reports/runs/a.json", "reports/runs/b.json"])


def _materialize_reports(tmp_path: Path, reports: list[dict], paths: tuple[str, ...]) -> list[tuple[dict, dict[str, str]]]:
    result = []
    for value, relative in zip(reports, paths, strict=True):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(value, allow_nan=False) + "\n").encode()
        path.write_bytes(raw)
        result.append((value, {"path": relative, "sha256": SUMMARY.sha256_bytes(raw)}))
    return result


def _build_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict, list[tuple[dict, dict[str, str]]]]:
    monkeypatch.setattr(SUMMARY, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(SUMMARY, "RUNS_DIR", tmp_path / "reports/runs")
    monkeypatch.setattr(SUMMARY, "CPU_OUTPUT", tmp_path / "reports/runs/g009_r0_rev23_matrix_observation_adapter_cpu_preflight_2x_s42.json")
    monkeypatch.setattr(SUMMARY, "FINAL_OUTPUT", tmp_path / "reports/runs/g009_r0_rev23_matrix_observation_adapter_synthesis_2x2_s42.json")
    cpu_entries = _materialize_reports(tmp_path, [report("cpu", 1), report("cpu", 2)], SUMMARY.CPU_PATHS)
    value = SUMMARY.cpu_preflight(cpu_entries, SUMMARY.CPU_OUTPUT)
    SUMMARY.CPU_OUTPUT.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return SUMMARY.CPU_OUTPUT, value, cpu_entries


def test_validate_cpu_preflight_detects_bound_report_hash_change(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _path, value, entries_value = _build_preflight(tmp_path, monkeypatch)
    SUMMARY.validate_cpu_preflight_value(value, tmp_path, SUMMARY.CPU_OUTPUT.relative_to(tmp_path).as_posix(), source_bundle())
    (tmp_path / entries_value[0][1]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        SUMMARY.validate_cpu_preflight_value(value, tmp_path, SUMMARY.CPU_OUTPUT.relative_to(tmp_path).as_posix(), source_bundle())


def test_final_requires_exact_preflight_binding_and_validates_2x2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preflight_path, preflight, cpu_entries = _build_preflight(tmp_path, monkeypatch)
    raw = preflight_path.read_bytes()
    binding = {
        "status": "validated_for_gpu",
        "path": preflight_path.relative_to(tmp_path).as_posix(),
        "sha256": SUMMARY.sha256_bytes(raw),
        "git_commit": source_bundle()["git_commit"],
        "probe_source_bundle_sha256": source_bundle()["source_bundle_sha256"],
        "input_reports": preflight["input_reports"],
    }
    gpu_reports = [report("cuda:0", 1), report("cuda:0", 2)]
    for value in gpu_reports:
        value["cpu_preflight_binding"] = binding
    gpu_entries = _materialize_reports(tmp_path, gpu_reports, SUMMARY.FINAL_PATHS[2:])
    value = SUMMARY.final_synthesis(cpu_entries + gpu_entries, preflight_path, SUMMARY.FINAL_OUTPUT)
    assert value["decision"] == {
        "outcome": "read_only_matrix_observation_adapter_runtime_2x2_validated",
        "next_step": "preregister_and_run_gpu_throughput_ladder_before_matrix_gate01",
        "third_run_allowed": False,
    }
    SUMMARY.validate_final_value(value, tmp_path, SUMMARY.FINAL_OUTPUT.relative_to(tmp_path).as_posix(), source_bundle())


def test_final_rejects_preflight_binding_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preflight_path, preflight, cpu_entries = _build_preflight(tmp_path, monkeypatch)
    raw = preflight_path.read_bytes()
    binding = {
        "status": "validated_for_gpu",
        "path": preflight_path.relative_to(tmp_path).as_posix(),
        "sha256": SUMMARY.sha256_bytes(raw),
        "git_commit": source_bundle()["git_commit"],
        "probe_source_bundle_sha256": source_bundle()["source_bundle_sha256"],
        "input_reports": preflight["input_reports"],
    }
    gpu_reports = [report("cuda:0", 1), report("cuda:0", 2)]
    gpu_reports[0]["cpu_preflight_binding"] = binding
    gpu_reports[1]["cpu_preflight_binding"] = {**binding, "sha256": "f" * 64}
    with pytest.raises(ValueError, match="exact-bind"):
        SUMMARY.final_synthesis(
            cpu_entries + _materialize_reports(tmp_path, gpu_reports, SUMMARY.FINAL_PATHS[2:]), preflight_path, SUMMARY.FINAL_OUTPUT
        )


def test_mode_output_rejects_wrong_or_arbitrary_canonical_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(SUMMARY, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(SUMMARY, "RUNS_DIR", tmp_path / "reports/runs")
    monkeypatch.setattr(SUMMARY, "CPU_OUTPUT", SUMMARY.RUNS_DIR / "cpu.json")
    monkeypatch.setattr(SUMMARY, "FINAL_OUTPUT", SUMMARY.RUNS_DIR / "final.json")
    assert SUMMARY._validate_mode_output("cpu-preflight", SUMMARY.CPU_OUTPUT) == SUMMARY.CPU_OUTPUT.resolve()
    assert SUMMARY._validate_mode_output("final", SUMMARY.FINAL_OUTPUT) == SUMMARY.FINAL_OUTPUT.resolve()
    with pytest.raises(ValueError, match="exact canonical"):
        SUMMARY._validate_mode_output("cpu-preflight", SUMMARY.FINAL_OUTPUT)
    with pytest.raises(ValueError, match="exact canonical"):
        SUMMARY._validate_mode_output("final", SUMMARY.RUNS_DIR / "arbitrary.json")


def test_write_is_exclusive_and_rolls_back_on_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(SUMMARY, "RUNS_DIR", tmp_path)
    output = tmp_path / "result.json"
    monkeypatch.setattr(SUMMARY, "CPU_OUTPUT", output)
    monkeypatch.setattr(SUMMARY, "FINAL_OUTPUT", tmp_path / "final.json")
    SUMMARY.write_json_exclusive(output, {"ok": True})
    with pytest.raises(ValueError, match="overwrite"):
        SUMMARY.write_json_exclusive(output, {"ok": False})
    output.unlink()
    monkeypatch.setattr(SUMMARY.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("forced failure")))
    with pytest.raises(OSError, match="forced failure"):
        SUMMARY.write_json_exclusive(output, {"ok": True})
    assert not output.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_race_preserves_destination_created_by_other_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(SUMMARY, "RUNS_DIR", tmp_path)
    output = tmp_path / "result.json"
    monkeypatch.setattr(SUMMARY, "CPU_OUTPUT", output)
    monkeypatch.setattr(SUMMARY, "FINAL_OUTPUT", tmp_path / "final.json")
    competing_payload = b'{"owner":"other-process"}\n'
    original_link = SUMMARY.os.link

    def race_link(source: Path, destination: Path) -> None:
        output.write_bytes(competing_payload)
        original_link(source, destination)

    monkeypatch.setattr(SUMMARY.os, "link", race_link)

    with pytest.raises(FileExistsError):
        SUMMARY.write_json_exclusive(output, {"owner": "summary"})

    assert output.read_bytes() == competing_payload
    assert list(tmp_path.glob("*.tmp")) == []
