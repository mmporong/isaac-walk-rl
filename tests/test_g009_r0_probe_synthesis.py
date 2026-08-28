from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g009_probe_synthesis_under_test",
    ROOT / "scripts" / "synthesize_g009_r0_probe.py",
)
assert SPEC is not None and SPEC.loader is not None
SYNTHESIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNTHESIS)


def _source_bundle() -> dict:
    files = {
        "configs/g009_r0.json": "1" * 64,
        "src/isaac_walk_g009/mdp/recover.py": "2" * 64,
    }
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return {
        "schema_version": 1,
        "git_commit": "c" * 40,
        "git_commit_valid": True,
        "source_binding_paths": list(files),
        "source_binding_files": files,
        "source_bundle_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "all_files_present": True,
        "missing_files": [],
        "clean": True,
        "dirty_source_paths": [],
    }


def _report(device: str) -> dict:
    is_cpu = device == "cpu"
    return {
        "schema_version": 3,
        "goal_id": "g009",
        "stage_id": "R0",
        "probe": "flat_recover_runtime_calibration",
        "contract_sha256": "a" * 64,
        "source_bundle": _source_bundle(),
        "task": "Isaac-G009-Recover-Flat-Go2-R0-v0",
        "seed": 42,
        "device": device,
        "num_envs": 8,
        "rollout_steps": 150,
        "pose_name_order": ["prone", "supine", "left_side", "right_side"],
        "run_health": {"passed": True},
        "runtime_contract": {"passed": True},
        "qualification": {"status": "not_run", "passed": None},
        "required_crosschecks": {
            "cpu_contact_separation": {
                "authority_device": "cpu",
                "this_run_is_authority": is_cpu,
                "data_available": is_cpu,
                "status": "observed" if is_cpu else "requires_cpu_crosscheck",
                "passed": True if is_cpu else None,
            }
        },
    }


def _write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report), encoding="utf-8")


def _valid_paths(tmp_path: Path) -> tuple[Path, Path]:
    gpu_path = tmp_path / "gpu.json"
    cpu_path = tmp_path / "cpu.json"
    _write_report(gpu_path, _report("cuda:0"))
    _write_report(cpu_path, _report("cpu"))
    return gpu_path, cpu_path


def test_synthesis_passes_runtime_without_qualifying_learned_policy(
    tmp_path: Path,
) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)

    result = SYNTHESIS.synthesize_reports(gpu_path, cpu_path)

    assert result["runtime_calibration_passed"] is True
    assert result["learned_policy_qualified"] is False
    assert result["learned_policy_qualification"] == {
        "status": "not_run",
        "passed": False,
        "reason": "runtime calibration does not evaluate a learned checkpoint",
    }


def test_synthesis_records_absolute_input_paths_and_file_sha256(tmp_path: Path) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)

    result = SYNTHESIS.synthesize_reports(gpu_path, cpu_path)

    assert result["inputs"]["gpu"] == {
        "absolute_path": str(gpu_path.resolve()),
        "sha256": hashlib.sha256(gpu_path.read_bytes()).hexdigest(),
    }
    assert result["inputs"]["cpu"]["absolute_path"] == str(cpu_path.resolve())
    assert result["schema_version"] == 2
    assert result["verified_identity"]["source_bundle"] == _source_bundle()


def test_synthesis_rejects_contract_sha256_mismatch(tmp_path: Path) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)
    cpu = _report("cpu")
    cpu["contract_sha256"] = "b" * 64
    _write_report(cpu_path, cpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match="contract_sha256"):
        SYNTHESIS.synthesize_reports(gpu_path, cpu_path)


def test_synthesis_rejects_dirty_or_mismatched_source_bundle(tmp_path: Path) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)
    cpu = _report("cpu")
    cpu["source_bundle"]["clean"] = False
    cpu["source_bundle"]["dirty_source_paths"] = [" M src/example.py"]
    _write_report(cpu_path, cpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match="source_bundle clean"):
        SYNTHESIS.synthesize_reports(gpu_path, cpu_path)


def test_synthesis_rejects_tampered_source_bundle_hash(tmp_path: Path) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)
    cpu = _report("cpu")
    cpu["source_bundle"]["source_binding_files"]["configs/g009_r0.json"] = "f" * 64
    _write_report(cpu_path, cpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match="source_bundle_sha256 mismatch"):
        SYNTHESIS.synthesize_reports(gpu_path, cpu_path)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("task", "another-task"),
        ("seed", 7),
        ("num_envs", 4),
        ("rollout_steps", 149),
        ("pose_name_order", ["supine", "prone", "left_side", "right_side"]),
    ],
)
def test_synthesis_rejects_execution_identity_mismatch(
    tmp_path: Path, field: str, changed: object
) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)
    cpu = _report("cpu")
    cpu[field] = changed
    _write_report(cpu_path, cpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match=field):
        SYNTHESIS.synthesize_reports(gpu_path, cpu_path)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("goal_id", "g008"),
        ("stage_id", "R1"),
        ("probe", "another_probe"),
    ],
)
def test_synthesis_rejects_report_for_another_calibration(
    tmp_path: Path, field: str, changed: str
) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)
    gpu = _report("cuda:0")
    gpu[field] = changed
    _write_report(gpu_path, gpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match=field):
        SYNTHESIS.synthesize_reports(gpu_path, cpu_path)


def test_synthesis_rejects_failed_gpu_runtime_contract(tmp_path: Path) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)
    gpu = _report("cuda:0")
    gpu["runtime_contract"]["passed"] = False
    _write_report(gpu_path, gpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match="GPU runtime_contract"):
        SYNTHESIS.synthesize_reports(gpu_path, cpu_path)


def test_synthesis_rejects_unavailable_cpu_separation_authority(
    tmp_path: Path,
) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)
    cpu = deepcopy(_report("cpu"))
    cpu["required_crosschecks"]["cpu_contact_separation"]["data_available"] = False
    _write_report(cpu_path, cpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match="data_available"):
        SYNTHESIS.synthesize_reports(gpu_path, cpu_path)


@pytest.mark.parametrize("input_name", ["gpu", "cpu"])
def test_synthesis_requires_input_qualification_to_be_not_run(
    tmp_path: Path, input_name: str
) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)
    path = gpu_path if input_name == "gpu" else cpu_path
    report = _report("cuda:0" if input_name == "gpu" else "cpu")
    report["qualification"] = {"status": "passed", "passed": True}
    _write_report(path, report)

    with pytest.raises(SYNTHESIS.SynthesisError, match="qualification"):
        SYNTHESIS.synthesize_reports(gpu_path, cpu_path)


def test_cli_exposes_no_threshold_override() -> None:
    parser = SYNTHESIS.build_parser()

    assert all("threshold" not in option for action in parser._actions for option in action.option_strings)
