from __future__ import annotations

import hashlib
import importlib.util
import json
import uuid
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


@pytest.fixture(autouse=True)
def _bind_synthesis_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SYNTHESIS, "REPO_ROOT", tmp_path)


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
        "contract_sha256": "ebee855c503c77bce93c0884535d4fdf66ee5a01538fa59eef0e1b7aabba7558",
        "source_bundle": _source_bundle(),
        "task": "Isaac-G009-Recover-Flat-Go2-R0-v0",
        "seed": 42,
        "device": device,
        "headless": True,
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
        "calibration_thresholds": {"max_nonfoot_force_bodyweights": 15.0},
        "pose_mode_metrics": [
            {
                "env_index": index,
                "pose_id": ["prone", "supine", "left_side", "right_side"][index % 4],
                "action_mode": "zero_normalized" if index < 4 else "reset_pose_hold",
                "max_nonfoot_force_bodyweights": float(index + 1),
                "max_nonfoot_force_physics_step": 10 + index,
                "max_nonfoot_force_body_index": 4 + index,
                "max_nonfoot_force_body_name": f"body_{index}",
            }
            for index in range(8)
        ],
        "checks": {
            "reset_pose_hold_action_diagnostics_finite": True,
            "reset_pose_hold_actions_unsaturated": True,
            "reset_pose_hold_reachable_targets_match_reset_positions": True,
            "nonfoot_peak_force_body_attribution_complete": True,
            "nonfoot_peak_force_bounded": True,
            "rigid_body_max_depenetration_velocity_matches_contract": True,
            "another_runtime_check": True,
        },
    }


def _write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report), encoding="utf-8")


def _bind_rev15_progression(report: dict) -> None:
    is_cpu = report["device"] == "cpu"
    report["contract_sha256"] = SYNTHESIS.REV15_CONTRACT_SHA256
    report["passed"] = True
    report["passed_semantics"] = "progression_gate_not_policy_qualification"
    report["progression_gate"] = {
        "passed": True,
        "status": (
            "passed"
            if is_cpu
            else "passed_runtime_contract_cpu_authority_not_evaluated"
        ),
        "device": report["device"],
        "cpu_contact_separation_required_this_run": is_cpu,
        "blocking_checks": {
            "runtime_contract": True,
            **({"cpu_contact_separation": True} if is_cpu else {}),
        },
    }


def _valid_paths(tmp_path: Path) -> tuple[Path, Path]:
    gpu_path = tmp_path / "gpu.json"
    cpu_path = tmp_path / "cpu.json"
    _write_report(gpu_path, _report("cuda:0"))
    _write_report(cpu_path, _report("cpu"))
    return gpu_path, cpu_path


def _repeated_paths(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    reports_dir = tmp_path / "reports" / "runs"
    reports_dir.mkdir(parents=True)
    gpu_paths: list[Path] = []
    cpu_paths: list[Path] = []
    for index in range(3):
        gpu_path = reports_dir / f"gpu_{index + 1}.json"
        cpu_path = reports_dir / f"cpu_{index + 1}.json"
        gpu = _report("cuda:0")
        cpu = _report("cpu")
        gpu["execution"] = {
            "execution_id": uuid.uuid4().hex,
            "started_at_utc": f"2026-08-28T01:00:0{index}Z",
            "no_overwrite": True,
            "output_path_repo_relative": f"reports/runs/{gpu_path.name}",
        }
        cpu["execution"] = {
            "execution_id": uuid.uuid4().hex,
            "started_at_utc": f"2026-08-28T01:01:0{index}+00:00",
            "no_overwrite": True,
            "output_path_repo_relative": f"reports/runs/{cpu_path.name}",
        }
        gpu["pose_mode_metrics"][7]["max_nonfoot_force_bodyweights"] = 8.0 + index
        cpu["pose_mode_metrics"][7]["max_nonfoot_force_bodyweights"] = 11.0 + index
        _write_report(gpu_path, gpu)
        _write_report(cpu_path, cpu)
        gpu_paths.append(gpu_path)
        cpu_paths.append(cpu_path)
    return gpu_paths, cpu_paths


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


def test_synthesis_writer_creates_once_and_refuses_target_or_temp_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthesis.json"
    value = {"runtime_calibration_passed": True}

    SYNTHESIS._write_json_atomic(output, value)

    assert json.loads(output.read_text(encoding="utf-8")) == value
    assert not output.with_suffix(".json.tmp").exists()
    with pytest.raises(FileExistsError, match="existing synthesis"):
        SYNTHESIS._write_json_atomic(output, value)

    output.unlink()
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text("in-progress", encoding="utf-8")
    with pytest.raises(FileExistsError, match="temporary synthesis"):
        SYNTHESIS._write_json_atomic(output, value)
    assert temporary.read_text(encoding="utf-8") == "in-progress"
    assert not output.exists()


def test_strict_repeated_synthesis_requires_and_records_all_six_runs(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)

    result = SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)

    assert result["schema_version"] == 3
    assert result["synthesis_mode"] == "strict_repeated_3x_gpu_3x_cpu"
    assert result["device_pass_counts"] == {
        "gpu": {"passed": 3, "required": 3},
        "cpu": {"passed": 3, "required": 3},
    }
    assert [entry["absolute_path"] for entry in result["inputs"]["gpu"]] == [
        str(path.resolve()) for path in gpu_paths
    ]
    assert all(len(entry["sha256"]) == 64 for entry in result["inputs"]["cpu"])
    assert result["force_verification"]["all_runs_bounded"] is True
    assert len({
        entry["execution_id"]
        for device in ("gpu", "cpu")
        for entry in result["execution_lineage"][device]
    }) == 6
    assert result["execution_lineage"]["gpu"][0]["input"] == result["inputs"]["gpu"][0]
    assert result["force_verification"]["worst_case"] == {
        "device": "cpu",
        "run_index": 3,
        "bodyweights": 13.0,
        "env_index": 7,
        "pose_id": "right_side",
        "action_mode": "reset_pose_hold",
        "body_index": 11,
        "body_name": "body_7",
        "physics_step": 17,
        "input": result["inputs"]["cpu"][2],
    }
    assert result["learned_policy_qualified"] is False


def test_strict_repeated_synthesis_rejects_missing_or_reused_runs(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)

    with pytest.raises(SYNTHESIS.SynthesisError, match="exactly 3 GPU and 3 CPU"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths[:2], cpu_paths)
    with pytest.raises(SYNTHESIS.SynthesisError, match="distinct input paths"):
        SYNTHESIS.synthesize_repeated_reports(
            [gpu_paths[0], gpu_paths[0], gpu_paths[2]], cpu_paths
        )


@pytest.mark.parametrize(
    ("device", "section", "field", "message"),
    [
        ("gpu", "runtime_contract", "passed", "GPU report 2 runtime_contract"),
        ("cpu", "runtime_contract", "passed", "CPU report 2 runtime_contract"),
        ("cpu", "run_health", "passed", "CPU report 2 run_health"),
    ],
)
def test_strict_repeated_synthesis_rejects_any_failed_run(
    tmp_path: Path, device: str, section: str, field: str, message: str
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    path = (gpu_paths if device == "gpu" else cpu_paths)[1]
    report = json.loads(path.read_text(encoding="utf-8"))
    report[section][field] = False
    _write_report(path, report)

    with pytest.raises(SYNTHESIS.SynthesisError, match=message):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


def test_strict_repeated_synthesis_rejects_failed_cpu_authority(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    cpu = json.loads(cpu_paths[2].read_text(encoding="utf-8"))
    cpu["required_crosschecks"]["cpu_contact_separation"]["passed"] = False
    _write_report(cpu_paths[2], cpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match="CPU report 3 authoritative separation"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


def test_strict_repeated_synthesis_rejects_cherry_picked_force_or_missing_attribution(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    cpu = json.loads(cpu_paths[0].read_text(encoding="utf-8"))
    cpu["checks"]["nonfoot_peak_force_bounded"] = False
    _write_report(cpu_paths[0], cpu)
    with pytest.raises(SYNTHESIS.SynthesisError, match="nonfoot_peak_force_bounded"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)

    cpu["checks"]["nonfoot_peak_force_bounded"] = True
    cpu["pose_mode_metrics"][7]["max_nonfoot_force_body_name"] = None
    _write_report(cpu_paths[0], cpu)
    with pytest.raises(SYNTHESIS.SynthesisError, match="body name"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


@pytest.mark.parametrize("required_check", sorted(SYNTHESIS.STRICT_REQUIRED_CHECKS))
@pytest.mark.parametrize("tamper", ["false", "missing", "non_boolean"])
def test_strict_repeated_synthesis_recomputes_required_checks_fail_closed(
    tmp_path: Path, required_check: str, tamper: str
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    gpu = json.loads(gpu_paths[1].read_text(encoding="utf-8"))
    assert gpu["runtime_contract"]["passed"] is True
    if tamper == "false":
        gpu["checks"][required_check] = False
        expected = "aggregate disagrees with failed checks"
    elif tamper == "missing":
        del gpu["checks"][required_check]
        expected = "missing required checks"
    else:
        gpu["checks"][required_check] = 1
        expected = "checks must contain only booleans"
    _write_report(gpu_paths[1], gpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match=expected):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_strict_repeated_synthesis_requires_max_depenetration_check_on_both_devices(
    tmp_path: Path, device: str
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    for path in [*gpu_paths, *cpu_paths]:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["contract_sha256"] = SYNTHESIS.REV14_CONTRACT_SHA256
        _write_report(path, report)
    path = (gpu_paths if device == "gpu" else cpu_paths)[1]
    report = json.loads(path.read_text(encoding="utf-8"))
    del report["checks"][
        "rigid_body_max_depenetration_velocity_matches_contract"
    ]
    _write_report(path, report)

    with pytest.raises(SYNTHESIS.SynthesisError, match="missing required checks"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


def test_strict_repeated_synthesis_accepts_complete_rev14_check_contract(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    for path in [*gpu_paths, *cpu_paths]:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["contract_sha256"] = SYNTHESIS.REV14_CONTRACT_SHA256
        _write_report(path, report)

    result = SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)

    assert result["runtime_calibration_passed"] is True


def test_strict_repeated_synthesis_accepts_complete_rev15_progression_contract(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    for path in [*gpu_paths, *cpu_paths]:
        report = json.loads(path.read_text(encoding="utf-8"))
        _bind_rev15_progression(report)
        _write_report(path, report)

    result = SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)

    assert result["runtime_calibration_passed"] is True


@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_strict_repeated_synthesis_requires_rev15_progression_gate(
    tmp_path: Path, device: str
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    for path in [*gpu_paths, *cpu_paths]:
        report = json.loads(path.read_text(encoding="utf-8"))
        _bind_rev15_progression(report)
        _write_report(path, report)
    target = (gpu_paths if device == "gpu" else cpu_paths)[1]
    report = json.loads(target.read_text(encoding="utf-8"))
    del report["progression_gate"]
    _write_report(target, report)

    with pytest.raises(SYNTHESIS.SynthesisError, match="required field: progression_gate"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


def test_strict_repeated_synthesis_rejects_rev15_gpu_claiming_cpu_authority(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    for path in [*gpu_paths, *cpu_paths]:
        report = json.loads(path.read_text(encoding="utf-8"))
        _bind_rev15_progression(report)
        _write_report(path, report)
    gpu = json.loads(gpu_paths[0].read_text(encoding="utf-8"))
    gpu["progression_gate"]["cpu_contact_separation_required_this_run"] = True
    gpu["progression_gate"]["blocking_checks"]["cpu_contact_separation"] = True
    _write_report(gpu_paths[0], gpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match="device authority scope mismatch"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


def test_strict_repeated_synthesis_keeps_rev12_required_check_compatibility(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    for path in [*gpu_paths, *cpu_paths]:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["contract_sha256"] = (
            "d4b48d2b5fc1ea7684684a6324ba22fbfae767effeae45668c7310df382392e0"
        )
        del report["checks"][
            "rigid_body_max_depenetration_velocity_matches_contract"
        ]
        _write_report(path, report)

    result = SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)

    assert result["runtime_calibration_passed"] is True


def test_strict_repeated_synthesis_rejects_unknown_contract_registry_entry(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    for path in [*gpu_paths, *cpu_paths]:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["contract_sha256"] = "f" * 64
        _write_report(path, report)

    with pytest.raises(SYNTHESIS.SynthesisError, match="no registered strict check contract"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


def test_strict_repeated_synthesis_rejects_false_unlisted_check_with_true_aggregate(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    cpu = json.loads(cpu_paths[0].read_text(encoding="utf-8"))
    assert cpu["runtime_contract"]["passed"] is True
    cpu["checks"]["another_runtime_check"] = False
    _write_report(cpu_paths[0], cpu)

    with pytest.raises(
        SYNTHESIS.SynthesisError, match="aggregate disagrees with failed checks"
    ):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


def test_strict_repeated_synthesis_rejects_identity_or_threshold_mismatch(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    cpu = json.loads(cpu_paths[1].read_text(encoding="utf-8"))
    cpu["source_bundle"]["git_commit"] = "d" * 40
    _write_report(cpu_paths[1], cpu)
    with pytest.raises(SYNTHESIS.SynthesisError, match="source_bundle mismatch"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)

    cpu["source_bundle"] = _source_bundle()
    cpu["calibration_thresholds"]["max_nonfoot_force_bodyweights"] = 14.0
    _write_report(cpu_paths[1], cpu)
    with pytest.raises(SYNTHESIS.SynthesisError, match="force threshold mismatch"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


def test_cli_accepts_one_or_three_reports_per_device() -> None:
    parser = SYNTHESIS.build_parser()

    old = parser.parse_args(
        ["--gpu-report", "gpu.json", "--cpu-report", "cpu.json", "--output", "out.json"]
    )
    repeated = parser.parse_args(
        [
            "--gpu-report", "g1.json", "g2.json", "g3.json",
            "--cpu-report", "c1.json", "c2.json", "c3.json",
            "--output", "out.json",
        ]
    )

    assert len(old.gpu_report) == len(old.cpu_report) == 1
    assert len(repeated.gpu_report) == len(repeated.cpu_report) == 3


def test_cli_dispatch_allows_only_exact_legacy_contract_for_single_run(
    tmp_path: Path,
) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)
    for path in (gpu_path, cpu_path):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["contract_sha256"] = SYNTHESIS.LEGACY_SINGLE_RUN_CONTRACT_SHA256
        _write_report(path, report)

    result = SYNTHESIS.synthesize_cli_inputs([gpu_path], [cpu_path])

    assert result["runtime_calibration_passed"] is True
    assert result["schema_version"] == 2
    assert (
        result["verified_identity"]["contract_sha256"]
        == SYNTHESIS.LEGACY_SINGLE_RUN_CONTRACT_SHA256
    )


@pytest.mark.parametrize(
    "contract_sha256",
    [
        "b5499b4a8c111788c3c601fd983bb03907cb3779106821ce2a0be6ef447d5912",
        "0679" + "0" * 60,
        "f" * 64,
    ],
    ids=["rev10", "rev11", "arbitrary"],
)
def test_cli_dispatch_rejects_nonlegacy_single_run_contract(
    tmp_path: Path, contract_sha256: str
) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)
    for path in (gpu_path, cpu_path):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["contract_sha256"] = contract_sha256
        _write_report(path, report)

    with pytest.raises(SYNTHESIS.SynthesisError, match=r"require strict 3\+3 reports"):
        SYNTHESIS.synthesize_cli_inputs([gpu_path], [cpu_path])


def test_cli_dispatch_rejects_mixed_legacy_and_new_single_run_contracts(
    tmp_path: Path,
) -> None:
    gpu_path, cpu_path = _valid_paths(tmp_path)
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    gpu["contract_sha256"] = SYNTHESIS.LEGACY_SINGLE_RUN_CONTRACT_SHA256
    _write_report(gpu_path, gpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match="exact legacy contract"):
        SYNTHESIS.synthesize_cli_inputs([gpu_path], [cpu_path])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task", "Isaac-G009-Another-Task-v0", "task must equal"),
        ("seed", 7, "seed must be integer 42"),
        ("seed", True, "seed must be integer 42"),
        ("headless", False, "headless must be boolean true"),
        ("headless", 1, "headless must be boolean true"),
    ],
)
def test_strict_repeated_synthesis_requires_exact_execution_conditions(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    for path in [*gpu_paths, *cpu_paths]:
        report = json.loads(path.read_text(encoding="utf-8"))
        report[field] = value
        _write_report(path, report)

    with pytest.raises(SYNTHESIS.SynthesisError, match=message):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


def test_strict_repeated_synthesis_rejects_missing_headless(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    cpu = json.loads(cpu_paths[1].read_text(encoding="utf-8"))
    del cpu["headless"]
    _write_report(cpu_paths[1], cpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match="missing required field: headless"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("execution_id", "not-a-uuid", "execution_id must be UUID4 hex"),
        ("execution_id", uuid.uuid1().hex, "execution_id must be UUID4 hex"),
        ("started_at_utc", "not-a-time", "valid UTC timestamp"),
        ("started_at_utc", "2026-08-28T10:00:00+09:00", "valid UTC timestamp"),
        ("no_overwrite", False, "no_overwrite must be true"),
        ("no_overwrite", 1, "no_overwrite must be true"),
        (
            "output_path_repo_relative",
            "reports/runs/tampered.json",
            "output path binding mismatch",
        ),
    ],
)
def test_strict_repeated_synthesis_rejects_tampered_execution_provenance(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    cpu = json.loads(cpu_paths[1].read_text(encoding="utf-8"))
    cpu["execution"][field] = value
    _write_report(cpu_paths[1], cpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match=message):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


def test_strict_repeated_synthesis_requires_execution_object(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    gpu = json.loads(gpu_paths[0].read_text(encoding="utf-8"))
    del gpu["execution"]
    _write_report(gpu_paths[0], gpu)

    with pytest.raises(SYNTHESIS.SynthesisError, match="missing required field: execution"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


def test_strict_repeated_synthesis_rejects_json_copied_to_another_path(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    copied = json.loads(gpu_paths[0].read_text(encoding="utf-8"))
    _write_report(gpu_paths[1], copied)

    with pytest.raises(SYNTHESIS.SynthesisError, match="output path binding mismatch"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)


def test_strict_repeated_synthesis_rejects_duplicate_execution_id(
    tmp_path: Path,
) -> None:
    gpu_paths, cpu_paths = _repeated_paths(tmp_path)
    gpu_first = json.loads(gpu_paths[0].read_text(encoding="utf-8"))
    gpu_second = json.loads(gpu_paths[1].read_text(encoding="utf-8"))
    gpu_second["execution"]["execution_id"] = gpu_first["execution"]["execution_id"]
    _write_report(gpu_paths[1], gpu_second)

    with pytest.raises(SYNTHESIS.SynthesisError, match="6 unique execution_id"):
        SYNTHESIS.synthesize_repeated_reports(gpu_paths, cpu_paths)
