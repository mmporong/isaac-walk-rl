from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_g009_r0_rev21_matrix_authority_safety_gate.py"
SPEC = importlib.util.spec_from_file_location("rev21_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def temp_artifacts(parent: Path) -> list[Path]:
    return [path for path in parent.iterdir() if ".tmp" in path.name or path.name.endswith(".part")]


def configure_passing_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, dict]:
    output = tmp_path / "reports/runs/gate.json"
    preregistration = json.loads(
        (ROOT / "configs/g009_r0_rev21_matrix_authority_safety_gate.json").read_bytes()
    )
    monkeypatch.setattr(RUNNER, "OUTPUT_PATH", output)
    monkeypatch.setattr(RUNNER, "_load_preregistration", lambda: (preregistration, b"{}"))
    monkeypatch.setattr(RUNNER, "_validate_output_path", lambda _prereg: output)
    monkeypatch.setattr(
        RUNNER, "source_bundle_provenance", lambda _prereg: {"path_scoped_clean": True}
    )
    monkeypatch.setattr(RUNNER, "_load_evidence", lambda _prereg: (b"{}", {}, b"{}", {}))
    monkeypatch.setattr(
        RUNNER.evaluator,
        "evaluate_evidence",
        lambda *_args: {
            "decision": {
                "passed": True,
                "primary_reason": "matrix_authority_safety_gate_passed_for_diagnostic_preregistration",
                "outcome": "matrix_authority_safety_gate_passed_for_diagnostic_preregistration",
                "next_step": "preregister_read_only_matrix_observation_adapter",
            },
            "checks": [
                {"reason": reason, "status": "pass"}
                for reason in RUNNER.evaluator.REASON_PRIORITY
            ],
        },
    )
    return output, preregistration


def test_runner_is_import_free_of_isaac_and_app_launcher() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "AppLauncher" not in source
    assert "import isaac" not in source.lower()


def test_preregistration_fixes_source_order_without_expected_self_aggregate() -> None:
    preregistration = json.loads(
        (ROOT / "configs/g009_r0_rev21_matrix_authority_safety_gate.json").read_bytes()
    )
    binding = preregistration["rev21_source_binding"]
    assert binding["ordered_paths"] == [
        "configs/g009_r0_rev21_matrix_authority_safety_gate.json",
        "scripts/summarize_g009_r0_rev21_matrix_authority_safety_gate.py",
        "scripts/run_g009_r0_rev21_matrix_authority_safety_gate.py",
    ]
    assert binding["expected_aggregate_sha256"] is None


def test_exclusive_writer_creates_final_once_and_removes_owned_temp(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    RUNNER.write_json_exclusive_owned(output, {"passed": True})
    assert json.loads(output.read_bytes()) == {"passed": True}
    assert temp_artifacts(tmp_path) == []


def test_exclusive_writer_preserves_preexisting_final_bytes(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    output.write_bytes(b"other-process")
    with pytest.raises(FileExistsError):
        RUNNER.write_json_exclusive_owned(output, {"passed": True})
    assert output.read_bytes() == b"other-process"
    assert temp_artifacts(tmp_path) == []


def test_exclusive_install_race_preserves_competing_final_and_removes_only_owned_temp(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"

    def competing_link(_temporary: Path, final: Path) -> None:
        final.write_bytes(b"race-winner")
        raise FileExistsError(final)

    with pytest.raises(FileExistsError):
        RUNNER.write_json_exclusive_owned(output, {"passed": True}, link=competing_link)
    assert output.read_bytes() == b"race-winner"
    assert temp_artifacts(tmp_path) == []


def test_final_install_operational_error_never_unlinks_another_process_final(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"

    def failing_link(_temporary: Path, final: Path) -> None:
        final.write_bytes(b"unrelated-final")
        raise PermissionError("injected final install failure")

    with pytest.raises(RUNNER.ExclusiveWriteFailure, match="injected") as error:
        RUNNER.write_json_exclusive_owned(output, {"passed": True}, link=failing_link)
    assert error.value.stage == "final"
    assert output.read_bytes() == b"unrelated-final"
    assert temp_artifacts(tmp_path) == []


def test_check_only_pass_is_completely_read_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output, _preregistration = configure_passing_run(monkeypatch, tmp_path)
    failure_root = tmp_path / "failed_attempts"
    monkeypatch.setattr(RUNNER, "FAILURE_ROOT", failure_root)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = RUNNER.run(check_only=True)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["decision"]["primary_reason"] == "matrix_authority_safety_gate_passed_for_diagnostic_preregistration"
    assert after == before


def test_check_only_rejects_existing_output_without_failure_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "reports/runs/gate.json"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing")
    failure_root = tmp_path / "failed_attempts"
    monkeypatch.setattr(RUNNER, "OUTPUT_PATH", output)
    monkeypatch.setattr(RUNNER, "FAILURE_ROOT", failure_root)
    monkeypatch.setattr(
        RUNNER,
        "run",
        lambda **_kwargs: (_ for _ in ()).throw(
            RUNNER.GateReject("canonical_output_already_exists", "existing")
        ),
    )
    assert RUNNER.main(["--check-only"]) == 2
    assert output.read_bytes() == b"existing"
    assert not failure_root.exists()
    assert temp_artifacts(output.parent) == []


def test_main_returns_zero_for_pass_two_for_contract_reject_and_three_for_operational_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passed = {
        "decision": {
            "primary_reason": "matrix_authority_safety_gate_passed_for_diagnostic_preregistration",
            "outcome": "matrix_authority_safety_gate_passed_for_diagnostic_preregistration",
        }
    }
    monkeypatch.setattr(RUNNER, "run", lambda **_kwargs: passed)
    assert RUNNER.main(["--check-only"]) == 0
    monkeypatch.setattr(
        RUNNER,
        "run",
        lambda **_kwargs: (_ for _ in ()).throw(
            RUNNER.GateReject("rev20_source_provenance_invalid", "injected reject")
        ),
    )
    assert RUNNER.main(["--check-only"]) == 2
    monkeypatch.setattr(
        RUNNER,
        "run",
        lambda **_kwargs: (_ for _ in ()).throw(
            RUNNER.OperationalFailure("runner_input_io_error", "injected operational failure")
        ),
    )
    assert RUNNER.main(["--check-only"]) == 3


def test_source_provenance_rejects_dirty_bound_path(monkeypatch: pytest.MonkeyPatch) -> None:
    preregistration = json.loads(
        (ROOT / "configs/g009_r0_rev21_matrix_authority_safety_gate.json").read_bytes()
    )
    monkeypatch.setattr(RUNNER, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        RUNNER,
        "_run_git_bytes",
        lambda args, **_kwargs: b" M scripts/run_g009_r0_rev21_matrix_authority_safety_gate.py\n"
        if args and args[0] == "status"
        else b"fixture",
    )
    with pytest.raises(RUNNER.GateReject, match="path-scoped clean") as error:
        RUNNER.source_bundle_provenance(preregistration)
    assert error.value.reason == "rev21_source_provenance_invalid"


def test_source_provenance_rejects_untracked_bound_path(monkeypatch: pytest.MonkeyPatch) -> None:
    preregistration = json.loads(
        (ROOT / "configs/g009_r0_rev21_matrix_authority_safety_gate.json").read_bytes()
    )
    monkeypatch.setattr(RUNNER, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        RUNNER,
        "_run_git_bytes",
        lambda args, **_kwargs: b"?? scripts/run_g009_r0_rev21_matrix_authority_safety_gate.py\n"
        if args and args[0] == "status"
        else b"fixture",
    )
    with pytest.raises(RUNNER.GateReject) as error:
        RUNNER.source_bundle_provenance(preregistration)
    assert error.value.reason == "rev21_source_provenance_invalid"


def test_source_provenance_rejects_head_blob_worktree_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    preregistration = json.loads(
        (ROOT / "configs/g009_r0_rev21_matrix_authority_safety_gate.json").read_bytes()
    )
    paths = preregistration["rev21_source_binding"]["ordered_paths"]
    files: dict[str, Path] = {}
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"worktree:{relative}".encode())
        files[relative] = path
    monkeypatch.setattr(RUNNER, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(RUNNER, "_run_git_bytes", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(RUNNER, "_canonical_repo_path", lambda relative, **_kwargs: files[relative])
    monkeypatch.setattr(RUNNER, "_git_blob", lambda _commit, relative, **_kwargs: b"different" if relative == paths[1] else files[relative].read_bytes())
    with pytest.raises(RUNNER.GateReject) as error:
        RUNNER.source_bundle_provenance(preregistration)
    assert error.value.reason == "rev21_source_provenance_invalid"
    assert "worktree bytes differ" in str(error.value)


def test_source_provenance_aggregate_uses_exact_declared_order_without_trailing_lf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    preregistration = json.loads(
        (ROOT / "configs/g009_r0_rev21_matrix_authority_safety_gate.json").read_bytes()
    )
    paths = preregistration["rev21_source_binding"]["ordered_paths"]
    files: dict[str, Path] = {}
    for index, relative in enumerate(paths):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"blob-{index}".encode())
        files[relative] = path
    monkeypatch.setattr(RUNNER, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(RUNNER, "_run_git_bytes", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(RUNNER, "_canonical_repo_path", lambda relative, **_kwargs: files[relative])
    monkeypatch.setattr(RUNNER, "_git_blob", lambda _commit, relative, **_kwargs: files[relative].read_bytes())
    result = RUNNER.source_bundle_provenance(preregistration)
    expected_files = {
        relative: hashlib.sha256(files[relative].read_bytes()).hexdigest()
        for relative in paths
    }
    serialized = "\n".join(f"{relative}:{expected_files[relative]}" for relative in paths).encode()
    assert result["source_binding_paths"] == paths
    assert list(result["source_binding_files"]) == paths
    assert result["source_binding_files"] == expected_files
    assert result["source_bundle_sha256"] == hashlib.sha256(serialized).hexdigest()
    assert not serialized.endswith(b"\n")


@pytest.mark.parametrize(
    ("stage", "expected_reason"),
    [
        ("temp", "runner_temp_write_failed"),
        ("final", "runner_final_install_failed"),
    ],
)
def test_run_maps_exclusive_writer_stage_to_operational_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str, expected_reason: str
) -> None:
    output, _preregistration = configure_passing_run(monkeypatch, tmp_path)
    monkeypatch.setattr(
        RUNNER,
        "write_json_exclusive_owned",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RUNNER.ExclusiveWriteFailure(stage, "injected writer failure")
        ),
    )
    with pytest.raises(RUNNER.OperationalFailure) as error:
        RUNNER.run(check_only=False)
    assert error.value.reason == expected_reason
    assert not output.exists()


def test_failure_envelope_write_failure_is_reported_as_operational_exit_three(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(RUNNER, "FAILURE_ROOT", tmp_path / "failed")
    monkeypatch.setattr(
        RUNNER,
        "run",
        lambda **_kwargs: (_ for _ in ()).throw(
            RUNNER.GateReject("rev20_source_provenance_invalid", "injected reject")
        ),
    )
    monkeypatch.setattr(
        RUNNER,
        "write_json_exclusive_owned",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RUNNER.ExclusiveWriteFailure("temp", "injected envelope failure")
        ),
    )
    assert RUNNER.main([]) == 3
    payload = json.loads(capsys.readouterr().err)
    assert payload["primary_reason"] == "runner_failure_envelope_write_failed"
    assert not (tmp_path / "failed").exists()


def test_check_only_operational_error_is_completely_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "existing.txt"
    marker.write_bytes(b"unchanged")
    monkeypatch.setattr(RUNNER, "FAILURE_ROOT", tmp_path / "failed")
    monkeypatch.setattr(
        RUNNER,
        "run",
        lambda **_kwargs: (_ for _ in ()).throw(
            RUNNER.OperationalFailure("runner_input_io_error", "injected")
        ),
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert RUNNER.main(["--check-only"]) == 3
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_runner_pass_artifact_records_all_seventeen_checks_in_priority_order() -> None:
    evaluator_checks = [
        {"reason": reason, "status": "pass"}
        for reason in RUNNER.evaluator.REASON_PRIORITY
    ]
    projection = {
        "decision": {
            "passed": True,
            "outcome": RUNNER.PASS_REASON,
            "primary_reason": RUNNER.PASS_REASON,
            "next_step": "preregister_read_only_matrix_observation_adapter",
        },
        "checks": evaluator_checks,
    }
    artifact = RUNNER._artifact(
        projection,
        {"path_scoped_clean": True},
        {
            "execution_id": "1234567890ab4def81234567890abcde",
            "started_at_utc": "2026-08-30T00:00:00Z",
            "output_path_repo_relative": "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json",
            "no_overwrite": True,
        },
    )
    assert [item["reason"] for item in artifact["checks"]] == list(
        RUNNER.evaluator.REASON_PRIORITY
    )
    assert [item["status"] for item in artifact["checks"]] == ["pass"] * 17


def test_evaluator_rejection_check_list_is_preserved_in_runner_failure_ledger() -> None:
    checks = []
    for index, reason in enumerate(RUNNER.evaluator.REASON_PRIORITY):
        status = "pass" if index < 9 else "fail" if index == 9 else "not_evaluated"
        item = {"reason": reason, "status": status}
        if status == "not_evaluated":
            item["detail"] = "depends on rev20_evidence_chain_binding_invalid"
        checks.append(item)
    rejection = RUNNER._evaluation_failure(
        {
            "decision": {
                "passed": False,
                "outcome": "rev20_evidence_chain_binding_invalid",
                "primary_reason": "rev20_evidence_chain_binding_invalid",
                "next_step": "stop_and_repair_evidence_chain",
            },
            "checks": checks,
        }
    )
    assert rejection is not None
    assert rejection.checks == checks
    execution = {
        "execution_id": "1234567890ab4def81234567890abcde",
        "started_at_utc": "2026-08-30T00:00:00Z",
        "output_path_repo_relative": "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json",
        "no_overwrite": True,
    }
    envelope = RUNNER._failure_envelope(rejection, execution, 2)
    assert envelope["checks"] == checks


@pytest.mark.parametrize(
    "projection",
    [
        {
            "decision": {
                "passed": False,
                "outcome": RUNNER.PASS_REASON,
                "primary_reason": RUNNER.PASS_REASON,
                "next_step": "preregister_read_only_matrix_observation_adapter",
            },
            "checks": [
                {"reason": reason, "status": "pass"}
                for reason in RUNNER.evaluator.REASON_PRIORITY
            ],
        },
        {
            "decision": {
                "passed": True,
                "outcome": RUNNER.PASS_REASON,
                "primary_reason": RUNNER.PASS_REASON,
                "next_step": "preregister_read_only_matrix_observation_adapter",
            },
            "checks": [
                {
                    "reason": reason,
                    "status": "fail" if reason == "rev20_matrix_numeric_invalid" else "pass",
                }
                for reason in RUNNER.evaluator.REASON_PRIORITY
            ],
        },
    ],
)
def test_contradictory_pass_projection_is_internal_error(projection: dict) -> None:
    with pytest.raises(RUNNER.OperationalFailure) as error:
        RUNNER._artifact(
            projection,
            {"path_scoped_clean": True},
            {
                "execution_id": "1234567890ab4def81234567890abcde",
                "started_at_utc": "2026-08-30T00:00:00Z",
                "output_path_repo_relative": "reports/runs/g009_r0_rev21_matrix_authority_safety_gate_s42.json",
                "no_overwrite": True,
            },
        )
    assert error.value.reason == "runner_internal_error"


def test_unexpected_evaluator_runtime_error_is_check_only_exit_three_and_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_passing_run(monkeypatch, tmp_path)
    marker = tmp_path / "marker.txt"
    marker.write_bytes(b"unchanged")
    monkeypatch.setattr(RUNNER, "FAILURE_ROOT", tmp_path / "failed")
    monkeypatch.setattr(
        RUNNER.evaluator,
        "evaluate_evidence",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("injected evaluator bug")),
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert RUNNER.main(["--check-only"]) == 3
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_git_reads_disable_optional_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = b"ok"
        stderr = b""

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs.get("env")
        return Completed()

    monkeypatch.setattr(RUNNER.subprocess, "run", fake_run)
    assert RUNNER._run_git_bytes(["status", "--porcelain=v1"], missing_reason="x") == b"ok"
    command = observed["command"]
    environment = observed["env"]
    assert "--no-optional-locks" in command or (
        isinstance(environment, dict) and environment.get("GIT_OPTIONAL_LOCKS") == "0"
    )
    assert isinstance(environment, dict)
    assert environment["LC_ALL"] == "C"
    assert environment["LANG"] == "C"


def test_unexpected_git_permission_failure_is_operational_exit_three_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Failed:
        returncode = 128
        stdout = b""
        stderr = b"fatal: Permission denied while reading the index"

    monkeypatch.setattr(RUNNER.subprocess, "run", lambda *_args, **_kwargs: Failed())
    monkeypatch.setattr(RUNNER, "FAILURE_ROOT", tmp_path / "failed")
    marker = tmp_path / "marker.txt"
    marker.write_bytes(b"unchanged")

    with pytest.raises(RUNNER.OperationalFailure) as error:
        RUNNER._run_git_bytes(["status", "--porcelain=v1"], missing_reason="x")
    assert error.value.reason == "runner_input_io_error"

    monkeypatch.setattr(
        RUNNER,
        "run",
        lambda **_kwargs: RUNNER._run_git_bytes(
            ["status", "--porcelain=v1"], missing_reason="x"
        ),
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert RUNNER.main(["--check-only"]) == 3
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_missing_git_blob_remains_a_deterministic_contract_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Missing:
        returncode = 128
        stdout = b""
        stderr = b"fatal: path 'missing.py' does not exist in 'abc123'"

    monkeypatch.setattr(RUNNER.subprocess, "run", lambda *_args, **_kwargs: Missing())
    with pytest.raises(RUNNER.GateReject) as error:
        RUNNER._run_git_bytes(
            ["show", "abc123:missing.py"],
            missing_reason="rev20_source_provenance_invalid",
        )
    assert error.value.reason == "rev20_source_provenance_invalid"
