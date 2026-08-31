from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_g009_r0_rev22_read_only_matrix_observation_adapter.py"
SPEC = importlib.util.spec_from_file_location("rev22_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def fake_source_binding() -> dict[str, object]:
    files = {path: RUNNER.sha256_bytes(f"blob:{path}".encode()) for path in RUNNER.evaluator.REQUIRED_SOURCE_PATHS}
    payload = "\n".join(f"{path}:{files[path]}" for path in RUNNER.evaluator.REQUIRED_SOURCE_PATHS)
    return {"schema_version": 1, "git_commit": "b" * 40, "source_binding_paths": list(RUNNER.evaluator.REQUIRED_SOURCE_PATHS), "source_binding_files": files, "source_bundle_sha256": RUNNER.sha256_bytes(payload.encode()), "path_scoped_clean": True}


def canonical_execution() -> dict[str, object]:
    return {"execution_id": "1234567890ab4def81234567890abcde", "started_at_utc": "2026-08-31T00:00:00Z", "output_path_repo_relative": "reports/runs/g009_r0_rev22_read_only_matrix_observation_adapter_preregistration_s42.json", "no_overwrite": True}


def configure_passing_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    prereg_raw = RUNNER.PREREGISTRATION_PATH.read_bytes()
    prereg = RUNNER.evaluator.validate_preregistration_bytes(prereg_raw)
    predecessor_raw = RUNNER.PREDECESSOR_PATH.read_bytes()
    output = tmp_path / "gate.json"
    source = fake_source_binding()
    monkeypatch.setattr(RUNNER, "_load_preregistration", lambda: (prereg, prereg_raw))
    monkeypatch.setattr(RUNNER, "_validate_output", lambda _prereg: output)
    monkeypatch.setattr(RUNNER, "source_bundle_provenance", lambda _prereg: source)
    monkeypatch.setattr(RUNNER, "_load_predecessor", lambda _prereg: predecessor_raw)
    return output


def test_exclusive_writer_creates_once_and_preserves_existing(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    RUNNER.write_json_exclusive_owned(output, {"passed": True})
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        RUNNER.write_json_exclusive_owned(output, {"passed": False})
    assert output.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_exclusive_install_race_preserves_competing_final(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    competing = b"competitor"

    def link_with_competitor(_source, destination) -> None:
        Path(destination).write_bytes(competing)
        raise FileExistsError("race")

    with pytest.raises(FileExistsError):
        RUNNER.write_json_exclusive_owned(output, {"passed": True}, link=link_with_competitor)
    assert output.read_bytes() == competing
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(("stage", "expected"), [("temp", "runner_temp_write_failed"), ("final", "runner_final_install_failed")])
def test_run_maps_writer_stage_to_operational_reason(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str, expected: str) -> None:
    configure_passing_run(monkeypatch, tmp_path)
    monkeypatch.setattr(RUNNER, "write_json_exclusive_owned", lambda *_args, **_kwargs: (_ for _ in ()).throw(RUNNER.ExclusiveWriteFailure(stage, "injected")))
    with pytest.raises(RUNNER.OperationalFailure) as error:
        RUNNER.run(check_only=False, _execution=canonical_execution())
    assert error.value.reason == expected


def test_check_only_is_completely_read_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = configure_passing_run(monkeypatch, tmp_path)
    before = list(tmp_path.rglob("*"))
    artifact = RUNNER.run(check_only=True, _execution=canonical_execution())
    assert artifact["decision"]["outcome"] == RUNNER.PASS_REASON
    assert not output.exists()
    assert list(tmp_path.rglob("*")) == before


def test_rejected_evaluator_preserves_exact_reason_ledger() -> None:
    reason = "adapter_filter_reduction_invalid"
    projection = {"decision": {"passed": False, "outcome": reason, "primary_reason": reason, "next_step": "stop"}, "checks": RUNNER.evaluator.complete_reason_ledger([], reason)}
    rejection = RUNNER._evaluation_failure(projection)
    assert isinstance(rejection, RUNNER.GateReject)
    assert rejection.reason == reason
    assert rejection.checks[RUNNER.evaluator.REASON_PRIORITY.index(reason)]["status"] == "fail"


def test_source_binding_includes_rev21_verifier_dependency_and_rejects_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    prereg = RUNNER.evaluator.validate_preregistration_bytes(RUNNER.PREREGISTRATION_PATH.read_bytes())
    assert "scripts/summarize_g009_r0_rev21_matrix_authority_safety_gate.py" in RUNNER.evaluator.REQUIRED_SOURCE_PATHS
    monkeypatch.setattr(RUNNER, "_git_commit", lambda: "b" * 40)

    def dirty_git(args: list[str], *, missing_reason: str) -> bytes:
        if args and args[0] == "status":
            return b" M scripts/summarize_g009_r0_rev21_matrix_authority_safety_gate.py\n"
        return b""

    monkeypatch.setattr(RUNNER, "_git_bytes", dirty_git)
    with pytest.raises(RUNNER.GateReject) as error:
        RUNNER.source_bundle_provenance(prereg)
    assert error.value.reason == "rev22_source_provenance_invalid"


def test_predecessor_loader_calls_fresh_rev21_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    prereg = RUNNER.evaluator.validate_preregistration_bytes(RUNNER.PREREGISTRATION_PATH.read_bytes())
    called: list[Path] = []
    monkeypatch.setattr(RUNNER.evaluator, "verify_predecessor_fresh", lambda path: called.append(path) or {})
    raw = RUNNER._load_predecessor(prereg)
    assert RUNNER.sha256_bytes(raw) == prereg["predecessor"]["sha256"]
    assert called == [RUNNER.PREDECESSOR_PATH.resolve(strict=True)]


def test_output_existing_has_priority_over_source_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    prereg_raw = RUNNER.PREREGISTRATION_PATH.read_bytes()
    prereg = RUNNER.evaluator.validate_preregistration_bytes(prereg_raw)
    monkeypatch.setattr(RUNNER, "_load_preregistration", lambda: (prereg, prereg_raw))
    monkeypatch.setattr(RUNNER, "_validate_output", lambda _prereg: (_ for _ in ()).throw(RUNNER.GateReject("canonical_output_already_exists", "exists")))
    monkeypatch.setattr(RUNNER, "source_bundle_provenance", lambda _prereg: (_ for _ in ()).throw(AssertionError("must not run")))
    with pytest.raises(RUNNER.GateReject) as error:
        RUNNER.run(check_only=True, _execution=canonical_execution())
    assert error.value.reason == "canonical_output_already_exists"


def test_failure_envelope_keeps_governance_zero() -> None:
    error = RUNNER.GateReject("adapter_dtype_device_or_shape_invalid", "bad")
    value = RUNNER._failure_envelope(error, canonical_execution(), 2)
    assert value["governance"]["reward_computed"] is False
    assert value["governance"]["ppo_updates"] == 0
    assert value["primary_reason"] == "adapter_dtype_device_or_shape_invalid"
