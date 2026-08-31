from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import probe_g009_r0_rev23_matrix_observation_adapter_runtime as probe  # noqa: E402
from isaac_walk_g009.matrix_observation_adapter import MatrixObservation  # noqa: E402


def _source() -> torch.Tensor:
    value = torch.zeros((8, 19, 1, 3), dtype=torch.float32)
    value[:, 0, 0, 2] = 9.81
    value[:, 4, 0, 0] = 3.0
    value[:, 4, 0, 1] = 4.0
    return value


def _sensor(source: torch.Tensor | None = None) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(force_matrix_w=_source() if source is None else source))


def _execution(device: str = "cpu", replicate: int = 1) -> dict[str, object]:
    return {
        "execution_id": uuid.uuid4().hex,
        "started_at_utc": "2026-08-31T00:00:00.000000Z",
        "output_path_repo_relative": probe.expected_output_relative(device, replicate),
        "no_overwrite": True,
    }


def _complete_adapter_runtime() -> dict[str, object]:
    accumulator = probe.AdapterRuntimeAccumulator(
        "cpu", "05105dbb7cf8646d0c7a5bf667cc9ab78de76131819a9654e43d9465a31d5b43"
    )
    sensor = _sensor()
    for step in range(1, 151):
        accumulator.observe(step, sensor, torch)
    return accumulator.snapshot()


def test_preregistration_matches_runtime_contract() -> None:
    value = probe.load_preregistration()
    assert value["evidence_id"] == "G009-5-E016"
    assert value["runtime"]["canonical_slot_order"] == [
        "cpu.rep1",
        "cpu.rep2",
        "cuda:0.rep1",
        "cuda:0.rep2",
    ]
    assert value["adapter_implementation"]["source_quantity_semantics"] == (
        "world_frame_filtered_normal_contact_force_vector"
    )


def test_predecessor_passes_full_rev22_verification() -> None:
    value = probe.validate_predecessor()
    assert value["full_verification_passed"] is True
    assert value["outcome"] == "read_only_matrix_observation_adapter_preregistration_passed"
    assert value["adapter_contract_sha256"] == (
        "05105dbb7cf8646d0c7a5bf667cc9ab78de76131819a9654e43d9465a31d5b43"
    )


@pytest.mark.parametrize(
    ("device", "replicate", "expected"),
    [
        ("cpu", 1, "reports/runs/g009_r0_rev23_matrix_observation_adapter_cpu_rep01_s42.json"),
        ("cpu", 2, "reports/runs/g009_r0_rev23_matrix_observation_adapter_cpu_rep02_s42.json"),
        ("cuda:0", 1, "reports/runs/g009_r0_rev23_matrix_observation_adapter_gpu_rep01_s42.json"),
        ("cuda:0", 2, "reports/runs/g009_r0_rev23_matrix_observation_adapter_gpu_rep02_s42.json"),
    ],
)
def test_expected_output_paths_are_canonical(device: str, replicate: int, expected: str) -> None:
    assert probe.expected_output_relative(device, replicate) == expected


def test_invalid_output_slot_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid rev23 runtime slot"):
        probe.expected_output_relative("cuda:1", 1)


def test_execution_validation_rejects_wrong_output_binding() -> None:
    execution = _execution()
    execution["output_path_repo_relative"] = probe.expected_output_relative("cpu", 2)
    with pytest.raises(ValueError, match="canonical no-overwrite"):
        probe.validate_execution(execution, "cpu", 1)


def test_runtime_contract_preserves_no_learning_governance() -> None:
    contract = probe.runtime_contract("cuda:0", 2, "a" * 64)
    assert contract["runtime"] == {
        "num_envs": 8,
        "physics_steps": 150,
        "physics_dt_s": 0.005,
        "headless": True,
        "fast_shutdown": False,
        "render": False,
    }
    assert contract["governance"]["reward_computed"] is False
    assert contract["governance"]["ppo_updates"] == 0


def test_accumulator_validates_all_150_steps_without_mutating_source() -> None:
    source = _source()
    sensor = _sensor(source)
    before = source.clone()
    before_pointer = source.untyped_storage().data_ptr()
    before_version = source._version
    accumulator = probe.AdapterRuntimeAccumulator("cpu", "a" * 64)

    for step in range(1, 151):
        accumulator.observe(step, sensor, torch)

    value = accumulator.snapshot()
    assert value["passed"] is True
    assert value["sample_count"] == 150
    assert value["source_mutation_steps"] == []
    assert value["oracle_mismatch_steps"] == []
    assert value["alias_violation_steps"] == []
    assert value["zero_source_vector_count_total"] > 0
    assert [item["step"] for item in value["representative_snapshots"]] == [1, 50, 100, 150]
    assert torch.equal(source, before)
    assert source.untyped_storage().data_ptr() == before_pointer
    assert source._version == before_version


def test_accumulator_supports_noncontiguous_runtime_source() -> None:
    backing = torch.zeros((8, 19, 1, 6), dtype=torch.float32)
    source = backing[..., ::2]
    source[:, 0, 0, 2] = 1.0
    assert not source.is_contiguous()
    accumulator = probe.AdapterRuntimeAccumulator("cpu", "a" * 64)
    for step in range(1, 151):
        accumulator.observe(step, _sensor(source), torch)
    assert accumulator.snapshot()["passed"] is True


def test_accumulator_records_nonfinite_source_as_fail_closed() -> None:
    source = _source()
    source[0, 0, 0, 0] = torch.nan
    accumulator = probe.AdapterRuntimeAccumulator("cpu", "a" * 64)
    accumulator.observe(1, _sensor(source), torch)
    value = accumulator.snapshot()
    assert value["passed"] is False
    assert value["error"].startswith("MatrixObservationSourceNonFiniteError")
    assert value["step_ledger"][0]["step"] == 1


def test_accumulator_detects_adapter_source_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    import isaac_walk_g009.matrix_observation_adapter as adapter

    def mutating_adapter(source: torch.Tensor) -> MatrixObservation:
        source.add_(1.0)
        world_xyz = source.sum(dim=2)
        magnitude = torch.linalg.vector_norm(world_xyz, dim=-1)
        return MatrixObservation(world_xyz, magnitude, magnitude > probe.CONTACT_THRESHOLD_N)

    monkeypatch.setattr(adapter, "adapt_terrain_pair_force_matrix_w", mutating_adapter)
    accumulator = probe.AdapterRuntimeAccumulator("cpu", "a" * 64)
    accumulator.observe(1, _sensor(), torch)
    value = accumulator.snapshot()
    assert value["passed"] is False
    assert value["source_mutation_steps"] == [1]
    assert value["oracle_mismatch_steps"] == [1]


def test_accumulator_detects_source_alias_output(monkeypatch: pytest.MonkeyPatch) -> None:
    import isaac_walk_g009.matrix_observation_adapter as adapter

    def aliasing_adapter(source: torch.Tensor) -> MatrixObservation:
        world_xyz = source[:, :, 0, :]
        magnitude = torch.linalg.vector_norm(world_xyz, dim=-1)
        return MatrixObservation(world_xyz, magnitude, magnitude > probe.CONTACT_THRESHOLD_N)

    monkeypatch.setattr(adapter, "adapt_terrain_pair_force_matrix_w", aliasing_adapter)
    accumulator = probe.AdapterRuntimeAccumulator("cpu", "a" * 64)
    accumulator.observe(1, _sensor(), torch)
    value = accumulator.snapshot()
    assert value["passed"] is False
    assert value["alias_violation_steps"] == [1]


def test_validate_adapter_runtime_accepts_complete_snapshot() -> None:
    probe.validate_adapter_runtime(_complete_adapter_runtime(), "cpu")


def test_validate_adapter_runtime_rejects_violation_ledger() -> None:
    value = _complete_adapter_runtime()
    value["source_mutation_steps"] = [1]
    with pytest.raises(ValueError, match="violation ledger"):
        probe.validate_adapter_runtime(value, "cpu")


def test_validate_adapter_runtime_rejects_step_only_ledger_tamper() -> None:
    value = _complete_adapter_runtime()
    value["step_ledger"] = [{"step": step} for step in range(1, 151)]
    with pytest.raises(ValueError, match="ledger row schema"):
        probe.validate_adapter_runtime(value, "cpu")


def test_validate_adapter_runtime_rejects_alias_pointer_tamper() -> None:
    value = _complete_adapter_runtime()
    row = value["step_ledger"][0]
    row["output_metadata"]["world_xyz"]["storage_data_ptr"] = row["source_before"]["storage_data_ptr"]
    with pytest.raises(ValueError, match="tensor alias metadata"):
        probe.validate_adapter_runtime(value, "cpu")


def test_validate_adapter_runtime_rejects_aggregate_tamper() -> None:
    value = _complete_adapter_runtime()
    value["magnitude_integral_n_s"] += 1.0
    with pytest.raises(ValueError, match="aggregate mismatch"):
        probe.validate_adapter_runtime(value, "cpu")


def test_validate_adapter_runtime_rejects_representative_tensor_tamper() -> None:
    value = _complete_adapter_runtime()
    value["representative_snapshots"][0]["world_xyz"][0][0][2] += 1.0
    with pytest.raises(ValueError, match="snapshot oracle mismatch"):
        probe.validate_adapter_runtime(value, "cpu")


def test_source_bundle_rejects_dirty_bound_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_git(args: list[str]) -> bytes:
        if args[:2] == ["rev-parse", "HEAD"]:
            return ("a" * 40 + "\n").encode()
        if args[:2] == ["status", "--porcelain=v1"]:
            return b" M scripts/probe_g009_r0_rev23_matrix_observation_adapter_runtime.py\n"
        raise AssertionError(args)

    monkeypatch.setattr(probe, "_git_bytes", fake_git)
    with pytest.raises(ValueError, match="committed and clean"):
        probe.source_bundle_provenance()


def test_recorded_source_bundle_does_not_require_current_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    relative = "source.py"
    payload = b"print('bound')\n"
    commit = "b" * 40
    (tmp_path / relative).write_bytes(payload)
    digest = probe.sha256_bytes(payload)
    value = {
        "schema_version": 1,
        "git_commit": commit,
        "source_binding_paths": [relative],
        "source_binding_files": {relative: digest},
        "source_bundle_sha256": probe.sha256_bytes(f"{relative}:{digest}".encode()),
        "path_scoped_clean": True,
    }
    calls: list[list[str]] = []

    def fake_git(args: list[str]) -> bytes:
        calls.append(args)
        if args[:2] == ["status", "--porcelain=v1"]:
            return b""
        if args == ["show", f"{commit}:{relative}"]:
            return payload
        raise AssertionError(args)

    monkeypatch.setattr(probe, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(probe, "_git_bytes", fake_git)
    assert probe.validate_recorded_source_bundle(value, (relative,)) == value
    assert not any(args[:2] == ["rev-parse", "HEAD"] for args in calls)


def test_cpu_preflight_artifact_uses_full_summary_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import summarize_g009_r0_rev23_matrix_observation_adapter_runtime as summary

    path = tmp_path / "cpu_preflight.json"
    path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(probe, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(probe, "CPU_PREFLIGHT_PATH", path)
    invoked: dict[str, object] = {}

    def reject_tampered(
        value: object, repo_root: Path, expected_output: str, source_bundle: object
    ) -> None:
        invoked.update(
            value=value,
            repo_root=repo_root,
            expected_output=expected_output,
            source_bundle=source_bundle,
        )
        raise ValueError("CPU repeatability mismatch")

    monkeypatch.setattr(summary, "validate_cpu_preflight_value", reject_tampered)
    with pytest.raises(ValueError, match="repeatability"):
        probe.validate_cpu_preflight_artifact(path, {"git_commit": "a" * 40})
    assert invoked["value"] == {}


def test_rev20_monkeypatches_restore_after_diagnose_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(
        device="cpu",
        replicate_index=1,
        isaaclab_root=ROOT.parent / "IsaacLab",
        _cpu_preflight_binding=probe.cpu_preflight_not_required_binding(),
    )
    original_accumulator = probe.rev20.MatrixSafetyAccumulator
    original_output_contract = probe.rev20.expected_output_relative
    monkeypatch.setattr(probe, "load_preregistration", lambda: {})
    monkeypatch.setattr(
        probe,
        "validate_predecessor",
        lambda: {"adapter_contract_sha256": "a" * 64},
    )
    monkeypatch.setattr(probe, "validate_runtime_parent_synthesis", lambda: {})
    monkeypatch.setattr(probe, "source_bundle_provenance", lambda: {})
    monkeypatch.setattr(probe.rev20, "load_preregistration", lambda: {})
    monkeypatch.setattr(probe.rev20, "validate_external_sources", lambda *_args: {})

    def fail_diagnose(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("diagnose failed")

    monkeypatch.setattr(probe.rev20, "diagnose", fail_diagnose)
    with pytest.raises(RuntimeError, match="diagnose failed"):
        probe.diagnose(args, _execution())
    assert probe.rev20.MatrixSafetyAccumulator is original_accumulator
    assert probe.rev20.expected_output_relative is original_output_contract


def test_prelaunch_rejects_rev20_source_before_cpu_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(
        device="cuda:0",
        cpu_preflight=Path("unused.json"),
        isaaclab_root=ROOT.parent / "IsaacLab",
    )
    monkeypatch.setattr(probe, "load_preregistration", lambda: {})
    monkeypatch.setattr(probe, "validate_predecessor", lambda: {})
    monkeypatch.setattr(probe, "validate_runtime_parent_synthesis", lambda: {})
    monkeypatch.setattr(probe, "source_bundle_provenance", lambda: {})
    monkeypatch.setattr(probe.rev20, "load_preregistration", lambda: {})
    monkeypatch.setattr(probe.rev20, "source_bundle_provenance", lambda: {"clean": False})
    monkeypatch.setattr(
        probe.rev20,
        "validate_source_bundle",
        lambda _value: (_ for _ in ()).throw(ValueError("rev20 source dirty")),
    )
    preflight_called = False

    def preflight(*_args: object) -> dict[str, object]:
        nonlocal preflight_called
        preflight_called = True
        return {}

    monkeypatch.setattr(probe, "validate_cpu_preflight_artifact", preflight)
    with pytest.raises(ValueError, match="rev20 source dirty"):
        probe.prelaunch_validate(args)
    assert preflight_called is False


def test_existing_output_is_rejected_before_app_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe.runtime_probe, "parse_prelaunch_output", lambda _argv: argparse.Namespace())
    monkeypatch.setattr(
        probe.runtime_probe,
        "prepare_execution",
        lambda _args: (_ for _ in ()).throw(FileExistsError("canonical output exists")),
    )
    before = set(sys.modules)
    assert probe.main(["--output", "unused.json"]) == 2
    assert "isaaclab.app" not in set(sys.modules) - before


def test_prelaunch_subprocess_failure_uses_operational_exit_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        device="cpu",
        replicate_index=1,
        isaaclab_root=ROOT.parent / "IsaacLab",
        cpu_preflight=None,
    )
    monkeypatch.setattr(probe.runtime_probe, "parse_prelaunch_output", lambda _argv: argparse.Namespace())
    monkeypatch.setattr(
        probe.runtime_probe,
        "prepare_execution",
        lambda _args: (ROOT / probe.expected_output_relative("cpu", 1), _execution()),
    )
    monkeypatch.setattr(probe, "parse_prelaunch_args", lambda _argv: args)
    full_parse_called = False

    def full_parse(_argv: object) -> argparse.Namespace:
        nonlocal full_parse_called
        full_parse_called = True
        return args

    monkeypatch.setattr(probe, "parse_args", full_parse)
    monkeypatch.setattr(probe, "validate_execution", lambda *_args: None)
    monkeypatch.setattr(
        probe,
        "prelaunch_validate",
        lambda _args: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["git", "show"], stderr=b"missing object")
        ),
    )
    before = set(sys.modules)
    assert probe.main([]) == 3
    assert full_parse_called is False
    assert "isaaclab.app" not in set(sys.modules) - before


def test_prelaunch_parser_is_import_free_and_ignores_launcher_only_options() -> None:
    before = set(sys.modules)
    args = probe.parse_prelaunch_args(
        [
            "--replicate-index",
            "1",
            "--output",
            str(ROOT / probe.expected_output_relative("cpu", 1)),
            "--device",
            "cpu",
            "--headless",
            "--launcher-only-option",
            "unused",
        ]
    )
    assert args.device == "cpu"
    assert args.replicate_index == 1
    assert args.headless is True
    assert "isaaclab.app" not in set(sys.modules) - before


def test_app_close_failure_cannot_consume_pass_canonical_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "canonical.json"
    failure = tmp_path / "failure.json"
    args = argparse.Namespace(
        task=probe.DEFAULT_TASK,
        seed=42,
        replicate_index=1,
        cpu_preflight=None,
        isaaclab_root=ROOT.parent / "IsaacLab",
        output=output,
        device="cpu",
        headless=True,
    )
    execution = _execution()
    writes: list[tuple[Path, dict[str, object]]] = []
    validation_called = False

    class FakeApp:
        def close(self) -> None:
            assert validation_called is True
            raise RuntimeError("close failed")

    class FakeAppLauncher:
        def __init__(self, _args: argparse.Namespace, **kwargs: object) -> None:
            assert kwargs == {"fast_shutdown": False}
            self.app = FakeApp()

    isaaclab_module = ModuleType("isaaclab")
    app_module = ModuleType("isaaclab.app")
    app_module.AppLauncher = FakeAppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setattr(probe.runtime_probe, "parse_prelaunch_output", lambda _argv: output)
    monkeypatch.setattr(probe.runtime_probe, "prepare_execution", lambda _path: (output, execution))
    monkeypatch.setattr(probe, "parse_prelaunch_args", lambda _argv: args)
    monkeypatch.setattr(probe, "parse_args", lambda _argv: args)
    monkeypatch.setattr(probe, "validate_execution", lambda *_args: None)
    monkeypatch.setattr(probe, "prelaunch_validate", lambda _args: probe.cpu_preflight_not_required_binding())
    monkeypatch.setattr(
        probe,
        "diagnose",
        lambda *_args: {
            "adapter_decision": {"passed": True},
            "adapter_runtime": {"sample_count": 150},
        },
    )
    def validate(_report: object) -> dict[str, object]:
        nonlocal validation_called
        validation_called = True
        return {}

    monkeypatch.setattr(probe, "validate_report", validate)
    monkeypatch.setattr(probe, "failed_attempt_path", lambda *_args: failure)
    monkeypatch.setattr(
        probe.runtime_probe,
        "_write_json_atomic",
        lambda path, value: writes.append((path, value)),
    )

    assert probe.main([]) == 3
    assert [path for path, _value in writes] == [failure]
    assert writes[0][1]["diagnostic_report"]["adapter_decision"]["passed"] is True
    assert output not in [path for path, _value in writes]


def test_failed_adapter_decision_stays_local_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "canonical.json"
    failure = tmp_path / "failure.json"
    args = argparse.Namespace(
        task=probe.DEFAULT_TASK,
        seed=42,
        replicate_index=1,
        cpu_preflight=None,
        isaaclab_root=ROOT.parent / "IsaacLab",
        output=output,
        device="cpu",
        headless=True,
    )
    writes: list[tuple[Path, dict[str, object]]] = []

    class FakeApp:
        def close(self) -> None:
            return None

    class FakeAppLauncher:
        def __init__(self, _args: argparse.Namespace, **kwargs: object) -> None:
            assert kwargs == {"fast_shutdown": False}
            self.app = FakeApp()

    app_module = ModuleType("isaaclab.app")
    app_module.AppLauncher = FakeAppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    monkeypatch.setattr(probe.runtime_probe, "parse_prelaunch_output", lambda _argv: output)
    monkeypatch.setattr(probe.runtime_probe, "prepare_execution", lambda _path: (output, _execution()))
    monkeypatch.setattr(probe, "parse_prelaunch_args", lambda _argv: args)
    monkeypatch.setattr(probe, "parse_args", lambda _argv: args)
    monkeypatch.setattr(probe, "validate_execution", lambda *_args: None)
    monkeypatch.setattr(probe, "prelaunch_validate", lambda _args: probe.cpu_preflight_not_required_binding())
    monkeypatch.setattr(
        probe,
        "diagnose",
        lambda *_args: {
            "adapter_decision": {"passed": False},
            "adapter_runtime": {"sample_count": 150},
        },
    )
    monkeypatch.setattr(
        probe,
        "validate_report",
        lambda _report: (_ for _ in ()).throw(AssertionError("failed report must not enter PASS validator")),
    )
    monkeypatch.setattr(probe, "failed_attempt_path", lambda *_args: failure)
    monkeypatch.setattr(
        probe.runtime_probe,
        "_write_json_atomic",
        lambda path, value: writes.append((path, value)),
    )

    assert probe.main([]) == 2
    assert [path for path, _value in writes] == [failure]
    assert writes[0][1]["diagnostic_report"]["adapter_decision"]["passed"] is False


def test_publish_recovers_verified_post_link_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "report.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    report = {"status": "complete", "value": 1}

    def publish_then_cleanup_fail(path: Path, value: dict[str, object]) -> None:
        payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        temporary.write_bytes(payload)
        os.link(temporary, path)
        raise PermissionError("temporary cleanup denied")

    monkeypatch.setattr(probe.runtime_probe, "_write_json_atomic", publish_then_cleanup_fail)
    warning = probe._publish_canonical_report(output, report)
    assert warning is not None and "cleanup warning" in warning
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert not temporary.exists()


def test_failure_envelope_keeps_no_learning_claims() -> None:
    args = argparse.Namespace(device="cpu", replicate_index=1)
    value = probe.failure_envelope(args, _execution(), RuntimeError("boom"))
    assert value["status"] == "failed_closed"
    assert value["governance"]["reward_computed"] is False
    assert value["governance"]["ppo_updates"] == 0
    assert value["error"] == {"type": "RuntimeError", "message": "boom"}
