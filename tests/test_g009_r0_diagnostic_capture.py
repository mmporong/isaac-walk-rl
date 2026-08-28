from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]


ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "record_g009_r0_diagnostic.py"
    spec = importlib.util.spec_from_file_location("record_g009_r0_diagnostic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnostic = _load()


def _valid_report(source_sha: str, checkpoint_sha: str) -> dict:
    files = {"source.py": source_sha}
    return {
        "run_name": diagnostic.EXPECTED_RUN_NAME,
        "task": diagnostic.DEFAULT_TASK,
        "num_envs": 1024,
        "max_iterations": 50,
        "last_iteration": 49,
        "iteration_target": 50,
        "seed": 42,
        "headless": True,
        "resume": {"enabled": False},
        "effective_hydra_overrides": [],
        "qualification_mode": {
            "enabled": False,
            "preflight_passed": None,
            "policy_qualification_status": "not_run",
        },
        "repository": {"commit": diagnostic.EXPECTED_TRAINING_COMMIT, "dirty": False},
        "source_bundle": {
            "sha256": diagnostic.source_bundle_sha256(files),
            "files": files,
            "matches_repository_commit": True,
        },
        "training_safety_aggregate": {
            "hard_joint_limit": {"maximum": 0.1},
            "numeric_invalid": {"maximum": 0.0},
        },
        "tensorboard": {
            "series_summary": {"Episode_Reward/stable_success_once": {"maximum": 0.0}}
        },
        "artifacts": {
            "checkpoint": "%USERPROFILE%\\IsaacLab\\logs\\model_49.pt",
            "checkpoint_sha256": checkpoint_sha,
        },
        "run_health_passed": True,
        "passed": True,
    }


def _install_valid_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source.py"
    checkpoint = tmp_path / "model_49.pt"
    report_path = tmp_path / "training.json"
    source.write_text("source\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    source_sha = diagnostic.file_sha256(source)
    checkpoint_sha = diagnostic.file_sha256(checkpoint)
    report = _valid_report(source_sha, checkpoint_sha)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_sha = diagnostic.file_sha256(report_path)
    bundle_sha = report["source_bundle"]["sha256"]
    monkeypatch.setattr(diagnostic, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "EXPECTED_TRAINING_REPORT_SHA256", report_sha)
    monkeypatch.setattr(diagnostic, "EXPECTED_CHECKPOINT_SHA256", checkpoint_sha)
    monkeypatch.setattr(diagnostic, "EXPECTED_SOURCE_BUNDLE_SHA256", bundle_sha)
    monkeypatch.setattr(
        diagnostic,
        "git_blob_sha256_candidates",
        lambda _commit, relative: frozenset({diagnostic.file_sha256(tmp_path / relative)}),
    )
    return report_path, checkpoint, report


def _install_dynamic_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_label: str,
    iterations: int,
    revision: str = "rev10",
) -> tuple[Path, Path, dict, str]:
    commit = "a" * 40
    run_name = f"go2_flat_recover_{revision}_prone_{gate_label}_s42_fixture"
    files: dict[str, str] = {}
    for relative in diagnostic.REQUIRED_SOURCE_BUNDLE_PATHS:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"{relative}\n", encoding="utf-8")
        files[relative] = diagnostic.file_sha256(source)
    checkpoint = tmp_path / "logs" / f"model_{iterations - 1}.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"checkpoint-{gate_label}".encode())
    report = _valid_report(next(iter(files.values())), diagnostic.file_sha256(checkpoint))
    report.update(
        run_name=run_name,
        max_iterations=iterations,
        last_iteration=iterations - 1,
        iteration_target=iterations,
    )
    report["repository"]["commit"] = commit
    report["source_bundle"] = {
        "sha256": diagnostic.source_bundle_sha256(files),
        "files": files,
        "matches_repository_commit": True,
    }
    report["artifacts"]["checkpoint"] = str(checkpoint)
    report_path = tmp_path / "reports" / "runs" / f"{run_name}.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(diagnostic, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "current_git_commit", lambda: commit)
    return report_path, checkpoint, report, run_name


def test_diagnostic_identity_and_local_only_filename() -> None:
    assert diagnostic.STAGE_NUMBER == "G009-5"
    assert diagnostic.STAGE_ID == "R0"
    assert diagnostic.output_name() == "g009_5_r0_diag_rev9_01_prone_s42.mp4"
    assert diagnostic.DEFAULT_OUTPUT_DIR.parts[-2:] == ("R0", "diagnostic")
    assert diagnostic.DEFAULT_REPORT_PATH.name == "g009_r0_diag_rev9_01_prone_capture_s42.json"
    assert "--/app/vulkan=false" in diagnostic.WINDOWS_KIT_ARGS
    assert diagnostic.CAPTURE_NUM_ENVS == 1
    assert diagnostic.CAMERA_EYE == (1.4, 1.4, 0.85)
    assert diagnostic.validate_output_dir(diagnostic.DEFAULT_OUTPUT_DIR) == diagnostic.DEFAULT_OUTPUT_DIR.resolve()
    with pytest.raises(ValueError, match="local-only"):
        diagnostic.validate_output_dir(ROOT / "docs" / "media" / "g009" / "R0")
    assert diagnostic.validate_report_path(diagnostic.DEFAULT_REPORT_PATH) == diagnostic.DEFAULT_REPORT_PATH.resolve()
    with pytest.raises(ValueError, match="report path is fixed"):
        diagnostic.validate_report_path(ROOT / "reports" / "runs" / "other.json")


def test_terminal_auto_reset_frame_is_removed() -> None:
    assert diagnostic.diagnostic_recorded_frame_count(400, terminated=True) == 400
    assert diagnostic.diagnostic_recorded_frame_count(400, terminated=False) == 401
    assert diagnostic.diagnostic_recorded_frame_count(1, terminated=True) == 1


def test_capture_report_or_mp4_cannot_be_overwritten(tmp_path: Path) -> None:
    report, video = tmp_path / "capture.json", tmp_path / "capture.mp4"
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        diagnostic.validate_new_capture_paths(report, video)
    report.unlink()
    video.write_bytes(b"video")
    with pytest.raises(FileExistsError):
        diagnostic.validate_new_capture_paths(report, video)


def test_existing_capture_report_is_rejected_before_app_launcher(tmp_path: Path) -> None:
    report = tmp_path / "capture.json"
    report.write_text("{}", encoding="utf-8")
    assert not (tmp_path / "capture.mp4").exists()
    with pytest.raises(FileExistsError):
        diagnostic.validate_new_report_path_before_launch(report)


def test_rev9_pilot_binding_accepts_only_exact_diagnostic_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path, checkpoint, _ = _install_valid_fixture(tmp_path, monkeypatch)
    binding = diagnostic.validate_diagnostic_training_report(report_path, checkpoint)
    assert binding["run_name"] == diagnostic.EXPECTED_RUN_NAME
    assert binding["protocol"] == {
        "num_envs": 1024,
        "max_iterations": 50,
        "seed": 42,
        "headless": True,
        "scratch": True,
        "qualification_enabled": False,
    }
    assert binding["checkpoint_sha256"] == diagnostic.file_sha256(checkpoint)


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (lambda report: report.update(max_iterations=300), "max_iterations"),
        (lambda report: report["qualification_mode"].update(enabled=True), "diagnostic_training"),
        (lambda report: report["resume"].update(enabled=True), "scratch"),
        (lambda report: report["source_bundle"].update(matches_repository_commit=False), "source_bundle_commit_match"),
        (lambda report: report["training_safety_aggregate"]["hard_joint_limit"].update(maximum=0.0), "rejected_hard_limit"),
        (lambda report: report["tensorboard"]["series_summary"]["Episode_Reward/stable_success_once"].update(maximum=1.0), "strict_success_absent"),
    ],
)
def test_rev9_binding_fails_closed_on_semantic_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    failure: str,
) -> None:
    report_path, checkpoint, report = _install_valid_fixture(tmp_path, monkeypatch)
    mutation(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(diagnostic, "EXPECTED_TRAINING_REPORT_SHA256", diagnostic.file_sha256(report_path))
    with pytest.raises(ValueError, match=failure):
        diagnostic.validate_diagnostic_training_report(report_path, checkpoint)


def test_rev9_binding_rejects_report_and_checkpoint_hash_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path, checkpoint, _ = _install_valid_fixture(tmp_path, monkeypatch)
    report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="report_sha256"):
        diagnostic.validate_diagnostic_training_report(report_path, checkpoint)

    report_path, checkpoint, _ = _install_valid_fixture(tmp_path, monkeypatch)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint_identity"):
        diagnostic.validate_diagnostic_training_report(report_path, checkpoint)


def test_rev10_gate_binding_is_derived_from_report_and_explicit_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path, checkpoint, report, run_name = _install_dynamic_fixture(
        tmp_path, monkeypatch, "gate50", 50
    )
    report["training_safety_aggregate"]["hard_joint_limit"]["maximum"] = 0.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    stem = "g009_5_r0_diag_rev10_gate50_01_prone"
    binding = diagnostic.validate_diagnostic_training_report(
        report_path,
        checkpoint,
        revision="rev10",
        gate_label="gate50",
        output_stem=stem,
        expected_run_name=run_name,
    )
    assert binding["run_name"] == report["run_name"]
    assert binding["repository"]["commit"] == "a" * 40
    assert binding["protocol"]["max_iterations"] == 50
    assert binding["sha256"] == diagnostic.file_sha256(report_path)
    assert diagnostic.output_name(stem) == f"{stem}_s42.mp4"
    assert diagnostic.expected_report_path(stem).name == f"{stem}_capture_s42.json"


def test_rev10_gate_binding_fails_closed_on_gate_or_checkpoint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path, checkpoint, _, run_name = _install_dynamic_fixture(
        tmp_path, monkeypatch, "gate10", 10
    )
    with pytest.raises(ValueError, match="revision/gate mismatch"):
        diagnostic.validate_diagnostic_training_report(
            report_path,
            checkpoint,
            revision="rev10",
            gate_label="gate01",
            output_stem="g009_5_r0_diag_rev10_gate01_01_prone",
            expected_run_name=run_name,
        )
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint_identity"):
        diagnostic.validate_diagnostic_training_report(
            report_path,
            checkpoint,
            revision="rev10",
            gate_label="gate10",
            output_stem="g009_5_r0_diag_rev10_gate10_01_prone",
            expected_run_name=run_name,
        )


@pytest.mark.parametrize(("gate_label", "iterations"), [("gate01", 1), ("gate10", 10), ("gate50", 50)])
def test_each_rev10_gate_accepts_its_exact_iteration_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_label: str,
    iterations: int,
) -> None:
    report_path, checkpoint, _, run_name = _install_dynamic_fixture(
        tmp_path, monkeypatch, gate_label, iterations
    )
    stem = f"g009_5_r0_diag_rev10_{gate_label}_01_prone"
    binding = diagnostic.validate_diagnostic_training_report(
        report_path,
        checkpoint,
        revision="rev10",
        gate_label=gate_label,
        output_stem=stem,
        expected_run_name=run_name,
    )
    assert binding["protocol"]["max_iterations"] == iterations


@pytest.mark.parametrize(("gate_label", "iterations"), [("gate01", 1), ("gate10", 10), ("gate50", 50)])
def test_each_rev11_gate_accepts_its_exact_identity_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_label: str,
    iterations: int,
) -> None:
    report_path, checkpoint, _, run_name = _install_dynamic_fixture(
        tmp_path, monkeypatch, gate_label, iterations, revision="rev11"
    )
    stem = f"g009_5_r0_diag_rev11_{gate_label}_01_prone"
    binding = diagnostic.validate_diagnostic_training_report(
        report_path,
        checkpoint,
        revision="rev11",
        gate_label=gate_label,
        output_stem=stem,
        expected_run_name=run_name,
    )
    assert binding["run_name"] == run_name
    assert binding["protocol"]["max_iterations"] == iterations
    assert binding["checkpoint_sha256"] == diagnostic.file_sha256(checkpoint)


@pytest.mark.parametrize(
    "run_name",
    [
        "prefix_go2_flat_recover_rev11_prone_gate01_s42_fixture",
        "go2_flat_recover_rev11_prone_gate01_s42_fixture_suffix/escape",
        "go2_flat_recover_rev11_prone_gate01_s42_",
    ],
)
def test_rev11_gate_rejects_noncanonical_run_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_name: str,
) -> None:
    report_path, checkpoint, report, _ = _install_dynamic_fixture(
        tmp_path, monkeypatch, "gate01", 1, revision="rev11"
    )
    report["run_name"] = run_name
    bad_path = report_path.parent / f"{run_name.replace('/', '-')}.json"
    bad_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="identity is not canonical"):
        diagnostic.validate_diagnostic_training_report(
            bad_path,
            checkpoint,
            revision="rev11",
            gate_label="gate01",
            output_stem="g009_5_r0_diag_rev11_gate01_01_prone",
            expected_run_name=run_name,
        )


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (lambda report: report.update(run_name="go2_flat_recover_rev10_prone_gate50_s42_tampered"), "expected_run_name"),
        (lambda report: report["repository"].update(commit="b" * 40), "training_commit"),
        (lambda report: report["source_bundle"]["files"].pop(next(iter(report["source_bundle"]["files"]))), "path set mismatch"),
        (lambda report: report["artifacts"].update(checkpoint="C:/tampered/model_49.pt"), "checkpoint_identity"),
    ],
)
def test_rev10_identity_rejects_run_commit_bundle_and_checkpoint_path_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    failure: str,
) -> None:
    report_path, checkpoint, report, run_name = _install_dynamic_fixture(
        tmp_path, monkeypatch, "gate50", 50
    )
    mutation(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=failure):
        diagnostic.validate_diagnostic_training_report(
            report_path,
            checkpoint,
            revision="rev10",
            gate_label="gate50",
            output_stem="g009_5_r0_diag_rev10_gate50_01_prone",
            expected_run_name=run_name,
        )


def test_rev10_identity_rejects_noncanonical_training_report_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path, checkpoint, _, run_name = _install_dynamic_fixture(
        tmp_path, monkeypatch, "gate01", 1
    )
    wrong_path = tmp_path / "reports" / "runs" / "renamed.json"
    wrong_path.write_bytes(report_path.read_bytes())
    with pytest.raises(ValueError, match="report path"):
        diagnostic.validate_diagnostic_training_report(
            wrong_path,
            checkpoint,
            revision="rev10",
            gate_label="gate01",
            output_stem="g009_5_r0_diag_rev10_gate01_01_prone",
            expected_run_name=run_name,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_iterations", True), ("last_iteration", False), ("iteration_target", True)],
)
def test_rev10_identity_rejects_boolean_iteration_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: bool,
) -> None:
    report_path, checkpoint, report, run_name = _install_dynamic_fixture(
        tmp_path, monkeypatch, "gate01", 1
    )
    report[field] = value
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="iteration fields must be integers"):
        diagnostic.validate_diagnostic_training_report(
            report_path,
            checkpoint,
            revision="rev10",
            gate_label="gate01",
            output_stem="g009_5_r0_diag_rev10_gate01_01_prone",
            expected_run_name=run_name,
        )


def test_rev9_historical_bundle_accepts_only_raw_or_crlf_git_blob_candidates() -> None:
    report_path = ROOT / "reports" / "runs" / "go2_flat_recover_rev9_prone_pilot_s42_20260828-1421.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    commit = report["repository"]["commit"]
    for relative, expected_sha in report["source_bundle"]["files"].items():
        assert expected_sha in diagnostic.git_blob_sha256_candidates(commit, relative)


def test_committed_rev9_training_artifacts_validate_when_available() -> None:
    report_path = ROOT / "reports" / "runs" / "go2_flat_recover_rev9_prone_pilot_s42_20260828-1421.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = diagnostic.resolve_portable_path(report["artifacts"]["checkpoint"])
    if not checkpoint.is_file():
        pytest.skip("local-only rev9 checkpoint is unavailable on this host")
    binding = diagnostic.validate_diagnostic_training_report(report_path, checkpoint)
    assert binding["source_bundle"]["sha256"] == report["source_bundle"]["sha256"]
