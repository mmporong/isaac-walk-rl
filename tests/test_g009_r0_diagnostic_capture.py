from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


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
    return report_path, checkpoint, report


def test_diagnostic_identity_and_local_only_filename() -> None:
    assert diagnostic.STAGE_NUMBER == "G009-5"
    assert diagnostic.STAGE_ID == "R0"
    assert diagnostic.output_name() == "g009_5_r0_diag_rev9_01_prone_s42.mp4"
    assert diagnostic.DEFAULT_OUTPUT_DIR.parts[-2:] == ("R0", "diagnostic")
    assert diagnostic.DEFAULT_REPORT_PATH.name == "g009_r0_diag_rev9_01_prone_capture_s42.json"
    assert "--/app/vulkan=false" in diagnostic.WINDOWS_KIT_ARGS
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
