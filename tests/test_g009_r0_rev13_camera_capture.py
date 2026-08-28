from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECORDER = load("g009_rev13_camera_recorder_test", "scripts/record_g009_r0_rev13_right_side.py")
MEDIA = load("g009_rev13_camera_media_test", "scripts/build_g009_r0_rev13_camera_media.py")


def test_capture_identity_is_numbered_and_local_only() -> None:
    assert RECORDER.OUTPUT_STEM == "g009_5_r0_diag_rev13_04_right_side_runtime"
    assert RECORDER.DEFAULT_VIDEO.name == f"{RECORDER.OUTPUT_STEM}_s42.mp4"
    assert RECORDER.portable_path(RECORDER.DEFAULT_VIDEO).startswith(
        "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\R0\\diagnostic\\"
    )


def test_capture_contract_matches_rev13_failure_cell() -> None:
    binding = RECORDER.validate_runtime_report(RECORDER.DEFAULT_RUNTIME_REPORT)
    assert binding["cell"]["env_index"] == 7
    assert binding["cell"]["pose_id"] == "right_side"
    assert binding["cell"]["action_mode"] == "reset_pose_hold"
    assert binding["cell"]["max_nonfoot_force_bodyweights"] == 15.97161865234375
    assert binding["cell"]["max_nonfoot_force_physics_step"] == 129
    assert binding["cell"]["max_nonfoot_force_body_name"] == "base"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("contract_sha256", "0" * 64, "contract mismatch"),
        ("seed", 41, "seed/env count mismatch"),
        ("rollout_steps", 149, "rollout length mismatch"),
    ],
)
def test_runtime_binding_fails_closed(monkeypatch, field: str, value: object, message: str) -> None:
    report = json.loads(RECORDER.DEFAULT_RUNTIME_REPORT.read_text(encoding="utf-8"))
    report[field] = value
    monkeypatch.setattr(RECORDER, "file_sha256", lambda _: RECORDER.EXPECTED_REPORT_SHA256)
    monkeypatch.setattr(RECORDER, "read_json", lambda _: report)
    with pytest.raises(ValueError, match=message):
        RECORDER.validate_runtime_report(RECORDER.DEFAULT_RUNTIME_REPORT)


def test_failure_cell_requires_exact_env_pose_action() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        RECORDER.failure_cell({"pose_mode_metrics": [{"env_index": 7, "pose_id": "right_side", "action_mode": "zero_normalized"}]})


def test_capture_commit_must_descend_from_rev13() -> None:
    assert RECORDER.contains_rev13_source_commit(RECORDER.git_source_state()["commit"]) is True


def valid_current_bundle() -> dict:
    return {
        "source_bundle_sha256": RECORDER.EXPECTED_SOURCE_BUNDLE_SHA256,
        "all_files_present": True,
        "git_commit_valid": True,
        "clean": True,
    }


def test_current_capture_binding_accepts_exact_rev13_bundle(monkeypatch) -> None:
    monkeypatch.setattr(RECORDER, "source_bundle_provenance", valid_current_bundle)
    monkeypatch.setattr(RECORDER, "recover_contract", lambda: {"contract": "rev13"})
    monkeypatch.setattr(RECORDER, "canonical_sha256", lambda _: RECORDER.EXPECTED_CONTRACT_SHA256)
    binding = RECORDER.validate_current_capture_binding()
    assert binding["source_bundle"]["source_bundle_sha256"] == RECORDER.EXPECTED_SOURCE_BUNDLE_SHA256
    assert binding["contract_sha256"] == RECORDER.EXPECTED_CONTRACT_SHA256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_bundle_sha256", "0" * 64, "bundle drifted"),
        ("all_files_present", False, "missing files"),
        ("git_commit_valid", False, "commit is invalid"),
        ("clean", False, "binding is dirty"),
    ],
)
def test_current_capture_binding_rejects_bundle_drift(monkeypatch, field: str, value: object, message: str) -> None:
    bundle = valid_current_bundle()
    bundle[field] = value
    monkeypatch.setattr(RECORDER, "source_bundle_provenance", lambda: bundle)
    with pytest.raises(ValueError, match=message):
        RECORDER.validate_current_capture_binding()


def test_current_capture_binding_rejects_contract_drift(monkeypatch) -> None:
    monkeypatch.setattr(RECORDER, "source_bundle_provenance", valid_current_bundle)
    monkeypatch.setattr(RECORDER, "recover_contract", lambda: {"contract": "drifted"})
    monkeypatch.setattr(RECORDER, "canonical_sha256", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="contract drifted"):
        RECORDER.validate_current_capture_binding()


def test_public_derivatives_are_numbered_and_git_scoped() -> None:
    assert MEDIA.DEFAULT_GIF.name == f"{MEDIA.OUTPUT_STEM}.gif"
    assert MEDIA.DEFAULT_PNG.name == f"{MEDIA.OUTPUT_STEM}.png"
    assert MEDIA.DEFAULT_VISUAL.name == f"{MEDIA.OUTPUT_STEM}_visual_evidence.json"
    assert MEDIA.repo_path(MEDIA.DEFAULT_GIF).startswith("docs/media/g009/R0/diagnostic/")


def test_required_labels_are_explicit() -> None:
    assert MEDIA.REQUIRED_LABELS == (
        "DIAGNOSTIC", "NOT QUALIFIED", "NO PPO", "RIGHT_SIDE",
        "RESET_POSE_HOLD", "REV13 REJECTED",
    )
    assert MEDIA.OVERLAY_TOP == "G009-5 | REV13 | DIAGNOSTIC | NOT QUALIFIED | NO PPO"
    assert MEDIA.OVERLAY_BOTTOM == "04 RIGHT_SIDE | RESET_POSE_HOLD | REJECTED"


def test_png_probe_does_not_require_duration_or_frame_rate(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "streams": [{"codec_type": "video", "codec_name": "png", "width": 1280, "height": 720}],
        "format": {},
    }
    completed = types.SimpleNamespace(stdout=json.dumps(payload))
    monkeypatch.setattr(MEDIA.subprocess, "run", lambda *args, **kwargs: completed)
    summary = MEDIA.ffprobe_summary(tmp_path / "still.png", "ffprobe", require_timing=False)
    assert summary == {"codec": "png", "width": 1280, "height": 720}


def test_gif_probe_requires_real_duration_and_frames(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "streams": [{"codec_type": "video", "codec_name": "gif", "width": 960, "height": 540, "avg_frame_rate": "10/1", "nb_read_frames": "30"}],
        "format": {"duration": "3.0"},
    }
    completed = types.SimpleNamespace(stdout=json.dumps(payload))
    monkeypatch.setattr(MEDIA.subprocess, "run", lambda *args, **kwargs: completed)
    summary = MEDIA.ffprobe_summary(tmp_path / "camera.gif", "ffprobe")
    assert summary["frames"] == 30
    assert summary["duration_s"] == 3.0


def valid_capture(tmp_path: Path) -> tuple[dict, Path]:
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"camera-footage")
    capture = {
        "camera_footage": True, "telemetry_animation": False,
        "headless": True, "offscreen": True,
        "diagnostic_only": True, "qualification_status": "not_run",
        "learned": False, "ppo_checkpoint_used": False,
        "labels": list(MEDIA.REQUIRED_LABELS),
        "pose_id": "right_side", "action_mode": "reset_pose_hold",
        "local_video": {"path": str(MEDIA.EXPECTED_LOCAL), "sha256": MEDIA.file_sha256(video), "bytes": video.stat().st_size},
    }
    return capture, video


def test_media_validation_accepts_only_actual_camera_footage(monkeypatch, tmp_path: Path) -> None:
    capture, video = valid_capture(tmp_path)
    monkeypatch.setattr(MEDIA, "read_json", lambda _: capture)
    monkeypatch.setattr(MEDIA, "resolve_portable", lambda _: video)
    actual, source = MEDIA.validate_capture(MEDIA.DEFAULT_CAPTURE)
    assert actual["camera_footage"] is True
    assert source == video.resolve()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("camera_footage", False, "camera footage"),
        ("telemetry_animation", True, "camera footage"),
        ("qualification_status", "passed", "diagnostic status"),
        ("learned", True, "NO PPO"),
        ("pose_id", "prone", "failure cell"),
    ],
)
def test_media_validation_rejects_claim_drift(monkeypatch, tmp_path: Path, field: str, value: object, message: str) -> None:
    capture, video = valid_capture(tmp_path)
    capture[field] = value
    monkeypatch.setattr(MEDIA, "read_json", lambda _: capture)
    monkeypatch.setattr(MEDIA, "resolve_portable", lambda _: video)
    with pytest.raises(ValueError, match=message):
        MEDIA.validate_capture(MEDIA.DEFAULT_CAPTURE)


def test_evidence_scope_does_not_claim_peak_reproduction() -> None:
    source = (ROOT / "scripts/record_g009_r0_rev13_right_side.py").read_text(encoding="utf-8")
    assert "condition-matched visual playback" in source
    assert "does not claim direct reproduction of the report peak" in source


def test_visual_sidecar_binds_numbered_overlay_and_builder_source() -> None:
    source = (ROOT / "scripts/build_g009_r0_rev13_camera_media.py").read_text(encoding="utf-8")
    assert '"overlay_labels": {"top": OVERLAY_TOP, "bottom": OVERLAY_BOTTOM}' in source
    assert '"source_builder"' in source
    assert 'file_sha256(Path(__file__))' in source
