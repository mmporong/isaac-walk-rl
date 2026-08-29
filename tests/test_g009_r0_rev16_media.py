from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECORDER = load_module(
    "rev16_recorder", "scripts/record_g009_r0_rev16_b_gpu_right_side.py"
)
CAMERA = load_module("rev16_camera", "scripts/build_g009_r0_rev16_camera_media.py")
TELEMETRY = load_module(
    "rev16_telemetry", "scripts/build_g009_r0_rev16_telemetry_media.py"
)


def test_stage08_is_exact_arm_b_gpu_camera_condition() -> None:
    binding = RECORDER.validate_runtime_report(RECORDER.DEFAULT_RUNTIME_REPORT)
    assert (
        RECORDER.OUTPUT_STEM == "g009_5_r0_diag_rev16_08_b_gpu_right_side_force_repro"
    )
    assert (RECORDER.SOURCE_ENV_INDEX, RECORDER.POSE_ID, RECORDER.ACTION_MODE) == (
        7,
        "right_side",
        "reset_pose_hold",
    )
    assert binding["report"]["contract"]["arm"] == {
        "id": "B",
        "meaning": "position_solver_only_16",
        "articulation_solver_position_iteration_count": 16,
        "articulation_solver_velocity_iteration_count": 0,
        "max_depenetration_velocity_m_s": 1.0,
    }
    assert binding["cell"]["max_nonfoot_force_bodyweights"] == pytest.approx(
        16.788277099400098
    )
    assert RECORDER.portable_path(RECORDER.DEFAULT_VIDEO).startswith(
        "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence"
    )


def test_stage08_runtime_hash_and_governance_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = json.loads(RECORDER.DEFAULT_RUNTIME_REPORT.read_text(encoding="utf-8"))
    value["governance"]["ppo"]["status"] = "complete"
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(RECORDER, "DEFAULT_RUNTIME_REPORT", path)
    monkeypatch.setattr(
        RECORDER,
        "EXPECTED_RUNTIME_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="governance"):
        RECORDER.validate_runtime_report(path)


def test_stage08_capture_binding_rejects_dirty_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RECORDER,
        "git_source_state",
        lambda: {
            "commit": RECORDER.SOURCE_COMMIT,
            "clean": False,
            "dirty_paths": [" M src/isaac_walk_g009/recover_env_cfg.py"],
        },
    )
    with pytest.raises(ValueError, match="dirty"):
        RECORDER.validate_current_capture_binding()


def test_stage08_capture_binding_rejects_runtime_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RECORDER,
        "git_source_state",
        lambda: {
            "commit": RECORDER.SOURCE_COMMIT,
            "clean": True,
            "dirty_paths": [],
        },
    )
    monkeypatch.setattr(
        RECORDER, "contains_expected_source_commit", lambda _commit: True
    )
    original_raw_sha256 = RECORDER.file_sha256
    original_canonical_sha256 = RECORDER.canonical_text_sha256

    def drift_raw_file(path: Path) -> str:
        if path == RECORDER.REPO_ROOT / "src/isaac_walk_g009/recover_env_cfg.py":
            return "0" * 64
        return original_raw_sha256(path)

    def drift_canonical_file(path: Path) -> str:
        if path == RECORDER.REPO_ROOT / "src/isaac_walk_g009/recover_env_cfg.py":
            return "1" * 64
        return original_canonical_sha256(path)

    monkeypatch.setattr(RECORDER, "file_sha256", drift_raw_file)
    monkeypatch.setattr(RECORDER, "canonical_text_sha256", drift_canonical_file)
    with pytest.raises(ValueError, match="drifted"):
        RECORDER.validate_current_capture_binding()


def test_stage08_output_paths_are_local_only_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "diagnostic"
    video = output / "camera.mp4"
    report = tmp_path / "capture.json"
    monkeypatch.setattr(RECORDER, "DEFAULT_OUTPUT_DIR", output)
    monkeypatch.setattr(RECORDER, "DEFAULT_VIDEO", video)
    monkeypatch.setattr(RECORDER, "DEFAULT_CAPTURE_REPORT", report)
    output.mkdir()
    video.write_bytes(b"user-owned")
    with pytest.raises(ValueError, match="overwrite"):
        RECORDER.validate_output_paths(output, video, report)
    assert video.read_bytes() == b"user-owned"


def test_stage09_is_telemetry_not_camera_and_is_inconclusive() -> None:
    value = TELEMETRY.read_synthesis(TELEMETRY.DEFAULT_INPUT)
    assert TELEMETRY.OUTPUT_STEM == "g009_5_r0_diag_rev16_09_four_group_telemetry"
    assert "TELEMETRY ANIMATION · NOT CAMERA FOOTAGE" in TELEMETRY.LABELS
    assert value["hypothesis"]["decision"] == "inconclusive"
    ratio = value["hypothesis"]["replicates"][0]["derived"][
        "b_gpu_over_b_cpu_concentration_ratio"
    ]
    assert ratio == pytest.approx(1.183556126964255)
    assert ratio < 1.20
    assert value["governance"]["position16_accepted"] is False


def test_stage09_four_group_force_peak_concentration_and_exposure() -> None:
    value = TELEMETRY.read_synthesis(TELEMETRY.DEFAULT_INPUT)
    rows = TELEMETRY.first_group_rows(value)
    assert [row["peak_base_force_physics_step"] for row in rows] == [131, 130, 130, 129]
    assert [row["peak_base_force_bodyweights"] for row in rows] == pytest.approx(
        [9.332860204105899, 8.79500775388691, 13.248280587723672, 16.788277099400098]
    )
    assert [row["concentration_index"] for row in rows] == pytest.approx(
        [
            0.48876082535996307,
            0.4543198511243181,
            0.6737326951635448,
            0.7974004592969541,
        ]
    )
    assert [
        row["contact_exposure"]["thresholds"]["over_15_bodyweights"]["step_count"]
        for row in rows
    ] == [0, 0, 0, 1]


def test_public_outputs_have_numbered_hash_size_and_governance() -> None:
    for summary_path, stage, camera in (
        (CAMERA.DEFAULT_SUMMARY, "08", True),
        (TELEMETRY.DEFAULT_SUMMARY, "09", False),
    ):
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        assert value["stage_number"] == stage
        assert value["source_commit"] == RECORDER.SOURCE_COMMIT
        assert value["diagnostic_only"] is True and value["status"] == "rejected"
        assert value["camera_footage"] is camera
        assert value["telemetry_animation"] is (not camera)
        assert value["governance"]["ppo"]["status"] == "not_run"
        assert value["governance"]["qualification"]["status"] == "not_run"
    for path in (
        CAMERA.DEFAULT_PNG,
        CAMERA.DEFAULT_GIF,
        TELEMETRY.DEFAULT_PNG,
        TELEMETRY.DEFAULT_GIF,
    ):
        assert path.is_file() and 0 < path.stat().st_size < 10 * 1024 * 1024
    assert CAMERA.DEFAULT_GIF.stat().st_size < 6 * 1024 * 1024
    assert TELEMETRY.DEFAULT_GIF.stat().st_size < 6 * 1024 * 1024
    camera_summary = json.loads(CAMERA.DEFAULT_SUMMARY.read_text(encoding="utf-8"))
    telemetry_summary = json.loads(
        TELEMETRY.DEFAULT_SUMMARY.read_text(encoding="utf-8")
    )
    capture = json.loads(CAMERA.DEFAULT_CAPTURE.read_text(encoding="utf-8"))
    capture_source = capture["capture_source"]
    capture_bundle = capture_source["source_bundle"]
    assert capture_bundle["git_commit"] == capture_source["capture_commit"]
    assert capture_bundle["runtime_source_commit"] == RECORDER.SOURCE_COMMIT
    assert capture_bundle["clean"] is True
    assert capture_bundle["dirty_paths"] == []
    assert camera_summary["capture_source"] == capture_source
    for key, path in (("png", CAMERA.DEFAULT_PNG), ("gif", CAMERA.DEFAULT_GIF)):
        assert (
            camera_summary["public_artifacts"][key]["sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
    for key, path in (("png", TELEMETRY.DEFAULT_PNG), ("gif", TELEMETRY.DEFAULT_GIF)):
        assert (
            telemetry_summary[key]["sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_public_builders_refuse_to_overwrite_published_evidence() -> None:
    with pytest.raises(ValueError, match="refusing to overwrite"):
        CAMERA.build(
            CAMERA.DEFAULT_CAPTURE,
            CAMERA.DEFAULT_LOCAL_VIDEO,
            CAMERA.DEFAULT_PNG,
            CAMERA.DEFAULT_GIF,
            CAMERA.DEFAULT_SUMMARY,
            "ffmpeg",
            "ffprobe",
        )
    with pytest.raises(ValueError, match="refusing to overwrite"):
        TELEMETRY.write_outputs(
            TELEMETRY.DEFAULT_INPUT,
            TELEMETRY.DEFAULT_PNG,
            TELEMETRY.DEFAULT_GIF,
            TELEMETRY.DEFAULT_SUMMARY,
        )


def test_no_public_mp4_and_only_fixed_local_rev16_mp4() -> None:
    assert list((ROOT / "docs").rglob("*.mp4")) == []
    summary = json.loads(CAMERA.DEFAULT_SUMMARY.read_text(encoding="utf-8"))
    local_video = summary["local_video"]
    assert local_video["path"] == RECORDER.portable_path(RECORDER.DEFAULT_VIDEO)
    assert local_video["git_policy"] == "local_only"
    assert local_video["bytes"] > 0
    assert len(local_video["sha256"]) == 64
    if RECORDER.DEFAULT_VIDEO.is_file():
        assert RECORDER.DEFAULT_VIDEO.stat().st_size == local_video["bytes"]
        assert (
            hashlib.sha256(RECORDER.DEFAULT_VIDEO.read_bytes()).hexdigest()
            == local_video["sha256"]
        )
