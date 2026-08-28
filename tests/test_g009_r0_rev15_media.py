from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECORDER = load_module(
    "g009_rev15_camera_recorder_test",
    "scripts/record_g009_r0_rev15_gpu_right_side.py",
)
CAMERA = load_module(
    "g009_rev15_camera_media_test",
    "scripts/build_g009_r0_rev15_camera_media.py",
)
TELEMETRY = load_module(
    "g009_rev15_cpu_gpu_media_test",
    "scripts/build_g009_r0_rev15_cpu_gpu_media.py",
)


def test_stage_06_camera_paths_and_condition_are_fixed() -> None:
    assert RECORDER.OUTPUT_STEM == "g009_5_r0_diag_rev15_06_gpu_right_side_force_fail"
    assert RECORDER.SOURCE_ENV_INDEX == 7
    assert RECORDER.POSE_ID == "right_side"
    assert RECORDER.ACTION_MODE == "reset_pose_hold"
    assert RECORDER.DEFAULT_VIDEO.name == f"{RECORDER.OUTPUT_STEM}_s42.mp4"
    assert RECORDER.portable_path(RECORDER.DEFAULT_VIDEO).startswith(
        "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\R0\\diagnostic"
    )
    assert "docs" not in RECORDER.portable_path(RECORDER.DEFAULT_VIDEO).lower()


def test_stage_06_runtime_binding_is_the_exact_gpu_failure_cell() -> None:
    binding = RECORDER.validate_runtime_report(RECORDER.DEFAULT_RUNTIME_REPORT)
    report = binding["report"]
    cell = binding["cell"]
    assert report["device"] == "cuda:0"
    assert report["run_health"]["passed"] is True
    assert report["runtime_contract"]["passed"] is False
    assert report["progression_gate"]["status"] == "runtime_contract_failed"
    assert [name for name, value in report["checks"].items() if not value] == [
        "nonfoot_peak_force_bounded"
    ]
    assert (cell["env_index"], cell["pose_id"], cell["action_mode"]) == (
        7,
        "right_side",
        "reset_pose_hold",
    )
    assert cell["max_nonfoot_force_bodyweights"] == pytest.approx(
        16.78827476501465
    )
    assert cell["max_nonfoot_force_body_name"] == "base"
    assert cell["max_nonfoot_force_physics_step"] == 129
    assert cell["max_nonfoot_force_time_s"] == pytest.approx(0.645)
    assert cell["min_contact_separation_m"] is None


def test_recorder_blocking_cell_schema_is_consumed_by_camera_builder() -> None:
    binding = RECORDER.validate_runtime_report(RECORDER.DEFAULT_RUNTIME_REPORT)
    payload = RECORDER.blocking_cell_payload(binding["cell"])
    CAMERA.validate_blocking_cell(payload)
    assert payload["max_nonfoot_force_threshold_bodyweights"] == 15.0
    assert payload["failed_check"] == "nonfoot_peak_force_bounded"
    assert payload["max_nonfoot_force_time_s"] == pytest.approx(0.645)


def test_stage_06_runtime_binding_rejects_metric_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = json.loads(RECORDER.DEFAULT_RUNTIME_REPORT.read_text(encoding="utf-8"))
    value["pose_mode_metrics"][7]["max_nonfoot_force_bodyweights"] = 15.0
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(RECORDER, "DEFAULT_RUNTIME_REPORT", path)
    monkeypatch.setattr(
        RECORDER,
        "EXPECTED_REPORT_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="nonfoot force peak mismatch"):
        RECORDER.validate_runtime_report(path)


def real_live_capture_contract() -> dict:
    report = json.loads(RECORDER.DEFAULT_RUNTIME_REPORT.read_text(encoding="utf-8"))
    physics = report["physics_readback"]
    return {
        "timing": {
            "physics_dt_s": 0.005,
            "control_dt_s": 0.02,
            "decimation": 4,
            "rollout_steps": 150,
            "rollout_duration_s": 3.0,
        },
        "live_physics_readback": {
            "checks": {
                "articulation_solver_iteration_counts_match_contract": True,
                "rigid_body_max_depenetration_velocity_matches_contract": True,
            },
            "articulation_solver_iterations": physics[
                "articulation_solver_iterations"
            ],
            "readback": physics["rigid_body_max_depenetration_velocity"],
        },
    }


def test_stage_06_live_physics_contract_requires_16_0_and_152_links() -> None:
    capture = real_live_capture_contract()
    CAMERA.validate_live_capture_contract(capture)
    readback = capture["live_physics_readback"]
    assert len(readback["articulation_solver_iterations"]["articulations"]) == 8
    assert readback["readback"]["rigid_body_count"] == 152


def test_stage_06_live_solver_drift_fails_closed() -> None:
    capture = copy.deepcopy(real_live_capture_contract())
    capture["live_physics_readback"]["articulation_solver_iterations"][
        "articulations"
    ][0]["solver_position_iteration_count"] = 8
    with pytest.raises(ValueError, match="solver iteration"):
        CAMERA.validate_live_capture_contract(capture)


def test_stage_06_live_max_depenetration_drift_fails_closed() -> None:
    capture = copy.deepcopy(real_live_capture_contract())
    capture["live_physics_readback"]["readback"]["articulations"][0]["links"][
        0
    ]["max_depenetration_velocity_m_s"] = 0.75
    with pytest.raises(ValueError, match="live link API/value"):
        CAMERA.validate_live_capture_contract(capture)


def test_stage_06_public_labels_cannot_be_mistaken_for_learned_policy() -> None:
    assert CAMERA.OVERLAY_TOP == "G009-5 | REV15 | DIAGNOSTIC | REJECTED | NO PPO"
    assert CAMERA.OVERLAY_BOTTOM.startswith("06 GPU RIGHT_SIDE")
    assert CAMERA.REQUIRED_LABELS == (
        "DIAGNOSTIC",
        "REJECTED",
        "NO PPO",
        "RIGHT_SIDE",
        "RESET_POSE_HOLD",
        "GPU FORCE FAIL",
    )
    assert PureWindowsPath(CAMERA.EXPECTED_LOCAL).suffix == ".mp4"


def test_stage_07_synthesis_is_exactly_bound_to_the_rejection() -> None:
    synthesis = TELEMETRY.read_synthesis(TELEMETRY.DEFAULT_INPUT)
    assert synthesis["evidence_synthesis_valid"] is True
    assert synthesis["candidate_runtime_calibration_passed"] is False
    assert synthesis["device_results"]["cpu"]["runtime_passed_runs"] == 3
    assert synthesis["device_results"]["gpu"]["runtime_passed_runs"] == 0
    assert synthesis["decision"] == {
        "strict_decision": "reject",
        "blocking_device": "gpu",
        "blocking_check": "nonfoot_peak_force_bounded",
        "threshold_bodyweights": 15.0,
        "observed_bodyweights": 16.78827476501465,
        "overrun_bodyweights": 1.7882747650146484,
        "overrun_percent_of_threshold": 11.921831766764322,
    }


def test_stage_07_rejects_a_false_ppo_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = json.loads(TELEMETRY.DEFAULT_INPUT.read_text(encoding="utf-8"))
    value["ppo_training"] = True
    path = tmp_path / "mutated-synthesis.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(TELEMETRY, "DEFAULT_INPUT", path)
    monkeypatch.setattr(
        TELEMETRY,
        "EXPECTED_SYNTHESIS_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="must remain rejected"):
        TELEMETRY.read_synthesis(path)


def test_stage_07_render_is_explicitly_telemetry_not_camera(
    tmp_path: Path,
) -> None:
    assert "TELEMETRY ANIMATION · NOT CAMERA FOOTAGE" in TELEMETRY.LABELS
    assert TELEMETRY.OUTPUT_STEM == "g009_5_r0_diag_rev15_07_cpu_gpu_telemetry"
    output = tmp_path / "frame.png"
    TELEMETRY.render_frame(
        TELEMETRY.read_synthesis(TELEMETRY.DEFAULT_INPUT), 1.0, output
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_stage_07_rejects_output_path_drift_before_writing(tmp_path: Path) -> None:
    png = tmp_path / "wrong.png"
    gif = tmp_path / "wrong.gif"
    summary = tmp_path / "wrong.json"
    with pytest.raises(ValueError, match="numbered input/output paths are fixed"):
        TELEMETRY.write_outputs(TELEMETRY.DEFAULT_INPUT, png, gif, summary)
    assert not any(path.exists() for path in (png, gif, summary))


def test_stage_07_preexisting_target_blocks_all_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    png = tmp_path / "telemetry.png"
    gif = tmp_path / "telemetry.gif"
    summary = tmp_path / "telemetry.json"
    png.write_bytes(b"user-owned")
    monkeypatch.setattr(TELEMETRY, "DEFAULT_PNG", png)
    monkeypatch.setattr(TELEMETRY, "DEFAULT_GIF", gif)
    monkeypatch.setattr(TELEMETRY, "DEFAULT_SUMMARY", summary)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        TELEMETRY.write_outputs(TELEMETRY.DEFAULT_INPUT, png, gif, summary)
    assert png.read_bytes() == b"user-owned"
    assert not gif.exists()
    assert not summary.exists()


def test_stage_07_staging_failure_leaves_no_partial_public_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image  # pyright: ignore[reportMissingImports]

    png = tmp_path / "telemetry.png"
    gif = tmp_path / "telemetry.gif"
    summary = tmp_path / "telemetry.json"
    monkeypatch.setattr(TELEMETRY, "DEFAULT_PNG", png)
    monkeypatch.setattr(TELEMETRY, "DEFAULT_GIF", gif)
    monkeypatch.setattr(TELEMETRY, "DEFAULT_SUMMARY", summary)

    def fake_render(_synthesis, _progress, destination: Path) -> None:
        Image.new("RGB", (16, 16), "black").save(destination)

    original_validate = TELEMETRY.validate_public_media

    def fail_on_gif(path: Path, signature: bytes) -> None:
        if signature == b"GIF8":
            raise ValueError("injected staged GIF failure")
        original_validate(path, signature)

    monkeypatch.setattr(TELEMETRY, "render_frame", fake_render)
    monkeypatch.setattr(TELEMETRY, "validate_public_media", fail_on_gif)
    with pytest.raises(ValueError, match="injected staged GIF failure"):
        TELEMETRY.write_outputs(TELEMETRY.DEFAULT_INPUT, png, gif, summary)
    assert not any(path.exists() for path in (png, gif, summary))


@pytest.mark.parametrize("builder", [CAMERA, TELEMETRY])
def test_exclusive_publish_never_replaces_a_racing_user_target(
    tmp_path: Path, builder
) -> None:
    staged = tmp_path / "staged.bin"
    final = tmp_path / "final.bin"
    staged.write_bytes(b"generated evidence")
    final.write_bytes(b"user-owned race winner")
    with pytest.raises(FileExistsError):
        builder.publish_new(staged, final)
    assert final.read_bytes() == b"user-owned race winner"
    assert staged.read_bytes() == b"generated evidence"


def test_no_mp4_is_public_under_docs() -> None:
    assert list((ROOT / "docs").rglob("*.mp4")) == []
