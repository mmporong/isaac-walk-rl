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


RECORDER = load(
    "g009_rev14_camera_recorder_test", "scripts/record_g009_r0_rev14_right_side.py"
)
CAMERA = load(
    "g009_rev14_camera_media_test", "scripts/build_g009_r0_rev14_camera_media.py"
)
TRADEOFF = load(
    "g009_rev14_tradeoff_media_test", "scripts/build_g009_r0_rev14_tradeoff_media.py"
)


def test_capture_identity_is_fixed_local_only_and_no_overwrite(tmp_path: Path) -> None:
    assert RECORDER.OUTPUT_STEM == "g009_5_r0_diag_rev14_04_right_side_tradeoff"
    assert RECORDER.DEFAULT_VIDEO.name == f"{RECORDER.OUTPUT_STEM}_s42.mp4"
    assert RECORDER.portable_path(RECORDER.DEFAULT_VIDEO).startswith(
        "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\R0\\diagnostic\\"
    )
    video, report = tmp_path / "video.mp4", tmp_path / "capture.json"
    video.write_bytes(b"existing")
    with pytest.raises(ValueError, match="fixed"):
        RECORDER.validate_output_paths(tmp_path, video, report)


def test_capture_rejects_existing_raw_prefix(monkeypatch, tmp_path: Path) -> None:
    monkey_video = tmp_path / f"{RECORDER.OUTPUT_STEM}_s42.mp4"
    monkey_report = tmp_path / "capture.json"
    raw_prefix = RECORDER.OUTPUT_STEM.replace("_", "-")
    (tmp_path / f"{raw_prefix}-episode-0.mp4").write_bytes(b"preserve-me")
    monkeypatch.setattr(RECORDER, "DEFAULT_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(RECORDER, "DEFAULT_VIDEO", monkey_video)
    monkeypatch.setattr(RECORDER, "DEFAULT_CAPTURE_REPORT", monkey_report)
    with pytest.raises(ValueError, match="raw prefix already exists"):
        RECORDER.validate_output_paths(tmp_path, monkey_video, monkey_report)


def test_capture_timing_and_live_topology_contract_are_explicit() -> None:
    assert RECORDER.EXPECTED_PHYSICS_DT_S == 0.005
    assert RECORDER.EXPECTED_CONTROL_DT_S == 0.02
    assert len(RECORDER.EXPECTED_BODY_NAMES) == 19
    assert RECORDER.EXPECTED_BODY_NAMES[0] == "base"
    assert RECORDER.EXPECTED_BODY_NAMES[-1] == "RR_foot"


def test_runtime_report_binding_matches_exact_tradeoff_cell() -> None:
    binding = RECORDER.validate_runtime_report(RECORDER.DEFAULT_RUNTIME_REPORT)
    cell = binding["cell"]
    assert (cell["env_index"], cell["pose_id"], cell["action_mode"]) == (
        3,
        "right_side",
        "zero_normalized",
    )
    assert cell["min_contact_separation_m"] == -0.010990187525749207
    assert cell["min_contact_separation_provenance"]["physics_step"] == 2
    assert cell["min_contact_separation_provenance"]["actor0_path"].endswith("/RL_foot")
    assert cell["max_nonfoot_force_bodyweights"] == 4.611482620239258
    assert cell["max_nonfoot_force_body_name"] == "FL_thigh"
    assert cell["max_nonfoot_force_physics_step"] == 11


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report.__setitem__("contract_sha256", "0" * 64),
            "contract mismatch",
        ),
        (
            lambda report: report["required_crosschecks"][
                "cpu_contact_separation"
            ].__setitem__("passed", True),
            "separation failure",
        ),
        (
            lambda report: report["pose_mode_metrics"][3].__setitem__(
                "min_contact_separation_m", -0.009
            ),
            "minimum separation",
        ),
    ],
)
def test_runtime_binding_fails_closed_without_writing_files(
    monkeypatch, mutation, message: str
) -> None:
    report = json.loads(RECORDER.DEFAULT_RUNTIME_REPORT.read_text(encoding="utf-8"))
    mutation(report)
    monkeypatch.setattr(
        RECORDER, "file_sha256", lambda _: RECORDER.EXPECTED_REPORT_SHA256
    )
    monkeypatch.setattr(RECORDER, "read_json", lambda _: report)
    with pytest.raises(ValueError, match=message):
        RECORDER.validate_runtime_report(RECORDER.DEFAULT_RUNTIME_REPORT)


def valid_bundle() -> dict:
    return {
        "source_bundle_sha256": RECORDER.EXPECTED_SOURCE_BUNDLE_SHA256,
        "all_files_present": True,
        "git_commit_valid": True,
        "clean": True,
    }


def test_current_capture_binding_requires_clean_exact_bundle_and_contract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(RECORDER, "source_bundle_provenance", valid_bundle)
    monkeypatch.setattr(RECORDER, "recover_contract", lambda: {"revision": "rev14"})
    monkeypatch.setattr(
        RECORDER, "canonical_sha256", lambda _: RECORDER.EXPECTED_CONTRACT_SHA256
    )
    assert (
        RECORDER.validate_current_capture_binding()["contract_sha256"]
        == RECORDER.EXPECTED_CONTRACT_SHA256
    )
    dirty = valid_bundle()
    dirty["clean"] = False
    monkeypatch.setattr(RECORDER, "source_bundle_provenance", lambda: dirty)
    with pytest.raises(ValueError, match="dirty"):
        RECORDER.validate_current_capture_binding()


def test_camera_labels_and_numbering_are_exact() -> None:
    assert CAMERA.OUTPUT_STEM == RECORDER.OUTPUT_STEM
    assert CAMERA.REQUIRED_LABELS == (
        "DIAGNOSTIC",
        "REJECTED",
        "NO PPO",
        "RIGHT_SIDE",
        "ZERO_NORMALIZED",
        "CPU SEPARATION FAIL",
    )
    assert CAMERA.OVERLAY_TOP == "G009-5 | REV14 | DIAGNOSTIC | REJECTED | NO PPO"
    assert (
        CAMERA.OVERLAY_BOTTOM == "04 RIGHT_SIDE | ZERO_NORMALIZED | CPU SEPARATION FAIL"
    )
    assert CAMERA.DEFAULT_GIF.name == f"{CAMERA.OUTPUT_STEM}.gif"
    assert CAMERA.DEFAULT_PNG.name == f"{CAMERA.OUTPUT_STEM}.png"


def valid_capture(tmp_path: Path) -> tuple[dict, Path]:
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"actual-camera")
    capture = {
        "camera_footage": True,
        "telemetry_animation": False,
        "headless": True,
        "offscreen": True,
        "status": "rejected",
        "diagnostic_only": True,
        "qualification_status": "not_run",
        "learned": False,
        "ppo_checkpoint_used": False,
        "timing": {
            "physics_dt_s": 0.005,
            "control_dt_s": 0.02,
            "decimation": 4,
            "rollout_steps": 150,
            "rollout_duration_s": 3.0,
        },
        "live_physics_readback": {
            "checks": {"rigid_body_max_depenetration_velocity_matches_contract": True},
            "readback": {
                "articulation_group_count": 8,
                "rigid_body_count": 152,
                "duplicate_link_prim_paths": [],
                "authoritative_body_names": list(CAMERA.EXPECTED_BODY_NAMES),
                "articulations": [
                    {
                        "articulation_index": articulation_index,
                        "authoritative_body_names": list(CAMERA.EXPECTED_BODY_NAMES),
                        "links": [
                            {
                                "body_index": body_index,
                                "body_name": body_name,
                                "prim_path": f"/World/envs/env_{articulation_index}/Robot/{body_name}",
                                "prim_valid": True,
                                "usd_rigid_body_api": True,
                                "physx_rigid_body_api": True,
                                "max_depenetration_velocity_m_s": 0.75,
                                "error": None,
                            }
                            for body_index, body_name in enumerate(
                                CAMERA.EXPECTED_BODY_NAMES
                            )
                        ],
                    }
                    for articulation_index in range(8)
                ],
            },
        },
        "completed_stages": {
            "cpu_runtime_3x": True,
            "gpu_runtime_3x": True,
            "strict_tradeoff_synthesis": True,
        },
        "blocked_stages": {"gate01": True, "gate10": True, "ppo_training": True},
        "labels": list(CAMERA.REQUIRED_LABELS),
        "source_env_index": 3,
        "pose_id": "right_side",
        "action_mode": "zero_normalized",
        "source": {
            "original_runtime_binding": {
                "commit": CAMERA.EXPECTED_RUNTIME_COMMIT,
                "bundle_sha256": CAMERA.EXPECTED_SOURCE_BUNDLE_SHA256,
                "contract_sha256": CAMERA.EXPECTED_CONTRACT_SHA256,
            },
            "current_capture_binding": {
                "capture_commit": "1" * 40,
                "contract_sha256": CAMERA.EXPECTED_CONTRACT_SHA256,
                "source_bundle": {
                    "source_bundle_sha256": CAMERA.EXPECTED_SOURCE_BUNDLE_SHA256,
                    "clean": True,
                    "all_files_present": True,
                },
            },
        },
        "original_rev14_report_binding": {
            "sha256": CAMERA.EXPECTED_RUNTIME_REPORT_SHA256,
            "execution_id": CAMERA.EXPECTED_EXECUTION_ID,
            "tradeoff_cell": {
                "min_contact_separation_m": -0.010990187525749207,
                "separation_body": "RL_foot",
                "separation_physics_step": 2,
                "max_nonfoot_force_bodyweights": 4.611482620239258,
                "max_nonfoot_force_body_name": "FL_thigh",
                "max_nonfoot_force_physics_step": 11,
            },
        },
        "local_video": {
            "path": str(CAMERA.EXPECTED_LOCAL),
            "sha256": CAMERA.file_sha256(video),
            "bytes": video.stat().st_size,
        },
    }
    return capture, video


def test_camera_media_accepts_only_actual_bound_footage(
    monkeypatch, tmp_path: Path
) -> None:
    capture, video = valid_capture(tmp_path)
    monkeypatch.setattr(CAMERA, "read_json", lambda _: capture)
    monkeypatch.setattr(CAMERA, "resolve_portable", lambda _: video)
    actual, source = CAMERA.validate_capture(CAMERA.DEFAULT_CAPTURE)
    assert actual["camera_footage"] is True and source == video.resolve()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("camera_footage", False, "camera footage"),
        ("telemetry_animation", True, "camera footage"),
        ("learned", True, "NO PPO"),
        ("action_mode", "reset_pose_hold", "failure cell"),
    ],
)
def test_camera_media_rejects_claim_drift(
    monkeypatch, tmp_path: Path, field: str, value: object, message: str
) -> None:
    capture, video = valid_capture(tmp_path)
    capture[field] = value
    monkeypatch.setattr(CAMERA, "read_json", lambda _: capture)
    monkeypatch.setattr(CAMERA, "resolve_portable", lambda _: video)
    with pytest.raises(ValueError, match=message):
        CAMERA.validate_capture(CAMERA.DEFAULT_CAPTURE)


def test_camera_media_rejects_timing_and_live_readback_drift(
    monkeypatch, tmp_path: Path
) -> None:
    capture, video = valid_capture(tmp_path)
    capture["timing"]["physics_dt_s"] = 0.01
    monkeypatch.setattr(CAMERA, "read_json", lambda _: capture)
    monkeypatch.setattr(CAMERA, "resolve_portable", lambda _: video)
    with pytest.raises(ValueError, match="timing contract"):
        CAMERA.validate_capture(CAMERA.DEFAULT_CAPTURE)

    capture, video = valid_capture(tmp_path)
    capture["live_physics_readback"]["readback"]["articulations"][0]["links"][0][
        "max_depenetration_velocity_m_s"
    ] = 1.0
    monkeypatch.setattr(CAMERA, "read_json", lambda _: capture)
    monkeypatch.setattr(CAMERA, "resolve_portable", lambda _: video)
    with pytest.raises(ValueError, match="API/value"):
        CAMERA.validate_capture(CAMERA.DEFAULT_CAPTURE)


def test_png_probe_is_pure_and_does_not_require_timing(
    monkeypatch, tmp_path: Path
) -> None:
    payload = {
        "streams": [
            {"codec_type": "video", "codec_name": "png", "width": 1280, "height": 720}
        ],
        "format": {},
    }
    monkeypatch.setattr(
        CAMERA.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout=json.dumps(payload)),
    )
    assert CAMERA.ffprobe_summary(
        tmp_path / "still.png", "ffprobe", require_timing=False
    ) == {"codec": "png", "width": 1280, "height": 720}


def test_tradeoff_contract_is_explicit_without_generated_media() -> None:
    assert TRADEOFF.LABELS == (
        "PUBLIC DIAGNOSTIC",
        "TELEMETRY ANIMATION",
        "NOT CAMERA FOOTAGE",
        "05 FORCE/SEPARATION",
        "NO PPO",
        "REJECTED",
    )
    assert (
        TRADEOFF.FORCE_OBSERVED_BW == 13.943856239318848 <= TRADEOFF.FORCE_THRESHOLD_BW
    )
    assert (
        TRADEOFF.SEPARATION_OBSERVED_M
        == -0.010990187525749207
        < TRADEOFF.SEPARATION_THRESHOLD_M
    )
    assert TRADEOFF.DEFAULT_GIF.name == f"{TRADEOFF.OUTPUT_STEM}.gif"
    assert "_05_cpu_tradeoff" in TRADEOFF.OUTPUT_STEM


def test_tradeoff_synthesis_contract_accepts_only_bound_real_summary() -> None:
    synthesis = TRADEOFF.read_synthesis(TRADEOFF.DEFAULT_INPUT)
    assert synthesis["tradeoff"]["strict_decision"] == "reject"


def test_tradeoff_synthesis_rejects_unbound_minimal_summary(
    monkeypatch, tmp_path: Path
) -> None:
    synthesis = {
        "experiment": "rev14_max_depenetration_velocity_tradeoff",
        "status": "rejected_before_gate01",
        "learned": False,
        "qualification_status": "not_run",
        "repeatability": {
            "unique_execution_ids": 6,
            "cpu": {"validated_runs": 3},
            "gpu": {"validated_runs": 3},
        },
        "tradeoff": {
            "strict_decision": "reject",
            "cpu_global_peak_bodyweights": TRADEOFF.FORCE_OBSERVED_BW,
            "separation_threshold_m": TRADEOFF.SEPARATION_THRESHOLD_M,
            "cpu_worst_separation_m": TRADEOFF.SEPARATION_OBSERVED_M,
        },
        "completed_stages": {
            "cpu_runtime_3x": True,
            "gpu_runtime_3x": True,
            "strict_tradeoff_synthesis": True,
        },
        "blocked_stages": {"gate01": True, "gate10": True, "ppo_training": True},
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(synthesis), encoding="utf-8")
    monkeypatch.setattr(TRADEOFF, "DEFAULT_INPUT", path)
    monkeypatch.setattr(
        TRADEOFF, "EXPECTED_SYNTHESIS_SHA256", TRADEOFF.file_sha256(path)
    )
    with pytest.raises(ValueError, match="source lineage"):
        TRADEOFF.read_synthesis(path)


def test_camera_and_telemetry_contracts_are_distinct() -> None:
    assert "DIAGNOSTIC" in CAMERA.REQUIRED_LABELS
    assert "NOT CAMERA FOOTAGE" not in CAMERA.REQUIRED_LABELS
    assert "TELEMETRY ANIMATION" in TRADEOFF.LABELS
    assert "NOT CAMERA FOOTAGE" in TRADEOFF.LABELS
