from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "record_g009_s0.py"


def _load_recorder():
    spec = importlib.util.spec_from_file_location("record_g009_s0", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_import_is_isaac_free_and_contract_uses_three_config_profiles() -> None:
    before = set(sys.modules)
    recorder = _load_recorder()
    newly_loaded = set(sys.modules) - before
    assert not any(name == "isaaclab" or name.startswith("isaaclab.") for name in newly_loaded)
    assert not any(name == "isaacsim" or name.startswith("isaacsim.") for name in newly_loaded)

    contract = recorder.load_capture_contract(ROOT / "configs" / "g009_s0.json")
    assert contract.task_id == "Isaac-G009-Velocity-Slope-Go2-S0-v0"
    assert contract.seed == 20260828
    assert [(item.profile_id, item.slope_deg, item.terrain_azimuth_deg) for item in contract.profiles] == [
        ("slope_05", 5.0, 0.0),
        ("slope_15", 15.0, 0.0),
        ("slope_25_stress", 25.0, 0.0),
    ]


def test_profile_lookup_and_sequence_are_fail_closed() -> None:
    recorder = _load_recorder()
    contract = recorder.load_capture_contract(ROOT / "configs" / "g009_s0.json")
    assert recorder.profile_by_id(contract, "slope_15").slope_deg == 15.0
    with pytest.raises(ValueError, match="not found or duplicated"):
        recorder.profile_by_id(contract, "missing")

    expected = [
        ("stand", 75, (0.0, 0.0, 0.0)),
        ("contour_left", 200, (0.4, 0.0, 0.0)),
        ("stand", 50, (0.0, 0.0, 0.0)),
        ("contour_right", 200, (-0.4, 0.0, 0.0)),
    ]
    assert [(item.name, item.steps, item.command) for item in contract.sequence] == expected
    assert recorder.command_at_step(contract.sequence, 74)[1].name == "stand"
    assert recorder.command_at_step(contract.sequence, 75)[1].name == "contour_left"
    assert recorder.command_at_step(contract.sequence, 274)[1].name == "contour_left"
    assert recorder.command_at_step(contract.sequence, 275)[1].name == "stand"
    assert recorder.command_at_step(contract.sequence, 524)[1].name == "contour_right"
    with pytest.raises(IndexError):
        recorder.command_at_step(contract.sequence, 525)


def test_invalid_sequence_is_rejected_from_config(tmp_path: Path) -> None:
    recorder = _load_recorder()
    config = json.loads((ROOT / "configs" / "g009_s0.json").read_text(encoding="utf-8"))
    config["visual_protocol"]["sequence"][1]["command"] = [-0.4, 0.0, 0.0]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="positive/negative body-x"):
        recorder.load_capture_contract(invalid)


def test_portable_path_round_trip_and_local_video_boundary() -> None:
    recorder = _load_recorder()
    local_video = recorder.DEFAULT_OUTPUT_DIR / "slope_15" / "capture.mp4"
    portable = recorder.portable_path(local_video)
    assert portable.startswith("%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\S0\\")
    assert recorder.resolve_portable_path(portable) == local_video.resolve()
    assert recorder.validate_output_dir(local_video.parent) == local_video.parent.resolve()
    with pytest.raises(ValueError, match="MP4 output must remain"):
        recorder.validate_output_dir(ROOT / "artifacts" / "videos")


def test_atomic_report_write_refuses_existing_output(tmp_path: Path) -> None:
    recorder = _load_recorder()
    output = tmp_path / "nested" / "capture.json"
    recorder._write_json_atomic_new(output, {"status": "complete"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "complete"}
    with pytest.raises(FileExistsError):
        recorder._write_json_atomic_new(output, {"status": "replacement"})
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_prelaunch_output_validation_is_local_and_fail_closed(tmp_path: Path) -> None:
    recorder = _load_recorder()
    contract = recorder.load_capture_contract(ROOT / "configs" / "g009_s0.json")
    profile = recorder.profile_by_id(contract, "slope_05")
    output_dir = recorder.DEFAULT_OUTPUT_DIR / "pytest-prelaunch"
    report = tmp_path / "capture.json"
    expected = recorder.validate_prelaunch_outputs(output_dir, report, profile, contract.seed)
    assert expected.name == "g009_s0_slope_05_s20260828.mp4"

    report.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        recorder.validate_prelaunch_outputs(output_dir, report, profile, contract.seed)


def test_terrain_axes_make_body_forward_contour_left_after_ninety_degree_yaw() -> None:
    recorder = _load_recorder()
    axes = recorder._terrain_axes_3d(25.0, 0.0)
    assert axes["contour_left"] == pytest.approx((0.0, 1.0, 0.0), abs=1.0e-12)
    assert axes["downhill"] == pytest.approx(
        (-0.9063077870, 0.0, -0.4226182617), abs=1.0e-9
    )
    assert sum(a * b for a, b in zip(axes["normal"], axes["downhill"])) == pytest.approx(0.0, abs=1.0e-12)


def test_config_checkpoint_path_is_portable_and_hash_is_pinned() -> None:
    recorder = _load_recorder()
    contract = recorder.load_capture_contract(ROOT / "configs" / "g009_s0.json")
    checkpoint = contract.raw["parent_checkpoint"]
    assert checkpoint["path"].startswith("%USERPROFILE%\\IsaacLab\\logs\\rsl_rl\\")
    assert len(checkpoint["sha256"]) == 64
    assert recorder.resolve_portable_path(checkpoint["path"]).is_absolute()


def test_generated_capture_report_matches_builder_schema() -> None:
    recorder = _load_recorder()
    contract = recorder.load_capture_contract(ROOT / "configs" / "g009_s0.json")
    profile = recorder.profile_by_id(contract, "slope_15")
    material = contract.raw["terrain"]["ground_material"]
    physics = {
        "slope_deg": profile.slope_deg,
        "terrain_azimuth_deg": profile.terrain_azimuth_deg,
        "ground_material": dict(material),
        "single_mesh": True,
    }
    report = recorder.build_capture_report(
        contract=contract,
        profile=profile,
        source_commit="a" * 40,
        dirty_paths=[],
        headless=True,
        step_dt_s=0.02,
        camera={"resolution": [1280, 720], "origin_type": "env"},
        checkpoint=dict(contract.raw["parent_checkpoint"]),
        physics_readback=physics,
        metrics={"segments": []},
        local_video={
            "path": "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\S0\\g009_s0_slope_15_s20260828.mp4",
            "sha256": "b" * 64,
            "bytes": 123,
            "git_policy": "local_only",
        },
        record_source_sha256="c" * 64,
    )

    assert report["config"] == {
        "path": "configs/g009_s0.json",
        "sha256": recorder.file_sha256(ROOT / "configs" / "g009_s0.json"),
    }
    assert report["dirty_paths"] == []
    assert report["profile"]["sequence"] == contract.raw["visual_protocol"]["sequence"]
    assert report["physics_readback"]["slope_deg"] == 15.0
    assert report["physics_readback"]["terrain_azimuth_deg"] == 0.0
    assert report["physics_readback"]["ground_material"] == material
    assert report["checkpoint"] == contract.raw["parent_checkpoint"]


def test_dirty_filter_allows_only_prior_sequential_capture_reports() -> None:
    recorder = _load_recorder()
    paths = [
        "reports/runs/g009_s0_slope_05_capture.json",
        "reports\\runs\\g009_s0_slope_15_capture.json",
        "reports/runs/g009_s0_slope_25_stress_capture.json",
        "reports/runs/g009_s0_visual_summary.json",
        "scripts/record_g009_s0.py",
    ]
    assert recorder.filter_source_dirty_paths(paths) == [
        "reports/runs/g009_s0_visual_summary.json",
        "scripts/record_g009_s0.py",
    ]


def test_source_snapshot_is_fail_closed_and_detects_mid_recording_change(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _load_recorder()
    clean = recorder.SourceSnapshot("a" * 40, (), "b" * 64, "c" * 64)
    recorder.require_clean_source_snapshot(clean)
    with pytest.raises(RuntimeError, match="clean source tree"):
        recorder.require_clean_source_snapshot(
            recorder.SourceSnapshot("a" * 40, ("scripts/record_g009_s0.py",), "b" * 64, "c" * 64)
        )

    monkeypatch.setattr(recorder, "capture_source_snapshot", lambda _path: clean)
    recorder.verify_source_snapshot_unchanged(clean, ROOT / "configs" / "g009_s0.json")
    changed = recorder.SourceSnapshot("a" * 40, (), "d" * 64, "c" * 64)
    monkeypatch.setattr(recorder, "capture_source_snapshot", lambda _path: changed)
    with pytest.raises(RuntimeError, match="changed during recording"):
        recorder.verify_source_snapshot_unchanged(clean, ROOT / "configs" / "g009_s0.json")


def test_base_contact_termination_is_required_without_fallback() -> None:
    recorder = _load_recorder()
    marker = object()
    recorder.require_base_contact_termination(lambda name: marker if name == "base_contact" else None)

    def missing(_name: str):
        raise KeyError("base_contact")

    with pytest.raises(RuntimeError, match="requires the base_contact"):
        recorder.require_base_contact_termination(missing)
    with pytest.raises(RuntimeError, match="requires the base_contact"):
        recorder.require_base_contact_termination(lambda _name: None)


def test_terrain_readback_validates_requested_slope_azimuth_and_normal() -> None:
    recorder = _load_recorder()
    profile = recorder.CaptureProfile("slope_25_stress", 25.0, 35.0)
    normal = recorder._terrain_axes_3d(profile.slope_deg, profile.terrain_azimuth_deg)["normal"]
    errors = recorder.validate_terrain_readback(
        {"measured_slope_deg": 25.0, "first_triangle_normal_w": list(normal)}, profile
    )
    assert errors == pytest.approx(
        {"slope_error_deg": 0.0, "azimuth_error_deg": 0.0, "normal_error_deg": 0.0},
        abs=1.0e-10,
    )
    with pytest.raises(RuntimeError, match="terrain geometry readback"):
        recorder.validate_terrain_readback(
            {"measured_slope_deg": 24.0, "first_triangle_normal_w": list(normal)}, profile
        )
    wrong_azimuth_normal = recorder._terrain_axes_3d(25.0, 36.0)["normal"]
    with pytest.raises(RuntimeError, match="terrain geometry readback"):
        recorder.validate_terrain_readback(
            {"measured_slope_deg": 25.0, "first_triangle_normal_w": list(wrong_azimuth_normal)}, profile
        )


def test_reset_alignment_gate_rejects_support_body_up_and_contour_errors() -> None:
    recorder = _load_recorder()
    recorder.validate_reset_alignment(
        support_normal_error_deg=0.01,
        body_up_error_deg=0.02,
        body_x_contour_left_error_deg=0.03,
    )
    for field in (
        "support_normal_error_deg",
        "body_up_error_deg",
        "body_x_contour_left_error_deg",
    ):
        values = {
            "support_normal_error_deg": 0.0,
            "body_up_error_deg": 0.0,
            "body_x_contour_left_error_deg": 0.0,
        }
        values[field] = recorder.RESET_ALIGNMENT_TOLERANCE_DEG + 0.01
        with pytest.raises(RuntimeError, match="reset alignment"):
            recorder.validate_reset_alignment(**values)
