from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reports" / "runs"
CAPTURE_REPORTS = (
    RUNS / "g008_stage_periodic_friction_capture.json",
    RUNS / "g008_stage_link_mass_hip_capture.json",
    RUNS / "g008_stage_link_mass_thigh_capture.json",
    RUNS / "g008_stage_link_mass_calf_capture.json",
    RUNS / "g008_stage_link_mass_foot_capture.json",
)
VISUAL_REPORTS = (
    RUNS / "g008_stage_periodic_friction_visual_evidence.json",
    RUNS / "g008_stage_link_mass_visual_evidence.json",
)
CHECKPOINT_HASHES = {
    "periodic_friction": "40af0a0f80489d705e1e8fdeedd2f765177d3d67bf757709b9195cc2bbeaaee0",
    "link_mass": "8976cfff6eee6d1a998c7aa554b23d98b01d3d64da02b43ac3133a9186ae97fa",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_ending_equivalent_hashes(path: Path) -> set[str]:
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    windows = normalized.replace(b"\n", b"\r\n")
    return {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(normalized).hexdigest(),
        hashlib.sha256(windows).hexdigest(),
    }


def test_stage_captures_bind_runtime_physics_and_local_only_videos() -> None:
    reports = [_load(path) for path in CAPTURE_REPORTS]
    profiles = [report["profile"] for report in reports]
    assert [profile["profile_id"] for profile in profiles] == [
        "periodic_friction_s1_mu020_010",
        "link_mass_hip_120",
        "link_mass_thigh_120",
        "link_mass_calf_120",
        "link_mass_foot_120",
    ]
    for report, profile in zip(reports, profiles):
        assert report["status"] == "complete"
        assert profile["headless"] is True
        assert profile["seed"] == 20260826
        assert profile["total_steps"] == 900
        assert profile["checkpoint"]["sha256"] == CHECKPOINT_HASHES[profile["stage"]]
        assert profile["local_video"]["git_policy"] == "local_only"
        assert profile["local_video"]["path"].startswith(
            "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g008\\"
        )
        assert report["record_source_sha256"] in _line_ending_equivalent_hashes(
            ROOT / "scripts" / "record_g008_stage_evidence.py"
        )

    periodic = profiles[0]["stage_physics"]
    assert periodic["case"]["id"] == "mixed_020_010"
    assert periodic["underlay"]["default_ground_collision_exists"] is False
    assert periodic["visual_overlay"]["collision_api_applied"] is False
    assert periodic["visual_overlay"]["physics_unchanged"] is True
    assert periodic["visual_overlay"]["low_friction"] == [0.2, 0.1]
    assert periodic["visual_overlay"]["high_friction"] == [0.8, 0.6]

    expected_totals = {
        "hip": 8.638399124145508,
        "thigh": 9.017599105834961,
        "calf": 8.219199180603027,
        "foot": 8.128000259399414,
    }
    for profile in profiles[1:]:
        physics = profile["stage_physics"]
        assert physics["mass_ratio"]["mean"] == pytest.approx(1.2, abs=1.0e-6)
        assert physics["total_leg_mass_kg"]["mean"] == pytest.approx(
            expected_totals[physics["group"]], abs=1.0e-6
        )
        assert physics["inertia_scale_absolute_error_max"] <= 1.0e-8
        assert physics["center_of_mass_changed"] is False


def test_stage_gifs_and_screenshots_are_public_but_mp4s_are_not() -> None:
    expected_stages = ["periodic_friction", "link_mass"]
    for report_path, expected_stage in zip(VISUAL_REPORTS, expected_stages):
        report = _load(report_path)
        assert report["status"] == "complete"
        assert report["stage"] == expected_stage
        assert report["composition"]["matched_command_sequence"] is True
        assert report["local_composite"]["git_policy"] == "local_only"
        assert report["local_composite"]["path"].startswith(
            "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g008\\"
        )
        assert report["record_source_sha256"] in _line_ending_equivalent_hashes(
            ROOT / "scripts" / "record_g008_stage_evidence.py"
        )
        assert report["builder_source_sha256"] in _line_ending_equivalent_hashes(
            ROOT / "scripts" / "build_g008_stage_media.py"
        )
        quantitative = ROOT / report["quantitative_report"]["path"]
        assert _sha256(quantitative) == report["quantitative_report"]["sha256"]

        gif = report["public_derivatives"]["gif"]
        screenshot = report["public_derivatives"]["contact_sheet"]
        for metadata in (gif, screenshot):
            path = ROOT / metadata["path"]
            assert metadata["git_policy"] == "git_public"
            assert path.stat().st_size == metadata["bytes"]
            assert _sha256(path) == metadata["sha256"]
        assert (ROOT / gif["path"]).read_bytes().startswith(b"GIF89a")
        assert gif["bytes"] <= 10 * 1024 * 1024
        assert (ROOT / screenshot["path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    assert not list(ROOT.rglob("*.mp4"))
