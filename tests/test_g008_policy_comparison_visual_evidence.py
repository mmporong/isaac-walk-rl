from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = REPO_ROOT / "reports" / "runs"
FINAL_REPORT = REPORTS_ROOT / "g008_policy_comparison_visual_evidence.json"
CAPTURE_REPORTS = (
    REPORTS_ROOT / "g008_policy_command_capture.json",
    REPORTS_ROOT / "g008_policy_friction_s1_capture.json",
    REPORTS_ROOT / "g008_policy_leg_mass_s1_capture.json",
)
EXPECTED_CHECKPOINT_HASHES = {
    "command": "53cc09043088bcd53618d2ae1f90c7f2e91d01eab7090cc63922486942b2ed47",
    "friction_s1": "40af0a0f80489d705e1e8fdeedd2f765177d3d67bf757709b9195cc2bbeaaee0",
    "leg_mass_s1": "8976cfff6eee6d1a998c7aa554b23d98b01d3d64da02b43ac3133a9186ae97fa",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_ending_equivalent_hashes(path: Path) -> set[str]:
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    return {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(normalized).hexdigest(),
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_isolated_captures_bind_checkpoints_and_runtime_domains() -> None:
    reports = [_load(path) for path in CAPTURE_REPORTS]
    assert [report["profile"]["profile_id"] for report in reports] == [
        "command",
        "friction_s1",
        "leg_mass_s1",
    ]

    for report in reports:
        profile = report["profile"]
        assert report["status"] == "complete"
        assert report["terrain_mode"] == "plane"
        assert report["seed"] == 42
        assert report["headless"] is True
        assert report["record_source_sha256"] in _line_ending_equivalent_hashes(
            REPO_ROOT / "scripts" / "record_g008_policy_comparison.py"
        )
        assert profile["checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_HASHES[profile["profile_id"]]
        assert profile["local_video"]["git_policy"] == "local_only"
        assert profile["local_video"]["path"].startswith(
            "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g008\\"
        )

    command, friction, leg_mass = [report["profile"]["runtime_domain"] for report in reports]
    assert command["leg_mass_scale"]["min"] == command["leg_mass_scale"]["max"] == 1.0
    assert command["foot_static_friction"]["mean"] == 0.800000011920929
    assert command["foot_dynamic_friction"]["mean"] == 0.6000000238418579

    assert 0.72 <= friction["foot_static_friction"]["min"] <= 0.88
    assert 0.52 <= friction["foot_dynamic_friction"]["min"] <= 0.68
    assert friction["leg_mass_scale"]["min"] == friction["leg_mass_scale"]["max"] == 1.0

    assert 0.95 <= leg_mass["leg_mass_scale"]["min"]
    assert leg_mass["leg_mass_scale"]["max"] <= 1.05
    assert leg_mass["leg_mass_scale"]["min"] < leg_mass["leg_mass_scale"]["max"]
    assert leg_mass["foot_static_friction"]["mean"] == 0.800000011920929


def test_comparison_derivatives_match_reported_files() -> None:
    report = _load(FINAL_REPORT)
    assert report["status"] == "complete"
    assert report["profile_order"] == ["command", "friction_s1", "leg_mass_s1"]
    assert report["composition"]["matched_command_sequence"] is True
    assert report["composition"]["synchronized_panels"] is True
    assert report["local_composite"]["git_policy"] == "local_only"
    assert report["local_composite"]["path"].startswith(
        "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g008\\"
    )

    assert report["record_source_sha256"] in _line_ending_equivalent_hashes(
        REPO_ROOT / "scripts" / "record_g008_policy_comparison.py"
    )
    assert report["builder_source_sha256"] in _line_ending_equivalent_hashes(
        REPO_ROOT / "scripts" / "build_g008_comparison_media.py"
    )

    capture_files = report["capture_report"]["files"]
    assert [item["path"] for item in capture_files] == [
        "reports/runs/g008_policy_command_capture.json",
        "reports/runs/g008_policy_friction_s1_capture.json",
        "reports/runs/g008_policy_leg_mass_s1_capture.json",
    ]
    for metadata in capture_files:
        assert metadata["sha256"] in _line_ending_equivalent_hashes(REPO_ROOT / metadata["path"])

    gif = report["public_derivatives"]["gif"]
    contact_sheet = report["public_derivatives"]["contact_sheet"]
    for metadata in (gif, contact_sheet):
        path = REPO_ROOT / metadata["path"]
        assert metadata["git_policy"] == "git_public"
        assert path.stat().st_size == metadata["bytes"]
        assert _sha256(path) == metadata["sha256"]

    assert (REPO_ROOT / gif["path"]).read_bytes().startswith(b"GIF89a")
    assert gif["bytes"] <= 10 * 1024 * 1024
    assert (REPO_ROOT / contact_sheet["path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(REPO_ROOT.rglob("*.mp4"))
