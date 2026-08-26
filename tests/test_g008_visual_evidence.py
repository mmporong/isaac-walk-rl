from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "reports" / "runs" / "g008_direction_visual_evidence.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_visual_derivatives_match_reported_hashes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    derivatives = report["public_derivatives"]
    assert _sha256(REPO_ROOT / "scripts" / "record_g008_directions.py") == report["record_source_sha256"]

    for key in ("gif", "contact_sheet"):
        metadata = derivatives[key]
        path = REPO_ROOT / metadata["path"]
        assert path.is_file()
        assert path.stat().st_size == metadata["bytes"]
        assert _sha256(path) == metadata["sha256"]

    assert (REPO_ROOT / derivatives["gif"]["path"]).read_bytes().startswith(b"GIF89a")
    assert (REPO_ROOT / derivatives["contact_sheet"]["path"]).read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )


def test_original_video_is_declared_local_only() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    local_video = report["local_video"]
    assert local_video["git_policy"] == "local_only"
    assert local_video["path"].startswith("%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g008\\")
    assert not (REPO_ROOT / "docs" / "media" / "g008" / "g008_directions_s42.mp4").exists()
