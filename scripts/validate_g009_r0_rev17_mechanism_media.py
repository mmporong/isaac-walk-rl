#!/usr/bin/env python3
"""Validate the dedicated G009-5-E010 diagnostic media bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_g009_r0_rev17_mechanism_media as media


DEFAULT_SIDECAR = media.DEFAULT_SIDECAR


def _record_path(
    record: Any,
    *,
    parent: Path,
    suffix: str,
    label: str,
) -> Path:
    media.require(isinstance(record, dict), f"{label} record must be an object")
    portable = record.get("path")
    media.require(isinstance(portable, str), f"{label} path must be a string")
    media.require(
        portable and "\\" not in portable and ":" not in portable,
        f"{label} must use a portable repository path",
    )
    relative = Path(portable)
    media.require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"{label} path escaped the repository",
    )
    resolved = media._resolve_direct_child(
        media.REPO_ROOT / relative,
        parent,
        label=label,
        suffix=suffix,
        must_exist=True,
    )
    media.require(media.repo_path(resolved) == portable, f"{label} path is not canonical")
    return resolved


def _local_video_path(record: Any) -> Path:
    media.require(isinstance(record, dict), "local video record must be an object")
    portable = record.get("path")
    prefix = "%USERPROFILE%\\"
    media.require(
        isinstance(portable, str) and portable.startswith(prefix),
        "local video must use a portable %USERPROFILE% path",
    )
    relative = portable[len(prefix) :].replace("\\", "/")
    media.require(".." not in Path(relative).parts, "local video path escaped USERPROFILE")
    resolved = media._resolve_direct_child(
        Path.home() / Path(relative),
        media.LOCAL_VIDEO_DIR,
        label="local video",
        suffix=".mp4",
        must_exist=True,
    )
    media.require(
        media.portable_local_path(resolved) == portable,
        "local video path is not canonical",
    )
    return resolved


def _validate_artifact_record(
    record: dict[str, Any],
    path: Path,
    *,
    label: str,
    local: bool,
) -> None:
    media.require(media._valid_sha256(record.get("sha256")), f"{label} hash is invalid")
    media.require(record.get("sha256") == media.file_sha256(path), f"{label} hash mismatch")
    media.require(record.get("bytes") == path.stat().st_size, f"{label} byte count mismatch")
    if local:
        media.require(
            record.get("tracked_in_git") is False
            and record.get("git_policy") == "local_only",
            "local video Git policy mismatch",
        )
    else:
        media.require(
            record.get("tracked_in_git") is True
            and record.get("git_policy") == "git_public",
            f"{label} Git policy mismatch",
        )


def validate_bundle(sidecar_path: Path = DEFAULT_SIDECAR) -> dict[str, Any]:
    resolved_sidecar = media._resolve_direct_child(
        sidecar_path,
        media.RUNS_DIR,
        label="E010 sidecar",
        suffix=".json",
        must_exist=True,
    )
    sidecar_raw = resolved_sidecar.read_bytes()
    sidecar = json.loads(sidecar_raw.decode("utf-8"))
    media.require(isinstance(sidecar, dict), "sidecar root must be an object")
    media.require(
        sidecar.get("schema_version")
        == "g009.r0.rev17.mechanism_visual_evidence.v1"
        and sidecar.get("goal_id") == "g009"
        and sidecar.get("stage_id") == "R0"
        and sidecar.get("stage_number") == "10"
        and sidecar.get("revision") == "rev17"
        and sidecar.get("evidence_id") == media.EVIDENCE_ID
        and sidecar.get("status") == "diagnostic_complete"
        and sidecar.get("diagnostic_only") is True,
        "E010 diagnostic sidecar identity mismatch",
    )
    media.require(
        sidecar.get("integrity") == {"passed": True, "hash_bound": True},
        "E010 sidecar integrity mismatch",
    )
    media.require(
        sidecar.get("decision")
        == {"outcome": "inconclusive", "selected_lever": None},
        "E010 sidecar decision must remain inconclusive",
    )
    contract = sidecar.get("contract")
    media.require(isinstance(contract, dict), "diagnostic contract is missing")
    media.require(
        contract.get("kind") == "g009_r0_diagnostic_extension",
        "diagnostic contract kind mismatch",
    )
    builder = contract.get("builder_source")
    builder_path = _record_path(
        builder,
        parent=media.REPO_ROOT / "scripts",
        suffix=".py",
        label="media builder",
    )
    _validate_artifact_record(builder, builder_path, label="media builder", local=False)
    media.require(
        builder_path == media.BUILDER_SOURCE,
        "media builder source binding mismatch",
    )
    dedicated = contract.get("dedicated_validator")
    validator_path = _record_path(
        dedicated,
        parent=media.REPO_ROOT / "scripts",
        suffix=".py",
        label="dedicated validator",
    )
    _validate_artifact_record(dedicated, validator_path, label="dedicated validator", local=False)
    media.require(
        validator_path == Path(__file__).resolve()
        and dedicated.get("command")
        == "%PYTHON% scripts/validate_g009_r0_rev17_mechanism_media.py --check-only",
        "dedicated validator binding mismatch",
    )
    standard = contract.get("standard_stage_validator")
    media.require(
        isinstance(standard, dict)
        and standard.get("path") == "scripts/validate_g009_media_contract.py"
        and standard.get("compatible") is False
        and isinstance(standard.get("reason"), str)
        and "diagnostic-only" in standard["reason"],
        "standard stage validator exception is not explicit",
    )

    provenance = sidecar.get("provenance")
    media.require(isinstance(provenance, dict), "sidecar provenance is missing")
    source_record = provenance.get("input")
    source_path = _record_path(
        source_record,
        parent=media.RUNS_DIR,
        suffix=".json",
        label="mechanism source",
    )
    source_raw = source_path.read_bytes()
    media.require(
        source_record.get("sha256") == hashlib.sha256(source_raw).hexdigest(),
        "mechanism source hash mismatch",
    )
    source = media.read_summary(source_path, raw=source_raw)

    summary_record = provenance.get("visual_summary")
    summary_path = _record_path(
        summary_record,
        parent=media.RUNS_DIR,
        suffix=".json",
        label="visual summary",
    )
    summary_raw = summary_path.read_bytes()
    _validate_artifact_record(
        summary_record,
        summary_path,
        label="visual summary",
        local=False,
    )
    media.require(
        summary_record.get("sha256") == hashlib.sha256(summary_raw).hexdigest(),
        "visual summary bytes changed while validating",
    )
    summary = json.loads(summary_raw.decode("utf-8"))
    media.require(
        isinstance(summary, dict)
        and summary.get("schema_version")
        == "g009.r0.rev17.mechanism_visual_summary.v1"
        and summary.get("goal_id") == "g009"
        and summary.get("stage_id") == "R0"
        and summary.get("stage_number") == "10"
        and summary.get("revision") == "rev17"
        and summary.get("evidence_id") == media.EVIDENCE_ID
        and summary.get("status") == "diagnostic_complete"
        and summary.get("diagnostic_only") is True
        and summary.get("camera_footage") is False
        and summary.get("telemetry_animation") is True
        and summary.get("learned_policy_qualified") is False,
        "visual summary identity/governance mismatch",
    )
    media.require(summary.get("source") == source_record, "visual summary source mismatch")
    media.require(
        summary.get("source_binding") == source.get("integrity"),
        "visual summary source lineage mismatch",
    )
    media.require(
        summary.get("decision") == sidecar.get("decision")
        and summary.get("governance")
        == {"ppo": {"status": "not_run"}, "qualification": {"status": "not_run"}},
        "visual summary decision/governance mismatch",
    )
    media.require(
        summary.get("labels") == list(media.LABELS)
        and summary.get("contact_authority")
        == {"cpu": "cpu_only", "gpu": "topology_unavailable"}
        and sidecar.get("contact_authority") == summary.get("contact_authority"),
        "visual summary label/contact-authority contract mismatch",
    )

    public = provenance.get("public_artifacts")
    media.require(isinstance(public, dict), "public artifact records are missing")
    png_record = public.get("png")
    gif_record = public.get("gif")
    png_path = _record_path(
        png_record,
        parent=media.PUBLIC_MEDIA_DIR,
        suffix=".png",
        label="public PNG",
    )
    gif_path = _record_path(
        gif_record,
        parent=media.PUBLIC_MEDIA_DIR,
        suffix=".gif",
        label="public GIF",
    )
    _validate_artifact_record(png_record, png_path, label="public PNG", local=False)
    _validate_artifact_record(gif_record, gif_path, label="public GIF", local=False)
    media.validate_media(png_path, "png")
    media.validate_media(gif_path, "gif")

    video_record = provenance.get("local_video")
    video_path = _local_video_path(video_record)
    _validate_artifact_record(video_record, video_path, label="local video", local=True)
    media.validate_media(video_path, "mp4")
    media.require(
        summary.get("public_artifacts") == public
        and summary.get("local_video") == video_record,
        "visual summary artifact bindings differ from sidecar",
    )
    media.require(
        sidecar.get("labels") == list(media.LABELS)
        and media.DECISION_BANNER.endswith("NO LEVER SELECTED"),
        "required diagnostic labels changed",
    )

    return {
        "schema_version": "g009.r0.rev17.mechanism_media_validation.v1",
        "goal_id": "g009",
        "stage_id": "R0",
        "evidence_id": media.EVIDENCE_ID,
        "mode": "check_only",
        "status": "pass",
        "diagnostic_only": True,
        "decision": sidecar["decision"],
        "sidecar": {
            "path": media.repo_path(resolved_sidecar),
            "sha256": hashlib.sha256(sidecar_raw).hexdigest(),
        },
        "checked": [
            "canonical_input_lineage",
            "builder_source_binding",
            "dedicated_validator_binding",
            "diagnostic_governance",
            "public_png",
            "public_gif",
            "local_mp4",
            "visual_summary",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--check-only", action="store_true", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = validate_bundle(args.sidecar)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema_version": "g009.r0.rev17.mechanism_media_validation.v1",
            "goal_id": "g009",
            "stage_id": "R0",
            "evidence_id": media.EVIDENCE_ID,
            "mode": "check_only",
            "status": "fail",
            "errors": [str(exc)],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
