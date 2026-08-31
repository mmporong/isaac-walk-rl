#!/usr/bin/env python3
"""Validate rev23 matrix-observation telemetry media and sidecar fail-closed."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, cast

import build_g009_r0_rev23_matrix_observation_adapter_media as media


def _artifact(record: Any, path: Path, *, local: bool, metadata: Mapping[str, Any] | None = None) -> None:
    media.require(isinstance(record, Mapping), "artifact record missing")
    value = cast(Mapping[str, Any], record)
    expected_path = media.portable_local_path(path) if local else media.repo_path(path)
    media.require(value.get("path") == expected_path, "artifact canonical path mismatch")
    media.require(value.get("sha256") == media.file_sha256(path) and value.get("bytes") == path.stat().st_size, "artifact hash/size mismatch")
    media.require(value.get("intended_for_git") is (not local) and value.get("git_policy") == ("local_only" if local else "git_public_after_review"), "artifact publication policy mismatch")
    if metadata:
        for key, expected in metadata.items(): media.require(value.get(key) == expected, f"artifact metadata mismatch: {key}")


def validate_bundle(phase: str, sidecar_path: Path | None = None) -> dict[str, Any]:
    paths = media.phase_paths(phase); sidecar_path = sidecar_path or paths["sidecar"]
    media.require(sidecar_path.resolve() == paths["sidecar"].resolve(), "sidecar canonical path mismatch")
    sidecar, _ = media.read_json(sidecar_path)
    sequence = "14.01" if phase == "cpu" else "14.02"
    expected_keys = {"schema_version", "goal_id", "stage_id", "stage_number", "sequence_number", "revision", "evidence_id", "phase", "status", "integrity", "labels", "claim_limits", "governance", "input_bindings", "source", "artifacts", "contract"}
    media.require(set(sidecar) == expected_keys, "sidecar top-level schema mismatch")
    media.require(sidecar.get("schema_version") == "g009.r0.rev23.matrix_observation_adapter_visual_evidence.v1" and sidecar.get("goal_id") == "g009" and sidecar.get("stage_id") == "R0" and sidecar.get("stage_number") == media.STAGE_NUMBER and sidecar.get("sequence_number") == sequence and sidecar.get("revision") == "rev23" and sidecar.get("evidence_id") == media.EVIDENCE_ID and sidecar.get("phase") == phase and sidecar.get("status") == "diagnostic_complete", "sidecar identity mismatch")
    media.require(sidecar.get("labels") == [sequence, media.HEADER], "required frame labels mismatch")
    media.require(sidecar.get("claim_limits") == media.CLAIM_LIMITS, "claim limits mismatch")
    media.require(sidecar.get("governance") == media.GOVERNANCE, "governance mismatch")
    media.require(sidecar.get("integrity") == {"passed": True, "hash_bound": True, "all_inputs_revalidated": True, "no_overwrite": True}, "integrity mismatch")

    canonical = media.validate_inputs(phase, media.expected_inputs(phase))
    media.require(sidecar.get("input_bindings") == canonical["input_bindings"], "input bindings mismatch")
    media.require(sidecar.get("source") == {"git_commit": canonical["git_commit"], "source_bundle_sha256": canonical["source_bundle_sha256"]}, "source binding mismatch")

    artifacts = cast(Mapping[str, Any], sidecar.get("artifacts", {})); public = cast(Mapping[str, Any], artifacts.get("public", {}))
    media.require(set(artifacts) == {"visual_summary", "public", "local_video"} and set(public) == {"gif", "png"}, "artifact set mismatch")
    for path, kind in ((paths["png"], "png"), (paths["gif"], "gif"), (paths["video"], "mp4")): media.validate_media(path, kind)
    _artifact(artifacts.get("visual_summary"), paths["summary"], local=False)
    _artifact(public.get("png"), paths["png"], local=False, metadata={"width": media.WIDTH, "height": media.HEIGHT, "representative_frame": media.FRAME_COUNT})
    _artifact(public.get("gif"), paths["gif"], local=False, metadata={"width": media.WIDTH, "height": media.HEIGHT, "frame_count": media.FRAME_COUNT, "duration_ms": media.FRAME_COUNT * media.FRAME_DURATION_MS})
    video_metadata = {"codec": "h264", "width": media.WIDTH, "height": media.HEIGHT, "fps": "30/1", "frames": media.VIDEO_FRAME_COUNT, "duration_seconds": media.VIDEO_DURATION_SECONDS}
    _artifact(artifacts.get("local_video"), paths["video"], local=True, metadata=video_metadata)
    media.require(media.ffprobe_metadata(paths["video"]) == video_metadata, "MP4 metadata mismatch")

    from PIL import Image
    with Image.open(paths["png"]) as image: media.require(image.format == "PNG" and image.size == (media.WIDTH, media.HEIGHT), "PNG metadata mismatch")
    with Image.open(paths["gif"]) as image:
        media.require(image.format == "GIF" and image.size == (media.WIDTH, media.HEIGHT) and getattr(image, "n_frames", 1) == media.FRAME_COUNT, "GIF metadata mismatch")
        durations = []
        for index in range(media.FRAME_COUNT): image.seek(index); durations.append(image.info.get("duration"))
        media.require(durations == [media.FRAME_DURATION_MS] * media.FRAME_COUNT, "GIF duration mismatch")

    summary, _ = media.read_json(paths["summary"])
    summary_keys = {"schema_version", "goal_id", "stage_id", "stage_number", "sequence_number", "revision", "evidence_id", "phase", "status", "labels", "claim_limits", "governance", "input_bindings", "source", "telemetry", "decision", "public_artifacts", "local_video"}
    media.require(set(summary) == summary_keys and summary.get("schema_version") == "g009.r0.rev23.matrix_observation_adapter_visual_summary.v1" and summary.get("sequence_number") == sequence and summary.get("phase") == phase and summary.get("status") == "diagnostic_complete", "summary identity/schema mismatch")
    media.require(summary.get("labels") == [sequence, media.HEADER] and summary.get("claim_limits") == media.CLAIM_LIMITS and summary.get("governance") == media.GOVERNANCE, "summary claims/governance mismatch")
    media.require(summary.get("input_bindings") == canonical["input_bindings"] and summary.get("source") == sidecar.get("source"), "summary binding mismatch")
    media.require(summary.get("telemetry") == {"reports": canonical["reports"], "repeatability": canonical["repeatability"]}, "summary telemetry mismatch")
    media.require(summary.get("decision") == {"outcome": canonical["decision"]}, "summary decision mismatch")
    media.require(summary.get("public_artifacts") == public and summary.get("local_video") == artifacts.get("local_video"), "summary artifact metadata mismatch")

    contract = cast(Mapping[str, Any], sidecar.get("contract", {})); media.require(set(contract) == {"builder", "validator"}, "tool contract set mismatch")
    for key, path in (("builder", media.BUILDER_SOURCE), ("validator", Path(__file__))):
        record = cast(Mapping[str, Any], contract.get(key, {})); media.require(record.get("path") == media.repo_path(path) and record.get("sha256") == media.file_sha256(path), f"{key} source binding mismatch")
    command = cast(Mapping[str, Any], contract["validator"]).get("command")
    media.require(command == f"%PYTHON% scripts/validate_g009_r0_rev23_matrix_observation_adapter_media.py --phase {phase} --check-only", "validator command mismatch")
    return {"schema_version": "g009.r0.rev23.matrix_observation_adapter_media_validation.v1", "status": "pass", "phase": phase, "sequence_number": sequence, "checked": ["magic", "codec", "frames", "dimensions", "duration", "artifact_hashes", "labels", "ledger", "input_bindings", "source_bundle", "claim_limits", "governance", "no_overwrite"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--phase", required=True, choices=("cpu", "final")); parser.add_argument("--sidecar", type=Path); parser.add_argument("--check-only", action="store_true", required=True); return parser


def main() -> int:
    args = build_parser().parse_args()
    try: result = validate_bundle(args.phase, args.sidecar)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, KeyError, IndexError, subprocess.CalledProcessError) as exc: result = {"schema_version": "g009.r0.rev23.matrix_observation_adapter_media_validation.v1", "status": "fail", "phase": args.phase, "errors": [str(exc)]}
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if result["status"] == "pass" else 1


if __name__ == "__main__": raise SystemExit(main())
