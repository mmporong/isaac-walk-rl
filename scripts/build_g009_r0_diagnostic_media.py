#!/usr/bin/env python3
"""Build clearly labeled public media from the G009 R0 rev9 diagnostic capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LOCAL_NAME = "g009_5_r0_diag_rev9_01_prone_s42.mp4"
EXPECTED_LOCAL_PARENT = PureWindowsPath("%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\R0\\diagnostic")
DEFAULT_CAPTURE = REPO_ROOT / "reports/runs/g009_r0_diag_rev9_01_prone_capture_s42.json"
DEFAULT_ANALYSIS = REPO_ROOT / "reports/runs/g009_r0_flat_diagnostic_rev9_prone_pilot_analysis.json"
DEFAULT_GIF = REPO_ROOT / "docs/media/g009/R0/diagnostic/g009_5_r0_diag_rev9_01_prone.gif"
DEFAULT_PNG = REPO_ROOT / "docs/media/g009/R0/diagnostic/g009_5_r0_diag_rev9_01_prone_still.png"
DEFAULT_SUMMARY = REPO_ROOT / "reports/runs/g009_r0_diag_rev9_01_prone_visual_summary.json"
DEFAULT_SIDECAR = REPO_ROOT / "reports/runs/g009_r0_diag_rev9_01_prone_visual_evidence.json"
DEFAULT_FONT = Path("C:/Windows/Fonts/arialbd.ttf")
EXPECTED_RECORD_SOURCE = "scripts/record_g009_r0_diagnostic.py"
EXPECTED_ANALYSIS_SOURCE = "scripts/analyze_g009_r0_pilot.py"
EXPECTED_CONFIG_SOURCE = "configs/g009_r0.json"
MAX_PUBLIC_MEDIA_BYTES = 10 * 1024 * 1024
GIF_SIGNATURES = (b"GIF87a", b"GIF89a")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def resolve_portable_path(value: str) -> Path:
    prefix = "%USERPROFILE%\\"
    if value.startswith(prefix):
        return Path.home() / value.removeprefix(prefix)
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def portable_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError as exc:
        raise ValueError(f"public artifact must be inside repository: {path}") from exc


def validate_fixed_paths(args: argparse.Namespace) -> None:
    expected = {
        "capture_report": DEFAULT_CAPTURE,
        "analysis_report": DEFAULT_ANALYSIS,
        "gif": DEFAULT_GIF,
        "png": DEFAULT_PNG,
        "summary": DEFAULT_SUMMARY,
        "sidecar": DEFAULT_SIDECAR,
    }
    mismatches = [name for name, path in expected.items() if Path(getattr(args, name)).resolve() != path.resolve()]
    _require(not mismatches, "diagnostic media paths are fixed: " + ", ".join(mismatches))


def validate_capture(capture_path: Path, analysis_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    capture = _read_json(capture_path)
    analysis = _read_json(analysis_path)
    _require(capture.get("goal_id") == "g009", "capture goal_id mismatch")
    _require(capture.get("stage_number") == "G009-5" and capture.get("stage_id") == "R0", "capture stage mismatch")
    _require(capture.get("status") == "diagnostic_complete", "capture must be diagnostic_complete")
    _require(capture.get("diagnostic_only") is True, "capture must be diagnostic_only")
    _require(capture.get("public_claim_eligible") is False, "diagnostic capture cannot be claim-eligible")
    _require(capture.get("qualification_status") == "not_run", "diagnostic qualification must be not_run")
    _require(
        capture.get("policy_result") in {"failure", "single_playback_success"},
        "diagnostic policy_result mismatch",
    )
    _require(
        type(capture.get("strict_success")) is int and capture["strict_success"] in {0, 1},
        "diagnostic strict_success must be integer 0 or 1",
    )
    _require(capture.get("task") == "Isaac-G009-Recover-Flat-Go2-R0-v0", "capture task mismatch")
    _require(capture.get("seed") == 42, "capture seed mismatch")
    _require(capture.get("headless") is True and capture.get("offscreen") is True, "headless off-screen capture is required")
    pose = capture.get("pose", {})
    _require(pose.get("index") == 1 and pose.get("pose_id") == "prone" and pose.get("source_class_id") == 0, "capture must be numbered 01 prone")
    playback_success = capture.get("metrics", {}).get("stable_success")
    _require(type(playback_success) is bool, "diagnostic stable_success must be boolean")
    _require(int(playback_success) == capture["strict_success"], "diagnostic strict_success/playback mismatch")
    expected_result = "single_playback_success" if playback_success else "failure"
    _require(capture["policy_result"] == expected_result, "diagnostic policy_result/playback mismatch")
    source_state = capture.get("source_state", {})
    _require(source_state.get("before") == source_state.get("after"), "source state changed during capture")
    _require(source_state.get("after", {}).get("clean") is True, "capture source state was dirty")

    video = capture.get("local_video", {})
    video_path = video.get("path")
    _require(isinstance(video_path, str), "capture local_video.path is required")
    portable = PureWindowsPath(video_path)
    _require(portable.parent == EXPECTED_LOCAL_PARENT, "diagnostic MP4 must remain in the local-only R0 directory")
    _require(portable.name == EXPECTED_LOCAL_NAME, "diagnostic MP4 numbered filename mismatch")
    _require(video.get("git_policy") == "local_only", "diagnostic MP4 must be local_only")
    source = resolve_portable_path(video_path).resolve()
    _require(source.is_file() and source.stat().st_size > 0, f"diagnostic MP4 is missing: {source}")
    _require(file_sha256(source) == video.get("sha256"), "diagnostic MP4 hash mismatch")
    _require(source.stat().st_size == video.get("bytes"), "diagnostic MP4 byte count mismatch")

    checkpoint = capture.get("checkpoint", {})
    checkpoint_path = resolve_portable_path(checkpoint.get("path", "")).resolve()
    _require(checkpoint_path.is_file(), "capture checkpoint is missing")
    _require(file_sha256(checkpoint_path) == checkpoint.get("sha256"), "capture checkpoint hash mismatch")
    training = capture.get("training_binding", {})
    training_path = resolve_portable_path(training.get("path", "")).resolve()
    _require(training_path.is_file(), "bound training report is missing")
    _require(file_sha256(training_path) == training.get("sha256"), "bound training report hash mismatch")
    _require(training.get("checkpoint_sha256") == checkpoint.get("sha256"), "training/checkpoint binding mismatch")

    source_bindings = capture.get("source_bindings", {})
    expected_sources = {
        "record_source": EXPECTED_RECORD_SOURCE,
        "config": EXPECTED_CONFIG_SOURCE,
    }
    for binding_name, expected_path in expected_sources.items():
        binding = source_bindings.get(binding_name, {})
        _require(binding.get("path") == expected_path, f"capture {binding_name} path mismatch")
        source_path = (REPO_ROOT / expected_path).resolve()
        _require(source_path.is_file(), f"capture {binding_name} source is missing")
        _require(file_sha256(source_path) == binding.get("sha256"), f"capture {binding_name} hash mismatch")

    _require(analysis.get("schema_version") == "g009.r0.pilot_analysis.v2", "analysis schema mismatch")
    _require(analysis.get("status") == "diagnostic_complete", "analysis must be diagnostic_complete")
    _require(analysis.get("diagnostic_only") is True, "analysis must be diagnostic_only")
    _require(analysis.get("qualification_allowed") is False, "analysis cannot allow qualification")
    _require(analysis.get("public_claim_eligible") is False, "analysis cannot be claim-eligible")
    _require(analysis.get("checkpoint", {}).get("sha256") == checkpoint.get("sha256"), "analysis/capture checkpoint mismatch")
    _require(analysis.get("training_report", {}).get("sha256") == training.get("sha256"), "analysis/capture training report mismatch")
    analysis_source = analysis.get("analysis_source", {})
    _require(analysis_source.get("path") == EXPECTED_ANALYSIS_SOURCE, "analysis source path mismatch")
    analysis_source_path = (REPO_ROOT / EXPECTED_ANALYSIS_SOURCE).resolve()
    _require(file_sha256(analysis_source_path) == analysis_source.get("sha256"), "analysis source hash mismatch")
    tensorboard = analysis.get("tensorboard", {})
    tensorboard_path = resolve_portable_path(tensorboard.get("path", "")).resolve()
    _require(tensorboard_path.is_dir(), "analysis TensorBoard directory is missing")
    event_files = tensorboard.get("event_files")
    _require(isinstance(event_files, Mapping) and bool(event_files), "analysis TensorBoard event files are missing")
    actual_event_files = {
        path.name: file_sha256(path)
        for path in sorted(tensorboard_path.glob("events.out.tfevents.*"), key=lambda item: item.name)
    }
    _require(actual_event_files == dict(event_files), "analysis TensorBoard event file hash mismatch")
    event_payload = "\n".join(f"{name}:{digest}" for name, digest in actual_event_files.items())
    event_bundle_sha = hashlib.sha256(event_payload.encode("utf-8")).hexdigest()
    _require(event_bundle_sha == tensorboard.get("event_bundle_sha256"), "analysis TensorBoard event bundle mismatch")
    reasons = set(analysis.get("qualification_block_reasons", []))
    _require(
        {"diagnostic_pilot_never_qualifies", "hard_joint_limit_nonzero", "strict_success_zero", "prone_curriculum_boundary_leak"}.issubset(reasons),
        "analysis is missing rev9 qualification blockers",
    )
    return capture, analysis, source


def _run(command: Sequence[str]) -> str:
    result = subprocess.run(list(command), check=True, capture_output=True, text=True, encoding="utf-8")
    return result.stdout


def _ffprobe(path: Path, executable: str) -> dict[str, Any]:
    return json.loads(_run([executable, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))


def _stream_summary(probe: Mapping[str, Any]) -> dict[str, Any]:
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    duration = probe.get("format", {}).get("duration") or video.get("duration")
    return {
        "codec": video["codec_name"],
        "width": int(video["width"]),
        "height": int(video["height"]),
        "duration_s": None if duration is None else float(duration),
        "frame_rate": video.get("avg_frame_rate"),
    }


def _font_expression(font: Path) -> str:
    return "fontfile='" + font.resolve().as_posix().replace(":", r"\:") + "'"


def _overlay_filter(font: Path, strict_success: int) -> str:
    font_expr = _font_expression(font)
    if strict_success:
        headline = "DIAGNOSTIC · NOT QUALIFIED · 01 PRONE · SINGLE PLAYBACK SUCCESS"
        footer = "REV9 PILOT - QUALIFICATION NOT RUN - STRICT SUCCESS 1"
    else:
        headline = "DIAGNOSTIC · NOT QUALIFIED · 01 PRONE · STRICT SUCCESS 0"
        footer = "REV9 PILOT - HARD LIMIT EVENTS - STRICT SUCCESS 0"
    return (
        "scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        "drawbox=x=0:y=0:w=iw:h=84:color=0x8B0000@0.92:t=fill,"
        f"drawtext={font_expr}:text='{headline}':"
        "x=(w-text_w)/2:y=20:fontsize=34:fontcolor=white:borderw=2:bordercolor=black,"
        "drawbox=x=0:y=h-48:w=iw:h=48:color=black@0.78:t=fill,"
        f"drawtext={font_expr}:text='{footer}':"
        "x=(w-text_w)/2:y=h-37:fontsize=22:fontcolor=yellow"
    )


def _build_gif(source: Path, destination: Path, ffmpeg: str, font: Path, strict_success: int) -> None:
    graph = (
        _overlay_filter(font, strict_success)
        + ",fps=6,scale=720:-2:flags=lanczos,split[base][palette_source];"
        "[palette_source]palettegen=max_colors=80:stats_mode=diff[palette];"
        "[base][palette]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )
    _run([ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(source), "-filter_complex", graph, "-loop", "0", str(destination)])


def _build_contact_sheet(
    source: Path,
    destination: Path,
    ffmpeg: str,
    font: Path,
    duration_s: float,
    strict_success: int,
) -> list[float]:
    timestamps = [duration_s * fraction for fraction in (0.1, 0.4, 0.7, 0.9)]
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
    for timestamp in timestamps:
        command.extend(["-ss", f"{timestamp:.3f}", "-i", str(source)])
    filters = []
    labels = []
    for index in range(4):
        filters.append(f"[{index}:v]{_overlay_filter(font, strict_success)},scale=640:360[p{index}]")
        labels.append(f"[p{index}]")
    filters.append("".join(labels) + "xstack=inputs=4:layout=0_0|640_0|0_360|640_360[out]")
    command.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-frames:v", "1", str(destination)])
    _run(command)
    return timestamps


def _validate_media(path: Path, kind: str) -> None:
    _require(path.is_file() and path.stat().st_size > 0, f"missing or empty {kind}: {path}")
    signature = path.read_bytes()[:8]
    if kind == "gif":
        _require(signature.startswith(GIF_SIGNATURES), "invalid GIF signature")
    elif kind == "png":
        _require(signature == PNG_SIGNATURE, "invalid PNG signature")
    _require(path.stat().st_size < MAX_PUBLIC_MEDIA_BYTES, f"{kind} must be below 10 MiB")


def _artifact(path: Path, evidence_type: str, git_policy: str, ffprobe: str) -> dict[str, Any]:
    return {
        "evidence_type": evidence_type,
        "path": portable_repo_path(path) if git_policy == "public_git" else "%USERPROFILE%\\" + str(path.resolve().relative_to(Path.home().resolve())),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "git_policy": git_policy,
        **_stream_summary(_ffprobe(path, ffprobe)),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _publish_transaction(pairs: Sequence[tuple[Path, Path]], validator: Any) -> None:
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for staged, final in pairs:
            _require(staged.is_file(), f"staged artifact is missing: {staged}")
            final.parent.mkdir(parents=True, exist_ok=True)
            backup = final.with_name(final.name + f".{uuid.uuid4().hex}.bak")
            if final.exists():
                os.replace(final, backup)
                backups.append((final, backup))
            os.replace(staged, final)
            installed.append(final)
        validator()
    except Exception:
        for final in reversed(installed):
            final.unlink(missing_ok=True)
        for final, backup in backups:
            if backup.exists():
                os.replace(backup, final)
        raise
    else:
        for _, backup in backups:
            backup.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    validate_fixed_paths(args)
    capture_path = args.capture_report.resolve()
    analysis_path = args.analysis_report.resolve()
    font = args.font.resolve()
    for path in (capture_path, analysis_path, font):
        if not path.is_file():
            raise FileNotFoundError(path)
    capture, analysis, source = validate_capture(capture_path, analysis_path)
    probe = _stream_summary(_ffprobe(source, args.ffprobe))
    duration_s = float(probe["duration_s"] or 0.02)
    final_gif, final_png = args.gif.resolve(), args.png.resolve()
    final_summary, final_sidecar = args.summary.resolve(), args.sidecar.resolve()
    for path in (final_gif, final_png, final_summary, final_sidecar):
        portable_repo_path(path)
    with tempfile.TemporaryDirectory(prefix="g009-r0-diagnostic-media-") as temporary:
        temporary_root = Path(temporary)
        staged_gif = temporary_root / final_gif.name
        staged_png = temporary_root / final_png.name
        staged_summary = temporary_root / final_summary.name
        staged_sidecar = temporary_root / final_sidecar.name
        strict_success = capture["strict_success"]
        _build_gif(source, staged_gif, args.ffmpeg, font, strict_success)
        timestamps = _build_contact_sheet(source, staged_png, args.ffmpeg, font, duration_s, strict_success)
        _validate_media(staged_gif, "gif")
        _validate_media(staged_png, "png")
        capture_binding = {"path": portable_repo_path(capture_path), "sha256": file_sha256(capture_path)}
        analysis_binding = {"path": portable_repo_path(analysis_path), "sha256": file_sha256(analysis_path)}
        local_video = {**capture["local_video"], **probe}
        gif = {
            "evidence_type": "diagnostic_animation",
            "path": portable_repo_path(final_gif),
            "sha256": file_sha256(staged_gif),
            "bytes": staged_gif.stat().st_size,
            "git_policy": "public_git",
            **_stream_summary(_ffprobe(staged_gif, args.ffprobe)),
        }
        png = {
            "evidence_type": "diagnostic_contact_sheet",
            "path": portable_repo_path(final_png),
            "sha256": file_sha256(staged_png),
            "bytes": staged_png.stat().st_size,
            "git_policy": "public_git",
            **_stream_summary(_ffprobe(staged_png, args.ffprobe)),
        }
        common = {
            "schema_version": "g009.r0.diagnostic_visual.v1",
            "goal_id": "g009",
            "stage_number": "G009-5",
            "stage_id": "R0",
            "status": "diagnostic_complete",
            "diagnostic_only": True,
            "public_claim_eligible": False,
            "qualification_status": "not_run",
            "policy_result": capture["policy_result"],
            "strict_success": capture["strict_success"],
            "warning_label": (
                "DIAGNOSTIC · NOT QUALIFIED · 01 PRONE · SINGLE PLAYBACK SUCCESS"
                if strict_success
                else "DIAGNOSTIC · NOT QUALIFIED · 01 PRONE · STRICT SUCCESS 0"
            ),
            "qualification_block_reasons": analysis["qualification_block_reasons"],
            "capture_report": capture_binding,
            "analysis_report": analysis_binding,
            "checkpoint": capture["checkpoint"],
            "training_binding": capture["training_binding"],
            "capture_commit": capture["capture_commit"],
            "source_bindings": {
                **capture["source_bindings"],
                "analysis_source": analysis["analysis_source"],
                "media_builder": {
                    "path": "scripts/build_g009_r0_diagnostic_media.py",
                    "sha256": file_sha256(Path(__file__)),
                },
            },
        }
        summary = {
            **common,
            "report_type": "diagnostic_visual_summary",
            "pose": capture["pose"],
            "metrics": capture["metrics"],
            "contact_sheet_timestamps_s": timestamps,
            "artifacts": {"local_mp4": local_video, "public_gif": gif, "public_png": png},
        }
        sidecar = {
            **common,
            "schema_version": "g009.r0.diagnostic_visual_evidence.v1",
            "report_type": "diagnostic_visual_evidence_sidecar",
            "official_sidecar_schema": False,
            "official_gate_evaluated": False,
            "artifacts": {"public_gif": gif, "public_png": png},
        }
        _write_json(staged_summary, summary)
        _write_json(staged_sidecar, sidecar)

        def validate_installed() -> None:
            _validate_media(final_gif, "gif")
            _validate_media(final_png, "png")
            installed = _read_json(final_sidecar)
            _require(installed.get("official_sidecar_schema") is False, "diagnostic sidecar became official")
            _require(installed.get("public_claim_eligible") is False, "diagnostic sidecar became claim-eligible")

        _publish_transaction(
            ((staged_gif, final_gif), (staged_png, final_png), (staged_summary, final_summary), (staged_sidecar, final_sidecar)),
            validate_installed,
        )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-report", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--analysis-report", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    summary = build(parse_args(argv))
    print(json.dumps({"status": summary["status"], "artifacts": summary["artifacts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
