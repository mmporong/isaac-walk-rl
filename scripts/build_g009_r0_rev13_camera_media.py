#!/usr/bin/env python3
"""Build numbered public GIF/PNG evidence from the rev13 Isaac Sim camera MP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_STEM = "g009_5_r0_diag_rev13_04_right_side_runtime"
DEFAULT_CAPTURE = REPO_ROOT / "reports/runs" / f"{OUTPUT_STEM}_capture_s42.json"
DEFAULT_GIF = REPO_ROOT / "docs/media/g009/R0/diagnostic" / f"{OUTPUT_STEM}.gif"
DEFAULT_PNG = REPO_ROOT / "docs/media/g009/R0/diagnostic" / f"{OUTPUT_STEM}.png"
DEFAULT_VISUAL = REPO_ROOT / "reports/runs" / f"{OUTPUT_STEM}_visual_evidence.json"
EXPECTED_LOCAL = PureWindowsPath(
    "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g009\\R0\\diagnostic\\"
    f"{OUTPUT_STEM}_s42.mp4"
)
REQUIRED_LABELS = ("DIAGNOSTIC", "NOT QUALIFIED", "NO PPO", "RIGHT_SIDE", "RESET_POSE_HOLD", "REV13 REJECTED")
MAX_PUBLIC_BYTES = 10 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("capture JSON root must be an object")
    return value


def resolve_portable(value: str) -> Path:
    prefix = "%USERPROFILE%\\"
    return Path.home() / value.removeprefix(prefix) if value.startswith(prefix) else (REPO_ROOT / value)


def repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")


def ffprobe_summary(path: Path, executable: str, *, require_timing: bool = True) -> dict[str, Any]:
    value = json.loads(subprocess.run(
        [executable, "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout)
    video = next(stream for stream in value["streams"] if stream.get("codec_type") == "video")
    summary = {"codec": video.get("codec_name"), "width": int(video["width"]), "height": int(video["height"])}
    if not require_timing:
        return summary
    duration = value.get("format", {}).get("duration") or video.get("duration")
    frame_count = video.get("nb_read_frames") or video.get("nb_frames")
    if duration is None:
        raise ValueError("timed media duration is unavailable")
    if frame_count is None or int(frame_count) <= 0:
        raise ValueError("timed media frame count is unavailable")
    return {
        **summary,
        "frame_rate": video.get("avg_frame_rate"),
        "frames": int(frame_count),
        "duration_s": float(duration),
    }


def validate_capture(path: Path) -> tuple[dict[str, Any], Path]:
    require(path.resolve() == DEFAULT_CAPTURE.resolve(), "capture sidecar path is fixed")
    capture = read_json(path)
    require(capture.get("camera_footage") is True and capture.get("telemetry_animation") is False, "actual camera footage is required")
    require(capture.get("headless") is True and capture.get("offscreen") is True, "headless/offscreen binding mismatch")
    require(capture.get("diagnostic_only") is True and capture.get("qualification_status") == "not_run", "diagnostic status mismatch")
    require(capture.get("learned") is False and capture.get("ppo_checkpoint_used") is False, "NO PPO binding mismatch")
    require(tuple(capture.get("labels", ())) == REQUIRED_LABELS, "required public labels mismatch")
    require(capture.get("pose_id") == "right_side" and capture.get("action_mode") == "reset_pose_hold", "failure cell mismatch")
    video = capture.get("local_video", {})
    require(PureWindowsPath(video.get("path", "")) == EXPECTED_LOCAL, "local MP4 path mismatch")
    source = resolve_portable(video["path"]).resolve()
    require(source.is_file(), "local-only camera MP4 is missing")
    require(file_sha256(source) == video.get("sha256") and source.stat().st_size == video.get("bytes"), "local MP4 integrity mismatch")
    return capture, source


def artifact(path: Path, evidence_type: str, ffprobe: str) -> dict[str, Any]:
    probe = ffprobe_summary(path, ffprobe, require_timing=evidence_type != "camera_footage_still")
    return {"path": repo_path(path), "sha256": file_sha256(path), "bytes": path.stat().st_size, "git_policy": "git_public", "evidence_type": evidence_type, **probe}


def build(args: argparse.Namespace) -> dict[str, Any]:
    require(args.gif.resolve() == DEFAULT_GIF.resolve(), "GIF path is fixed")
    require(args.png.resolve() == DEFAULT_PNG.resolve(), "PNG path is fixed")
    require(args.visual.resolve() == DEFAULT_VISUAL.resolve(), "visual sidecar path is fixed")
    require(not any(path.exists() for path in (args.gif, args.png, args.visual)), "public evidence refuses overwrite")
    capture, source = validate_capture(args.capture)
    source_probe = ffprobe_summary(source, args.ffprobe)
    require(source_probe["codec"] == "h264" and source_probe["width"] == 1280 and source_probe["height"] == 720, "source must be H264 1280x720")
    args.gif.parent.mkdir(parents=True, exist_ok=True)
    args.visual.parent.mkdir(parents=True, exist_ok=True)
    font = str(args.font).replace("\\", "/").replace(":", "\\:")
    top = "DIAGNOSTIC | NOT QUALIFIED | NO PPO"
    bottom = "RIGHT_SIDE | RESET_POSE_HOLD | REV13 REJECTED"
    overlay = (
        f"drawbox=x=0:y=0:w=iw:h=58:color=black@0.72:t=fill,"
        f"drawtext=fontfile='{font}':text='{top}':fontcolor=white:fontsize=26:x=(w-text_w)/2:y=14,"
        f"drawbox=x=0:y=h-58:w=iw:h=58:color=black@0.72:t=fill,"
        f"drawtext=fontfile='{font}':text='{bottom}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=h-44"
    )
    with tempfile.TemporaryDirectory(prefix="g009-rev13-camera-") as directory:
        temp = Path(directory)
        staged_png = temp / "still.png"
        staged_gif = temp / "camera.gif"
        subprocess.run([args.ffmpeg, "-y", "-ss", "0.70", "-i", str(source), "-frames:v", "1", "-vf", overlay, str(staged_png)], check=True)
        gif_filter = f"{overlay},fps=10,scale=960:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
        subprocess.run([args.ffmpeg, "-y", "-t", "3.0", "-i", str(source), "-filter_complex", gif_filter, "-loop", "0", str(staged_gif)], check=True)
        require(staged_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "PNG signature mismatch")
        require(staged_gif.read_bytes()[:6] in {b"GIF87a", b"GIF89a"}, "GIF signature mismatch")
        require(staged_png.stat().st_size <= MAX_PUBLIC_BYTES and staged_gif.stat().st_size <= MAX_PUBLIC_BYTES, "public media exceeds 10 MiB")
        staged_png.replace(args.png)
        staged_gif.replace(args.gif)
    visual = {
        "schema_version": "g009.r0.rev13.camera_visual_evidence.v1",
        "status": "diagnostic_complete", "diagnostic_only": True,
        "camera_footage": True, "telemetry_animation": False,
        "qualification_status": "not_run", "learned": False,
        "labels": list(REQUIRED_LABELS),
        "pose_id": "right_side", "action_mode": "reset_pose_hold",
        "source_capture": {"path": repo_path(args.capture), "sha256": file_sha256(args.capture)},
        "source": capture["source"], "solver_live_readback": capture["solver_live_readback"],
        "original_rev13_report_binding": capture["original_rev13_report_binding"],
        "headless": True, "offscreen": True,
        "evidence_scope": capture["evidence_scope"],
        "blocked_stages": capture["blocked_stages"],
        "local_video": {**capture["local_video"], **source_probe},
        "public_artifacts": {"gif": artifact(args.gif, "camera_footage_gif", args.ffprobe), "png": artifact(args.png, "camera_footage_still", args.ffprobe)},
    }
    args.visual.write_text(json.dumps(visual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return visual


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--visual", type=Path, default=DEFAULT_VISUAL)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/arialbd.ttf"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build(args)
    print(json.dumps({"gif": result["public_artifacts"]["gif"], "png": result["public_artifacts"]["png"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
