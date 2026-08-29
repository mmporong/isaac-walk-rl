#!/usr/bin/env python3
"""Build numbered public GIF/PNG evidence from the local rev16 camera MP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "9ac874f48a1403e0ed838beb5e75938db5873d1c"
OUTPUT_STEM = "g009_5_r0_diag_rev16_08_b_gpu_right_side_force_repro"
DEFAULT_CAPTURE = REPO_ROOT / "reports/runs" / f"{OUTPUT_STEM}_capture_s42.json"
DEFAULT_LOCAL_VIDEO = (
    Path.home()
    / "IsaacLab/logs/visual_evidence/g009/R0/diagnostic"
    / f"{OUTPUT_STEM}_s42.mp4"
)
DEFAULT_PNG = REPO_ROOT / "docs/media/g009/R0/diagnostic" / f"{OUTPUT_STEM}.png"
DEFAULT_GIF = REPO_ROOT / "docs/media/g009/R0/diagnostic" / f"{OUTPUT_STEM}.gif"
DEFAULT_SUMMARY = REPO_ROOT / "reports/runs" / f"{OUTPUT_STEM}_visual_evidence.json"
OVERLAY_TOP = "G009-5 | REV16 | DIAGNOSTIC | REJECTED | NO PPO | NOT QUALIFIED"
OVERLAY_BOTTOM = "08 ARM B CUDA | RIGHT_SIDE | RESET_POSE_HOLD | 16.788 BW > 15 BW"
REQUIRED_LABELS = (
    "DIAGNOSTIC",
    "REJECTED",
    "NO PPO",
    "NOT QUALIFIED",
    "CAMERA FOOTAGE",
)
MAX_PUBLIC_BYTES = 10 * 1024 * 1024
MAX_GIF_BYTES = 6 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")


def portable_path(path: Path) -> str:
    return "%USERPROFILE%\\" + str(path.resolve().relative_to(Path.home().resolve()))


def read_capture(path: Path) -> dict[str, Any]:
    require(path.resolve() == DEFAULT_CAPTURE.resolve(), "capture path is fixed")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(
        value.get("schema_version") == "g009.r0.rev16.camera_capture.v1"
        and value.get("stage_number") == "08"
        and value.get("camera_footage") is True
        and value.get("telemetry_animation") is False,
        "camera capture identity mismatch",
    )
    require(
        value.get("headless") is True and value.get("offscreen") is True,
        "off-screen capture required",
    )
    require(
        value.get("source", {}).get("commit") == SOURCE_COMMIT, "source commit mismatch"
    )
    capture_source = value.get("capture_source", {})
    capture_bundle = capture_source.get("source_bundle", {})
    capture_commit = capture_source.get("capture_commit")
    require(
        isinstance(capture_commit, str)
        and len(capture_commit) == 40
        and capture_bundle.get("git_commit") == capture_commit
        and capture_bundle.get("runtime_source_commit") == SOURCE_COMMIT,
        "camera capture source commit binding mismatch",
    )
    require(
        capture_bundle.get("all_files_present") is True
        and capture_bundle.get("clean") is True
        and capture_bundle.get("dirty_paths") == []
        and isinstance(capture_bundle.get("source_bundle_sha256"), str)
        and len(capture_bundle["source_bundle_sha256"]) == 64,
        "camera capture source bundle is not clean and complete",
    )
    source_files = capture_bundle.get("source_binding_files", {})
    require(
        "scripts/record_g009_r0_rev16_b_gpu_right_side.py" in source_files
        and "reports/runs/g009_r0_rev16_arm_b_gpu_rep01_retry01_s42.json"
        in source_files,
        "camera capture source files are incomplete",
    )
    require(
        value.get("labels")
        == [
            "DIAGNOSTIC",
            "REJECTED",
            "NO PPO",
            "NOT QUALIFIED",
            "RIGHT_SIDE",
            "RESET_POSE_HOLD",
        ],
        "governance labels mismatch",
    )
    governance = value.get("governance", {})
    require(
        governance.get("ppo", {}).get("status") == "not_run"
        and governance.get("gate01", {}).get("status") == "forbidden"
        and governance.get("gate10", {}).get("status") == "forbidden"
        and governance.get("qualification", {}).get("status") == "not_run",
        "governance mismatch",
    )
    return value


def publish_new(staged: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    with final.open("xb") as output, staged.open("rb") as source:
        while block := source.read(1024 * 1024):
            output.write(block)
        output.flush()
        os.fsync(output.fileno())


def probe(path: Path, ffprobe: str) -> dict[str, Any]:
    data = json.loads(
        subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    stream = next(row for row in data["streams"] if row.get("codec_type") == "video")
    duration = data.get("format", {}).get("duration") or stream.get("duration")
    result = {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream.get("nb_frames") or 0),
    }
    if duration is not None:
        result["duration_s"] = float(duration)
    return result


def validate_media(path: Path, signature: bytes, gif: bool = False) -> None:
    require(
        path.is_file() and path.read_bytes().startswith(signature),
        f"invalid media: {path}",
    )
    require(path.stat().st_size < MAX_PUBLIC_BYTES, "public artifact exceeds 10 MiB")
    if gif:
        require(
            path.stat().st_size < MAX_GIF_BYTES, "GIF exceeds preferred 6 MiB limit"
        )


def build(
    capture_path: Path,
    video: Path,
    png: Path,
    gif: Path,
    summary: Path,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, Any]:
    require(
        capture_path.resolve() == DEFAULT_CAPTURE.resolve()
        and video.resolve() == DEFAULT_LOCAL_VIDEO.resolve()
        and png.resolve() == DEFAULT_PNG.resolve()
        and gif.resolve() == DEFAULT_GIF.resolve()
        and summary.resolve() == DEFAULT_SUMMARY.resolve(),
        "numbered input/output paths are fixed",
    )
    for path in (png, gif, summary):
        require(not path.exists(), f"refusing to overwrite output: {path}")
    require(video.is_file(), "local-only MP4 is missing")
    capture = read_capture(capture_path)
    local_probe = probe(video, ffprobe)
    require(
        local_probe["codec"] == "h264"
        and local_probe["width"] == 1280
        and local_probe["height"] == 720,
        "camera MP4 must be H264 1280x720",
    )
    draw = "drawbox=x=0:y=0:w=iw:h=74:color=black@0.72:t=fill,drawbox=x=0:y=h-62:w=iw:h=62:color=black@0.72:t=fill,drawtext=text='G009-5 | REV16 | DIAGNOSTIC | REJECTED | NO PPO | NOT QUALIFIED':x=(w-text_w)/2:y=20:fontsize=27:fontcolor=white,drawtext=text='08 ARM B CUDA | RIGHT_SIDE | RESET_POSE_HOLD | 16.788 BW > 15 BW':x=(w-text_w)/2:y=h-45:fontsize=24:fontcolor=yellow"
    with tempfile.TemporaryDirectory(prefix="g009-rev16-camera-") as directory:
        temp = Path(directory)
        staged_png, staged_gif = temp / "camera.png", temp / "camera.gif"
        subprocess.run(
            [
                ffmpeg,
                "-n",
                "-ss",
                "1.5",
                "-i",
                str(video),
                "-vf",
                draw,
                "-frames:v",
                "1",
                str(staged_png),
            ],
            check=True,
        )
        gif_filter = f"{draw},fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer"
        subprocess.run(
            [
                ffmpeg,
                "-n",
                "-i",
                str(video),
                "-filter_complex",
                gif_filter,
                "-loop",
                "0",
                str(staged_gif),
            ],
            check=True,
        )
        validate_media(staged_png, b"\x89PNG\r\n\x1a\n")
        validate_media(staged_gif, b"GIF8", gif=True)
        png_probe, gif_probe = probe(staged_png, ffprobe), probe(staged_gif, ffprobe)
        value = {
            "schema_version": "g009.r0.rev16.camera_visual_evidence.v1",
            "goal_id": "g009",
            "stage_id": "R0",
            "stage_number": "08",
            "revision": "rev16",
            "status": "rejected",
            "diagnostic_only": True,
            "camera_footage": True,
            "telemetry_animation": False,
            "headless": True,
            "offscreen": True,
            "labels": list(REQUIRED_LABELS),
            "overlay_labels": {"top": OVERLAY_TOP, "bottom": OVERLAY_BOTTOM},
            "source_commit": SOURCE_COMMIT,
            "source_capture": {
                "path": repo_path(capture_path),
                "sha256": file_sha256(capture_path),
            },
            "input_report": capture["source"],
            "capture_source": capture["capture_source"],
            "governance": capture["governance"],
            "local_video": {
                "path": portable_path(video),
                "sha256": file_sha256(video),
                "bytes": video.stat().st_size,
                "git_policy": "local_only",
                **local_probe,
            },
            "public_artifacts": {
                "png": {
                    "path": repo_path(png),
                    "sha256": file_sha256(staged_png),
                    "bytes": staged_png.stat().st_size,
                    "git_policy": "git_public",
                    **png_probe,
                },
                "gif": {
                    "path": repo_path(gif),
                    "sha256": file_sha256(staged_gif),
                    "bytes": staged_gif.stat().st_size,
                    "git_policy": "git_public",
                    **gif_probe,
                },
            },
        }
        staged_summary = temp / "summary.json"
        staged_summary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        published: list[Path] = []
        try:
            for source, destination in (
                (staged_png, png),
                (staged_gif, gif),
                (staged_summary, summary),
            ):
                publish_new(source, destination)
                published.append(destination)
        except Exception:
            for path in published:
                path.unlink(missing_ok=True)
            raise
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--video", type=Path, default=DEFAULT_LOCAL_VIDEO)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.capture,
                args.video,
                args.png,
                args.gif,
                args.summary,
                args.ffmpeg,
                args.ffprobe,
            ),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
