#!/usr/bin/env python3
"""Build a synchronized G008 road/reward comparison GIF and contact sheet."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from build_g008_comparison_media import (
    _artifact_metadata,
    _drawtext,
    _font_expression,
    file_sha256,
    portable_path,
    resolve_portable,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTACT_TIMES_S = (3.0, 7.5, 12.0, 16.5)


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def _timeline_filters(font: str) -> list[str]:
    return [
        _drawtext(font, "STAND", size=20, y="748", enable=r"between(t\,0\,1)+between(t\,4.5\,5.5)+between(t\,9\,10)+between(t\,13.5\,14.5)"),
        _drawtext(font, "FORWARD  vx +0.60 m/s", size=20, y="748", enable=r"between(t\,1\,4.5)"),
        _drawtext(font, "BACKWARD  vx -0.40 m/s", size=20, y="748", enable=r"between(t\,5.5\,9)"),
        _drawtext(font, "LEFT TURN  wz +0.50 rad/s", size=20, y="748", enable=r"between(t\,10\,13.5)"),
        _drawtext(font, "RIGHT TURN  wz -0.50 rad/s", size=20, y="748", enable=r"between(t\,14.5\,18)"),
    ]


def _build_composite(sources: list[Path], labels: list[str], destination: Path, ffmpeg: str, font_path: Path) -> None:
    font = _font_expression(font_path)
    filters = [
        (
            "[0:v]crop=640:720:(iw-640)/2:0,drawbox=x=0:y=0:w=iw:h=48:color=black@0.72:t=fill,"
            f"{_drawtext(font, labels[0], size=20, y='12')}[left]"
        ),
        (
            "[1:v]crop=640:720:(iw-640)/2:0,drawbox=x=0:y=0:w=iw:h=48:color=black@0.72:t=fill,"
            f"{_drawtext(font, labels[1], size=20, y='12')}[right]"
        ),
        "[left][right]hstack=inputs=2[grid]",
        "[grid]pad=1280:780:0:0:black,drawbox=x=0:y=720:w=1280:h=60:color=black:t=fill,"
        + ",".join(_timeline_filters(font))
        + "[out]",
    ]
    _run([
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-i", str(sources[0]), "-i", str(sources[1]),
        "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ])


def _build_gif(source: Path, destination: Path, ffmpeg: str) -> None:
    graph = (
        "fps=4,scale=720:-2:flags=lanczos,split[base][palette_source];"
        "[palette_source]palettegen=max_colors=48:stats_mode=diff[palette];"
        "[base][palette]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )
    _run([ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(source), "-filter_complex", graph, "-loop", "0", str(destination)])


def _build_contact_sheet(source: Path, destination: Path, ffmpeg: str) -> None:
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
    for timestamp in CONTACT_TIMES_S:
        command.extend(["-ss", str(timestamp), "-i", str(source)])
    command.extend([
        "-filter_complex",
        "[0:v]scale=640:390:flags=lanczos[a];[1:v]scale=640:390:flags=lanczos[b];"
        "[2:v]scale=640:390:flags=lanczos[c];[3:v]scale=640:390:flags=lanczos[d];"
        "[a][b][c][d]xstack=inputs=4:layout=0_0|640_0|0_390|640_390[out]",
        "-map", "[out]", "-frames:v", "1", str(destination),
    ])
    _run(command)


def build(args: argparse.Namespace) -> dict[str, Any]:
    capture_paths = [path.resolve() for path in args.capture_reports]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in capture_paths]
    profiles = [report["profile"] for report in reports]
    if any(report.get("status") != "complete" or profile["stage"] != "road_reward_curriculum" for report, profile in zip(reports, profiles)):
        raise ValueError("capture report is not a completed road/reward profile")
    if len({json.dumps(profile["sequence"], sort_keys=True) for profile in profiles}) != 1:
        raise ValueError("capture command sequences differ")
    sources = [resolve_portable(profile["local_video"]["path"]).resolve() for profile in profiles]
    for source, profile in zip(sources, profiles):
        if not source.is_file() or file_sha256(source) != profile["local_video"]["sha256"]:
            raise RuntimeError(f"local video is missing or changed: {source}")
    outputs = (args.local_composite.resolve(), args.public_gif.resolve(), args.public_contact_sheet.resolve())
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not args.rebuild_existing:
            raise FileExistsError(output)
    _build_composite(sources, [profile["label"] for profile in profiles], outputs[0], args.ffmpeg, args.font.resolve())
    _build_gif(outputs[0], outputs[1], args.ffmpeg)
    _build_contact_sheet(outputs[0], outputs[2], args.ffmpeg)
    if outputs[1].stat().st_size >= 10 * 1024 * 1024 or outputs[2].stat().st_size >= 10 * 1024 * 1024:
        raise RuntimeError("public visual derivative exceeds the 10 MiB repository limit")
    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "stage": "road_reward_curriculum",
        "purpose": "visual comparison of the G0 inherited policy and the turn-aware reward checkpoint",
        "capture_reports": [{"path": portable_path(path), "sha256": file_sha256(path)} for path in capture_paths],
        "profile_order": [profile["profile_id"] for profile in profiles],
        "composition": {
            "seed": profiles[0]["seed"],
            "matched_command_sequence": True,
            "synchronized_panels": True,
            "interpretation": "visual playback only; linked multi-environment reports decide PASS/FAIL",
        },
        "tool": _run([args.ffmpeg, "-version"]).splitlines()[0],
        "local_composite": _artifact_metadata(outputs[0], args.ffprobe, "local_only"),
        "public_derivatives": {
            "gif": _artifact_metadata(outputs[1], args.ffprobe, "git_public"),
            "contact_sheet": {
                **_artifact_metadata(outputs[2], args.ffprobe, "git_public"),
                "frame_times_seconds": list(CONTACT_TIMES_S),
                "layout_order": ["forward", "backward", "left_turn", "right_turn"],
            },
        },
        "record_source_sha256": file_sha256(REPO_ROOT / "scripts" / "record_g008_road_curriculum.py"),
        "builder_source_sha256": file_sha256(Path(__file__)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-reports", nargs=2, required=True, type=Path)
    parser.add_argument("--local-composite", required=True, type=Path)
    parser.add_argument("--public-gif", required=True, type=Path)
    parser.add_argument("--public-contact-sheet", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/arial.ttf"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--rebuild-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build(args)
    output = args.output_report.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"report": str(output), "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
