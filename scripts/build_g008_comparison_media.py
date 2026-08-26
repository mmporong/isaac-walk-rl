#!/usr/bin/env python3
"""Build a local comparison MP4 and public GIF/PNG derivatives with FFmpeg."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROFILE_ORDER = ("command", "friction_s1", "leg_mass_s1")
CONTACT_TIMES_S = (3.0, 7.5, 12.0, 16.5)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        try:
            relative = resolved.relative_to(Path.home().resolve())
        except ValueError:
            return str(resolved)
        return "%USERPROFILE%\\" + str(relative)
    return str(relative).replace("\\", "/")


def resolve_portable(path_value: str) -> Path:
    if path_value.startswith("%USERPROFILE%\\"):
        return Path.home() / path_value.removeprefix("%USERPROFILE%\\")
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    return result.stdout


def _probe(path: Path, ffprobe: str) -> dict[str, Any]:
    return json.loads(
        _run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def _stream_summary(probe: dict[str, Any]) -> dict[str, Any]:
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    duration_value = probe.get("format", {}).get("duration") or video.get("duration")
    return {
        "codec": video["codec_name"],
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frame_rate": video.get("avg_frame_rate"),
        "frames": int(video["nb_frames"]) if video.get("nb_frames") else None,
        "duration_s": float(duration_value) if duration_value is not None else None,
    }


def _font_expression(font_path: Path) -> str:
    escaped = font_path.resolve().as_posix().replace(":", "\\:")
    return f"fontfile='{escaped}'"


def _panel_detail(profile: dict[str, Any]) -> str:
    runtime = profile["runtime_domain"]
    if profile["profile_id"] == "leg_mass_s1":
        scale = runtime["leg_mass_scale"]
        return f"mass scale {scale['min']:.3f}..{scale['max']:.3f} mean {scale['mean']:.3f}"
    static = runtime["foot_static_friction"]["mean"]
    dynamic = runtime["foot_dynamic_friction"]["mean"]
    prefix = "sampled" if profile["profile_id"] == "friction_s1" else "nominal"
    return f"{prefix} muS {static:.3f} muD {dynamic:.3f}"


def _drawtext(font: str, text: str, *, size: int, y: str, enable: str | None = None) -> str:
    expression = (
        f"drawtext={font}:text='{text}':x=(w-text_w)/2:y={y}:"
        f"fontsize={size}:fontcolor=white"
    )
    if enable is not None:
        expression += f":enable='{enable}'"
    return expression


def _build_composite(
    inputs: list[Path],
    profiles: list[dict[str, Any]],
    destination: Path,
    ffmpeg: str,
    font_path: Path,
) -> None:
    font = _font_expression(font_path)
    filters = []
    for index, profile in enumerate(profiles):
        label = profile["label"]
        detail = _panel_detail(profile)
        filters.append(
            f"[{index}:v]crop=480:360:(iw-480)/2:(ih-360)/2,"
            "scale=426:320:flags=lanczos,"
            "drawbox=x=0:y=0:w=iw:h=62:color=black@0.72:t=fill,"
            f"{_drawtext(font, label, size=21, y='7')},"
            f"{_drawtext(font, detail, size=15, y='36')}[panel{index}]"
        )
    filters.append("[panel0][panel1][panel2]hstack=inputs=3[stack]")
    timeline = [
        "[stack]pad=1280:380:1:0:black",
        "drawbox=x=0:y=320:w=1280:h=60:color=black:t=fill",
        _drawtext(
            font,
            "Plane | seed 42 | matched commands | inference playback",
            size=16,
            y="326",
        ),
        _drawtext(
            font,
            "STAND",
            size=20,
            y="350",
            enable=(
                "between(t\\,0\\,1)+between(t\\,4.5\\,5.5)+"
                "between(t\\,9\\,10)+between(t\\,13.5\\,14.5)"
            ),
        ),
        _drawtext(font, "FORWARD  vx +0.60 m/s", size=20, y="350", enable="between(t\\,1\\,4.5)"),
        _drawtext(font, "BACKWARD  vx -0.40 m/s", size=20, y="350", enable="between(t\\,5.5\\,9)"),
        _drawtext(font, "LEFT TURN  wz +0.50 rad/s", size=20, y="350", enable="between(t\\,10\\,13.5)"),
        _drawtext(font, "RIGHT TURN  wz -0.50 rad/s", size=20, y="350", enable="between(t\\,14.5\\,18)"),
    ]
    filters.append(",".join(timeline) + "[out]")
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
    for path in inputs:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    _run(command)


def _build_gif(source: Path, destination: Path, ffmpeg: str) -> None:
    filter_graph = (
        "fps=5,scale=720:-2:flags=lanczos,split[base][palette_source];"
        "[palette_source]palettegen=max_colors=64:stats_mode=diff[palette];"
        "[base][palette]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            filter_graph,
            "-loop",
            "0",
            str(destination),
        ]
    )


def _build_contact_sheet(source: Path, destination: Path, ffmpeg: str) -> None:
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
    for timestamp in CONTACT_TIMES_S:
        command.extend(["-ss", str(timestamp), "-i", str(source)])
    command.extend(
        [
            "-filter_complex",
            (
                "[0:v]scale=640:190:flags=lanczos[a];"
                "[1:v]scale=640:190:flags=lanczos[b];"
                "[2:v]scale=640:190:flags=lanczos[c];"
                "[3:v]scale=640:190:flags=lanczos[d];"
                "[a][b][c][d]xstack=inputs=4:layout=0_0|640_0|0_190|640_190[out]"
            ),
            "-map",
            "[out]",
            "-frames:v",
            "1",
            str(destination),
        ]
    )
    _run(command)


def _artifact_metadata(path: Path, ffprobe: str, git_policy: str) -> dict[str, Any]:
    return {
        "path": portable_path(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "git_policy": git_policy,
        **_stream_summary(_probe(path, ffprobe)),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    capture_report_paths = [path.resolve() for path in args.capture_reports]
    capture_reports = [json.loads(path.read_text(encoding="utf-8")) for path in capture_report_paths]
    if any(report.get("status") != "complete" for report in capture_reports):
        raise ValueError("all isolated capture reports must be complete")
    if len({report["seed"] for report in capture_reports}) != 1:
        raise ValueError("capture seeds do not match")
    if len({report["terrain_mode"] for report in capture_reports}) != 1:
        raise ValueError("capture terrain modes do not match")
    sequence_contracts = {json.dumps(report["sequence"], sort_keys=True) for report in capture_reports}
    if len(sequence_contracts) != 1:
        raise ValueError("capture command sequences do not match")
    profiles = [report["profile"] for report in capture_reports]
    if tuple(profile["profile_id"] for profile in profiles) != EXPECTED_PROFILE_ORDER:
        raise ValueError("capture profile order does not match the comparison contract")

    input_paths = [resolve_portable(profile["local_video"]["path"]).resolve() for profile in profiles]
    outputs = (args.local_composite, args.public_gif, args.public_contact_sheet)
    for path in (*input_paths, args.font):
        if not path.is_file():
            raise FileNotFoundError(path)
    for output in outputs:
        output.resolve().parent.mkdir(parents=True, exist_ok=True)
        if output.resolve().exists() and not (args.reuse_existing or args.rebuild_existing):
            raise FileExistsError(output.resolve())

    local_composite = args.local_composite.resolve()
    public_gif = args.public_gif.resolve()
    public_contact_sheet = args.public_contact_sheet.resolve()
    if args.rebuild_existing or not local_composite.exists():
        _build_composite(input_paths, profiles, local_composite, args.ffmpeg, args.font)
    if args.rebuild_existing or not public_gif.exists():
        _build_gif(local_composite, public_gif, args.ffmpeg)
    if args.rebuild_existing or not public_contact_sheet.exists():
        _build_contact_sheet(local_composite, public_contact_sheet, args.ffmpeg)

    ffmpeg_version = _run([args.ffmpeg, "-version"]).splitlines()[0]
    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "purpose": "public comparison of command, friction S1, and leg-mass S1 policies",
        "capture_report": {
            "files": [
                {"path": portable_path(path), "sha256": file_sha256(path)}
                for path in capture_report_paths
            ]
        },
        "profile_order": list(EXPECTED_PROFILE_ORDER),
        "composition": {
            "terrain_mode": capture_reports[0]["terrain_mode"],
            "seed": capture_reports[0]["seed"],
            "matched_command_sequence": True,
            "synchronized_panels": True,
            "interpretation": "visual playback; quantitative gate decisions remain in evaluation JSON",
        },
        "tool": ffmpeg_version,
        "local_composite": _artifact_metadata(local_composite, args.ffprobe, "local_only"),
        "public_derivatives": {
            "gif": _artifact_metadata(public_gif, args.ffprobe, "git_public"),
            "contact_sheet": {
                **_artifact_metadata(public_contact_sheet, args.ffprobe, "git_public"),
                "frame_times_seconds": list(CONTACT_TIMES_S),
                "layout_order": ["forward", "backward", "left_turn", "right_turn"],
            },
        },
        "record_source_sha256": file_sha256(REPO_ROOT / "scripts" / "record_g008_policy_comparison.py"),
        "builder_source_sha256": file_sha256(Path(__file__)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-reports", required=True, nargs=3, type=Path)
    parser.add_argument("--local-composite", required=True, type=Path)
    parser.add_argument("--public-gif", required=True, type=Path)
    parser.add_argument("--public-contact-sheet", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/arial.ttf"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument("--reuse-existing", action="store_true")
    reuse_group.add_argument("--rebuild-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build(args)
    output_report = args.output_report.resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_report.with_suffix(output_report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_report)
    print(json.dumps({"report": str(output_report), "status": report["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
