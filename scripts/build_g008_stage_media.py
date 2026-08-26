#!/usr/bin/env python3
"""Build local annotated MP4s and public GIF/PNG evidence for G008 stage changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_g008_comparison_media import (
    _artifact_metadata,
    _drawtext,
    _font_expression,
    _run,
    file_sha256,
    portable_path,
    resolve_portable,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTACT_TIMES_S = (3.0, 7.5, 12.0, 16.5)
EXPECTED_PROFILES = {
    "periodic_friction": ("periodic_friction_s1_mu020_010",),
    "link_mass": (
        "link_mass_hip_120",
        "link_mass_thigh_120",
        "link_mass_calf_120",
        "link_mass_foot_120",
    ),
}
QUANTITATIVE_REPORTS = {
    "periodic_friction": (
        REPO_ROOT
        / "reports"
        / "runs"
        / "g008_periodic_friction_sweep_command_vs_friction_s1_e32_h500_s20260826.json"
    ),
    "link_mass": (
        REPO_ROOT
        / "reports"
        / "runs"
        / "g008_link_mass_sensitivity_command_vs_leg_mass_s1_e800_h300_s20260826.json"
    ),
}


def _timeline_filters(font: str, y: str) -> list[str]:
    return [
        _drawtext(
            font,
            "STAND",
            size=20,
            y=y,
            enable=(
                "between(t\\,0\\,1)+between(t\\,4.5\\,5.5)+"
                "between(t\\,9\\,10)+between(t\\,13.5\\,14.5)"
            ),
        ),
        _drawtext(
            font,
            "FORWARD  vx +0.60 m/s",
            size=20,
            y=y,
            enable="between(t\\,1\\,4.5)",
        ),
        _drawtext(
            font,
            "BACKWARD  vx -0.40 m/s",
            size=20,
            y=y,
            enable="between(t\\,5.5\\,9)",
        ),
        _drawtext(
            font,
            "LEFT TURN  wz +0.50 rad/s",
            size=20,
            y=y,
            enable="between(t\\,10\\,13.5)",
        ),
        _drawtext(
            font,
            "RIGHT TURN  wz -0.50 rad/s",
            size=20,
            y=y,
            enable="between(t\\,14.5\\,18)",
        ),
    ]


def _build_periodic_video(
    source: Path,
    destination: Path,
    ffmpeg: str,
    font_path: Path,
) -> None:
    font = _font_expression(font_path)
    filters = [
        "[0:v]scale=1280:720:flags=lanczos",
        "drawbox=x=0:y=0:w=iw:h=78:color=black@0.72:t=fill",
        _drawtext(font, "Periodic mixed-friction floor | Friction S1", size=26, y="8"),
        _drawtext(
            font,
            "blue low muS/muD 0.2/0.1 | brown high 0.8/0.6 | stripe 0.5 m",
            size=18,
            y="45",
        ),
        "pad=1280:780:0:0:black",
        "drawbox=x=0:y=720:w=1280:h=60:color=black:t=fill",
        *_timeline_filters(font, "748"),
    ]
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(source),
            "-vf",
            ",".join(filters),
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


def _mass_panel_detail(profile: dict[str, Any]) -> str:
    physics = profile["stage_physics"]
    total_mass = physics["total_leg_mass_kg"]["mean"]
    return f"{physics['group']} x{physics['factor']:.2f} | total leg {total_mass:.3f} kg"


def _build_mass_video(
    sources: list[Path],
    profiles: list[dict[str, Any]],
    destination: Path,
    ffmpeg: str,
    font_path: Path,
) -> None:
    font = _font_expression(font_path)
    filters = []
    for index, profile in enumerate(profiles):
        filters.append(
            f"[{index}:v]crop=640:360:(iw-640)/2:(ih-360)/2,"
            "drawbox=x=0:y=0:w=iw:h=55:color=black@0.72:t=fill,"
            f"{_drawtext(font, profile['label'], size=20, y='5')},"
            f"{_drawtext(font, _mass_panel_detail(profile), size=15, y='32')}[panel{index}]"
        )
    filters.extend(
        (
            "[panel0][panel1]hstack=inputs=2[top]",
            "[panel2][panel3]hstack=inputs=2[bottom]",
            "[top][bottom]vstack=inputs=2[grid]",
        )
    )
    grid_filters = [
        "[grid]pad=1280:780:0:0:black",
        "drawbox=x=0:y=720:w=1280:h=60:color=black:t=fill",
        *_timeline_filters(font, "748"),
    ]
    filters.append(",".join(grid_filters) + "[out]")
    command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y"]
    for source in sources:
        command.extend(["-i", str(source)])
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
        "fps=4,scale=720:-2:flags=lanczos,split[base][palette_source];"
        "[palette_source]palettegen=max_colors=48:stats_mode=diff[palette];"
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
                "[0:v]scale=640:390:flags=lanczos[a];"
                "[1:v]scale=640:390:flags=lanczos[b];"
                "[2:v]scale=640:390:flags=lanczos[c];"
                "[3:v]scale=640:390:flags=lanczos[d];"
                "[a][b][c][d]xstack=inputs=4:layout=0_0|640_0|0_390|640_390[out]"
            ),
            "-map",
            "[out]",
            "-frames:v",
            "1",
            str(destination),
        ]
    )
    _run(command)


def build(args: argparse.Namespace) -> dict[str, Any]:
    capture_paths = [path.resolve() for path in args.capture_reports]
    capture_reports = [json.loads(path.read_text(encoding="utf-8")) for path in capture_paths]
    if any(report.get("status") != "complete" for report in capture_reports):
        raise ValueError("all capture reports must be complete")
    profiles = [report["profile"] for report in capture_reports]
    expected = EXPECTED_PROFILES[args.stage]
    actual = tuple(profile["profile_id"] for profile in profiles)
    if actual != expected:
        raise ValueError(f"capture order mismatch: expected={expected}, actual={actual}")
    if any(profile["stage"] != args.stage for profile in profiles):
        raise ValueError("capture stage mismatch")
    seeds = {profile["seed"] for profile in profiles}
    sequences = {json.dumps(profile["sequence"], sort_keys=True) for profile in profiles}
    if len(seeds) != 1 or len(sequences) != 1:
        raise ValueError("capture seeds or command sequences do not match")

    sources = [resolve_portable(profile["local_video"]["path"]).resolve() for profile in profiles]
    quantitative_report = QUANTITATIVE_REPORTS[args.stage]
    for path in (*sources, args.font.resolve(), quantitative_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    outputs = (
        args.local_composite.resolve(),
        args.public_gif.resolve(),
        args.public_contact_sheet.resolve(),
    )
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not args.rebuild_existing:
            raise FileExistsError(output)

    if args.stage == "periodic_friction":
        _build_periodic_video(sources[0], outputs[0], args.ffmpeg, args.font.resolve())
    else:
        _build_mass_video(sources, profiles, outputs[0], args.ffmpeg, args.font.resolve())
    _build_gif(outputs[0], outputs[1], args.ffmpeg)
    _build_contact_sheet(outputs[0], outputs[2], args.ffmpeg)
    ffmpeg_version = _run([args.ffmpeg, "-version"]).splitlines()[0]
    return {
        "schema_version": 1,
        "goal": "G008",
        "status": "complete",
        "stage": args.stage,
        "purpose": "visual evidence for a dynamics stage change",
        "capture_reports": [
            {"path": portable_path(path), "sha256": file_sha256(path)} for path in capture_paths
        ],
        "profile_order": list(actual),
        "composition": {
            "seed": profiles[0]["seed"],
            "matched_command_sequence": True,
            "synchronized_panels": len(profiles) > 1,
            "interpretation": "visual playback only; quantitative gates remain in the linked evaluation report",
        },
        "quantitative_report": {
            "path": portable_path(quantitative_report),
            "sha256": file_sha256(quantitative_report),
        },
        "tool": ffmpeg_version,
        "local_composite": _artifact_metadata(outputs[0], args.ffprobe, "local_only"),
        "public_derivatives": {
            "gif": _artifact_metadata(outputs[1], args.ffprobe, "git_public"),
            "contact_sheet": {
                **_artifact_metadata(outputs[2], args.ffprobe, "git_public"),
                "frame_times_seconds": list(CONTACT_TIMES_S),
                "layout_order": ["forward", "backward", "left_turn", "right_turn"],
            },
        },
        "record_source_sha256": file_sha256(
            REPO_ROOT / "scripts" / "record_g008_stage_evidence.py"
        ),
        "builder_source_sha256": file_sha256(Path(__file__)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=tuple(EXPECTED_PROFILES))
    parser.add_argument("--capture-reports", required=True, nargs="+", type=Path)
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
    output_report = args.output_report.resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_report.with_suffix(output_report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_report)
    print(json.dumps({"report": str(output_report), "status": report["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
