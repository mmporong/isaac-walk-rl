#!/usr/bin/env python3
"""Render rev16 four-group force/peak/concentration telemetry evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "9ac874f48a1403e0ed838beb5e75938db5873d1c"
EXPECTED_INPUT_SHA256 = (
    "d39931ad6ddf6104095a6276e9b6db3a047d044d203e034f2d38f1f172e0288d"
)
OUTPUT_STEM = "g009_5_r0_diag_rev16_09_four_group_telemetry"
DEFAULT_INPUT = (
    REPO_ROOT / "reports/runs/g009_r0_rev16_synthesis_12_full_retry01_s42.json"
)
DEFAULT_PNG = REPO_ROOT / "docs/media/g009/R0/diagnostic" / f"{OUTPUT_STEM}.png"
DEFAULT_GIF = REPO_ROOT / "docs/media/g009/R0/diagnostic" / f"{OUTPUT_STEM}.gif"
DEFAULT_SUMMARY = REPO_ROOT / "reports/runs" / f"{OUTPUT_STEM}_visual_evidence.json"
LABELS = (
    "PUBLIC DIAGNOSTIC",
    "TELEMETRY ANIMATION · NOT CAMERA FOOTAGE",
    "09 FOUR-GROUP DYNAMICS",
    "NO PPO",
    "REJECTED",
    "NOT QUALIFIED",
)
MAX_PUBLIC_BYTES = 10 * 1024 * 1024
MAX_GIF_BYTES = 6 * 1024 * 1024
GROUP_LABELS = ("A CPU", "A GPU", "B CPU", "B GPU")


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


def read_synthesis(path: Path) -> dict[str, Any]:
    require(path.resolve() == DEFAULT_INPUT.resolve(), "synthesis path is fixed")
    require(file_sha256(path) == EXPECTED_INPUT_SHA256, "synthesis hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(
        value.get("schema_version") == "g009.r0.rev16.backend_divergence_synthesis.v1"
        and value.get("source_commit") == SOURCE_COMMIT
        and value.get("evidence_synthesis_valid") is True
        and value.get("input_report_count") == 12
        and value.get("completed_group_count") == 4,
        "full rev16 synthesis binding mismatch",
    )
    require(
        value.get("hypothesis", {}).get("decision") == "inconclusive",
        "hypothesis decision must be inconclusive",
    )
    ratios = [
        float(row["derived"]["b_gpu_over_b_cpu_concentration_ratio"])
        for row in value["hypothesis"]["replicates"]
    ]
    require(
        len(ratios) == 3
        and all(ratio == ratios[0] for ratio in ratios)
        and ratios[0] < 1.20,
        "1.20 concentration criterion mismatch",
    )
    governance = value.get("governance", {})
    require(
        governance.get("position16_accepted") is False
        and governance.get("ppo", {}).get("status") == "not_run"
        and governance.get("gate01", {}).get("status") == "forbidden"
        and governance.get("gate10", {}).get("status") == "forbidden"
        and governance.get("qualification", {}).get("status") == "not_run",
        "rejection governance mismatch",
    )
    require(
        [(g["arm"], g["device"]) for g in value["groups"]]
        == [("A", "cpu"), ("A", "cuda:0"), ("B", "cpu"), ("B", "cuda:0")],
        "group order mismatch",
    )
    return value


def first_group_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [group["runs"][0] for group in value["groups"]]
    require(
        all(group["replicate_count"] == 3 for group in value["groups"]),
        "each group requires 3 replicates",
    )
    return rows


def render_frame(value: dict[str, Any], progress: float, destination: Path) -> None:
    import matplotlib  # pyright: ignore[reportMissingImports]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    rows = first_group_rows(value)
    forces = [float(row["peak_base_force_bodyweights"]) * progress for row in rows]
    concentrations = [float(row["concentration_index"]) * progress for row in rows]
    steps = [int(row["peak_base_force_physics_step"]) for row in rows]
    over = [
        [
            int(
                row["contact_exposure"]["thresholds"][f"over_{n}_bodyweights"][
                    "step_count"
                ]
            )
            for n in (5, 10, 15)
        ]
        for row in rows
    ]
    ratio = float(
        value["hypothesis"]["replicates"][0]["derived"][
            "b_gpu_over_b_cpu_concentration_ratio"
        ]
    )
    colors = ["#4ca3dd", "#7fc8f8", "#f2a65a", "#e45756"]
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#101722")
    grid = figure.add_gridspec(
        3, 2, height_ratios=[0.25, 1.0, 0.45], hspace=0.43, wspace=0.30
    )
    title = figure.add_subplot(grid[0, :])
    title.axis("off")
    title.text(
        0.5,
        0.74,
        "G009-5 · REV16 · FOUR-GROUP DYNAMICS",
        ha="center",
        color="white",
        fontsize=19,
        fontweight="bold",
    )
    title.text(
        0.5,
        0.18,
        "DIAGNOSTIC · REJECTED · NO PPO · NOT QUALIFIED · TELEMETRY (NOT CAMERA)",
        ha="center",
        color="#ffcf66",
        fontsize=11.5,
        fontweight="bold",
    )
    force_axis = figure.add_subplot(grid[1, 0])
    force_axis.set_facecolor("#17202d")
    force_axis.bar(GROUP_LABELS, forces, color=colors)
    force_axis.axhline(15, color="#f2cf5b", linestyle="--", linewidth=2)
    force_axis.set_ylim(0, 19)
    force_axis.set_title("PEAK BASE FORCE (BW)", color="white", fontweight="bold")
    force_axis.tick_params(colors="white")
    for i, force in enumerate(forces):
        force_axis.text(
            i, force + 0.35, f"{force:.2f}", ha="center", color="white", fontsize=9
        )
    concentration_axis = figure.add_subplot(grid[1, 1])
    concentration_axis.set_facecolor("#17202d")
    concentration_axis.bar(GROUP_LABELS, concentrations, color=colors)
    concentration_axis.set_ylim(0, 0.9)
    concentration_axis.set_title(
        "PEAK / 17-STEP WINDOW IMPULSE", color="white", fontweight="bold"
    )
    concentration_axis.tick_params(colors="white")
    for i, concentration in enumerate(concentrations):
        concentration_axis.text(
            i,
            concentration + 0.025,
            f"{concentration:.3f}",
            ha="center",
            color="white",
            fontsize=9,
        )
    note = figure.add_subplot(grid[2, :])
    note.axis("off")
    note.text(
        0.02,
        0.78,
        "Peak step: "
        + " · ".join(
            f"{label} {step}" for label, step in zip(GROUP_LABELS, steps, strict=True)
        ),
        color="white",
        fontsize=11.5,
        fontweight="bold",
    )
    note.text(
        0.02,
        0.49,
        "Exposure steps >5/>10/>15 BW: "
        + " · ".join(
            f"{label} {a}/{b}/{c}"
            for label, (a, b, c) in zip(GROUP_LABELS, over, strict=True)
        ),
        color="#d8dde8",
        fontsize=10.5,
    )
    note.text(
        0.02,
        0.17,
        f"INCONCLUSIVE: B GPU / B CPU concentration = {ratio:.6f} < 1.20 · Position16 remains REJECTED",
        color="#ffcf66",
        fontsize=11.5,
        fontweight="bold",
    )
    figure.savefig(destination, facecolor=figure.get_facecolor())
    plt.close(figure)


def publish_new(staged: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    with final.open("xb") as output, staged.open("rb") as source:
        while block := source.read(1024 * 1024):
            output.write(block)
        output.flush()
        os.fsync(output.fileno())


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


def write_outputs(
    input_path: Path, png: Path, gif: Path, summary: Path
) -> dict[str, Any]:
    from PIL import Image  # pyright: ignore[reportMissingImports]

    require(
        input_path.resolve() == DEFAULT_INPUT.resolve()
        and png.resolve() == DEFAULT_PNG.resolve()
        and gif.resolve() == DEFAULT_GIF.resolve()
        and summary.resolve() == DEFAULT_SUMMARY.resolve(),
        "numbered input/output paths are fixed",
    )
    for path in (png, gif, summary):
        require(not path.exists(), f"refusing to overwrite output: {path}")
    value = read_synthesis(input_path)
    rows = first_group_rows(value)
    with tempfile.TemporaryDirectory(prefix="g009-rev16-telemetry-") as directory:
        temp = Path(directory)
        frames: list[Image.Image] = []
        for index, progress in enumerate((0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
            frame = temp / f"frame_{index}.png"
            render_frame(value, progress, frame)
            frames.append(Image.open(frame).convert("RGB").copy())
        staged_png, staged_gif = temp / "telemetry.png", temp / "telemetry.gif"
        frames[-1].save(staged_png, optimize=True)
        palette = [
            frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=96)
            for frame in frames
        ]
        palette[0].save(
            staged_gif,
            save_all=True,
            append_images=palette[1:],
            duration=850,
            loop=0,
            optimize=True,
        )
        validate_media(staged_png, b"\x89PNG\r\n\x1a\n")
        validate_media(staged_gif, b"GIF8", gif=True)
        ratio = value["hypothesis"]["replicates"][0]["derived"][
            "b_gpu_over_b_cpu_concentration_ratio"
        ]
        output = {
            "schema_version": "g009.r0.rev16.four_group_visual_evidence.v1",
            "goal_id": "g009",
            "stage_id": "R0",
            "stage_number": "09",
            "revision": "rev16",
            "status": "rejected",
            "diagnostic_only": True,
            "camera_footage": False,
            "telemetry_animation": True,
            "headless": True,
            "labels": list(LABELS),
            "source_commit": SOURCE_COMMIT,
            "source": {
                "path": repo_path(input_path),
                "sha256": file_sha256(input_path),
            },
            "governance": value["governance"],
            "hypothesis": {
                "decision": "inconclusive",
                "concentration_ratio": ratio,
                "required_ratio": 1.20,
                "passed": False,
            },
            "groups": [
                {
                    "label": label,
                    "peak_force_bodyweights": row["peak_base_force_bodyweights"],
                    "peak_step": row["peak_base_force_physics_step"],
                    "concentration_index": row["concentration_index"],
                    "exposure_steps": {
                        n: row["contact_exposure"]["thresholds"][
                            f"over_{n}_bodyweights"
                        ]["step_count"]
                        for n in (5, 10, 15)
                    },
                }
                for label, row in zip(GROUP_LABELS, rows, strict=True)
            ],
            "png": {
                "path": repo_path(png),
                "sha256": file_sha256(staged_png),
                "bytes": staged_png.stat().st_size,
                "width": 1280,
                "height": 720,
            },
            "gif": {
                "path": repo_path(gif),
                "sha256": file_sha256(staged_gif),
                "bytes": staged_gif.stat().st_size,
                "frames": 6,
                "duration_ms": 5100,
            },
        }
        staged_summary = temp / "summary.json"
        staged_summary.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
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
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    print(
        json.dumps(
            write_outputs(args.input, args.png, args.gif, args.summary),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
