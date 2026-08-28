#!/usr/bin/env python3
"""Build public telemetry media for the rejected rev14 force/separation trade-off."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_STEM = "g009_5_r0_diag_rev14_05_cpu_tradeoff"
DEFAULT_INPUT = (
    REPO_ROOT
    / "reports/runs/g009_r0_runtime_probe_rev14_tradeoff_synthesis_3x3_s42.json"
)
DEFAULT_PNG = REPO_ROOT / "docs/media/g009/R0/diagnostic" / f"{OUTPUT_STEM}.png"
DEFAULT_GIF = REPO_ROOT / "docs/media/g009/R0/diagnostic" / f"{OUTPUT_STEM}.gif"
DEFAULT_SUMMARY = REPO_ROOT / "reports/runs" / f"{OUTPUT_STEM}_visual_evidence.json"
FORCE_THRESHOLD_BW = 15.0
FORCE_OBSERVED_BW = 13.943856239318848
SEPARATION_THRESHOLD_M = -0.01
SEPARATION_OBSERVED_M = -0.010990187525749207
MAX_PUBLIC_BYTES = 10 * 1024 * 1024
LABELS = (
    "PUBLIC DIAGNOSTIC",
    "TELEMETRY ANIMATION",
    "NOT CAMERA FOOTAGE",
    "NO PPO",
    "REJECTED",
)
EXPECTED_SYNTHESIS_SHA256 = (
    "605fa6a6080109c5b8699364216102fc61d41c761a07316c4441cf30de97553c"
)
EXPECTED_LINEAGE = {
    "source_commit": "e9c1eff15bb2679c67e325546a749dbe7f98b07c",
    "source_bundle_sha256": (
        "5c3cfa41a9c6b61a5579ed48ed17eb4f0f363eeebb9f970b61eada09fca8bacc"
    ),
    "contract_sha256": (
        "744c53d3c8d1e608f849af405c7d0fad314b01234fc4cb9a4ab1000c69140506"
    ),
}
EXPECTED_RUN_BINDINGS = {
    "cpu": (
        (
            "3cc04ef58582b8a9dc7c77b46ce8747965db66561c949f70f4ff40c3241faa1f",
            "a0fe0250b7cb4f51bb691deaa4d986af",
        ),
        (
            "c0608e42fd128027c3945cd944a76eaa56fa95ae343b335fc6cf6b32d1704149",
            "c5ffbdba22774960ae9a896b79a3bdc4",
        ),
        (
            "8d47c402359b0cf48d2c3dcc54b64641cd5f5794f1885be9cd6f5dfbe0e701fb",
            "260e7bb0cd584ffaae3c2401cdf58d95",
        ),
    ),
    "gpu": (
        (
            "42ee16fc344a75f10f9bfc6db580d7fff74c946c19b06ccc902564305b8af4a9",
            "1f09537096c34236bdda36e53c100b01",
        ),
        (
            "c23515c70936ab72a65c848be1beba6638a22a3f5e6e8fc614fd6702942583ec",
            "65d7381c4dbc45ce8248e106f2aac69d",
        ),
        (
            "bf0c3c6cd26d6a67a1957341f0e55cf7e70b7716620370efce65ddddcfc0704e",
            "6c6f2320328843cbb12fd6dcce9a71ae",
        ),
    ),
}


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
    require(path.resolve() == DEFAULT_INPUT.resolve(), "synthesis input path is fixed")
    require(
        path.is_file() and file_sha256(path) == EXPECTED_SYNTHESIS_SHA256,
        "synthesis input hash mismatch",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "synthesis root must be an object")
    require(
        value.get("experiment") == "rev14_max_depenetration_velocity_tradeoff",
        "experiment mismatch",
    )
    require(
        value.get("status") == "rejected_before_gate01"
        and value.get("learned") is False,
        "rev14 tradeoff must remain rejected pre-Gate01 evidence",
    )
    require(
        value.get("qualification_status") == "not_run", "qualification claim mismatch"
    )
    require(value.get("lineage") == EXPECTED_LINEAGE, "source lineage mismatch")
    repeatability = value.get("repeatability", {})
    require(
        repeatability.get("unique_execution_ids") == 6,
        "six independent executions are required",
    )
    require(
        repeatability.get("cpu", {}).get("validated_runs") == 3
        and repeatability.get("gpu", {}).get("validated_runs") == 3,
        "3x CPU and 3x GPU validation is required",
    )
    for device, expected in EXPECTED_RUN_BINDINGS.items():
        observed = tuple(
            (run.get("sha256"), run.get("execution_id"))
            for run in repeatability.get(device, {}).get("inputs", ())
            if isinstance(run, dict)
        )
        require(observed == expected, f"{device} run binding mismatch")
    require(
        value.get("physics_readback")
        == {
            "articulations_per_run": 8,
            "links_per_articulation": 19,
            "rigid_bodies_per_run": 152,
            "max_depenetration_velocity_m_s": 0.75,
            "all_paths_and_apis_valid": True,
        },
        "152-link physics readback mismatch",
    )
    tradeoff = value.get("tradeoff", {})
    require(tradeoff.get("strict_decision") == "reject", "strict decision mismatch")
    require(
        tradeoff.get("cpu_global_peak_bodyweights") == FORCE_OBSERVED_BW
        and FORCE_OBSERVED_BW <= FORCE_THRESHOLD_BW,
        "force PASS binding mismatch",
    )
    require(
        tradeoff.get("separation_threshold_m") == SEPARATION_THRESHOLD_M
        and tradeoff.get("cpu_worst_separation_m") == SEPARATION_OBSERVED_M,
        "separation FAIL binding mismatch",
    )
    require(
        value.get("blocked_stages")
        == {"gate01": True, "gate10": True, "ppo_training": True},
        "downstream stages must remain blocked",
    )
    require(
        value.get("completed_stages")
        == {
            "cpu_runtime_3x": True,
            "gpu_runtime_3x": True,
            "strict_tradeoff_synthesis": True,
        },
        "completed diagnostic stages mismatch",
    )
    return value


def render_frame(synthesis: dict[str, Any], progress: float, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tradeoff = synthesis["tradeoff"]
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#10131a")
    grid = figure.add_gridspec(
        3, 2, height_ratios=(0.58, 2.15, 1.25), hspace=0.42, wspace=0.32
    )
    title = figure.add_subplot(grid[0, :])
    title.set_facecolor("#8b1111")
    title.text(
        0.5,
        0.68,
        "PUBLIC DIAGNOSTIC · TELEMETRY ANIMATION · NOT CAMERA FOOTAGE",
        ha="center",
        va="center",
        color="white",
        fontsize=17,
        fontweight="bold",
    )
    title.text(
        0.5,
        0.22,
        "G009 R0 REV14 · NO PPO · REJECTED",
        ha="center",
        va="center",
        color="#ffe36e",
        fontsize=20,
        fontweight="bold",
    )
    title.set_xticks([])
    title.set_yticks([])
    for spine in title.spines.values():
        spine.set_visible(False)

    force_axis = figure.add_subplot(grid[1, 0])
    shown_force = float(tradeoff["cpu_global_peak_bodyweights"]) * progress
    force_axis.bar(["non-foot peak"], [shown_force], color="#2ca02c", width=0.55)
    force_axis.axhline(
        FORCE_THRESHOLD_BW,
        color="#f2cf5b",
        linewidth=2.4,
        linestyle="--",
        label="limit 15 BW",
    )
    force_axis.set_ylim(0, 17)
    force_axis.set_ylabel("body weight", color="white")
    force_axis.set_title("FORCE PASS", color="#79df79", fontweight="bold")
    force_axis.tick_params(axis="both", colors="white")
    force_axis.legend(loc="upper right")
    force_axis.text(
        0,
        shown_force + 0.35,
        f"{shown_force:.3f} BW",
        ha="center",
        color="white",
        fontweight="bold",
    )

    sep_axis = figure.add_subplot(grid[1, 1])
    threshold_mm = float(tradeoff["separation_threshold_m"]) * 1000
    shown_mm = (
        threshold_mm
        + (float(tradeoff["cpu_worst_separation_m"]) * 1000 - threshold_mm) * progress
    )
    sep_axis.bar(["minimum separation"], [shown_mm], color="#e45756", width=0.55)
    sep_axis.axhline(
        threshold_mm,
        color="#f2cf5b",
        linewidth=2.4,
        linestyle="--",
        label="limit -10 mm",
    )
    sep_axis.set_ylim(-12, 0)
    sep_axis.set_ylabel("separation [mm]", color="white")
    sep_axis.set_title("CPU SEPARATION FAIL", color="#ff7979", fontweight="bold")
    sep_axis.tick_params(axis="both", colors="white")
    sep_axis.legend(loc="upper right")
    sep_axis.text(
        0,
        shown_mm - 0.45,
        f"{shown_mm:.3f} mm",
        ha="center",
        color="white",
        fontweight="bold",
    )

    note = figure.add_subplot(grid[2, :])
    note.axis("off")
    note.text(
        0.02,
        0.78,
        "Validated envelope: 3 CPU + 3 GPU runs · seed 42 · all runtime force checks pass",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    note.text(
        0.02,
        0.48,
        "Global force PASS: 13.944 BW ≤ 15 BW · CPU separation FAIL: -10.990 mm < -10 mm",
        color="#d8dde8",
        fontsize=12,
    )
    note.text(
        0.02,
        0.18,
        "Interpretation: lower collision force does not prove acceptable contact topology; GPU validation completed, Gate01/PPO remain blocked.",
        color="#f2cf5b",
        fontsize=11.5,
    )
    figure.savefig(destination, format="png", facecolor=figure.get_facecolor())
    plt.close(figure)


def validate_public_media(path: Path, signature: bytes) -> None:
    require(
        path.is_file()
        and path.stat().st_size > 0
        and path.read_bytes().startswith(signature),
        f"invalid media: {path}",
    )
    require(
        path.stat().st_size < MAX_PUBLIC_BYTES, f"public media exceeds 10 MiB: {path}"
    )


def write_outputs(
    synthesis_path: Path, png_path: Path, gif_path: Path, summary_path: Path
) -> dict[str, Any]:
    from PIL import Image

    for path in (png_path, gif_path, summary_path):
        require(not path.exists(), f"refusing to overwrite output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    synthesis = read_synthesis(synthesis_path)
    frames: list[Image.Image] = []
    with tempfile.TemporaryDirectory(prefix="g009-rev14-tradeoff-") as directory:
        for index, progress in enumerate((0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
            temporary = Path(directory) / f"frame_{index:03d}.png"
            render_frame(synthesis, progress, temporary)
            frames.append(Image.open(temporary).convert("RGB").copy())
    frames[-1].save(png_path, optimize=True)
    palette_frames = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=96)
        for frame in frames
    ]
    palette_frames[0].save(
        gif_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=900,
        loop=0,
        optimize=True,
    )
    validate_public_media(png_path, b"\x89PNG\r\n\x1a\n")
    validate_public_media(gif_path, b"GIF8")
    summary = {
        "schema_version": "g009.r0.rev14.tradeoff_visual_evidence.v1",
        "goal_id": "g009",
        "stage_number": "G009-5",
        "stage_id": "R0",
        "status": "rejected",
        "diagnostic_only": True,
        "telemetry_animation": True,
        "camera_footage": False,
        "public_claim_eligible": False,
        "qualification_status": "not_run",
        "learned_policy_qualified": False,
        "ppo_training_run": False,
        "completed_stages": synthesis["completed_stages"],
        "blocked_stages": synthesis["blocked_stages"],
        "labels": list(LABELS),
        "source": {
            "path": repo_path(synthesis_path),
            "sha256": file_sha256(synthesis_path),
        },
        "source_builder": {
            "path": repo_path(Path(__file__)),
            "sha256": file_sha256(Path(__file__)),
        },
        "force_check": {
            "passed": True,
            "threshold_bodyweights": FORCE_THRESHOLD_BW,
            "observed_bodyweights": FORCE_OBSERVED_BW,
        },
        "separation_check": {
            "passed": False,
            "threshold_m": SEPARATION_THRESHOLD_M,
            "observed_m": SEPARATION_OBSERVED_M,
        },
        "png": {
            "path": repo_path(png_path),
            "sha256": file_sha256(png_path),
            "bytes": png_path.stat().st_size,
            "width": 1280,
            "height": 720,
        },
        "gif": {
            "path": repo_path(gif_path),
            "sha256": file_sha256(gif_path),
            "bytes": gif_path.stat().st_size,
            "frames": 6,
            "duration_ms": 5400,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(
        json.dumps(
            write_outputs(args.input, args.png, args.gif, args.summary),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
