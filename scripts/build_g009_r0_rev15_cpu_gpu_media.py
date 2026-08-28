#!/usr/bin/env python3
"""Build public telemetry media for the rejected rev15 CPU/GPU divergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_STEM = "g009_5_r0_diag_rev15_07_cpu_gpu_telemetry"
DEFAULT_INPUT = (
    REPO_ROOT
    / "reports/runs/g009_r0_runtime_probe_rev15_rejection_synthesis_3x3_s42.json"
)
DEFAULT_PNG = REPO_ROOT / "docs/media/g009/R0/diagnostic" / f"{OUTPUT_STEM}.png"
DEFAULT_GIF = REPO_ROOT / "docs/media/g009/R0/diagnostic" / f"{OUTPUT_STEM}.gif"
DEFAULT_SUMMARY = REPO_ROOT / "reports/runs" / f"{OUTPUT_STEM}_visual_evidence.json"
FORCE_THRESHOLD_BW = 15.0
CPU_FORCE_OBSERVED_BW = 13.248281478881836
GPU_FORCE_OBSERVED_BW = 16.78827476501465
SEPARATION_THRESHOLD_M = -0.01
CPU_SEPARATION_OBSERVED_M = -0.009353086352348328
MAX_PUBLIC_BYTES = 10 * 1024 * 1024
LABELS = (
    "PUBLIC DIAGNOSTIC",
    "TELEMETRY ANIMATION · NOT CAMERA FOOTAGE",
    "07 CPU/GPU DIVERGENCE",
    "NO PPO",
    "REJECTED",
)
EXPECTED_SYNTHESIS_SHA256 = (
    "9cbb31e22d355c0c9f1f855dcac05486cf92c0268f4788621c85ad6d78e01f85"
)
EXPECTED_LINEAGE = {
    "source_commit": "bc999d504e226011ff3d83e68a416b9049b406cb",
    "source_bundle_sha256": (
        "218671a84f2748f7b94a426490057318b0896e2160454f6928c4277dee7435df"
    ),
    "contract_sha256": (
        "5f29ba19458404b5009d3734294c57e79294efecc7fe03bf8c71c71656129832"
    ),
}
EXPECTED_RUN_BINDINGS = {
    "cpu": (
        (
            "426f4fe1085aeddad52c77d98fc74a55907dcc90d7084ebe8b4fde736b60e9d5",
            "144b636184f542f1ae9319bbe2ebabed",
        ),
        (
            "e972efc874205d076e50c543659cb0aa2a5e3f74300f8627466272d5448bdace",
            "cbf7bc72e330486986ce48306d4feca4",
        ),
        (
            "49252da40105efd2309d88cc23d5ea1aab1207e56e80adbf5566af851242537a",
            "7ae4bca660994d3bb855c9ed73a7b4ad",
        ),
    ),
    "gpu": (
        (
            "e24674a1ed33c38fbe5f12d19dc068167b9787e75323efbe55629bf059839b91",
            "fc715b20c45242b19b86445f733fa02b",
        ),
        (
            "ef0871ee8ea82d614f613f959a7a45cb84e57ce00edd90d04a3226ac241c87d7",
            "99d201671b7d4763845bc4d0a38d40d1",
        ),
        (
            "ca64da55ae2828f3d259afec3281e5afa1f4c631130c1733f3bbe70080688b03",
            "9dc377f6a76f4dc092683b1cca2083a9",
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


def publish_new(staged: Path, final: Path) -> str:
    """Exclusively publish staged bytes without replacing an existing target."""

    created = False
    try:
        with staged.open("rb") as source, final.open("xb") as destination:
            created = True
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        digest = file_sha256(staged)
        require(file_sha256(final) == digest, "exclusive publish integrity mismatch")
        return digest
    except Exception:
        if created:
            final.unlink(missing_ok=True)
        raise


def cleanup_owned_outputs(published: list[tuple[Path, str]]) -> None:
    for path, expected_digest in published:
        if path.is_file() and file_sha256(path) == expected_digest:
            path.unlink()


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
        value.get("experiment") == "rev15_position_solver_cpu_gpu_divergence",
        "experiment mismatch",
    )
    require(
        value.get("status") == "rejected_before_gate01"
        and value.get("evidence_synthesis_valid") is True
        and value.get("candidate_runtime_calibration_passed") is False
        and value.get("learned") is False
        and value.get("ppo_training") is False
        and value.get("ppo_training_status") == "not_run",
        "rev15 must remain rejected pre-Gate01 evidence",
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
            "solver_position_iterations": 16,
            "solver_velocity_iterations": 0,
            "max_depenetration_velocity_m_s": 1.0,
            "all_paths_and_apis_valid": True,
        },
        "152-link physics readback mismatch",
    )
    decision = value.get("decision", {})
    require(
        decision.get("strict_decision") == "reject"
        and decision.get("blocking_device") == "gpu"
        and decision.get("blocking_check") == "nonfoot_peak_force_bounded",
        "strict decision mismatch",
    )
    device_results = value.get("device_results", {})
    cpu = device_results.get("cpu", {})
    gpu = device_results.get("gpu", {})
    require(
        cpu.get("peak_nonfoot_force_bodyweights") == CPU_FORCE_OBSERVED_BW
        and cpu.get("runtime_passed_runs") == 3
        and CPU_FORCE_OBSERVED_BW <= FORCE_THRESHOLD_BW,
        "CPU force PASS binding mismatch",
    )
    require(
        gpu.get("peak_nonfoot_force_bodyweights") == GPU_FORCE_OBSERVED_BW
        and gpu.get("runtime_passed_runs") == 0
        and gpu.get("failed_checks") == ["nonfoot_peak_force_bounded"]
        and GPU_FORCE_OBSERVED_BW > FORCE_THRESHOLD_BW,
        "GPU force FAIL binding mismatch",
    )
    require(
        cpu.get("separation_threshold_m") == SEPARATION_THRESHOLD_M
        and cpu.get("worst_contact_separation_m") == CPU_SEPARATION_OBSERVED_M
        and CPU_SEPARATION_OBSERVED_M >= SEPARATION_THRESHOLD_M,
        "CPU separation PASS binding mismatch",
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
            "strict_rejection_synthesis": True,
        },
        "completed diagnostic stages mismatch",
    )
    return value


def render_frame(synthesis: dict[str, Any], progress: float, destination: Path) -> None:
    import matplotlib  # pyright: ignore[reportMissingImports]

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

    device_results = synthesis["device_results"]
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
        "G009-5 R0 REV15 · 07 CPU/GPU DIVERGENCE · NO PPO · REJECTED",
        ha="center",
        va="center",
        color="#ffe36e",
        fontsize=18,
        fontweight="bold",
    )
    title.set_xticks([])
    title.set_yticks([])
    for spine in title.spines.values():
        spine.set_visible(False)

    force_axis = figure.add_subplot(grid[1, 0])
    shown_cpu_force = (
        float(device_results["cpu"]["peak_nonfoot_force_bodyweights"]) * progress
    )
    shown_gpu_force = (
        float(device_results["gpu"]["peak_nonfoot_force_bodyweights"]) * progress
    )
    force_axis.bar(
        ["CPU", "GPU"],
        [shown_cpu_force, shown_gpu_force],
        color=["#2ca02c", "#e45756"],
        width=0.58,
    )
    force_axis.axhline(
        FORCE_THRESHOLD_BW,
        color="#f2cf5b",
        linewidth=2.4,
        linestyle="--",
        label="limit 15 BW",
    )
    force_axis.set_ylim(0, 19)
    force_axis.set_ylabel("body weight", color="white")
    force_axis.set_title(
        "SAME CONTRACT · CPU PASS / GPU FAIL", color="#ffcf66", fontweight="bold"
    )
    force_axis.tick_params(axis="both", colors="white")
    force_axis.legend(loc="upper right")
    force_axis.text(
        0,
        max(0.6, shown_cpu_force - 0.8),
        f"{shown_cpu_force:.3f} BW",
        ha="center",
        color="white",
        fontweight="bold",
    )
    force_axis.text(
        1,
        max(0.6, shown_gpu_force - 0.8),
        f"{shown_gpu_force:.3f} BW",
        ha="center",
        color="white",
        fontweight="bold",
    )

    sep_axis = figure.add_subplot(grid[1, 1])
    threshold_mm = float(device_results["cpu"]["separation_threshold_m"]) * 1000
    shown_mm = (
        threshold_mm
        + (
            float(device_results["cpu"]["worst_contact_separation_m"]) * 1000
            - threshold_mm
        )
        * progress
    )
    sep_axis.bar(["CPU minimum separation"], [shown_mm], color="#2ca02c", width=0.55)
    sep_axis.axhline(
        threshold_mm,
        color="#f2cf5b",
        linewidth=2.4,
        linestyle="--",
        label="limit -10 mm",
    )
    sep_axis.set_ylim(-12, 0)
    sep_axis.set_ylabel("separation [mm]", color="white")
    sep_axis.set_title("CPU SEPARATION PASS", color="#79df79", fontweight="bold")
    sep_axis.tick_params(axis="both", colors="white")
    sep_axis.legend(loc="upper right")
    sep_axis.text(
        0,
        shown_mm + 0.65,
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
        "Validated evidence: 3 CPU + 3 GPU runs · seed 42 · 16/0 solver · 1.0 m/s depenetration",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    note.text(
        0.02,
        0.48,
        "CPU force PASS: 13.248 BW ≤ 15 BW · GPU force FAIL: 16.788 BW > 15 BW · CPU separation PASS: -9.353 mm",
        color="#d8dde8",
        fontsize=12,
    )
    note.text(
        0.02,
        0.18,
        "Interpretation: the candidate is rejected before Gate01 because the same contract diverges on GPU; PPO was not run.",
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
    from PIL import Image  # pyright: ignore[reportMissingImports]

    require(
        synthesis_path.resolve() == DEFAULT_INPUT.resolve()
        and png_path.resolve() == DEFAULT_PNG.resolve()
        and gif_path.resolve() == DEFAULT_GIF.resolve()
        and summary_path.resolve() == DEFAULT_SUMMARY.resolve(),
        "numbered input/output paths are fixed",
    )
    for path in (png_path, gif_path, summary_path):
        require(not path.exists(), f"refusing to overwrite output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    synthesis = read_synthesis(synthesis_path)
    with tempfile.TemporaryDirectory(prefix="g009-rev15-cpu-gpu-") as directory:
        staging = Path(directory)
        staged_png = staging / "telemetry.png"
        staged_gif = staging / "telemetry.gif"
        staged_summary = staging / "visual_evidence.json"
        frames: list[Image.Image] = []
        for index, progress in enumerate((0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
            temporary = staging / f"frame_{index:03d}.png"
            render_frame(synthesis, progress, temporary)
            frames.append(Image.open(temporary).convert("RGB").copy())
        frames[-1].save(staged_png, optimize=True)
        palette_frames = [
            frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=96)
            for frame in frames
        ]
        palette_frames[0].save(
            staged_gif,
            save_all=True,
            append_images=palette_frames[1:],
            duration=900,
            loop=0,
            optimize=True,
        )
        validate_public_media(staged_png, b"\x89PNG\r\n\x1a\n")
        validate_public_media(staged_gif, b"GIF8")
        summary = {
            "schema_version": "g009.r0.rev15.cpu_gpu_visual_evidence.v1",
            "goal_id": "g009",
            "stage_number": "G009-5",
            "stage_id": "R0",
            "status": "rejected",
            "diagnostic_only": True,
            "telemetry_animation": True,
            "camera_footage": False,
            "public_claim_eligible": False,
            "qualification_status": "not_run",
            "qualification_passed": None,
            "learned_policy_qualified": False,
            "ppo_training_run": False,
            "candidate_runtime_calibration_passed": False,
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
            "cpu_force_check": {
                "passed": True,
                "threshold_bodyweights": FORCE_THRESHOLD_BW,
                "observed_bodyweights": CPU_FORCE_OBSERVED_BW,
            },
            "gpu_force_check": {
                "passed": False,
                "threshold_bodyweights": FORCE_THRESHOLD_BW,
                "observed_bodyweights": GPU_FORCE_OBSERVED_BW,
                "blocking_check": "nonfoot_peak_force_bounded",
            },
            "cpu_separation_check": {
                "passed": True,
                "threshold_m": SEPARATION_THRESHOLD_M,
                "observed_m": CPU_SEPARATION_OBSERVED_M,
            },
            "png": {
                "path": repo_path(png_path),
                "sha256": file_sha256(staged_png),
                "bytes": staged_png.stat().st_size,
                "width": 1280,
                "height": 720,
            },
            "gif": {
                "path": repo_path(gif_path),
                "sha256": file_sha256(staged_gif),
                "bytes": staged_gif.stat().st_size,
                "frames": 6,
                "duration_ms": 5400,
            },
        }
        staged_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        require(
            json.loads(staged_summary.read_text(encoding="utf-8")) == summary,
            "staged summary round-trip mismatch",
        )
        published: list[tuple[Path, str]] = []
        try:
            for staged, final in (
                (staged_png, png_path),
                (staged_gif, gif_path),
                (staged_summary, summary_path),
            ):
                published.append((final, publish_new(staged, final)))
            require(
                file_sha256(png_path) == summary["png"]["sha256"]
                and file_sha256(gif_path) == summary["gif"]["sha256"]
                and json.loads(summary_path.read_text(encoding="utf-8")) == summary,
                "published telemetry evidence integrity mismatch",
            )
        except Exception:
            cleanup_owned_outputs(published)
            raise
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
