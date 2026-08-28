#!/usr/bin/env python3
"""Reproduce the G009 R0 rev9 prone-pilot diagnostic from TensorBoard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from record_g009_r0_diagnostic import validate_diagnostic_training_report

DEFAULT_TRAINING_REPORT = REPO_ROOT / "reports/runs/go2_flat_recover_rev9_prone_pilot_s42_20260828-1421.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports/runs/g009_r0_flat_diagnostic_rev9_prone_pilot_analysis.json"
EXPECTED_TASK = "Isaac-G009-Recover-Flat-Go2-R0-v0"
EXPECTED_RUN_NAME = "go2_flat_recover_rev9_prone_pilot_s42_20260828-1421"
EXPECTED_TAGS = (
    "Episode_Reward/stable_support",
    "Episode_Reward/upright_hold",
    "Episode_Reward/stable_success_once",
    "Episode_Termination/stable_success",
    "Episode_Termination/numeric_invalid",
    "Episode_Termination/hard_joint_limit",
    "Curriculum/recover_pose_distribution/phase_index",
    "Curriculum/recover_pose_distribution/probability_prone",
    "Curriculum/recover_pose_distribution/probability_supine",
    "Curriculum/recover_pose_distribution/probability_left_side",
    "Curriculum/recover_pose_distribution/probability_right_side",
    "Policy/mean_noise_std",
    "Train/mean_reward",
)


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


def validate_pilot_report(path: Path) -> tuple[dict[str, Any], Path]:
    report = _read_json(path)
    _require(report.get("run_name") == EXPECTED_RUN_NAME, "unexpected rev9 pilot run_name")
    _require(report.get("task") == EXPECTED_TASK, "unexpected rev9 pilot task")
    _require(report.get("num_envs") == 1024, "rev9 pilot requires 1024 environments")
    _require(report.get("max_iterations") == 50, "rev9 pilot requires 50 iterations")
    _require(report.get("last_iteration") == 49, "rev9 pilot did not reach iteration 49")
    _require(report.get("seed") == 42, "rev9 pilot requires seed 42")
    _require(report.get("headless") is True, "rev9 pilot must be headless")
    qualification = report.get("qualification_mode", {})
    _require(qualification.get("enabled") is False, "pilot must not use qualification mode")
    _require(qualification.get("policy_qualification_status") == "not_run", "pilot qualification must be not_run")
    _require(report.get("resume", {}).get("enabled") is False, "pilot must be scratch training")
    _require(report.get("effective_hydra_overrides") == [], "pilot must not use Hydra overrides")
    _require(report.get("repository", {}).get("dirty") is False, "pilot training source was dirty")
    _require(report.get("source_bundle", {}).get("matches_repository_commit") is True, "pilot source bundle was not commit-bound")
    _require(report.get("run_health_passed") is True and report.get("exit_code") == 0, "pilot run health failed")
    tensorboard_dir = resolve_portable_path(report.get("artifacts", {}).get("tensorboard_directory", "")).resolve()
    _require(tensorboard_dir.is_dir(), f"TensorBoard directory is missing: {tensorboard_dir}")
    _require(any(tensorboard_dir.glob("events.out.tfevents.*")), "TensorBoard event file is missing")
    checkpoint = resolve_portable_path(report.get("artifacts", {}).get("checkpoint", "")).resolve()
    _require(checkpoint.is_file(), f"pilot checkpoint is missing: {checkpoint}")
    _require(file_sha256(checkpoint) == report.get("artifacts", {}).get("checkpoint_sha256"), "pilot checkpoint hash mismatch")
    validate_diagnostic_training_report(path, checkpoint)
    return report, tensorboard_dir


def load_scalar_series(tensorboard_dir: Path) -> dict[str, list[dict[str, float | int]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:  # pragma: no cover - exercised by Isaac bundled Python
        raise RuntimeError("TensorBoard is unavailable; run with Isaac Sim bundled Python") from exc
    accumulator = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing = sorted(set(EXPECTED_TAGS) - available)
    _require(not missing, f"required TensorBoard tags are missing: {missing}")
    return {
        tag: [
            {"step": int(item.step), "wall_time": float(item.wall_time), "value": float(item.value)}
            for item in accumulator.Scalars(tag)
        ]
        for tag in EXPECTED_TAGS
    }


def tensorboard_event_bundle(tensorboard_dir: Path) -> dict[str, Any]:
    event_files = sorted(tensorboard_dir.glob("events.out.tfevents.*"), key=lambda path: path.name)
    _require(bool(event_files), "TensorBoard event bundle is empty")
    files = {path.name: file_sha256(path) for path in event_files}
    payload = "\n".join(f"{name}:{digest}" for name, digest in files.items())
    return {
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "files": files,
    }


def validate_recomputed_summaries(
    report: Mapping[str, Any], series: Mapping[str, Sequence[Mapping[str, float | int]]]
) -> None:
    stored = report.get("tensorboard", {}).get("series_summary", {})
    for tag in EXPECTED_TAGS:
        _require(tag in stored, f"training report summary is missing: {tag}")
        recomputed = _summary(series[tag])
        recorded = stored[tag]
        for field in ("sample_count", "nonzero_sample_count"):
            _require(recomputed[field] == recorded.get(field), f"training report {tag}.{field} mismatch")
        for field in ("latest", "minimum", "maximum", "mean"):
            actual = float(recomputed[field])
            expected = float(recorded.get(field))
            _require(
                math.isclose(actual, expected, rel_tol=1.0e-9, abs_tol=1.0e-12),
                f"training report {tag}.{field} mismatch",
            )


def _summary(series: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    values = [float(item["value"]) for item in series]
    _require(bool(values), "TensorBoard scalar series is empty")
    return {
        "sample_count": len(values),
        "latest": values[-1],
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
        "nonzero_sample_count": sum(abs(value) > 1.0e-12 for value in values),
    }


def build_analysis(
    report: Mapping[str, Any],
    report_path: Path,
    tensorboard_dir: Path,
    series: Mapping[str, Sequence[Mapping[str, float | int]]],
    event_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    lengths = {tag: len(series[tag]) for tag in EXPECTED_TAGS}
    _require(set(lengths.values()) == {50}, f"expected exactly 50 scalar samples per tag: {lengths}")
    reference_steps = [int(item["step"]) for item in series["Train/mean_reward"]]
    for tag in EXPECTED_TAGS:
        _require([int(item["step"]) for item in series[tag]] == reference_steps, f"TensorBoard step alignment mismatch: {tag}")

    summaries = {tag: _summary(series[tag]) for tag in EXPECTED_TAGS}
    event_bundle = dict(event_bundle or tensorboard_event_bundle(tensorboard_dir))
    hard = summaries["Episode_Termination/hard_joint_limit"]
    success = summaries["Episode_Termination/stable_success"]
    success_reward = summaries["Episode_Reward/stable_success_once"]
    phase = summaries["Curriculum/recover_pose_distribution/phase_index"]
    prone = summaries["Curriculum/recover_pose_distribution/probability_prone"]
    non_prone_tags = (
        "Curriculum/recover_pose_distribution/probability_supine",
        "Curriculum/recover_pose_distribution/probability_left_side",
        "Curriculum/recover_pose_distribution/probability_right_side",
    )
    leak = (
        float(phase["maximum"]) > 0.0
        or float(prone["minimum"]) < 1.0
        or any(float(summaries[tag]["maximum"]) > 0.0 for tag in non_prone_tags)
    )
    reasons = ["diagnostic_pilot_never_qualifies"]
    if int(hard["nonzero_sample_count"]) > 0:
        reasons.append("hard_joint_limit_nonzero")
    if int(success["nonzero_sample_count"]) == 0 and int(success_reward["nonzero_sample_count"]) == 0:
        reasons.append("strict_success_zero")
    if leak:
        reasons.append("prone_curriculum_boundary_leak")

    tail = []
    for index in range(40, 50):
        tail.append(
            {
                "iteration": reference_steps[index],
                "mean_reward": float(series["Train/mean_reward"][index]["value"]),
                "stable_support": float(series["Episode_Reward/stable_support"][index]["value"]),
                "upright_hold": float(series["Episode_Reward/upright_hold"][index]["value"]),
                "stable_success": float(series["Episode_Termination/stable_success"][index]["value"]),
                "stable_success_once": float(series["Episode_Reward/stable_success_once"][index]["value"]),
                "hard_joint_limit": float(series["Episode_Termination/hard_joint_limit"][index]["value"]),
                "numeric_invalid": float(series["Episode_Termination/numeric_invalid"][index]["value"]),
                "curriculum_phase_index": float(series["Curriculum/recover_pose_distribution/phase_index"][index]["value"]),
                "probability_prone": float(series["Curriculum/recover_pose_distribution/probability_prone"][index]["value"]),
                "probability_left_side": float(series["Curriculum/recover_pose_distribution/probability_left_side"][index]["value"]),
                "probability_right_side": float(series["Curriculum/recover_pose_distribution/probability_right_side"][index]["value"]),
                "policy_mean_noise_std": float(series["Policy/mean_noise_std"][index]["value"]),
            }
        )

    return {
        "schema_version": "g009.r0.pilot_analysis.v2",
        "goal_id": "g009",
        "stage_number": "G009-5",
        "stage_id": "R0",
        "status": "diagnostic_complete",
        "diagnostic_only": True,
        "public_claim_eligible": False,
        "qualification_allowed": False,
        "qualification_status": "not_run",
        "qualification_block_reasons": reasons,
        "analysis_source": {
            "path": "scripts/analyze_g009_r0_pilot.py",
            "sha256": file_sha256(Path(__file__)),
        },
        "training_report": {
            "path": str(report_path.resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": file_sha256(report_path),
            "run_name": report["run_name"],
            "source_commit": report["repository"]["commit"],
            "source_bundle_sha256": report["source_bundle"]["sha256"],
        },
        "tensorboard": {
            "path": report["artifacts"]["tensorboard_directory"],
            "event_bundle_sha256": event_bundle["sha256"],
            "event_files": event_bundle["files"],
            "sample_count_per_tag": 50,
        },
        "checkpoint": {
            "path": report["artifacts"]["checkpoint"],
            "sha256": report["artifacts"]["checkpoint_sha256"],
        },
        "pilot": {"num_envs": 1024, "iterations": 50, "seed": 42, "headless": True, "scratch": True},
        "observations": {
            "hard_joint_limit": hard,
            "numeric_invalid": summaries["Episode_Termination/numeric_invalid"],
            "stable_support": summaries["Episode_Reward/stable_support"],
            "upright_hold": summaries["Episode_Reward/upright_hold"],
            "strict_stable_success": success,
            "strict_success_reward": success_reward,
            "curriculum_boundary_leak": {
                "detected": leak,
                "phase_index_maximum": phase["maximum"],
                "prone_probability_minimum": prone["minimum"],
                "left_side_probability_maximum": summaries[non_prone_tags[1]]["maximum"],
                "right_side_probability_maximum": summaries[non_prone_tags[2]]["maximum"],
            },
            "policy_mean_noise_std": summaries["Policy/mean_noise_std"],
        },
        "tail_10_iterations": tail,
        "interpretation": "부분 자세 회복 신호는 관측됐지만 엄격 성공은 없었고 관절 한계 종료와 prone 경계 누수가 있어 다음 리비전은 scratch로 진행해야 한다.",
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, default=DEFAULT_TRAINING_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = args.training_report.resolve()
    report, tensorboard_dir = validate_pilot_report(report_path)
    event_bundle_before = tensorboard_event_bundle(tensorboard_dir)
    series = load_scalar_series(tensorboard_dir)
    event_bundle_after = tensorboard_event_bundle(tensorboard_dir)
    _require(event_bundle_before == event_bundle_after, "TensorBoard event bundle changed during analysis")
    validate_recomputed_summaries(report, series)
    analysis = build_analysis(
        report,
        report_path,
        tensorboard_dir,
        series,
        event_bundle=event_bundle_before,
    )
    _write_json_atomic(args.output.resolve(), analysis)
    print(json.dumps({"status": analysis["status"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
