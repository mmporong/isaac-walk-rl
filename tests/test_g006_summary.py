from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PACKAGE = ModuleType("isaac_walk_g006")
PACKAGE.__path__ = [str(ROOT / "src" / "isaac_walk_g006")]
sys.modules.setdefault("isaac_walk_g006", PACKAGE)

from isaac_walk_g006.evaluation.protocol import (  # noqa: E402
    EXPECTED_SUCCESS_CRITERIA,
    compute_evaluation_source_bundle,
    deterministic_hierarchical_paired_bootstrap,
)

SPEC = importlib.util.spec_from_file_location("summarize_g006_tested", ROOT / "scripts" / "summarize_g006.py")
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def complete_evidence_fixture(tmp_path: Path, queue_status: str) -> tuple[Path, Path]:
    manifest = json.loads((ROOT / "configs" / "g006_rough_push.json").read_text(encoding="utf-8"))
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    protocol_hash = SUMMARY.canonical_sha256(manifest["evaluation_protocol"])

    training_entrypoint = ROOT / "scripts" / "bootstrap_train_g006.py"
    evaluator = ROOT / "scripts" / "evaluate_push_recovery.py"
    training_files = [{
        "path": "scripts/bootstrap_train_g006.py",
        "sha256": SUMMARY.file_sha256(training_entrypoint),
    }]
    training_bundle = SUMMARY.compute_declared_source_bundle(training_files)
    evaluation_bundle = compute_evaluation_source_bundle(ROOT)
    jobs = []
    tasks = {item["name"]: item["task"] for item in manifest["variants"]}
    for variant in ("baseline", "push_curriculum"):
        for seed in (42, 43, 44):
            checkpoint = tmp_path / f"{variant}-s{seed}.pt"
            checkpoint.write_bytes(f"checkpoint:{variant}:{seed}".encode())
            checkpoint_hash = SUMMARY.file_sha256(checkpoint)
            training_report_path = tmp_path / f"training-{variant}-s{seed}.json"
            write_json(training_report_path, {
                "passed": True,
                "task": tasks[variant],
                "seed": seed,
                "training_source_bundle_sha256": training_bundle["sha256"],
                "training_entrypoint": {"sha256": SUMMARY.file_sha256(training_entrypoint)},
                "artifacts": {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                },
            })
            report_paths = {}
            for mode in ("push", "guardrail"):
                report_path = tmp_path / f"eval-{variant}-s{seed}-{mode}.json"
                write_json(report_path, eval_report(variant, seed, mode, protocol_hash, checkpoint_hash))
                report_paths[mode] = report_path
            jobs.append({
                "id": f"{variant}-s{seed}",
                "variant": variant,
                "seed": seed,
                "task": tasks[variant],
                "status": "complete",
                "report_path": str(training_report_path),
                "checkpoint_sha256": checkpoint_hash,
                "training_source_bundle_sha256": training_bundle["sha256"],
                "evaluation_source_bundle_sha256": evaluation_bundle["sha256"],
                "push_report_path": str(report_paths["push"]),
                "guardrail_report_path": str(report_paths["guardrail"]),
                "push_script_sha256": SUMMARY.file_sha256(evaluator),
                "guardrail_script_sha256": SUMMARY.file_sha256(evaluator),
            })
    queue_path = tmp_path / "queue.json"
    write_json(queue_path, {
        "goal": "G006",
        "mode": "production",
        "status": queue_status,
        "config_sha256": SUMMARY.canonical_sha256(manifest),
        "protocol_sha256": protocol_hash,
        "training_entrypoint_sha256": SUMMARY.file_sha256(training_entrypoint),
        "training_source_bundle_sha256": training_bundle["sha256"],
        "evaluation_source_bundle_sha256": evaluation_bundle["sha256"],
        "source_bundles": {
            "training": training_bundle,
            "evaluation": evaluation_bundle,
        },
        "jobs": jobs,
    })
    return manifest_path, queue_path


def terrain_evidence() -> dict:
    entries = []
    for col in range(10):
        for row, value in ((1, 1.0), (4, 2.0), (8, 3.0)):
            entries.append({
                "row": row,
                "col": col,
                "raw_sha256": f"{len(entries) + 1:064x}",
                "mesh_sha256": f"{len(entries) + 101:064x}",
                "metrics": {
                    "height_rms_m": value,
                    "height_p90_abs_m": value,
                    "face_normal_slope_rms_rad": value,
                    "face_normal_slope_p90_rad": value,
                },
            })
    return {"selected_tiles": entries, "difficulty_aggregates": {}}


def eval_report(variant: str, seed: int, mode: str, protocol_hash: str, checkpoint_hash: str) -> dict:
    count = 1080 if mode == "push" else 90
    trials = []
    for index in range(count):
        stratum = f"s{index // 10:03d}" if mode == "push" else f"g{index // 10:02d}"
        recovered = mode == "push" and variant == "push_curriculum"
        trials.append({
            "trial_id": f"{variant}-s{seed}:{mode}:{index}",
            "pair_id": f"{variant}-s{seed}",
            "paired_trial_key": f"{mode}:{index}",
            "stratum_id": stratum,
            "eligible": mode == "push",
            "criterion_met": recovered,
            "recovered": recovered,
            "guardrail_survived": True,
            "guardrail_eligible": True,
            "failed": mode == "push" and not recovered,
            "recovery_failed": mode == "push" and not recovered,
            "prepush_failure": False,
            "survived_to_horizon": True,
            "physical_failure": False,
            "protocol_blocked": False,
            "recovery_step": 225 if recovered else None,
            "excluded_reason": None,
            "tracking_error_sq_mean": 0.1,
            "yaw_error_sq_mean": 0.1,
            "torque_l2_mean": 1.0,
            "absolute_mechanical_power_mean": 1.0,
            "action_rate_l2_mean": 0.1,
        })
    cell_count = 108 if mode == "push" else 9
    bundle = compute_evaluation_source_bundle(ROOT)
    return {
        "schema_version": 1,
        "goal": "G006",
        "status": "complete",
        "protocol_compliant": True,
        "experimental_use": "g006_production_evaluation",
        "mode": mode,
        "variant": variant,
        "training_seed": seed,
        "protocol": {"sha256": protocol_hash},
        "checkpoint": {"sha256": checkpoint_hash},
        "evaluation_source_bundle_sha256": bundle["sha256"],
        "evaluation_source_bundle_files": bundle["files"],
        "success_criteria": dict(EXPECTED_SUCCESS_CRITERIA),
        "runtime": {
            "task": "Isaac-Velocity-Rough-Unitree-Go2-v0",
            "terrain_levels_runtime": None,
            "observation_corruption": False,
            "events_enabled": [],
            "base_contact_threshold_n": 1.0,
            "push_injection_completed_steps": 200,
            "horizon_completed_step": 600,
            "preliminary": False,
            "exit_code": 0,
            "app_close_completed": True,
            "finalized_after_process_exit": True,
            "gpu_measurement_complete": True,
            "process_recovered": True,
            "gpu_recovered_to_baseline": True,
            "fatal_patterns": [],
        },
        "terrain_evidence": terrain_evidence(),
        "trials": trials,
        "cells": [{"cell_id": f"c{index}", "trial_count": 10, "raw_metrics": {}} for index in range(cell_count)],
        "aggregate": {
            "trial_count": count,
            "eligible_count": count if mode == "push" else 0,
            "criterion_met_count": count if recovered else 0,
            "recovered_count": count if recovered else 0,
            "survived_to_horizon_count": count,
            "boundary_violation_count": 0,
            "auto_reset_excluded_count": 0,
        },
    }


def test_eval_schema_detects_pair_mismatch() -> None:
    report = eval_report("baseline", 42, "push", "a" * 64, "b" * 64)
    SUMMARY.validate_eval_report(report, mode="push", variant="baseline", seed=42, protocol_hash="a" * 64, checkpoint_hash="b" * 64)
    report["trials"][0]["pair_id"] = "wrong"
    with pytest.raises(SUMMARY.ValidationError, match="pair IDs"):
        SUMMARY.validate_eval_report(report, mode="push", variant="baseline", seed=42, protocol_hash="a" * 64, checkpoint_hash="b" * 64)


def test_eval_schema_blocks_tile_violation() -> None:
    report = eval_report("baseline", 42, "guardrail", "a" * 64, "b" * 64)
    report["aggregate"]["boundary_violation_count"] = 1
    with pytest.raises(SUMMARY.ValidationError, match="boundary"):
        SUMMARY.validate_eval_report(report, mode="guardrail", variant="baseline", seed=42, protocol_hash="a" * 64, checkpoint_hash="b" * 64)


def test_summary_rejects_recovery_false_positive_without_horizon_survival() -> None:
    report = eval_report("push_curriculum", 42, "push", "a" * 64, "b" * 64)
    report["trials"][0]["criterion_met"] = True
    report["trials"][0]["survived_to_horizon"] = False
    report["trials"][0]["recovered"] = True
    report["trials"][0]["physical_failure"] = True
    with pytest.raises(SUMMARY.ValidationError, match="criterion and horizon survival"):
        SUMMARY.validate_eval_report(report, mode="push", variant="push_curriculum", seed=42, protocol_hash="a" * 64, checkpoint_hash="b" * 64)


def test_summary_rejects_auto_reset_protocol_block() -> None:
    report = eval_report("baseline", 42, "push", "a" * 64, "b" * 64)
    report["trials"][0]["protocol_blocked"] = True
    report["trials"][0]["excluded_reason"] = "auto_reset_poison"
    report["aggregate"]["auto_reset_excluded_count"] = 1
    with pytest.raises(SUMMARY.ValidationError, match="protocol-blocked"):
        SUMMARY.validate_eval_report(report, mode="push", variant="baseline", seed=42, protocol_hash="a" * 64, checkpoint_hash="b" * 64)


def test_cross_variant_pair_key_mismatch_is_rejected() -> None:
    baseline = eval_report("baseline", 42, "push", "a" * 64, "b" * 64)["trials"]
    curriculum = eval_report("push_curriculum", 42, "push", "a" * 64, "b" * 64)["trials"]
    curriculum[0]["paired_trial_key"] = "tampered"
    with pytest.raises(SUMMARY.ValidationError, match="paired trial mismatch"):
        SUMMARY.build_paired_recovery_deltas({42: baseline}, {42: curriculum})


def test_bootstrap_known_constant_fixture_is_exact_and_deterministic() -> None:
    paired = {
        seed: {f"s{stratum:03d}": [0.25] * 10 for stratum in range(108)}
        for seed in (42, 43, 44)
    }
    first = deterministic_hierarchical_paired_bootstrap(paired, bootstrap_seed=20260824, draws=10_000)
    second = deterministic_hierarchical_paired_bootstrap(paired, bootstrap_seed=20260824, draws=10_000)
    assert first == second
    assert first["estimate"] == 0.25
    assert first["ci95"] == [0.25, 0.25]


def test_real_summarizer_accepts_pre_final_summarizing_queue(tmp_path: Path) -> None:
    manifest_path, queue_path = complete_evidence_fixture(tmp_path, "summarizing")

    result = SUMMARY.summarize(manifest_path, queue_path)

    assert result["status"] == "complete"
    assert len(result["jobs"]) == 6
    assert result["comparisons"]["recovery"]["baseline"]["total"] == 3240
    assert result["comparisons"]["recovery"]["push_curriculum"]["successes"] == 3240


def test_real_summarizer_rejects_failed_queue_state(tmp_path: Path) -> None:
    manifest_path, queue_path = complete_evidence_fixture(tmp_path, "failed")

    with pytest.raises(SUMMARY.ValidationError, match="summarizing or complete"):
        SUMMARY.summarize(manifest_path, queue_path)
