#!/usr/bin/env python3
"""G006S1 CPU tooling: materialize grid manifests, apply the grid rule, summarize phase B.

Subcommands
  grid       write configs/g006s1_grid_<label>.json for three ladder magnitudes
  select     read calibration reports (A1[, A2]) and print the pre-registered decision
  summarize  aggregate the six phase-B reports into reports/runs/g006s1_summary.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if "isaac_walk_g006" not in sys.modules:
    package = ModuleType("isaac_walk_g006")
    package.__path__ = [str(SRC_ROOT / "isaac_walk_g006")]
    sys.modules["isaac_walk_g006"] = package

from isaac_walk_g006.evaluation.protocol import (  # noqa: E402
    deterministic_hierarchical_paired_bootstrap,
    wilson_interval,
)
from isaac_walk_g006.sweep.push_strength import (  # noqa: E402
    PER_MAGNITUDE_TRIALS,
    REPORT_EXPERIMENTAL_USE,
    REPORT_GOAL,
    build_grid_manifest,
    grid_label,
    needs_phase_a2,
    per_magnitude_rates,
    select_comparison_grid,
    validate_grid,
)

SPEC = importlib.util.spec_from_file_location("summarize_g006_reused", REPO_ROOT / "scripts" / "summarize_g006.py")
assert SPEC and SPEC.loader
G006_SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G006_SUMMARY)

CONTRACT = REPO_ROOT / "configs" / "g006s1_push_strength_sweep.json"
RUNS = REPO_ROOT / "reports" / "runs"
VARIANTS = ("baseline", "push_curriculum")
SEEDS = (42, 43, 44)


def read_json(path: Path) -> dict[str, Any]:
    return G006_SUMMARY.read_json(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write LF bytes so committed hashes match on every checkout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def report_path(phase: str, variant: str, seed: int) -> Path:
    return RUNS / f"g006s1_{phase}_{variant}_s{seed}_push.json"


def grid_manifest_path(grid: list[float]) -> Path:
    return REPO_ROOT / "configs" / f"g006s1_grid_{grid_label(grid)}.json"


def load_report(path: Path, *, phase: str, variant: str, seed: int, grid: list[float]) -> dict[str, Any]:
    report = read_json(path)
    G006_SUMMARY.require(report.get("status") == "complete", f"{path.name} status is not complete")
    G006_SUMMARY.require(
        report.get("goal") == REPORT_GOAL and report.get("experimental_use") == REPORT_EXPERIMENTAL_USE,
        f"{path.name} is not a G006S1 report",
    )
    binding = report.get("g006s1", {})
    G006_SUMMARY.require(binding.get("phase") == phase, f"{path.name} phase mismatch")
    G006_SUMMARY.require(binding.get("evaluation_source_bundle_matches_g006") is True, f"{path.name} evaluator bundle not bound")
    G006_SUMMARY.require(
        [float(value) for value in binding.get("grid_manifest", {}).get("push_magnitudes_mps", [])] == grid,
        f"{path.name} grid mismatch",
    )
    G006_SUMMARY.require(report.get("variant") == variant and int(report.get("training_seed")) == seed, f"{path.name} identity mismatch")
    trials = report.get("trials", [])
    G006_SUMMARY.require(len(trials) == 3 * PER_MAGNITUDE_TRIALS, f"{path.name} must contain 1080 trials")
    G006_SUMMARY.require(all(not trial["protocol_blocked"] for trial in trials), f"{path.name} contains protocol-blocked trials")
    return report


def command_grid(args: argparse.Namespace) -> int:
    contract = read_json(CONTRACT)
    grid = validate_grid(args.grid, contract["calibration"]["magnitude_ladder_mps"])
    base = read_json(REPO_ROOT / contract["base"]["manifest"])
    manifest = build_grid_manifest(base, grid, sweep_sha256=G006_SUMMARY.file_sha256(CONTRACT))
    path = grid_manifest_path(grid)
    write_json(path, manifest)
    print(json.dumps({"grid_manifest": str(path.relative_to(REPO_ROOT)), "grid_mps": grid}))
    return 0


def calibration_rates(contract: dict[str, Any]) -> tuple[dict[float, float], list[dict[str, Any]]]:
    rates: dict[float, float] = {}
    evidence = []
    for run in contract["calibration"]["runs"]:
        path = report_path(run["phase"], "baseline", 42)
        if not path.is_file():
            continue
        grid = [float(value) for value in run["grid_mps"]]
        report = load_report(path, phase=run["phase"], variant="baseline", seed=42, grid=grid)
        per_mag = per_magnitude_rates(report)
        for magnitude, counts in per_mag.items():
            G006_SUMMARY.require(counts["trials"] == PER_MAGNITUDE_TRIALS, "calibration magnitude must have 360 trials")
            rates[magnitude] = counts["recovery_rate"] if counts["recovery_rate"] is not None else 0.0
        evidence.append({
            "phase": run["phase"],
            "path": "reports/runs/" + path.name,
            "sha256": G006_SUMMARY.file_sha256(path),
            "per_magnitude": {f"{key:.2f}": value for key, value in per_mag.items()},
        })
    return rates, evidence


def decide(contract: dict[str, Any]) -> dict[str, Any]:
    calibration = contract["calibration"]
    threshold = float(calibration["ceiling_threshold"])
    rates, evidence = calibration_rates(contract)
    G006_SUMMARY.require(bool(evidence), "phase A1 report is missing")
    a1_grid = [float(value) for value in calibration["runs"][0]["grid_mps"]]
    if len(evidence) == 1 and needs_phase_a2({key: rates[key] for key in a1_grid}, threshold):
        decision: dict[str, Any] = {"verdict": "run_phase_A2"}
    else:
        decision = select_comparison_grid(rates, calibration["magnitude_ladder_mps"], threshold)
    decision["ceiling_threshold"] = threshold
    decision["calibration_rates"] = {f"{key:.2f}": value for key, value in sorted(rates.items())}
    decision["calibration_evidence"] = evidence
    return decision


def command_select(args: argparse.Namespace) -> int:
    decision = decide(read_json(CONTRACT))
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


def summarize(grid: list[float]) -> dict[str, Any]:
    contract = read_json(CONTRACT)
    decision = decide(contract)
    G006_SUMMARY.require(decision.get("verdict") == "grid_selected", "calibration did not select a grid")
    G006_SUMMARY.require(decision["grid_mps"] == grid, "phase B grid differs from the pre-registered selection")
    knee = float(decision["knee_magnitude_mps"])
    reports = {
        (variant, seed): load_report(report_path("B", variant, seed), phase="B", variant=variant, seed=seed, grid=grid)
        for variant in VARIANTS
        for seed in SEEDS
    }

    per_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        pooled: dict[float, list[int]] = {}
        per_seed: dict[str, Any] = {}
        raw: dict[str, list[float]] = {key: [] for key in ("tracking_error_sq_mean", "torque_l2_mean", "absolute_mechanical_power_mean")}
        for seed in SEEDS:
            rates = per_magnitude_rates(reports[(variant, seed)])
            per_seed[str(seed)] = {
                f"{magnitude:.2f}": {**counts, "wilson95": list(wilson_interval(counts["recovered"], counts["eligible"]))}
                for magnitude, counts in rates.items()
            }
            for magnitude, counts in rates.items():
                bucket = pooled.setdefault(magnitude, [0, 0])
                bucket[0] += counts["recovered"]
                bucket[1] += counts["eligible"]
            for trial in reports[(variant, seed)]["trials"]:
                for key in raw:
                    if trial.get(key) is not None:
                        raw[key].append(float(trial[key]))
        total_recovered = sum(value[0] for value in pooled.values())
        total_eligible = sum(value[1] for value in pooled.values())
        per_variant[variant] = {
            "recovered": total_recovered,
            "eligible": total_eligible,
            "recovery_rate": total_recovered / total_eligible,
            "wilson95": list(wilson_interval(total_recovered, total_eligible)),
            "per_magnitude": {
                f"{magnitude:.2f}": {
                    "recovered": counts[0],
                    "eligible": counts[1],
                    "recovery_rate": None if counts[1] == 0 else counts[0] / counts[1],
                    "wilson95": list(wilson_interval(counts[0], counts[1])),
                }
                for magnitude, counts in sorted(pooled.items())
            },
            "per_seed": per_seed,
            "raw_metrics": {key: (math.fsum(values) / len(values) if values else None) for key, values in raw.items()},
        }

    def paired_for(seeds: tuple[int, ...]) -> dict[int, dict[str, list[float]]]:
        return G006_SUMMARY.build_paired_recovery_deltas(
            {seed: reports[("baseline", seed)]["trials"] for seed in seeds},
            {seed: reports[("push_curriculum", seed)]["trials"] for seed in seeds},
        )

    bootstrap_seed = int(contract["base"].get("bootstrap_seed", 20260824))
    primary = deterministic_hierarchical_paired_bootstrap(paired_for(SEEDS), bootstrap_seed=bootstrap_seed, draws=10_000)
    without_calibration = deterministic_hierarchical_paired_bootstrap(paired_for((43, 44)), bootstrap_seed=bootstrap_seed, draws=10_000)
    threshold = float(contract["calibration"]["ceiling_threshold"])
    knee_rate = per_variant["baseline"]["per_magnitude"][f"{knee:.2f}"]["recovery_rate"]
    ci_low, ci_high = primary["ci95"]
    direction = "no_detectable_difference"
    if ci_low > 0:
        direction = "push_curriculum_higher"
    elif ci_high < 0:
        direction = "baseline_higher"
    return {
        "schema_version": 1,
        "goal": REPORT_GOAL,
        "status": "complete",
        "contract": {"path": "configs/g006s1_push_strength_sweep.json", "sha256": G006_SUMMARY.file_sha256(CONTRACT)},
        "calibration_decision": decision,
        "grid_mps": grid,
        "reports": {
            f"{variant}-s{seed}": {
                "path": "reports/runs/" + report_path("B", variant, seed).name,
                "sha256": G006_SUMMARY.file_sha256(report_path("B", variant, seed)),
                "checkpoint_sha256": reports[(variant, seed)]["checkpoint"]["sha256"],
            }
            for variant in VARIANTS
            for seed in SEEDS
        },
        "recovery": per_variant,
        "ceiling_exit": {
            "knee_magnitude_mps": knee,
            "pooled_baseline_rate_at_knee": knee_rate,
            "threshold": threshold,
            "confirmed": knee_rate <= threshold,
        },
        "paired_delta_push_curriculum_minus_baseline": {
            "primary_all_seeds": primary,
            "secondary_seeds_43_44": without_calibration,
            "direction_rule_result": direction,
        },
        "warnings": [
            "Evaluation-only analysis of existing G006 checkpoints; no policy was retrained.",
            "Three training seeds per variant; seed 42 baseline also selected the grid.",
            "Push magnitudes are instantaneous simulated root velocity deltas, not calibrated real forces.",
            "Mechanical power is a simulation proxy and is not electrical energy consumption.",
        ],
    }


def command_summarize(args: argparse.Namespace) -> int:
    summary = summarize([float(value) for value in args.grid])
    write_json(RUNS / "g006s1_summary.json", summary)
    print(json.dumps({
        "ceiling_exit": summary["ceiling_exit"],
        "rates": {variant: summary["recovery"][variant]["recovery_rate"] for variant in VARIANTS},
        "paired": summary["paired_delta_push_curriculum_minus_baseline"]["primary_all_seeds"]["ci95"],
        "direction": summary["paired_delta_push_curriculum_minus_baseline"]["direction_rule_result"],
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    grid = sub.add_parser("grid")
    grid.add_argument("--grid", required=True, nargs=3, type=float)
    grid.set_defaults(func=command_grid)
    select = sub.add_parser("select")
    select.set_defaults(func=command_select)
    summary = sub.add_parser("summarize")
    summary.add_argument("--grid", required=True, nargs=3, type=float)
    summary.set_defaults(func=command_summarize)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
