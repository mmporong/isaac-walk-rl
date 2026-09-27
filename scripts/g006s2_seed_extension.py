#!/usr/bin/env python3
"""G006S2 CPU tooling: training source bundle check, per-run training gate, five-seed summary.

Subcommands
  bundle     print the current G006 training source bundle and whether it equals the contract
  gate       check one seed-extension training report against the contract gates
  summarize  aggregate seeds 42-46 at the G006S1 grid into reports/runs/g006s2_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
from isaac_walk_g006.sweep.push_strength import per_magnitude_rates  # noqa: E402

SPEC = importlib.util.spec_from_file_location("summarize_g006_for_s2", REPO_ROOT / "scripts" / "summarize_g006.py")
assert SPEC and SPEC.loader
G006_SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G006_SUMMARY)

CONTRACT = REPO_ROOT / "configs" / "g006s2_seed_extension.json"
RUNS = REPO_ROOT / "reports" / "runs"
VARIANTS = ("baseline", "push_curriculum")
require = G006_SUMMARY.require


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def training_bundle(queue: dict[str, Any]) -> dict[str, Any]:
    return G006_SUMMARY.compute_declared_source_bundle(queue["source_bundles"]["training"]["files"])


def expected_command(g006_command: list[Any], seed: int, run_name: str) -> list[str]:
    """G006 command with seed/run_name substituted and the documented explicit device inserted."""

    command = [str(item) for item in g006_command]
    command[command.index("--seed") + 1] = str(seed)
    command[command.index("--run_name") + 1] = run_name
    seed_index = command.index("--seed")
    return command[: seed_index + 2] + ["--device", "cuda:0"] + command[seed_index + 2 :]


def normalize_command(command: list[Any]) -> list[str]:
    """Compare everything except the repository location of the entrypoint (its bytes are hashed separately)."""

    values = [str(item) for item in command]
    values[1] = Path(values[1].replace("\\", "/")).name
    return values


def gate(variant: str, seed: int, *, runs: Path = RUNS, stdout_text: str | None = None) -> dict[str, Any]:
    contract = read_json(CONTRACT)
    queue = read_json(REPO_ROOT / contract["base"]["queue_state"])
    entry = next(item for item in contract["training_reports"] if item["variant"] == variant and item["seed"] == seed)
    report_path = runs / Path(entry["path"]).name
    report = read_json(report_path)
    g006_job = next(job for job in queue["jobs"] if job["variant"] == variant and int(job["seed"]) == 42)
    g006_training = read_json(RUNS / f"g006_production_{variant}_e4096_i1500_s42.json")
    run_name = contract["training"]["run_name"].format(variant=variant, seed=seed)
    if stdout_text is None:
        stdout_path = Path(report["artifacts"]["raw_stdout"].replace("%USERPROFILE%", str(Path.home())))
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
    bundle = training_bundle(queue)
    checks = {
        "harness_passed": report.get("passed") is True,
        "run_name": report.get("run_name") == run_name,
        "task": report.get("task") == contract["training"]["tasks"][variant] == g006_job["task"],
        "budget": report.get("num_envs") == 4096 and report.get("max_iterations") == 1500 and report.get("seed") == seed,
        "no_hydra_overrides": list(report.get("effective_hydra_overrides") or []) == [],
        "command_matches_g006": normalize_command(report.get("command", [])) == normalize_command(
            expected_command(g006_training["command"], seed, run_name)
        ),
        "entrypoint_sha256": report.get("training_entrypoint", {}).get("sha256") == contract["base"]["training_entrypoint_sha256"],
        "training_source_bundle": bundle["sha256"] == contract["base"]["training_source_bundle_sha256"],
        "device_cuda0_in_stdout": "Using device: cuda:0" in stdout_text,
        "peak_vram_within_limit": int(report.get("gpu", {}).get("peak_used_mib", 10**9)) <= int(contract["resource_gates"]["vram_limit_mib"]),
    }
    return {
        "variant": variant,
        "seed": seed,
        "report": {"path": "reports/runs/" + report_path.name, "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()},
        "checkpoint_sha256": report.get("artifacts", {}).get("checkpoint_sha256"),
        "training_source_bundle_sha256": bundle["sha256"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def load_eval(path: Path, variant: str, seed: int, grid: list[float]) -> dict[str, Any]:
    report = read_json(path)
    require(report.get("status") == "complete", f"{path.name} not complete")
    require(report.get("variant") == variant and int(report.get("training_seed")) == seed, f"{path.name} identity mismatch")
    require(report["g006s1"]["evaluation_source_bundle_matches_g006"] is True, f"{path.name} evaluator not bound")
    require([float(value) for value in report["g006s1"]["grid_manifest"]["push_magnitudes_mps"]] == grid, f"{path.name} grid mismatch")
    require(len(report["trials"]) == 1080 and not any(trial["protocol_blocked"] for trial in report["trials"]), f"{path.name} trials invalid")
    return report


def eval_path(variant: str, seed: int, runs: Path) -> Path:
    if seed in (42, 43, 44):
        return runs / f"g006s1_B_{variant}_s{seed}_push.json"
    return runs / f"g006s2_S2_{variant}_s{seed}_push.json"


def summarize(runs: Path = RUNS) -> dict[str, Any]:
    contract = read_json(CONTRACT)
    seeds = tuple(int(seed) for seed in contract["design"]["seeds"])
    grid = [float(value) for value in contract["evaluation"]["grid_mps"]]
    reports = {(variant, seed): load_eval(eval_path(variant, seed, runs), variant, seed, grid) for variant in VARIANTS for seed in seeds}
    protocol_hashes = {report["protocol"]["sha256"] for report in reports.values()}
    require(len(protocol_hashes) == 1, "evaluation protocol differs between seed groups")
    for seed in (45, 46):
        for variant in VARIANTS:
            require(reports[(variant, seed)].get("goal") == contract["goal"], f"{variant}-s{seed} is not a G006S2 report")

    recovery: dict[str, Any] = {}
    for variant in VARIANTS:
        pooled: dict[float, list[int]] = {}
        direction: dict[str, list[int]] = {}
        per_seed = {}
        for seed in seeds:
            trials = reports[(variant, seed)]["trials"]
            rates = per_magnitude_rates(reports[(variant, seed)])
            per_seed[str(seed)] = {f"{m:.2f}": c["recovery_rate"] for m, c in rates.items()}
            for magnitude, counts in rates.items():
                bucket = pooled.setdefault(magnitude, [0, 0])
                bucket[0] += counts["recovered"]
                bucket[1] += counts["eligible"]
            for trial in trials:
                bucket = direction.setdefault(trial["push_direction_id"], [0, 0])
                bucket[0] += int(bool(trial["recovered"]))
                bucket[1] += 1
        recovered = sum(value[0] for value in pooled.values())
        total = sum(value[1] for value in pooled.values())

        def table(buckets: dict[Any, list[int]]) -> dict[str, Any]:
            return {
                (f"{key:.2f}" if isinstance(key, float) else str(key)): {
                    "recovered": value[0],
                    "trials": value[1],
                    "recovery_rate": value[0] / value[1],
                    "wilson95": list(wilson_interval(value[0], value[1])),
                }
                for key, value in sorted(buckets.items())
            }

        recovery[variant] = {
            "recovered": recovered,
            "trials": total,
            "recovery_rate": recovered / total,
            "wilson95": list(wilson_interval(recovered, total)),
            "per_magnitude": table(pooled),
            "per_direction": table(direction),
            "per_seed": per_seed,
        }

    def paired(selected: tuple[int, ...]) -> dict[str, Any]:
        deltas = G006_SUMMARY.build_paired_recovery_deltas(
            {seed: reports[("baseline", seed)]["trials"] for seed in selected},
            {seed: reports[("push_curriculum", seed)]["trials"] for seed in selected},
        )
        return deterministic_hierarchical_paired_bootstrap(deltas, bootstrap_seed=20260824, draws=10_000)

    primary = paired(seeds)
    new_only = paired((45, 46))
    low, high = primary["ci95"]
    direction_result = "push_curriculum_higher" if low > 0 else "baseline_higher" if high < 0 else "no_detectable_difference"
    return {
        "schema_version": 1,
        "goal": contract["goal"],
        "status": "complete",
        "contract": {"path": "configs/" + CONTRACT.name, "sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest()},
        "seeds": list(seeds),
        "grid_mps": grid,
        "protocol_sha256": protocol_hashes.pop(),
        "reports": {
            f"{variant}-s{seed}": {
                "path": "reports/runs/" + eval_path(variant, seed, runs).name,
                "sha256": hashlib.sha256(eval_path(variant, seed, runs).read_bytes()).hexdigest(),
                "checkpoint_sha256": reports[(variant, seed)]["checkpoint"]["sha256"],
            }
            for variant in VARIANTS
            for seed in seeds
        },
        "recovery": recovery,
        "paired_delta_push_curriculum_minus_baseline": {
            "primary_seeds_42_46": primary,
            "secondary_seeds_45_46": new_only,
            "direction_rule_result": direction_result,
        },
        "warnings": [
            "Both variants trained with pushes of at most 1.0 m/s; the evaluation grid is outside both training distributions.",
            "Simulation-only; push magnitudes are instantaneous root velocity deltas.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("bundle")
    gate_parser = sub.add_parser("gate")
    gate_parser.add_argument("--variant", required=True, choices=VARIANTS)
    gate_parser.add_argument("--seed", required=True, type=int)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "bundle":
        contract = read_json(CONTRACT)
        bundle = training_bundle(read_json(REPO_ROOT / contract["base"]["queue_state"]))
        print(json.dumps({"sha256": bundle["sha256"], "matches": bundle["sha256"] == contract["base"]["training_source_bundle_sha256"]}))
        return 0 if bundle["sha256"] == contract["base"]["training_source_bundle_sha256"] else 1
    if args.command == "gate":
        result = gate(args.variant, args.seed)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["passed"] else 1
    summary = summarize()
    write_json(RUNS / "g006s2_summary.json", summary)
    paired_result = summary["paired_delta_push_curriculum_minus_baseline"]
    print(json.dumps({
        "rates": {variant: summary["recovery"][variant]["recovery_rate"] for variant in VARIANTS},
        "primary": paired_result["primary_seeds_42_46"]["ci95"],
        "direction": paired_result["direction_rule_result"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
