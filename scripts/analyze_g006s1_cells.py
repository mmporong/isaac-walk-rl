#!/usr/bin/env python3
"""Break G006S1 phase-B recovery down by command, push direction, terrain row, magnitude and failure kind.

Reads only the committed phase-B reports; writes reports/runs/g006s1_cell_analysis.json (LF bytes).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "reports" / "runs"
VARIANTS = ("baseline", "push_curriculum")
SEEDS = (42, 43, 44)
AXES = ("command_id", "push_direction_id", "terrain_row", "push_magnitude_mps")


def failure_kind(trial: dict[str, Any]) -> str:
    if trial["recovered"]:
        return "recovered"
    return "fell_before_horizon" if not trial["survived_to_horizon"] else "survived_without_recovery"


def rate_table(trials: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for trial in trials:
        label = "|".join(str(trial[key]) for key in keys)
        buckets[label][1] += 1
        buckets[label][0] += int(bool(trial["recovered"]))
    return {
        label: {"recovered": recovered, "trials": total, "recovery_rate": recovered / total}
        for label, (recovered, total) in sorted(buckets.items())
    }


def analyze(runs: Path = RUNS) -> dict[str, Any]:
    reports: dict[tuple[str, int], list[dict[str, Any]]] = {}
    sources = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            path = runs / f"g006s1_B_{variant}_s{seed}_push.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("status") != "complete" or len(report["trials"]) != 1080:
                raise ValueError(f"{path.name} is not a complete 1080-trial report")
            reports[(variant, seed)] = report["trials"]
            sources[f"{variant}-s{seed}"] = {"path": "reports/runs/" + path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    per_checkpoint = {}
    for (variant, seed), trials in reports.items():
        kinds: dict[str, int] = defaultdict(int)
        for trial in trials:
            kinds[failure_kind(trial)] += 1
        per_checkpoint[f"{variant}-s{seed}"] = {
            "by_axis": {axis: rate_table(trials, (axis,)) for axis in AXES},
            "direction_x_magnitude": rate_table(trials, ("push_direction_id", "push_magnitude_mps")),
            "command_x_direction": rate_table(trials, ("command_id", "push_direction_id")),
            "failure_kind": dict(sorted(kinds.items())),
        }

    pooled = {}
    for variant in VARIANTS:
        trials = [trial for seed in SEEDS for trial in reports[(variant, seed)]]
        pooled[variant] = {
            "by_axis": {axis: rate_table(trials, (axis,)) for axis in AXES},
            "direction_x_magnitude": rate_table(trials, ("push_direction_id", "push_magnitude_mps")),
        }

    # Spread of per-seed rates within a variant, per axis level: how much of the variance a factor carries.
    seed_spread = {}
    for variant in VARIANTS:
        seed_spread[variant] = {}
        for axis in AXES:
            levels = per_checkpoint[f"{variant}-s{SEEDS[0]}"]["by_axis"][axis]
            seed_spread[variant][axis] = {
                level: {
                    "min": min(per_checkpoint[f"{variant}-s{seed}"]["by_axis"][axis][level]["recovery_rate"] for seed in SEEDS),
                    "max": max(per_checkpoint[f"{variant}-s{seed}"]["by_axis"][axis][level]["recovery_rate"] for seed in SEEDS),
                }
                for level in levels
            }

    return {
        "schema_version": 1,
        "goal": "G006S1",
        "analysis": "phase-B cell breakdown (descriptive, no new simulation)",
        "sources": sources,
        "per_checkpoint": per_checkpoint,
        "pooled": pooled,
        "seed_spread": seed_spread,
        "warnings": [
            "Descriptive breakdown of existing trials; each cell has 30-360 trials and no multiple-comparison correction.",
            "Push directions are in the robot body frame at injection time.",
        ],
    }


def main() -> int:
    result = analyze()
    output = RUNS / "g006s1_cell_analysis.json"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": "reports/runs/" + output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
