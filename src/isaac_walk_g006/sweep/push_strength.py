"""Pure helpers for the G006S1 push-strength sweep (evaluation of existing G006 checkpoints).

Nothing here imports Isaac Sim. The evaluator itself is reused unchanged; this module only
derives grid manifests, applies the pre-registered grid selection rule, and aggregates reports.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

REPORT_GOAL = "G006S1"
REPORT_EXPERIMENTAL_USE = "g006s1_push_strength_sweep"
PER_MAGNITUDE_TRIALS = 360


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def grid_label(grid: Sequence[float]) -> str:
    return "_".join(f"m{float(value):.2f}".replace(".", "p") for value in grid)


def validate_grid(grid: Sequence[float], ladder: Sequence[float]) -> list[float]:
    values = [float(value) for value in grid]
    if len(values) != 3 or len(set(values)) != 3 or values != sorted(values):
        raise ValueError("grid must be three strictly increasing magnitudes")
    if any(value not in [float(item) for item in ladder] for value in values):
        raise ValueError("grid magnitudes must come from the registered ladder")
    return values


def build_grid_manifest(base_manifest: Mapping[str, Any], grid: Sequence[float], *, sweep_sha256: str) -> dict[str, Any]:
    """Copy the G006 manifest and change only evaluation_protocol.push_magnitudes_mps."""

    derived = copy.deepcopy(dict(base_manifest))
    protocol = derived.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("base manifest requires evaluation_protocol")
    base_protocol_sha256 = canonical_sha256(protocol)
    protocol["push_magnitudes_mps"] = [float(value) for value in grid]
    derived["goal"] = REPORT_GOAL
    derived["derived_from"] = {
        "manifest": "configs/g006_rough_push.json",
        "protocol_sha256": base_protocol_sha256,
        "sweep_contract_sha256": sweep_sha256,
        "changed_keys": ["goal", "evaluation_protocol.push_magnitudes_mps"],
    }
    return derived


def protocol_diff_keys(base_protocol: Mapping[str, Any], derived_protocol: Mapping[str, Any]) -> list[str]:
    keys = set(base_protocol) | set(derived_protocol)
    return sorted(key for key in keys if base_protocol.get(key) != derived_protocol.get(key))


def per_magnitude_rates(report: Mapping[str, Any]) -> dict[float, dict[str, Any]]:
    """Recovery counts per push magnitude from one evaluator report."""

    buckets: dict[float, list[int]] = {}
    for trial in report["trials"]:
        magnitude = float(trial["push_magnitude_mps"])
        counts = buckets.setdefault(magnitude, [0, 0])
        if trial["eligible"]:
            counts[1] += 1
            counts[0] += int(bool(trial["recovered"]))
    result = {}
    for magnitude, (recovered, eligible) in sorted(buckets.items()):
        result[magnitude] = {
            "recovered": recovered,
            "eligible": eligible,
            "trials": sum(1 for trial in report["trials"] if float(trial["push_magnitude_mps"]) == magnitude),
            "recovery_rate": None if eligible == 0 else recovered / eligible,
        }
    return result


def select_comparison_grid(
    calibration_rates: Mapping[float, float], ladder: Sequence[float], ceiling_threshold: float
) -> dict[str, Any]:
    """Apply the pre-registered rule. calibration_rates must cover a ladder prefix."""

    values = [float(item) for item in ladder]
    if len(values) < 3 or values != sorted(values):
        raise ValueError("ladder must be increasing with at least three values")
    measured = [value for value in values if value in calibration_rates]
    if measured != values[: len(measured)]:
        raise ValueError("calibration must cover a contiguous ladder prefix")
    knee_index = next(
        (index for index, value in enumerate(measured) if float(calibration_rates[value]) <= ceiling_threshold),
        None,
    )
    if knee_index is None:
        if len(measured) < len(values):
            return {"verdict": "need_more_calibration", "next_start_index": len(measured)}
        return {"verdict": "no_knee_within_ladder"}
    start = max(0, min(knee_index - 1, len(values) - 3))
    return {
        "verdict": "grid_selected",
        "knee_index": knee_index,
        "knee_magnitude_mps": values[knee_index],
        "grid_mps": values[start : start + 3],
    }


def needs_phase_a2(a1_rates: Mapping[float, float], ceiling_threshold: float) -> bool:
    return all(float(rate) > ceiling_threshold for rate in a1_rates.values())
