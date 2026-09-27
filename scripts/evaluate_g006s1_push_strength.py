#!/usr/bin/env python3
"""Run the unchanged G006 push evaluator on one checkpoint for the G006S1 strength sweep.

Pre-launch gates: frozen evaluator bundle hash, checkpoint hash from the G006 queue, and a grid
manifest that differs from the G006 protocol only in push_magnitudes_mps. The evaluator report is
relabelled as G006S1 so it can never be mistaken for G006 production evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
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

from isaac_walk_g006.evaluation.protocol import compute_evaluation_source_bundle  # noqa: E402
from isaac_walk_g006.sweep.push_strength import (  # noqa: E402
    REPORT_EXPERIMENTAL_USE,
    REPORT_GOAL,
    canonical_sha256,
    protocol_diff_keys,
)

SPEC = importlib.util.spec_from_file_location("g006_evaluator", REPO_ROOT / "scripts" / "evaluate_push_recovery.py")
assert SPEC and SPEC.loader
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def preflight(sweep_path: Path, grid_manifest_path: Path, checkpoint: Path, variant: str, seed: int) -> dict[str, Any]:
    sweep = read_json(sweep_path)
    base = read_json(REPO_ROOT / sweep["base"]["manifest"])
    queue = read_json(REPO_ROOT / sweep["base"]["queue_state"])
    grid_manifest = read_json(grid_manifest_path)

    base_protocol_sha = canonical_sha256(base["evaluation_protocol"])
    if base_protocol_sha != sweep["base"]["protocol_sha256"] or queue["protocol_sha256"] != base_protocol_sha:
        raise RuntimeError("G006 base protocol hash mismatch")
    diff = protocol_diff_keys(base["evaluation_protocol"], grid_manifest["evaluation_protocol"])
    if diff != ["push_magnitudes_mps"]:
        raise RuntimeError(f"grid manifest must change only push_magnitudes_mps, changed={diff}")
    if grid_manifest.get("derived_from", {}).get("sweep_contract_sha256") != EVALUATOR.file_sha256(sweep_path):
        raise RuntimeError("grid manifest was not derived from the current sweep contract")

    bundle = compute_evaluation_source_bundle(REPO_ROOT)
    if bundle["sha256"] != sweep["base"]["evaluation_source_bundle_sha256"]:
        raise RuntimeError("evaluation source bundle differs from G006")

    job = next(
        (item for item in queue["jobs"] if item["variant"] == variant and int(item["seed"]) == seed),
        None,
    )
    if job is None:
        raise RuntimeError("checkpoint job missing from G006 queue")
    checkpoint_sha = EVALUATOR.file_sha256(checkpoint)
    if checkpoint_sha != job["checkpoint_sha256"]:
        raise RuntimeError("checkpoint sha256 differs from G006 queue")
    return {
        "sweep_contract": {"path": "configs/" + sweep_path.name, "sha256": EVALUATOR.file_sha256(sweep_path)},
        "grid_manifest": {
            "path": "configs/" + grid_manifest_path.name,
            "sha256": EVALUATOR.file_sha256(grid_manifest_path),
            "push_magnitudes_mps": grid_manifest["evaluation_protocol"]["push_magnitudes_mps"],
        },
        "base_protocol_sha256": base_protocol_sha,
        "evaluation_source_bundle_matches_g006": True,
        "checkpoint_matches_g006_queue": job["id"],
        "wrapper_sha256": EVALUATOR.file_sha256(Path(__file__).resolve()),
    }


def split_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sweep-contract", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("A1", "A2", "B"))
    own, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return own


def main() -> int:
    own = split_args()
    args = EVALUATOR.parse_args()
    started_at = time.time()
    try:
        if args.mode != "push":
            raise ValueError("G006S1 evaluates push mode only")
        binding = preflight(
            own.sweep_contract.resolve(), args.protocol.resolve(), args.checkpoint.resolve(), args.variant, args.training_seed
        )
        from isaaclab.app import AppLauncher

        app = AppLauncher(args).app
        try:
            report = EVALUATOR.evaluate(args)
        finally:
            app.close()
        report["goal"] = REPORT_GOAL
        report["experimental_use"] = REPORT_EXPERIMENTAL_USE
        report["g006s1"] = {"phase": own.phase, **binding}
        report["runtime"]["started_at_epoch"] = started_at
        report["runtime"]["finished_at_epoch"] = time.time()
        EVALUATOR.write_json_atomic(args.output.resolve(), report)
        print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}), flush=True)
        return 0 if report["status"] == "complete" else 1
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "goal": REPORT_GOAL,
            "status": "failed",
            "phase": own.phase,
            "variant": args.variant,
            "training_seed": args.training_seed,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        EVALUATOR.write_json_atomic(args.output.resolve(), failure)
        print(json.dumps(failure["error"]), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
