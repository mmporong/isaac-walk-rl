#!/usr/bin/env python3
"""Fail-closed CPU preflight and final synthesis for G009-5-E013 rev20."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import probe_g009_r0_rev20_terrain_contact_matrix as probe


RUNS_DIR = REPO_ROOT / "reports/runs"
CPU_SCHEMA = "g009.r0.rev20.terrain_contact_matrix_cpu_preflight.v1"
FINAL_SCHEMA = "g009.r0.rev20.terrain_contact_matrix_synthesis.v1"
CPU_OUTPUT = RUNS_DIR / "g009_r0_rev20_terrain_contact_matrix_cpu_preflight_2x_s42.json"
FINAL_OUTPUT = RUNS_DIR / "g009_r0_rev20_terrain_contact_matrix_synthesis_2x2_s42.json"
CPU_PATHS = tuple(probe.EXPECTED_PATHS[("cpu", replicate)] for replicate in (1, 2))
FINAL_PATHS = CPU_PATHS + tuple(probe.EXPECTED_PATHS[("cuda:0", replicate)] for replicate in (1, 2))
ABS_TOL = 1e-5
REL_TOL = 1e-6


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {token}")))
    require(isinstance(value, dict), "report root must be an object")
    return value, raw


def binding(path: Path, raw: bytes) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    require(resolved.parent == RUNS_DIR.resolve(), "input must be a direct reports/runs JSON")
    return {"path": resolved.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_bytes(raw)}


def load_inputs(paths: Sequence[Path], expected_paths: Sequence[str]) -> list[tuple[dict[str, Any], dict[str, str]]]:
    require(len(paths) == len(expected_paths), f"exactly {len(expected_paths)} reports required")
    entries = [(value, binding(path, raw)) for path in paths for value, raw in [_read_json(path)]]
    bindings = [item for _, item in entries]
    require(all(list(item.keys()) == ["path", "sha256"] for item in bindings), "input binding key order mismatch")
    require([item["path"] for item in bindings] == list(expected_paths), "canonical input path/order mismatch")
    require(len({item["path"] for item in bindings}) == len(bindings), "duplicate report path")
    require(len({item["sha256"] for item in bindings}) == len(bindings), "duplicate report SHA")
    return entries


def execution_id(report: Mapping[str, Any]) -> str:
    return probe.validate_uuid4_hex(report.get("execution", {}).get("execution_id"))


def numeric_close(left: float, right: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= max(ABS_TOL, REL_TOL * max(abs(left), abs(right)))


def repeatability(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) == 2, "repeatability requires exactly two rows")
    exact_keys = ("availability_state", "sensor_paths_sha256", "raw_filter_paths_sha256", "logical_filter_paths_sha256", "force_body_names_sha256", "raw_and_reshaped_tensor_shapes", "per_env_overlap_step_indices", "source_env_overlap_step_indices", "safety_checks")
    numeric_keys = ("all_env_matrix_peak_force_n", "source_env_matrix_peak_force_n", "all_env_matrix_force_integral_n_s", "source_env_matrix_force_integral_n_s")
    exact = all(rows[0][key] == rows[1][key] for key in exact_keys)
    numeric = all(numeric_close(float(rows[0][key]), float(rows[1][key])) for key in numeric_keys)
    return {"exact_fields_match": exact, "numeric_fields_within_tolerance": numeric, "repeatable": exact and numeric, "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL}


def row(report: Mapping[str, Any], item_binding: Mapping[str, str]) -> dict[str, Any]:
    probe.validate_report(report)
    matrix = report["terrain_contact_matrix"]
    overlap = matrix["same_step_overlap"]
    return {
        "slot": f"{report['device']}.rep{report['replicate_index']}", "device": report["device"], "replicate_index": report["replicate_index"], "binding": dict(item_binding),
        "execution_id": execution_id(report), "availability_state": matrix["availability_state"],
        "sensor_paths_sha256": matrix["path_order"]["sensor_paths_sha256"], "raw_filter_paths_sha256": matrix["path_order"]["raw_filter_paths_sha256"], "logical_filter_paths_sha256": matrix["path_order"]["logical_filter_paths_sha256"], "force_body_names_sha256": matrix["path_order"]["force_body_names_sha256"],
        "raw_and_reshaped_tensor_shapes": [matrix["shapes"]["raw"], matrix["shapes"]["reshaped"]],
        "per_env_overlap_step_indices": overlap["per_env_overlap_step_indices"], "source_env_overlap_step_indices": overlap["source_env_overlap_step_indices"],
        "safety_checks": matrix["checks"], "all_env_matrix_peak_force_n": overlap["all_env_matrix_peak_force_n"], "source_env_matrix_peak_force_n": overlap["source_env_matrix_peak_force_n"],
        "all_env_matrix_force_integral_n_s": overlap["all_env_matrix_force_integral_n_s"], "source_env_matrix_force_integral_n_s": overlap["source_env_matrix_force_integral_n_s"],
        "source_bundle_sha256": report["source_bundle"]["source_bundle_sha256"], "cpu_preflight_binding": report["cpu_preflight_binding"],
        "git_commit": report["source_bundle"]["git_commit"], "probe_valid": report["feasibility"]["probe_valid"],
        "structural_probe_valid": matrix["structural_probe_valid"], "overlap_available": matrix["overlap_available"], "baseline_passed": report["baseline_snapshot"]["all_match"],
        "device_passed": report["device_readback"]["gpu_dynamics_matches_device"], "live_readback_passed": probe.live_readback_valid(report),
        "external_passed": report["external_source_binding"]["all_hashes_match"], "safety_passed": matrix["safety_valid"],
    }


def validate_entry_uniqueness(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len({item["binding"]["path"] for item in rows}) == len(rows), "duplicate report path")
    require(len({item["binding"]["sha256"] for item in rows}) == len(rows), "duplicate report SHA")
    require(len({item["execution_id"] for item in rows}) == len(rows), "duplicate execution_id")


def synthesis_source_bundle_provenance() -> dict[str, Any]:
    paths = probe.SYNTHESIS_SOURCE_BINDING_PATHS
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *paths], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    require(not dirty, "synthesis source paths must be committed and clean")
    files = {path: probe._git_blob_sha256(path, commit) for path in paths}
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return {"schema_version": 1, "git_commit": commit, "git_commit_valid": bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "source_binding_paths": list(paths), "source_binding_files": files, "source_bundle_sha256": sha256_bytes(payload.encode()), "clean": True}


def new_execution(output: Path, forbidden_ids: set[str]) -> dict[str, Any]:
    execution_id_value = uuid.uuid4().hex
    require(execution_id_value not in forbidden_ids, "synthesis execution_id collides with input")
    return {"execution_id": execution_id_value, "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"), "output_path_repo_relative": output.resolve().relative_to(REPO_ROOT.resolve()).as_posix(), "no_overwrite": True}


def cpu_preflight(entries: Sequence[tuple[dict[str, Any], dict[str, str]]], output: Path = CPU_OUTPUT) -> dict[str, Any]:
    require(len(entries) == 2, "CPU preflight requires exactly two reports")
    rows = [row(report, item_binding) for report, item_binding in entries]
    validate_entry_uniqueness(rows)
    require([item["slot"] for item in rows] == ["cpu.rep1", "cpu.rep2"], "CPU slot order mismatch")
    repeated = repeatability(rows)
    ids = [item["execution_id"] for item in rows]
    source_digests = {item["source_bundle_sha256"] for item in rows}; require(len(source_digests) == 1, "CPU source bundle drift")
    synthesis_bundle = synthesis_source_bundle_provenance()
    commits = {item["git_commit"] for item in rows}; require(commits == {synthesis_bundle["git_commit"]}, "CPU report/synthesis git commit mismatch")
    if not all(item["baseline_passed"] and item["device_passed"] and item["live_readback_passed"] and item["external_passed"] for item in rows): outcome = "probe_invalid"
    elif not all(item["structural_probe_valid"] for item in rows): outcome = "terrain_matrix_probe_invalid"
    elif not all(item["safety_passed"] for item in rows): outcome = "safety_limit_exceeded"
    elif [item["availability_state"] for item in rows] == ["unavailable", "unavailable"]: outcome = "cpu_terrain_matrix_unavailable_gpu_forbidden"
    elif [item["availability_state"] for item in rows] != ["observed_valid", "observed_valid"] or not repeated["repeatable"]: outcome = "inconclusive_nondeterministic_gpu_forbidden"
    else: outcome = "gpu_stage_authorized"
    gpu_allowed = outcome == "gpu_stage_authorized"
    return {
        "schema_version": CPU_SCHEMA, "evidence_id": "G009-5-E013", "status": "complete", "mode": "cpu_preflight_2x",
        "input_report_count": 2, "input_reports": [item["binding"] for item in rows],
        "integrity": {"passed": True, "hash_bound": True, "unique_report_paths": True, "unique_report_sha256": True, "unique_execution_ids": True, "exact_slots": [item["slot"] for item in rows], "git_commit": synthesis_bundle["git_commit"], "probe_source_bundle_sha256": next(iter(source_digests)), "synthesis_source_bundle_sha256": synthesis_bundle["source_bundle_sha256"]},
        "cpu_preflight": {"passed": gpu_allowed, "required_checks_passed": all(item["structural_probe_valid"] and item["safety_passed"] and item["baseline_passed"] and item["device_passed"] and item["live_readback_passed"] and item["external_passed"] for item in rows), "within_cpu_repeatability_passed": repeated["repeatable"], "gpu_stage_allowed": gpu_allowed},
        "decision": {"outcome": outcome, "third_run_allowed": False, "repeatability": repeated},
        "governance": probe.governance(), "synthesis_source_bundle": synthesis_bundle, "execution": new_execution(output, set(ids)),
    }


def _read_preflight(path: Path) -> tuple[dict[str, Any], bytes]:
    require(path.resolve(strict=True) == CPU_OUTPUT.resolve(), "final synthesis requires canonical CPU preflight")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {token}")))
    require(isinstance(value, dict), "CPU preflight root must be object")
    probe.validate_cpu_preflight_value(value, REPO_ROOT, CPU_OUTPUT.relative_to(REPO_ROOT).as_posix())
    return value, raw


def final_synthesis(entries: Sequence[tuple[dict[str, Any], dict[str, str]]], preflight_path: Path, output: Path = FINAL_OUTPUT) -> dict[str, Any]:
    require(len(entries) == 4, "final synthesis requires exactly four reports")
    rows = [row(report, item_binding) for report, item_binding in entries]
    validate_entry_uniqueness(rows)
    require([item["slot"] for item in rows] == ["cpu.rep1", "cpu.rep2", "cuda:0.rep1", "cuda:0.rep2"], "final slot order mismatch")
    preflight, preflight_raw = _read_preflight(preflight_path)
    require(preflight["input_reports"] == [item["binding"] for item in rows[:2]], "final CPU bindings differ from preflight")
    expected_gpu_binding = {"status": "validated_for_gpu", "path": CPU_OUTPUT.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_bytes(preflight_raw), "git_commit": preflight["integrity"]["git_commit"], "probe_source_bundle_sha256": preflight["integrity"]["probe_source_bundle_sha256"], "input_reports": preflight["input_reports"]}
    require(rows[2]["cpu_preflight_binding"] == expected_gpu_binding and rows[3]["cpu_preflight_binding"] == expected_gpu_binding, "GPU reports do not exact-bind the same preflight")
    ids = [item["execution_id"] for item in rows] + [preflight["execution"]["execution_id"]]
    require(len(set(ids)) == 5, "duplicate report/preflight execution_id")
    require(len({item["source_bundle_sha256"] for item in rows}) == 1 and len({item["git_commit"] for item in rows}) == 1, "final report source bundle/commit drift")
    require(rows[0]["source_bundle_sha256"] == preflight["integrity"]["probe_source_bundle_sha256"] and rows[0]["git_commit"] == preflight["integrity"]["git_commit"], "final reports differ from preflight source binding")
    cpu_repeat, gpu_repeat = repeatability(rows[:2]), repeatability(rows[2:])
    states = [item["availability_state"] for item in rows]
    if not all(item["baseline_passed"] and item["device_passed"] and item["live_readback_passed"] and item["external_passed"] for item in rows): outcome = "probe_invalid"
    elif not all(item["structural_probe_valid"] for item in rows): outcome = "terrain_matrix_probe_invalid"
    elif not all(item["safety_passed"] for item in rows): outcome = "safety_limit_exceeded"
    elif not cpu_repeat["repeatable"] or states[:2] != ["observed_valid", "observed_valid"]: outcome = "inconclusive_nondeterministic_gpu_forbidden"
    elif not gpu_repeat["repeatable"]: outcome = "inconclusive_nondeterministic"
    elif states[2:] == ["observed_valid", "observed_valid"]: outcome = "terrain_pair_matrix_authority_candidate_validated"
    elif states[2:] == ["unavailable", "unavailable"]: outcome = "gpu_terrain_matrix_unavailable"
    else: outcome = "inconclusive_nondeterministic"
    synthesis_bundle = synthesis_source_bundle_provenance()
    require({item["git_commit"] for item in rows} == {synthesis_bundle["git_commit"]}, "final report/synthesis git commit mismatch")
    return {
        "schema_version": FINAL_SCHEMA, "evidence_id": "G009-5-E013", "status": "complete", "mode": "final_2x2",
        "input_report_count": 4, "input_reports": [item["binding"] for item in rows],
        "integrity": {"passed": True, "hash_bound": True, "unique_report_paths": True, "unique_report_sha256": True, "unique_execution_ids": True, "exact_slots": [item["slot"] for item in rows], "preflight": {"path": CPU_OUTPUT.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_bytes(preflight_raw)}},
        "repeatability": {"cpu": cpu_repeat, "cuda:0": gpu_repeat}, "rows": rows,
        "decision": {"outcome": outcome, "next_step": "preregister_matrix_authority_safety_gate" if outcome == "terrain_pair_matrix_authority_candidate_validated" else "stop_and_fix_filter_path_or_view_only", "third_run_allowed": False},
        "claim_limits": {"terrain_pair_aggregated_normal_force_authority_candidate_only": True, "gpu_contact_absence_claimed": False, "physics_failure_claimed": False, "callback_count_used": False},
        "governance": probe.governance(), "synthesis_source_bundle": synthesis_bundle, "execution": new_execution(output, set(ids)),
    }


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    require(resolved.parent == RUNS_DIR.resolve() and resolved.suffix == ".json", "output must be direct reports/runs JSON")
    require(not resolved.exists(), "refusing to overwrite output")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        with resolved.open("xb") as destination, temporary.open("rb") as source:
            while block := source.read(1024 * 1024): destination.write(block)
            destination.flush(); os.fsync(destination.fileno())
    except BaseException:
        resolved.unlink(missing_ok=True); raise
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode", required=True, choices=("cpu-preflight", "final")); parser.add_argument("--inputs", nargs="+", required=True, type=Path); parser.add_argument("--cpu-preflight", type=Path); parser.add_argument("--output", required=True, type=Path); return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "cpu-preflight":
        require(args.cpu_preflight is None, "CPU mode must not bind a preflight"); entries = load_inputs(args.inputs, CPU_PATHS); value = cpu_preflight(entries, args.output)
    else:
        require(args.cpu_preflight is not None, "final mode requires --cpu-preflight"); entries = load_inputs(args.inputs, FINAL_PATHS); value = final_synthesis(entries, args.cpu_preflight, args.output)
    write_json_exclusive(args.output, value); print(json.dumps({"output": str(args.output), "outcome": value["decision"]["outcome"]}, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
