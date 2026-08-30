#!/usr/bin/env python3
"""Synthesize the fail-closed G009 R0 rev18 CPU/GPU 2x2 capability probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "reports/runs"
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import probe_g009_r0_rev18_gpu_raw_contact as probe
import probe_g009_recover_runtime as runtime_probe

SCHEMA_VERSION = "g009.r0.rev18.gpu_raw_contact_synthesis.v1"
EXPECTED_SLOTS = (("cpu", 1), ("cpu", 2), ("cuda:0", 1), ("cuda:0", 2))
RAW_CHECK_NAMES = (
    "raw_subscription_attempted",
    "raw_subscription_succeeded",
    "raw_callback_well_formed",
    "raw_steps_monotonic_aligned",
    "nonempty_source_robot_ground_datum",
    "absolute_pair_paths",
    "finite_raw_vectors_and_separation",
    "unit_normals",
    "nonzero_impulse",
)
NUMERIC_TOLERANCES = {
    "position_w_m": {"absolute": 1.0e-6, "relative": 1.0e-5},
    "normal_w": {"absolute": 1.0e-6, "relative": 1.0e-5},
    "impulse_n_s": {"absolute": 1.0e-7, "relative": 1.0e-5},
    "separation_m": {"absolute": 1.0e-7, "relative": 1.0e-5},
}
SYNTHESIS_SOURCE_BINDING_PATHS = (
    "scripts/summarize_g009_r0_rev18_gpu_raw_contact.py",
    "scripts/probe_g009_r0_rev18_gpu_raw_contact.py",
    "scripts/probe_g009_recover_runtime.py",
    "src/isaac_walk_g009/recover_contracts.py",
)


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def committed_synthesis_blob_sha256(relative_path: str, commit: str) -> str:
    require(relative_path in SYNTHESIS_SOURCE_BINDING_PATHS, "unexpected synthesis source path")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "invalid synthesis source commit")
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return sha256_bytes(result.stdout)


def validate_synthesis_source_bundle(bundle: Any) -> dict[str, Any]:
    require(isinstance(bundle, dict), "synthesis source bundle must be an object")
    expected_keys = {
        "schema_version",
        "role",
        "git_commit",
        "git_commit_valid",
        "source_binding_paths",
        "source_binding_files",
        "source_bundle_sha256",
        "all_files_present",
        "missing_files",
        "clean",
        "dirty_source_paths",
    }
    require(set(bundle) == expected_keys, "synthesis source bundle schema mismatch")
    require(bundle.get("schema_version") == 1 and bundle.get("role") == "offline_synthesis_implementation", "synthesis source bundle role mismatch")
    commit = bundle.get("git_commit")
    require(isinstance(commit, str) and bool(re.fullmatch(r"[0-9a-f]{40}", commit)) and bundle.get("git_commit_valid") is True, "synthesis source commit identity mismatch")
    paths = list(SYNTHESIS_SOURCE_BINDING_PATHS)
    require(bundle.get("source_binding_paths") == paths, "synthesis source path order mismatch")
    files = bundle.get("source_binding_files")
    require(isinstance(files, dict) and list(files) == paths, "synthesis source file map mismatch")
    assert isinstance(files, dict)
    require(all(isinstance(files[path], str) and bool(re.fullmatch(r"[0-9a-f]{64}", files[path])) for path in paths), "synthesis source file hash format mismatch")
    require(bundle.get("all_files_present") is True and bundle.get("missing_files") == [] and bundle.get("clean") is True and bundle.get("dirty_source_paths") == [], "synthesis source bundle must be complete and clean")
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(paths))
    digest = sha256_bytes(payload.encode("utf-8"))
    require(bundle.get("source_bundle_sha256") == digest, "synthesis source aggregate SHA256 mismatch")
    for path in paths:
        require(files[path] == committed_synthesis_blob_sha256(path, commit), f"synthesis source committed blob mismatch: {path}")
    return bundle


def synthesis_source_bundle_provenance() -> dict[str, Any]:
    missing: list[str] = []
    for relative in SYNTHESIS_SOURCE_BINDING_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            missing.append(relative)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *SYNTHESIS_SOURCE_BINDING_PATHS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    require(not missing, "synthesis source files are missing")
    require(not dirty, "synthesis source paths must be clean")
    files = {
        relative: committed_synthesis_blob_sha256(relative, commit)
        for relative in SYNTHESIS_SOURCE_BINDING_PATHS
    }
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    bundle = {
        "schema_version": 1,
        "role": "offline_synthesis_implementation",
        "git_commit": commit,
        "git_commit_valid": bool(re.fullmatch(r"[0-9a-f]{40}", commit)),
        "source_binding_paths": list(SYNTHESIS_SOURCE_BINDING_PATHS),
        "source_binding_files": files,
        "source_bundle_sha256": sha256_bytes(payload.encode("utf-8")) if files else None,
        "all_files_present": not missing and len(files) == len(SYNTHESIS_SOURCE_BINDING_PATHS),
        "missing_files": missing,
        "clean": not dirty,
        "dirty_source_paths": dirty,
    }
    return validate_synthesis_source_bundle(bundle)


def _read_json_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value, raw


def _binding(path: Path, raw: bytes) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    require(resolved.parent == RUNS_DIR.resolve(), "input report must be a direct child of reports/runs")
    return {
        "path": resolved.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_bytes(raw),
    }


def load_inputs(paths: Sequence[Path]) -> list[tuple[dict[str, Any], dict[str, str]]]:
    require(len(paths) == 4, "exactly four raw reports are required")
    entries: list[tuple[dict[str, Any], dict[str, str]]] = []
    for path in paths:
        report, raw = _read_json_object(path)
        entries.append((report, _binding(path, raw)))
    return entries


def _execution_id(report: Mapping[str, Any]) -> str:
    execution = report.get("execution")
    require(isinstance(execution, Mapping), "execution block is missing")
    assert isinstance(execution, Mapping)
    value = execution.get("execution_id")
    require(isinstance(value, str), "execution_id is missing")
    assert isinstance(value, str)
    try:
        parsed = uuid.UUID(hex=value)
    except ValueError as error:
        raise ValueError("execution_id must be a lowercase UUID4 hex string") from error
    require(parsed.version == 4 and parsed.hex == value, "execution_id must be a lowercase UUID4 hex string")
    return value


def _replicate_index(report: Mapping[str, Any]) -> int:
    value = report.get("replicate_index")
    require(type(value) is int and value in (1, 2), "replicate_index must be 1 or 2")
    assert type(value) is int
    return value


def _canonical_source_bundle(report: Mapping[str, Any]) -> str:
    bundle = report.get("source_bundle")
    require(isinstance(bundle, Mapping), "source bundle is missing")
    assert isinstance(bundle, Mapping)
    require(bundle.get("all_files_present") is True, "source bundle is incomplete")
    require(bundle.get("clean") is True, "source bundle is dirty")
    expected_paths = list(probe.SOURCE_BINDING_PATHS)
    require(
        bundle.get("source_binding_paths") == expected_paths,
        "source bundle path set changed",
    )
    files = bundle.get("source_binding_files")
    require(
        isinstance(files, Mapping) and set(files) == set(expected_paths),
        "source bundle file map changed",
    )
    assert isinstance(files, Mapping)
    require(
        all(
            isinstance(files[path], str) and len(files[path]) == 64
            for path in expected_paths
        ),
        "source bundle file SHA256 is invalid",
    )
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(expected_paths))
    recomputed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    value = bundle.get("source_bundle_sha256")
    require(value == recomputed, "source bundle SHA256 is invalid")
    assert isinstance(value, str)
    return value


def _raw_structure(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = report.get("raw_contact_observation")
    require(isinstance(raw, Mapping), "raw contact observation is missing")
    assert isinstance(raw, Mapping)
    events = raw.get("events")
    require(isinstance(events, list), "raw contact events are missing")
    assert isinstance(events, list)
    structure: list[dict[str, Any]] = []
    for event in events:
        require(isinstance(event, Mapping), "raw contact event must be an object")
        headers = event.get("headers")
        require(isinstance(headers, list), "raw contact headers are missing")
        for header in headers:
            require(isinstance(header, Mapping), "raw contact header must be an object")
            if header.get("env_index") != probe.SOURCE_ENV_INDEX:
                continue
            points = header.get("contact_points")
            require(isinstance(points, list), "raw contact points are missing")
            structure.append(
                {
                    "physics_step": event.get("physics_step"),
                    "event_type": header.get("event_type"),
                    "env_index": header.get("env_index"),
                    "actor0_path": header.get("actor0_path"),
                    "actor1_path": header.get("actor1_path"),
                    "collider0_path": header.get("collider0_path"),
                    "collider1_path": header.get("collider1_path"),
                    "point_count": len(points),
                }
            )
    return structure


def _numeric_trace(report: Mapping[str, Any]) -> dict[str, list[float]]:
    trace = {name: [] for name in NUMERIC_TOLERANCES}
    raw = report["raw_contact_observation"]
    assert isinstance(raw, Mapping)
    events = raw["events"]
    assert isinstance(events, list)
    for event in events:
        assert isinstance(event, Mapping)
        headers = event["headers"]
        assert isinstance(headers, list)
        for header in headers:
            assert isinstance(header, Mapping)
            if header.get("env_index") != probe.SOURCE_ENV_INDEX:
                continue
            points = header["contact_points"]
            assert isinstance(points, list)
            for point in points:
                require(isinstance(point, Mapping), "raw contact point must be an object")
                for name in ("position_w_m", "normal_w", "impulse_n_s"):
                    vector = point.get(name)
                    require(isinstance(vector, list) and len(vector) == 3, f"{name} shape changed")
                    trace[name].extend(float(value) for value in vector)
                trace["separation_m"].append(float(point["separation_m"]))
    return trace


def _numeric_repeatable(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_trace = _numeric_trace(left)
    right_trace = _numeric_trace(right)
    for field, tolerance in NUMERIC_TOLERANCES.items():
        if len(left_trace[field]) != len(right_trace[field]):
            return False
        if not all(
            math.isclose(
                left_value,
                right_value,
                rel_tol=tolerance["relative"],
                abs_tol=tolerance["absolute"],
            )
            for left_value, right_value in zip(left_trace[field], right_trace[field], strict=True)
        ):
            return False
    return True


def _unavailable_signature(report: Mapping[str, Any], derived: Mapping[str, Any]) -> dict[str, Any]:
    raw = report.get("raw_contact_observation")
    require(isinstance(raw, Mapping), "raw contact observation is missing")
    assert isinstance(raw, Mapping)
    checks = derived.get("checks")
    require(isinstance(checks, Mapping), "derived checks are missing")
    assert isinstance(checks, Mapping)
    events = raw.get("events")
    require(isinstance(events, list), "raw contact events are missing")
    return {
        "scope": f"source_env_{probe.SOURCE_ENV_INDEX}_robot_ground_headers_only",
        "failed_raw_checks": [name for name in RAW_CHECK_NAMES if checks.get(name) is not True],
        "subscription_attempted": raw.get("subscription_attempted"),
        "subscription_succeeded": raw.get("subscription_succeeded"),
        "subscription_error": raw.get("subscription_error"),
        "malformed_callback_count": raw.get("malformed_callback_count"),
        "first_callback_error": raw.get("first_callback_error"),
        "event_structure": _raw_structure(report),
    }


def _device_repeatability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(rows) == 2, "within-device repeatability requires exactly two reports")
    left, right = rows
    left_pass = left["raw_observation_passed"] is True
    right_pass = right["raw_observation_passed"] is True
    if left_pass and right_pass:
        structure_equal = _raw_structure(left["report"]) == _raw_structure(right["report"])
        numeric_equal = structure_equal and _numeric_repeatable(left["report"], right["report"])
        return {
            "mode": "observed_raw_contact",
            "repeatable": structure_equal and numeric_equal,
            "structure_exact": structure_equal,
            "numeric_within_tolerance": numeric_equal,
            "unavailable_signature_exact": None,
        }
    if not left_pass and not right_pass:
        exact = left["unavailable_signature"] == right["unavailable_signature"]
        return {
            "mode": "unavailable_raw_contact",
            "repeatable": exact,
            "structure_exact": None,
            "numeric_within_tolerance": None,
            "unavailable_signature_exact": exact,
        }
    return {
        "mode": "split_raw_contact",
        "repeatable": False,
        "structure_exact": None,
        "numeric_within_tolerance": None,
        "unavailable_signature_exact": None,
    }


def _governance() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "selected_lever": None,
        "learned": False,
        "ppo": {"allowed": False, "status": "not_run", "updates": 0},
        "qualification": {"eligible": False, "status": "not_run", "passed": None},
        "gate01": {"allowed": False, "status": "forbidden"},
    }


def synthesize_loaded(
    entries: Sequence[tuple[dict[str, Any], dict[str, str]]],
) -> dict[str, Any]:
    require(len(entries) == 4, "exactly four raw reports are required")
    synthesis_source_bundle = synthesis_source_bundle_provenance()
    paths = [binding.get("path") for _, binding in entries]
    hashes = [binding.get("sha256") for _, binding in entries]
    require(len(set(paths)) == 4, "duplicate input report path")
    require(len(set(hashes)) == 4, "duplicate raw report hash")

    rows: list[dict[str, Any]] = []
    execution_ids: list[str] = []
    slots: list[tuple[str, int]] = []
    source_bundles: list[str] = []
    source_bundle_payloads: list[Mapping[str, Any]] = []
    predecessors: list[Mapping[str, Any]] = []
    for report, binding in entries:
        derived = probe.validate_report(report)
        recomputed = probe.derive_feasibility(report)
        require(derived == recomputed, "probe validator did not return recomputed feasibility")
        device = str(report.get("device", "")).lower()
        replicate = _replicate_index(report)
        slot = (device, replicate)
        slots.append(slot)
        execution_id = _execution_id(report)
        execution_ids.append(execution_id)
        execution = report.get("execution")
        assert isinstance(execution, Mapping)
        require(
            execution.get("output_path_repo_relative") == binding.get("path"),
            "raw input binding path must match execution output path",
        )
        source_bundles.append(_canonical_source_bundle(report))
        source_bundle = report.get("source_bundle")
        assert isinstance(source_bundle, Mapping)
        source_bundle_payloads.append(source_bundle)
        predecessor = report.get("predecessor")
        require(isinstance(predecessor, Mapping), "predecessor binding is missing")
        assert isinstance(predecessor, Mapping)
        predecessors.append(predecessor)
        checks = derived.get("checks")
        require(isinstance(checks, Mapping), "derived checks are missing")
        assert isinstance(checks, Mapping)
        raw_passed = derived.get("raw_observation_passed") is True
        probe_valid = derived.get("probe_valid") is True
        rows.append(
            {
                "slot": f"{device}.rep{replicate}",
                "device": device,
                "replicate_index": replicate,
                "binding": dict(binding),
                "execution_id": execution_id,
                "probe_valid": probe_valid,
                "positive_force_stimulus_present": checks.get(
                    "positive_force_stimulus_present"
                )
                is True,
                "raw_observation_passed": raw_passed,
                "instrumentation_bundle_complete": derived.get("supporting_bundle_complete") is True,
                "unavailable_signature": None if raw_passed else _unavailable_signature(report, derived),
                "report": report,
            }
        )

    require(sorted(slots) == sorted(EXPECTED_SLOTS), "reports must fill CPU rep1/2 and GPU rep1/2 exactly")
    require(len(set(execution_ids)) == 4, "execution_id values must be unique")
    require(len(set(source_bundles)) == 1, "source bundle changed across the 2x2 probe")
    require(
        all(item == source_bundle_payloads[0] for item in source_bundle_payloads),
        "source bundle payload changed across the 2x2 probe",
    )
    require(all(item == predecessors[0] for item in predecessors), "predecessor binding changed across the 2x2 probe")
    require(predecessors[0] == probe.validate_predecessor(), "rev17 predecessor binding mismatch")

    rows.sort(key=lambda row: EXPECTED_SLOTS.index((row["device"], row["replicate_index"])))
    by_device = {
        device: [row for row in rows if row["device"] == device]
        for device in ("cpu", "cuda:0")
    }
    repeatability = {
        device: _device_repeatability(device_rows)
        for device, device_rows in by_device.items()
    }
    cpu_ready = (
        all(row["probe_valid"] and row["raw_observation_passed"] for row in by_device["cpu"])
        and repeatability["cpu"]["repeatable"] is True
    )
    gpu_passes = [row["raw_observation_passed"] for row in by_device["cuda:0"]]
    gpu_probe_valid = all(row["probe_valid"] for row in by_device["cuda:0"])
    if not cpu_ready or not gpu_probe_valid:
        outcome = "probe_invalid"
        next_step = "repair_probe_or_instrumentation"
    elif gpu_passes == [True, True] and repeatability["cuda:0"]["repeatable"] is True:
        outcome = "gpu_pair_attribution_available"
        next_step = "authoritative_cpu_gpu_raw_topology_comparison"
    elif gpu_passes == [False, False] and repeatability["cuda:0"]["unavailable_signature_exact"] is True:
        outcome = "unavailable_on_gpu"
        next_step = "pre_registered_single_variable_intervention"
    else:
        outcome = "inconclusive_nondeterministic"
        next_step = "stop_without_third_run_majority_vote"

    complete_count = sum(row["instrumentation_bundle_complete"] for row in rows)
    bundle_status = "complete" if complete_count == 4 else ("unavailable" if complete_count == 0 else "partial")
    public_rows = [
        {key: value for key, value in row.items() if key != "report"}
        for row in rows
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": "G009-5-E011",
        "goal_id": "g009",
        "stage_id": "R0",
        "revision": "rev18",
        "status": "pass",
        "mode": "offline_fail_closed_2x2_synthesis",
        "input_report_count": 4,
        "input_reports": [row["binding"] for row in public_rows],
        "integrity": {
            "passed": True,
            "hash_bound": True,
            "unique_execution_ids": True,
            "exact_slots": [f"{device}.rep{replicate}" for device, replicate in EXPECTED_SLOTS],
            "source_bundle_sha256": source_bundles[0],
            "raw_probe_source_bundle_sha256": source_bundles[0],
            "synthesis_source_bundle_sha256": synthesis_source_bundle[
                "source_bundle_sha256"
            ],
            "predecessor": dict(predecessors[0]),
        },
        "synthesis_source_bundle": synthesis_source_bundle,
        "raw_contact_feasibility": {
            "outcome": outcome,
            "gpu_pair_attribution_available": outcome == "gpu_pair_attribution_available",
            "cpu_control_2_of_2_passed_repeatable": cpu_ready,
            "third_run_majority_vote_allowed": False,
            "cross_device_numeric_equality_required": False,
            "within_device_numeric_tolerances": NUMERIC_TOLERANCES,
            "repeatability": repeatability,
            "runs": public_rows,
        },
        "instrumentation_bundle": {
            "status": bundle_status,
            "complete_report_count": complete_count,
            "required_report_count": 4,
            "independent_of_raw_contact_feasibility": True,
        },
        "decision": {
            "outcome": outcome,
            "next_step": next_step,
            "selected_lever": None,
        },
        "governance": _governance(),
    }


def synthesize(paths: Sequence[Path]) -> dict[str, Any]:
    return synthesize_loaded(load_inputs(paths))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    output, execution = runtime_probe.prepare_execution(
        runtime_probe.parse_prelaunch_output(argv)
    )
    args = parse_args(argv)
    require(len(args.report) == 4, "exactly four --report arguments are required")
    report = synthesize(args.report)
    report["created_at_utc"] = execution["started_at_utc"]
    report["execution"] = execution
    runtime_probe._write_json_atomic(output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
