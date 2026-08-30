#!/usr/bin/env python3
"""Fail-closed synthesis for the eight preregistered G009-5-E012 runs."""

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
from typing import Any, Mapping, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for search_root in (SCRIPT_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import probe_g009_r0_rev19_contact_offset_intervention as probe
import probe_g009_recover_runtime as runtime_probe
import summarize_g009_r0_rev18_gpu_raw_contact as base_summary


SCHEMA_VERSION = "g009.r0.rev19.contact_offset_intervention_synthesis.v1"
RUNS_DIR = REPO_ROOT / "reports/runs"
FINAL_SYNTHESIS_PATH = RUNS_DIR / "g009_r0_rev19_contact_offset_intervention_synthesis_2x2x2_s42.json"
CPU_SLOTS = tuple((arm, "cpu", replicate) for arm in ("A", "B") for replicate in (1, 2))
GPU_SLOTS = tuple((arm, "cuda:0", replicate) for arm in ("A", "B") for replicate in (1, 2))
EXPECTED_SLOTS = CPU_SLOTS + GPU_SLOTS
EXPECTED_PATHS = tuple(
    probe.expected_output_relative(arm, device, replicate)
    for arm, device, replicate in EXPECTED_SLOTS
)
SYNTHESIS_SOURCE_BINDING_PATHS = probe.SYNTHESIS_SOURCE_BINDING_PATHS


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def committed_synthesis_blob_sha256(relative_path: str, commit: str) -> str:
    return probe.committed_synthesis_blob_sha256(relative_path, commit)


def synthesis_source_bundle_provenance() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *SYNTHESIS_SOURCE_BINDING_PATHS], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    missing = [path for path in SYNTHESIS_SOURCE_BINDING_PATHS if not (REPO_ROOT / path).is_file()]
    require(not missing, "synthesis source files are missing")
    require(not dirty, "synthesis source paths must be clean")
    files = {path: committed_synthesis_blob_sha256(path, commit) for path in SYNTHESIS_SOURCE_BINDING_PATHS}
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    bundle = {
        "schema_version": 1,
        "role": "offline_synthesis_implementation",
        "git_commit": commit,
        "git_commit_valid": bool(re.fullmatch(r"[0-9a-f]{40}", commit)),
        "source_binding_paths": list(SYNTHESIS_SOURCE_BINDING_PATHS),
        "source_binding_files": files,
        "source_bundle_sha256": sha256_bytes(payload.encode("utf-8")),
        "all_files_present": True,
        "missing_files": [],
        "clean": True,
        "dirty_source_paths": [],
    }
    return validate_synthesis_source_bundle(bundle)


def validate_synthesis_source_bundle(bundle: Any) -> dict[str, Any]:
    require(isinstance(bundle, Mapping) and isinstance(bundle.get("git_commit"), str), "synthesis source commit missing")
    return probe.validate_synthesis_source_bundle(bundle, cast(str, bundle.get("git_commit")))


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {item}")))
    require(isinstance(value, dict), "report root must be an object")
    return value, raw


def _binding(path: Path, raw: bytes) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    require(resolved.parent == RUNS_DIR.resolve(), "input report must be a direct reports/runs child")
    return {"path": resolved.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_bytes(raw)}


def load_inputs(paths: Sequence[Path], expected_count: int) -> list[tuple[dict[str, Any], dict[str, str]]]:
    require(len(paths) == expected_count, f"exactly {expected_count} rev19 reports are required")
    return [(report, _binding(path, raw)) for path in paths for report, raw in [_read_json(path)]]


def _execution_id(report: Mapping[str, Any]) -> str:
    execution = report.get("execution")
    require(isinstance(execution, Mapping), "execution block missing")
    execution = cast(Mapping[str, Any], execution)
    value = execution.get("execution_id")
    require(isinstance(value, str), "execution_id missing")
    parsed = uuid.UUID(hex=cast(str, value))
    require(parsed.version == 4 and parsed.hex == value, "execution_id must be lowercase UUID4 hex")
    return cast(str, value)


def _source_bundle_digest(report: Mapping[str, Any]) -> str:
    bundle = probe.validate_source_bundle(report.get("source_bundle"))
    digest = bundle.get("source_bundle_sha256")
    require(isinstance(digest, str), "raw source bundle digest missing")
    return cast(str, digest)


def _unavailable_signature(report: Mapping[str, Any], derived: Mapping[str, Any]) -> dict[str, Any]:
    return base_summary._unavailable_signature(report, derived)


def _cell_repeatability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return probe.cell_repeatability(rows)


def _raw_cell_state(rows: list[dict[str, Any]], repeatability: Mapping[str, Any]) -> str:
    values = [row["raw_observation_passed"] for row in rows]
    if values == [True, True] and repeatability.get("raw_repeatable") is True:
        return "observed_2_of_2"
    if values == [False, False] and repeatability.get("raw_repeatable") is True:
        return "unavailable_2_of_2"
    return "split_or_nonrepeatable"


def _governance() -> dict[str, Any]:
    return probe.synthesis_governance()


def _validated_rows(
    entries: Sequence[tuple[dict[str, Any], dict[str, str]]],
    expected_slots: Sequence[tuple[str, str, int]],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    require(len(entries) == len(expected_slots), f"exactly {len(expected_slots)} rev19 reports are required")
    synthesis_bundle = synthesis_source_bundle_provenance()
    rows: list[dict[str, Any]] = []
    execution_ids: list[str] = []
    source_digests: list[str] = []
    source_payloads: list[Any] = []
    slots: list[tuple[str, str, int]] = []
    for report, binding in entries:
        derived = probe.validate_report(report)
        require(derived == probe.derive_feasibility(report), "probe validation must recompute feasibility")
        arm = str(report.get("arm", "")).upper()
        device = str(report.get("device", "")).lower()
        replicate = report.get("replicate_index")
        require(type(replicate) is int, "replicate index missing")
        slot = (arm, device, cast(int, replicate))
        slots.append(slot)
        require(binding.get("path") == probe.expected_output_relative(*slot), "canonical report path mismatch")
        execution = report.get("execution")
        require(isinstance(execution, Mapping) and execution.get("output_path_repo_relative") == binding.get("path"), "execution/input path binding mismatch")
        execution_ids.append(_execution_id(report))
        source_digests.append(_source_bundle_digest(report))
        source_payloads.append(report.get("source_bundle"))
        offset = cast(Mapping[str, Any], report["offset_integrity"])
        before = cast(Mapping[str, Any], offset["before"])
        after = cast(Mapping[str, Any], offset["after"])
        safety = cast(Mapping[str, Any], report["manual_probe_safety"])
        mass = cast(Mapping[str, Any], safety["mass_evidence"])
        mass_tensor = cast(Mapping[str, Any], mass["tensor"])
        raw_passed = derived["raw_observation_passed"] is True
        rows.append({
            "slot": f"{arm}.{device}.rep{replicate}",
            "arm": arm,
            "device": device,
            "replicate_index": replicate,
            "binding": dict(binding),
            "execution_id": execution_ids[-1],
            "probe_valid": derived["probe_valid"] is True,
            "run_interpretable": derived["run_interpretable"] is True,
            "raw_observation_passed": raw_passed,
            "offset_integrity_passed": derived["offset_integrity_passed"] is True,
            "solver_live_readback_8_0": derived["solver_live_readback_8_0"] is True,
            "safety_available": safety.get("available") is True,
            "safety_passed": safety.get("passed") is True,
            "mass_tensor_sha256": mass_tensor["sha256"],
            "mass_body_names_sha256": mass["body_names_sha256"],
            "force_body_names_sha256": mass["contact_force_body_names_sha256"],
            "offset_scale": offset["contact_offset_scale"],
            "offset_baseline_contact_sha256": cast(Mapping[str, Any], before["contact_offset"])["sha256"],
            "offset_baseline_rest_sha256": cast(Mapping[str, Any], before["rest_offset"])["sha256"],
            "offset_after_contact_sha256": cast(Mapping[str, Any], after["contact_offset"])["sha256"],
            "unavailable_signature": None if raw_passed else _unavailable_signature(report, derived),
            "report": report,
        })
    require(slots == list(expected_slots), "reports must use exact canonical slot order")
    require([binding["path"] for _, binding in entries] == [probe.expected_output_relative(*slot) for slot in expected_slots], "report paths must use exact canonical order")
    require(len(set(execution_ids)) == len(expected_slots), "execution_id values must be unique")
    require(len(set(source_digests)) == 1 and all(payload == source_payloads[0] for payload in source_payloads), "source bundle changed across rev19 runs")
    require(len({row["mass_tensor_sha256"] for row in rows}) == 1, "default mass tensor changed across rev19 runs")
    require(len({row["mass_body_names_sha256"] for row in rows}) == 1, "mass body ordering changed across rev19 runs")
    require(len({row["force_body_names_sha256"] for row in rows}) == 1, "contact force body ordering changed across rev19 runs")
    baseline_contact = {row["offset_baseline_contact_sha256"] for row in rows}
    baseline_rest = {row["offset_baseline_rest_sha256"] for row in rows}
    require(len(baseline_contact) == 1 and len(baseline_rest) == 1, "A/B baseline offset vectors differ")
    require(all(row["offset_scale"] == probe.ARM_SCALES[row["arm"]] for row in rows), "A/B scale assignment mismatch")
    return rows, source_digests[0], synthesis_bundle


def _cells(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    devices = sorted({cast(str, row["device"]) for row in rows})
    return {(arm, device): [row for row in rows if row["arm"] == arm and row["device"] == device] for arm in ("A", "B") for device in devices}


def _decision(*, probe_integrity: bool, safety_available: bool, cpu_preflight_passed: bool, safety_passed: bool, any_split: bool, gpu_a: str, gpu_b: str) -> tuple[str, str]:
    if not probe_integrity:
        return "probe_invalid", "repair_offset_or_probe_integrity"
    if not safety_available:
        return "safety_unavailable", "repair_manual_safety_observation"
    if not cpu_preflight_passed:
        return "cpu_preflight_failed_gpu_results_not_interpretable", "do_not_authorize_gpu_stage"
    if not safety_passed:
        return "safety_limit_exceeded", "stop_and_inspect_manual_probe_safety"
    if any_split:
        return "inconclusive_nondeterministic", "stop_without_third_run_majority_vote"
    if gpu_a == "observed_2_of_2" and gpu_b == "observed_2_of_2":
        return "gpu_raw_available_both_arms", "retain_no_lever_and_plan_separate_physics_validation"
    if gpu_a == "unavailable_2_of_2" and gpu_b == "unavailable_2_of_2":
        return "gpu_raw_unavailable_both_arms", "stop_without_gpu_contact_absence_claim"
    if gpu_a == "unavailable_2_of_2" and gpu_b == "observed_2_of_2":
        return "gpu_raw_enabled_by_contact_offset", "replicate_in_separate_authority_stage_without_selecting_lever"
    return "gpu_raw_regressed_with_contact_offset", "stop_without_physics_failure_claim"


def synthesize_cpu_preflight_loaded(entries: Sequence[tuple[dict[str, Any], dict[str, str]]]) -> dict[str, Any]:
    rows, source_digest, synthesis_bundle = _validated_rows(entries, CPU_SLOTS)
    cells = _cells(rows)
    repeatability = {f"{arm}.cpu": _cell_repeatability(cells[(arm, "cpu")]) for arm in ("A", "B")}
    integrity_passed = all(row["probe_valid"] and row["offset_integrity_passed"] and row["solver_live_readback_8_0"] for row in rows)
    ready = all(row["raw_observation_passed"] and row["safety_available"] and row["safety_passed"] for row in rows)
    repeatable = all(value["repeatable"] is True for value in repeatability.values())
    require(integrity_passed and ready and repeatable, "CPU preflight reports must all raw PASS, probe valid, safety PASS, and repeatable")
    source_payload = cast(Mapping[str, Any], rows[0]["report"]["source_bundle"])
    return {
        "schema_version": probe.CPU_PREFLIGHT_SCHEMA_VERSION,
        "evidence_id": "G009-5-E012",
        "goal_id": "g009",
        "stage_id": "R0",
        "revision": "rev19",
        "status": "complete",
        "mode": "cpu_preflight_2x2",
        "input_report_count": 4,
        "input_reports": [row["binding"] for row in rows],
        "integrity": {"passed": True, "hash_bound": True, "unique_execution_ids": True, "exact_slots": [f"{arm}.cpu.rep{replicate}" for arm, _, replicate in CPU_SLOTS], "git_commit": source_payload["git_commit"], "probe_source_bundle_sha256": source_digest, "synthesis_source_bundle_sha256": synthesis_bundle["source_bundle_sha256"], "mass_tensor_sha256": rows[0]["mass_tensor_sha256"], "mass_body_names_sha256": rows[0]["mass_body_names_sha256"], "force_body_names_sha256": rows[0]["force_body_names_sha256"]},
        "cpu_preflight": {"passed": True, "raw_pass_probe_valid_safety_pass": True, "within_arm_repeatability_passed": True, "gpu_stage_allowed": True},
        "decision": {"outcome": "gpu_stage_authorized", "selected_lever": None, "third_run_majority_vote_allowed": False, "repeatability": repeatability},
        "governance": _governance(),
        "synthesis_source_bundle": synthesis_bundle,
    }


def synthesize_loaded(entries: Sequence[tuple[dict[str, Any], dict[str, str]]], cpu_preflight_path: Path = probe.CPU_PREFLIGHT_PATH) -> dict[str, Any]:
    rows, source_digest, synthesis_bundle = _validated_rows(entries, EXPECTED_SLOTS)
    source_payload = cast(Mapping[str, Any], rows[0]["report"]["source_bundle"])
    preflight_binding = probe.validate_cpu_preflight_artifact(cpu_preflight_path, source_payload)
    preflight_value = json.loads(cpu_preflight_path.read_text(encoding="utf-8"))
    require([row["binding"] for row in rows[:4]] == preflight_value.get("input_reports"), "final CPU inputs differ from immutable preflight")
    require(all(row["report"]["cpu_preflight_binding"] == probe.cpu_preflight_not_required_binding() for row in rows[:4]), "CPU report preflight binding mismatch")
    require(all(row["report"]["cpu_preflight_binding"] == preflight_binding for row in rows[4:]), "GPU report preflight binding drift")
    baseline_contact = {row["offset_baseline_contact_sha256"] for row in rows}
    baseline_rest = {row["offset_baseline_rest_sha256"] for row in rows}
    cells = _cells(rows)
    repeatability = {f"{arm}.{device}": _cell_repeatability(cell_rows) for (arm, device), cell_rows in cells.items()}
    raw_states = {f"{arm}.{device}": _raw_cell_state(cell_rows, repeatability[f"{arm}.{device}"]) for (arm, device), cell_rows in cells.items()}
    any_split = any(state == "split_or_nonrepeatable" for state in raw_states.values())
    offset_probe_valid = all(row["probe_valid"] and row["offset_integrity_passed"] and row["solver_live_readback_8_0"] for row in rows)
    safety_available = all(row["safety_available"] for row in rows)
    safety_passed = all(row["safety_passed"] for row in rows)
    cpu_preflight_passed = preflight_binding.get("status") == "validated_for_gpu"
    gpu_a = raw_states["A.cuda:0"]
    gpu_b = raw_states["B.cuda:0"]
    outcome, next_step = _decision(
        probe_integrity=offset_probe_valid,
        safety_available=safety_available,
        cpu_preflight_passed=cpu_preflight_passed,
        safety_passed=safety_passed,
        any_split=any_split,
        gpu_a=gpu_a,
        gpu_b=gpu_b,
    )
    public_rows = [{key: value for key, value in row.items() if key != "report"} for row in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": "G009-5-E012",
        "goal_id": "g009",
        "stage_id": "R0",
        "revision": "rev19",
        "status": "complete",
        "mode": "offline_fail_closed_2x2x2_synthesis",
        "input_report_count": 8,
        "input_reports": [row["binding"] for row in public_rows],
        "integrity": {"passed": True, "hash_bound": True, "unique_execution_ids": True, "exact_slots": [f"{arm}.{device}.rep{replicate}" for arm, device, replicate in EXPECTED_SLOTS], "source_bundle_sha256": source_digest, "synthesis_source_bundle_sha256": synthesis_bundle["source_bundle_sha256"], "mass_tensor_sha256": rows[0]["mass_tensor_sha256"], "mass_body_names_sha256": rows[0]["mass_body_names_sha256"], "force_body_names_sha256": rows[0]["force_body_names_sha256"], "predecessor": probe.validate_predecessor(), "preregistration": probe.probe_contract("A", "cpu", 1)["preregistration"], "single_variable_difference": {"solver": "8/0_both_arms", "rest_offset": "unchanged_both_arms", "contact_offset_scale": {"A": 1.0, "B": 1.5}, "baseline_contact_sha256": next(iter(baseline_contact)), "baseline_rest_sha256": next(iter(baseline_rest)), "comparison_authority": "rev19_arm_A_vs_arm_B_only"}},
        "synthesis_source_bundle": synthesis_bundle,
        "cpu_preflight": {"required_before_gpu": True, "binding": preflight_binding, "passed": cpu_preflight_passed, "gpu_stage_allowed": cpu_preflight_passed},
        "raw_callback_observation": {"outcome": outcome, "physics_ground_truth_authority": False, "gpu_contact_absence_claimed": False, "physics_failure_claimed": False, "third_run_majority_vote_allowed": False, "states": raw_states, "repeatability": repeatability, "runs": public_rows},
        "manual_probe_safety": {"label": "manual_probe_observation_not_gate", "available": safety_available, "passed": safety_passed, "thresholds": {"hard_joint_limit_margin_rad": probe.HARD_JOINT_LIMIT_MARGIN_RAD, "non_foot_peak_force_body_weight_max": probe.NON_FOOT_PEAK_FORCE_BODY_WEIGHT_MAX, "cpu_raw_minimum_separation_m": probe.CPU_RAW_MINIMUM_SEPARATION_M}},
        "decision": {"outcome": outcome, "next_step": next_step, "selected_lever": None},
        "governance": _governance(),
    }


def synthesize(paths: Sequence[Path], cpu_preflight_path: Path) -> dict[str, Any]:
    return synthesize_loaded(load_inputs(paths, 8), cpu_preflight_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu-preflight", "final"), required=True)
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--cpu-preflight", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    output, execution = runtime_probe.prepare_execution(runtime_probe.parse_prelaunch_output(argv))
    args = parse_args(argv)
    if args.mode == "cpu-preflight":
        require(len(args.report) == 4 and args.cpu_preflight is None, "CPU preflight mode requires four reports and no --cpu-preflight input")
        require(output.resolve() == probe.CPU_PREFLIGHT_PATH.resolve(), "CPU preflight output must use canonical path")
        report = synthesize_cpu_preflight_loaded(load_inputs(args.report, 4))
    else:
        require(len(args.report) == 8 and args.cpu_preflight is not None, "final mode requires eight reports and --cpu-preflight")
        require(output.resolve() == FINAL_SYNTHESIS_PATH.resolve(), "final synthesis output must use canonical path")
        report = synthesize(args.report, cast(Path, args.cpu_preflight))
    report["created_at_utc"] = execution["started_at_utc"]
    report["execution"] = execution
    runtime_probe._write_json_atomic(output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
