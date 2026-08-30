#!/usr/bin/env python3
"""Run the preregistered G009-5-E012 rev19 contact-offset A/B probe.

This diagnostic-only probe reuses the rev18 raw-contact observation loop, but
creates a new within-revision control: both arms use solver 8/0 and call the
same startup-event ``root_physx_view.set_contact_offsets`` path, with scales
1.0 and 1.5 respectively.  Rest-offset setters and USD Apply are forbidden.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for search_root in (SCRIPT_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import probe_g009_r0_rev18_gpu_raw_contact as base_probe
import probe_g009_recover_runtime as runtime_probe
import summarize_g009_r0_rev18_gpu_raw_contact as base_summary


DEFAULT_TASK = base_probe.DEFAULT_TASK
SCHEMA_VERSION = "g009.r0.rev19.contact_offset_intervention.v1"
FAILURE_SCHEMA_VERSION = "g009.r0.rev19.contact_offset_intervention_failure.v1"
PREREGISTRATION_PATH = REPO_ROOT / "configs/g009_r0_rev19_contact_offset_intervention.json"
PREDECESSOR_PATH = REPO_ROOT / "reports/runs/g009_r0_rev18_raw_contact_feasibility_synthesis_2x2_s42.json"
PREDECESSOR_SHA256 = "9ca8007d88e771a5f24ca68afa46a670097e733f9e613c31fc4cc62f3fb9e01e"
CPU_PREFLIGHT_PATH = REPO_ROOT / "reports/runs/g009_r0_rev19_contact_offset_cpu_preflight_2x2_s42.json"
CPU_PREFLIGHT_SCHEMA_VERSION = "g009.r0.rev19.contact_offset_cpu_preflight.v1"
NUM_ENVS = 8
SOURCE_ENV_INDEX = 7
POSE_ID = "right_side"
ACTION_MODE = "reset_pose_hold"
PHYSICS_SUBSTEPS = 150
PHYSICS_DT_S = 0.005
POSITION_SOLVER_ITERATIONS = 8
VELOCITY_SOLVER_ITERATIONS = 0
ARM_SCALES = {"A": 1.0, "B": 1.5}
COLLISION_SHAPES_PER_ARTICULATION = 27
BODY_COUNT = 19
HARD_JOINT_LIMIT_MARGIN_RAD = 0.01
NON_FOOT_PEAK_FORCE_BODY_WEIGHT_MAX = 15.0
CPU_RAW_MINIMUM_SEPARATION_M = -0.01
AUTHORITY_SCOPE = base_probe.AUTHORITY_SCOPE
SOURCE_BINDING_PATHS = (
    "configs/g009_r0.json",
    "configs/g009_r0_rev19_contact_offset_intervention.json",
    "scripts/probe_g009_recover_runtime.py",
    "scripts/probe_g009_r0_rev18_gpu_raw_contact.py",
    "scripts/probe_g009_r0_rev19_contact_offset_intervention.py",
    "reports/runs/g009_r0_rev18_raw_contact_feasibility_synthesis_2x2_s42.json",
    "src/isaac_walk_g009/mdp/events.py",
    "src/isaac_walk_g009/recover_contracts.py",
    "src/isaac_walk_g009/recover_env_cfg.py",
    "src/isaac_walk_g009/registry.py",
)
SYNTHESIS_SOURCE_BINDING_PATHS = (
    "configs/g009_r0_rev19_contact_offset_intervention.json",
    "scripts/probe_g009_r0_rev18_gpu_raw_contact.py",
    "scripts/probe_g009_r0_rev19_contact_offset_intervention.py",
    "scripts/summarize_g009_r0_rev18_gpu_raw_contact.py",
    "scripts/summarize_g009_r0_rev19_contact_offset_intervention.py",
    "scripts/probe_g009_recover_runtime.py",
    "src/isaac_walk_g009/recover_contracts.py",
)
_STARTUP_EVIDENCE_SINK: dict[str, Any] | None = None


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def governance() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "headless_required": True,
        "rendering_allowed": False,
        "ppo_updates": 0,
        "reward_computed": False,
        "qualification_eligible": False,
        "gate_execution_allowed": False,
        "learned": False,
        "physics_ground_truth_authority": False,
        "authority_scope": AUTHORITY_SCOPE,
        "proxy_can_upgrade_raw_observation": False,
        "manual_probe_safety_not_gate": True,
    }


def synthesis_governance() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "selected_lever": None,
        "learned": False,
        "locomotion_or_physics_performance_approved": False,
        "physics_ground_truth_authority": False,
        "ppo": {"allowed": False, "status": "not_run", "updates": 0},
        "qualification": {"eligible": False, "status": "not_run", "passed": None},
        "gate01": {"allowed": False, "status": "forbidden"},
    }


def load_preregistration() -> dict[str, Any]:
    value = json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "rev19 preregistration root must be an object")
    require(
        value.get("schema_version") == "g009.r0.rev19.contact_offset_intervention_preregistration.v1"
        and value.get("evidence_id") == "G009-5-E012"
        and value.get("revision") == "rev19"
        and value.get("seed") == 42,
        "rev19 preregistration identity mismatch",
    )
    design = value.get("design", {})
    runtime = value.get("runtime", {})
    safety = value.get("manual_probe_safety", {})
    require(
        design.get("solver_position_iterations") == POSITION_SOLVER_ITERATIONS
        and design.get("solver_velocity_iterations") == VELOCITY_SOLVER_ITERATIONS
        and design.get("arms", {}).get("A", {}).get("contact_offset_scale") == 1.0
        and design.get("arms", {}).get("B", {}).get("contact_offset_scale") == 1.5
        and design.get("arms", {}).get("A", {}).get("contact_offset_setter_called") is True
        and design.get("arms", {}).get("B", {}).get("contact_offset_setter_called") is True
        and design.get("rest_offset_setter_called") is False
        and design.get("third_run_majority_vote_allowed") is False
        and runtime.get("num_envs") == NUM_ENVS
        and runtime.get("source_env_index") == SOURCE_ENV_INDEX
        and runtime.get("physics_substeps") == PHYSICS_SUBSTEPS
        and runtime.get("physics_dt_s") == PHYSICS_DT_S
        and runtime.get("usd_schema_apply_allowed") is False
        and value.get("measured_shape_topology", {}).get("collision_shapes_per_articulation") == COLLISION_SHAPES_PER_ARTICULATION
        and value.get("measured_shape_topology", {}).get("absolute_contact_offset_hardcode_allowed") is False
        and value.get("measured_shape_topology", {}).get("tensor_column_path_mapping_authority") is False
        and value.get("cpu_preflight", {}).get("canonical_path") == CPU_PREFLIGHT_PATH.relative_to(REPO_ROOT).as_posix()
        and value.get("cpu_preflight", {}).get("required_before_gpu_app_launcher") is True
        and safety.get("hard_joint_limit_margin_rad") == HARD_JOINT_LIMIT_MARGIN_RAD
        and safety.get("non_foot_peak_force_body_weight_max") == NON_FOOT_PEAK_FORCE_BODY_WEIGHT_MAX
        and safety.get("cpu_raw_minimum_separation_m") == CPU_RAW_MINIMUM_SEPARATION_M,
        "rev19 preregistration constants changed",
    )
    return value


def validate_predecessor() -> dict[str, str]:
    require(PREDECESSOR_PATH.is_file(), "rev18 E011 predecessor is missing")
    actual = sha256_bytes(PREDECESSOR_PATH.read_bytes())
    require(actual == PREDECESSOR_SHA256, "rev18 E011 predecessor SHA256 mismatch")
    return {"path": PREDECESSOR_PATH.relative_to(REPO_ROOT).as_posix(), "sha256": actual}


def current_git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def committed_blob_sha256(relative_path: str, commit: str) -> str:
    require(relative_path in SOURCE_BINDING_PATHS, "unexpected source bundle path")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "invalid Git commit")
    result = subprocess.run(["git", "show", f"{commit}:{relative_path}"], cwd=REPO_ROOT, check=True, capture_output=True)
    return sha256_bytes(result.stdout)


def source_binding_status() -> list[str]:
    return subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_BINDING_PATHS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def source_bundle_provenance() -> dict[str, Any]:
    commit = current_git_commit()
    dirty = source_binding_status()
    files: dict[str, str] = {}
    missing: list[str] = []
    for relative in SOURCE_BINDING_PATHS:
        if not (REPO_ROOT / relative).is_file():
            missing.append(relative)
            continue
        try:
            files[relative] = committed_blob_sha256(relative, commit)
        except (subprocess.CalledProcessError, ValueError):
            missing.append(relative)
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return {
        "schema_version": 1,
        "git_commit": commit,
        "git_commit_valid": bool(re.fullmatch(r"[0-9a-f]{40}", commit)),
        "source_binding_paths": list(SOURCE_BINDING_PATHS),
        "source_binding_files": files,
        "source_bundle_sha256": sha256_bytes(payload.encode("utf-8")) if files else None,
        "all_files_present": not missing and len(files) == len(SOURCE_BINDING_PATHS),
        "missing_files": missing,
        "clean": not dirty,
        "dirty_source_paths": dirty,
    }


def validate_source_bundle(bundle: Any) -> dict[str, Any]:
    require(isinstance(bundle, dict), "source bundle must be an object")
    bundle = cast(dict[str, Any], bundle)
    require(
        set(bundle) == {"schema_version", "git_commit", "git_commit_valid", "source_binding_paths", "source_binding_files", "source_bundle_sha256", "all_files_present", "missing_files", "clean", "dirty_source_paths"},
        "source bundle schema mismatch",
    )
    commit = bundle.get("git_commit")
    require(isinstance(commit, str) and bool(re.fullmatch(r"[0-9a-f]{40}", commit)) and bundle.get("git_commit_valid") is True, "source bundle commit mismatch")
    require(bundle.get("source_binding_paths") == list(SOURCE_BINDING_PATHS), "source bundle path order mismatch")
    files = bundle.get("source_binding_files")
    require(isinstance(files, dict) and list(files) == list(SOURCE_BINDING_PATHS), "source bundle file map mismatch")
    files = cast(dict[str, Any], files)
    require(all(isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value)) for value in files.values()), "source bundle file hash invalid")
    require(bundle.get("all_files_present") is True and bundle.get("missing_files") == [] and bundle.get("clean") is True and bundle.get("dirty_source_paths") == [], "source bundle must be complete and clean")
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    require(bundle.get("source_bundle_sha256") == sha256_bytes(payload.encode("utf-8")), "source bundle aggregate hash mismatch")
    for path in SOURCE_BINDING_PATHS:
        require(files[path] == committed_blob_sha256(path, cast(str, commit)), f"source bundle committed blob mismatch: {path}")
    return bundle


def probe_contract(arm: str, device: str, replicate_index: int) -> dict[str, Any]:
    arm = arm.upper()
    device = device.lower()
    require(arm in ARM_SCALES, "arm must be A or B")
    require(device in {"cpu", "cuda:0"}, "device must be cpu or cuda:0")
    require(replicate_index in {1, 2}, "replicate_index must be 1 or 2")
    return {
        "goal_id": "g009",
        "stage_id": "R0",
        "experiment_id": "G009-5-E012",
        "revision": "rev19",
        "preregistration": {"path": PREREGISTRATION_PATH.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_bytes(PREREGISTRATION_PATH.read_bytes())},
        "predecessor": {"path": PREDECESSOR_PATH.relative_to(REPO_ROOT).as_posix(), "sha256": PREDECESSOR_SHA256},
        "controlled_cell": {
            "arm": arm,
            "contact_offset_scale": ARM_SCALES[arm],
            "contact_offset_setter_called": True,
            "rest_offset_setter_called": False,
            "solver_position_iterations": POSITION_SOLVER_ITERATIONS,
            "solver_velocity_iterations": VELOCITY_SOLVER_ITERATIONS,
            "seed": 42,
            "num_envs": NUM_ENVS,
            "source_env_index": SOURCE_ENV_INDEX,
            "pose_id": POSE_ID,
            "action_mode": ACTION_MODE,
            "device": device,
            "replicate_index": replicate_index,
        },
        "execution": {"manual_inner_loop": True, "physics_substeps": PHYSICS_SUBSTEPS, "physics_dt_s": PHYSICS_DT_S, "simulated_duration_s": 0.75, "headless": True, "render": False, "ppo_updates": 0, "gate_runs": 0},
        "comparison_authority": {"scope": "rev19_arm_A_vs_arm_B_only", "rev18_solver_16_as_control_allowed": False, "third_run_majority_vote_allowed": False},
        "cpu_preflight": {
            "required_before_app_launcher": device == "cuda:0",
            "canonical_path": CPU_PREFLIGHT_PATH.relative_to(REPO_ROOT).as_posix(),
        },
    }


def cpu_preflight_not_required_binding() -> dict[str, Any]:
    return {
        "status": "not_required_for_cpu",
        "path": None,
        "sha256": None,
        "git_commit": None,
        "probe_source_bundle_sha256": None,
    }


def _canonical_cpu_report_paths() -> list[str]:
    return [
        expected_output_relative(arm, "cpu", replicate)
        for arm in ("A", "B")
        for replicate in (1, 2)
    ]


def committed_synthesis_blob_sha256(relative_path: str, commit: str) -> str:
    require(relative_path in SYNTHESIS_SOURCE_BINDING_PATHS, "unexpected synthesis source path")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "invalid synthesis commit")
    result = subprocess.run(["git", "show", f"{commit}:{relative_path}"], cwd=REPO_ROOT, check=True, capture_output=True)
    return sha256_bytes(result.stdout)


def validate_synthesis_source_bundle(bundle: Any, expected_git_commit: str) -> dict[str, Any]:
    require(isinstance(bundle, dict), "synthesis source bundle must be an object")
    bundle = cast(dict[str, Any], bundle)
    require(set(bundle) == {"schema_version", "role", "git_commit", "git_commit_valid", "source_binding_paths", "source_binding_files", "source_bundle_sha256", "all_files_present", "missing_files", "clean", "dirty_source_paths"}, "synthesis source schema mismatch")
    commit = bundle.get("git_commit")
    require(bundle.get("schema_version") == 1 and bundle.get("role") == "offline_synthesis_implementation", "synthesis source identity mismatch")
    require(isinstance(commit, str) and commit == expected_git_commit and bool(re.fullmatch(r"[0-9a-f]{40}", commit)) and bundle.get("git_commit_valid") is True, "synthesis source commit mismatch")
    require(bundle.get("source_binding_paths") == list(SYNTHESIS_SOURCE_BINDING_PATHS), "synthesis source path order mismatch")
    files = bundle.get("source_binding_files")
    require(isinstance(files, Mapping) and list(files) == list(SYNTHESIS_SOURCE_BINDING_PATHS), "synthesis source file map mismatch")
    files = cast(Mapping[str, Any], files)
    require(all(isinstance(files[path], str) and bool(re.fullmatch(r"[0-9a-f]{64}", cast(str, files[path]))) for path in SYNTHESIS_SOURCE_BINDING_PATHS), "synthesis source file hash invalid")
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    require(bundle.get("source_bundle_sha256") == sha256_bytes(payload.encode("utf-8")), "synthesis source aggregate hash mismatch")
    require(bundle.get("all_files_present") is True and bundle.get("missing_files") == [] and bundle.get("clean") is True and bundle.get("dirty_source_paths") == [], "synthesis source bundle must be complete and clean")
    for path in SYNTHESIS_SOURCE_BINDING_PATHS:
        require(files[path] == committed_synthesis_blob_sha256(path, commit), f"synthesis source committed blob mismatch: {path}")
    return bundle


def repeatability_row(report: Mapping[str, Any], derived: Mapping[str, Any]) -> dict[str, Any]:
    offset = cast(Mapping[str, Any], report["offset_integrity"])
    before = cast(Mapping[str, Any], offset["before"])
    after = cast(Mapping[str, Any], offset["after"])
    safety = cast(Mapping[str, Any], report["manual_probe_safety"])
    raw_passed = derived.get("raw_observation_passed") is True
    return {
        "raw_observation_passed": raw_passed,
        "offset_baseline_contact_sha256": cast(Mapping[str, Any], before["contact_offset"])["sha256"],
        "offset_baseline_rest_sha256": cast(Mapping[str, Any], before["rest_offset"])["sha256"],
        "offset_after_contact_sha256": cast(Mapping[str, Any], after["contact_offset"])["sha256"],
        "safety_available": safety.get("available") is True,
        "safety_passed": safety.get("passed") is True,
        "unavailable_signature": None if raw_passed else base_summary._unavailable_signature(report, derived),
        "report": report,
    }


def cell_repeatability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(rows) == 2, "cell repeatability requires two replicates")
    raw = base_summary._device_repeatability(rows)
    offset_exact = rows[0]["offset_baseline_contact_sha256"] == rows[1]["offset_baseline_contact_sha256"] and rows[0]["offset_baseline_rest_sha256"] == rows[1]["offset_baseline_rest_sha256"] and rows[0]["offset_after_contact_sha256"] == rows[1]["offset_after_contact_sha256"]
    safety_status_exact = rows[0]["safety_available"] == rows[1]["safety_available"] and rows[0]["safety_passed"] == rows[1]["safety_passed"]
    return {**raw, "raw_repeatable": raw["repeatable"] is True, "offset_hashes_exact": offset_exact, "safety_status_exact": safety_status_exact, "repeatable": raw["repeatable"] is True and offset_exact and safety_status_exact}


def validate_cpu_preflight_artifact(
    path: Path,
    expected_source_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    require(resolved == CPU_PREFLIGHT_PATH.resolve(), "CPU preflight must use canonical path")
    require(resolved.parent == (REPO_ROOT / "reports/runs").resolve(), "CPU preflight must be a direct reports/runs child")
    raw = resolved.read_bytes()
    value = json.loads(raw.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {item}")))
    require(isinstance(value, dict), "CPU preflight root must be an object")
    require(
        set(value)
        == {
            "schema_version", "evidence_id", "goal_id", "stage_id", "revision",
            "status", "mode", "input_report_count", "input_reports", "integrity",
            "cpu_preflight", "decision", "governance", "synthesis_source_bundle",
            "created_at_utc", "execution",
        },
        "CPU preflight top-level schema mismatch",
    )
    require(
        value.get("schema_version") == CPU_PREFLIGHT_SCHEMA_VERSION
        and value.get("evidence_id") == "G009-5-E012"
        and value.get("goal_id") == "g009"
        and value.get("stage_id") == "R0"
        and value.get("revision") == "rev19"
        and value.get("status") == "complete"
        and value.get("mode") == "cpu_preflight_2x2"
        and value.get("input_report_count") == 4,
        "CPU preflight identity mismatch",
    )
    bindings = value.get("input_reports")
    require(isinstance(bindings, list) and len(bindings) == 4 and all(isinstance(item, Mapping) and set(item) == {"path", "sha256"} and isinstance(item.get("sha256"), str) and bool(re.fullmatch(r"[0-9a-f]{64}", cast(str, item.get("sha256")))) for item in bindings) and [item.get("path") for item in bindings if isinstance(item, Mapping)] == _canonical_cpu_report_paths(), "CPU preflight exact report order/binding mismatch")
    integrity = value.get("integrity")
    require(
        isinstance(integrity, Mapping)
        and set(integrity) == {"passed", "hash_bound", "unique_execution_ids", "exact_slots", "git_commit", "probe_source_bundle_sha256", "synthesis_source_bundle_sha256", "mass_tensor_sha256", "mass_body_names_sha256"}
        and integrity.get("passed") is True
        and integrity.get("hash_bound") is True
        and integrity.get("unique_execution_ids") is True
        and integrity.get("exact_slots") == ["A.cpu.rep1", "A.cpu.rep2", "B.cpu.rep1", "B.cpu.rep2"]
        and integrity.get("git_commit") == expected_source_bundle.get("git_commit")
        and integrity.get("probe_source_bundle_sha256") == expected_source_bundle.get("source_bundle_sha256"),
        "CPU preflight source/integrity binding mismatch",
    )
    synthesis_bundle = validate_synthesis_source_bundle(value.get("synthesis_source_bundle"), cast(str, expected_source_bundle.get("git_commit")))
    require(integrity.get("synthesis_source_bundle_sha256") == synthesis_bundle.get("source_bundle_sha256"), "CPU preflight synthesis bundle digest mismatch")
    require(value.get("governance") == synthesis_governance(), "CPU preflight governance mismatch")
    created = value.get("created_at_utc")
    require(isinstance(created, str) and created.endswith("Z"), "CPU preflight created timestamp must be UTC")
    try:
        created_time = datetime.fromisoformat(cast(str, created)[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("CPU preflight created timestamp must be UTC") from error
    require(created_time.utcoffset() == timezone.utc.utcoffset(created_time), "CPU preflight created timestamp must be UTC")
    execution = value.get("execution")
    require(isinstance(execution, Mapping) and set(execution) == {"execution_id", "started_at_utc", "output_path_repo_relative", "no_overwrite"}, "CPU preflight execution schema mismatch")
    execution_id = execution.get("execution_id") if isinstance(execution, Mapping) else None
    require(isinstance(execution_id, str), "CPU preflight execution UUID missing")
    try:
        parsed_execution_id = uuid.UUID(hex=cast(str, execution_id))
    except ValueError as error:
        raise ValueError("CPU preflight execution UUID invalid") from error
    require(
        parsed_execution_id.version == 4
        and parsed_execution_id.hex == execution_id
        and execution.get("started_at_utc") == created
        and execution.get("output_path_repo_relative") == CPU_PREFLIGHT_PATH.relative_to(REPO_ROOT).as_posix()
        and execution.get("no_overwrite") is True,
        "CPU preflight execution binding mismatch",
    )
    execution_ids: list[str] = []
    source_payload: Any = None
    repeatability_rows: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    mass_tensor_hashes: set[str] = set()
    mass_body_hashes: set[str] = set()
    for binding in cast(list[Mapping[str, Any]], bindings):
        relative = binding.get("path")
        digest = binding.get("sha256")
        require(isinstance(relative, str) and isinstance(digest, str), "CPU preflight input binding invalid")
        report_path = (REPO_ROOT / cast(str, relative)).resolve(strict=True)
        require(report_path.parent == (REPO_ROOT / "reports/runs").resolve() and sha256_bytes(report_path.read_bytes()) == digest, "CPU preflight input hash mismatch")
        report = json.loads(report_path.read_text(encoding="utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {item}")))
        require(isinstance(report, dict), "CPU preflight report root invalid")
        derived = validate_report(report, validate_gpu_preflight=False)
        require(report.get("device") == "cpu" and report.get("cpu_preflight_binding") == cpu_preflight_not_required_binding(), "CPU preflight contains non-CPU or recursively bound report")
        feasibility = report.get("feasibility", {})
        safety = report.get("manual_probe_safety", {})
        require(feasibility.get("raw_observation_passed") is True and feasibility.get("probe_valid") is True and safety.get("available") is True and safety.get("passed") is True, "CPU preflight report is not ready")
        execution_ids.append(report["execution"]["execution_id"])
        arm = cast(str, report.get("arm"))
        require(arm in repeatability_rows, "CPU preflight report arm invalid")
        repeatability_rows[arm].append(repeatability_row(report, derived))
        mass = cast(Mapping[str, Any], cast(Mapping[str, Any], report["manual_probe_safety"])["mass_evidence"])
        mass_tensor_hashes.add(cast(str, cast(Mapping[str, Any], mass["tensor"])["sha256"]))
        mass_body_hashes.add(cast(str, mass["body_names_sha256"]))
        if source_payload is None:
            source_payload = report["source_bundle"]
        require(report["source_bundle"] == source_payload, "CPU preflight report source bundle drift")
    require(len(set(execution_ids)) == 4 and source_payload == dict(expected_source_bundle), "CPU preflight execution/source uniqueness mismatch")
    require(len(mass_tensor_hashes) == 1 and integrity.get("mass_tensor_sha256") == next(iter(mass_tensor_hashes)) and len(mass_body_hashes) == 1 and integrity.get("mass_body_names_sha256") == next(iter(mass_body_hashes)), "CPU preflight mass integrity mismatch")
    recomputed_repeatability = {f"{arm}.cpu": cell_repeatability(repeatability_rows[arm]) for arm in ("A", "B")}
    require(all(item.get("repeatable") is True for item in recomputed_repeatability.values()), "CPU preflight source reports are not repeatable")
    require(value.get("cpu_preflight") == {"passed": True, "raw_pass_probe_valid_safety_pass": True, "within_arm_repeatability_passed": True, "gpu_stage_allowed": True}, "CPU preflight did not authorize GPU")
    require(value.get("decision") == {"outcome": "gpu_stage_authorized", "selected_lever": None, "third_run_majority_vote_allowed": False, "repeatability": recomputed_repeatability}, "CPU preflight decision/repeatability mismatch")
    return {
        "status": "validated_for_gpu",
        "path": resolved.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_bytes(raw),
        "git_commit": integrity["git_commit"],
        "probe_source_bundle_sha256": integrity["probe_source_bundle_sha256"],
    }


def expected_output_relative(arm: str, device: str, replicate_index: int) -> str:
    require(arm.upper() in ARM_SCALES, "arm must be A or B")
    require(device.lower() in {"cpu", "cuda:0"}, "device must be cpu or cuda:0")
    require(replicate_index in {1, 2}, "replicate_index must be 1 or 2")
    label = "cpu" if device.lower() == "cpu" else "gpu"
    return f"reports/runs/g009_r0_rev19_contact_offset_arm{arm.upper()}_{label}_rep0{replicate_index}_s42.json"


def validate_execution_metadata(execution: Any, arm: str, device: str, replicate_index: int) -> dict[str, Any]:
    require(isinstance(execution, dict), "execution metadata must be an object")
    execution = cast(dict[str, Any], execution)
    require(set(execution) == {"execution_id", "started_at_utc", "output_path_repo_relative", "no_overwrite"}, "execution metadata key set mismatch")
    execution_id = execution.get("execution_id")
    require(isinstance(execution_id, str), "execution_id must be UUID4 lowercase hex")
    try:
        parsed = uuid.UUID(hex=cast(str, execution_id))
    except ValueError as error:
        raise ValueError("execution_id must be UUID4 lowercase hex") from error
    require(parsed.version == 4 and parsed.hex == execution_id, "execution_id must be UUID4 lowercase hex")
    started = execution.get("started_at_utc")
    require(isinstance(started, str) and started.endswith("Z"), "started_at_utc must be UTC")
    parsed_time = datetime.fromisoformat(cast(str, started)[:-1] + "+00:00")
    require(parsed_time.utcoffset() == timezone.utc.utcoffset(parsed_time), "started_at_utc must be UTC")
    require(execution.get("output_path_repo_relative") == expected_output_relative(arm, device, replicate_index), "execution output binding mismatch")
    require(execution.get("no_overwrite") is True, "execution must be no-overwrite")
    return execution


def _tensor_values(tensor: Any) -> Any:
    value = tensor.detach() if hasattr(tensor, "detach") else tensor
    value = value.cpu() if hasattr(value, "cpu") else value
    return value.tolist() if hasattr(value, "tolist") else value


def tensor_snapshot(tensor: Any) -> dict[str, Any]:
    values = _tensor_values(tensor)
    require(isinstance(values, list), "offset tensor must serialize to a list")
    shape = list(getattr(tensor, "shape", ()))
    flat: list[float] = []

    def flatten(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                flatten(item)
        else:
            number = float(value)
            require(math.isfinite(number), "offset tensor must be finite")
            flat.append(number)

    flatten(values)
    require(flat, "offset tensor must not be empty")
    payload = json.dumps({"shape": shape, "values": values}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"shape": shape, "dtype": str(getattr(tensor, "dtype", "unknown")), "device": str(getattr(tensor, "device", "unknown")), "sha256": sha256_bytes(payload), "minimum": min(flat), "maximum": max(flat), "values": values}


def _snapshots_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.get("shape") == right.get("shape") and left.get("sha256") == right.get("sha256") and left.get("values") == right.get("values")


def apply_contact_offset_intervention(root_physx_view: Any, arm: str, torch_module: Any) -> dict[str, Any]:
    arm = arm.upper()
    require(arm in ARM_SCALES, "arm must be A or B")
    before_contact_tensor = root_physx_view.get_contact_offsets().clone()
    before_rest_tensor = root_physx_view.get_rest_offsets().clone()
    require(len(before_contact_tensor.shape) == 2 and tuple(int(value) for value in before_contact_tensor.shape) == (NUM_ENVS, COLLISION_SHAPES_PER_ARTICULATION), "contact offsets must cover 8 envs x 27 collision shapes")
    require(tuple(before_contact_tensor.shape) == tuple(before_rest_tensor.shape), "contact/rest offset shape mismatch")
    env_ids = torch_module.arange(NUM_ENVS, device="cpu")
    root_physx_view.set_contact_offsets(before_contact_tensor * ARM_SCALES[arm], env_ids)
    setter_called = True
    after_contact_tensor = root_physx_view.get_contact_offsets().clone()
    after_rest_tensor = root_physx_view.get_rest_offsets().clone()
    expected_contact_tensor = before_contact_tensor * ARM_SCALES[arm]
    contact_matches = bool(torch_module.equal(after_contact_tensor, expected_contact_tensor))
    rest_unchanged = bool(torch_module.equal(before_rest_tensor, after_rest_tensor))
    contact_above_rest = bool(torch_module.all(after_contact_tensor > after_rest_tensor).item())
    before_contact = tensor_snapshot(before_contact_tensor)
    after_contact = tensor_snapshot(after_contact_tensor)
    before_rest = tensor_snapshot(before_rest_tensor)
    after_rest = tensor_snapshot(after_rest_tensor)
    return {
        "api": "root_physx_view_get_set",
        "application_boundary": "env_cfg_custom_startup_event_during_gym_make_before_first_reset_or_physics_step",
        "usd_schema_apply_called": False,
        "arm": arm,
        "contact_offset_scale": ARM_SCALES[arm],
        "contact_offset_setter_called": setter_called,
        "rest_offset_setter_called": False,
        "shape_tensor_columns": [f"shape_index_{index:02d}" for index in range(COLLISION_SHAPES_PER_ARTICULATION)],
        "tensor_column_path_mapping_authority": False,
        "before": {"contact_offset": before_contact, "rest_offset": before_rest},
        "after": {"contact_offset": after_contact, "rest_offset": after_rest},
        "checks": {
            "all_envs_all_shapes_recorded": before_contact["shape"] == [NUM_ENVS, COLLISION_SHAPES_PER_ARTICULATION],
            "heterogeneous_baseline_observed": float(before_contact["minimum"]) < float(before_contact["maximum"]),
            "rest_baseline_zero": float(before_rest["minimum"]) == 0.0 and float(before_rest["maximum"]) == 0.0,
            "contact_after_matches_scale": contact_matches,
            "contact_unchanged_for_control": arm != "A" or _snapshots_equal(before_contact, after_contact),
            "rest_bitwise_unchanged": rest_unchanged and _snapshots_equal(before_rest, after_rest),
            "contact_strictly_above_rest": contact_above_rest,
            "symmetric_setter_policy": setter_called,
        },
    }


def startup_scale_contact_offsets(env: Any, env_ids: Any, arm: str) -> None:
    """Apply the preregistered heterogeneous-vector scale as a startup event."""

    import torch

    require(env_ids is None, "rev19 contact-offset startup event must target all envs")
    global _STARTUP_EVIDENCE_SINK
    require(_STARTUP_EVIDENCE_SINK is not None and not _STARTUP_EVIDENCE_SINK, "startup evidence sink must be armed exactly once")
    sink = cast(dict[str, Any], _STARTUP_EVIDENCE_SINK)
    evidence = apply_contact_offset_intervention(env.scene["robot"].root_physx_view, arm, torch)
    evidence["topology"] = collision_topology_evidence(env)
    sink.update(evidence)
    env._g009_rev19_contact_offset_evidence = copy.deepcopy(evidence)


def _collision_topology_from_prims(prims: Any, collision_api: Any) -> dict[str, Any]:
    """Normalize read-only collision prim observations across cloned environments."""

    records = [
        {
            "path": str(prim.GetPath()),
            "is_instance_proxy": bool(prim.IsInstanceProxy()),
        }
        for prim in prims
        if str(prim.GetPath()).startswith("/World/envs/env_")
        and prim.HasAPI(collision_api)
    ]
    per_env: dict[str, list[str]] = {}
    per_env_records: dict[str, list[dict[str, Any]]] = {}
    per_env_counts: dict[str, int] = {}
    per_env_unique_counts: dict[str, int] = {}
    per_env_instance_proxy_counts: dict[str, int] = {}
    template: list[str] | None = None
    for env_index in range(NUM_ENVS):
        prefix = f"/World/envs/env_{env_index}/Robot/"
        env_records = sorted(
            (
                {
                    "path": record["path"],
                    "is_instance_proxy": record["is_instance_proxy"],
                }
                for record in records
                if record["path"].startswith(prefix)
            ),
            key=lambda record: record["path"],
        )
        paths = [record["path"] for record in env_records]
        unique_count = len(set(paths))
        per_env_counts[str(env_index)] = len(paths)
        per_env_unique_counts[str(env_index)] = unique_count
        per_env_instance_proxy_counts[str(env_index)] = sum(
            1
            for record in records
            if record["path"].startswith(prefix) and record["is_instance_proxy"]
        )
        require(
            len(paths) == COLLISION_SHAPES_PER_ARTICULATION and unique_count == len(paths),
            "collision topology must contain 27 unique paths per env; "
            f"observed_counts={json.dumps(per_env_counts, sort_keys=True)}; "
            f"observed_unique_counts={json.dumps(per_env_unique_counts, sort_keys=True)}",
        )
        relative = [path[len(prefix):] for path in paths]
        if template is None:
            template = relative
        require(
            relative == template,
            "collision topology differs across envs; "
            f"env_index={env_index}; observed_relative_paths={json.dumps(relative)}",
        )
        per_env[str(env_index)] = paths
        per_env_records[str(env_index)] = env_records
    require(template is not None and len(template) == COLLISION_SHAPES_PER_ARTICULATION, "collision topology template unavailable")
    return {
        "source": "read_only_usd_collision_api_traversal",
        "traversal_predicate": "Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies())",
        "instance_proxy_traversal_enabled": True,
        "collision_shapes_per_articulation": COLLISION_SHAPES_PER_ARTICULATION,
        "collision_shape_paths_total": sum(per_env_counts.values()),
        "sorted_unique_template_paths": template,
        "per_env_sorted_paths": per_env,
        "per_env_sorted_path_records": per_env_records,
        "per_env_collision_path_counts": per_env_counts,
        "per_env_unique_collision_path_counts": per_env_unique_counts,
        "per_env_instance_proxy_collision_path_counts": per_env_instance_proxy_counts,
        "instance_proxy_collision_path_count": sum(per_env_instance_proxy_counts.values()),
        "all_envs_topology_identical": True,
        "setter_call_scope": "robot.root_physx_view_only",
        "ground_setter_called": False,
        "usd_schema_apply_called": False,
        "ground_runtime_offset_unchanged_claimed": False,
        "tensor_column_path_mapping_authority": False,
    }


def collision_topology_evidence(env: Any) -> dict[str, Any]:
    """Read collision prim topology without applying or mutating USD schemas."""

    import omni.usd  # pyright: ignore[reportMissingImports]
    from pxr import Usd, UsdPhysics  # pyright: ignore[reportMissingImports]

    stage = omni.usd.get_context().get_stage()
    require(stage is not None, "USD stage unavailable for collision topology")
    predicate = Usd.TraverseInstanceProxies()
    robot_root_paths = [f"/World/envs/env_{env_index}/Robot" for env_index in range(NUM_ENVS)]
    robot_root_instance_state: dict[str, dict[str, Any]] = {}
    collision_scope_prims: list[Any] = []
    for env_index, robot_root_path in enumerate(robot_root_paths):
        robot_root = stage.GetPrimAtPath(robot_root_path)
        require(
            bool(robot_root.IsValid()),
            f"robot root unavailable for collision topology: {robot_root_path}",
        )
        robot_root_instance_state[str(env_index)] = {
            "path": robot_root_path,
            "is_instance": bool(robot_root.IsInstance()),
            "is_instanceable": bool(robot_root.IsInstanceable()),
            "is_instance_proxy": bool(robot_root.IsInstanceProxy()),
        }
        collision_scope_prims.extend(Usd.PrimRange(robot_root, predicate))
    evidence = _collision_topology_from_prims(collision_scope_prims, UsdPhysics.CollisionAPI)
    evidence["robot_root_paths"] = robot_root_paths
    evidence["per_env_robot_root_instance_state"] = robot_root_instance_state
    evidence["robot_root_scope_validated"] = True
    return evidence


class SafetyAccumulator:
    def __init__(self) -> None:
        self.sample_count = 0
        self.finite_joint_position_and_contact_force = True
        self.hard_limit_with_margin = True
        self.all_env_non_foot_peak_bw = 0.0
        self.source_non_foot_peak_bw = 0.0
        self.per_env_non_foot_peak_bw = [0.0] * NUM_ENVS
        self.per_env_non_foot_peak_n = [0.0] * NUM_ENVS
        self.finite_violation_steps_all_env: list[int] = []
        self.finite_violation_steps_source_env: list[int] = []
        self.hard_limit_violation_steps_all_env: list[int] = []
        self.hard_limit_violation_steps_source_env: list[int] = []
        self.mass_snapshot: dict[str, Any] | None = None
        self.mass_body_names: list[str] | None = None
        self.mass_body_names_sha256: str | None = None
        self.mass_changed_steps: list[int] = []
        self.mass_tensor_reference: Any = None
        self.error: str | None = None

    def observe(self, sensor: Any, robot: Any, torch_module: Any) -> None:
        try:
            forces = sensor.data.net_forces_w.detach()
            positions = robot.data.joint_pos.detach()
            limits = robot.data.joint_pos_limits.detach()
            require(forces.shape[0] == NUM_ENVS and positions.shape[0] == NUM_ENVS, "safety env dimension mismatch")
            step = self.sample_count + 1
            finite_by_env = torch_module.isfinite(forces).all(dim=(1, 2)) & torch_module.isfinite(positions).all(dim=1)
            if not bool(finite_by_env.all().item()):
                self.finite_violation_steps_all_env.append(step)
            if not bool(finite_by_env[SOURCE_ENV_INDEX].item()):
                self.finite_violation_steps_source_env.append(step)
            self.finite_joint_position_and_contact_force = self.finite_joint_position_and_contact_force and bool(finite_by_env.all().item())
            lower = limits[..., 0] - HARD_JOINT_LIMIT_MARGIN_RAD
            upper = limits[..., 1] + HARD_JOINT_LIMIT_MARGIN_RAD
            hard_ok_by_env = ((positions >= lower) & (positions <= upper)).all(dim=1)
            if not bool(hard_ok_by_env.all().item()):
                self.hard_limit_violation_steps_all_env.append(step)
            if not bool(hard_ok_by_env[SOURCE_ENV_INDEX].item()):
                self.hard_limit_violation_steps_source_env.append(step)
            self.hard_limit_with_margin = self.hard_limit_with_margin and bool(hard_ok_by_env.all().item())
            names = list(sensor.body_names)
            non_foot = [index for index, name in enumerate(names) if "foot" not in name.lower()]
            require(non_foot, "non-foot body set is empty")
            magnitudes = torch_module.linalg.vector_norm(forces[:, non_foot, :], dim=-1)
            peak = magnitudes.amax(dim=1)
            mass_tensor = robot.data.default_mass.detach().to(device=forces.device)
            require(tuple(int(value) for value in mass_tensor.shape) == (NUM_ENVS, BODY_COUNT), "default_mass must be 8x19")
            body_names = list(robot.body_names)
            require(len(body_names) == BODY_COUNT and body_names == list(sensor.body_names) and len(set(body_names)) == BODY_COUNT, "mass/contact body ordering mismatch")
            if self.mass_snapshot is None:
                self.mass_tensor_reference = mass_tensor.clone()
                self.mass_snapshot = tensor_snapshot(mass_tensor)
                self.mass_body_names = body_names
                self.mass_body_names_sha256 = sha256_bytes(json.dumps(body_names, separators=(",", ":")).encode("utf-8"))
            elif not bool(torch_module.equal(mass_tensor, self.mass_tensor_reference)):
                self.mass_changed_steps.append(step)
            masses = mass_tensor.sum(dim=1)
            require(bool(torch_module.isfinite(mass_tensor).all().item()) and bool((mass_tensor > 0.0).all().item()), "default_mass must be finite and strictly positive")
            ratios = peak / (masses * 9.81)
            for env_index, (force_n, ratio) in enumerate(zip(peak.tolist(), ratios.tolist(), strict=True)):
                self.per_env_non_foot_peak_n[env_index] = max(self.per_env_non_foot_peak_n[env_index], float(force_n))
                self.per_env_non_foot_peak_bw[env_index] = max(self.per_env_non_foot_peak_bw[env_index], float(ratio))
            self.all_env_non_foot_peak_bw = max(self.all_env_non_foot_peak_bw, float(ratios.max().item()))
            self.source_non_foot_peak_bw = max(self.source_non_foot_peak_bw, float(ratios[SOURCE_ENV_INDEX].item()))
            self.sample_count += 1
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def snapshot(self, raw_observation: Mapping[str, Any], device: str) -> dict[str, Any]:
        separations_by_env: dict[int, list[float]] = {index: [] for index in range(NUM_ENVS)}
        for event in raw_observation.get("events", []):
            if not isinstance(event, Mapping):
                continue
            for header in event.get("headers", []):
                if not isinstance(header, Mapping) or type(header.get("env_index")) is not int:
                    continue
                env_index = cast(int, header["env_index"])
                if env_index not in separations_by_env:
                    continue
                for point in header.get("contact_points", []):
                    if isinstance(point, Mapping) and type(point.get("separation_m")) in (int, float):
                        separations_by_env[env_index].append(float(point["separation_m"]))
        per_env_minimum = {
            str(index): min(values) if values else None
            for index, values in separations_by_env.items()
        }
        observed_minima = [value for value in per_env_minimum.values() if value is not None]
        all_env_minimum = min(observed_minima) if observed_minima else None
        source_minimum = per_env_minimum[str(SOURCE_ENV_INDEX)]
        cpu_separation_observed = device != "cpu" or all(value is not None for value in per_env_minimum.values())
        checks = {
            "exact_150_manual_samples": self.sample_count == PHYSICS_SUBSTEPS,
            "finite_joint_position_and_contact_force": self.finite_joint_position_and_contact_force,
            "hard_joint_limit_with_margin": self.hard_limit_with_margin,
            "all_env_non_foot_peak_force_within_15_bw": self.all_env_non_foot_peak_bw <= NON_FOOT_PEAK_FORCE_BODY_WEIGHT_MAX,
            "source_env_non_foot_peak_force_within_15_bw": self.source_non_foot_peak_bw <= NON_FOOT_PEAK_FORCE_BODY_WEIGHT_MAX,
            "cpu_raw_minimum_separation_observed": cpu_separation_observed,
            "cpu_raw_minimum_separation_within_limit": device != "cpu" or (all_env_minimum is not None and all_env_minimum >= CPU_RAW_MINIMUM_SEPARATION_M and source_minimum is not None and source_minimum >= CPU_RAW_MINIMUM_SEPARATION_M),
            "default_mass_8x19_finite_positive_unchanged": self.mass_snapshot is not None and self.mass_changed_steps == [] and self.sample_count == PHYSICS_SUBSTEPS,
            "collection_error_absent": self.error is None,
        }
        complete = all(checks.values())
        available = self.error is None and self.sample_count == PHYSICS_SUBSTEPS and cpu_separation_observed
        require(self.mass_snapshot is not None and self.mass_body_names is not None and self.mass_body_names_sha256 is not None, "mass evidence unavailable")
        mass_snapshot = cast(dict[str, Any], self.mass_snapshot)
        mass_values = cast(list[list[float]], mass_snapshot["values"])
        total_mass = [math.fsum(float(item) for item in row) for row in mass_values]
        return {
            "label": "manual_probe_observation_not_gate",
            "required_scopes": ["all_envs", "source_env_7"],
            "thresholds": {"hard_joint_limit_margin_rad": HARD_JOINT_LIMIT_MARGIN_RAD, "non_foot_peak_force_body_weight_max": NON_FOOT_PEAK_FORCE_BODY_WEIGHT_MAX, "cpu_raw_minimum_separation_m": CPU_RAW_MINIMUM_SEPARATION_M},
            "mass_evidence": {"source": "robot.data.default_mass", "shape": [NUM_ENVS, BODY_COUNT], "body_names": self.mass_body_names, "body_names_sha256": self.mass_body_names_sha256, "tensor": mass_snapshot, "per_env_total_mass_kg": total_mass, "per_env_body_weight_n": [mass * 9.81 for mass in total_mass], "unchanged_for_150_steps": self.mass_changed_steps == [] and self.sample_count == PHYSICS_SUBSTEPS, "changed_steps": self.mass_changed_steps},
            "observations": {"sample_count": self.sample_count, "finite_violation_steps": {"all_envs": self.finite_violation_steps_all_env, "source_env_7": self.finite_violation_steps_source_env}, "hard_joint_limit_violation_steps": {"all_envs": self.hard_limit_violation_steps_all_env, "source_env_7": self.hard_limit_violation_steps_source_env}, "non_foot_peak_force_n_per_env": {str(index): value for index, value in enumerate(self.per_env_non_foot_peak_n)}, "non_foot_peak_force_body_weight_per_env": {str(index): value for index, value in enumerate(self.per_env_non_foot_peak_bw)}, "all_env_non_foot_peak_force_body_weight": self.all_env_non_foot_peak_bw, "source_env_non_foot_peak_force_body_weight": self.source_non_foot_peak_bw, "cpu_raw_minimum_separation_m": {"per_env": per_env_minimum, "all_env_minimum": all_env_minimum, "source_env_7_minimum": source_minimum}, "error": self.error},
            "checks": checks,
            "available": available,
            "passed": complete if available else None,
        }


def _base_view(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "schema_version", "goal_id", "stage_id", "experiment_id", "revision", "status", "headless", "device", "seed", "replicate_index", "num_envs", "source_env_index", "physics_substeps", "physics_dt_s", "manual_inner_loop", "finished_at_utc", "execution", "contract", "contract_sha256", "predecessor", "source_bundle", "governance", "pose_action_assignment", "live_physics_readback", "device_readback", "residual_capability", "physics_step_clock", "raw_contact_observation", "supporting_telemetry", "feasibility"
    }
    view = {key: copy.deepcopy(report[key]) for key in keys if key in report}
    view["schema_version"] = base_probe.SCHEMA_VERSION
    view["experiment_id"] = "G009-5-E011"
    view["revision"] = "rev18"
    return view


def derive_feasibility(report: Mapping[str, Any]) -> dict[str, Any]:
    base = base_probe.derive_feasibility(_base_view(report))
    offset = report.get("offset_integrity")
    offset_checks = offset.get("checks", {}) if isinstance(offset, Mapping) else {}
    safety = report.get("manual_probe_safety")
    safety_available = isinstance(safety, Mapping) and safety.get("available") is True
    safety_passed = isinstance(safety, Mapping) and safety.get("passed") is True
    solver = report.get("live_physics_readback", {}).get("solver", {}) if isinstance(report.get("live_physics_readback"), Mapping) else {}
    articulations = solver.get("articulations", []) if isinstance(solver, Mapping) else []
    solver_8_0 = isinstance(articulations, list) and len(articulations) == NUM_ENVS and all(isinstance(row, Mapping) and row.get("solver_position_iteration_count") == 8 and row.get("solver_velocity_iteration_count") == 0 for row in articulations)
    offset_integrity_passed = isinstance(offset_checks, Mapping) and all(offset_checks.get(name) is True for name in ("all_envs_all_shapes_recorded", "heterogeneous_baseline_observed", "rest_baseline_zero", "contact_after_matches_scale", "contact_unchanged_for_control", "rest_bitwise_unchanged", "contact_strictly_above_rest", "symmetric_setter_policy"))
    return {
        **base,
        "offset_integrity_passed": offset_integrity_passed,
        "solver_live_readback_8_0": solver_8_0,
        "manual_probe_safety_available": safety_available,
        "manual_probe_safety_passed": safety_passed,
        "run_interpretable": base["probe_valid"] and offset_integrity_passed and solver_8_0 and safety_available,
    }


def validate_collision_topology(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "collision topology must be an object")
    topology = cast(Mapping[str, Any], value)
    expected_keys = {
        "source",
        "traversal_predicate",
        "instance_proxy_traversal_enabled",
        "robot_root_paths",
        "per_env_robot_root_instance_state",
        "robot_root_scope_validated",
        "collision_shapes_per_articulation",
        "collision_shape_paths_total",
        "sorted_unique_template_paths",
        "per_env_sorted_paths",
        "per_env_sorted_path_records",
        "per_env_collision_path_counts",
        "per_env_unique_collision_path_counts",
        "per_env_instance_proxy_collision_path_counts",
        "instance_proxy_collision_path_count",
        "all_envs_topology_identical",
        "setter_call_scope",
        "ground_setter_called",
        "usd_schema_apply_called",
        "ground_runtime_offset_unchanged_claimed",
        "tensor_column_path_mapping_authority",
    }
    require(set(topology) == expected_keys, "collision topology schema mismatch")
    require(
        topology.get("source") == "read_only_usd_collision_api_traversal"
        and topology.get("traversal_predicate") == "Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies())"
        and topology.get("instance_proxy_traversal_enabled") is True
        and topology.get("robot_root_scope_validated") is True
        and topology.get("collision_shapes_per_articulation") == COLLISION_SHAPES_PER_ARTICULATION
        and topology.get("all_envs_topology_identical") is True
        and topology.get("setter_call_scope") == "robot.root_physx_view_only"
        and topology.get("ground_setter_called") is False
        and topology.get("usd_schema_apply_called") is False
        and topology.get("ground_runtime_offset_unchanged_claimed") is False
        and topology.get("tensor_column_path_mapping_authority") is False,
        "collision topology identity mismatch",
    )

    expected_env_keys = {str(index) for index in range(NUM_ENVS)}
    expected_root_paths = [f"/World/envs/env_{env_index}/Robot" for env_index in range(NUM_ENVS)]
    require(topology.get("robot_root_paths") == expected_root_paths, "collision topology robot roots mismatch")
    root_states = topology.get("per_env_robot_root_instance_state")
    per_env_paths = topology.get("per_env_sorted_paths")
    per_env_records = topology.get("per_env_sorted_path_records")
    serialized_counts = topology.get("per_env_collision_path_counts")
    serialized_unique_counts = topology.get("per_env_unique_collision_path_counts")
    serialized_proxy_counts = topology.get("per_env_instance_proxy_collision_path_counts")
    require(
        isinstance(root_states, Mapping)
        and isinstance(per_env_paths, Mapping)
        and isinstance(per_env_records, Mapping)
        and isinstance(serialized_counts, Mapping)
        and isinstance(serialized_unique_counts, Mapping)
        and isinstance(serialized_proxy_counts, Mapping)
        and set(root_states) == expected_env_keys
        and set(per_env_paths) == expected_env_keys
        and set(per_env_records) == expected_env_keys
        and set(serialized_counts) == expected_env_keys
        and set(serialized_unique_counts) == expected_env_keys
        and set(serialized_proxy_counts) == expected_env_keys,
        "collision topology per-env schema mismatch",
    )
    root_states = cast(Mapping[str, Any], root_states)
    per_env_paths = cast(Mapping[str, Any], per_env_paths)
    per_env_records = cast(Mapping[str, Any], per_env_records)
    serialized_counts = cast(Mapping[str, Any], serialized_counts)
    serialized_unique_counts = cast(Mapping[str, Any], serialized_unique_counts)
    serialized_proxy_counts = cast(Mapping[str, Any], serialized_proxy_counts)

    computed_template: list[str] | None = None
    computed_counts: dict[str, int] = {}
    computed_unique_counts: dict[str, int] = {}
    computed_proxy_counts: dict[str, int] = {}
    for env_index in range(NUM_ENVS):
        env_key = str(env_index)
        root_path = expected_root_paths[env_index]
        prefix = f"{root_path}/"
        root_state = root_states[env_key]
        require(
            isinstance(root_state, Mapping)
            and set(root_state) == {"path", "is_instance", "is_instanceable", "is_instance_proxy"}
            and root_state.get("path") == root_path
            and all(isinstance(root_state.get(name), bool) for name in ("is_instance", "is_instanceable", "is_instance_proxy")),
            f"collision topology robot root state mismatch for env {env_index}",
        )

        paths = per_env_paths[env_key]
        path_records = per_env_records[env_key]
        require(
            isinstance(paths, list)
            and all(isinstance(path, str) and path.startswith(prefix) for path in paths)
            and paths == sorted(set(paths)),
            f"collision topology canonical paths mismatch for env {env_index}",
        )
        require(
            isinstance(path_records, list)
            and all(
                isinstance(record, Mapping)
                and set(record) == {"path", "is_instance_proxy"}
                and isinstance(record.get("path"), str)
                and isinstance(record.get("is_instance_proxy"), bool)
                for record in path_records
            )
            and [record["path"] for record in path_records] == paths,
            f"collision topology path records mismatch for env {env_index}",
        )
        relative_paths = [path[len(prefix):] for path in paths]
        if computed_template is None:
            computed_template = relative_paths
        require(relative_paths == computed_template, f"collision topology relative template mismatch for env {env_index}")
        computed_counts[env_key] = len(paths)
        computed_unique_counts[env_key] = len(set(paths))
        computed_proxy_counts[env_key] = sum(1 for record in path_records if record["is_instance_proxy"])

    require(computed_template is not None, "collision topology serialized template missing")
    computed_template = cast(list[str], computed_template)
    require(
        len(computed_template) == COLLISION_SHAPES_PER_ARTICULATION
        and computed_template == sorted(set(computed_template))
        and topology.get("sorted_unique_template_paths") == computed_template,
        "collision topology serialized template mismatch",
    )
    require(
        computed_counts == {env_key: COLLISION_SHAPES_PER_ARTICULATION for env_key in expected_env_keys}
        and serialized_counts == computed_counts
        and serialized_unique_counts == computed_unique_counts
        and topology.get("collision_shape_paths_total") == sum(computed_counts.values()),
        "collision topology count relationship mismatch",
    )
    require(
        serialized_proxy_counts == computed_proxy_counts
        and topology.get("instance_proxy_collision_path_count") == sum(computed_proxy_counts.values()),
        "collision topology instance-proxy count relationship mismatch",
    )
    return dict(topology)


def validate_offset_integrity(value: Any, arm: str) -> dict[str, Any]:
    require(isinstance(value, dict), "offset integrity must be an object")
    value = cast(dict[str, Any], value)
    require(value.get("api") == "root_physx_view_get_set" and value.get("application_boundary") == "env_cfg_custom_startup_event_during_gym_make_before_first_reset_or_physics_step" and value.get("usd_schema_apply_called") is False, "offset API boundary mismatch")
    require(value.get("arm") == arm and value.get("contact_offset_scale") == ARM_SCALES[arm] and value.get("contact_offset_setter_called") is True and value.get("rest_offset_setter_called") is False, "offset intervention identity mismatch")
    require(value.get("shape_tensor_columns") == [f"shape_index_{index:02d}" for index in range(COLLISION_SHAPES_PER_ARTICULATION)] and value.get("tensor_column_path_mapping_authority") is False, "shape tensor column authority mismatch")
    before, after = value.get("before"), value.get("after")
    require(isinstance(before, Mapping) and isinstance(after, Mapping), "offset before/after snapshots missing")
    before_contact, before_contact_record = _validate_tensor_snapshot(before.get("contact_offset"), "before contact")
    before_rest, before_rest_record = _validate_tensor_snapshot(before.get("rest_offset"), "before rest")
    after_contact, after_contact_record = _validate_tensor_snapshot(after.get("contact_offset"), "after contact")
    after_rest, after_rest_record = _validate_tensor_snapshot(after.get("rest_offset"), "after rest")
    require(before_rest == after_rest and _snapshots_equal(before_rest_record, after_rest_record), "rest offset changed")
    require(all(item == 0.0 for row in before_rest for item in row), "rest offset baseline must remain zero")
    require(min(item for row in before_contact for item in row) < max(item for row in before_contact for item in row), "contact offset baseline must remain heterogeneous")
    require(
        all(
            math.isclose(after_contact[env][shape], before_contact[env][shape] * ARM_SCALES[arm], rel_tol=1.0e-6, abs_tol=1.0e-12)
            for env in range(NUM_ENVS)
            for shape in range(COLLISION_SHAPES_PER_ARTICULATION)
        ),
        "contact offset scale mismatch",
    )
    require(all(after_contact[env][shape] > after_rest[env][shape] for env in range(NUM_ENVS) for shape in range(COLLISION_SHAPES_PER_ARTICULATION)), "contact offset must remain above rest offset")
    require((arm != "A") or _snapshots_equal(before_contact_record, after_contact_record), "Arm A contact offset changed")
    checks = value.get("checks")
    require(isinstance(checks, Mapping) and set(checks) == {"all_envs_all_shapes_recorded", "heterogeneous_baseline_observed", "rest_baseline_zero", "contact_after_matches_scale", "contact_unchanged_for_control", "rest_bitwise_unchanged", "contact_strictly_above_rest", "symmetric_setter_policy"} and all(checks.get(name) is True for name in checks), "offset integrity check failed")
    validate_collision_topology(value.get("topology"))
    return value


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _validate_tensor_snapshot(snapshot: Any, label: str) -> tuple[list[list[float]], dict[str, Any]]:
    require(isinstance(snapshot, Mapping), f"{label} snapshot must be an object")
    snapshot = cast(Mapping[str, Any], snapshot)
    require(snapshot.get("shape") == [NUM_ENVS, COLLISION_SHAPES_PER_ARTICULATION], f"{label} shape mismatch")
    values = snapshot.get("values")
    require(isinstance(values, list) and len(values) == NUM_ENVS, f"{label} values env shape mismatch")
    values = cast(list[Any], values)
    matrix: list[list[float]] = []
    for row in values:
        require(isinstance(row, list) and len(row) == COLLISION_SHAPES_PER_ARTICULATION, f"{label} values collision shape mismatch")
        converted = [float(item) for item in row]
        require(all(math.isfinite(item) for item in converted), f"{label} values must be finite")
        matrix.append(converted)
    flat = [item for row in matrix for item in row]
    payload = json.dumps({"shape": [NUM_ENVS, COLLISION_SHAPES_PER_ARTICULATION], "values": values}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    require(_valid_hash(snapshot.get("sha256")) and snapshot.get("sha256") == sha256_bytes(payload), f"{label} SHA256 mismatch")
    require(snapshot.get("minimum") == min(flat) and snapshot.get("maximum") == max(flat), f"{label} min/max mismatch")
    require(isinstance(snapshot.get("dtype"), str) and isinstance(snapshot.get("device"), str), f"{label} dtype/device missing")
    return matrix, dict(snapshot)


def validate_manual_probe_safety(value: Any, device: str) -> dict[str, Any]:
    require(isinstance(value, dict), "manual probe safety must be an object")
    value = cast(dict[str, Any], value)
    require(value.get("label") == "manual_probe_observation_not_gate" and value.get("required_scopes") == ["all_envs", "source_env_7"], "manual probe safety identity mismatch")
    require(value.get("thresholds") == {"hard_joint_limit_margin_rad": HARD_JOINT_LIMIT_MARGIN_RAD, "non_foot_peak_force_body_weight_max": NON_FOOT_PEAK_FORCE_BODY_WEIGHT_MAX, "cpu_raw_minimum_separation_m": CPU_RAW_MINIMUM_SEPARATION_M}, "manual probe safety threshold mismatch")
    observations = value.get("observations")
    checks = value.get("checks")
    require(isinstance(observations, Mapping) and isinstance(checks, Mapping), "manual probe safety payload missing")
    observations = cast(Mapping[str, Any], observations)
    checks = cast(Mapping[str, Any], checks)
    expected_check_names = {"exact_150_manual_samples", "finite_joint_position_and_contact_force", "hard_joint_limit_with_margin", "all_env_non_foot_peak_force_within_15_bw", "source_env_non_foot_peak_force_within_15_bw", "cpu_raw_minimum_separation_observed", "cpu_raw_minimum_separation_within_limit", "default_mass_8x19_finite_positive_unchanged", "collection_error_absent"}
    require(set(checks) == expected_check_names, "manual probe safety check set mismatch")
    all_force = observations.get("all_env_non_foot_peak_force_body_weight")
    source_force = observations.get("source_env_non_foot_peak_force_body_weight")
    require(type(all_force) in (int, float) and type(source_force) in (int, float), "manual probe force summary invalid")
    all_force = cast(float | int, all_force)
    source_force = cast(float | int, source_force)
    require(math.isfinite(float(all_force)) and math.isfinite(float(source_force)), "manual probe force summary invalid")
    finite_steps = observations.get("finite_violation_steps")
    hard_steps = observations.get("hard_joint_limit_violation_steps")
    per_env_force = observations.get("non_foot_peak_force_body_weight_per_env")
    per_env_force_n = observations.get("non_foot_peak_force_n_per_env")
    require(isinstance(finite_steps, Mapping) and set(finite_steps) == {"all_envs", "source_env_7"} and all(isinstance(item, list) and all(type(step) is int and 1 <= step <= PHYSICS_SUBSTEPS for step in item) for item in finite_steps.values()), "finite violation trace invalid")
    require(isinstance(hard_steps, Mapping) and set(hard_steps) == {"all_envs", "source_env_7"} and all(isinstance(item, list) and all(type(step) is int and 1 <= step <= PHYSICS_SUBSTEPS for step in item) for item in hard_steps.values()), "hard-limit violation trace invalid")
    require(isinstance(per_env_force, Mapping) and set(per_env_force) == {str(index) for index in range(NUM_ENVS)} and all(type(item) in (int, float) and math.isfinite(float(item)) for item in per_env_force.values()), "per-env force summary invalid")
    require(isinstance(per_env_force_n, Mapping) and set(per_env_force_n) == {str(index) for index in range(NUM_ENVS)} and all(type(item) in (int, float) and math.isfinite(float(item)) and float(item) >= 0.0 for item in per_env_force_n.values()), "per-env force numerator invalid")
    finite_steps = cast(Mapping[str, list[int]], finite_steps)
    hard_steps = cast(Mapping[str, list[int]], hard_steps)
    per_env_force = cast(Mapping[str, float | int], per_env_force)
    per_env_force_n = cast(Mapping[str, float | int], per_env_force_n)
    require(float(all_force) == max(float(item) for item in per_env_force.values()) and float(source_force) == float(per_env_force[str(SOURCE_ENV_INDEX)]), "force scope summary mismatch")
    mass = value.get("mass_evidence")
    require(isinstance(mass, Mapping) and mass.get("source") == "robot.data.default_mass" and mass.get("shape") == [NUM_ENVS, BODY_COUNT], "mass evidence identity mismatch")
    body_names = mass.get("body_names")
    require(isinstance(body_names, list) and len(body_names) == BODY_COUNT and len(set(body_names)) == BODY_COUNT and all(isinstance(name, str) and name for name in body_names), "mass body ordering invalid")
    require(mass.get("body_names_sha256") == sha256_bytes(json.dumps(body_names, separators=(",", ":")).encode("utf-8")), "mass body ordering hash mismatch")
    tensor = mass.get("tensor")
    require(isinstance(tensor, Mapping) and tensor.get("shape") == [NUM_ENVS, BODY_COUNT], "mass tensor shape mismatch")
    tensor = cast(Mapping[str, Any], tensor)
    mass_values = tensor.get("values")
    require(isinstance(mass_values, list) and len(mass_values) == NUM_ENVS and all(isinstance(row, list) and len(row) == BODY_COUNT for row in mass_values), "mass tensor values shape mismatch")
    mass_values = cast(list[list[Any]], mass_values)
    numeric_mass = [[float(item) for item in row] for row in mass_values]
    require(all(math.isfinite(item) and item > 0.0 for row in numeric_mass for item in row), "mass tensor must be finite and strictly positive")
    mass_payload = json.dumps({"shape": [NUM_ENVS, BODY_COUNT], "values": mass_values}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    flat_mass = [item for row in numeric_mass for item in row]
    require(tensor.get("sha256") == sha256_bytes(mass_payload) and tensor.get("minimum") == min(flat_mass) and tensor.get("maximum") == max(flat_mass), "mass tensor hash/min/max mismatch")
    totals = [math.fsum(row) for row in numeric_mass]
    weights = [total * 9.81 for total in totals]
    require(mass.get("per_env_total_mass_kg") == totals and mass.get("per_env_body_weight_n") == weights and mass.get("unchanged_for_150_steps") is True and mass.get("changed_steps") == [], "mass totals or stability mismatch")
    require(all(math.isclose(float(per_env_force[str(index)]), float(per_env_force_n[str(index)]) / weights[index], rel_tol=1.0e-6, abs_tol=1.0e-12) for index in range(NUM_ENVS)), "body-weight normalization does not match mass snapshot")
    separation = observations.get("cpu_raw_minimum_separation_m")
    require(isinstance(separation, Mapping), "manual probe separation summary missing")
    separation = cast(Mapping[str, Any], separation)
    per_env = separation.get("per_env")
    require(isinstance(per_env, Mapping) and set(per_env) == {str(index) for index in range(NUM_ENVS)}, "manual probe per-env separation scope mismatch")
    per_env = cast(Mapping[str, Any], per_env)
    cpu_values: list[float] = []
    if device == "cpu":
        require(all(type(item) in (int, float) and math.isfinite(float(item)) for item in per_env.values()), "CPU per-env separation observation incomplete")
        cpu_values = [float(cast(float | int, item)) for item in per_env.values()]
        recomputed_minimum = min(cpu_values)
        require(separation.get("all_env_minimum") == recomputed_minimum and separation.get("source_env_7_minimum") == per_env[str(SOURCE_ENV_INDEX)], "CPU separation minima mismatch")
    else:
        require(all(item is None or (type(item) in (int, float) and math.isfinite(float(item))) for item in per_env.values()), "GPU separation summary invalid")
    recomputed = {
        "exact_150_manual_samples": observations.get("sample_count") == PHYSICS_SUBSTEPS,
        "finite_joint_position_and_contact_force": finite_steps["all_envs"] == [] and finite_steps["source_env_7"] == [],
        "hard_joint_limit_with_margin": hard_steps["all_envs"] == [] and hard_steps["source_env_7"] == [],
        "all_env_non_foot_peak_force_within_15_bw": float(all_force) <= NON_FOOT_PEAK_FORCE_BODY_WEIGHT_MAX,
        "source_env_non_foot_peak_force_within_15_bw": float(source_force) <= NON_FOOT_PEAK_FORCE_BODY_WEIGHT_MAX,
        "cpu_raw_minimum_separation_observed": device != "cpu" or all(item is not None for item in per_env.values()),
        "cpu_raw_minimum_separation_within_limit": device != "cpu" or min(cpu_values) >= CPU_RAW_MINIMUM_SEPARATION_M,
        "default_mass_8x19_finite_positive_unchanged": mass.get("unchanged_for_150_steps") is True and mass.get("changed_steps") == [],
        "collection_error_absent": observations.get("error") is None,
    }
    require(dict(checks) == recomputed, "manual probe safety checks differ from recomputation")
    available = recomputed["exact_150_manual_samples"] and recomputed["cpu_raw_minimum_separation_observed"] and recomputed["collection_error_absent"]
    require(value.get("available") is available and value.get("passed") is (all(recomputed.values()) if available else None), "manual probe safety outcome mismatch")
    return value


def validate_report(report: Mapping[str, Any], *, validate_gpu_preflight: bool = True) -> dict[str, Any]:
    expected_keys = {"schema_version", "goal_id", "stage_id", "experiment_id", "revision", "status", "headless", "arm", "device", "seed", "replicate_index", "num_envs", "source_env_index", "physics_substeps", "physics_dt_s", "manual_inner_loop", "finished_at_utc", "execution", "contract", "contract_sha256", "preregistration", "predecessor", "source_bundle", "cpu_preflight_binding", "governance", "pose_action_assignment", "live_physics_readback", "device_readback", "residual_capability", "offset_integrity", "physics_step_clock", "raw_contact_observation", "supporting_telemetry", "manual_probe_safety", "feasibility"}
    require(set(report) == expected_keys, "report top-level field set mismatch")
    arm = str(report.get("arm", "")).upper()
    device = str(report.get("device", "")).lower()
    replicate = report.get("replicate_index")
    require(report.get("schema_version") == SCHEMA_VERSION and report.get("goal_id") == "g009" and report.get("stage_id") == "R0" and report.get("experiment_id") == "G009-5-E012" and report.get("revision") == "rev19" and report.get("status") == "complete", "rev19 report identity mismatch")
    require(arm in ARM_SCALES and device in {"cpu", "cuda:0"} and replicate in {1, 2}, "rev19 slot identity mismatch")
    require(report.get("headless") is True and report.get("seed") == 42 and report.get("num_envs") == NUM_ENVS and report.get("source_env_index") == SOURCE_ENV_INDEX and report.get("physics_substeps") == PHYSICS_SUBSTEPS and report.get("physics_dt_s") == PHYSICS_DT_S, "rev19 runtime contract mismatch")
    contract = probe_contract(arm, device, cast(int, replicate))
    require(report.get("contract") == contract and report.get("contract_sha256") == canonical_sha256(contract), "probe contract mismatch")
    require(report.get("preregistration") == contract["preregistration"] and report.get("predecessor") == contract["predecessor"], "predecessor/preregistration binding mismatch")
    require(report.get("governance") == governance(), "governance must remain closed")
    require(report.get("pose_action_assignment") == {"class_ids": [0, 1, 2, 3, 0, 1, 2, 3]}, "pose assignment mismatch")
    finished = report.get("finished_at_utc")
    require(isinstance(finished, str) and finished.endswith("Z"), "finished timestamp must be UTC")
    try:
        finished_time = datetime.fromisoformat(cast(str, finished)[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("finished timestamp must be UTC") from error
    require(finished_time.utcoffset() == timezone.utc.utcoffset(finished_time), "finished timestamp must be UTC")
    validate_execution_metadata(report.get("execution"), arm, device, cast(int, replicate))
    source_bundle = validate_source_bundle(report.get("source_bundle"))
    preflight_binding = report.get("cpu_preflight_binding")
    if device == "cpu":
        require(preflight_binding == cpu_preflight_not_required_binding(), "CPU report preflight binding mismatch")
    else:
        require(isinstance(preflight_binding, Mapping) and preflight_binding.get("status") == "validated_for_gpu", "GPU report requires validated CPU preflight binding")
        if validate_gpu_preflight:
            require(preflight_binding == validate_cpu_preflight_artifact(CPU_PREFLIGHT_PATH, source_bundle), "GPU CPU-preflight binding drift")
    validate_offset_integrity(report.get("offset_integrity"), arm)
    safety = validate_manual_probe_safety(report.get("manual_probe_safety"), device)
    manual = report.get("manual_inner_loop")
    require(
        isinstance(manual, Mapping)
        and manual.get("control_decimation") == 4
        and manual.get("action_process_steps") == list(range(1, PHYSICS_SUBSTEPS + 1, 4))
        and manual.get("action_process_count") == 38
        and manual.get("manager_post_step_executed") is False
        and manual.get("reward_computed") is False
        and manual.get("termination_computed") is False
        and manual.get("trajectory_equivalence_claimed") is False
        and manual.get("scope") == "capability_only",
        "manual inner-loop contract mismatch",
    )
    device_readback = report.get("device_readback")
    require(
        isinstance(device_readback, Mapping)
        and device_readback.get("requested_device") == device
        and device_readback.get("runtime_device") == device
        and device_readback.get("gpu_dynamics_enabled") is (device == "cuda:0")
        and device_readback.get("gpu_dynamics_matches_device") is True
        and device_readback.get("error") is None,
        "device readback mismatch",
    )
    mass_names = cast(Mapping[str, Any], safety["mass_evidence"])["body_names"]
    live = report.get("live_physics_readback")
    require(isinstance(live, Mapping), "live physics readback missing")
    live = cast(Mapping[str, Any], live)
    max_dep = live.get("max_depenetration_velocity")
    require(isinstance(max_dep, dict), "max-depenetration readback missing")
    max_dep = cast(dict[str, Any], max_dep)
    max_dep_checks = runtime_probe.rigid_body_max_depenetration_velocity_checks(
        max_dep,
        expected_velocity_m_s=1.0,
        expected_articulation_count=NUM_ENVS,
        expected_body_names=cast(list[str], mass_names),
    )
    require(max_dep_checks.get("rigid_body_max_depenetration_velocity_matches_contract") is True, "max-depenetration readback mismatch")
    derived = derive_feasibility(report)
    require(report.get("feasibility") == derived, "serialized feasibility differs from recomputation")
    require(derived["offset_integrity_passed"] is True and derived["solver_live_readback_8_0"] is True, "single-variable intervention integrity failed")
    return derived


def diagnose(args: argparse.Namespace, execution: dict[str, Any]) -> dict[str, Any]:
    global _STARTUP_EVIDENCE_SINK
    arm = args.arm.upper()
    device = args.device.lower()
    require(args.seed == 42 and args.task == DEFAULT_TASK and bool(args.headless), "rev19 launch contract mismatch")
    validate_predecessor()
    load_preregistration()
    source_bundle = source_bundle_provenance()
    require(source_bundle["all_files_present"] and source_bundle["clean"], "source bundle must be committed and clean")
    validate_source_bundle(source_bundle)
    import isaaclab_tasks.utils as task_utils  # pyright: ignore[reportMissingImports]
    import torch
    from isaaclab.managers import EventTermCfg  # pyright: ignore[reportMissingImports]

    original_parse_env_cfg = task_utils.parse_env_cfg
    original_proxy = base_probe._proxy_row
    offset_record: dict[str, Any] = {}
    safety = SafetyAccumulator()
    saved = {
        "POSITION_SOLVER_ITERATIONS": base_probe.POSITION_SOLVER_ITERATIONS,
        "VELOCITY_SOLVER_ITERATIONS": base_probe.VELOCITY_SOLVER_ITERATIONS,
        "PREDECESSOR_PATH": base_probe.PREDECESSOR_PATH,
        "PREDECESSOR_SHA256": base_probe.PREDECESSOR_SHA256,
        "SOURCE_BINDING_PATHS": base_probe.SOURCE_BINDING_PATHS,
        "expected_output_relative": base_probe.expected_output_relative,
    }

    def wrapped_parse_env_cfg(*parse_args: Any, **parse_kwargs: Any) -> Any:
        env_cfg = original_parse_env_cfg(*parse_args, **parse_kwargs)
        env_cfg.events.rev19_contact_offset_intervention = EventTermCfg(
            func=startup_scale_contact_offsets,
            mode="startup",
            params={"arm": arm},
        )
        return env_cfg

    def wrapped_proxy(*proxy_args: Any, **proxy_kwargs: Any) -> dict[str, Any]:
        row = original_proxy(*proxy_args, **proxy_kwargs)
        safety.observe(proxy_kwargs["sensor"], proxy_kwargs["robot"], torch)
        return row

    try:
        _STARTUP_EVIDENCE_SINK = offset_record
        task_utils.parse_env_cfg = wrapped_parse_env_cfg
        base_probe._proxy_row = wrapped_proxy
        base_probe.expected_output_relative = lambda requested_device, requested_replicate: expected_output_relative(arm, requested_device, requested_replicate)
        base_probe.POSITION_SOLVER_ITERATIONS = POSITION_SOLVER_ITERATIONS
        base_probe.VELOCITY_SOLVER_ITERATIONS = VELOCITY_SOLVER_ITERATIONS
        base_probe.PREDECESSOR_PATH = PREDECESSOR_PATH
        base_probe.PREDECESSOR_SHA256 = PREDECESSOR_SHA256
        base_probe.SOURCE_BINDING_PATHS = SOURCE_BINDING_PATHS
        base_args = argparse.Namespace(**vars(args))
        base_report = base_probe.diagnose(base_args, execution)
    finally:
        task_utils.parse_env_cfg = original_parse_env_cfg
        _STARTUP_EVIDENCE_SINK = None
        base_probe._proxy_row = original_proxy
        for name, value in saved.items():
            setattr(base_probe, name, value)
    require(offset_record, "offset intervention did not run before reset")
    report = dict(base_report)
    report.update({"schema_version": SCHEMA_VERSION, "experiment_id": "G009-5-E012", "revision": "rev19", "arm": arm})
    contract = probe_contract(arm, device, args.replicate_index)
    report["contract"] = contract
    report["contract_sha256"] = canonical_sha256(contract)
    report["preregistration"] = contract["preregistration"]
    report["predecessor"] = validate_predecessor()
    report["source_bundle"] = source_bundle
    report["cpu_preflight_binding"] = getattr(args, "_cpu_preflight_binding", cpu_preflight_not_required_binding())
    report["governance"] = governance()
    report["offset_integrity"] = offset_record
    report["manual_probe_safety"] = safety.snapshot(cast(Mapping[str, Any], report["raw_contact_observation"]), device)
    report["feasibility"] = derive_feasibility(report)
    validate_report(report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, choices=(DEFAULT_TASK,))
    parser.add_argument("--seed", type=int, default=42, choices=(42,))
    parser.add_argument("--arm", required=True, choices=("A", "B", "a", "b"))
    parser.add_argument("--replicate-index", required=True, type=int, choices=(1, 2))
    parser.add_argument("--cpu-preflight", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if not getattr(args, "device_explicit", False):
        parser.error("--device must be supplied explicitly as cpu or cuda:0")
    if args.device not in {"cpu", "cuda:0"}:
        parser.error("--device must be cpu or cuda:0")
    if args.device == "cuda:0" and args.cpu_preflight is None:
        parser.error("GPU runs require --cpu-preflight")
    if args.device == "cpu" and args.cpu_preflight is not None:
        parser.error("CPU runs must not supply --cpu-preflight")
    return args


def failure_envelope(args: argparse.Namespace, execution: dict[str, Any], error: BaseException) -> dict[str, Any]:
    arm = str(getattr(args, "arm", "")).upper()
    device = str(getattr(args, "device", "")).lower()
    replicate = getattr(args, "replicate_index", None)
    valid_slot = arm in ARM_SCALES and device in {"cpu", "cuda:0"} and replicate in {1, 2}
    try:
        source_bundle = source_bundle_provenance()
    except Exception as bundle_error:
        source_bundle = {
            "all_files_present": False,
            "clean": False,
            "error": f"{type(bundle_error).__name__}: {bundle_error}",
        }
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "goal_id": "g009",
        "stage_id": "R0",
        "experiment_id": "G009-5-E012",
        "revision": "rev19",
        "status": "failed_closed",
        "arm": arm or None,
        "device": device or None,
        "seed": getattr(args, "seed", None),
        "replicate_index": replicate,
        "execution": execution,
        "contract": probe_contract(arm, device, cast(int, replicate)) if valid_slot else None,
        "predecessor": {"path": PREDECESSOR_PATH.relative_to(REPO_ROOT).as_posix(), "expected_sha256": PREDECESSOR_SHA256, "observed_sha256": sha256_bytes(PREDECESSOR_PATH.read_bytes()) if PREDECESSOR_PATH.is_file() else None},
        "source_bundle": source_bundle,
        "governance": governance(),
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def main(argv: list[str] | None = None) -> int:
    output, execution = runtime_probe.prepare_execution(runtime_probe.parse_prelaunch_output(argv))
    args = parse_args(argv)
    validate_execution_metadata(execution, args.arm, args.device, args.replicate_index)
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    app = None
    try:
        try:
            validate_predecessor()
            load_preregistration()
            preflight = source_bundle_provenance()
            require(preflight["all_files_present"] and preflight["clean"], "source bundle must be committed and clean")
            validate_source_bundle(preflight)
            if args.device == "cuda:0":
                args._cpu_preflight_binding = validate_cpu_preflight_artifact(args.cpu_preflight, preflight)
            else:
                args._cpu_preflight_binding = cpu_preflight_not_required_binding()
            app = AppLauncher(args).app
            report = diagnose(args, execution)
        except Exception as error:
            report = failure_envelope(args, execution, error)
        runtime_probe._write_json_atomic(output, report)
        interpretable = bool(report.get("feasibility", {}).get("run_interpretable", False))
        print(json.dumps({"output": str(output), "run_interpretable": interpretable, "raw_observation_passed": bool(report.get("feasibility", {}).get("raw_observation_passed", False))}, ensure_ascii=False), flush=True)
        return 0 if interpretable else 2
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
