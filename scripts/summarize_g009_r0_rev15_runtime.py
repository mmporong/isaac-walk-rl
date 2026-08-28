#!/usr/bin/env python3
"""Create a fail-closed rev15 CPU/GPU runtime rejection synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "reports/runs"
DEFAULT_CPU = tuple(
    RUNS_DIR / f"g009_r0_runtime_probe_rev15_cpu_rep{i:02d}_s42.json"
    for i in range(1, 4)
)
DEFAULT_GPU = tuple(
    RUNS_DIR / f"g009_r0_runtime_probe_rev15_gpu_rep{i:02d}_s42.json"
    for i in range(1, 4)
)
DEFAULT_OUTPUT = (
    RUNS_DIR / "g009_r0_runtime_probe_rev15_rejection_synthesis_3x3_s42.json"
)

REV15_CONTRACT = "5f29ba19458404b5009d3734294c57e79294efecc7fe03bf8c71c71656129832"
REV15_SOURCE_COMMIT = "bc999d504e226011ff3d83e68a416b9049b406cb"
REV15_SOURCE_BUNDLE = "218671a84f2748f7b94a426490057318b0896e2160454f6928c4277dee7435df"
EXPECTED_SOLVER_POSITION_ITERATIONS = 16
EXPECTED_SOLVER_VELOCITY_ITERATIONS = 0
EXPECTED_MAX_DEPENETRATION_VELOCITY_M_S = 1.0
EXPECTED_CPU_FORCE_BODYWEIGHTS = 13.248281478881836
EXPECTED_GPU_FORCE_BODYWEIGHTS = 16.78827476501465
EXPECTED_CPU_SEPARATION_M = -0.009353086352348328
FORCE_THRESHOLD_BODYWEIGHTS = 15.0
SEPARATION_THRESHOLD_M = -0.01
EXPECTED_BODY_NAMES = (
    "base",
    "FL_hip",
    "FR_hip",
    "Head_upper",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "Head_lower",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
)
EXPECTED_RUNTIME_CHECKS = (
    "articulation_solver_iteration_counts_match_contract",
    "rigid_body_max_depenetration_velocity_matches_contract",
    "joint_action_type_matches_contract",
    "joint_action_scale_matches_contract",
    "joint_action_ema_alpha_matches_contract",
    "joint_action_ema_history_matches_reset_joint_positions",
    "soft_joint_limit_factor_matches_contract",
    "policy_observation_dim_83",
    "critic_observation_dim_107",
    "action_dim_12",
    "eight_env_pose_mode_stratification",
    "eight_env_pose_one_hot",
    "pose_orientation_matches_contract",
    "observations_finite_at_reset",
    "critic_base_height_matches_root_height",
    "critic_terrain_normal_finite_unit_length",
    "body_range_camera_world_pose_matches_base_offset",
    "body_range_camera_attached_to_base",
    "body_range_camera_targets_ground",
    "body_range_camera_distance_output",
    "body_range_camera_pattern_5x3",
    "body_range_camera_max_distance_1m",
    "body_range_camera_offset_matches_contract",
    "body_range_camera_rotation_matches_contract",
    "actor_foot_load_finite_nonnegative",
    "body_range_finite_unit_interval",
    "body_range_mask_binary",
    "body_range_no_hit_is_one",
    "prone_has_at_least_one_camera_hit_both_modes",
    "supine_has_zero_camera_hits_both_modes",
    "actor_foot_load_valid_through_rollout",
    "body_range_valid_through_rollout",
    "body_range_mask_valid_through_rollout",
    "body_range_no_hit_is_one_through_rollout",
    "actor_foot_load_independent_of_ray_no_hit",
    "forced_ray_no_hit_maps_to_valid_actor_observation",
    "foot_material_readback_matches_startup",
    "effective_friction_matches_multiply_combine",
    "nominal_mass_readback_matches_contract",
    "actor_privileged_perturbation_invariant",
    "actor_call_path_avoids_live_mass_read",
    "reward_terminal_contribution_exact_10",
    "reward_potential_contribution_exact_0_495",
    "stable_success_25_step_one_shot",
    "stable_success_partial_reset_isolated",
    "stable_success_uses_60pct_foot_load_gate",
    "body_supported_state_does_not_latch_success",
    "low_foot_load_state_does_not_latch_success",
    "rollout_state_finite_all_pose_modes",
    "no_numeric_invalid_termination",
    "no_hard_joint_limit_termination",
    "joint_positions_within_hard_limit_margin",
    "torque_within_runtime_limit",
    "joint_speed_within_runtime_limit",
    "root_height_above_2cm",
    "at_least_one_contact_type_exercised_per_pose_mode",
    "foot_contact_exercised_globally",
    "nonfoot_contact_exercised_globally",
    "nonfoot_peak_force_bounded",
    "nonfoot_peak_force_body_attribution_complete",
    "nonfoot_excess_impulse_bounded",
    "tail_horizontal_speed_settled",
    "tail_vertical_speed_settled",
    "tail_angular_speed_settled",
    "reset_pose_hold_actions_bounded",
    "reset_pose_hold_action_diagnostics_finite",
    "reset_pose_hold_actions_unsaturated",
    "reset_pose_hold_reachable_targets_match_reset_positions",
    "zero_normalized_actions_are_zero",
    "base_contact_not_a_termination",
    "source_binding_files_present",
    "source_binding_git_commit_valid",
    "source_binding_clean",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: Any, label: str) -> float:
    require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{label} must be a finite JSON number",
    )
    return float(value)


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    resolved = path.resolve(strict=True)
    require(
        resolved.parent == RUNS_DIR.resolve(),
        f"input must be a direct child of reports/runs: {path}",
    )
    raw = resolved.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value, {
        "path": f"reports/runs/{resolved.name}",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def validate_execution(
    report: dict[str, Any], evidence: dict[str, str]
) -> tuple[str, str]:
    execution = report.get("execution")
    require(isinstance(execution, dict), "execution is required")
    assert isinstance(execution, dict)
    execution_id = execution.get("execution_id")
    require(isinstance(execution_id, str), "execution_id must be UUID4 hex")
    assert isinstance(execution_id, str)
    try:
        parsed = uuid.UUID(hex=execution_id)
    except ValueError as exc:
        raise ValueError("execution_id must be UUID4 hex") from exc
    require(
        parsed.version == 4 and parsed.hex == execution_id,
        "execution_id must be lowercase UUID4 hex",
    )
    require(execution.get("no_overwrite") is True, "input must be no-overwrite")
    started_at_utc = execution.get("started_at_utc")
    require(
        isinstance(started_at_utc, str) and started_at_utc.endswith("Z"),
        "execution timestamp must be RFC3339 UTC with a Z suffix",
    )
    assert isinstance(started_at_utc, str)
    try:
        parsed_time = datetime.fromisoformat(
            started_at_utc.removesuffix("Z") + "+00:00"
        )
    except ValueError as exc:
        raise ValueError("execution timestamp must be valid RFC3339 UTC") from exc
    require(parsed_time.tzinfo == timezone.utc, "execution timestamp must use UTC")
    require(
        execution.get("output_path_repo_relative") == evidence["path"],
        "execution path binding mismatch",
    )
    return execution_id, started_at_utc


def validate_lineage(report: dict[str, Any]) -> tuple[str, str, str]:
    require(report.get("contract_sha256") == REV15_CONTRACT, "contract hash mismatch")
    bundle = report.get("source_bundle")
    require(isinstance(bundle, dict), "source_bundle is required")
    assert isinstance(bundle, dict)
    require(bundle.get("git_commit_valid") is True, "source commit must be validated")
    require(
        bundle.get("clean") is True and bundle.get("all_files_present") is True,
        "source bundle must be clean and complete",
    )
    require(
        bundle.get("missing_files") == [] and bundle.get("dirty_source_paths") == [],
        "source bundle lists unresolved files",
    )
    commit = bundle.get("git_commit")
    digest = bundle.get("source_bundle_sha256")
    require(
        isinstance(commit, str)
        and len(commit) == 40
        and all(c in "0123456789abcdef" for c in commit),
        "invalid source commit",
    )
    assert isinstance(commit, str)
    require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest),
        "invalid source bundle hash",
    )
    assert isinstance(digest, str)
    paths = bundle.get("source_binding_paths")
    files = bundle.get("source_binding_files")
    require(
        isinstance(paths, list) and len(paths) > 0 and len(paths) == len(set(paths)),
        "source binding paths must be non-empty and unique",
    )
    assert isinstance(paths, list)
    require(
        isinstance(files, dict) and set(files) == set(paths),
        "source binding files must match paths",
    )
    assert isinstance(files, dict)
    require(
        all(
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)
            for value in files.values()
        ),
        "invalid source binding file hash",
    )
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    require(
        hashlib.sha256(payload.encode("utf-8")).hexdigest() == digest,
        "source bundle digest mismatch",
    )
    return commit, digest, REV15_CONTRACT


def termination_totals(report: dict[str, Any]) -> dict[str, int]:
    metrics = report.get("pose_mode_metrics")
    require(isinstance(metrics, list) and len(metrics) == 8, "eight pose metrics required")
    assert isinstance(metrics, list)
    totals = {"numeric_invalid": 0, "hard_joint_limit": 0}
    for item in metrics:
        require(isinstance(item, dict), "pose metric must be an object")
        counts = item.get("termination_counts")
        require(isinstance(counts, dict), "termination_counts must be an object")
        assert isinstance(counts, dict)
        for name in totals:
            value = counts.get(name)
            require(type(value) is int and value >= 0, f"invalid {name} count")
            assert isinstance(value, int)
            totals[name] += value
    return totals


def validate_pose_mapping(report: dict[str, Any]) -> None:
    pose_order = ["prone", "supine", "left_side", "right_side"]
    action_modes = ["zero_normalized", "reset_pose_hold"]
    require(report.get("pose_name_order") == pose_order, "pose name order mismatch")
    require(report.get("action_modes") == action_modes, "action mode order mismatch")
    metrics = report.get("pose_mode_metrics")
    require(isinstance(metrics, list) and len(metrics) == 8, "eight pose metrics required")
    assert isinstance(metrics, list)
    expected = [
        (env_index, pose, action_mode)
        for env_index, (action_mode, pose) in enumerate(
            (action_mode, pose)
            for action_mode in action_modes
            for pose in pose_order
        )
    ]
    observed = [
        (item.get("env_index"), item.get("pose_id"), item.get("action_mode"))
        for item in metrics
        if isinstance(item, dict)
    ]
    require(observed == expected, "pose/action environment mapping mismatch")


def validate_common(report: dict[str, Any], expected_device: str) -> None:
    require(report.get("schema_version") == 3, "probe schema must be 3")
    require(
        report.get("goal_id") == "g009" and report.get("stage_id") == "R0",
        "goal/stage mismatch",
    )
    require(report.get("probe") == "flat_recover_runtime_calibration", "probe mismatch")
    require(report.get("task") == "Isaac-G009-Recover-Flat-Go2-R0-v0", "task mismatch")
    require(
        report.get("seed") == 42
        and report.get("num_envs") == 8
        and report.get("rollout_steps") == 150
        and report.get("headless") is True,
        "seed/environment/rollout/headless mismatch",
    )
    device = report.get("device")
    require(
        device == "cpu"
        if expected_device == "cpu"
        else device == "cuda:0",
        "device mismatch",
    )
    require(report.get("run_health", {}).get("passed") is True, "run_health must pass")
    require(
        report.get("passed_semantics") == "progression_gate_not_policy_qualification",
        "passed semantics mismatch",
    )
    qualification = report.get("qualification")
    require(
        isinstance(qualification, dict)
        and qualification.get("status") == "not_run"
        and qualification.get("passed") is None,
        "qualification must remain not_run/null",
    )
    require(
        termination_totals(report) == {"numeric_invalid": 0, "hard_joint_limit": 0},
        "safety termination count must be zero",
    )
    validate_pose_mapping(report)
    thresholds = report.get("calibration_thresholds")
    require(isinstance(thresholds, dict), "calibration thresholds are required")
    assert isinstance(thresholds, dict)
    require(
        math.isclose(
            finite_number(
                thresholds.get("max_nonfoot_force_bodyweights"), "force threshold"
            ),
            FORCE_THRESHOLD_BODYWEIGHTS,
            abs_tol=1e-12,
        ),
        "force threshold changed",
    )


def validate_readback(report: dict[str, Any]) -> None:
    physics = report.get("physics_readback")
    require(isinstance(physics, dict), "physics readback is required")
    assert isinstance(physics, dict)
    solver = physics.get("articulation_solver_iterations")
    require(isinstance(solver, dict), "solver readback is required")
    assert isinstance(solver, dict)
    solver_articulations = solver.get("articulations")
    require(
        isinstance(solver_articulations, list) and len(solver_articulations) == 8,
        "eight solver articulation readbacks are required",
    )
    assert isinstance(solver_articulations, list)
    for index, item in enumerate(solver_articulations):
        require(isinstance(item, dict), "solver articulation must be an object")
        require(
            item.get("prim_path") == f"/World/envs/env_{index}/Robot/base",
            "solver articulation path mismatch",
        )
        require(
            item.get("solver_position_iteration_count")
            == EXPECTED_SOLVER_POSITION_ITERATIONS,
            "position solver iteration count mismatch",
        )
        require(
            item.get("solver_velocity_iteration_count")
            == EXPECTED_SOLVER_VELOCITY_ITERATIONS,
            "velocity solver iteration count mismatch",
        )

    readback = physics.get("rigid_body_max_depenetration_velocity")
    require(isinstance(readback, dict), "rigid-body readback is required")
    assert isinstance(readback, dict)
    require(
        readback.get("articulation_group_count") == 8
        and readback.get("rigid_body_count") == 152,
        "readback must cover 8x19 rigid bodies",
    )
    require(readback.get("duplicate_link_prim_paths") == [], "readback paths duplicated")
    require(
        readback.get("authoritative_body_names") == list(EXPECTED_BODY_NAMES),
        "authoritative body order mismatch",
    )
    articulations = readback.get("articulations")
    require(
        isinstance(articulations, list) and len(articulations) == 8,
        "eight rigid-body articulation readbacks are required",
    )
    assert isinstance(articulations, list)
    observed_paths: set[str] = set()
    for env_index, articulation in enumerate(articulations):
        require(isinstance(articulation, dict), "articulation must be an object")
        container = f"/World/envs/env_{env_index}/Robot"
        require(
            articulation.get("articulation_index") == env_index
            and articulation.get("robot_container_prim_path") == container
            and articulation.get("articulation_prim_path") == f"{container}/base",
            "articulation identity mismatch",
        )
        require(
            articulation.get("root_link_prim_path") == f"{container}/base"
            and articulation.get("authoritative_body_names")
            == list(EXPECTED_BODY_NAMES),
            "articulation root/body order mismatch",
        )
        links = articulation.get("links")
        require(
            isinstance(links, list) and len(links) == len(EXPECTED_BODY_NAMES),
            "each articulation must expose 19 links",
        )
        assert isinstance(links, list)
        for body_index, (body_name, link) in enumerate(
            zip(EXPECTED_BODY_NAMES, links, strict=True)
        ):
            require(isinstance(link, dict), "link readback must be an object")
            expected_path = f"{container}/{body_name}"
            require(
                link.get("body_index") == body_index
                and link.get("body_name") == body_name
                and link.get("prim_path") == expected_path,
                "link identity/path mismatch",
            )
            require(expected_path not in observed_paths, "duplicate link path")
            observed_paths.add(expected_path)
            require(
                link.get("prim_valid") is True
                and link.get("usd_rigid_body_api") is True
                and link.get("physx_rigid_body_api") is True
                and link.get("error") is None,
                "link API/path readback invalid",
            )
            require(
                math.isclose(
                    finite_number(
                        link.get("max_depenetration_velocity_m_s"),
                        "max depenetration velocity",
                    ),
                    EXPECTED_MAX_DEPENETRATION_VELOCITY_M_S,
                    abs_tol=1e-12,
                ),
                "max depenetration velocity mismatch",
            )
    require(len(observed_paths) == 152, "readback must contain 152 unique paths")


def extrema(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["pose_mode_metrics"]
    assert isinstance(metrics, list)
    peak_metric = max(
        metrics,
        key=lambda item: finite_number(
            item.get("max_nonfoot_force_bodyweights"), "non-foot force"
        ),
    )
    peak = finite_number(
        peak_metric.get("max_nonfoot_force_bodyweights"), "global non-foot force"
    )
    available_separations = [
        finite_number(item.get("min_contact_separation_m"), "contact separation")
        for item in metrics
        if item.get("min_contact_separation_m") is not None
    ]
    return {
        "peak_force_bodyweights": peak,
        "peak_force_env_index": peak_metric.get("env_index"),
        "peak_force_pose": peak_metric.get("pose_id"),
        "peak_force_action_mode": peak_metric.get("action_mode"),
        "peak_force_body_name": peak_metric.get("max_nonfoot_force_body_name"),
        "peak_force_body_index": peak_metric.get("max_nonfoot_force_body_index"),
        "peak_force_physics_step": peak_metric.get("max_nonfoot_force_physics_step"),
        "peak_force_time_s": finite_number(
            peak_metric.get("max_nonfoot_force_time_s"), "peak force time"
        ),
        "worst_separation_m": min(available_separations)
        if available_separations
        else None,
    }


def validate_cpu(
    report: dict[str, Any], evidence: dict[str, str]
) -> dict[str, Any]:
    validate_common(report, "cpu")
    validate_readback(report)
    execution_id, started_at_utc = validate_execution(report, evidence)
    checks = report.get("checks")
    require(
        isinstance(checks, dict)
        and len(checks) > 0
        and tuple(checks) == EXPECTED_RUNTIME_CHECKS
        and all(type(value) is bool and value for value in checks.values()),
        "all CPU runtime checks must pass",
    )
    runtime = report.get("runtime_contract")
    require(
        isinstance(runtime, dict)
        and runtime.get("passed") is True
        and runtime.get("blocking_checks") == list(EXPECTED_RUNTIME_CHECKS),
        "CPU runtime contract/check list mismatch",
    )
    progression = report.get("progression_gate")
    require(
        isinstance(progression, dict)
        and progression.get("passed") is True
        and progression.get("status") == "passed"
        and progression.get("device") == "cpu"
        and progression.get("cpu_contact_separation_required_this_run") is True
        and progression.get("blocking_checks")
        == {"runtime_contract": True, "cpu_contact_separation": True},
        "CPU progression gate must pass with CPU separation authority",
    )
    require(report.get("passed") is True, "CPU top-level passed must be true")
    crosscheck = report.get("required_crosschecks", {}).get("cpu_contact_separation")
    require(
        isinstance(crosscheck, dict)
        and crosscheck.get("authority_device") == "cpu"
        and crosscheck.get("this_run_is_authority") is True
        and crosscheck.get("data_available") is True
        and crosscheck.get("threshold_m") == SEPARATION_THRESHOLD_M
        and crosscheck.get("threshold_passed") is True
        and crosscheck.get("status") == "observed"
        and crosscheck.get("passed") is True,
        "CPU separation authority evidence must pass",
    )
    values = extrema(report)
    require(
        math.isclose(
            values["peak_force_bodyweights"],
            EXPECTED_CPU_FORCE_BODYWEIGHTS,
            abs_tol=1e-12,
        ),
        "CPU peak force changed",
    )
    require(
        values["peak_force_env_index"] == 7
        and values["peak_force_pose"] == "right_side"
        and values["peak_force_action_mode"] == "reset_pose_hold"
        and values["peak_force_body_name"] == "base",
        "CPU peak force attribution changed",
    )
    require(
        values["peak_force_body_index"] == 0
        and values["peak_force_physics_step"] == 130
        and math.isclose(values["peak_force_time_s"], 0.65, abs_tol=1e-12),
        "CPU peak force step/time attribution changed",
    )
    separation = values["worst_separation_m"]
    require(
        separation is not None
        and math.isclose(separation, EXPECTED_CPU_SEPARATION_M, abs_tol=1e-12)
        and separation >= SEPARATION_THRESHOLD_M,
        "CPU contact separation changed or failed",
    )
    return {
        **evidence,
        "execution_id": execution_id,
        "started_at_utc": started_at_utc,
        **values,
    }


def validate_gpu(
    report: dict[str, Any], evidence: dict[str, str]
) -> dict[str, Any]:
    validate_common(report, "gpu")
    validate_readback(report)
    execution_id, started_at_utc = validate_execution(report, evidence)
    checks = report.get("checks")
    require(
        isinstance(checks, dict)
        and len(checks) > 0
        and tuple(checks) == EXPECTED_RUNTIME_CHECKS,
        "GPU runtime checks are missing, reordered, or unexpected",
    )
    assert isinstance(checks, dict)
    require(
        all(type(value) is bool for value in checks.values()),
        "GPU runtime checks must be booleans",
    )
    failed = [name for name, value in checks.items() if not value]
    require(
        failed == ["nonfoot_peak_force_bounded"],
        "GPU must fail only nonfoot_peak_force_bounded",
    )
    runtime = report.get("runtime_contract")
    require(
        isinstance(runtime, dict)
        and runtime.get("passed") is False
        and runtime.get("blocking_checks") == list(EXPECTED_RUNTIME_CHECKS),
        "GPU runtime contract failure evidence mismatch",
    )
    progression = report.get("progression_gate")
    require(
        isinstance(progression, dict)
        and progression.get("passed") is False
        and progression.get("status") == "runtime_contract_failed"
        and progression.get("device") == "cuda:0"
        and progression.get("cpu_contact_separation_required_this_run") is False
        and progression.get("blocking_checks") == {"runtime_contract": False},
        "GPU progression gate must reject the runtime failure",
    )
    require(report.get("passed") is False, "GPU top-level passed must be false")
    crosscheck = report.get("required_crosschecks", {}).get("cpu_contact_separation")
    require(
        isinstance(crosscheck, dict)
        and crosscheck.get("authority_device") == "cpu"
        and crosscheck.get("this_run_is_authority") is False
        and crosscheck.get("data_available") is False
        and crosscheck.get("threshold_m") == SEPARATION_THRESHOLD_M
        and crosscheck.get("threshold_passed") is None
        and crosscheck.get("status") == "requires_cpu_crosscheck"
        and crosscheck.get("passed") is None,
        "GPU must not claim CPU separation authority",
    )
    values = extrema(report)
    require(
        math.isclose(
            values["peak_force_bodyweights"],
            EXPECTED_GPU_FORCE_BODYWEIGHTS,
            abs_tol=1e-12,
        )
        and values["peak_force_bodyweights"] > FORCE_THRESHOLD_BODYWEIGHTS,
        "GPU peak force changed or no longer exceeds the gate",
    )
    require(
        values["peak_force_env_index"] == 7
        and values["peak_force_pose"] == "right_side"
        and values["peak_force_action_mode"] == "reset_pose_hold"
        and values["peak_force_body_name"] == "base",
        "GPU peak force attribution changed",
    )
    require(
        values["peak_force_body_index"] == 0
        and values["peak_force_physics_step"] == 129
        and math.isclose(values["peak_force_time_s"], 0.645, abs_tol=1e-12),
        "GPU peak force step/time attribution changed",
    )
    require(values["worst_separation_m"] is None, "GPU must not fabricate separation")
    require(
        all(
            item.get("min_contact_separation_m") is None
            and item.get("min_contact_separation_provenance") is None
            for item in report["pose_mode_metrics"]
        ),
        "GPU separation value/provenance must remain null",
    )
    return {
        **evidence,
        "execution_id": execution_id,
        "started_at_utc": started_at_utc,
        **values,
    }


def summarize(
    cpu_paths: Iterable[Path], gpu_paths: Iterable[Path]
) -> dict[str, Any]:
    cpu_paths, gpu_paths = tuple(cpu_paths), tuple(gpu_paths)
    require(
        len(cpu_paths) == len(gpu_paths) == 3,
        "exactly three CPU and three GPU reports are required",
    )
    all_paths = cpu_paths + gpu_paths
    require(
        len({path.resolve() for path in all_paths}) == 6,
        "six distinct report paths are required",
    )
    loaded = [read_json(path) for path in all_paths]
    reports = [item[0] for item in loaded]
    evidence = [item[1] for item in loaded]
    lineages = [validate_lineage(report) for report in reports]
    require(len(set(lineages)) == 1, "all runs must share one source lineage")
    expected_lineage = (REV15_SOURCE_COMMIT, REV15_SOURCE_BUNDLE, REV15_CONTRACT)
    require(lineages[0] == expected_lineage, "rev15 source lineage mismatch")

    cpu_runs = [validate_cpu(reports[index], evidence[index]) for index in range(3)]
    gpu_runs = [
        validate_gpu(reports[index], evidence[index]) for index in range(3, 6)
    ]
    require(
        len({run["execution_id"] for run in cpu_runs + gpu_runs}) == 6,
        "six unique UUID4 execution IDs are required",
    )
    require(
        len({run["started_at_utc"] for run in cpu_runs + gpu_runs}) == 6,
        "six unique RFC3339 UTC execution timestamps are required",
    )
    cpu_signatures = {
        (
            run["peak_force_bodyweights"],
            run["peak_force_env_index"],
            run["peak_force_pose"],
            run["peak_force_action_mode"],
            run["peak_force_body_name"],
            run["peak_force_body_index"],
            run["peak_force_physics_step"],
            run["peak_force_time_s"],
            run["worst_separation_m"],
        )
        for run in cpu_runs
    }
    gpu_signatures = {
        (
            run["peak_force_bodyweights"],
            run["peak_force_env_index"],
            run["peak_force_pose"],
            run["peak_force_action_mode"],
            run["peak_force_body_name"],
            run["peak_force_body_index"],
            run["peak_force_physics_step"],
            run["peak_force_time_s"],
            run["worst_separation_m"],
        )
        for run in gpu_runs
    }
    require(
        len(cpu_signatures) == len(gpu_signatures) == 1,
        "CPU/GPU semantic results must each reproduce across three runs",
    )

    force_excess = EXPECTED_GPU_FORCE_BODYWEIGHTS - FORCE_THRESHOLD_BODYWEIGHTS
    separation_margin = EXPECTED_CPU_SEPARATION_M - SEPARATION_THRESHOLD_M
    return {
        "schema_version": 1,
        "goal_id": "g009",
        "stage_number": "G009-5",
        "stage_id": "R0",
        "experiment": "rev15_position_solver_cpu_gpu_divergence",
        "status": "rejected_before_gate01",
        "evidence_synthesis_valid": True,
        "candidate_runtime_calibration_passed": False,
        "learned": False,
        "ppo_training": False,
        "ppo_training_status": "not_run",
        "qualification_status": "not_run",
        "qualification_passed": None,
        "conclusion": (
            "rev15 passes the authoritative CPU runtime and separation gate, but "
            "the GPU runtime reproducibly exceeds the 15 BW non-foot contact-force gate"
        ),
        "lineage": {
            "source_commit": lineages[0][0],
            "source_bundle_sha256": lineages[0][1],
            "contract_sha256": lineages[0][2],
        },
        "controlled_change": {
            "baseline_revision": "rev12",
            "solver_position_iterations": {"baseline": 8, "candidate": 16},
            "solver_velocity_iterations": {"baseline": 0, "candidate": 0},
            "max_depenetration_velocity_m_s": {"baseline": 1.0, "candidate": 1.0},
        },
        "repeatability": {
            "cpu": {
                "required_runs": 3,
                "validated_runs": 3,
                "semantically_identical": True,
                "inputs": cpu_runs,
            },
            "gpu": {
                "required_runs": 3,
                "validated_runs": 3,
                "semantically_identical": True,
                "inputs": gpu_runs,
            },
            "unique_execution_ids": 6,
        },
        "physics_readback": {
            "articulations_per_run": 8,
            "links_per_articulation": 19,
            "rigid_bodies_per_run": 152,
            "solver_position_iterations": EXPECTED_SOLVER_POSITION_ITERATIONS,
            "solver_velocity_iterations": EXPECTED_SOLVER_VELOCITY_ITERATIONS,
            "max_depenetration_velocity_m_s": EXPECTED_MAX_DEPENETRATION_VELOCITY_M_S,
            "all_paths_and_apis_valid": True,
        },
        "device_results": {
            "cpu": {
                "runtime_passed": True,
                "progression_gate_passed": True,
                "validated_runs": 3,
                "run_health_passed_runs": 3,
                "runtime_passed_runs": 3,
                "progression_gate_passed_runs": 3,
                "peak_nonfoot_force_bodyweights": EXPECTED_CPU_FORCE_BODYWEIGHTS,
                "force_threshold_bodyweights": FORCE_THRESHOLD_BODYWEIGHTS,
                "worst_contact_separation_m": EXPECTED_CPU_SEPARATION_M,
                "separation_threshold_m": SEPARATION_THRESHOLD_M,
            },
            "gpu": {
                "runtime_passed": False,
                "progression_gate_passed": False,
                "validated_runs": 3,
                "run_health_passed_runs": 3,
                "runtime_passed_runs": 0,
                "progression_gate_passed_runs": 0,
                "failed_checks": ["nonfoot_peak_force_bounded"],
                "peak_nonfoot_force_bodyweights": EXPECTED_GPU_FORCE_BODYWEIGHTS,
                "force_threshold_bodyweights": FORCE_THRESHOLD_BODYWEIGHTS,
                "contact_separation_authority": "cpu_only_not_evaluated_on_gpu",
            },
        },
        "divergence": {
            "same_seed_and_contract": True,
            "peak_force_difference_gpu_minus_cpu_bodyweights": (
                EXPECTED_GPU_FORCE_BODYWEIGHTS - EXPECTED_CPU_FORCE_BODYWEIGHTS
            ),
            "gpu_force_excess_bodyweights": force_excess,
            "gpu_force_excess_percent_of_threshold": (
                100.0 * force_excess / FORCE_THRESHOLD_BODYWEIGHTS
            ),
            "cpu_separation_margin_m": separation_margin,
            "cpu_separation_margin_mm": 1000.0 * separation_margin,
            "strict_decision": "reject",
        },
        "decision": {
            "strict_decision": "reject",
            "blocking_device": "gpu",
            "blocking_check": "nonfoot_peak_force_bounded",
            "threshold_bodyweights": FORCE_THRESHOLD_BODYWEIGHTS,
            "observed_bodyweights": EXPECTED_GPU_FORCE_BODYWEIGHTS,
            "overrun_bodyweights": force_excess,
            "overrun_percent_of_threshold": (
                100.0 * force_excess / FORCE_THRESHOLD_BODYWEIGHTS
            ),
        },
        "safety": {
            "numeric_invalid_terminations": 0,
            "hard_joint_limit_terminations": 0,
        },
        "completed_stages": {
            "cpu_runtime_3x": True,
            "gpu_runtime_3x": True,
            "strict_rejection_synthesis": True,
        },
        "blocked_stages": {"gate01": True, "gate10": True, "ppo_training": True},
    }


def write_summary(
    cpu_paths: Iterable[Path], gpu_paths: Iterable[Path], output_path: Path
) -> dict[str, Any]:
    require(not output_path.exists(), f"refusing to overwrite output: {output_path}")
    summary = summarize(cpu_paths, gpu_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", nargs=3, type=Path, default=DEFAULT_CPU)
    parser.add_argument("--gpu", nargs=3, type=Path, default=DEFAULT_GPU)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            write_summary(args.cpu, args.gpu, args.output),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
