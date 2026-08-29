from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g009_rev16_summary_test",
    ROOT / "scripts" / "summarize_g009_r0_rev16_backend_divergence.py",
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)
RAW_SPEC = importlib.util.spec_from_file_location(
    "g009_rev16_raw_fixture_test",
    ROOT / "tests" / "test_g009_r0_rev16_backend_divergence.py",
)
assert RAW_SPEC is not None and RAW_SPEC.loader is not None
RAW_TEST = importlib.util.module_from_spec(RAW_SPEC)
RAW_SPEC.loader.exec_module(RAW_TEST)
REAL_VALIDATE_PREDECESSOR_FILE = SUMMARY._validate_predecessor_file
REAL_RAW_VALIDATE_PREDECESSOR = SUMMARY.raw_probe.validate_predecessor_synthesis


def _expected_prefix_evidence(count: int) -> list[dict[str, str]]:
    values = []
    for arm, device in SUMMARY.GROUP_ORDER:
        for replicate in range(1, 4):
            path = (
                f"reports/runs/rev16_{arm}_{device.replace(':', '_')}_{replicate}.json"
            )
            values.append(
                {"path": path, "sha256": hashlib.sha256(path.encode()).hexdigest()}
            )
    return values[:count]


@pytest.fixture(autouse=True)
def _avoid_shared_repo_predecessor_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SUMMARY,
        "_validate_predecessor_file",
        lambda *args, **kwargs: _expected_prefix_evidence(kwargs["expected_count"]),
    )
    monkeypatch.setattr(
        SUMMARY.raw_probe,
        "validate_predecessor_synthesis",
        lambda path, **kwargs: {
            "path": f"reports/runs/{Path(path).name}",
            "sha256": f"{SUMMARY.PREDECESSOR_REQUIREMENTS[(kwargs['arm'], kwargs['device'])][0]:x}"
            * 64,
            "evidence_synthesis_valid": True,
            "validated_run_count": SUMMARY.PREDECESSOR_REQUIREMENTS[
                (kwargs["arm"], kwargs["device"])
            ][0],
            "next_group": SUMMARY.PREDECESSOR_REQUIREMENTS[
                (kwargs["arm"], kwargs["device"])
            ][1],
            "source_commit": kwargs["source_bundle"]["git_commit"],
            "source_bundle_sha256": kwargs["source_bundle"]["source_bundle_sha256"],
        },
    )


def _source_bundle() -> dict:
    files = {path: "1" * 64 for path in SUMMARY.raw_probe.SOURCE_BINDING_PATHS}
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return {
        "git_commit_valid": True,
        "all_files_present": True,
        "clean": True,
        "missing_files": [],
        "dirty_source_paths": [],
        "git_commit": "2" * 40,
        "source_binding_paths": list(files),
        "source_binding_files": files,
        "source_bundle_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def _contract(arm: str, device: str) -> dict:
    return SUMMARY.raw_probe.rev16_contract(arm, device)


FORCE_BODY_NAMES = [
    "base",
    *[f"link_{index}" for index in range(1, 15)],
    *[f"leg_{index}_foot" for index in range(4)],
]
LINK_BODY_NAMES = [
    FORCE_BODY_NAMES[0],
    FORCE_BODY_NAMES[2],
    FORCE_BODY_NAMES[1],
    *FORCE_BODY_NAMES[3:],
]


def _mass_evidence() -> dict:
    all_components = [[1.0] * 19 for _ in range(8)]
    all_components[7][0] = 2.0
    canonical_totals = [math.fsum(row) for row in all_components]
    return {
        "mass_accumulation": dict(SUMMARY.raw_probe.MASS_ACCUMULATION_CONTRACT),
        "body_mass_kg": all_components[7].copy(),
        "all_env_body_mass_kg": all_components,
        "total_mass_kg": canonical_totals[7],
        "all_env_total_mass_kg": canonical_totals,
        "body_weight_n": canonical_totals[7] * 9.81,
    }


def _forge_non_float32_mass_component(topology: dict) -> None:
    forged = 2.000000000000001
    topology["body_mass_kg"][0] = forged
    topology["all_env_body_mass_kg"][7][0] = forged
    canonical = math.fsum(topology["all_env_body_mass_kg"][7])
    topology["all_env_total_mass_kg"][7] = canonical
    topology["total_mass_kg"] = canonical
    topology["body_weight_n"] = canonical * 9.81


def _physics_rows(
    peak_step: int, peak_bw: float, neighbor_bw: float, body_weight: float
) -> list[dict]:
    rows = RAW_TEST._zero_physics_rows()
    for row in rows:
        requested_bw = peak_bw if row["physics_step"] == peak_step else neighbor_bw
        force = SUMMARY.raw_probe._float32(requested_bw * body_weight)
        base_bw = force / body_weight
        impulse = force * 0.005
        row["per_body_force_vector_n"][0] = [force, 0.0, 0.0]
        row["per_body_force_magnitude_n"][0] = force
        row["per_body_impulse_vector_n_s"][0] = [impulse, 0.0, 0.0]
        row["base_force_magnitude_n"] = force
        row["base_force_bodyweights"] = base_bw
        row["base_impulse_n_s"] = impulse
        row["nonfoot_resultant_force_vector_n"] = [force, 0.0, 0.0]
        row["nonfoot_resultant_force_n"] = force
        row["nonfoot_total_force_n"] = force
        row["nonfoot_impulse_vector_n_s"] = [impulse, 0.0, 0.0]
        row["nonfoot_impulse_n_s"] = impulse
    return rows


def _control_rows(speed: float) -> list[dict]:
    rows = RAW_TEST._zero_control_rows(LINK_BODY_NAMES)
    for row in rows:
        row["root_state_w"][10] = speed
        row["joint_velocity_rad_s"][0] = speed
    return rows


def _report(arm: str, device: str, replicate: int) -> tuple[dict, dict[str, str]]:
    group_values = {
        ("A", "cpu"): (131, 9.3, 0.5, 2.0),
        ("A", "cuda:0"): (130, 9.0, 0.5, 4.0),
        ("B", "cpu"): (130, 13.2, 1.0, 3.0),
        ("B", "cuda:0"): (129, 16.8, 0.01, 6.0),
    }
    peak_step, peak_bw, neighbor_bw, speed = group_values[(arm, device)]
    contract = _contract(arm, device)
    path = f"reports/runs/rev16_{arm}_{device.replace(':', '_')}_{replicate}.json"
    physics_clock = RAW_TEST._valid_clock_snapshot()
    mass_evidence = _mass_evidence()
    predecessor_requirement = SUMMARY.PREDECESSOR_REQUIREMENTS[(arm, device)]
    predecessor = None
    if predecessor_requirement is not None:
        validated_count, next_group = predecessor_requirement
        predecessor = {
            "path": f"reports/runs/rev16_prefix_{validated_count}.json",
            "sha256": f"{validated_count:x}" * 64,
            "evidence_synthesis_valid": True,
            "validated_run_count": validated_count,
            "next_group": next_group,
            "source_commit": "2" * 40,
            "source_bundle_sha256": _source_bundle()["source_bundle_sha256"],
        }
    report = {
        "schema_version": "g009.r0.rev16.backend_divergence.v1",
        "goal_id": "g009",
        "stage_id": "R0",
        "revision": "rev16",
        "status": "complete",
        "diagnostic_capture_complete": True,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "replicate_index": replicate,
        "device": device,
        "runtime_device": device,
        "headless": True,
        "seed": 42,
        "task": SUMMARY.raw_probe.DEFAULT_TASK,
        "num_envs": 8,
        "rollout_steps": 150,
        "pose_action_assignment": {
            "class_ids": [0, 1, 2, 3, 0, 1, 2, 3],
            "mapping": SUMMARY.raw_probe.expected_pose_action_assignment(),
        },
        "execution": {
            "execution_id": uuid.uuid4().hex,
            "no_overwrite": True,
            "output_path_repo_relative": path,
            "started_at_utc": "2026-08-29T00:00:00Z",
        },
        "predecessor_synthesis": predecessor,
        "contract": contract,
        "contract_sha256": SUMMARY.canonical_sha256(contract),
        "source_bundle": _source_bundle(),
        "governance": SUMMARY.raw_probe.governance(),
        "controlled_cell": {
            "source_env_index": 7,
            "pose_id": "right_side",
            "action_mode": "reset_pose_hold",
            "target_body_index": 0,
            "target_body_name": "base",
        },
        "safety_termination_counts": {
            "numeric_invalid": 0,
            "hard_joint_limit": 0,
        },
        "live_physics_readback": {
            "checks": {
                "articulation_solver_iteration_counts_match_contract": True,
                "rigid_body_max_depenetration_velocity_matches_contract": True,
            }
        },
        "physics_step_clock": physics_clock,
        "cpu_contact_authority": {
            "authority_device": "cpu",
            "this_run_is_authority": device == "cpu",
            "status": "observed" if device == "cpu" else "unavailable_on_gpu",
            "data_available": device == "cpu",
            "error": None,
            "subsequent_error_count": 0,
            "passed": True if device == "cpu" else None,
            "callback_event_count": 10 if device == "cpu" else 0,
            "physics_step_clock": physics_clock,
            "events": (
                [
                    {
                        "physics_step": step,
                        "callback_event_index": index,
                        "headers": [
                            {
                                "env_index": 7,
                                "event_type": "CONTACT_LOST",
                                "actor0_path": "/World/envs/env_7/Robot/base",
                                "actor1_path": "/World/ground",
                                "collider0_path": "/World/envs/env_7/Robot/base/collider",
                                "collider1_path": "/World/ground/collider",
                                "contact_points": [],
                            }
                        ],
                        "complete": True,
                    }
                    for index, step in zip((1, 5, 10), (1, 200, 600), strict=True)
                ]
                if device == "cpu"
                else None
            ),
            "all_env_minimum_separation_m": [-0.001] * 8 if device == "cpu" else None,
        },
        "historical_runtime_summary": {
            "reference_contract_id": SUMMARY.EXPECTED_REFERENCE[arm][0],
            "reference_contract_sha256": SUMMARY.EXPECTED_REFERENCE[arm][1],
            "matches_historical_reference": True,
            **SUMMARY.load_historical_target(arm, device),
        },
        "runtime_topology": {
            "force_body_names": FORCE_BODY_NAMES.copy(),
            "link_body_names": LINK_BODY_NAMES.copy(),
            "joint_names": [f"joint_{index}" for index in range(12)],
            "base_force_body_id": 0,
            "foot_force_body_ids": list(range(15, 19)),
            "nonfoot_force_body_ids": list(range(15)),
            "body_mass_body_names": LINK_BODY_NAMES.copy(),
            **mass_evidence,
        },
        "telemetry_timing": {
            "physics_dt_s": 0.005,
            "control_dt_s": 0.02,
            "control_decimation": 4,
            "history_order": "newest_to_oldest",
            "peak_window_radius_physics_steps": 8,
            "physics_row_derivation": SUMMARY.raw_probe.PHYSICS_ROW_DERIVATION,
        },
        "physics_substep_telemetry": _physics_rows(
            peak_step, peak_bw, neighbor_bw, mass_evidence["body_weight_n"]
        ),
        "control_step_telemetry": _control_rows(speed),
        "active_terminations": [
            "time_out",
            "stable_success",
            "numeric_invalid",
            "hard_joint_limit",
        ],
    }
    report["diagnostic_check_snapshot"] = (
        SUMMARY.raw_probe.build_diagnostic_check_snapshot(report)
    )
    report["diagnostic_capture_complete"] = report["diagnostic_check_snapshot"][
        "all_passed"
    ]
    evidence = {"path": path, "sha256": hashlib.sha256(path.encode()).hexdigest()}
    return report, evidence


def _entries(count: int = 12) -> list[tuple[dict, dict[str, str]]]:
    values = []
    for arm, device in SUMMARY.GROUP_ORDER:
        for replicate in range(1, 4):
            values.append(_report(arm, device, replicate))
    return values[:count]


@pytest.mark.parametrize(
    ("count", "next_group"),
    [(3, "A.cuda:0"), (6, "B.cpu"), (9, "B.cuda:0"), (12, None)],
)
def test_sequential_prefixes_are_valid_and_bound_for_next_group(
    count: int, next_group: str | None
) -> None:
    result = SUMMARY.synthesize_loaded(_entries(count))

    assert result["evidence_synthesis_valid"] is True
    assert result["run_matrix"]["validated_run_count"] == count
    assert result["next_group"] == next_group
    assert result["source_commit"] == "2" * 40
    assert len(result["input_reports"]) == count
    assert result["governance"]["position16_accepted"] is False
    assert result["governance"]["ppo"] == {
        "allowed": False,
        "status": "not_run",
    }


def test_historical_target_initializes_distinct_equal_projections() -> None:
    target = SUMMARY.load_historical_target("A", "cpu")

    assert target["historical_projection_derivation"] == (
        SUMMARY.raw_probe.HISTORICAL_PROJECTION_DERIVATION
    )
    assert target["historical_comparison_pose_metrics"] == target["pose_metrics"]
    assert target["historical_comparison_pose_metrics"] is not target["pose_metrics"]
    assert all(
        candidate is not historical
        for historical, candidate in zip(
            target["pose_metrics"],
            target["historical_comparison_pose_metrics"],
            strict=True,
        )
    )


def test_legacy_projection_can_reproduce_while_canonical_candidate_fails() -> None:
    report, _ = _report("B", "cuda:0", 1)
    summary = report["historical_runtime_summary"]

    historical_reproduced, runtime_candidate_passed = (
        SUMMARY._validate_historical_summary(report, "B", "cuda:0")
    )

    assert historical_reproduced is True
    assert runtime_candidate_passed is False
    assert summary["projection_pair_crosscheck"]["passed"] is True
    assert summary["checks"]["pose_metric_fingerprint_within_1e_6"] is True
    assert (
        summary["runtime_candidate_checks"]["max_nonfoot_force_at_most_15_bodyweights"]
        is False
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda summary: summary["historical_projection_derivation"].update(
                scope="canonical_runtime_candidate"
            ),
            "projection derivation",
        ),
        (
            lambda summary: summary.pop("historical_comparison_pose_metrics"),
            "historical comparison summary must contain 8 pose metrics",
        ),
        (
            lambda summary: summary["historical_comparison_pose_metrics"][0].update(
                pose_id="supine"
            ),
            "historical comparison summary pose/action mapping",
        ),
        (
            lambda summary: summary["pose_metrics"][0].update(
                max_nonfoot_force_bodyweights=15.1
            ),
            "compatibility tolerance",
        ),
        (
            lambda summary: summary["pose_metrics"][0].pop("max_joint_speed_rad_s"),
            "runtime candidate summary pose metric field set mismatch",
        ),
        (
            lambda summary: summary["projection_pair_crosscheck"].update(passed=False),
            "projection pair crosscheck",
        ),
    ],
)
def test_projection_derivation_identity_missing_and_forgery_fail_closed(
    mutation, message: str
) -> None:
    report, _ = _report("A", "cpu", 1)
    mutation(report["historical_runtime_summary"])

    with pytest.raises(ValueError, match=message):
        SUMMARY._validate_historical_summary(report, "A", "cpu")


def test_projection_pair_blocks_self_consistent_candidate_force_forgery() -> None:
    report, _ = _report("B", "cuda:0", 1)
    summary = report["historical_runtime_summary"]
    summary["pose_metrics"][7]["max_nonfoot_force_bodyweights"] = 0.0
    candidate_checks = SUMMARY._runtime_candidate_checks(
        summary["pose_metrics"], "cuda:0"
    )
    summary["runtime_candidate_checks"] = candidate_checks
    summary["runtime_candidate_passed"] = all(candidate_checks.values())

    with pytest.raises(ValueError, match="compatibility tolerance"):
        SUMMARY._validate_historical_summary(report, "B", "cuda:0")


def test_projection_pair_rejects_split_15_bw_classification_within_tolerance() -> None:
    report, _ = _report("A", "cpu", 1)
    summary = report["historical_runtime_summary"]
    summary["pose_metrics"][0]["max_nonfoot_force_bodyweights"] = 15.000001
    summary["historical_comparison_pose_metrics"][0][
        "max_nonfoot_force_bodyweights"
    ] = 14.999999

    with pytest.raises(ValueError, match="threshold classification"):
        SUMMARY._validate_historical_summary(report, "A", "cpu")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("max_nonfoot_force_physics_step", 999),
        ("max_nonfoot_force_body_index", 14),
        ("max_nonfoot_force_body_name", "forged_body"),
        ("max_root_angular_speed_rad_s", 123.0),
        ("max_joint_speed_rad_s", 456.0),
        ("min_contact_separation_m", -0.009),
        (
            "termination_counts",
            {
                "time_out": 0,
                "stable_success": 0,
                "numeric_invalid": 1,
                "hard_joint_limit": 0,
            },
        ),
    ],
)
def test_projection_pair_rejects_shared_field_mutation(
    field: str,
    replacement,
) -> None:
    report, _ = _report("A", "cpu", 1)
    report["historical_runtime_summary"]["pose_metrics"][7][field] = replacement

    with pytest.raises(ValueError, match=f"shared field {field} mismatch"):
        SUMMARY._validate_historical_summary(report, "A", "cpu")


def test_full_matrix_supports_hypothesis_three_of_three_without_accepting_arm_b() -> (
    None
):
    entries = _entries()
    assert all(
        report["runtime_topology"]["force_body_names"]
        != report["runtime_topology"]["link_body_names"]
        and report["runtime_topology"]["body_mass_body_names"]
        == report["runtime_topology"]["link_body_names"]
        for report, _ in entries
    )
    result = SUMMARY.synthesize_loaded(entries)

    assert result["hypothesis"]["decision"] == "supported"
    assert result["hypothesis"]["supported_3_of_3"] is True
    assert [item["passed"] for item in result["hypothesis"]["replicates"]] == [
        True,
        True,
        True,
    ]
    assert result["governance"]["position16_status"] == (
        "rejected_even_if_hypothesis_supported"
    )
    assert result["groups"][-1]["hypothesis_supported"] is True
    assert result["groups"][-1]["progression_allowed"] is False
    assert (
        len(result["hypothesis"]["replicates"][0]["derived"]["trace_pair_errors"]) == 6
    )
    divergences = result["hypothesis"]["replicates"][0]["derived"][
        "first_divergence_pairs"
    ]
    assert len(divergences) == 6
    assert divergences["a_cpu_vs_a_gpu"]["first_physics_divergence"]["step"] == 130
    assert divergences["b_cpu_vs_b_gpu"]["first_physics_divergence"]["step"] == 1
    assert divergences["a_cpu_vs_a_gpu"]["first_control_divergence"]["step"] == 1
    assert divergences["b_cpu_vs_b_gpu"]["first_overall_divergence"]["domain"] == (
        "physics"
    )
    exposure = result["groups"][0]["runs"][0]["contact_exposure"]
    assert exposure["first_any_contact_physics_step"] == 1
    assert exposure["first_base_contact_physics_step"] == 1
    assert set(exposure["thresholds"]) == {
        "over_5_bodyweights",
        "over_10_bodyweights",
        "over_15_bodyweights",
    }
    assert all(
        "traces" not in run for group in result["groups"] for run in group["runs"]
    )
    assert all(
        "physics_base_force_series" not in run and "control_series" not in run
        for group in result["groups"]
        for run in group["runs"]
    )


def _divergence_run() -> dict:
    control_step = {
        variable: [0.0]
        for variable in SUMMARY.DIVERGENCE_TOLERANCES
        if variable != "base_force_bodyweights"
    }
    return {
        "physics_base_force_series": [0.0] * 600,
        "control_series": [copy.deepcopy(control_step) for _ in range(150)],
    }


def test_divergence_tolerances_are_preregistered_and_boundary_is_not_divergence() -> (
    None
):
    assert set(SUMMARY.DIVERGENCE_TOLERANCES.values()) == {1.0e-6}
    left = _divergence_run()
    right = _divergence_run()
    right["physics_base_force_series"][9] = 1.0e-6
    for variable in right["control_series"][2]:
        right["control_series"][2][variable][0] = 1.0e-6

    result = SUMMARY._first_divergence(left, right)

    assert result["first_physics_divergence"] is None
    assert result["first_control_divergence"] is None
    assert result["first_overall_divergence"] is None


@pytest.mark.parametrize(
    "variable",
    [
        "root_state_w",
        "joint_position_rad",
        "joint_velocity_rad_s",
        "applied_torque_nm",
        "raw_action",
        "processed_ema_target_rad",
        "ema_previous_before_rad",
        "ema_previous_after_rad",
    ],
)
def test_first_control_divergence_identifies_each_variable(variable: str) -> None:
    left = _divergence_run()
    right = _divergence_run()
    right["control_series"][2][variable][0] = 1.0e-6 + 1.0e-9

    result = SUMMARY._first_divergence(left, right)

    assert result["first_control_divergence"] == {
        "step": 3,
        "time_s": 0.06,
        "variable": variable,
        "max_abs_delta": pytest.approx(1.001e-6),
        "tolerance": 1.0e-6,
    }
    assert result["first_overall_divergence"]["domain"] == "control"


def test_first_physics_divergence_precedes_later_control_divergence() -> None:
    left = _divergence_run()
    right = _divergence_run()
    right["physics_base_force_series"][9] = 2.0e-6
    right["control_series"][4]["root_state_w"][0] = 2.0e-6

    result = SUMMARY._first_divergence(left, right)

    assert result["first_physics_divergence"]["step"] == 10
    assert result["first_control_divergence"]["step"] == 5
    assert result["first_overall_divergence"] == {
        "domain": "physics",
        "step": 10,
        "time_s": 0.05,
        "variable": "base_force_bodyweights",
        "max_abs_delta": 2.0e-6,
        "tolerance": 1.0e-6,
    }


def test_first_contact_uses_one_newton_runtime_threshold() -> None:
    report, _ = _report("A", "cpu", 1)
    row = report["physics_substep_telemetry"][0]
    body_weight = report["runtime_topology"]["body_weight_n"]
    row["base_force_magnitude_n"] = 0.5
    row["base_force_bodyweights"] = 0.5 / body_weight
    row["base_impulse_n_s"] = 0.5 * 0.005
    row["foot_total_force_n"] = 0.0
    row["nonfoot_total_force_n"] = 0.5

    derived = SUMMARY._validate_physics_rows(report)

    assert derived["contact_exposure"]["first_any_contact_physics_step"] == 2
    assert derived["contact_exposure"]["first_base_contact_physics_step"] == 2


@pytest.mark.parametrize("count", [0, 1, 2, 4, 5, 7, 8, 10, 11, 13])
def test_only_complete_sequential_group_counts_are_allowed(count: int) -> None:
    entries = _entries()
    while len(entries) < count:
        entries.append(copy.deepcopy(entries[-1]))
    with pytest.raises(ValueError, match="exactly 3, 6, 9, or 12"):
        SUMMARY.synthesize_loaded(entries[:count])


@pytest.mark.parametrize(
    ("mutation", "_message"),
    [
        (lambda report: report.update(status="failed_closed"), "incomplete"),
        (lambda report: report.update(device="cuda:0"), "device/order"),
        (lambda report: report.update(replicate_index=2), "replicate"),
        (lambda report: report["execution"].update(no_overwrite=False), "no-overwrite"),
        (lambda report: report["execution"].update(execution_id="bad"), "UUID4"),
        (
            lambda report: report["source_bundle"].update(clean=False),
            "complete and clean",
        ),
        (lambda report: report.update(contract_sha256="0" * 64), "contract hash"),
        (lambda report: report["governance"]["ppo"].update(allowed=True), "governance"),
        (
            lambda report: report["controlled_cell"].update(pose_id="supine"),
            "controlled cell",
        ),
        (
            lambda report: report["safety_termination_counts"].update(
                numeric_invalid=1
            ),
            "safety",
        ),
        (
            lambda report: report["live_physics_readback"]["checks"].update(
                articulation_solver_iteration_counts_match_contract=False
            ),
            "readback",
        ),
        (
            lambda report: report["physics_step_clock"].update(callback_count=599),
            "physics step clock",
        ),
        (
            lambda report: report["cpu_contact_authority"]["events"][1].update(
                physics_step=0
            ),
            "outside 1..600",
        ),
        (
            lambda report: report["cpu_contact_authority"].update(
                callback_event_count=2
            ),
            "authority incomplete",
        ),
        (
            lambda report: report["historical_runtime_summary"].update(
                matches_historical_reference=False
            ),
            "reproduction flags",
        ),
        (
            lambda report: report["historical_runtime_summary"].update(
                reference_contract_sha256="0" * 64
            ),
            "historical reference contract hash",
        ),
        (
            lambda report: report["historical_runtime_summary"].update(passed=False),
            "reproduction flags",
        ),
        (lambda report: report["physics_substep_telemetry"].pop(), "600 physics"),
        (
            lambda report: report["physics_substep_telemetry"][0].pop(
                "per_body_force_vector_n"
            ),
            "physics row keys",
        ),
        (
            lambda report: report["physics_substep_telemetry"][5].update(
                physics_step=99
            ),
            "contiguous",
        ),
        (
            lambda report: report["physics_substep_telemetry"][5].update(
                contact_force_history_slot=3
            ),
            "history slot",
        ),
        (lambda report: report["control_step_telemetry"].pop(), "150 control"),
        (
            lambda report: report["control_step_telemetry"][0].pop("link_state_w"),
            "control row keys",
        ),
        (
            lambda report: report["control_step_telemetry"][0][
                "root_state_w"
            ].__setitem__(0, float("nan")),
            "finite",
        ),
        (
            lambda report: report["control_step_telemetry"][0][
                "termination_flags"
            ].update(hard_joint_limit=True),
            "safety termination",
        ),
    ],
)
def test_raw_report_mutations_fail_closed(mutation, _message: str) -> None:
    entries = _entries(3)
    mutation(entries[0][0])
    with pytest.raises(ValueError):
        SUMMARY.synthesize_loaded(entries)


@pytest.mark.parametrize(
    "mutation",
    [
        _forge_non_float32_mass_component,
        lambda topology: topology.__setitem__("native_total_mass_kg", 20.0),
        lambda topology: topology.__setitem__("native_minus_canonical_kg", 0.0),
        lambda topology: topology.__setitem__(
            "total_mass_kg", topology["total_mass_kg"] + 1.0
        ),
        lambda topology: topology["mass_accumulation"].__setitem__(
            "canonical_sum_method", "sum(serialized_components)"
        ),
    ],
)
def test_mass_provenance_and_canonical_derivation_mutations_fail_closed(
    mutation,
) -> None:
    entries = _entries(3)
    mutation(entries[0][0]["runtime_topology"])

    with pytest.raises(ValueError):
        SUMMARY.synthesize_loaded(entries)


def test_duplicate_execution_report_hash_and_source_binding_fail_closed() -> None:
    entries = _entries(3)
    entries[1][0]["execution"]["execution_id"] = entries[0][0]["execution"][
        "execution_id"
    ]
    with pytest.raises(ValueError, match="execution IDs"):
        SUMMARY.synthesize_loaded(entries)

    entries = _entries(3)
    entries[1][1]["sha256"] = entries[0][1]["sha256"]
    with pytest.raises(ValueError, match="report hashes"):
        SUMMARY.synthesize_loaded(entries)

    entries = _entries(3)
    entries[1][0]["source_bundle"] = _source_bundle()
    entries[1][0]["source_bundle"]["git_commit"] = "3" * 40
    with pytest.raises(ValueError, match="one source binding"):
        SUMMARY.synthesize_loaded(entries)


@pytest.mark.parametrize(
    ("field", "value", "_message"),
    [
        ("validated_run_count", 2, "sequence"),
        ("next_group", "B.cpu", "sequence"),
        ("source_commit", "3" * 40, "source binding"),
        ("sha256", "z" * 64, "lowercase hexadecimal"),
    ],
)
def test_predecessor_synthesis_binding_mutations_fail_closed(
    field: str, value, _message: str
) -> None:
    entries = _entries(6)
    entries[3][0]["predecessor_synthesis"][field] = value
    with pytest.raises(ValueError):
        SUMMARY.synthesize_loaded(entries)


def test_group_replicates_must_share_one_predecessor_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _entries(6)
    entries[4][0]["predecessor_synthesis"]["path"] = (
        "reports/runs/rev16_prefix_alt.json"
    )
    entries[4][0]["predecessor_synthesis"]["sha256"] = "a" * 64
    monkeypatch.setattr(
        SUMMARY.raw_probe,
        "validate_predecessor_synthesis",
        lambda path, **kwargs: {
            **entries[3][0]["predecessor_synthesis"],
            "path": f"reports/runs/{Path(path).name}",
            "sha256": "a" * 64 if Path(path).name.endswith("alt.json") else "3" * 64,
        },
    )
    with pytest.raises(ValueError, match="share one predecessor"):
        SUMMARY.synthesize_loaded(entries)


@pytest.mark.parametrize("mutation", ["swap", "missing", "foreign"])
def test_predecessor_input_reports_exactly_match_validated_prefix(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = _expected_prefix_evidence(3)
    if mutation == "swap":
        prefix[0], prefix[1] = prefix[1], prefix[0]
    elif mutation == "missing":
        prefix.pop()
    else:
        prefix[-1] = {
            "path": "reports/runs/foreign.json",
            "sha256": "f" * 64,
        }
    monkeypatch.setattr(
        SUMMARY, "_validate_predecessor_file", lambda *args, **kwargs: prefix
    )

    with pytest.raises(ValueError, match="validated prefix"):
        SUMMARY.synthesize_loaded(_entries(6))


def test_predecessor_binding_reopens_direct_child_and_verifies_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SUMMARY, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(SUMMARY.raw_probe, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        SUMMARY, "_validate_predecessor_file", REAL_VALIDATE_PREDECESSOR_FILE
    )
    runs = tmp_path / "reports" / "runs"
    runs.mkdir(parents=True)
    source = _source_bundle()
    payload = {
        "evidence_synthesis_valid": True,
        "run_matrix": {"validated_run_count": 3},
        "next_group": "A.cuda:0",
        "source_commit": source["git_commit"],
        "source_bundle_sha256": source["source_bundle_sha256"],
        "input_reports": _expected_prefix_evidence(3),
    }
    path = runs / "prefix.json"
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    binding = {
        "path": "reports/runs/prefix.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "evidence_synthesis_valid": True,
        "validated_run_count": 3,
        "next_group": "A.cuda:0",
        "source_commit": source["git_commit"],
        "source_bundle_sha256": source["source_bundle_sha256"],
    }
    monkeypatch.setattr(
        SUMMARY.raw_probe,
        "validate_predecessor_synthesis",
        lambda *args, **kwargs: dict(binding),
    )
    report = {"predecessor_synthesis": binding, "source_bundle": source}

    validated_binding, input_reports = SUMMARY._validate_predecessor_binding(
        report,
        "A",
        "cuda:0",
        source["git_commit"],
        source["source_bundle_sha256"],
    )
    assert validated_binding == binding
    assert input_reports == _expected_prefix_evidence(3)

    binding["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        SUMMARY._validate_predecessor_binding(
            report,
            "A",
            "cuda:0",
            source["git_commit"],
            source["source_bundle_sha256"],
        )

    binding["sha256"] = hashlib.sha256(raw).hexdigest()
    binding["path"] = "reports/runs/missing.json"
    with pytest.raises(FileNotFoundError):
        SUMMARY._validate_predecessor_binding(
            report,
            "A",
            "cuda:0",
            source["git_commit"],
            source["source_bundle_sha256"],
        )


def test_later_group_after_failed_predecessor_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _entries(6)
    validated = []
    for index, (report, evidence) in enumerate(entries):
        arm, device = SUMMARY.GROUP_ORDER[index // 3]
        validated.append(
            SUMMARY.validate_raw_report(report, evidence, arm, device, index % 3 + 1)
        )
    for run in validated[:3]:
        run["runtime_candidate_passed"] = False
    validated_iterator = iter(validated)
    monkeypatch.setattr(
        SUMMARY,
        "validate_raw_report",
        lambda *_args: next(validated_iterator),
    )

    with pytest.raises(ValueError, match="later group"):
        SUMMARY.synthesize_loaded(entries)


def test_hypothesis_requires_all_three_replicates_and_trace_equivalence() -> None:
    entries = _entries()
    entries[9][0]["control_step_telemetry"][0]["raw_action"][0] = 2.0e-6
    result = SUMMARY.synthesize_loaded(entries)

    assert result["hypothesis"]["decision"] == "inconclusive"
    assert result["hypothesis"]["supported_3_of_3"] is False
    assert (
        result["hypothesis"]["replicates"][0]["checks"][
            "action_and_ema_trace_error_at_most_1e_6"
        ]
        is False
    )


def test_main_refuses_to_overwrite_before_reading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SUMMARY, "RUNS_DIR", tmp_path)
    output = tmp_path / "exists.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        SUMMARY.main(["missing.json", "--output", str(output)])
