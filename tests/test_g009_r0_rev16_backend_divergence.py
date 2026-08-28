from __future__ import annotations

import importlib.util
import json
import math
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g009_rev16_backend_probe_test",
    ROOT / "scripts" / "probe_g009_r0_rev16_backend_divergence.py",
)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_rev16_arms_change_position_solver_only_and_forbid_progression() -> None:
    arm_a = PROBE.rev16_contract("A")
    arm_b = PROBE.rev16_contract("b")

    assert arm_a["arm"]["articulation_solver_position_iteration_count"] == 8
    assert arm_b["arm"]["articulation_solver_position_iteration_count"] == 16
    assert arm_a["historical_reference"] == {
        "contract_id": "g009_r0_recover_rev12",
        "canonical_sha256": (
            "d4b48d2b5fc1ea7684684a6324ba22fbfae767effeae45668c7310df382392e0"
        ),
        "role": "accepted_baseline_parameter_reference",
        "reference_scope": "contract_identity_and_physics_tuple_only",
        "historical_checkpoint_loaded": False,
        "historical_training_resumed": False,
        "current_rev16_execution_is_fresh": True,
    }
    assert arm_b["historical_reference"] == {
        "contract_id": "g009_r0_recover_rev15",
        "canonical_sha256": (
            "5f29ba19458404b5009d3734294c57e79294efecc7fe03bf8c71c71656129832"
        ),
        "role": "rejected_comparison_parameter_reference",
        "reference_scope": "contract_identity_and_physics_tuple_only",
        "historical_checkpoint_loaded": False,
        "historical_training_resumed": False,
        "current_rev16_execution_is_fresh": True,
    }
    for contract in (arm_a, arm_b):
        assert contract["diagnostic_only"] is True
        assert contract["qualification_eligible"] is False
        assert contract["arm"]["articulation_solver_velocity_iteration_count"] == 0
        assert contract["arm"]["max_depenetration_velocity_m_s"] == 1.0
        assert contract["training_and_gate_policy"] == {
            "ppo_allowed": False,
            "gate01_allowed": False,
            "gate10_allowed": False,
        }
        assert contract["controlled_cell"] == {
            "num_envs": 8,
            "source_env_index": 7,
            "pose_id": "right_side",
            "action_mode": "reset_pose_hold",
            "assignment_mode": "stratified",
            "pose_xy_range_m": [0.0, 0.0],
            "yaw_range_rad": [0.0, 0.0],
        }

    a_without_identity = {
        **arm_a,
        "arm": {**arm_a["arm"], "id": "B", "meaning": arm_b["arm"]["meaning"]},
    }
    assert {
        key: value
        for key, value in a_without_identity["arm"].items()
        if key != "articulation_solver_position_iteration_count"
    } == {
        key: value
        for key, value in arm_b["arm"].items()
        if key != "articulation_solver_position_iteration_count"
    }


def test_rev16_rejects_unknown_arm() -> None:
    with pytest.raises(ValueError, match="A or B"):
        PROBE.rev16_contract("C")


def test_source_binding_includes_new_probe_and_existing_runtime_contract() -> None:
    assert {
        "scripts/probe_g009_r0_rev16_backend_divergence.py",
        "scripts/probe_g009_recover_runtime.py",
        "src/isaac_walk_g009/recover_contracts.py",
        "src/isaac_walk_g009/recover_env_cfg.py",
    } <= set(PROBE.SOURCE_BINDING_PATHS)


def test_physics_history_rows_preserve_newest_first_slot_and_impulse() -> None:
    history = torch.zeros((4, 3, 3), dtype=torch.float64)
    history[0, 1] = torch.tensor([3.0, 4.0, 0.0])
    history[3, 2] = torch.tensor([0.0, 0.0, 10.0])

    rows = PROBE.physics_history_rows(
        history,
        control_step=2,
        physics_dt_s=0.005,
        body_names=["base", "FL_foot", "calf"],
        nonfoot_ids=[0, 2],
        foot_ids=[1],
        base_body_id=0,
        total_mass_kg=10.0,
    )

    assert [row["physics_step"] for row in rows] == [8, 7, 6, 5]
    assert [row["contact_force_history_slot"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["per_body_impulse_vector_n_s"][1] == pytest.approx(
        [0.015, 0.02, 0.0]
    )
    assert rows[3]["nonfoot_total_force_n"] == pytest.approx(10.0)
    assert rows[3]["nonfoot_impulse_n_s"] == pytest.approx(0.05)
    assert rows[0]["per_body_force_magnitude_n"] == pytest.approx([0.0, 5.0, 0.0])
    assert rows[0]["foot_total_force_n"] == pytest.approx(5.0)
    assert rows[3]["base_force_bodyweights"] == pytest.approx(0.0)


def test_physics_history_rows_reject_nonfinite_force() -> None:
    history = torch.zeros((4, 1, 3))
    history[2, 0, 1] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        PROBE.physics_history_rows(
            history,
            control_step=1,
            physics_dt_s=0.005,
            body_names=["base"],
            nonfoot_ids=[0],
            foot_ids=[],
            base_body_id=0,
            total_mass_kg=10.0,
        )


class _Header:
    actor0 = 1
    actor1 = 2
    collider0 = 3
    collider1 = 4
    contact_data_offset = 0
    num_contact_data = 1
    type = 2


class _EventTypes:
    CONTACT_FOUND = 0
    CONTACT_LOST = 1
    CONTACT_PERSIST = 2


def test_cpu_contact_extraction_records_pair_normal_and_separation() -> None:
    paths = {
        1: "/World/envs/env_7/Robot/base",
        2: "/World/ground",
        3: "/World/envs/env_7/Robot/base/collider",
        4: "/World/ground/collider",
    }
    datum = SimpleNamespace(
        separation=-0.002,
        position=(1.0, 2.0, 3.0),
        normal=(0.0, 0.0, 1.0),
        impulse=(0.0, 0.0, 0.25),
    )

    result = PROBE.extract_cpu_contact_points(
        [_Header()], [datum], source_env_index=7, int_to_path=paths.__getitem__
    )

    assert result["complete"] is True
    assert result["robot_ground_header_count"] == 1
    assert result["contact_points"][0]["contact_normal_w"] == [0.0, 0.0, 1.0]
    assert result["contact_points"][0]["contact_position_w_m"] == [1.0, 2.0, 3.0]
    assert result["contact_points"][0]["reported_contact_impulse_n_s"] == [
        0.0,
        0.0,
        0.25,
    ]
    assert result["contact_points"][0]["separation_m"] == pytest.approx(-0.002)
    assert result["contact_points"][0]["actor0_path"].endswith("/Robot/base")


def test_cpu_contact_authority_fails_closed_and_gpu_never_claims_it() -> None:
    paths = {
        1: "/World/envs/env_7/Robot/base",
        2: "/World/ground",
        3: "/World/envs/env_7/Robot/base/collider",
        4: "/World/ground/collider",
    }
    clock = PROBE.PhysicsStepClock(0.005)
    for _ in range(600):
        clock(0.005)
    accumulator = PROBE.CpuContactAuthorityAccumulator(
        7, 8, paths.__getitem__, clock, _EventTypes
    )
    accumulator([_Header()], [SimpleNamespace(separation=-0.001, normal=None)])

    cpu = accumulator.snapshot("cpu")
    gpu = accumulator.snapshot("cuda:0")
    assert cpu["status"] == "authority_unavailable"
    assert cpu["passed"] is False
    assert gpu["status"] == "unavailable_on_gpu"
    assert gpu["passed"] is None
    assert gpu["events"] is None


def test_contact_authority_preserves_first_error_and_counts_later_errors() -> None:
    clock = PROBE.PhysicsStepClock(0.005)
    accumulator = PROBE.CpuContactAuthorityAccumulator(
        7, 8, lambda value: str(value), clock, _EventTypes
    )

    accumulator.mark_unavailable(RuntimeError("first failure"))
    accumulator.mark_unavailable(ValueError("second failure"))
    accumulator.mark_unavailable(TypeError("third failure"))
    snapshot = accumulator.snapshot("cpu")

    assert snapshot["error"] == "RuntimeError: first failure"
    assert snapshot["subsequent_error_count"] == 2
    assert snapshot["passed"] is False


def test_contact_lost_with_zero_data_is_valid_and_stamped_by_physics_clock() -> None:
    lost = SimpleNamespace(
        actor0=1,
        actor1=2,
        collider0=3,
        collider1=4,
        contact_data_offset=0,
        num_contact_data=0,
        type=_EventTypes.CONTACT_LOST,
    )
    paths = {
        1: "/World/envs/env_7/Robot/base",
        2: "/World/ground",
        3: "/World/envs/env_7/Robot/base/collider",
        4: "/World/ground/collider",
    }
    clock = PROBE.PhysicsStepClock(0.005)
    clock(0.005)
    accumulator = PROBE.CpuContactAuthorityAccumulator(
        7, 8, paths.__getitem__, clock, _EventTypes
    )

    accumulator([lost], [])

    assert accumulator.events == [
        {
            "physics_step": 1,
            "callback_event_index": 1,
            "headers": [
                {
                    "env_index": 7,
                    "event_type": "CONTACT_LOST",
                    "actor0_path": paths[1],
                    "actor1_path": paths[2],
                    "collider0_path": paths[3],
                    "collider1_path": paths[4],
                    "contact_points": [],
                }
            ],
            "complete": True,
        }
    ]


@pytest.mark.parametrize(
    ("arm", "device", "count", "next_group"),
    [
        ("A", "cuda:0", 3, "A.cuda:0"),
        ("B", "cpu", 6, "B.cpu"),
        ("B", "cuda:0", 9, "B.cuda:0"),
    ],
)
def test_predecessor_synthesis_enforces_sequential_run_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arm: str,
    device: str,
    count: int,
    next_group: str,
) -> None:
    path, source = _write_valid_predecessor(monkeypatch, tmp_path, count)

    binding = PROBE.validate_predecessor_synthesis(
        path, arm=arm, device=device, source_bundle=source
    )

    assert binding["validated_run_count"] == count
    assert binding["next_group"] == next_group


def test_a_cpu_rejects_predecessor_and_later_groups_require_one(tmp_path: Path) -> None:
    source = {"git_commit": "1" * 40, "source_bundle_sha256": "2" * 64}
    assert (
        PROBE.validate_predecessor_synthesis(
            None, arm="A", device="cpu", source_bundle=source
        )
        is None
    )
    with pytest.raises(ValueError, match="must not accept"):
        PROBE.validate_predecessor_synthesis(
            tmp_path / "unused.json", arm="A", device="cpu", source_bundle=source
        )
    with pytest.raises(ValueError, match="requires --predecessor"):
        PROBE.validate_predecessor_synthesis(
            None, arm="B", device="cpu", source_bundle=source
        )


def test_predecessor_rejects_minimal_forged_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path, source = _write_valid_predecessor(monkeypatch, tmp_path, 3)
    path.write_text(
        json.dumps(
            {
                "evidence_synthesis_valid": True,
                "run_matrix": {"validated_run_count": 3},
                "next_group": "A.cuda:0",
                "source_commit": source["git_commit"],
                "source_bundle_sha256": source["source_bundle_sha256"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity/status/evidence"):
        PROBE.validate_predecessor_synthesis(
            path, arm="A", device="cuda:0", source_bundle=source
        )


@pytest.mark.parametrize("field", ["sha256", "path"])
def test_predecessor_rejects_input_hash_or_path_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    path, source = _write_valid_predecessor(monkeypatch, tmp_path, 3)
    synthesis = json.loads(path.read_text(encoding="utf-8"))
    synthesis["input_reports"][0][field] = (
        "0" * 64 if field == "sha256" else "reports/runs/../raw_01.json"
    )
    path.write_text(json.dumps(synthesis), encoding="utf-8")

    with pytest.raises(ValueError, match="input"):
        PROBE.validate_predecessor_synthesis(
            path, arm="A", device="cuda:0", source_bundle=source
        )


def _zero_physics_rows() -> list[dict]:
    body_names = [
        "base",
        *[f"link_{index}" for index in range(1, 15)],
        *[f"leg_{index}_foot" for index in range(4)],
    ]
    rows = []
    for control_step in range(1, 151):
        rows.extend(
            PROBE.physics_history_rows(
                torch.zeros((4, 19, 3), dtype=torch.float64),
                control_step=control_step,
                physics_dt_s=0.005,
                body_names=body_names,
                nonfoot_ids=list(range(15)),
                foot_ids=list(range(15, 19)),
                base_body_id=0,
                total_mass_kg=19.0,
            )
        )
    return sorted(rows, key=lambda row: row["physics_step"])


def _zero_control_rows(link_body_names: list[str] | None = None) -> list[dict]:
    body_names = link_body_names or [
        "base",
        *[f"link_{index}" for index in range(1, 15)],
        *[f"leg_{index}_foot" for index in range(4)],
    ]
    joint_names = [f"joint_{index}" for index in range(12)]
    return [
        {
            "control_step": step,
            "time_s": step * 0.02,
            "termination_flags": {
                "time_out": False,
                "stable_success": False,
                "numeric_invalid": False,
                "hard_joint_limit": False,
            },
            "root_state_w": [0.0] * 13,
            "link_state_field": "body_link_state_w",
            "link_names": body_names,
            "link_state_w": [[0.0] * 13 for _ in range(19)],
            "joint_names": joint_names,
            "joint_position_rad": [0.0] * 12,
            "joint_velocity_rad_s": [0.0] * 12,
            "applied_torque_nm": [0.0] * 12,
            "input_action": [0.0] * 12,
            "raw_action": [0.0] * 12,
            "processed_ema_target_rad": [0.0] * 12,
            "ema_previous_before_rad": [0.0] * 12,
            "ema_previous_after_rad": [0.0] * 12,
        }
        for step in range(1, 151)
    ]


def _valid_clock_snapshot(dt_s: float = 0.005) -> dict:
    clock = PROBE.PhysicsStepClock(0.005)
    for _ in range(600):
        clock(dt_s)
    return clock.snapshot()


def _complete_report(
    device: str = "cpu",
    arm: str = "A",
    predecessor: dict | None = None,
    *,
    link_body_names: list[str] | None = None,
) -> dict:
    contract = PROBE.rev16_contract(arm, device)
    reference_path = PROBE.REPO_ROOT / PROBE.HISTORICAL_REPORTS[(arm, device)]["path"]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    observed_metrics = PROBE._historical_pose_projection(reference, device)
    body_names = [
        "base",
        *[f"link_{index}" for index in range(1, 15)],
        *[f"leg_{index}_foot" for index in range(4)],
    ]
    link_body_names = link_body_names or body_names.copy()
    joint_names = [f"joint_{index}" for index in range(12)]
    mass_evidence = PROBE.build_mass_evidence(
        torch.ones((8, 19), dtype=torch.float32), 7
    )
    return {
        "schema_version": "g009.r0.rev16.backend_divergence.v1",
        "goal_id": "g009",
        "stage_id": "R0",
        "revision": "rev16",
        "status": "complete",
        "diagnostic_only": True,
        "qualification_eligible": False,
        "replicate_index": 1,
        "headless": True,
        "device": device,
        "runtime_device": device,
        "task": PROBE.DEFAULT_TASK,
        "seed": 42,
        "num_envs": 8,
        "rollout_steps": 150,
        "execution": {"execution_id": "fresh", "no_overwrite": True},
        "contract": contract,
        "contract_sha256": PROBE.canonical_sha256(contract),
        "source_bundle": {
            "all_files_present": True,
            "clean": True,
            "source_binding_paths": list(PROBE.SOURCE_BINDING_PATHS),
            "git_commit": "1" * 40,
            "source_bundle_sha256": "2" * 64,
        },
        "predecessor_synthesis": predecessor,
        "governance": PROBE.governance(),
        "controlled_cell": {
            "source_env_index": 7,
            "pose_id": "right_side",
            "action_mode": "reset_pose_hold",
            "target_body_index": 0,
            "target_body_name": "base",
        },
        "pose_action_assignment": {
            "class_ids": [0, 1, 2, 3, 0, 1, 2, 3],
            "mapping": PROBE.expected_pose_action_assignment(),
        },
        "live_physics_readback": {"checks": {"solver": True, "depenetration": True}},
        "runtime_topology": {
            "force_body_names": body_names,
            "link_body_names": link_body_names,
            "joint_names": joint_names,
            "base_force_body_id": 0,
            "foot_force_body_ids": list(range(15, 19)),
            "nonfoot_force_body_ids": list(range(15)),
            "body_mass_body_names": link_body_names,
            **mass_evidence,
        },
        "telemetry_timing": {
            "physics_dt_s": 0.005,
            "control_dt_s": 0.02,
            "control_decimation": 4,
            "history_order": "newest_to_oldest",
            "peak_window_radius_physics_steps": 8,
        },
        "physics_substep_telemetry": _zero_physics_rows(),
        "control_step_telemetry": _zero_control_rows(link_body_names),
        "active_terminations": [
            "time_out",
            "stable_success",
            "numeric_invalid",
            "hard_joint_limit",
        ],
        "physics_step_clock": _valid_clock_snapshot(),
        "safety_termination_counts": {
            "numeric_invalid": 0,
            "hard_joint_limit": 0,
        },
        "cpu_contact_authority": {
            "authority_device": "cpu",
            "this_run_is_authority": device == "cpu",
            "status": "observed" if device == "cpu" else "unavailable_on_gpu",
            "data_available": device == "cpu",
            "error": None,
            "subsequent_error_count": 0,
            "passed": True if device == "cpu" else None,
            "callback_event_count": 1 if device == "cpu" else 0,
            "physics_step_clock": _valid_clock_snapshot(),
            "events": [
                {
                    "physics_step": 1,
                    "callback_event_index": 1,
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
            ]
            if device == "cpu"
            else None,
            "all_env_minimum_separation_m": [-0.001] * 8 if device == "cpu" else None,
        },
        "historical_runtime_summary": PROBE.build_historical_runtime_summary(
            arm, device, observed_metrics
        ),
        "diagnostic_capture_complete": True,
    }


def _write_valid_predecessor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    count: int = 3,
) -> tuple[Path, dict[str, str]]:
    original_root = PROBE.REPO_ROOT
    source = {"git_commit": "1" * 40, "source_bundle_sha256": "2" * 64}
    runs_dir = tmp_path / "reports" / "runs"
    runs_dir.mkdir(parents=True)
    for reference in PROBE.HISTORICAL_REPORTS.values():
        destination = tmp_path / reference["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((original_root / reference["path"]).read_bytes())
    group_order = (("A", "cpu"), ("A", "cuda:0"), ("B", "cpu"))
    evidence_list: list[dict[str, str]] = []
    raw_reports: list[dict] = []
    for index in range(count):
        group_index = index // 3
        replicate = index % 3 + 1
        arm, device = group_order[group_index]
        predecessor = None
        if group_index > 0:
            predecessor_count = group_index * 3
            predecessor = {
                "path": f"reports/runs/prefix_{predecessor_count}.json",
                "sha256": f"{predecessor_count:x}" * 64,
                "evidence_synthesis_valid": True,
                "validated_run_count": predecessor_count,
                "next_group": f"{arm}.{device}",
                "source_commit": source["git_commit"],
                "source_bundle_sha256": source["source_bundle_sha256"],
            }
        report = _complete_report(device, arm, predecessor)
        report["replicate_index"] = replicate
        report["source_bundle"]["git_commit"] = source["git_commit"]
        report["source_bundle"]["source_bundle_sha256"] = source["source_bundle_sha256"]
        relative_path = f"reports/runs/raw_{index + 1:02d}.json"
        report["execution"] = {
            "execution_id": uuid.uuid4().hex,
            "started_at_utc": "2026-08-29T00:00:00Z",
            "output_path_repo_relative": relative_path,
            "no_overwrite": True,
        }
        raw_path = tmp_path / relative_path
        raw_bytes = (
            json.dumps(report, ensure_ascii=False, allow_nan=False).encode("utf-8")
            + b"\n"
        )
        raw_path.write_bytes(raw_bytes)
        evidence_list.append(
            {
                "path": relative_path,
                "sha256": PROBE.hashlib.sha256(raw_bytes).hexdigest(),
            }
        )
        raw_reports.append(report)
    groups = []
    for group_index in range(count // 3):
        arm, device = group_order[group_index]
        runs = []
        for replicate_offset in range(3):
            evidence_index = group_index * 3 + replicate_offset
            raw_report = raw_reports[evidence_index]
            runs.append(
                {
                    "evidence": evidence_list[evidence_index],
                    "arm": arm,
                    "device": device,
                    "replicate_index": replicate_offset + 1,
                    "source_commit": source["git_commit"],
                    "source_bundle_sha256": source["source_bundle_sha256"],
                    "contract_sha256": raw_report["contract_sha256"],
                    "historical_reproduction_passed": True,
                    "runtime_candidate_passed": True,
                }
            )
        groups.append(
            {
                "sequence_index": group_index + 1,
                "arm": arm,
                "device": device,
                "replicate_count": 3,
                "historical_reproduction_3_of_3": True,
                "runtime_candidate_expected_3_of_3": True,
                "sequence_gate_passed": True,
                "progression_allowed": True,
                "runs": runs,
            }
        )
    next_group = {3: "A.cuda:0", 6: "B.cpu", 9: "B.cuda:0"}[count]
    synthesis = {
        "schema_version": "g009.r0.rev16.backend_divergence_synthesis.v1",
        "goal_id": "g009",
        "stage_id": "R0",
        "revision": "rev16",
        "status": "complete",
        "evidence_synthesis_valid": True,
        "input_report_count": count,
        "input_reports": evidence_list,
        "source_commit": source["git_commit"],
        "source_bundle_sha256": source["source_bundle_sha256"],
        "run_matrix": {
            "validated_run_count": count,
            "validated_group_count": count // 3,
        },
        "next_group": next_group,
        "required_sequence": ["A.cpu", "A.cuda:0", "B.cpu", "B.cuda:0"],
        "completed_group_count": count // 3,
        "groups": groups,
        "hypothesis": {
            "decision": "pending_sequential_groups",
            "supported_3_of_3": None,
            "replicates": [],
        },
        "governance": {
            "position16_accepted": False,
            "position16_status": "rejected_even_if_hypothesis_supported",
            "diagnostic_only": True,
            "learned": False,
            "ppo": {"allowed": False, "status": "not_run"},
            "gate01": {"allowed": False, "status": "forbidden"},
            "gate10": {"allowed": False, "status": "forbidden"},
            "qualification": {"eligible": False, "status": "not_run", "passed": None},
        },
    }
    synthesis_path = runs_dir / f"synthesis_{count}.json"
    synthesis_path.write_text(json.dumps(synthesis), encoding="utf-8")
    monkeypatch.setattr(PROBE, "REPO_ROOT", tmp_path)
    return synthesis_path, source


def test_report_contract_accepts_complete_diagnostic_and_rejects_progression() -> None:
    report = _complete_report()
    PROBE.validate_report_contract(report)

    report["governance"] = {
        **report["governance"],
        "ppo": {"allowed": True, "status": "not_run"},
    }
    with pytest.raises(ValueError, match="governance"):
        PROBE.validate_report_contract(report)


def test_report_contract_accepts_distinct_sensor_and_robot_body_orders() -> None:
    force_order = _complete_report()["runtime_topology"]["force_body_names"]
    link_order = [*force_order[1:], force_order[0]]
    report = _complete_report(link_body_names=link_order)

    PROBE.validate_report_contract(report)

    assert report["physics_substep_telemetry"][0]["body_names"] == force_order
    assert report["control_step_telemetry"][0]["link_names"] == link_order
    assert report["runtime_topology"]["body_mass_body_names"] == link_order


def test_report_contract_rejects_sensor_robot_body_set_mismatch() -> None:
    report = _complete_report()
    report["runtime_topology"]["link_body_names"][-1] = "foreign_link"
    report["runtime_topology"]["body_mass_body_names"][-1] = "foreign_link"
    report["control_step_telemetry"][0]["link_names"][-1] = "foreign_link"

    with pytest.raises(ValueError, match="force/link/mass body topology mismatch"):
        PROBE.validate_report_contract(report)


def test_report_contract_rejects_mass_order_not_bound_to_robot_links() -> None:
    report = _complete_report()
    mass_order = report["runtime_topology"]["body_mass_body_names"].copy()
    report["runtime_topology"]["body_mass_body_names"] = [
        *mass_order[1:],
        mass_order[0],
    ]

    with pytest.raises(ValueError, match="force/link/mass body topology mismatch"):
        PROBE.validate_report_contract(report)


def test_canonical_mass_fsum_excludes_large_native_float32_reduction_error() -> None:
    components = torch.tensor([1.0e8, *([1.0] * 18)], dtype=torch.float32)
    masses = components.repeat(8, 1)
    evidence = PROBE.build_mass_evidence(masses, 7)

    PROBE.validate_mass_evidence(evidence)

    native_total = float(masses[7].sum().item())
    assert evidence["total_mass_kg"] == 100000018.0
    assert native_total == 100000000.0
    assert abs(native_total - evidence["total_mass_kg"]) > 1.0e-6
    force_history = torch.zeros((4, 19, 3), dtype=torch.float32)
    force_history[:, 0, 0] = 1000.0
    rows = PROBE.physics_history_rows(
        force_history,
        control_step=1,
        physics_dt_s=0.005,
        body_names=["base", *[f"link_{index}" for index in range(1, 19)]],
        nonfoot_ids=list(range(19)),
        foot_ids=[],
        base_body_id=0,
        total_mass_kg=evidence["total_mass_kg"],
    )
    canonical_bw = 1000.0 / (evidence["total_mass_kg"] * 9.81)
    native_bw = 1000.0 / (native_total * 9.81)
    assert rows[0]["base_force_bodyweights"] == pytest.approx(canonical_bw, abs=1.0e-15)
    assert rows[0]["base_force_bodyweights"] != native_bw


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence: evidence.__setitem__(
            "total_mass_kg", evidence["total_mass_kg"] + 1.0
        ),
        lambda evidence: evidence["body_mass_kg"].__setitem__(0, 2.0),
        lambda evidence: evidence["mass_accumulation"].__setitem__(
            "canonical_sum_method", "sum"
        ),
        lambda evidence: evidence["mass_accumulation"].__setitem__(
            "component_storage_dtype", "torch.float64"
        ),
        lambda evidence: evidence["all_env_total_mass_kg"].pop(),
        lambda evidence: evidence["all_env_total_mass_kg"].__setitem__(0, True),
    ],
)
def test_mass_evidence_rejects_forged_components_methods_and_derivations(
    mutation,
) -> None:
    evidence = PROBE.build_mass_evidence(torch.ones((8, 19), dtype=torch.float32), 7)
    mutation(evidence)

    with pytest.raises(ValueError):
        PROBE.validate_mass_evidence(evidence)


@pytest.mark.parametrize("forged_component", [1.000000000000001, -1000.0, 1.0e100])
def test_mass_evidence_rejects_rehashed_non_float32_or_nonpositive_components(
    forged_component: float,
) -> None:
    evidence = PROBE.build_mass_evidence(torch.ones((8, 19), dtype=torch.float32), 7)
    evidence["body_mass_kg"][0] = forged_component
    evidence["all_env_body_mass_kg"][7][0] = forged_component
    canonical = math.fsum(evidence["all_env_body_mass_kg"][7])
    evidence["total_mass_kg"] = canonical
    evidence["all_env_total_mass_kg"][7] = canonical
    evidence["body_weight_n"] = canonical * 9.81

    with pytest.raises(ValueError, match="exact float32"):
        PROBE.validate_mass_evidence(evidence)


@pytest.mark.parametrize("forged_native_total", [-1000.0, 1.0e100, 20.0])
def test_mass_evidence_rejects_retired_native_total_provenance(
    forged_native_total: float,
) -> None:
    evidence = PROBE.build_mass_evidence(torch.ones((8, 19), dtype=torch.float32), 7)
    evidence["native_total_mass_kg"] = forged_native_total

    with pytest.raises(ValueError, match="mass evidence fields"):
        PROBE.validate_mass_evidence(evidence)


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("base_force_body_id", True),
        ("foot_force_body_ids", [15, 16, 17, 17]),
        ("foot_force_body_ids", [True, 16, 17, 18]),
        ("foot_force_body_ids", [15, 16, 17, 19]),
        ("nonfoot_force_body_ids", [*range(14), 13]),
        ("nonfoot_force_body_ids", [True, *range(1, 15)]),
    ],
)
def test_report_contract_rejects_non_strict_force_body_ids(
    field: str, mutation: object
) -> None:
    report = _complete_report()
    report["runtime_topology"][field] = mutation

    with pytest.raises(ValueError, match="body id|foot/nonfoot topology"):
        PROBE.validate_report_contract(report)


def test_report_contract_rejects_force_ids_not_matching_sensor_labels() -> None:
    report = _complete_report()
    report["runtime_topology"]["foot_force_body_ids"] = [14, 16, 17, 18]
    report["runtime_topology"]["nonfoot_force_body_ids"] = [*range(14), 15]

    with pytest.raises(ValueError, match="force body labels"):
        PROBE.validate_report_contract(report)


def test_predecessor_rejects_rehashed_raw_report_with_duplicate_nonfoot_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    synthesis_path, source = _write_valid_predecessor(monkeypatch, tmp_path, 3)
    raw_path = tmp_path / "reports" / "runs" / "raw_01.json"
    raw_report = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_report["runtime_topology"]["nonfoot_force_body_ids"] = [*range(14), 13]
    raw_bytes = (
        json.dumps(raw_report, ensure_ascii=False, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    raw_path.write_bytes(raw_bytes)
    evidence = {
        "path": "reports/runs/raw_01.json",
        "sha256": PROBE.hashlib.sha256(raw_bytes).hexdigest(),
    }
    synthesis = json.loads(synthesis_path.read_text(encoding="utf-8"))
    synthesis["input_reports"][0] = evidence
    synthesis["groups"][0]["runs"][0]["evidence"] = evidence
    synthesis_path.write_text(json.dumps(synthesis), encoding="utf-8")

    with pytest.raises(ValueError, match="foot/nonfoot topology"):
        PROBE.validate_predecessor_synthesis(
            synthesis_path, arm="A", device="cuda:0", source_bundle=source
        )


def test_report_contract_requires_exact_eight_env_pose_action_assignment() -> None:
    report = _complete_report()
    PROBE.validate_report_contract(report)

    report["pose_action_assignment"]["class_ids"][-1] = 2
    with pytest.raises(ValueError, match="pose/action assignment"):
        PROBE.validate_report_contract(report)

    report = _complete_report()
    report["pose_action_assignment"]["mapping"][4]["action_mode"] = "zero_normalized"
    with pytest.raises(ValueError, match="pose/action assignment"):
        PROBE.validate_report_contract(report)


def test_report_contract_rejects_rehashed_historical_lineage_drift() -> None:
    report = _complete_report()
    report["contract"]["historical_reference"]["canonical_sha256"] = "0" * 64
    report["contract_sha256"] = PROBE.canonical_sha256(report["contract"])

    with pytest.raises(ValueError, match="historical lineage"):
        PROBE.validate_report_contract(report)


def test_report_contract_requires_full_time_series_and_cpu_authority() -> None:
    report = _complete_report("cpu")
    PROBE.validate_report_contract(report)

    report["physics_substep_telemetry"].pop()
    with pytest.raises(ValueError, match="600 physics"):
        PROBE.validate_report_contract(report)

    report = _complete_report("cpu")
    report["cpu_contact_authority"]["passed"] = False
    with pytest.raises(ValueError, match="CPU contact authority"):
        PROBE.validate_report_contract(report)


def test_report_contract_requires_explicit_replicate_and_headless_device() -> None:
    report = _complete_report()
    report["replicate_index"] = 4
    with pytest.raises(ValueError, match="replicate_index"):
        PROBE.validate_report_contract(report)

    report = _complete_report()
    report["headless"] = False
    with pytest.raises(ValueError, match="headless"):
        PROBE.validate_report_contract(report)

    report = _complete_report()
    report["device"] = "cuda:1"
    with pytest.raises(ValueError, match="device"):
        PROBE.validate_report_contract(report)

    report = _complete_report()
    report["task"] = "wrong-task"
    with pytest.raises(ValueError, match="task"):
        PROBE.validate_report_contract(report)

    report = _complete_report()
    report["seed"] = 43
    with pytest.raises(ValueError, match="seed"):
        PROBE.validate_report_contract(report)

    report = _complete_report()
    report["contract"]["execution_conditions"]["runtime_device"] = "cuda:0"
    report["contract_sha256"] = PROBE.canonical_sha256(report["contract"])
    with pytest.raises(ValueError, match="contract or historical"):
        PROBE.validate_report_contract(report)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["physics_substep_telemetry"][0].update(
                physics_step=2
            ),
            "physics steps",
        ),
        (
            lambda report: report["physics_substep_telemetry"][0].update(
                contact_force_history_slot=1
            ),
            "history-slot",
        ),
        (
            lambda report: report["physics_substep_telemetry"][0].pop(
                "base_impulse_n_s"
            ),
            "keys",
        ),
        (
            lambda report: report["physics_substep_telemetry"][0][
                "per_body_force_vector_n"
            ].pop(),
            "19x3",
        ),
        (
            lambda report: report["physics_substep_telemetry"][0][
                "per_body_force_vector_n"
            ][0].__setitem__(0, float("nan")),
            "finite",
        ),
        (
            lambda report: report["physics_substep_telemetry"][0].update(
                base_force_bodyweights=1.0
            ),
            "base BW",
        ),
        (
            lambda report: report["physics_substep_telemetry"][0][
                "per_body_impulse_vector_n_s"
            ][0].__setitem__(0, 1.0),
            "body impulse",
        ),
    ],
)
def test_raw_physics_validator_rejects_mutations(mutation, message: str) -> None:
    report = _complete_report()
    mutation(report)

    with pytest.raises(ValueError, match=message):
        PROBE.validate_report_contract(report)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["control_step_telemetry"][0].update(control_step=2),
            "control steps",
        ),
        (
            lambda report: report["control_step_telemetry"][0].pop("root_state_w"),
            "keys",
        ),
        (
            lambda report: report["control_step_telemetry"][0]["link_state_w"].pop(),
            "19x13",
        ),
        (
            lambda report: report["control_step_telemetry"][0][
                "joint_position_rad"
            ].pop(),
            "shape",
        ),
        (
            lambda report: report["control_step_telemetry"][0][
                "root_state_w"
            ].__setitem__(0, float("inf")),
            "finite",
        ),
    ],
)
def test_raw_control_validator_rejects_mutations(mutation, message: str) -> None:
    report = _complete_report()
    mutation(report)

    with pytest.raises(ValueError, match=message):
        PROBE.validate_report_contract(report)


def test_historical_summary_rejects_reference_fingerprint_mutation() -> None:
    report = _complete_report()
    report["historical_runtime_summary"]["pose_metrics"][0][
        "max_root_angular_speed_rad_s"
    ] += 2.0e-6

    with pytest.raises(ValueError, match="historical runtime summary"):
        PROBE.validate_report_contract(report)


@pytest.mark.parametrize("callback_dt", [0.004999999888241291, 0.005])
def test_physics_clock_accepts_float32_and_double_contract_values(
    callback_dt: float,
) -> None:
    snapshot = _valid_clock_snapshot(callback_dt)

    assert snapshot["evidence_kind"] == "pre_step_notification_count"
    assert snapshot["contract_expected_dt_s"] == 0.005
    assert snapshot["callback_expected_float32_dt_s"] == 0.004999999888241291
    assert snapshot["callback_dt_abs_tolerance_s"] == 2.5e-10
    assert snapshot["callback_count"] == 600
    assert snapshot["mismatch_count"] == 0
    assert snapshot["nonfinite_count"] == 0
    assert snapshot["first_mismatch"] is None
    assert snapshot["passed"] is True


@pytest.mark.parametrize("callback_dt", [0.005000000353902578, float("nan")])
def test_physics_clock_rejects_outside_boundary_and_nonfinite_dt(
    callback_dt: float,
) -> None:
    clock = PROBE.PhysicsStepClock(0.005)
    for _ in range(600):
        clock(callback_dt)
    snapshot = clock.snapshot()

    assert snapshot["mismatch_count"] == 600
    assert snapshot["nonfinite_count"] == (600 if math.isnan(callback_dt) else 0)
    assert snapshot["first_mismatch"]["callback_index"] == 1
    assert snapshot["passed"] is False


def test_physics_clock_rejects_callback_count_drift() -> None:
    report = _complete_report()
    report["physics_step_clock"]["callback_count"] = 599
    report["physics_step_clock"]["passed"] = False

    with pytest.raises(ValueError, match="600 pre-step notifications"):
        PROBE.validate_report_contract(report)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("callback_count", True, "count types"),
        ("mismatch_count", True, "count types"),
        ("max_abs_error_s", float("nan"), "finite"),
        ("max_abs_error_s", 1.0e-9, "derivation"),
    ],
)
def test_physics_clock_validator_rejects_type_nonfinite_and_derived_mutations(
    field: str, value: object, message: str
) -> None:
    report = _complete_report()
    report["physics_step_clock"][field] = value

    with pytest.raises(ValueError, match=message):
        PROBE.validate_report_contract(report)


def test_failure_envelope_keeps_structured_partial_clock_evidence() -> None:
    clock = PROBE.PhysicsStepClock(0.005)
    for _ in range(599):
        clock(0.004999999888241291)
    evidence = clock.snapshot()
    args = SimpleNamespace(
        arm="A",
        replicate_index=1,
        headless=True,
        device="cpu",
        seed=42,
        task=PROBE.DEFAULT_TASK,
    )

    report = PROBE.failure_envelope(
        args,
        {"execution_id": "fresh", "no_overwrite": True},
        PROBE.PhysicsStepClockEvidenceError(evidence),
    )

    assert report["status"] == "failed_closed"
    assert report["physics_step_clock"] == evidence
    assert report["physics_step_clock"]["callback_count"] == 599
    assert report["error"] == {
        "type": "PhysicsStepClockEvidenceError",
        "message": "physics pre-step notification clock validation failed",
    }


def test_failure_envelope_preserves_mass_evidence_and_original_error() -> None:
    evidence = PROBE.build_mass_evidence(torch.ones((8, 19), dtype=torch.float32), 7)
    args = SimpleNamespace(
        arm="A",
        replicate_index=1,
        headless=True,
        device="cpu",
        seed=42,
        task=PROBE.DEFAULT_TASK,
    )

    report = PROBE.failure_envelope(
        args,
        {"execution_id": "fresh", "no_overwrite": True},
        PROBE.MassEvidenceError(evidence, ValueError("forged canonical total")),
    )

    assert report["mass_evidence"] == evidence
    assert report["physics_step_clock"] is None
    assert report["error"] == {
        "type": "ValueError",
        "message": "forged canonical total",
    }


def test_failure_envelope_preserves_completed_clock_and_mass_without_telemetry() -> (
    None
):
    clock = _valid_clock_snapshot()
    mass = PROBE.build_mass_evidence(torch.ones((8, 19), dtype=torch.float32), 7)
    args = SimpleNamespace(
        arm="A",
        replicate_index=1,
        headless=True,
        device="cpu",
        seed=42,
        task=PROBE.DEFAULT_TASK,
    )

    report = PROBE.failure_envelope(
        args,
        {"execution_id": "fresh", "no_overwrite": True},
        PROBE.DiagnosticEvidenceError(
            ValueError("final report rejected"),
            physics_step_clock=clock,
            mass_evidence=mass,
        ),
    )

    assert report["physics_step_clock"] == clock
    assert report["mass_evidence"] == mass
    assert report["error"] == {
        "type": "ValueError",
        "message": "final report rejected",
    }
    assert "physics_substep_telemetry" not in report
    assert "control_step_telemetry" not in report


def test_contact_event_rejects_callback_stamp_and_lost_data_mutations() -> None:
    report = _complete_report()
    report["cpu_contact_authority"]["events"][0]["physics_step"] = 0
    with pytest.raises(ValueError, match="physics-step stamp"):
        PROBE.validate_report_contract(report)

    report = _complete_report()
    report["cpu_contact_authority"]["events"][0]["headers"][0]["contact_points"] = [{}]
    with pytest.raises(ValueError, match="FOUND/PERSIST/LOST"):
        PROBE.validate_report_contract(report)


def test_failure_envelope_preserves_original_when_provenance_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_provenance() -> dict:
        raise RuntimeError("secondary provenance failure")

    monkeypatch.setattr(PROBE, "source_bundle_provenance", broken_provenance)
    args = SimpleNamespace(
        arm="A",
        replicate_index=1,
        headless=True,
        device="cpu",
        seed=42,
        task=PROBE.DEFAULT_TASK,
    )

    report = PROBE.failure_envelope(
        args,
        {"execution_id": "fresh", "no_overwrite": True},
        RuntimeError("primary AppLauncher failure"),
    )

    assert report["status"] == "failed_closed"
    assert report["error"] == {
        "type": "RuntimeError",
        "message": "primary AppLauncher failure",
    }
    assert report["source_bundle"]["error"].endswith("secondary provenance failure")
    assert report["failure_envelope_errors"] == [
        "source_bundle: RuntimeError: secondary provenance failure"
    ]
