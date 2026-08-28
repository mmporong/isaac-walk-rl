from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g009_runtime_probe_under_test", ROOT / "scripts" / "probe_g009_recover_runtime.py"
)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_calibration_budget_requires_eight_environments_and_three_seconds() -> None:
    PROBE.validate_calibration_budget(8, 150)

    with pytest.raises(ValueError, match="8 environments"):
        PROBE.validate_calibration_budget(4, 150)
    with pytest.raises(ValueError, match="150 control steps"):
        PROBE.validate_calibration_budget(8, 10)


def test_source_bundle_covers_training_and_probe_semantics() -> None:
    paths = set(PROBE.SOURCE_BINDING_PATHS)

    assert {
        "configs/g009_r0.json",
        "scripts/bootstrap_train_g009.py",
        "scripts/probe_g009_recover_runtime.py",
        "scripts/run_training.ps1",
        "scripts/synthesize_g009_r0_probe.py",
        "src/isaac_walk_g009/mdp/recover.py",
        "src/isaac_walk_g009/recover_contracts.py",
        "src/isaac_walk_g009/recover_env_cfg.py",
        "src/isaac_walk_g009/registry.py",
    } <= paths
    assert len(paths) == len(PROBE.SOURCE_BINDING_PATHS)


def test_source_bundle_records_current_file_hashes_and_git_identity() -> None:
    result = PROBE.source_bundle_provenance()

    assert result["schema_version"] == 1
    assert result["git_commit_valid"] is True
    assert result["all_files_present"] is True
    assert result["missing_files"] == []
    assert len(result["source_bundle_sha256"]) == 64
    assert set(result["source_binding_files"]) == set(PROBE.SOURCE_BINDING_PATHS)


def test_reward_temporal_expectations_preserve_terminal_and_potential_magnitudes() -> None:
    result = PROBE.reward_temporal_expectations(0.02)

    assert result["terminal_raw_pulse"] == pytest.approx(50.0)
    assert result["terminal_contribution"] == pytest.approx(10.0)
    assert result["potential_raw_rate"] == pytest.approx(12.375)
    assert result["potential_contribution"] == pytest.approx(0.495)


@pytest.mark.parametrize("step_dt", [0.0, -0.02, float("inf"), float("nan")])
def test_reward_temporal_expectations_reject_invalid_step_dt(step_dt: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        PROBE.reward_temporal_expectations(step_dt)


def test_pose_mode_rows_assign_each_pose_to_both_action_modes() -> None:
    rows = PROBE.pose_mode_rows(["prone", "supine", "left_side", "right_side"])

    assert [(row["pose_id"], row["action_mode"]) for row in rows] == [
        ("prone", "zero_normalized"),
        ("supine", "zero_normalized"),
        ("left_side", "zero_normalized"),
        ("right_side", "zero_normalized"),
        ("prone", "reset_pose_hold"),
        ("supine", "reset_pose_hold"),
        ("left_side", "reset_pose_hold"),
        ("right_side", "reset_pose_hold"),
    ]


def test_physics_thresholds_are_blocking_calibration_values() -> None:
    assert PROBE.MIN_ROOT_HEIGHT_M == pytest.approx(0.02)
    assert PROBE.MIN_CONTACT_SEPARATION_M == pytest.approx(-0.01)
    assert PROBE.MAX_NON_FOOT_FORCE_BODYWEIGHTS == pytest.approx(15.0)
    assert PROBE.MAX_EXCESS_CONTACT_DELTA_V_M_S == pytest.approx(3.0)
    assert PROBE.TAIL_STEPS == 25
    assert PROBE.MAX_TAIL_HORIZONTAL_SPEED_M_S == pytest.approx(0.5)
    assert PROBE.MAX_TAIL_VERTICAL_SPEED_M_S == pytest.approx(0.25)
    assert PROBE.MAX_TAIL_ANGULAR_SPEED_RAD_S == pytest.approx(2.0)


def test_rev3_observation_schema_uses_explicit_actor_and_critic_indices() -> None:
    assert PROBE.ACTOR_FOOT_LOAD_SLICE == slice(49, 53)
    assert PROBE.ACTOR_RANGE_SLICE == slice(53, 68)
    assert PROBE.ACTOR_RANGE_MASK_SLICE == slice(68, 83)
    assert PROBE.CRITIC_TERRAIN_NORMAL_SLICE == slice(83, 86)
    assert PROBE.CRITIC_BASE_HEIGHT_INDEX == 86


def test_camera_observation_checks_accept_prone_hits_and_supine_no_hits() -> None:
    load = torch.zeros((8, 4))
    ranges = torch.ones((8, 15))
    masks = torch.zeros((8, 15))
    masks[[0, 4], 7] = 1.0
    ranges[[0, 4], 7] = 0.2

    result = PROBE.camera_observation_checks(load, ranges, masks)

    assert result["checks"] == {
        "actor_foot_load_finite_nonnegative": True,
        "body_range_finite_unit_interval": True,
        "body_range_mask_binary": True,
        "body_range_no_hit_is_one": True,
        "prone_has_at_least_one_camera_hit_both_modes": True,
        "supine_has_zero_camera_hits_both_modes": True,
    }


@pytest.mark.parametrize("side_mask_value", [0.0, 1.0])
def test_camera_observation_checks_only_observe_side_pose_hits(
    side_mask_value: float,
) -> None:
    load = torch.zeros((8, 4))
    ranges = torch.ones((8, 15))
    masks = torch.zeros((8, 15))
    masks[[0, 4], 0] = 1.0
    ranges[[0, 4], 0] = 0.2
    masks[[2, 3, 6, 7]] = side_mask_value
    ranges[[2, 3, 6, 7]] = 0.2 if side_mask_value else 1.0

    result = PROBE.camera_observation_checks(load, ranges, masks)

    assert all(result["checks"].values())
    assert result["hit_count_per_env"][2] == int(side_mask_value * 15)


def test_camera_observation_checks_reject_no_hit_range_below_one() -> None:
    load = torch.zeros((8, 4))
    ranges = torch.ones((8, 15))
    masks = torch.zeros((8, 15))
    masks[[0, 4], 0] = 1.0
    ranges[[0, 4], 0] = 0.2
    ranges[1, 3] = 0.9

    result = PROBE.camera_observation_checks(load, ranges, masks)

    assert result["checks"]["body_range_no_hit_is_one"] is False


def test_camera_config_checks_require_body_fixed_five_by_three_geometry() -> None:
    offset = type(
        "Offset",
        (),
        {
            "pos": (0.0, 0.0, -0.05),
            "rot": (2**-0.5, 0.0, 2**-0.5, 0.0),
            "convention": "world",
        },
    )()
    pattern = type(
        "Pattern",
        (),
        {
            "width": 5,
            "height": 3,
            "focal_length": 24.0,
            "horizontal_aperture": 20.955,
        },
    )()
    cfg = type(
        "CameraCfg",
        (),
        {
            "prim_path": "{ENV_REGEX_NS}/Robot/base",
            "mesh_prim_paths": ["/World/ground"],
            "offset": offset,
            "data_types": ["distance_to_camera"],
            "max_distance": 1.0,
            "pattern_cfg": pattern,
        },
    )()

    result = PROBE.camera_config_readback(cfg)

    assert all(result["checks"].values())


def test_contact_gate_allows_each_pose_to_use_the_physically_available_body_contact() -> None:
    result = PROBE.contact_exercise_checks(
        torch.tensor([True, False, True, True]),
        torch.tensor([False, True, True, False]),
    )

    assert result == {
        "at_least_one_contact_type_exercised_per_pose_mode": True,
        "foot_contact_exercised_globally": True,
        "nonfoot_contact_exercised_globally": True,
    }


def test_contact_gate_fails_when_one_pose_has_no_contact_signal() -> None:
    result = PROBE.contact_exercise_checks(
        torch.tensor([True, False]),
        torch.tensor([False, False]),
    )

    assert result["at_least_one_contact_type_exercised_per_pose_mode"] is False


def test_inverse_mapping_round_trips_joint_targets_inside_soft_limits() -> None:
    limits = torch.tensor([[[-2.0, 2.0], [-3.0, 1.0]]])
    target = torch.tensor([[0.8, -1.8]])
    action_scale = 0.8

    normalized = PROBE._inverse_map_position_targets(
        target, limits, action_scale=action_scale
    )
    scaled = normalized * action_scale
    reconstructed = limits[..., 0] + (scaled + 1.0) * 0.5 * (
        limits[..., 1] - limits[..., 0]
    )

    torch.testing.assert_close(reconstructed, target)
    assert bool((normalized.abs() <= 1.0).all())


@pytest.mark.parametrize("action_scale", [0.0, -0.1, 1.01, float("inf")])
def test_inverse_mapping_rejects_invalid_action_scale(action_scale: float) -> None:
    limits = torch.tensor([[[-2.0, 2.0]]])
    target = torch.tensor([[0.0]])

    with pytest.raises(ValueError, match="action_scale"):
        PROBE._inverse_map_position_targets(
            target, limits, action_scale=action_scale
        )


def test_actuator_limit_readback_uses_explicit_motor_limits_in_joint_order() -> None:
    actuator = type(
        "Actuator",
        (),
        {
            "joint_indices": slice(None),
            "num_joints": 3,
            "effort_limit": torch.tensor([[23.5, 23.5, 23.5], [23.5, 23.5, 23.5]]),
        },
    )()
    robot = type(
        "Robot",
        (),
        {
            "num_instances": 2,
            "num_joints": 3,
            "device": "cpu",
            "actuators": {"legs": actuator},
        },
    )()

    result = PROBE._actuator_joint_limits(robot, "effort_limit", torch)

    torch.testing.assert_close(result, torch.full((2, 3), 23.5))


def test_contact_report_separation_maps_robot_ground_points_to_environments() -> None:
    header_type = type(
        "Header",
        (),
        {
            "actor0": 1,
            "actor1": 2,
            "collider0": 3,
            "collider1": 4,
            "contact_data_offset": 0,
            "num_contact_data": 2,
        },
    )
    second_header = type(
        "Header",
        (),
        {
            "actor0": 5,
            "actor1": 2,
            "collider0": 6,
            "collider1": 4,
            "contact_data_offset": 2,
            "num_contact_data": 1,
        },
    )
    paths = {
        1: "/World/envs/env_0/Robot/base",
        2: "/World/ground",
        3: "/World/envs/env_0/Robot/base/collisions",
        4: "/World/ground/collision",
        5: "/World/envs/env_1/Robot/RR_calf",
        6: "/World/envs/env_1/Robot/RR_calf/collisions",
    }
    data = [
        type("Contact", (), {"separation": -0.002})(),
        type("Contact", (), {"separation": 0.001})(),
        type("Contact", (), {"separation": -0.007})(),
    ]

    result = PROBE.contact_report_separations(
        [header_type(), second_header()],
        data,
        num_envs=2,
        int_to_path=paths.__getitem__,
    )

    assert result == {
        "minimum_separation_m": [-0.002, -0.007],
        "minimum_separation_provenance": [
            {
                "separation_m": -0.002,
                "actor0_path": "/World/envs/env_0/Robot/base",
                "actor1_path": "/World/ground",
                "collider0_path": "/World/envs/env_0/Robot/base/collisions",
                "collider1_path": "/World/ground/collision",
            },
            {
                "separation_m": -0.007,
                "actor0_path": "/World/envs/env_1/Robot/RR_calf",
                "actor1_path": "/World/ground",
                "collider0_path": "/World/envs/env_1/Robot/RR_calf/collisions",
                "collider1_path": "/World/ground/collision",
            },
        ],
        "contact_point_count": [2, 1],
        "header_count": 2,
        "robot_ground_header_count": 2,
    }


def test_contact_report_separation_ignores_non_ground_contacts() -> None:
    header = type(
        "Header",
        (),
        {
            "actor0": 1,
            "actor1": 2,
            "collider0": 1,
            "collider1": 2,
            "contact_data_offset": 0,
            "num_contact_data": 1,
        },
    )()
    paths = {
        1: "/World/envs/env_0/Robot/base",
        2: "/World/envs/env_0/Robot/FL_thigh",
    }

    result = PROBE.contact_report_separations(
        [header],
        [type("Contact", (), {"separation": -0.2})()],
        num_envs=1,
        int_to_path=paths.__getitem__,
    )

    assert result["minimum_separation_m"] == [float("inf")]
    assert result["minimum_separation_provenance"] == [None]
    assert result["contact_point_count"] == [0]
    assert result["robot_ground_header_count"] == 0


def test_contact_report_separation_rejects_non_finite_ground_points() -> None:
    header = type(
        "Header",
        (),
        {
            "actor0": 1,
            "actor1": 2,
            "collider0": 1,
            "collider1": 2,
            "contact_data_offset": 0,
            "num_contact_data": 2,
        },
    )()
    paths = {
        1: "/World/envs/env_0/Robot/base",
        2: "/World/ground",
    }

    result = PROBE.contact_report_separations(
        [header],
        [
            type("Contact", (), {"separation": float("nan")})(),
            type("Contact", (), {"separation": float("inf")})(),
        ],
        num_envs=1,
        int_to_path=paths.__getitem__,
    )

    assert result["minimum_separation_m"] == [float("inf")]
    assert result["minimum_separation_provenance"] == [None]
    assert result["contact_point_count"] == [0]
    assert result["robot_ground_header_count"] == 1


def test_contact_report_accumulator_captures_each_physics_step_before_buffer_expires() -> None:
    paths = {
        1: "/World/envs/env_0/Robot/base",
        2: "/World/ground",
    }
    accumulator = PROBE.ContactReportAccumulator(
        1, paths.__getitem__, physics_dt_s=0.005
    )
    header = type(
        "Header",
        (),
        {
            "actor0": 1,
            "actor1": 2,
            "collider0": 1,
            "collider1": 2,
            "contact_data_offset": 0,
            "num_contact_data": 1,
        },
    )()

    accumulator([header], [type("Contact", (), {"separation": -0.003})()])
    accumulator([header], [type("Contact", (), {"separation": -0.008})()])

    assert accumulator.snapshot() == {
        "available": True,
        "error": None,
        "event_count": 2,
        "minimum_separation_m": [-0.008],
        "minimum_separation_provenance": [
            {
                "separation_m": -0.008,
                "actor0_path": "/World/envs/env_0/Robot/base",
                "actor1_path": "/World/ground",
                "collider0_path": "/World/envs/env_0/Robot/base",
                "collider1_path": "/World/ground",
                "physics_step": 2,
                "time_s": 0.01,
            }
        ],
        "contact_point_count": [2],
        "header_count": 2,
        "robot_ground_header_count": 2,
    }


def test_contact_report_accumulator_reset_excludes_pre_rollout_events() -> None:
    accumulator = PROBE.ContactReportAccumulator(1, lambda _: "/unmatched")
    accumulator([], [])

    accumulator.reset()

    assert accumulator.snapshot()["event_count"] == 0


def test_contact_report_accumulator_marks_path_conversion_failure_unavailable() -> None:
    def fail_path_conversion(_: int) -> str:
        raise RuntimeError("expired contact token")

    accumulator = PROBE.ContactReportAccumulator(1, fail_path_conversion)
    header = type(
        "Header",
        (),
        {
            "actor0": 1,
            "actor1": 2,
            "collider0": 1,
            "collider1": 2,
            "contact_data_offset": 0,
            "num_contact_data": 0,
        },
    )()

    accumulator([header], [])

    assert accumulator.snapshot()["available"] is False
    assert "expired contact token" in accumulator.snapshot()["error"]


def test_hard_joint_limit_gate_allows_violation_inside_contract_margin() -> None:
    result = PROBE.within_hard_joint_limit_margin(
        torch.tensor([0.006457, 0.010001]), 0.01
    )

    assert result.tolist() == [True, False]


def test_gpu_separation_crosscheck_requires_external_cpu_authority() -> None:
    result = PROBE.separation_crosscheck_status(
        device="cuda:0",
        data_available=False,
        threshold_passed=False,
    )

    assert result["status"] == "requires_cpu_crosscheck"
    assert result["this_run_is_authority"] is False
    assert result["passed"] is None


def test_cpu_separation_crosscheck_preserves_failed_exact_threshold() -> None:
    result = PROBE.separation_crosscheck_status(
        device="cpu",
        data_available=True,
        threshold_passed=False,
    )

    assert result["status"] == "observed"
    assert result["this_run_is_authority"] is True
    assert result["passed"] is False


def test_runtime_failure_does_not_claim_policy_qualification() -> None:
    status = PROBE.summarize_status(
        {"finite": True, "contact_separation_available": False},
        ("finite",),
    )

    assert status["run_health"]["passed"] is True
    assert status["runtime_contract"]["passed"] is False
    assert status["qualification"] == {
        "status": "not_run",
        "passed": None,
        "reason": "runtime calibration does not evaluate a learned checkpoint",
    }
    assert status["passed"] is False


def test_all_runtime_checks_can_pass_without_becoming_policy_qualification() -> None:
    status = PROBE.summarize_status({"finite": True, "physics": True}, ("finite",))

    assert status["run_health"]["passed"] is True
    assert status["runtime_contract"]["passed"] is True
    assert status["qualification"]["passed"] is None
    assert status["passed_semantics"] == "runtime_contract_only_not_policy_qualification"
