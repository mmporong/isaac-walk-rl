from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaac_walk_g009.recover_contracts import (
    ACTION_EMA_ALPHA,
    ACTION_SCALE,
    ACTOR_OBSERVATION_DIM,
    CONTROL_DT_S,
    CRITIC_OBSERVATION_DIM,
    EFFECTIVE_ACTION_TARGET_HARD_LIMIT_RANGE_FRACTION,
    GO2_SOFT_JOINT_LIMIT_FACTOR,
    MAX_BASE_HEIGHT_M,
    MAX_LINEAR_SPEED_M_S,
    MIN_TOTAL_FOOT_SUPPORT_RATIO,
    POSE_CURRICULUM_PHASE_END_CONTROL_STEPS,
    POSE_CURRICULUM_PROBABILITIES,
    PPO_INIT_NOISE_STD,
    PPO_GAMMA,
    RECOVER_POSES,
    R0_REWARD_TERMS,
    SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
    canonical_sha256,
    recover_contract,
)


MANIFEST = ROOT / "configs" / "g009_r0.json"


def _rotate_body_up_wxyz(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    return (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )


def test_four_curated_poses_are_uniform_and_source_class_matches() -> None:
    assert tuple(RECOVER_POSES) == ("prone", "supine", "left_side", "right_side")
    assert sum(pose.probability for pose in RECOVER_POSES.values()) == pytest.approx(1.0)

    for index, pose in enumerate(RECOVER_POSES.values()):
        assert pose.probability == pytest.approx(0.25)
        assert pose.source_class_one_hot == tuple(float(i == index) for i in range(4))
        assert pose.root_height_m > 0.0
        assert math.sqrt(sum(value * value for value in pose.root_quaternion_wxyz)) == pytest.approx(1.0)
        assert _rotate_body_up_wxyz(pose.root_quaternion_wxyz) == pytest.approx(
            pose.expected_body_up, abs=1.0e-12
        )
    assert [pose.root_height_m for pose in RECOVER_POSES.values()] == [0.165, 0.06, 0.163, 0.163]


def test_actor_is_deployable_83_and_critic_adds_only_24_privileged_values() -> None:
    contract = recover_contract()
    actor = contract["observations"]["actor"]
    critic = contract["observations"]["critic"]

    assert [item["dim"] for item in actor["terms"]] == [3, 3, 3, 12, 12, 12, 4, 4, 15, 15]
    assert ACTOR_OBSERVATION_DIM == actor["dim"] == 83
    assert actor["deployability_status"] == "conditional_adapter_required"
    assert len(actor["required_hardware_adapters"]) == 4
    assert all(item["privileged"] is False for item in actor["terms"])
    assert all(
        item["invalid_policy"] == "nonfinite_signal_to_numeric_invalid_termination"
        for item in actor["terms"][:-2]
    )
    assert all(
        item["invalid_policy"].startswith("no_hit_or_out_of_range_to_one")
        for item in actor["terms"][-2:]
    )

    assert critic["actor_prefix_dim"] == 83
    assert [item["dim"] for item in critic["privileged_terms"]] == [3, 1, 8, 3, 1, 3, 1, 4]
    assert sum(item["dim"] for item in critic["privileged_terms"]) == 24
    assert CRITIC_OBSERVATION_DIM == critic["dim"] == 107
    assert critic["terms"][: len(actor["terms"])] == actor["terms"]
    assert all(item["privileged"] is True for item in critic["privileged_terms"])


def test_r0_reward_and_ppo_contract_match_the_frozen_document_revision() -> None:
    assert [(term.name, term.weight) for term in R0_REWARD_TERMS] == [
        ("upright_progress", 2.0),
        ("gated_base_height_progress", 2.0),
        ("soft_stand_progress", 2.0),
        ("stable_support", 0.5),
        ("upright_hold", 5.0),
        ("stable_success_once", 10.0),
        ("gated_angvel_l2", -0.05),
        ("joint_limit", -2.0),
        ("torque_l2", -0.0002),
        ("joint_acc_l2", -2.5e-7),
        ("gated_action_rate_l2", -0.01),
        ("mechanical_power_proxy", -1.0e-5),
        ("undesired_collision", -1.0),
    ]

    reward = recover_contract()["reward"]
    assert reward["aggregation"] == "sum_i(weight_i * raw_rate_i * control_dt_s)"
    assert [term["temporal_semantics"] for term in reward["terms"][:6]] == [
        "discounted_potential_difference",
        "discounted_potential_difference",
        "discounted_potential_difference",
        "continuous_rate",
        "continuous_rate",
        "terminal_impulse",
    ]
    assert reward["potential_shaping"]["soft_stand"]["formula"] == "u*z*(0.5*c+0.5*l)"
    assert reward["potential_discount_factor"] == PPO_GAMMA == 0.99
    assert "terminal transitions" in reward["potential_difference_definition"]
    assert reward["state_gated_regularization"]["gated_angvel_l2"][
        "fallen_effective_weight"
    ] == -0.005
    assert reward["state_gated_regularization"]["gated_action_rate_l2"][
        "fallen_effective_weight"
    ] == -0.002

    ppo = recover_contract()["ppo"]
    assert ppo == {
        "algorithm": "RSL-RL PPO",
        "rsl_rl_version": "2.3.3",
        "rollout_horizon_steps_per_env": 24,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "init_noise_std": PPO_INIT_NOISE_STD,
        "learning_epochs": 5,
        "mini_batches": 4,
        "optimizer_updates_per_iteration": 20,
        "clip_parameter": 0.2,
        "entropy_coefficient": 0.01,
        "gamma": PPO_GAMMA,
        "gae_lambda": 0.95,
        "learning_rate": 0.001,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "max_gradient_norm": 1.0,
        "empirical_normalization": False,
        "scratch_training": True,
        "qualification_num_envs": 1024,
        "qualification_iterations": 300,
        "qualification_seed": 42,
    }


def test_runtime_dynamics_and_success_gate_are_hash_bound() -> None:
    contract = recover_contract()
    assert contract["contract_id"] == "g009_r0_recover_rev10"
    assert contract["timing"] == {
        "physics_dt_s": 0.005,
        "control_decimation": 4,
        "control_dt_s": CONTROL_DT_S,
        "episode_length_s": 8.0,
        "max_episode_steps": 400,
    }
    assert contract["success_gate"]["base_height_m"] == [0.30, MAX_BASE_HEIGHT_M]
    assert contract["success_gate"]["base_linear_speed_m_s_max"] == MAX_LINEAR_SPEED_M_S
    assert contract["success_gate"]["foot_contact_definition"] == (
        "positive contact-force projection onto the true support normal"
    )
    assert contract["success_gate"]["total_foot_normal_support_ratio_min"] == 0.60
    assert MIN_TOTAL_FOOT_SUPPORT_RATIO == 0.60
    assert contract["success_gate"]["non_foot_contacts_max"] == 0
    assert contract["success_gate"]["non_foot_contact_force_threshold_n"] == 1.0
    assert "tangential force or body/leg bracing" in contract["success_gate"]["mechanics_rationale"]
    assert contract["success_gate"]["stable_dwell_steps"] == 25
    assert contract["success_gate"]["stable_dwell_s"] == pytest.approx(0.5)
    assert contract["reset"]["folded_joint_angles_rad"] == {
        "left_hip": 0.1,
        "right_hip": -0.1,
        "thigh": 1.5,
        "calf": -2.4,
    }
    assert contract["materials"]["effective_foot_ground_friction"]["missing_value_policy"] == "fail_closed"
    assert contract["action"]["normalized_clip"] == [-1.0, 1.0]
    assert contract["action"]["type"] == "EMAJointPositionToLimitsAction"
    assert contract["action"]["scale"] == ACTION_SCALE == 0.70
    assert contract["action"]["ema_alpha"] == ACTION_EMA_ALPHA == 0.2
    assert contract["action"]["ema_control_frequency_hz"] == 50.0
    assert contract["action"]["ema_time_constant_s"] == pytest.approx(0.08963, rel=1.0e-4)
    assert contract["action"]["asset_soft_joint_limit_factor"] == (
        GO2_SOFT_JOINT_LIMIT_FACTOR
    ) == 0.9
    assert contract["action"]["effective_target_hard_limit_range_fraction"] == (
        EFFECTIVE_ACTION_TARGET_HARD_LIMIT_RANGE_FRACTION
    ) == pytest.approx(0.63)
    assert contract["action"]["hard_limit_margin_fraction_per_side"] == pytest.approx(0.185)
    assert contract["collision_penalty"]["active_when_base_height_m_min"] == 0.30
    assert contract["observations"]["actor"]["range_sensor"]["no_hit_semantics"] == (
        "range=1.0, hit_mask=0.0, not numeric_invalid"
    )
    privileged_by_name = {
        term["name"]: term
        for term in contract["observations"]["critic"]["privileged_terms"]
    }
    for name, implementation in (
        ("commanded_wrench", "critic_zero_external_wrench"),
        ("normalized_pulse_time_remaining", "critic_zero_disturbance_pulse"),
    ):
        assert privileged_by_name[name]["r0_runtime_state"] == "constant_zero_inactive"
        assert privileged_by_name[name]["activation_stage"] == "D1_external_wrench"
        assert privileged_by_name[name]["implementation"] == implementation
    assert contract["termination"]["solver_joint_limit_tolerance_rad"] == 0.01
    assert SOLVER_JOINT_LIMIT_TOLERANCE_RAD == 0.01
    calibration = contract["termination"]["solver_joint_limit_tolerance_calibration"]
    assert calibration["gpu_reset_hold_max_raw_excess_rad"] == 0.007703
    assert calibration["cpu_reset_hold_max_raw_excess_rad"] == 0.003262
    curriculum = contract["reset"]["pose_curriculum"]
    assert tuple(curriculum["phase_end_control_steps_exclusive"]) == (
        POSE_CURRICULUM_PHASE_END_CONTROL_STEPS
    ) == (1201, 2401)
    assert [tuple(phase["pose_probabilities"].values()) for phase in curriculum["phases"]] == list(
        POSE_CURRICULUM_PROBABILITIES
    )
    assert curriculum["actor_observation_exposure"] is False


def test_json_manifest_is_canonically_bound_to_the_import_light_code_contract() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = recover_contract()

    assert manifest["contract"] == contract
    assert manifest["contract_sha256"] == canonical_sha256(contract)
    assert len(manifest["contract_sha256"]) == 64
