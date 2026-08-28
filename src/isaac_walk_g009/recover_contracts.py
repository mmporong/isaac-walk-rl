"""Import-light contracts for the G009 R0 recovery policy.

This module deliberately depends only on the Python standard library.  Isaac
Lab adapters may consume these values, but importing the contract must never
initialize Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any


@dataclass(frozen=True)
class RecoverPose:
    """One curated R0 reset pose and its critic-only source label."""

    name: str
    probability: float
    root_height_m: float
    root_quaternion_wxyz: tuple[float, float, float, float]
    expected_body_up: tuple[float, float, float]
    source_class_one_hot: tuple[float, float, float, float]


@dataclass(frozen=True)
class RewardTerm:
    """One signed term in the frozen R0 reward revision."""

    name: str
    weight: float
    temporal_semantics: str


_SQRT_HALF = math.sqrt(0.5)

PHYSICS_DT_S = 0.005
CONTROL_DECIMATION = 4
CONTROL_DT_S = PHYSICS_DT_S * CONTROL_DECIMATION
EPISODE_LENGTH_S = 8.0

ACTION_SCALE = 0.70
ACTION_EMA_ALPHA = 0.2
GO2_SOFT_JOINT_LIMIT_FACTOR = 0.9
EFFECTIVE_ACTION_TARGET_HARD_LIMIT_RANGE_FRACTION = (
    ACTION_SCALE * GO2_SOFT_JOINT_LIMIT_FACTOR
)
PPO_INIT_NOISE_STD = 0.5
PPO_GAMMA = 0.99

RESET_POSE_XY_RANGE_M = (-0.05, 0.05)
RESET_YAW_RANGE_RAD = (-math.pi, math.pi)
FOLDED_JOINT_ANGLES_RAD = {
    "left_hip": 0.1,
    "right_hip": -0.1,
    "thigh": 1.5,
    "calf": -2.37,
}

UPRIGHT_ANGLE_DEG = 20.0
UPRIGHT_COS_THRESHOLD = math.cos(math.radians(UPRIGHT_ANGLE_DEG))
MIN_BASE_HEIGHT_M = 0.30
MAX_BASE_HEIGHT_M = 0.60
MAX_LINEAR_SPEED_M_S = 0.50
MAX_ANGULAR_SPEED_RAD_S = 1.0
MIN_FOOT_CONTACTS = 3
CONTACT_FORCE_THRESHOLD_N = 1.0
MIN_TOTAL_FOOT_SUPPORT_RATIO = 0.60
STABLE_DWELL_STEPS = 25

HEIGHT_SHAPING_MIN_M = 0.06
HEIGHT_ORIENTATION_GATE_START = 0.0
HEIGHT_ORIENTATION_GATE_FULL = 0.8
REGULARIZATION_ORIENTATION_GATE_START = 0.50
REGULARIZATION_ORIENTATION_GATE_FULL = UPRIGHT_COS_THRESHOLD
REGULARIZATION_HEIGHT_GATE_START_M = 0.20
REGULARIZATION_HEIGHT_GATE_FULL_M = MIN_BASE_HEIGHT_M
FALLEN_ANGVEL_PENALTY_MULTIPLIER = 0.10
FALLEN_ACTION_RATE_PENALTY_MULTIPLIER = 0.20

POSE_CURRICULUM_PHASE_END_CONTROL_STEPS = (1201, 2401)
POSE_CURRICULUM_PROBABILITIES = (
    (1.0, 0.0, 0.0, 0.0),
    (0.50, 0.0, 0.25, 0.25),
    (0.25, 0.25, 0.25, 0.25),
)

NOMINAL_TOTAL_MASS_KG = 15.019
GROUND_STATIC_FRICTION = 0.8
GROUND_DYNAMIC_FRICTION = 0.6
FOOT_STATIC_FRICTION = 1.0
FOOT_DYNAMIC_FRICTION = 1.0
SOLVER_JOINT_LIMIT_TOLERANCE_RAD = 0.01
NON_FOOT_COLLISION_THRESHOLD_N = 1.0
GRAVITY_MAGNITUDE_M_S2 = 9.81

RECOVER_POSES: dict[str, RecoverPose] = {
    "prone": RecoverPose(
        name="prone",
        probability=0.25,
        root_height_m=0.165,
        root_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        expected_body_up=(0.0, 0.0, 1.0),
        source_class_one_hot=(1.0, 0.0, 0.0, 0.0),
    ),
    "supine": RecoverPose(
        name="supine",
        probability=0.25,
        root_height_m=0.06,
        root_quaternion_wxyz=(0.0, 1.0, 0.0, 0.0),
        expected_body_up=(0.0, 0.0, -1.0),
        source_class_one_hot=(0.0, 1.0, 0.0, 0.0),
    ),
    "left_side": RecoverPose(
        name="left_side",
        probability=0.25,
        root_height_m=0.163,
        root_quaternion_wxyz=(_SQRT_HALF, _SQRT_HALF, 0.0, 0.0),
        expected_body_up=(0.0, -1.0, 0.0),
        source_class_one_hot=(0.0, 0.0, 1.0, 0.0),
    ),
    "right_side": RecoverPose(
        name="right_side",
        probability=0.25,
        root_height_m=0.163,
        root_quaternion_wxyz=(_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0),
        expected_body_up=(0.0, 1.0, 0.0),
        source_class_one_hot=(0.0, 0.0, 0.0, 1.0),
    ),
}


ACTOR_OBSERVATION_TERMS = (
    ("base_linear_velocity", 3),
    ("base_angular_velocity", 3),
    ("projected_gravity", 3),
    ("relative_joint_position", 12),
    ("relative_joint_velocity", 12),
    ("last_recover_action", 12),
    ("four_foot_contact_state", 4),
    ("normalized_four_foot_load", 4),
    ("body_fixed_range", 15),
    ("body_fixed_range_hit_mask", 15),
)

PRIVILEGED_OBSERVATION_TERMS = (
    ("terrain_normal_gt", 3),
    ("base_height_gt", 1),
    ("four_foot_effective_static_dynamic_friction", 8),
    ("whole_body_com_base", 3),
    ("total_mass", 1),
    ("commanded_wrench", 3),
    ("normalized_pulse_time_remaining", 1),
    ("source_fall_class_one_hot", 4),
)

PRIVILEGED_OBSERVATION_METADATA: dict[str, dict[str, Any]] = {
    "commanded_wrench": {
        "r0_runtime_state": "constant_zero_inactive",
        "activation_stage": "D1_external_wrench",
        "implementation": "critic_zero_external_wrench",
        "rationale": "reserved critic-only channel; R0 has no external-wrench event",
    },
    "normalized_pulse_time_remaining": {
        "r0_runtime_state": "constant_zero_inactive",
        "activation_stage": "D1_external_wrench",
        "implementation": "critic_zero_disturbance_pulse",
        "rationale": "reserved critic-only channel; R0 has no disturbance pulse",
    },
}

ACTOR_OBSERVATION_METADATA: dict[str, dict[str, Any]] = {
    "base_linear_velocity": {
        "sim_source": "articulation base-frame velocity",
        "hardware_source": "base state estimator",
        "noise_uniform": [-0.1, 0.1],
    },
    "base_angular_velocity": {
        "sim_source": "articulation base-frame angular velocity",
        "hardware_source": "IMU gyroscope",
        "noise_uniform": [-0.2, 0.2],
    },
    "projected_gravity": {
        "sim_source": "articulation projected gravity",
        "hardware_source": "IMU attitude estimator",
        "noise_uniform": [-0.05, 0.05],
    },
    "relative_joint_position": {
        "sim_source": "articulation joint position",
        "hardware_source": "joint encoders",
        "noise_uniform": [-0.01, 0.01],
    },
    "relative_joint_velocity": {
        "sim_source": "articulation joint velocity",
        "hardware_source": "filtered joint encoder derivative",
        "noise_uniform": [-1.5, 1.5],
    },
    "last_recover_action": {
        "sim_source": "action manager previous command",
        "hardware_source": "controller command buffer",
        "noise_uniform": None,
    },
    "four_foot_contact_state": {
        "sim_source": "foot ContactSensor threshold",
        "hardware_source": "foot contact/load estimator",
        "noise_uniform": None,
    },
    "normalized_four_foot_load": {
        "sim_source": "foot ContactSensor force magnitude",
        "hardware_source": "foot load or actuator-torque force estimator",
        "noise_uniform": [-0.02, 0.02],
    },
    "body_fixed_range": {
        "sim_source": "body-fixed RayCasterCamera distance_to_camera",
        "hardware_source": "body-mounted downward range/depth camera",
        "noise_uniform": None,
    },
    "body_fixed_range_hit_mask": {
        "sim_source": "finite in-range distance_to_camera mask",
        "hardware_source": "range/depth camera validity mask",
        "noise_uniform": None,
    },
}

ACTOR_OBSERVATION_DIM = sum(dim for _, dim in ACTOR_OBSERVATION_TERMS)
PRIVILEGED_OBSERVATION_DIM = sum(dim for _, dim in PRIVILEGED_OBSERVATION_TERMS)
CRITIC_OBSERVATION_DIM = ACTOR_OBSERVATION_DIM + PRIVILEGED_OBSERVATION_DIM

R0_REWARD_TERMS = (
    RewardTerm("upright_progress", 2.0, "discounted_potential_difference"),
    RewardTerm("gated_base_height_progress", 2.0, "discounted_potential_difference"),
    RewardTerm("soft_stand_progress", 2.0, "discounted_potential_difference"),
    RewardTerm("stable_support", 0.5, "continuous_rate"),
    RewardTerm("upright_hold", 5.0, "continuous_rate"),
    RewardTerm("stable_success_once", 10.0, "terminal_impulse"),
    RewardTerm("gated_angvel_l2", -0.05, "continuous_rate"),
    RewardTerm("joint_limit", -2.0, "continuous_rate"),
    RewardTerm("torque_l2", -0.0002, "continuous_rate"),
    RewardTerm("joint_acc_l2", -2.5e-7, "continuous_rate"),
    RewardTerm("gated_action_rate_l2", -0.01, "continuous_rate"),
    RewardTerm("mechanical_power_proxy", -1.0e-5, "continuous_rate"),
    RewardTerm("undesired_collision", -1.0, "continuous_rate"),
)


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with the repository canonical encoding."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _observation_term(name: str, dim: int, *, privileged: bool) -> dict[str, Any]:
    term = {"name": name, "dim": dim, "privileged": privileged}
    if privileged:
        term.update(PRIVILEGED_OBSERVATION_METADATA.get(name, {}))
    else:
        term.update(ACTOR_OBSERVATION_METADATA[name])
        if name in {"body_fixed_range", "body_fixed_range_hit_mask"}:
            term["invalid_policy"] = (
                "no_hit_or_out_of_range_to_one_without_termination; "
                "negative_finite_or_nan_to_numeric_invalid"
            )
        else:
            term["invalid_policy"] = "nonfinite_signal_to_numeric_invalid_termination"
    return term


def recover_contract() -> dict[str, Any]:
    """Return the canonical, JSON-compatible G009 R0 contract."""

    actor_terms = [
        _observation_term(name, dim, privileged=False) for name, dim in ACTOR_OBSERVATION_TERMS
    ]
    privileged_terms = [
        _observation_term(name, dim, privileged=True)
        for name, dim in PRIVILEGED_OBSERVATION_TERMS
    ]
    poses = [
        {
            "name": pose.name,
            "probability": pose.probability,
            "root_height_m": pose.root_height_m,
            "root_quaternion_wxyz": list(pose.root_quaternion_wxyz),
            "expected_body_up": list(pose.expected_body_up),
            "source_class_one_hot": list(pose.source_class_one_hot),
        }
        for pose in RECOVER_POSES.values()
    ]
    return {
        "contract_id": "g009_r0_recover_rev11",
        "stage_id": "R0",
        "policy_schema": "P-RECOVER-83/C-RECOVER-107",
        "poses": poses,
        "timing": {
            "physics_dt_s": PHYSICS_DT_S,
            "control_decimation": CONTROL_DECIMATION,
            "control_dt_s": CONTROL_DT_S,
            "episode_length_s": EPISODE_LENGTH_S,
            "max_episode_steps": int(round(EPISODE_LENGTH_S / CONTROL_DT_S)),
        },
        "reset": {
            "assignment_mode": "random",
            "pose_xy_range_m": list(RESET_POSE_XY_RANGE_M),
            "yaw_range_rad": list(RESET_YAW_RANGE_RAD),
            "root_and_joint_state_written_atomically": True,
            "root_velocity_m_s_and_rad_s": [0.0] * 6,
            "folded_joint_angles_rad": FOLDED_JOINT_ANGLES_RAD,
            "action_envelope_requirement": (
                "every folded reset joint target must be reachable without normalized-action saturation"
            ),
            "pose_curriculum": {
                "clock": "env.common_step_counter",
                "control_steps_per_ppo_iteration": 24,
                "phase_end_control_steps_exclusive": list(
                    POSE_CURRICULUM_PHASE_END_CONTROL_STEPS
                ),
                "phases": [
                    {
                        "ppo_iterations": [0, 49],
                        "pose_probabilities": dict(zip(RECOVER_POSES, POSE_CURRICULUM_PROBABILITIES[0])),
                    },
                    {
                        "ppo_iterations": [50, 99],
                        "pose_probabilities": dict(zip(RECOVER_POSES, POSE_CURRICULUM_PROBABILITIES[1])),
                    },
                    {
                        "ppo_iterations": [100, 299],
                        "pose_probabilities": dict(zip(RECOVER_POSES, POSE_CURRICULUM_PROBABILITIES[2])),
                    },
                ],
                "evaluation_override": "stratified_equal_pose_assignment",
                "actor_observation_exposure": False,
            },
        },
        "action": {
            "type": "EMAJointPositionToLimitsAction",
            "dim": 12,
            "scale": ACTION_SCALE,
            "ema_alpha": ACTION_EMA_ALPHA,
            "ema_control_frequency_hz": 1.0 / CONTROL_DT_S,
            "ema_time_constant_s": -CONTROL_DT_S / math.log(1.0 - ACTION_EMA_ALPHA),
            "normalized_clip": [-1.0, 1.0],
            "rescale_to_soft_joint_limits": True,
            "runner_clip_actions": 1.0,
            "asset_soft_joint_limit_factor": GO2_SOFT_JOINT_LIMIT_FACTOR,
            "effective_target_hard_limit_range_fraction": (
                EFFECTIVE_ACTION_TARGET_HARD_LIMIT_RANGE_FRACTION
            ),
            "hard_limit_margin_fraction_per_side": (
                1.0 - EFFECTIVE_ACTION_TARGET_HARD_LIMIT_RANGE_FRACTION
            )
            / 2.0,
            "safety_rationale": (
                "seed-42 rev3 smoke terminated 75% of completed episodes at the hard joint limit; "
                "scale-only rev4 still terminated 62.5%; scale 0.8 plus alpha 0.2 removed it in the "
                "64-env rev5 smoke, but rev6 still measured 0.0416667 in the 1024-env stress smoke; "
                "reducing PPO initial noise to 0.5 produced zero in the rev7 1024-env stress smoke; "
                "rev9 prone pilot returned hard-limit terminations in 23 of 50 iterations with a "
                "maximum rate of 0.4583333, so rev10 reduced only the action scale to 0.70; "
                "rev10 then reproduced a 16.066175 body-weight CPU non-foot contact peak while the "
                "-2.40 rad calf reset independently saturated the normalized hold action and shifted "
                "its reachable target by 0.026014 rad; rev11 tests this mechanical hypothesis by moving "
                "only that reset seed to -2.37 rad inside the scale-0.70 action envelope while retaining "
                "EMA alpha 0.2, PPO initial noise 0.5, and the 0.01 rad solver tolerance; "
                "stochastic-training terminations remain diagnostic and the official deterministic "
                "evaluation still requires zero; rev9 numeric-invalid terminations remained zero "
                "throughout the pilot"
            ),
        },
        "materials": {
            "ground": {
                "static_friction": GROUND_STATIC_FRICTION,
                "dynamic_friction": GROUND_DYNAMIC_FRICTION,
                "friction_combine_mode": "multiply",
            },
            "feet": {
                "static_friction": FOOT_STATIC_FRICTION,
                "dynamic_friction": FOOT_DYNAMIC_FRICTION,
                "restitution": 0.0,
            },
            "effective_foot_ground_friction": {
                "static": GROUND_STATIC_FRICTION * FOOT_STATIC_FRICTION,
                "dynamic": GROUND_DYNAMIC_FRICTION * FOOT_DYNAMIC_FRICTION,
                "source": "startup PhysX foot-material readback multiplied by the ground material",
                "missing_value_policy": "fail_closed",
            },
        },
        "observations": {
            "actor": {
                "schema_id": "P-RECOVER-83",
                "dim": ACTOR_OBSERVATION_DIM,
                "deployability_status": "conditional_adapter_required",
                "required_hardware_adapters": [
                    "base state estimator and IMU",
                    "joint encoder interface",
                    "foot contact/load estimator",
                    "body-mounted downward range/depth camera adapter",
                ],
                "terrain_geometry_source": "body-fixed 5x3 pinhole range image",
                "range_sensor": {
                    "type": "RayCasterCameraCfg",
                    "data_type": "distance_to_camera",
                    "ray_alignment": "base",
                    "width": 5,
                    "height": 3,
                    "max_distance_m": 1.0,
                    "depth_clipping_behavior": "none",
                    "mount_position_m": [0.0, 0.0, -0.05],
                    "mount_quaternion_wxyz": [_SQRT_HALF, 0.0, _SQRT_HALF, 0.0],
                    "mount_convention": "world",
                    "forward_axis_mapping": "camera +X to body -Z",
                    "no_hit_semantics": "range=1.0, hit_mask=0.0, not numeric_invalid",
                },
                "foot_load_denominator": {
                    "type": "frozen_nominal_mass",
                    "nominal_total_mass_kg": NOMINAL_TOTAL_MASS_KG,
                },
                "terms": actor_terms,
            },
            "critic": {
                "schema_id": "C-RECOVER-107",
                "dim": CRITIC_OBSERVATION_DIM,
                "actor_prefix_dim": ACTOR_OBSERVATION_DIM,
                "terms": actor_terms + privileged_terms,
                "privileged_terms": privileged_terms,
            },
        },
        "success_gate": {
            "upright_angle_deg_max": UPRIGHT_ANGLE_DEG,
            "upright_cos_min": UPRIGHT_COS_THRESHOLD,
            "base_height_m": [MIN_BASE_HEIGHT_M, MAX_BASE_HEIGHT_M],
            "foot_contacts_min": MIN_FOOT_CONTACTS,
            "contact_force_threshold_n": CONTACT_FORCE_THRESHOLD_N,
            "foot_contact_definition": "positive contact-force projection onto the true support normal",
            "total_foot_normal_support_ratio_min": MIN_TOTAL_FOOT_SUPPORT_RATIO,
            "nominal_total_mass_kg": NOMINAL_TOTAL_MASS_KG,
            "gravity_magnitude_m_s2": GRAVITY_MAGNITUDE_M_S2,
            "non_foot_contacts_max": 0,
            "non_foot_contact_force_threshold_n": NON_FOOT_COLLISION_THRESHOLD_N,
            "base_linear_speed_m_s_max": MAX_LINEAR_SPEED_M_S,
            "base_angular_speed_rad_s_max": MAX_ANGULAR_SPEED_RAD_S,
            "stable_dwell_steps": STABLE_DWELL_STEPS,
            "stable_dwell_s": STABLE_DWELL_STEPS * CONTROL_DT_S,
            "mechanics_rationale": "success requires gravity-supporting foot load, not tangential force or body/leg bracing; at least 60% of nominal weight must be carried by three or more feet",
        },
        "reward": {
            "revision": "R0-rev4-gated-potential-and-regularization",
            "aggregation": "sum_i(weight_i * raw_rate_i * control_dt_s)",
            "framework_semantics": "Isaac Lab RewardManager multiplies every weighted raw term by control_dt_s",
            "terms": [
                {
                    "name": term.name,
                    "weight": term.weight,
                    "temporal_semantics": term.temporal_semantics,
                }
                for term in R0_REWARD_TERMS
            ],
            "potential_difference_definition": (
                "(gamma * potential_t - potential_t-1) / control_dt_s; "
                "potential_t is zero on terminal transitions"
            ),
            "potential_discount_factor": PPO_GAMMA,
            "episode_boundary_semantics": (
                "previous potential is reset to zero, the first transition contributes "
                "gamma*potential_t, and terminal potential is forced to zero so the full "
                "discounted shaping return telescopes to zero"
            ),
            "potential_shaping": {
                "height": {
                    "weight": 2.0,
                    "normalized_height_m": [HEIGHT_SHAPING_MIN_M, MIN_BASE_HEIGHT_M],
                    "orientation_gate_cosine": [
                        HEIGHT_ORIENTATION_GATE_START,
                        HEIGHT_ORIENTATION_GATE_FULL,
                    ],
                },
                "soft_stand": {
                    "weight": 2.0,
                    "formula": "u*z*(0.5*c+0.5*l)",
                    "upright_score": "clip((alignment+1)/2,0,1)",
                    "normalized_height_m": [HEIGHT_SHAPING_MIN_M, MIN_BASE_HEIGHT_M],
                    "positive_normal_load_ratio_target": MIN_TOTAL_FOOT_SUPPORT_RATIO,
                    "contact_count_target": MIN_FOOT_CONTACTS,
                    "tangential_or_negative_load_credit": 0.0,
                },
                "strict_success_gate_is_separate": True,
                "pilot_basis": (
                    "rev7 and rev8 uniform 1024x50 pilots produced zero stable-support, hold, "
                    "or success events because the prone height potential maximum was 0.135 "
                    "before regularization"
                ),
            },
            "state_gated_regularization": {
                "stand_gate": {
                    "orientation_cosine": [
                        REGULARIZATION_ORIENTATION_GATE_START,
                        REGULARIZATION_ORIENTATION_GATE_FULL,
                    ],
                    "base_height_m": [
                        REGULARIZATION_HEIGHT_GATE_START_M,
                        REGULARIZATION_HEIGHT_GATE_FULL_M,
                    ],
                    "combination": "orientation_gate*height_gate",
                },
                "gated_angvel_l2": {
                    "configured_weight": -0.05,
                    "fallen_multiplier": FALLEN_ANGVEL_PENALTY_MULTIPLIER,
                    "fallen_effective_weight": -0.005,
                    "standing_effective_weight": -0.05,
                },
                "gated_action_rate_l2": {
                    "configured_weight": -0.01,
                    "fallen_multiplier": FALLEN_ACTION_RATE_PENALTY_MULTIPLIER,
                    "fallen_effective_weight": -0.002,
                    "standing_effective_weight": -0.01,
                },
                "unchanged_safety_terms": [
                    "joint_limit",
                    "torque_l2",
                    "joint_acc_l2",
                    "mechanical_power_proxy",
                    "undesired_collision",
                ],
            },
            "terminal_impulse_definition": "one_shot_indicator / control_dt_s",
            "stable_success_once_requires_consecutive_hold": True,
        },
        "ppo": {
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
        },
        "termination": {
            "active_terms": [
                "timeout",
                "stable_success",
                "numeric_invalid",
                "urdf_hard_joint_limit_violation",
            ],
            "base_contact_termination": False,
            "solver_joint_limit_tolerance_rad": SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
            "solver_joint_limit_tolerance_calibration": {
                "gpu_reset_hold_max_raw_excess_rad": 0.007703,
                "cpu_reset_hold_max_raw_excess_rad": 0.003262,
                "rationale": "0.01 rad exceeds measured reset-hold solver overshoot on both GPU and CPU while remaining a strict URDF-limit guard",
            },
        },
        "collision_penalty": {
            "non_foot_contact_force_threshold_n": NON_FOOT_COLLISION_THRESHOLD_N,
            "active_when_base_height_m_min": MIN_BASE_HEIGHT_M,
            "rationale": "allow required body and leg pivots while fallen; penalize renewed collision after standing height is reached",
        },
    }


class StableHoldLatch:
    """Vectorized consecutive-hold detector with a sticky one-shot latch.

    Invalid/non-finite samples are treated as unstable and clear that
    environment's in-progress counter.  A latch already reached during the
    episode remains sticky until :meth:`reset`; this keeps the one-shot reward
    independent from later samples while still failing closed before success.
    """

    def __init__(self, num_envs: int, required_consecutive_steps: int = 1) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if required_consecutive_steps <= 0:
            raise ValueError("required_consecutive_steps must be positive")
        self.num_envs = num_envs
        self.required_consecutive_steps = required_consecutive_steps
        self._counters = [0] * num_envs
        self._latched = [False] * num_envs

    @property
    def counters(self) -> tuple[int, ...]:
        return tuple(self._counters)

    @property
    def latched(self) -> tuple[bool, ...]:
        return tuple(self._latched)

    def update(self, stable: Sequence[object]) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
        if len(stable) != self.num_envs:
            raise ValueError(f"stable length must be {self.num_envs}, got {len(stable)}")

        pulse = [False] * self.num_envs
        for index, value in enumerate(stable):
            is_stable = self._finite_truth(value)
            self._counters[index] = self._counters[index] + 1 if is_stable else 0
            if not self._latched[index] and self._counters[index] >= self.required_consecutive_steps:
                self._latched[index] = True
                pulse[index] = True
        return self.latched, tuple(pulse)

    def reset(self, env_indices: Iterable[int] | None = None) -> None:
        indices = range(self.num_envs) if env_indices is None else tuple(env_indices)
        for index in indices:
            if index < 0 or index >= self.num_envs:
                raise IndexError(f"environment index out of range: {index}")
            self._counters[index] = 0
            self._latched[index] = False

    @staticmethod
    def _finite_truth(value: object) -> bool:
        try:
            return math.isfinite(value) and bool(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False


assert ACTOR_OBSERVATION_DIM == 83
assert PRIVILEGED_OBSERVATION_DIM == 24
assert CRITIC_OBSERVATION_DIM == 107


__all__ = [
    "ACTION_EMA_ALPHA",
    "ACTION_SCALE",
    "ACTOR_OBSERVATION_DIM",
    "ACTOR_OBSERVATION_METADATA",
    "ACTOR_OBSERVATION_TERMS",
    "CONTACT_FORCE_THRESHOLD_N",
    "CONTROL_DECIMATION",
    "CONTROL_DT_S",
    "CRITIC_OBSERVATION_DIM",
    "EPISODE_LENGTH_S",
    "EFFECTIVE_ACTION_TARGET_HARD_LIMIT_RANGE_FRACTION",
    "FALLEN_ACTION_RATE_PENALTY_MULTIPLIER",
    "FALLEN_ANGVEL_PENALTY_MULTIPLIER",
    "FOLDED_JOINT_ANGLES_RAD",
    "FOOT_DYNAMIC_FRICTION",
    "FOOT_STATIC_FRICTION",
    "GROUND_DYNAMIC_FRICTION",
    "GROUND_STATIC_FRICTION",
    "GO2_SOFT_JOINT_LIMIT_FACTOR",
    "HEIGHT_ORIENTATION_GATE_FULL",
    "HEIGHT_ORIENTATION_GATE_START",
    "HEIGHT_SHAPING_MIN_M",
    "MAX_ANGULAR_SPEED_RAD_S",
    "MAX_BASE_HEIGHT_M",
    "MAX_LINEAR_SPEED_M_S",
    "MIN_BASE_HEIGHT_M",
    "MIN_FOOT_CONTACTS",
    "NOMINAL_TOTAL_MASS_KG",
    "NON_FOOT_COLLISION_THRESHOLD_N",
    "PHYSICS_DT_S",
    "PPO_INIT_NOISE_STD",
    "PPO_GAMMA",
    "POSE_CURRICULUM_PHASE_END_CONTROL_STEPS",
    "POSE_CURRICULUM_PROBABILITIES",
    "PRIVILEGED_OBSERVATION_DIM",
    "PRIVILEGED_OBSERVATION_TERMS",
    "R0_REWARD_TERMS",
    "RECOVER_POSES",
    "RESET_POSE_XY_RANGE_M",
    "RESET_YAW_RANGE_RAD",
    "RecoverPose",
    "RewardTerm",
    "STABLE_DWELL_STEPS",
    "SOLVER_JOINT_LIMIT_TOLERANCE_RAD",
    "StableHoldLatch",
    "REGULARIZATION_HEIGHT_GATE_FULL_M",
    "REGULARIZATION_HEIGHT_GATE_START_M",
    "REGULARIZATION_ORIENTATION_GATE_FULL",
    "REGULARIZATION_ORIENTATION_GATE_START",
    "UPRIGHT_ANGLE_DEG",
    "UPRIGHT_COS_THRESHOLD",
    "canonical_sha256",
    "recover_contract",
]
