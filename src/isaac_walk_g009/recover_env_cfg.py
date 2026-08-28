"""Flat-ground environment configuration for G009 R0 self-righting."""

from __future__ import annotations

import math

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster import RayCasterCameraCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaac_walk_g008.env_cfg import G008CommandEnvCfg

from .mdp import recover as recover_mdp
from .mdp.events import (
    ApplyRecoverFootMaterial,
    recovery_pose_curriculum,
    reset_root_and_joints_for_recovery,
)
from .recover_contracts import (
    ACTION_EMA_ALPHA,
    ACTION_SCALE,
    ARTICULATION_SOLVER_POSITION_ITERATION_COUNT,
    CONTACT_FORCE_THRESHOLD_N,
    CONTROL_DECIMATION,
    EPISODE_LENGTH_S,
    FALLEN_ACTION_RATE_PENALTY_MULTIPLIER,
    FALLEN_ANGVEL_PENALTY_MULTIPLIER,
    FOOT_DYNAMIC_FRICTION,
    FOOT_STATIC_FRICTION,
    GRAVITY_MAGNITUDE_M_S2,
    GROUND_DYNAMIC_FRICTION,
    GROUND_STATIC_FRICTION,
    GO2_SOFT_JOINT_LIMIT_FACTOR,
    HEIGHT_ORIENTATION_GATE_FULL,
    HEIGHT_ORIENTATION_GATE_START,
    HEIGHT_SHAPING_MIN_M,
    MAX_ANGULAR_SPEED_RAD_S,
    MAX_BASE_HEIGHT_M,
    MAX_LINEAR_SPEED_M_S,
    MIN_BASE_HEIGHT_M,
    MIN_FOOT_CONTACTS,
    MIN_TOTAL_FOOT_SUPPORT_RATIO,
    NOMINAL_TOTAL_MASS_KG,
    NON_FOOT_COLLISION_THRESHOLD_N,
    PHYSICS_DT_S,
    PPO_GAMMA,
    POSE_CURRICULUM_PHASE_END_CONTROL_STEPS,
    POSE_CURRICULUM_PROBABILITIES,
    REGULARIZATION_HEIGHT_GATE_FULL_M,
    REGULARIZATION_HEIGHT_GATE_START_M,
    REGULARIZATION_ORIENTATION_GATE_FULL,
    REGULARIZATION_ORIENTATION_GATE_START,
    SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
    STABLE_DWELL_STEPS,
    UPRIGHT_COS_THRESHOLD,
)

_ROBOT = SceneEntityCfg("robot")
_FEET = SceneEntityCfg("contact_forces", body_names=".*_foot")
_BODY_RANGE_CAMERA = SceneEntityCfg("body_range_camera")
_SQRT_HALF = math.sqrt(0.5)
_NON_FOOT_BODIES = SceneEntityCfg(
    "contact_forces",
    body_names=["base", ".*_hip", ".*_thigh", ".*_calf"],
)
_STABLE_PARAMS = {
    "upright_threshold": UPRIGHT_COS_THRESHOLD,
    "min_base_height": MIN_BASE_HEIGHT_M,
    "max_base_height": MAX_BASE_HEIGHT_M,
    "min_contacts": MIN_FOOT_CONTACTS,
    "max_linear_speed": MAX_LINEAR_SPEED_M_S,
    "max_angular_speed": MAX_ANGULAR_SPEED_RAD_S,
    "contact_force_threshold": CONTACT_FORCE_THRESHOLD_N,
    "non_foot_contact_force_threshold": NON_FOOT_COLLISION_THRESHOLD_N,
    "min_total_foot_support_ratio": MIN_TOTAL_FOOT_SUPPORT_RATIO,
    "nominal_total_mass_kg": NOMINAL_TOTAL_MASS_KG,
    "gravity_magnitude": GRAVITY_MAGNITUDE_M_S2,
    "asset_cfg": _ROBOT,
    "sensor_cfg": _FEET,
    "non_foot_sensor_cfg": _NON_FOOT_BODIES,
}


def _actor_terms() -> dict[str, ObsTerm]:
    """Build the ordered deployable P-RECOVER-83 observation terms."""
    return {
        "base_linear_velocity": ObsTerm(
            func=base_mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
        "base_angular_velocity": ObsTerm(
            func=base_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObsTerm(
            func=base_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "relative_joint_position": ObsTerm(
            func=base_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "relative_joint_velocity": ObsTerm(
            func=base_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "last_recover_action": ObsTerm(func=base_mdp.last_action),
        "four_foot_contact_state": ObsTerm(
            func=recover_mdp.four_foot_contact_state,
            params={
                "sensor_cfg": _FEET,
                "contact_force_threshold": CONTACT_FORCE_THRESHOLD_N,
            },
        ),
        "normalized_four_foot_load": ObsTerm(
            func=recover_mdp.normalized_four_foot_load,
            params={
                "sensor_cfg": _FEET,
                "nominal_total_mass_kg": NOMINAL_TOTAL_MASS_KG,
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "body_fixed_range": ObsTerm(
            func=recover_mdp.body_fixed_range,
            params={
                "sensor_cfg": _BODY_RANGE_CAMERA,
                "max_distance": 1.0,
                "expected_ray_count": 15,
            },
            clip=(0.0, 1.0),
        ),
        "body_fixed_range_hit_mask": ObsTerm(
            func=recover_mdp.body_fixed_range_hit_mask,
            params={
                "sensor_cfg": _BODY_RANGE_CAMERA,
                "max_distance": 1.0,
                "expected_ray_count": 15,
            },
        ),
    }


@configclass
class RecoverPolicyCfg(ObsGroup):
    """Deployable, corrupted P-RECOVER-83 actor group."""

    base_linear_velocity = _actor_terms()["base_linear_velocity"]
    base_angular_velocity = _actor_terms()["base_angular_velocity"]
    projected_gravity = _actor_terms()["projected_gravity"]
    relative_joint_position = _actor_terms()["relative_joint_position"]
    relative_joint_velocity = _actor_terms()["relative_joint_velocity"]
    last_recover_action = _actor_terms()["last_recover_action"]
    four_foot_contact_state = _actor_terms()["four_foot_contact_state"]
    normalized_four_foot_load = _actor_terms()["normalized_four_foot_load"]
    body_fixed_range = _actor_terms()["body_fixed_range"]
    body_fixed_range_hit_mask = _actor_terms()["body_fixed_range_hit_mask"]

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class RecoverCriticCfg(ObsGroup):
    """Uncorrupted actor prefix plus the C-RECOVER privileged suffix."""

    base_linear_velocity = _actor_terms()["base_linear_velocity"]
    base_angular_velocity = _actor_terms()["base_angular_velocity"]
    projected_gravity = _actor_terms()["projected_gravity"]
    relative_joint_position = _actor_terms()["relative_joint_position"]
    relative_joint_velocity = _actor_terms()["relative_joint_velocity"]
    last_recover_action = _actor_terms()["last_recover_action"]
    four_foot_contact_state = _actor_terms()["four_foot_contact_state"]
    normalized_four_foot_load = _actor_terms()["normalized_four_foot_load"]
    body_fixed_range = _actor_terms()["body_fixed_range"]
    body_fixed_range_hit_mask = _actor_terms()["body_fixed_range_hit_mask"]
    terrain_normal_gt = ObsTerm(func=recover_mdp.terrain_normal_gt)
    base_height_gt = ObsTerm(func=recover_mdp.base_height_gt)
    four_foot_effective_static_dynamic_friction = ObsTerm(
        func=recover_mdp.four_foot_effective_static_dynamic_friction,
        params={
            "configured_static_friction": GROUND_STATIC_FRICTION * FOOT_STATIC_FRICTION,
            "configured_dynamic_friction": GROUND_DYNAMIC_FRICTION * FOOT_DYNAMIC_FRICTION,
        },
    )
    whole_body_com_base = ObsTerm(func=recover_mdp.whole_body_com_base)
    total_mass = ObsTerm(func=recover_mdp.total_mass)
    commanded_wrench = ObsTerm(func=recover_mdp.commanded_wrench)
    normalized_pulse_time_remaining = ObsTerm(func=recover_mdp.normalized_pulse_time_remaining)
    source_fall_class_one_hot = ObsTerm(func=recover_mdp.source_fall_class_one_hot)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class RecoverObservationsCfg:
    policy: RecoverPolicyCfg = RecoverPolicyCfg()
    critic: RecoverCriticCfg = RecoverCriticCfg()


@configclass
class RecoverRewardsCfg:
    upright_progress = RewTerm(
        func=recover_mdp.UprightProgress,
        weight=2.0,
        params={"discount_factor": PPO_GAMMA},
    )
    gated_base_height_progress = RewTerm(
        func=recover_mdp.GatedBaseHeightProgress,
        weight=2.0,
        params={
            "min_height": HEIGHT_SHAPING_MIN_M,
            "target_height": MIN_BASE_HEIGHT_M,
            "orientation_gate_start": HEIGHT_ORIENTATION_GATE_START,
            "orientation_gate_full": HEIGHT_ORIENTATION_GATE_FULL,
            "discount_factor": PPO_GAMMA,
        },
    )
    soft_stand_progress = RewTerm(
        func=recover_mdp.SoftStandProgress,
        weight=2.0,
        params={
            "min_height": HEIGHT_SHAPING_MIN_M,
            "target_height": MIN_BASE_HEIGHT_M,
            "contact_force_threshold": CONTACT_FORCE_THRESHOLD_N,
            "min_contacts": MIN_FOOT_CONTACTS,
            "min_total_foot_support_ratio": MIN_TOTAL_FOOT_SUPPORT_RATIO,
            "nominal_total_mass_kg": NOMINAL_TOTAL_MASS_KG,
            "gravity_magnitude": GRAVITY_MAGNITUDE_M_S2,
            "discount_factor": PPO_GAMMA,
            "asset_cfg": _ROBOT,
            "sensor_cfg": _FEET,
        },
    )
    stable_support = RewTerm(func=recover_mdp.stable_support, weight=0.5, params=_STABLE_PARAMS)
    upright_hold = RewTerm(func=recover_mdp.upright_hold, weight=5.0, params=_STABLE_PARAMS)
    stable_success_once = RewTerm(func=recover_mdp.stable_success_once, weight=10.0)
    gated_angvel_l2 = RewTerm(
        func=recover_mdp.gated_angvel_l2,
        weight=-0.05,
        params={
            "fallen_multiplier": FALLEN_ANGVEL_PENALTY_MULTIPLIER,
            "orientation_gate_start": REGULARIZATION_ORIENTATION_GATE_START,
            "orientation_gate_full": REGULARIZATION_ORIENTATION_GATE_FULL,
            "height_gate_start": REGULARIZATION_HEIGHT_GATE_START_M,
            "height_gate_full": REGULARIZATION_HEIGHT_GATE_FULL_M,
            "asset_cfg": _ROBOT,
        },
    )
    joint_limit = RewTerm(func=base_mdp.joint_pos_limits, weight=-2.0)
    torque_l2 = RewTerm(func=base_mdp.joint_torques_l2, weight=-0.0002)
    joint_acc_l2 = RewTerm(func=base_mdp.joint_acc_l2, weight=-2.5e-7)
    gated_action_rate_l2 = RewTerm(
        func=recover_mdp.gated_action_rate_l2,
        weight=-0.01,
        params={
            "fallen_multiplier": FALLEN_ACTION_RATE_PENALTY_MULTIPLIER,
            "orientation_gate_start": REGULARIZATION_ORIENTATION_GATE_START,
            "orientation_gate_full": REGULARIZATION_ORIENTATION_GATE_FULL,
            "height_gate_start": REGULARIZATION_HEIGHT_GATE_START_M,
            "height_gate_full": REGULARIZATION_HEIGHT_GATE_FULL_M,
            "asset_cfg": _ROBOT,
        },
    )
    mechanical_power_proxy = RewTerm(func=recover_mdp.mechanical_power_proxy, weight=-1.0e-5)
    undesired_collision = RewTerm(
        func=recover_mdp.undesired_collision,
        weight=-1.0,
        params={
            "sensor_cfg": _NON_FOOT_BODIES,
            "contact_force_threshold": NON_FOOT_COLLISION_THRESHOLD_N,
            "min_base_height": MIN_BASE_HEIGHT_M,
            "asset_cfg": _ROBOT,
        },
    )


@configclass
class RecoverTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    stable_success = DoneTerm(
        func=recover_mdp.StableSuccess,
        params={"required_consecutive_steps": STABLE_DWELL_STEPS, **_STABLE_PARAMS},
    )
    numeric_invalid = DoneTerm(func=recover_mdp.numeric_invalid)
    hard_joint_limit = DoneTerm(
        func=recover_mdp.urdf_hard_joint_limit_violation,
        params={"margin": SOLVER_JOINT_LIMIT_TOLERANCE_RAD},
    )


@configclass
class G009FlatRecoverEnvCfg(G008CommandEnvCfg):
    """Scratch PPO task for recovery from four curated flat-ground poses."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity = None
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = (
            ARTICULATION_SOLVER_POSITION_ITERATION_COUNT
        )
        self.scene.robot.soft_joint_pos_limit_factor = GO2_SOFT_JOINT_LIMIT_FACTOR
        self.actions.joint_pos = base_mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=ACTION_SCALE,
            alpha=ACTION_EMA_ALPHA,
            rescale_to_limits=True,
        )
        self.observations = RecoverObservationsCfg()
        self.rewards = RecoverRewardsCfg()
        self.terminations = RecoverTerminationsCfg()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.prim_path = "/World/ground"
        self.scene.terrain.physics_material.static_friction = GROUND_STATIC_FRICTION
        self.scene.terrain.physics_material.dynamic_friction = GROUND_DYNAMIC_FRICTION
        self.scene.terrain.physics_material.friction_combine_mode = "multiply"
        self.scene.height_scanner = None
        self.scene.body_range_camera = RayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            mesh_prim_paths=["/World/ground"],
            update_period=CONTROL_DECIMATION * PHYSICS_DT_S,
            offset=RayCasterCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, -0.05),
                rot=(_SQRT_HALF, 0.0, _SQRT_HALF, 0.0),
                convention="world",
            ),
            data_types=["distance_to_camera"],
            depth_clipping_behavior="none",
            max_distance=1.0,
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
                width=5,
                height=3,
            ),
            debug_vis=False,
        )
        self.curriculum.terrain_levels = None
        self.curriculum.recover_pose_distribution = CurrTerm(
            func=recovery_pose_curriculum,
            params={
                "phase_end_control_steps": POSE_CURRICULUM_PHASE_END_CONTROL_STEPS,
                "phase_probabilities": POSE_CURRICULUM_PROBABILITIES,
            },
        )

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.base_com = None
        self.events.reset_robot_joints = None
        self.events.reset_base = EventTerm(
            func=reset_root_and_joints_for_recovery,
            mode="reset",
            params={"asset_cfg": _ROBOT},
        )
        self.events.physics_material = EventTerm(
            func=ApplyRecoverFootMaterial,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "static_friction_range": (FOOT_STATIC_FRICTION, FOOT_STATIC_FRICTION),
                "dynamic_friction_range": (FOOT_DYNAMIC_FRICTION, FOOT_DYNAMIC_FRICTION),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
                "ground_static_friction": GROUND_STATIC_FRICTION,
                "ground_dynamic_friction": GROUND_DYNAMIC_FRICTION,
                "make_consistent": True,
            },
        )

        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = PHYSICS_DT_S
        self.decimation = CONTROL_DECIMATION
        self.sim.render_interval = self.decimation
        self.scene.body_range_camera.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physics_material = self.scene.terrain.physics_material


__all__ = [
    "CONTACT_FORCE_THRESHOLD_N",
    "G009FlatRecoverEnvCfg",
    "MAX_ANGULAR_SPEED_RAD_S",
    "MIN_BASE_HEIGHT_M",
    "MIN_FOOT_CONTACTS",
    "RecoverCriticCfg",
    "RecoverObservationsCfg",
    "RecoverPolicyCfg",
    "RecoverRewardsCfg",
    "RecoverTerminationsCfg",
    "STABLE_DWELL_STEPS",
    "UPRIGHT_COS_THRESHOLD",
]
