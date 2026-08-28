import atexit
import math
import pathlib
import sys
import types

import pytest
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaaclab.app import AppLauncher


_APP = AppLauncher({"headless": True}).app
atexit.register(_APP.close)

import gymnasium as gym

from isaac_walk_g009.agent_cfg import G009RecoverPPORunnerCfg
from isaac_walk_g009.recover_contracts import (
    ACTION_EMA_ALPHA,
    ACTION_SCALE,
    ARTICULATION_SOLVER_POSITION_ITERATION_COUNT,
    ARTICULATION_SOLVER_VELOCITY_ITERATION_COUNT,
    ACTOR_OBSERVATION_TERMS,
    GO2_SOFT_JOINT_LIMIT_FACTOR,
    MAX_BASE_HEIGHT_M,
    MAX_LINEAR_SPEED_M_S,
    NOMINAL_TOTAL_MASS_KG,
    MIN_TOTAL_FOOT_SUPPORT_RATIO,
    POSE_CURRICULUM_PHASE_END_CONTROL_STEPS,
    POSE_CURRICULUM_PROBABILITIES,
    PRIVILEGED_OBSERVATION_TERMS,
    PPO_INIT_NOISE_STD,
    R0_REWARD_TERMS,
    SOLVER_JOINT_LIMIT_TOLERANCE_RAD,
)
from isaac_walk_g009.recover_env_cfg import (
    G009FlatRecoverEnvCfg,
    MAX_ANGULAR_SPEED_RAD_S,
    MIN_BASE_HEIGHT_M,
    MIN_FOOT_CONTACTS,
    STABLE_DWELL_STEPS,
    UPRIGHT_COS_THRESHOLD,
)
from isaac_walk_g009.mdp.events import (
    ApplyRecoverFootMaterial,
    recovery_pose_curriculum,
    reset_root_and_joints_for_recovery,
)
from isaac_walk_g009.registry import AGENT_ENTRY_POINTS, RECOVER_TASK_ENTRY_POINTS, register_tasks


def _term_names(group):
    return tuple(
        name
        for name, value in vars(group).items()
        if not name.startswith("_") and value is not None and hasattr(value, "func")
    )


def test_r0_pose_curriculum_uses_global_control_step_without_actor_exposure():
    env = types.SimpleNamespace(common_step_counter=0, device="cpu")
    expected = (
        (0, POSE_CURRICULUM_PROBABILITIES[0]),
        (1199, POSE_CURRICULUM_PROBABILITIES[0]),
        (1200, POSE_CURRICULUM_PROBABILITIES[0]),
        (1201, POSE_CURRICULUM_PROBABILITIES[1]),
        (2399, POSE_CURRICULUM_PROBABILITIES[1]),
        (2400, POSE_CURRICULUM_PROBABILITIES[1]),
        (2401, POSE_CURRICULUM_PROBABILITIES[2]),
    )
    for control_step, probabilities in expected:
        env.common_step_counter = control_step
        state = recovery_pose_curriculum(env, [])
        torch.testing.assert_close(
            env._g009_recover_pose_probabilities,
            torch.tensor(probabilities, dtype=torch.float32),
        )
        assert state["common_control_step"] == float(control_step)
    assert not hasattr(env, "_g009_recover_fall_class_one_hot")


def test_r0_pose_curriculum_keeps_first_1200_training_steps_prone_only():
    env = types.SimpleNamespace(common_step_counter=0, device="cpu")
    expected = torch.tensor(POSE_CURRICULUM_PROBABILITIES[0], dtype=torch.float32)

    for control_step in range(1, 1201):
        env.common_step_counter = control_step
        state = recovery_pose_curriculum(env, [])
        torch.testing.assert_close(env._g009_recover_pose_probabilities, expected)
        assert state["phase_index"] == 0.0


def test_r0_registry_uses_task_specific_runner():
    task_id = "Isaac-G009-Recover-Flat-Go2-R0-v0"
    register_tasks()
    register_tasks()
    assert RECOVER_TASK_ENTRY_POINTS == {
        task_id: "isaac_walk_g009.recover_env_cfg:G009FlatRecoverEnvCfg"
    }
    spec = gym.spec(task_id)
    assert spec.kwargs["env_cfg_entry_point"] == RECOVER_TASK_ENTRY_POINTS[task_id]
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == AGENT_ENTRY_POINTS[task_id]


def test_r0_flat_scene_and_disabled_randomization_contract():
    cfg = G009FlatRecoverEnvCfg()
    assert cfg.commands.base_velocity is None
    assert (
        cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count
        == ARTICULATION_SOLVER_POSITION_ITERATION_COUNT
        == 8
    )
    assert (
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count
        == ARTICULATION_SOLVER_VELOCITY_ITERATION_COUNT
        == 1
    )
    assert cfg.actions.joint_pos.class_type.__name__ == "EMAJointPositionToLimitsAction"
    assert cfg.actions.joint_pos.scale == ACTION_SCALE == 0.70
    assert cfg.actions.joint_pos.alpha == ACTION_EMA_ALPHA == 0.2
    assert cfg.actions.joint_pos.rescale_to_limits is True
    assert cfg.scene.robot.soft_joint_pos_limit_factor == GO2_SOFT_JOINT_LIMIT_FACTOR == 0.9
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None
    assert cfg.scene.terrain.max_init_terrain_level is None
    assert cfg.scene.terrain.prim_path == "/World/ground"
    assert cfg.scene.height_scanner is None
    camera = cfg.scene.body_range_camera
    assert camera.prim_path == "{ENV_REGEX_NS}/Robot/base"
    assert camera.mesh_prim_paths == ["/World/ground"]
    assert camera.data_types == ["distance_to_camera"]
    assert camera.ray_alignment == "base"
    assert camera.max_distance == 1.0
    assert camera.depth_clipping_behavior == "none"
    assert camera.pattern_cfg.width == 5
    assert camera.pattern_cfg.height == 3
    assert camera.pattern_cfg.width * camera.pattern_cfg.height == 15
    assert camera.offset.pos == (0.0, 0.0, -0.05)
    assert camera.offset.convention == "world"
    assert camera.offset.rot == pytest.approx((math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0))
    w, x, y, z = camera.offset.rot
    camera_forward_in_body = (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    )
    assert camera_forward_in_body == pytest.approx((0.0, 0.0, -1.0), abs=1.0e-12)
    assert cfg.scene.terrain.physics_material.static_friction == 0.8
    assert cfg.scene.terrain.physics_material.dynamic_friction == 0.6
    assert cfg.scene.terrain.physics_material.friction_combine_mode == "multiply"
    assert cfg.events.physics_material.params["static_friction_range"] == (1.0, 1.0)
    assert cfg.events.physics_material.params["dynamic_friction_range"] == (1.0, 1.0)
    assert cfg.events.physics_material.func is ApplyRecoverFootMaterial
    assert cfg.events.physics_material.params["ground_static_friction"] == 0.8
    assert cfg.events.physics_material.params["ground_dynamic_friction"] == 0.6
    assert cfg.events.push_robot is None
    assert cfg.events.add_base_mass is None
    assert cfg.events.base_external_force_torque is None
    assert cfg.events.base_com is None
    assert cfg.events.reset_robot_joints is None
    assert cfg.events.reset_base.func is reset_root_and_joints_for_recovery
    assert cfg.events.reset_base.mode == "reset"
    assert cfg.curriculum.terrain_levels is None
    assert cfg.curriculum.recover_pose_distribution.func is recovery_pose_curriculum
    assert cfg.curriculum.recover_pose_distribution.params == {
        "phase_end_control_steps": POSE_CURRICULUM_PHASE_END_CONTROL_STEPS,
        "phase_probabilities": POSE_CURRICULUM_PROBABILITIES,
    }
    assert cfg.episode_length_s == 8.0
    assert cfg.sim.dt == 0.005
    assert cfg.decimation == 4


def test_r0_observation_order_corruption_and_dimensions():
    cfg = G009FlatRecoverEnvCfg()
    actor_names = tuple(name for name, _ in ACTOR_OBSERVATION_TERMS)
    privileged_names = tuple(name for name, _ in PRIVILEGED_OBSERVATION_TERMS)
    assert _term_names(cfg.observations.policy) == actor_names
    assert _term_names(cfg.observations.critic) == actor_names + privileged_names
    assert cfg.observations.policy.enable_corruption is True
    assert cfg.observations.critic.enable_corruption is False
    assert cfg.observations.critic.base_height_gt.func.__name__ == "critic_base_height_gt"
    expected_noise = {
        "base_linear_velocity": (-0.1, 0.1),
        "base_angular_velocity": (-0.2, 0.2),
        "projected_gravity": (-0.05, 0.05),
        "relative_joint_position": (-0.01, 0.01),
        "relative_joint_velocity": (-1.5, 1.5),
        "normalized_four_foot_load": (-0.02, 0.02),
    }
    for name, bounds in expected_noise.items():
        noise = getattr(cfg.observations.policy, name).noise
        assert (noise.n_min, noise.n_max) == bounds
    assert cfg.observations.policy.body_fixed_range.clip == (0.0, 1.0)
    assert cfg.observations.policy.body_fixed_range.func.__name__ == "body_fixed_range"
    assert (
        cfg.observations.policy.body_fixed_range_hit_mask.func.__name__
        == "body_fixed_range_hit_mask"
    )
    assert (
        cfg.observations.policy.normalized_four_foot_load.params["nominal_total_mass_kg"]
        == NOMINAL_TOTAL_MASS_KG
    )
    for name in actor_names:
        if name not in expected_noise:
            assert getattr(cfg.observations.policy, name).noise is None
    assert sum(dim for _, dim in ACTOR_OBSERVATION_TERMS) == 83
    assert sum(dim for _, dim in PRIVILEGED_OBSERVATION_TERMS) == 24


def test_r0_reward_termination_and_threshold_contract():
    cfg = G009FlatRecoverEnvCfg()
    assert _term_names(cfg.rewards) == tuple(term.name for term in R0_REWARD_TERMS)
    assert tuple(getattr(cfg.rewards, term.name).weight for term in R0_REWARD_TERMS) == tuple(
        term.weight for term in R0_REWARD_TERMS
    )
    assert _term_names(cfg.terminations) == (
        "time_out",
        "stable_success",
        "numeric_invalid",
        "hard_joint_limit",
    )
    assert cfg.rewards.stable_success_once.func.__name__ == "stable_success_once"
    assert cfg.rewards.gated_base_height_progress.func.__name__ == "GatedBaseHeightProgress"
    assert cfg.rewards.soft_stand_progress.func.__name__ == "SoftStandProgress"
    assert cfg.rewards.gated_angvel_l2.func.__name__ == "gated_angvel_l2"
    assert cfg.rewards.gated_action_rate_l2.func.__name__ == "gated_action_rate_l2"
    assert cfg.terminations.stable_success.params["required_consecutive_steps"] == 25
    assert cfg.terminations.stable_success.params["contact_force_threshold"] == 1.0
    assert cfg.terminations.stable_success.params["non_foot_contact_force_threshold"] == 1.0
    assert (
        cfg.terminations.stable_success.params["min_total_foot_support_ratio"]
        == MIN_TOTAL_FOOT_SUPPORT_RATIO
        == 0.60
    )
    assert cfg.terminations.stable_success.params["nominal_total_mass_kg"] == NOMINAL_TOTAL_MASS_KG
    assert cfg.terminations.stable_success.params["non_foot_sensor_cfg"].body_names == [
        "base",
        ".*_hip",
        ".*_thigh",
        ".*_calf",
    ]
    assert cfg.terminations.hard_joint_limit.params["margin"] == 0.01
    assert SOLVER_JOINT_LIMIT_TOLERANCE_RAD == 0.01
    assert cfg.rewards.undesired_collision.params["contact_force_threshold"] == 1.0
    assert cfg.rewards.undesired_collision.params["min_base_height"] == 0.30
    assert UPRIGHT_COS_THRESHOLD == math.cos(math.radians(20.0))
    assert MIN_BASE_HEIGHT_M == 0.30
    assert MAX_BASE_HEIGHT_M == 0.60
    assert MAX_LINEAR_SPEED_M_S == 0.50
    assert MAX_ANGULAR_SPEED_RAD_S == 1.0
    assert MIN_FOOT_CONTACTS == 3
    assert STABLE_DWELL_STEPS == 25


def test_r0_runner_budget_and_ppo_inheritance():
    cfg = G009RecoverPPORunnerCfg()
    assert cfg.experiment_name == "g009_recover_r0"
    assert cfg.max_iterations == 300
    assert cfg.save_interval == 50
    assert cfg.clip_actions == 1.0
    assert cfg.policy.init_noise_std == PPO_INIT_NOISE_STD == 0.5
    assert cfg.num_steps_per_env == 24
    assert cfg.algorithm.num_learning_epochs == 5
    assert cfg.algorithm.num_mini_batches == 4
    assert cfg.policy.actor_hidden_dims == [512, 256, 128]
    assert cfg.policy.critic_hidden_dims == [512, 256, 128]
    assert cfg.policy.activation == "elu"
    assert cfg.policy.init_noise_std == PPO_INIT_NOISE_STD
    assert cfg.algorithm.clip_param == 0.2
    assert cfg.algorithm.entropy_coef == 0.01
    assert cfg.algorithm.gamma == 0.99
    assert cfg.algorithm.lam == 0.95
    assert cfg.algorithm.learning_rate == 0.001
    assert cfg.algorithm.schedule == "adaptive"
    assert cfg.algorithm.desired_kl == 0.01
    assert cfg.algorithm.max_grad_norm == 1.0
    assert cfg.empirical_normalization is False
