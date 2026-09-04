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
    MAX_DEPENETRATION_VELOCITY_M_S,
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
    G009FlatRecoverMatrixEnvCfg,
    G009FlatRecoverMatrixGate01EnvCfg,
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
from isaac_walk_g009.matrix_gate01 import (
    MATRIX_CRITIC_OBSERVATION_DIM,
    MATRIX_POLICY_OBSERVATION_DIM,
    NOMINAL_BODY_WEIGHT_N,
    ORDERED_BODY_NAMES,
    ORDERED_BODY_NAMES_SHA256,
    reset_runtime_telemetry,
    runtime_telemetry,
    whole_body_terrain_contact_matrix_base_normalized,
)
import isaac_walk_g009.matrix_gate01 as matrix_gate01
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
    production_matrix_task_id = "Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0"
    matrix_task_id = "Isaac-G009-Recover-Flat-Go2-R0-MatrixGate01-v0"
    register_tasks()
    register_tasks()
    assert RECOVER_TASK_ENTRY_POINTS == {
        task_id: "isaac_walk_g009.recover_env_cfg:G009FlatRecoverEnvCfg",
        production_matrix_task_id: (
            "isaac_walk_g009.recover_env_cfg:G009FlatRecoverMatrixEnvCfg"
        ),
        matrix_task_id: "isaac_walk_g009.recover_env_cfg:G009FlatRecoverMatrixGate01EnvCfg",
    }
    spec = gym.spec(task_id)
    assert spec.kwargs["env_cfg_entry_point"] == RECOVER_TASK_ENTRY_POINTS[task_id]
    assert spec.kwargs["rsl_rl_cfg_entry_point"] == AGENT_ENTRY_POINTS[task_id]
    matrix_spec = gym.spec(matrix_task_id)
    assert matrix_spec.kwargs["env_cfg_entry_point"] == RECOVER_TASK_ENTRY_POINTS[matrix_task_id]
    assert matrix_spec.kwargs["rsl_rl_cfg_entry_point"] == AGENT_ENTRY_POINTS[matrix_task_id]
    production_spec = gym.spec(production_matrix_task_id)
    assert production_spec.kwargs["env_cfg_entry_point"] == RECOVER_TASK_ENTRY_POINTS[
        production_matrix_task_id
    ]
    assert production_spec.kwargs["rsl_rl_cfg_entry_point"] == AGENT_ENTRY_POINTS[
        production_matrix_task_id
    ]


def test_matrix_production_and_gate01_are_isolated_from_baseline():
    baseline = G009FlatRecoverEnvCfg()
    baseline_policy_terms = _term_names(baseline.observations.policy)
    baseline_critic_terms = _term_names(baseline.observations.critic)
    baseline_filter_paths = tuple(baseline.scene.contact_forces.filter_prim_paths_expr)
    production = G009FlatRecoverMatrixEnvCfg()
    gate = G009FlatRecoverMatrixGate01EnvCfg()
    baseline_after = G009FlatRecoverEnvCfg()
    name = "whole_body_terrain_contact_matrix_base_normalized"

    assert not hasattr(baseline.observations.policy, name)
    assert getattr(production.observations.policy, name).params["collect_gate_telemetry"] is False
    assert getattr(production.observations.critic, name).params["collect_gate_telemetry"] is False
    assert getattr(gate.observations.policy, name).params["collect_gate_telemetry"] is True
    assert getattr(gate.observations.critic, name).params["collect_gate_telemetry"] is True
    assert _term_names(production.observations.policy) == _term_names(gate.observations.policy)
    assert _term_names(production.observations.critic) == _term_names(gate.observations.critic)
    assert _term_names(production.observations.critic)[:11] == _term_names(
        production.observations.policy
    )
    assert _term_names(production.observations.critic)[11:] == _term_names(
        baseline.observations.critic
    )[10:]
    assert production.observations.policy.enable_corruption is True
    assert production.observations.critic.enable_corruption is False
    assert production.scene.contact_forces.filter_prim_paths_expr == list(
        matrix_gate01.TERRAIN_FILTER_PATHS
    )
    assert production.scene.contact_forces.history_length == 1
    assert MATRIX_POLICY_OBSERVATION_DIM == 140
    assert MATRIX_CRITIC_OBSERVATION_DIM == 164
    assert _term_names(baseline_after.observations.policy) == baseline_policy_terms
    assert _term_names(baseline_after.observations.critic) == baseline_critic_terms
    assert tuple(baseline_after.scene.contact_forces.filter_prim_paths_expr) == baseline_filter_paths


def test_matrix_production_flags_and_sanitizes_nonfinite_rows_without_host_sync(monkeypatch):
    source = torch.zeros((4, 19, 1, 3), dtype=torch.float32)
    source[0, 0, 0] = torch.tensor((10.0, -4.0, 2.0))
    source[1, 0, 0, 0] = torch.nan
    source[2, 0, 0, 1] = torch.inf
    quaternions = torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 4)
    quaternions[3, 0] = torch.nan
    sensor = types.SimpleNamespace(
        body_names=list(ORDERED_BODY_NAMES), data=types.SimpleNamespace(force_matrix_w=source)
    )
    asset = types.SimpleNamespace(data=types.SimpleNamespace(root_quat_w=quaternions))

    class Scene(types.SimpleNamespace):
        def __getitem__(self, name):
            return {"robot": asset}[name]

    env = types.SimpleNamespace(scene=Scene(sensors={"contact_forces": sensor}))
    monkeypatch.setattr(
        matrix_gate01,
        "adapt_terrain_pair_force_matrix_w",
        lambda _source: (_ for _ in ()).throw(AssertionError("authority adapter called")),
    )
    monkeypatch.setattr(
        matrix_gate01,
        "_live_contract_readback",
        lambda *_args: (_ for _ in ()).throw(AssertionError("live USD readback called")),
    )

    with monkeypatch.context() as no_sync:
        no_sync.setattr(
            torch.Tensor,
            "item",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("host sync called")),
        )
        result = whole_body_terrain_contact_matrix_base_normalized(
            env, collect_gate_telemetry=False
        )

    expected_valid = torch.tanh(source[0].sum(dim=1).reshape(-1) / NOMINAL_BODY_WEIGHT_N)
    torch.testing.assert_close(result[0], expected_valid)
    torch.testing.assert_close(result[1:], torch.zeros_like(result[1:]))
    assert torch.isfinite(result).all()
    assert env._g009_actor_signal_invalid.tolist() == [False, True, True, True]


def test_matrix_gate01_isolated_policy_observation_contract():
    baseline = G009FlatRecoverEnvCfg()
    matrix = G009FlatRecoverMatrixGate01EnvCfg()
    assert not hasattr(baseline.observations.policy, "whole_body_terrain_contact_matrix_base_normalized")
    assert matrix.scene.contact_forces.filter_prim_paths_expr == [
        "/World/ground/terrain/GroundPlane/CollisionPlane"
    ]
    assert matrix.scene.contact_forces.history_length == 1
    term = matrix.observations.policy.whole_body_terrain_contact_matrix_base_normalized
    assert term.func.__name__ == "whole_body_terrain_contact_matrix_base_normalized"
    assert term.noise is None
    assert term.clip is None
    assert matrix.observations.policy.enable_corruption is True
    assert matrix.observations.critic.enable_corruption is False
    assert _term_names(matrix.observations.policy)[-1] == "whole_body_terrain_contact_matrix_base_normalized"
    assert _term_names(matrix.observations.critic)[:11] == _term_names(matrix.observations.policy)
    assert _term_names(matrix.observations.critic)[11:] == _term_names(baseline.observations.critic)[10:]
    assert matrix.scene.robot.spawn.articulation_props.solver_position_iteration_count == 8
    assert matrix.scene.robot.spawn.articulation_props.solver_velocity_iteration_count == 0


def test_matrix_gate01_projects_world_force_to_bounded_base_frame_and_records_runtime(monkeypatch):
    source = torch.zeros((2, 19, 1, 3), dtype=torch.float32)
    source[0, 3, 0] = torch.tensor((1.0, -2.0, 3.0))
    source_before = source.clone()
    sensor = types.SimpleNamespace(
        body_names=list(ORDERED_BODY_NAMES), data=types.SimpleNamespace(force_matrix_w=source)
    )
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1))
    )

    class Scene(types.SimpleNamespace):
        def __getitem__(self, name):
            return {"robot": asset}[name]

    env = types.SimpleNamespace(scene=Scene(sensors={"contact_forces": sensor}))
    live_contract = {
        "solver_position_velocity": [[8, 0], [8, 0]],
        "max_depenetration_velocity_m_s": [1.0, 1.0],
        "action_scale": 0.65,
        "action_ema_alpha": 0.2,
    }
    monkeypatch.setattr(matrix_gate01, "_live_contract_readback", lambda _env, _asset: live_contract)
    reset_runtime_telemetry()

    result = whole_body_terrain_contact_matrix_base_normalized(env, collect_gate_telemetry=True)

    assert result.shape == (2, 57)
    torch.testing.assert_close(
        result[0, 9:12], torch.tanh(torch.tensor((1.0, -2.0, 3.0)) / NOMINAL_BODY_WEIGHT_N)
    )
    torch.testing.assert_close(source, source_before)
    assert runtime_telemetry() == {
        "call_count": 1,
        "all_source_finite": True,
        "all_output_finite": True,
        "source_unchanged": True,
        "positive_magnitude_count": 1,
        "nonzero_output_count": 3,
        "maximum_magnitude_n": pytest.approx(math.sqrt(14.0)),
        "output_minimum": pytest.approx(math.tanh(-2.0 / NOMINAL_BODY_WEIGHT_N)),
        "output_maximum": pytest.approx(math.tanh(3.0 / NOMINAL_BODY_WEIGHT_N)),
        "output_variance_maximum": pytest.approx(float(result.var(unbiased=False))),
        "ordered_body_names": list(ORDERED_BODY_NAMES),
        "ordered_body_names_sha256": ORDERED_BODY_NAMES_SHA256,
        "body_order_consistent": True,
        "live_contract": live_contract,
        "source_shapes": ["2x19x1x3"],
        "output_shapes": ["2x57"],
        "source_dtypes": ["torch.float32"],
        "source_devices": ["cpu"],
    }


def test_matrix_gate01_projection_is_invariant_to_joint_world_yaw_rotation():
    reset_runtime_telemetry()
    half = math.sqrt(0.5)
    quaternions = torch.tensor(((1.0, 0.0, 0.0, 0.0), (half, 0.0, 0.0, half)))
    base_force = torch.tensor(((10.0, 4.0, 2.0), (10.0, 4.0, 2.0)))
    world_force = torch.stack((base_force[0], torch.tensor((-4.0, 10.0, 2.0))))
    source = torch.zeros((2, 19, 1, 3), dtype=torch.float32)
    source[:, 0, 0] = world_force
    sensor = types.SimpleNamespace(
        body_names=list(ORDERED_BODY_NAMES), data=types.SimpleNamespace(force_matrix_w=source)
    )
    asset = types.SimpleNamespace(data=types.SimpleNamespace(root_quat_w=quaternions))

    class Scene(types.SimpleNamespace):
        def __getitem__(self, name):
            return {"robot": asset}[name]

    result = whole_body_terrain_contact_matrix_base_normalized(
        types.SimpleNamespace(scene=Scene(sensors={"contact_forces": sensor}))
    )
    torch.testing.assert_close(result[0], result[1], atol=1.0e-6, rtol=1.0e-6)
    assert runtime_telemetry()["call_count"] == 0


def test_matrix_gate01_production_projection_avoids_syncing_authority_adapter(monkeypatch):
    source = torch.zeros((1, 19, 1, 3), dtype=torch.float32)
    source[:, 0, 0, 2] = 10.0
    sensor = types.SimpleNamespace(
        body_names=list(ORDERED_BODY_NAMES), data=types.SimpleNamespace(force_matrix_w=source)
    )
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(root_quat_w=torch.tensor(((1.0, 0.0, 0.0, 0.0),)))
    )

    class Scene(types.SimpleNamespace):
        def __getitem__(self, name):
            return {"robot": asset}[name]

    monkeypatch.setattr(
        matrix_gate01,
        "adapt_terrain_pair_force_matrix_w",
        lambda _source: (_ for _ in ()).throw(AssertionError("host-sync adapter called")),
    )
    monkeypatch.setattr(
        matrix_gate01,
        "_live_contract_readback",
        lambda *_args: (_ for _ in ()).throw(AssertionError("live telemetry called")),
    )
    result = whole_body_terrain_contact_matrix_base_normalized(
        types.SimpleNamespace(scene=Scene(sensors={"contact_forces": sensor})),
        collect_gate_telemetry=False,
    )
    assert result.shape == (1, 57)
    assert result[0, 2] > 0.0


def test_matrix_gate01_rejects_noncanonical_body_order_before_projection():
    source = torch.zeros((1, 19, 1, 3), dtype=torch.float32)
    sensor = types.SimpleNamespace(
        body_names=list(reversed(ORDERED_BODY_NAMES)),
        data=types.SimpleNamespace(force_matrix_w=source),
    )
    asset = types.SimpleNamespace(
        data=types.SimpleNamespace(root_quat_w=torch.tensor(((1.0, 0.0, 0.0, 0.0),)))
    )

    class Scene(types.SimpleNamespace):
        def __getitem__(self, name):
            return {"robot": asset}[name]

    with pytest.raises(ValueError, match="sensor order mismatch"):
        whole_body_terrain_contact_matrix_base_normalized(
            types.SimpleNamespace(scene=Scene(sensors={"contact_forces": sensor}))
        )


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
        == 0
    )
    assert (
        cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity
        == MAX_DEPENETRATION_VELOCITY_M_S
        == 1.0
    )
    assert cfg.actions.joint_pos.class_type.__name__ == "EMAJointPositionToLimitsAction"
    assert cfg.actions.joint_pos.scale == ACTION_SCALE == 0.65
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
    assert cfg.algorithm.entropy_coef == 0.0
    assert cfg.algorithm.gamma == 0.99
    assert cfg.algorithm.lam == 0.95
    assert cfg.algorithm.learning_rate == 0.001
    assert cfg.algorithm.schedule == "adaptive"
    assert cfg.algorithm.desired_kl == 0.01
    assert cfg.algorithm.max_grad_norm == 1.0
    assert cfg.empirical_normalization is False
