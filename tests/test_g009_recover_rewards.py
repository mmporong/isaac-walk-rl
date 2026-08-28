from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _install_isaaclab_stubs() -> dict[str, object]:
    """Install only the import surface needed by the pure tensor helpers."""
    isaaclab = types.ModuleType("isaaclab")
    assets = types.ModuleType("isaaclab.assets")
    managers = types.ModuleType("isaaclab.managers")
    sensors = types.ModuleType("isaaclab.sensors")
    utils = types.ModuleType("isaaclab.utils")
    math_module = types.ModuleType("isaaclab.utils.math")

    class _ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

    class _SceneEntityCfg:
        def __init__(self, name, **kwargs):
            self.name = name
            self.body_ids = kwargs.get("body_ids", slice(None))
            self.joint_ids = kwargs.get("joint_ids", slice(None))

    def quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        q_vec = quaternion[:, 1:]
        uv = torch.linalg.cross(q_vec, vector, dim=-1)
        uuv = torch.linalg.cross(q_vec, uv, dim=-1)
        return vector + 2.0 * (quaternion[:, :1] * uv + uuv)

    def quat_apply_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        conjugate = torch.cat((quaternion[:, :1], -quaternion[:, 1:]), dim=-1)
        return quat_apply(conjugate, vector)

    assets.Articulation = object
    managers.ManagerTermBase = _ManagerTermBase
    managers.ManagerTermBaseCfg = object
    managers.SceneEntityCfg = _SceneEntityCfg
    sensors.ContactSensor = object
    math_module.quat_apply = quat_apply
    math_module.quat_apply_inverse = quat_apply_inverse
    utils.math = math_module

    stubs = {
        "isaaclab": isaaclab,
        "isaaclab.assets": assets,
        "isaaclab.managers": managers,
        "isaaclab.sensors": sensors,
        "isaaclab.utils": utils,
        "isaaclab.utils.math": math_module,
    }
    previous = {name: sys.modules[name] for name in stubs if name in sys.modules}
    sys.modules.update(stubs)
    return previous


PREVIOUS_ISAACLAB_MODULES = _install_isaaclab_stubs()
SPEC = importlib.util.spec_from_file_location("g009_recover_under_test", ROOT / "src/isaac_walk_g009/mdp/recover.py")
assert SPEC is not None and SPEC.loader is not None
RECOVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVER)
for module_name in tuple(sys.modules):
    if module_name == "isaaclab" or module_name.startswith("isaaclab."):
        sys.modules.pop(module_name)
sys.modules.update(PREVIOUS_ISAACLAB_MODULES)
StableSuccess = RECOVER.StableSuccess
PotentialProgress = RECOVER._PotentialProgress
_progress_delta = RECOVER._progress_delta
_gated_height_potential_from_values = RECOVER._gated_height_potential_from_values
_positive_support_normal_load = RECOVER._positive_support_normal_load
_soft_stand_potential_from_values = RECOVER._soft_stand_potential_from_values
_stand_regularization_gate_from_values = RECOVER._stand_regularization_gate_from_values
_stable_mask_from_values = RECOVER._stable_mask_from_values
_stable_predicate = RECOVER._stable_predicate
_upright_alignment = RECOVER._upright_alignment
stable_success_once = RECOVER.stable_success_once
undesired_collision = RECOVER.undesired_collision
body_fixed_range = RECOVER.body_fixed_range
body_fixed_range_hit_mask = RECOVER.body_fixed_range_hit_mask
normalized_foot_load = RECOVER.normalized_foot_load
STABLE_TERM_PARAMS = {
    "upright_threshold": 0.95,
    "min_base_height": 0.25,
    "max_base_height": 0.4,
    "min_contacts": 3,
    "max_linear_speed": 0.1,
    "max_angular_speed": 0.2,
    "contact_force_threshold": 1.0,
    "non_foot_contact_force_threshold": 1.0,
    "min_total_foot_support_ratio": 0.60,
    "nominal_total_mass_kg": 15.0,
    "gravity_magnitude": 9.81,
    "asset_cfg": RECOVER.SceneEntityCfg("robot"),
    "sensor_cfg": RECOVER.SceneEntityCfg("contact_forces", body_ids=[0, 1, 2, 3]),
    "non_foot_sensor_cfg": RECOVER.SceneEntityCfg("contact_forces", body_ids=[4, 5]),
}


def test_upright_alignment_uses_full_quaternion_and_support_normal() -> None:
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    upside_down = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])

    assert _upright_alignment(identity, normal).item() == pytest.approx(1.0)
    assert _upright_alignment(upside_down, normal).item() == pytest.approx(-1.0)


def test_potential_progress_preserves_sign_and_scale() -> None:
    current = torch.tensor([0.8, 0.2])
    previous = torch.tensor([0.5, 0.4])
    torch.testing.assert_close(_progress_delta(current, previous, scale=2.0), torch.tensor([0.6, -0.4]))


def test_gated_height_potential_is_normalized_and_requires_upright_progress() -> None:
    potential = _gated_height_potential_from_values(
        alignment=torch.tensor([1.0, 0.4, 0.8, -1.0]),
        base_height=torch.tensor([0.18, 0.30, 0.30, 0.30]),
        min_height=0.06,
        target_height=0.30,
        orientation_gate_start=0.0,
        orientation_gate_full=0.8,
    )
    torch.testing.assert_close(potential, torch.tensor([0.5, 0.5, 1.0, 0.0]))


def test_soft_stand_potential_is_bounded_and_dense_before_strict_success() -> None:
    potential = _soft_stand_potential_from_values(
        alignment=torch.tensor([1.0, 1.0, 0.0, -1.0]),
        base_height=torch.full((4,), 0.30),
        foot_normal_load=torch.tensor(
            [
                [30.0, 30.0, 30.0, 0.0],
                [30.0, 30.0, 0.0, 0.0],
                [30.0, 30.0, 0.0, 0.0],
                [30.0, 30.0, 30.0, 0.0],
            ]
        ),
        min_height=0.06,
        target_height=0.30,
        contact_force_threshold=1.0,
        min_contacts=3,
        target_total_foot_load=90.0,
    )
    torch.testing.assert_close(
        potential,
        torch.tensor([1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]),
    )
    assert torch.all((potential >= 0.0) & (potential <= 1.0))


def test_regularization_gate_is_off_while_fallen_and_full_when_standing() -> None:
    gate = _stand_regularization_gate_from_values(
        alignment=torch.tensor([1.0, 0.5, 1.0]),
        base_height=torch.tensor([0.30, 0.30, 0.20]),
        orientation_gate_start=0.5,
        orientation_gate_full=0.94,
        height_gate_start=0.20,
        height_gate_full=0.30,
    )
    torch.testing.assert_close(gate, torch.tensor([1.0, 0.0, 0.0]))


def test_stable_predicate_applies_every_threshold_and_fails_closed() -> None:
    result = _stable_mask_from_values(
        alignment=torch.tensor([0.96, 0.94, float("nan")]),
        base_height=torch.tensor([0.31, 0.31, 0.31]),
        contact_count=torch.tensor([4, 4, 4]),
        total_foot_normal_load=torch.tensor([100.0, 100.0, 100.0]),
        non_foot_contact_count=torch.tensor([0, 0, 0]),
        linear_speed=torch.tensor([0.05, 0.05, 0.05]),
        angular_speed=torch.tensor([0.1, 0.1, 0.1]),
        upright_threshold=0.95,
        min_base_height=0.25,
        max_base_height=0.4,
        min_contacts=3,
        min_total_foot_normal_load=90.0,
        max_linear_speed=0.1,
        max_angular_speed=0.2,
    )
    assert result.tolist() == [True, False, False]


def test_positive_support_load_rejects_tangential_and_negative_force() -> None:
    forces_w = torch.tensor(
        [[[50.0, 0.0, 0.0], [0.0, 0.0, 25.0], [0.0, 0.0, -10.0]]]
    )
    support_normal_w = torch.tensor([[0.0, 0.0, 1.0]])

    torch.testing.assert_close(
        _positive_support_normal_load(forces_w, support_normal_w),
        torch.tensor([[0.0, 25.0, 0.0]]),
    )


def test_body_fixed_range_maps_no_hit_and_out_of_range_to_sentinel_without_invalidating() -> None:
    depth = torch.tensor(
        [[[0.0, 0.25, 1.0, float("inf"), 2.0]]]
    ).repeat(2, 3, 1)
    camera = types.SimpleNamespace(data=types.SimpleNamespace(output={"distance_to_camera": depth}))
    env = types.SimpleNamespace(
        num_envs=2,
        scene=types.SimpleNamespace(sensors={"body_range_camera": camera}),
    )
    cfg = RECOVER.SceneEntityCfg("body_range_camera")

    ranges = body_fixed_range(env, cfg, max_distance=1.0, expected_ray_count=15)
    mask = body_fixed_range_hit_mask(env, cfg, max_distance=1.0, expected_ray_count=15)

    assert ranges.shape == mask.shape == (2, 15)
    torch.testing.assert_close(ranges[0, :5], torch.tensor([0.0, 0.25, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(mask[0, :5], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))
    assert not env._g009_actor_signal_invalid.any()


def test_body_fixed_range_fails_closed_on_negative_finite_sample() -> None:
    depth = torch.full((1, 3, 5), 0.5)
    depth[0, 1, 2] = -0.01
    camera = types.SimpleNamespace(data=types.SimpleNamespace(output={"distance_to_camera": depth}))
    env = types.SimpleNamespace(
        num_envs=1,
        scene=types.SimpleNamespace(sensors={"body_range_camera": camera}),
    )

    ranges = body_fixed_range(env, RECOVER.SceneEntityCfg("body_range_camera"), 1.0, 15)

    assert ranges[0, 7].item() == 1.0
    assert env._g009_actor_signal_invalid.tolist() == [True]


def test_normalized_foot_load_uses_force_magnitude_without_range_sensor() -> None:
    forces = torch.tensor(
        [[[3.0, 4.0, 0.0], [0.0, 0.0, 10.0], [6.0, 8.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=forces))
    env = types.SimpleNamespace(
        num_envs=1,
        scene=types.SimpleNamespace(sensors={"contact_forces": sensor}),
    )
    cfg = RECOVER.SceneEntityCfg("contact_forces", body_ids=[0, 1, 2, 3])

    load = normalized_foot_load(
        env,
        cfg,
        nominal_total_mass_kg=10.0,
        gravity_magnitude=10.0,
    )

    torch.testing.assert_close(load, torch.tensor([[0.05, 0.10, 0.10, 0.0]]))


def test_manager_stable_gate_requires_feet_to_carry_weight_without_body_support() -> None:
    class _Scene:
        def __init__(self, robot, sensor):
            self._robot = robot
            self.sensors = {"contact_forces": sensor}

        def __getitem__(self, name):
            assert name == "robot"
            return self._robot

    foot_forces = torch.tensor(
        [
            [[0.0, 0.0, 40.0], [0.0, 0.0, 30.0], [0.0, 0.0, 20.0], [0.0, 0.0, 0.0]],
            [[50.0, 0.0, 0.0], [50.0, 0.0, 0.0], [50.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
            [[0.0, 0.0, 40.0], [0.0, 0.0, 30.0], [0.0, 0.0, 20.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 10.0], [0.0, 0.0, 10.0], [0.0, 0.0, 10.0], [0.0, 0.0, 10.0]],
        ]
    )
    non_foot_forces = torch.zeros((4, 2, 3))
    non_foot_forces[2, 0, 0] = 2.0
    sensor = types.SimpleNamespace(
        data=types.SimpleNamespace(net_forces_w=torch.cat((foot_forces, non_foot_forces), dim=1))
    )
    robot = types.SimpleNamespace(
        device="cpu",
        data=types.SimpleNamespace(
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
            root_pos_w=torch.tensor([[0.0, 0.0, 0.31]]).repeat(4, 1),
            root_lin_vel_b=torch.zeros((4, 3)),
            root_ang_vel_b=torch.zeros((4, 3)),
        )
    )
    env = types.SimpleNamespace(
        scene=_Scene(robot, sensor),
        _g009_terrain_normal_w=torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1),
    )

    stable = _stable_predicate(env, **STABLE_TERM_PARAMS)

    assert stable.tolist() == [True, False, False, False]


class _TerminationManager:
    def __init__(self, value: torch.Tensor):
        self.value = value
        self.requested = None

    def get_term(self, name: str) -> torch.Tensor:
        self.requested = name
        return self.value


def test_success_reward_reuses_termination_result_without_another_latch() -> None:
    env = types.SimpleNamespace(
        step_dt=0.02,
        termination_manager=_TerminationManager(torch.tensor([True, False])),
    )
    reward = stable_success_once(env)
    assert env.termination_manager.requested == "stable_success"
    torch.testing.assert_close(reward, torch.tensor([50.0, 0.0]))
    torch.testing.assert_close(reward * 10.0 * env.step_dt, torch.tensor([10.0, 0.0]))


def test_discounted_potential_difference_survives_reward_manager_dt_scaling() -> None:
    env = types.SimpleNamespace(num_envs=1, device="cpu")
    term = PotentialProgress(cfg=object(), env=env)
    first = term._difference(
        torch.tensor([0.25]), scale=1.0, step_dt=0.02, discount_factor=0.99
    )
    second = term._difference(
        torch.tensor([0.50]), scale=1.0, step_dt=0.02, discount_factor=0.99
    )

    torch.testing.assert_close(first, torch.tensor([12.375]))
    torch.testing.assert_close(second, torch.tensor([12.25]))
    torch.testing.assert_close(first * 2.0 * 0.02, torch.tensor([0.495]))
    torch.testing.assert_close(second * 2.0 * 0.02, torch.tensor([0.49]))


def test_discounted_potential_cycle_has_zero_discounted_return() -> None:
    env = types.SimpleNamespace(num_envs=1, device="cpu")
    term = PotentialProgress(cfg=object(), env=env)
    term._difference(torch.tensor([0.0]), 1.0, 0.02, 0.99)
    rise = term._difference(torch.tensor([1.0]), 1.0, 0.02, 0.99) * 2.0 * 0.02
    fall = term._difference(torch.tensor([0.0]), 1.0, 0.02, 0.99) * 2.0 * 0.02

    torch.testing.assert_close(rise + 0.99 * fall, torch.tensor([0.0]), atol=1.0e-6, rtol=0.0)


def test_terminal_transition_forces_potential_to_zero() -> None:
    env = types.SimpleNamespace(
        num_envs=1,
        device="cpu",
        reset_buf=torch.tensor([False]),
    )
    term = PotentialProgress(cfg=object(), env=env)
    first = term._difference(torch.tensor([0.25]), 1.0, 0.02, 0.99)
    env.reset_buf.fill_(True)
    terminal = term._difference(torch.tensor([1.0]), 1.0, 0.02, 0.99)

    torch.testing.assert_close(terminal, torch.tensor([-12.5]))
    torch.testing.assert_close(first + 0.99 * terminal, torch.tensor([0.0]), atol=1.0e-6, rtol=0.0)


def test_multi_step_discounted_potential_return_telescopes_to_zero() -> None:
    gamma = 0.99
    env = types.SimpleNamespace(
        num_envs=1,
        device="cpu",
        reset_buf=torch.tensor([False]),
    )
    term = PotentialProgress(cfg=object(), env=env)
    rates = [
        term._difference(torch.tensor([0.25]), 1.0, 0.02, gamma),
        term._difference(torch.tensor([0.70]), 1.0, 0.02, gamma),
        term._difference(torch.tensor([0.40]), 1.0, 0.02, gamma),
    ]
    env.reset_buf.fill_(True)
    rates.append(term._difference(torch.tensor([0.90]), 1.0, 0.02, gamma))
    discounted_return = sum((gamma**index) * rate for index, rate in enumerate(rates))

    torch.testing.assert_close(
        discounted_return,
        torch.tensor([0.0]),
        atol=1.0e-5,
        rtol=0.0,
    )


def test_stable_success_requires_consecutive_steps_and_resets_selected_envs(monkeypatch) -> None:
    env = types.SimpleNamespace(num_envs=2, device="cpu")
    term = StableSuccess(cfg=object(), env=env)
    states = iter(
        [
            torch.tensor([True, True]),
            torch.tensor([True, False]),
            torch.tensor([True, True]),
            torch.tensor([True, True]),
        ]
    )
    monkeypatch.setattr(RECOVER, "_stable_predicate", lambda *args, **kwargs: next(states))

    assert term(env, required_consecutive_steps=3, **STABLE_TERM_PARAMS).tolist() == [False, False]
    assert term(env, required_consecutive_steps=3, **STABLE_TERM_PARAMS).tolist() == [False, False]
    assert term(env, required_consecutive_steps=3, **STABLE_TERM_PARAMS).tolist() == [True, False]
    term.reset([0])
    assert term(env, required_consecutive_steps=3, **STABLE_TERM_PARAMS).tolist() == [False, False]


def test_stable_success_rejects_nonpositive_dwell() -> None:
    env = types.SimpleNamespace(num_envs=1, device="cpu")
    term = StableSuccess(cfg=object(), env=env)
    with pytest.raises(ValueError, match="positive"):
        term(env, required_consecutive_steps=0, **STABLE_TERM_PARAMS)


def test_undesired_collision_allows_fallen_pivots_and_activates_after_raise() -> None:
    class _Scene:
        def __init__(self, robot, sensor):
            self._robot = robot
            self.sensors = {"contact_forces": sensor}

        def __getitem__(self, name):
            assert name == "robot"
            return self._robot

    forces = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )
    robot = types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.tensor([[0.0, 0.0, 0.10], [0.0, 0.0, 0.30], [0.0, 0.0, 0.40]])))
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=forces))
    env = types.SimpleNamespace(scene=_Scene(robot, sensor))

    reward = undesired_collision(
        env,
        sensor_cfg=RECOVER.SceneEntityCfg("contact_forces", body_ids=[0, 1]),
        contact_force_threshold=1.0,
        min_base_height=0.30,
        asset_cfg=RECOVER.SceneEntityCfg("robot"),
    )

    torch.testing.assert_close(reward, torch.tensor([0.0, 2.0, 1.0]))
