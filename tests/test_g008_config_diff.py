import atexit
import pathlib
import sys
import unittest
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaaclab.app import AppLauncher


_APP = AppLauncher({"headless": True}).app
atexit.register(_APP.close)

from isaac_walk_g008.commands import BalancedVelocityCommandCfg
from isaac_walk_g008.contracts import GO2_LEG_BODY_PATTERN
from isaac_walk_g008.env_cfg import (
    G008CommandEnvCfg,
    G008FrictionStage1EnvCfg,
    G008FrictionStage2EnvCfg,
    G008FrictionStage3EnvCfg,
    G008IrregularRoadGeometryEnvCfg,
    G008IrregularRoadStage1EnvCfg,
    G008IrregularRoadTurnAirTimeEnvCfg,
    G008LegMassStage1EnvCfg,
    G008LegMassStage2EnvCfg,
    G008LegMassStage3EnvCfg,
)
from isaac_walk_g008.registry import AGENT_ENTRY_POINT, TASK_ENTRY_POINTS, register_tasks
from isaac_walk_g008.rewards import feet_air_time_turn_aware
from isaaclab.utils.dict import class_to_dict
import torch


def changed_paths(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in left.keys() | right.keys():
            child = f"{prefix}.{key}" if prefix else key
            paths |= changed_paths(left.get(key), right.get(key), child)
        return paths
    if left != right:
        return {prefix}
    return set()


class G008ConfigContractTests(unittest.TestCase):
    def test_registry_is_idempotent_and_string_based(self):
        import gymnasium as gym

        register_tasks()
        register_tasks()
        self.assertEqual(len(TASK_ENTRY_POINTS), 10)
        for task_id, env_entry_point in TASK_ENTRY_POINTS.items():
            spec = gym.spec(task_id)
            self.assertEqual(spec.kwargs["env_cfg_entry_point"], env_entry_point)
            self.assertEqual(spec.kwargs["rsl_rl_cfg_entry_point"], AGENT_ENTRY_POINT)

    def test_command_distribution_uses_direct_yaw_and_balanced_generator(self):
        cfg = G008CommandEnvCfg()
        self.assertIsInstance(cfg.commands.base_velocity, BalancedVelocityCommandCfg)
        self.assertFalse(cfg.commands.base_velocity.heading_command)
        self.assertEqual(cfg.commands.base_velocity.primitive_probability, 0.8)
        self.assertEqual(cfg.commands.base_velocity.resampling_time_range, (4.0, 6.0))

    def test_friction_part_changes_only_material_event(self):
        command = class_to_dict(G008CommandEnvCfg())
        friction = class_to_dict(G008FrictionStage1EnvCfg())
        changes = changed_paths(command, friction)
        self.assertTrue(changes)
        self.assertTrue(all(path.startswith("events.physics_material") for path in changes), changes)
        params = G008FrictionStage2EnvCfg().events.physics_material.params
        self.assertEqual(params["asset_cfg"].body_names, ".*_foot")
        self.assertTrue(params["make_consistent"])

        stage_changes = changed_paths(
            class_to_dict(G008FrictionStage2EnvCfg()),
            class_to_dict(G008FrictionStage3EnvCfg()),
        )
        self.assertEqual(
            stage_changes,
            {
                "events.physics_material.params.static_friction_range",
                "events.physics_material.params.dynamic_friction_range",
            },
        )

    def test_mass_part_adds_only_independent_leg_mass_event(self):
        command = class_to_dict(G008CommandEnvCfg())
        mass = class_to_dict(G008LegMassStage1EnvCfg())
        changes = changed_paths(command, mass)
        self.assertTrue(changes)
        self.assertTrue(all(path.startswith("events.add_leg_mass") for path in changes), changes)
        params = G008LegMassStage2EnvCfg().events.add_leg_mass.params
        self.assertEqual(params["asset_cfg"].body_names, GO2_LEG_BODY_PATTERN)
        self.assertEqual(params["operation"], "scale")
        self.assertEqual(params["distribution"], "uniform")
        self.assertTrue(params["recompute_inertia"])

        stage_changes = changed_paths(
            class_to_dict(G008LegMassStage2EnvCfg()),
            class_to_dict(G008LegMassStage3EnvCfg()),
        )
        self.assertEqual(stage_changes, {"events.add_leg_mass.params.mass_distribution_params"})

    def test_irregular_road_replaces_the_underlay_and_uses_neutral_foot_material(self):
        cfg = G008IrregularRoadStage1EnvCfg()
        self.assertIsNone(cfg.scene.terrain)
        self.assertEqual(
            cfg.scene.height_scanner.mesh_prim_paths,
            ["/World/irregular_road_field/surface"],
        )
        self.assertEqual(cfg.scene.irregular_road_field.collision_group, -1)
        self.assertEqual(cfg.scene.irregular_road_field.spawn.cell_size_m, 0.25)
        self.assertEqual(cfg.scene.irregular_road_field.spawn.static_friction, (0.25, 0.4, 0.6, 0.8))
        self.assertEqual(cfg.events.physics_material.params["asset_cfg"].body_names, ".*_foot")
        self.assertEqual(cfg.events.physics_material.params["static_friction_range"], (1.0, 1.0))
        self.assertIsNone(cfg.curriculum.terrain_levels)

    def test_geometry_stage_changes_only_the_ground_material_buckets(self):
        geometry = G008IrregularRoadGeometryEnvCfg()
        mixed = G008IrregularRoadStage1EnvCfg()
        changes = changed_paths(class_to_dict(geometry), class_to_dict(mixed))
        self.assertEqual(
            changes,
            {
                "scene.irregular_road_field.spawn.static_friction",
                "scene.irregular_road_field.spawn.dynamic_friction",
                "scene.irregular_road_field.spawn.colors_rgb",
            },
        )
        road = geometry.scene.irregular_road_field.spawn
        self.assertEqual(road.static_friction, (0.8,))
        self.assertEqual(road.dynamic_friction, (0.6,))

    def test_turn_air_time_stage_changes_only_one_reward_function_and_parameter(self):
        geometry = class_to_dict(G008IrregularRoadGeometryEnvCfg())
        turn_air = class_to_dict(G008IrregularRoadTurnAirTimeEnvCfg())
        changes = changed_paths(geometry, turn_air)
        self.assertEqual(
            changes,
            {
                "rewards.feet_air_time.func",
                "rewards.feet_air_time.params.yaw_command_threshold",
            },
        )
        term = G008IrregularRoadTurnAirTimeEnvCfg().rewards.feet_air_time
        self.assertEqual(term.func.__name__, "feet_air_time_turn_aware")
        self.assertEqual(term.params["yaw_command_threshold"], 0.1)

    def test_turn_air_time_reward_activates_for_pure_yaw_but_not_zero_command(self):
        class ContactSensorStub:
            data = SimpleNamespace(last_air_time=torch.tensor([[0.7, 0.7]]))

            @staticmethod
            def compute_first_contact(_step_dt):
                return torch.tensor([[1.0, 1.0]])

        class CommandManagerStub:
            command = torch.tensor([[0.0, 0.0, 0.5]])

            @classmethod
            def get_command(cls, _name):
                return cls.command

        env = SimpleNamespace(
            step_dt=0.02,
            scene=SimpleNamespace(sensors={"contact_forces": ContactSensorStub()}),
            command_manager=CommandManagerStub(),
        )
        sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
        pure_yaw_reward = feet_air_time_turn_aware(
            env,
            command_name="base_velocity",
            sensor_cfg=sensor_cfg,
            threshold=0.5,
            yaw_command_threshold=0.1,
        )
        self.assertAlmostEqual(float(pure_yaw_reward.item()), 0.4, places=6)

        CommandManagerStub.command = torch.zeros((1, 3))
        zero_command_reward = feet_air_time_turn_aware(
            env,
            command_name="base_velocity",
            sensor_cfg=sensor_cfg,
            threshold=0.5,
            yaw_command_threshold=0.1,
        )
        self.assertEqual(float(zero_command_reward.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
