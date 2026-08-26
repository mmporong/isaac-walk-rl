import atexit
import pathlib
import sys
import unittest


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
    G008IrregularRoadStage1EnvCfg,
    G008LegMassStage1EnvCfg,
    G008LegMassStage2EnvCfg,
    G008LegMassStage3EnvCfg,
)
from isaac_walk_g008.registry import AGENT_ENTRY_POINT, TASK_ENTRY_POINTS, register_tasks
from isaaclab.utils.dict import class_to_dict


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
        self.assertEqual(len(TASK_ENTRY_POINTS), 8)
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


if __name__ == "__main__":
    unittest.main()
