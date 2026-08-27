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

from isaac_walk_g009.env_cfg import G009SlopeWalkEnvCfg
from isaac_walk_g009.mdp import reset_root_state_on_slope
from isaac_walk_g009.registry import AGENT_ENTRY_POINT, TASK_ENTRY_POINTS, register_tasks
from isaac_walk_g009.sim_terrain import (
    G009_SLOPE_FIELD_PRIM,
    G009_SLOPE_SURFACE_PRIM,
    SlopeTerrainFieldCfg,
    spawn_slope_terrain,
)


class G009ConfigContractTests(unittest.TestCase):
    def test_registry_is_idempotent_and_uses_string_entry_points(self):
        import gymnasium as gym

        register_tasks()
        register_tasks()
        self.assertEqual(len(TASK_ENTRY_POINTS), 1)
        for task_id, env_cfg_entry_point in TASK_ENTRY_POINTS.items():
            spec = gym.spec(task_id)
            self.assertEqual(spec.kwargs["env_cfg_entry_point"], env_cfg_entry_point)
            self.assertEqual(spec.kwargs["rsl_rl_cfg_entry_point"], AGENT_ENTRY_POINT)

    def test_s0_replaces_inherited_terrain_and_disturbances(self):
        cfg = G009SlopeWalkEnvCfg()
        self.assertIsNone(cfg.scene.terrain)
        self.assertIsNone(cfg.curriculum.terrain_levels)
        self.assertIsNone(cfg.events.push_robot)
        self.assertIsNone(cfg.events.add_base_mass)
        self.assertIsNone(cfg.events.base_external_force_torque)
        self.assertEqual(cfg.scene.slope_field.prim_path, G009_SLOPE_FIELD_PRIM)
        self.assertEqual(cfg.scene.height_scanner.mesh_prim_paths, [G009_SLOPE_SURFACE_PRIM])
        self.assertTrue(cfg.scene.contact_forces.track_pose)

    def test_s0_ground_and_foot_material_contract(self):
        cfg = G009SlopeWalkEnvCfg()
        ground = cfg.scene.slope_field.spawn
        self.assertIs(ground.func, spawn_slope_terrain)
        self.assertEqual(ground.static_friction, (0.8,))
        self.assertEqual(ground.dynamic_friction, (0.6,))
        foot = cfg.events.physics_material.params
        self.assertEqual(foot["asset_cfg"].body_names, ".*_foot")
        self.assertEqual(foot["static_friction_range"], (1.0, 1.0))
        self.assertEqual(foot["dynamic_friction_range"], (1.0, 1.0))
        self.assertTrue(foot["make_consistent"])

    def test_spawner_creates_one_collision_and_ray_cast_surface(self):
        import isaacsim.core.utils.stage as stage_utils
        from pxr import UsdGeom, UsdPhysics, UsdShade

        stage_utils.create_new_stage()
        cfg = SlopeTerrainFieldCfg(
            x_min_m=-1.0,
            x_max_m=1.0,
            y_min_m=-1.0,
            y_max_m=1.0,
            cell_size_m=0.5,
            slope_deg=10.0,
        )
        spawn_slope_terrain(G009_SLOPE_FIELD_PRIM, cfg)
        stage = stage_utils.get_current_stage()
        surface = stage.GetPrimAtPath(G009_SLOPE_SURFACE_PRIM)
        mesh_paths = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
        self.assertEqual(mesh_paths, [G009_SLOPE_SURFACE_PRIM])
        self.assertTrue(surface.HasAPI(UsdPhysics.CollisionAPI))
        self.assertTrue(surface.HasAPI(UsdPhysics.MeshCollisionAPI))
        material_path = UsdShade.MaterialBindingAPI(surface).GetDirectBinding("physics").GetMaterialPath()
        material = stage.GetPrimAtPath(material_path)
        self.assertAlmostEqual(material.GetAttribute("physics:staticFriction").Get(), 0.8, places=6)
        self.assertAlmostEqual(material.GetAttribute("physics:dynamicFriction").Get(), 0.6, places=6)
        self.assertEqual(material.GetAttribute("physxMaterial:frictionCombineMode").Get(), "multiply")

    def test_slope_controls_are_exposed_on_the_scene_cfg(self):
        ground = G009SlopeWalkEnvCfg().scene.slope_field.spawn
        ground.slope_deg = 20.0
        ground.azimuth_deg = 90.0
        ground.seed = 7
        ground.residual_amplitude_m = 0.025
        self.assertEqual(
            (ground.slope_deg, ground.azimuth_deg, ground.seed, ground.residual_amplitude_m),
            (20.0, 90.0, 7, 0.025),
        )

    def test_reset_uses_slope_event_and_records_zero_velocity_by_default(self):
        term = G009SlopeWalkEnvCfg().events.reset_base
        self.assertIs(term.func, reset_root_state_on_slope)
        self.assertEqual(term.params["terrain_cfg_name"], "slope_field")
        self.assertEqual(
            term.params["velocity_range"],
            {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")},
        )

    def test_reset_places_and_aligns_robot_and_records_velocity(self):
        import torch

        class RobotStub:
            device = "cpu"
            data = SimpleNamespace(
                default_root_state=torch.tensor(
                    [[0.0, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                )
            )

            def write_root_pose_to_sim(self, pose, env_ids):
                self.pose = pose
                self.pose_env_ids = env_ids

            def write_root_velocity_to_sim(self, velocity, env_ids):
                self.velocity = velocity
                self.velocity_env_ids = env_ids

        class SceneStub:
            env_origins = torch.tensor([[0.5, -0.25, 0.0]], dtype=torch.float32)

            def __init__(self, robot):
                self.robot = robot

            def __getitem__(self, name):
                return getattr(self, name)

        robot = RobotStub()
        spawn_cfg = SlopeTerrainFieldCfg(
            x_min_m=-2.0,
            x_max_m=2.0,
            y_min_m=-2.0,
            y_max_m=2.0,
            cell_size_m=0.5,
            slope_deg=20.0,
            azimuth_deg=45.0,
        )
        env = SimpleNamespace(
            scene=SceneStub(robot),
            cfg=SimpleNamespace(scene=SimpleNamespace(slope_field=SimpleNamespace(spawn=spawn_cfg))),
            extras={},
        )
        env_ids = torch.tensor([0], dtype=torch.long)
        zero_pose = {axis: (0.0, 0.0) for axis in ("x", "y", "z", "yaw")}
        fixed_velocity = {
            "x": (0.1, 0.1),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.2, 0.2),
        }
        reset_root_state_on_slope(env, env_ids, zero_pose, fixed_velocity)

        expected_plane_z = (0.5 - 0.25) * (2.0**-0.5) * torch.tan(torch.deg2rad(torch.tensor(20.0)))
        self.assertAlmostEqual(float(robot.pose[0, 2]), float(expected_plane_z + 0.42), places=5)
        self.assertTrue(torch.allclose(robot.velocity[0], torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.2])))
        from isaaclab.utils import math as math_utils

        body_up = math_utils.quat_apply(robot.pose[:, 3:7], torch.tensor([[0.0, 0.0, 1.0]]))
        self.assertTrue(torch.allclose(body_up, env.extras["g009_reset"]["support_normal_w"], atol=1.0e-5))
        self.assertTrue(torch.equal(env.extras["g009_reset"]["env_ids"], env_ids))
        self.assertTrue(torch.equal(env.extras["g009_reset"]["root_velocity_w"], robot.velocity))


if __name__ == "__main__":
    unittest.main()
