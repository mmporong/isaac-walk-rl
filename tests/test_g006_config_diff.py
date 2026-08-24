import hashlib
import json
import pathlib
import sys
import unittest
import atexit

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaaclab.app import AppLauncher

_APP = AppLauncher({"headless": True}).app
atexit.register(_APP.close)

from isaac_walk_g006.registry import AGENT_ENTRY_POINT, BASELINE_TASK_ID, PUSH_TASK_ID, register_tasks
from isaac_walk_g006.rough_env_cfg import G006RoughBaselineEnvCfg, G006RoughPushCurriculumEnvCfg
from isaaclab.utils.dict import class_to_dict
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


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


class ConfigContractTests(unittest.TestCase):
    def test_registry_is_idempotent_and_uses_string_entrypoints(self):
        import gymnasium as gym

        register_tasks()
        register_tasks()
        for task_id in (BASELINE_TASK_ID, PUSH_TASK_ID):
            spec = gym.spec(task_id)
            self.assertIsInstance(spec.entry_point, str)
            self.assertIsInstance(spec.kwargs["env_cfg_entry_point"], str)
            self.assertEqual(spec.kwargs["rsl_rl_cfg_entry_point"], AGENT_ENTRY_POINT)

    def test_only_push_robot_differs(self):
        baseline = class_to_dict(G006RoughBaselineEnvCfg())
        push = class_to_dict(G006RoughPushCurriculumEnvCfg())
        changes = changed_paths(baseline, push)
        self.assertTrue(changes)
        self.assertTrue(all(path.startswith("events.push_robot") for path in changes), changes)

    def test_manifest_hash_and_logger_contract(self):
        manifest = json.loads((ROOT / "configs" / "g006_rough_push.json").read_text(encoding="utf-8"))
        self.assertEqual(canonical_sha(manifest["training"]["agent_learning_config"]), manifest["training"]["agent_learning_config_sha256"])
        self.assertEqual([v["task"] for v in manifest["variants"]], [BASELINE_TASK_ID, PUSH_TASK_ID])
        expected_terrain = {"mean_level", "p10_level", "p50_level", "p90_level", "low_fraction", "mid_fraction", "high_fraction"}
        self.assertEqual(set(manifest["logger"]["terrain_keys"]), expected_terrain)
        self.assertEqual(len(manifest["evaluation_protocol"]["initial_states"]), 10)
        self.assertEqual(len({tuple(x["root_relative_pos_m"]) for x in manifest["evaluation_protocol"]["initial_states"]}), 10)

    def test_evaluation_success_criteria_are_exact(self):
        manifest = json.loads((ROOT / "configs" / "g006_rough_push.json").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["evaluation_protocol"]["success_criteria"],
            {
                "lin_vel_error_mps_max": 0.30,
                "yaw_rate_error_radps_max": 0.30,
                "roll_abs_rad_max": 0.35,
                "pitch_abs_rad_max": 0.35,
                "consecutive_post_push_samples": 25,
                "recovery_completed_step_start": 201,
                "recovery_completed_step_end": 450,
                "horizon_completed_step": 600,
                "push_injection_completed_steps": 200,
                "base_contact_allowed": False,
                "survival_to_horizon_required": True,
            },
        )
        self.assertNotIn("push_steps", manifest["evaluation_protocol"]["timing"])

    def test_go2_default_root_height_is_contract_value(self):
        self.assertAlmostEqual(float(UNITREE_GO2_CFG.init_state.pos[2]), 0.4, places=8)


if __name__ == "__main__":
    unittest.main()
