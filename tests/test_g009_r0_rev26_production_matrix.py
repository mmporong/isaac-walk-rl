from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs/g009_r0_rev26_qualification.json"


def test_rev26_preregistration_freezes_training_and_evaluation_contract():
    value = json.loads(PREREG.read_text(encoding="utf-8"))
    assert value["task"] == "Isaac-G009-Recover-Flat-Go2-R0-Matrix-v0"
    assert (value["seed"], value["num_envs"], value["num_steps_per_env"]) == (42, 1024, 24)
    assert value["max_iterations"] == 300
    assert value["optimizer_mini_batch_updates"] == 300 * 5 * 4 == 6000
    assert value["expected_checkpoint_name"] == "model_299.pt"
    assert value["training"] == {
        "task": value["task"],
        "seed": 42,
        "headless": True,
        "scratch": True,
        "num_envs": 1024,
        "num_steps_per_env": 24,
        "max_iterations": 300,
        "ppo_num_learning_epochs": 5,
        "ppo_num_mini_batches": 4,
        "optimizer_mini_batch_updates": 6000,
        "expected_checkpoint_name": "model_299.pt",
    }
    assert value["evaluation"] == {
        "seed": 1042,
        "num_envs": 1024,
        "poses": ["prone", "supine", "left_side", "right_side"],
        "environments_per_pose": 256,
        "actor_corruption_enabled": True,
        "minimum_success_rate_per_pose": 0.8,
        "minimum_successes_per_pose": 205,
        "maximum_median_recovery_time_seconds": 4.0,
        "maximum_safety_terminations": 0,
        "checkpoint_name": "model_299.pt",
    }


def test_rev26_source_manifest_and_matrix_contract_are_canonical():
    value = json.loads(PREREG.read_text(encoding="utf-8"))
    paths = value["source_binding_paths"]
    assert len(paths) == 16
    assert paths == sorted(paths)
    assert value["source_binding_path_manifest_sha256"] == hashlib.sha256(
        json.dumps(paths, separators=(",", ":")).encode()
    ).hexdigest()
    assert value["policy_observation"]["total_dimension"] == 140
    assert value["policy_observation"]["collect_gate_telemetry"] is False
    assert value["critic_observation"] == {
        "actor_prefix_dimension": 140,
        "privileged_suffix_dimension": 24,
        "total_dimension": 164,
        "actor_prefix_exact_order_required": True,
    }
    assert value["claim_limits"]["production_training_qualified"] is False
    assert value["claim_limits"]["recovery_success"] is False


def test_production_sanitizer_contains_no_host_sync_calls():
    tree = ast.parse(
        (ROOT / "src/isaac_walk_g009/matrix_gate01.py").read_text(encoding="utf-8")
    )
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_sanitize_production_projection"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "item"
        for node in ast.walk(function)
    )
