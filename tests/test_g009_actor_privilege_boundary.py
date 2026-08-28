from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECOVER_SOURCE = ROOT / "src" / "isaac_walk_g009" / "mdp" / "recover.py"


def _attribute_name(node: ast.Attribute) -> str:
    parts = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def test_actor_observation_call_graph_excludes_privileged_simulator_state() -> None:
    tree = ast.parse(RECOVER_SOURCE.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    roots = {
        "body_fixed_range",
        "body_fixed_range_hit_mask",
        "foot_contact_flags",
        "normalized_foot_load",
    }
    reachable = set(roots)
    pending = list(roots)
    while pending:
        current = pending.pop()
        for call in (node for node in ast.walk(functions[current]) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Name) and call.func.id in functions and call.func.id not in reachable:
                reachable.add(call.func.id)
                pending.append(call.func.id)

    forbidden = {
        "root_physx_view",
        "get_masses",
        "root_pos_w",
        "root_quat_w",
        "ray_hits_w",
        "pos_w",
        "quat_w",
        "_g009_terrain_normal_w",
        "_g009_effective_foot_friction",
        "_g009_recover_fall_class",
    }
    observed = {
        _attribute_name(node)
        for name in reachable
        for node in ast.walk(functions[name])
        if isinstance(node, ast.Attribute)
    }
    violations = sorted(
        attribute
        for attribute in observed
        if any(part in forbidden for part in attribute.split("."))
    )
    assert violations == []

    actor_attributes = {
        attribute
        for name in reachable
        for node in ast.walk(functions[name])
        if isinstance(node, ast.Attribute)
        for attribute in [_attribute_name(node)]
    }
    assert "data.output" in actor_attributes
