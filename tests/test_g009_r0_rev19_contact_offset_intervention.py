from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts/probe_g009_r0_rev19_contact_offset_intervention.py"
SPEC = importlib.util.spec_from_file_location("g009_rev19_probe", PROBE_PATH)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def topology_fixture() -> dict:
    template = [f"link_{index:02d}/collision" for index in range(27)]
    return {
        "source": "read_only_usd_collision_api_traversal",
        "traversal_predicate": "Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies())",
        "instance_proxy_traversal_enabled": True,
        "robot_root_paths": [f"/World/envs/env_{env}/Robot" for env in range(8)],
        "per_env_robot_root_instance_state": {
            str(env): {
                "path": f"/World/envs/env_{env}/Robot",
                "is_instance": False,
                "is_instanceable": False,
                "is_instance_proxy": False,
            }
            for env in range(8)
        },
        "robot_root_scope_validated": True,
        "collision_shapes_per_articulation": 27,
        "collision_shape_paths_total": 8 * 27,
        "sorted_unique_template_paths": template,
        "per_env_sorted_paths": {
            str(env): [f"/World/envs/env_{env}/Robot/{path}" for path in template]
            for env in range(8)
        },
        "per_env_sorted_path_records": {
            str(env): [
                {
                    "path": f"/World/envs/env_{env}/Robot/{path}",
                    "is_instance_proxy": env > 0,
                }
                for path in template
            ]
            for env in range(8)
        },
        "per_env_collision_path_counts": {str(env): 27 for env in range(8)},
        "per_env_unique_collision_path_counts": {str(env): 27 for env in range(8)},
        "per_env_instance_proxy_collision_path_counts": {
            str(env): 0 if env == 0 else 27 for env in range(8)
        },
        "instance_proxy_collision_path_count": 7 * 27,
        "all_envs_topology_identical": True,
        "setter_call_scope": "robot.root_physx_view_only",
        "ground_setter_called": False,
        "usd_schema_apply_called": False,
        "ground_runtime_offset_unchanged_claimed": False,
        "tensor_column_path_mapping_authority": False,
    }


class FakeCollisionPrim:
    def __init__(self, path: str, *, collision: bool = True, instance_proxy: bool = False) -> None:
        self.path = path
        self.collision = collision
        self.instance_proxy = instance_proxy

    def GetPath(self) -> str:
        return self.path

    def HasAPI(self, _api: object) -> bool:
        return self.collision

    def IsInstanceProxy(self) -> bool:
        return self.instance_proxy


def collision_prims() -> list[FakeCollisionPrim]:
    prims = [
        FakeCollisionPrim(
            f"/World/envs/env_{env}/Robot/link_{shape:02d}/collision",
            instance_proxy=env > 0,
        )
        for env in range(8)
        for shape in range(27)
    ]
    prims.extend(
        [
            FakeCollisionPrim("/World/ground", collision=True),
            FakeCollisionPrim("/World/envs/env_0/Robot/debug_visual", collision=False),
        ]
    )
    random.Random(42).shuffle(prims)
    return prims


def test_collision_topology_uses_instance_proxy_predicate_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicate = object()

    class FakeStage:
        def __init__(self) -> None:
            self.requested_root_paths: list[str] = []

        def GetPrimAtPath(self, path: str):
            self.requested_root_paths.append(path)
            return types.SimpleNamespace(
                path=path,
                IsValid=lambda: True,
                IsInstance=lambda: False,
                IsInstanceable=lambda: False,
                IsInstanceProxy=lambda: False,
            )

    stage = FakeStage()
    all_prims = collision_prims()

    def prim_range(root: Any, value: object):
        assert value is predicate
        prefix = f"{root.path}/"
        return [prim for prim in all_prims if prim.path.startswith(prefix)]

    usd_module = types.SimpleNamespace(
        TraverseInstanceProxies=lambda: predicate,
        PrimRange=prim_range,
    )
    usd_physics_module = types.SimpleNamespace(CollisionAPI=object())
    omni_usd_module = types.ModuleType("omni.usd")
    setattr(omni_usd_module, "get_context", lambda: types.SimpleNamespace(get_stage=lambda: stage))
    omni_module = types.ModuleType("omni")
    setattr(omni_module, "usd", omni_usd_module)
    pxr_module = types.ModuleType("pxr")
    setattr(pxr_module, "Usd", usd_module)
    setattr(pxr_module, "UsdPhysics", usd_physics_module)
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd_module)
    monkeypatch.setitem(sys.modules, "pxr", pxr_module)

    result = PROBE.collision_topology_evidence(SimpleNamespace())

    assert stage.requested_root_paths == [f"/World/envs/env_{env}/Robot" for env in range(8)]
    assert result["collision_shape_paths_total"] == 216
    assert result["per_env_collision_path_counts"] == {str(env): 27 for env in range(8)}
    assert result["per_env_instance_proxy_collision_path_counts"] == {
        str(env): 0 if env == 0 else 27 for env in range(8)
    }
    assert result["sorted_unique_template_paths"] == sorted(result["sorted_unique_template_paths"])


def test_collision_topology_fails_closed_with_observed_counts() -> None:
    prims = collision_prims()
    prims = [prim for prim in prims if prim.path != "/World/envs/env_3/Robot/link_26/collision"]

    with pytest.raises(ValueError, match=r'observed_counts=.*"3": 26'):
        PROBE._collision_topology_from_prims(prims, object())


def test_collision_topology_fails_closed_on_duplicate_path() -> None:
    prims = collision_prims()
    prims.append(FakeCollisionPrim("/World/envs/env_2/Robot/link_00/collision", instance_proxy=True))

    with pytest.raises(ValueError, match=r'observed_counts=.*"2": 28'):
        PROBE._collision_topology_from_prims(prims, object())


def test_collision_topology_fails_closed_on_cross_env_template_drift() -> None:
    prims = collision_prims()
    target = next(prim for prim in prims if prim.path == "/World/envs/env_6/Robot/link_26/collision")
    target.path = "/World/envs/env_6/Robot/link_drift/collision"

    with pytest.raises(ValueError, match="collision topology differs across envs; env_index=6"):
        PROBE._collision_topology_from_prims(prims, object())


@pytest.mark.parametrize(
    "tamper",
    ["forged_prefix", "serialized_template", "stale_count", "proxy_redistribution"],
)
def test_offset_validator_recomputes_topology_relationships(tamper: str) -> None:
    value = offset_fixture("A")
    topology = value["topology"]
    if tamper == "forged_prefix":
        forged = [f"/forged/not-env3/path_{index:02d}" for index in range(27)]
        topology["per_env_sorted_paths"]["3"] = forged
        topology["per_env_sorted_path_records"]["3"] = [
            {"path": path, "is_instance_proxy": True} for path in forged
        ]
    elif tamper == "serialized_template":
        topology["sorted_unique_template_paths"][0] = "unrelated/collision"
        topology["sorted_unique_template_paths"].sort()
    elif tamper == "stale_count":
        topology["per_env_collision_path_counts"]["4"] = 26
        topology["collision_shape_paths_total"] = 215
    elif tamper == "proxy_redistribution":
        topology["per_env_instance_proxy_collision_path_counts"]["0"] = 1
        topology["per_env_instance_proxy_collision_path_counts"]["1"] = 26

    with pytest.raises(ValueError):
        PROBE.validate_offset_integrity(value, "A")


def offset_fixture(arm: str = "A") -> dict:
    import torch

    baseline = torch.linspace(0.0004905, 0.00188, 27).repeat(8, 1)
    rest = torch.zeros((8, 27))

    class View:
        def __init__(self) -> None:
            self.contact = baseline.clone()
            self.rest = rest.clone()
            self.contact_calls = 0

        def get_contact_offsets(self):
            return self.contact

        def get_rest_offsets(self):
            return self.rest

        def set_contact_offsets(self, value, env_ids):
            assert env_ids.tolist() == list(range(8))
            self.contact_calls += 1
            self.contact = value.clone()

        def set_rest_offsets(self, *_args):
            raise AssertionError("rest setter forbidden")

    view = View()
    value = PROBE.apply_contact_offset_intervention(view, arm, torch)
    value["topology"] = topology_fixture()
    assert view.contact_calls == 1
    return value


def source_bundle_fixture() -> dict:
    files = {path: hashlib.sha256(path.encode()).hexdigest() for path in PROBE.SOURCE_BINDING_PATHS}
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return {
        "schema_version": 1,
        "git_commit": "1" * 40,
        "git_commit_valid": True,
        "source_binding_paths": list(PROBE.SOURCE_BINDING_PATHS),
        "source_binding_files": files,
        "source_bundle_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "all_files_present": True,
        "missing_files": [],
        "clean": True,
        "dirty_source_paths": [],
    }


def report_fixture(arm: str = "A", device: str = "cpu", replicate: int = 1) -> dict:
    label = "cpu" if device == "cpu" else "gpu"
    source = ROOT / "reports/runs" / f"g009_r0_rev18_raw_contact_{label}_rep01_s42.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    report.update(
        {
            "schema_version": PROBE.SCHEMA_VERSION,
            "experiment_id": "G009-5-E012",
            "revision": "rev19",
            "arm": arm,
            "device": device,
            "replicate_index": replicate,
            "execution": {
                "execution_id": uuid.uuid4().hex,
                "started_at_utc": "2026-08-30T00:00:00.000000Z",
                "output_path_repo_relative": PROBE.expected_output_relative(arm, device, replicate),
                "no_overwrite": True,
            },
            "preregistration": PROBE.probe_contract(arm, device, replicate)["preregistration"],
            "predecessor": {"path": PROBE.PREDECESSOR_PATH.relative_to(ROOT).as_posix(), "sha256": PROBE.PREDECESSOR_SHA256},
            "source_bundle": source_bundle_fixture(),
            "cpu_preflight_binding": PROBE.cpu_preflight_not_required_binding() if device == "cpu" else {
                "status": "validated_for_gpu",
                "path": PROBE.CPU_PREFLIGHT_PATH.relative_to(ROOT).as_posix(),
                "sha256": "3" * 64,
                "git_commit": "1" * 40,
                "probe_source_bundle_sha256": source_bundle_fixture()["source_bundle_sha256"],
            },
            "governance": PROBE.governance(),
            "offset_integrity": offset_fixture(arm),
            "manual_probe_safety": {},
        }
    )
    body_names = report["live_physics_readback"]["max_depenetration_velocity"]["authoritative_body_names"]
    import torch
    mass_tensor = torch.ones((8, 19))
    mass_snapshot = PROBE.tensor_snapshot(mass_tensor)
    body_names_sha = hashlib.sha256(json.dumps(body_names, separators=(",", ":")).encode()).hexdigest()
    body_weight = 19.0 * 9.81
    report["manual_probe_safety"] = {
                "label": "manual_probe_observation_not_gate",
                "required_scopes": ["all_envs", "source_env_7"],
                "thresholds": {
                    "hard_joint_limit_margin_rad": 0.01,
                    "non_foot_peak_force_body_weight_max": 15.0,
                    "cpu_raw_minimum_separation_m": -0.01,
                },
                "observations": {
                    "sample_count": 150,
                    "finite_violation_steps": {"all_envs": [], "source_env_7": []},
                    "hard_joint_limit_violation_steps": {"all_envs": [], "source_env_7": []},
                    "non_foot_peak_force_body_weight_per_env": {str(index): 1.0 for index in range(8)},
                    "non_foot_peak_force_n_per_env": {str(index): body_weight for index in range(8)},
                    "all_env_non_foot_peak_force_body_weight": 1.0,
                    "source_env_non_foot_peak_force_body_weight": 1.0,
                    "cpu_raw_minimum_separation_m": {
                        "per_env": {str(index): -0.001 for index in range(8)},
                        "all_env_minimum": -0.001,
                        "source_env_7_minimum": -0.001,
                    },
                    "error": None,
                },
                "checks": {
                    "exact_150_manual_samples": True,
                    "finite_joint_position_and_contact_force": True,
                    "hard_joint_limit_with_margin": True,
                    "all_env_non_foot_peak_force_within_15_bw": True,
                    "source_env_non_foot_peak_force_within_15_bw": True,
                    "cpu_raw_minimum_separation_observed": True,
                    "cpu_raw_minimum_separation_within_limit": True,
                    "default_mass_8x19_finite_positive_unchanged": True,
                    "collection_error_absent": True,
                },
                "mass_evidence": {
                    "source": "robot.data.default_mass",
                    "shape": [8, 19],
                    "body_names": body_names,
                    "body_names_sha256": body_names_sha,
                    "tensor": mass_snapshot,
                    "per_env_total_mass_kg": [19.0] * 8,
                    "per_env_body_weight_n": [body_weight] * 8,
                    "unchanged_for_150_steps": True,
                    "changed_steps": [],
                },
                "available": True,
                "passed": True,
            }
    for row in report["live_physics_readback"]["solver"]["articulations"]:
        row["solver_position_iteration_count"] = 8
        row["solver_velocity_iteration_count"] = 0
    contract = PROBE.probe_contract(arm, device, replicate)
    report["contract"] = contract
    report["contract_sha256"] = PROBE.canonical_sha256(contract)
    report["feasibility"] = PROBE.derive_feasibility(report)
    return report


def test_preregistration_locks_new_control_and_measured_topology() -> None:
    value = PROBE.load_preregistration()
    assert value["canonical_slots"] == [
        "A.cpu.rep1", "A.cpu.rep2", "B.cpu.rep1", "B.cpu.rep2",
        "A.cuda:0.rep1", "A.cuda:0.rep2", "B.cuda:0.rep1", "B.cuda:0.rep2",
    ]
    assert value["design"]["solver_position_iterations"] == 8
    assert value["design"]["arms"]["A"]["contact_offset_setter_called"] is True
    assert value["design"]["arms"]["B"]["contact_offset_setter_called"] is True
    assert value["design"]["rest_offset_setter_called"] is False
    assert value["measured_shape_topology"]["tensor_column_labels"] == "shape_index_00..shape_index_26"
    assert value["measured_shape_topology"]["tensor_column_path_mapping_authority"] is False
    assert value["measured_shape_topology"]["collision_path_inventory_traversal"] == "Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies())"
    assert value["measured_shape_topology"]["collision_path_inventory_read_only"] is True
    assert value["measured_shape_topology"]["collision_path_inventory_can_map_tensor_columns"] is False


@pytest.mark.parametrize(("arm", "scale"), [("A", 1.0), ("B", 1.5)])
def test_both_arms_use_symmetric_setter_with_only_scale_difference(arm: str, scale: float) -> None:
    value = offset_fixture(arm)
    assert value["contact_offset_setter_called"] is True
    assert value["rest_offset_setter_called"] is False
    assert value["contact_offset_scale"] == scale
    assert value["before"]["contact_offset"]["shape"] == [8, 27]
    assert value["before"]["contact_offset"]["minimum"] < value["before"]["contact_offset"]["maximum"]
    assert value["before"]["rest_offset"]["minimum"] == value["before"]["rest_offset"]["maximum"] == 0.0
    assert all(value["checks"].values())
    PROBE.validate_offset_integrity(value, arm)


def test_startup_event_records_env_and_never_mutates_ground_or_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    baseline = torch.linspace(0.0005, 0.0018, 27).repeat(8, 1)

    class View:
        def __init__(self) -> None:
            self.contact = baseline.clone()
            self.rest = torch.zeros_like(baseline)

        def get_contact_offsets(self): return self.contact
        def get_rest_offsets(self): return self.rest
        def set_contact_offsets(self, value, _ids): self.contact = value.clone()
        def set_rest_offsets(self, *_args): pytest.fail("rest setter forbidden")

    env = SimpleNamespace(scene={"robot": SimpleNamespace(root_physx_view=View())})
    monkeypatch.setattr(PROBE, "collision_topology_evidence", lambda _env: topology_fixture())
    sink: dict = {}
    monkeypatch.setattr(PROBE, "_STARTUP_EVIDENCE_SINK", sink)
    PROBE.startup_scale_contact_offsets(env, None, "B")
    assert env._g009_rev19_contact_offset_evidence == sink
    assert sink["topology"]["setter_call_scope"] == "robot.root_physx_view_only"
    assert sink["topology"]["ground_setter_called"] is False
    assert sink["topology"]["ground_runtime_offset_unchanged_claimed"] is False


def test_contract_and_output_names_keep_rev19_arms_separate() -> None:
    a = PROBE.probe_contract("A", "cpu", 1)
    b = PROBE.probe_contract("B", "cpu", 1)
    assert a["controlled_cell"]["solver_position_iterations"] == b["controlled_cell"]["solver_position_iterations"] == 8
    assert a["controlled_cell"]["contact_offset_setter_called"] is b["controlled_cell"]["contact_offset_setter_called"] is True
    assert a["controlled_cell"]["contact_offset_scale"] == 1.0
    assert b["controlled_cell"]["contact_offset_scale"] == 1.5
    assert a["comparison_authority"]["rev18_solver_16_as_control_allowed"] is False
    assert PROBE.expected_output_relative("B", "cuda:0", 2).endswith("armB_gpu_rep02_s42.json")


def test_safety_uses_hard_limits_and_same_device_mass_readback() -> None:
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    forces = torch.zeros((8, 19, 3), device=device)
    forces[:, 0, 2] = 9.81
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces), body_names=["base"] + [f"link_{i}" for i in range(14)] + [f"foot_{i}" for i in range(4)])
    joint_pos = torch.zeros((8, 12), device=device)
    hard = torch.stack((torch.full_like(joint_pos, -1.0), torch.full_like(joint_pos, 1.0)), dim=-1)
    robot = SimpleNamespace(body_names=sensor.body_names, data=SimpleNamespace(joint_pos=joint_pos, joint_pos_limits=hard, soft_joint_pos_limits=None, default_mass=torch.ones((8, 19), device=device)))
    accumulator = PROBE.SafetyAccumulator()
    accumulator.observe(sensor, robot, torch)
    assert accumulator.error is None
    assert accumulator.sample_count == 1
    assert accumulator.hard_limit_with_margin is True


def test_cpu_safety_records_all_env_and_source_minima() -> None:
    import torch

    accumulator = PROBE.SafetyAccumulator()
    names = ["base"] + [f"link_{i}" for i in range(14)] + [f"foot_{i}" for i in range(4)]
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros((8, 19, 3))), body_names=names)
    positions = torch.zeros((8, 12))
    limits = torch.stack((torch.full_like(positions, -1.0), torch.full_like(positions, 1.0)), dim=-1)
    robot = SimpleNamespace(body_names=names, data=SimpleNamespace(joint_pos=positions, joint_pos_limits=limits, default_mass=torch.ones((8, 19))))
    for _ in range(150):
        accumulator.observe(sensor, robot, torch)
    raw = {
        "events": [
            {
                "headers": [
                    {"env_index": index, "contact_points": [{"separation_m": -0.001 - index * 0.0001}]}
                    for index in range(8)
                ]
            }
        ]
    }
    value = accumulator.snapshot(raw, "cpu")
    separation = value["observations"]["cpu_raw_minimum_separation_m"]
    assert set(separation["per_env"]) == {str(index) for index in range(8)}
    assert separation["source_env_7_minimum"] == pytest.approx(-0.0017)
    assert separation["all_env_minimum"] == pytest.approx(-0.0017)
    assert value["available"] is True and value["passed"] is True


def test_cpu_safety_fails_closed_when_any_env_separation_missing() -> None:
    import torch

    accumulator = PROBE.SafetyAccumulator()
    names = ["base"] + [f"link_{i}" for i in range(14)] + [f"foot_{i}" for i in range(4)]
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros((8, 19, 3))), body_names=names)
    positions = torch.zeros((8, 12))
    limits = torch.stack((torch.full_like(positions, -1.0), torch.full_like(positions, 1.0)), dim=-1)
    robot = SimpleNamespace(body_names=names, data=SimpleNamespace(joint_pos=positions, joint_pos_limits=limits, default_mass=torch.ones((8, 19))))
    for _ in range(150):
        accumulator.observe(sensor, robot, torch)
    raw = {"events": [{"headers": [{"env_index": 7, "contact_points": [{"separation_m": 0.0}]}]}]}
    value = accumulator.snapshot(raw, "cpu")
    assert value["available"] is False
    assert value["passed"] is None
    assert value["checks"]["cpu_raw_minimum_separation_observed"] is False


def test_validate_report_recomputes_offset_solver_and_governance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PROBE, "validate_source_bundle", lambda value: value)
    report = report_fixture()
    derived = PROBE.validate_report(report)
    assert derived["offset_integrity_passed"] is True
    assert derived["solver_live_readback_8_0"] is True
    assert derived["manual_probe_safety_available"] is True
    assert derived["manual_probe_safety_passed"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report: report["offset_integrity"]["checks"].update(rest_bitwise_unchanged=False), "offset integrity"),
        (lambda report: report["offset_integrity"]["topology"].update(ground_setter_called=True), "topology"),
        (lambda report: report["live_physics_readback"]["solver"]["articulations"][0].update(solver_position_iteration_count=16), "serialized feasibility"),
        (lambda report: report["governance"].update(ppo_updates=1), "governance"),
    ],
)
def test_validate_report_fails_closed(monkeypatch: pytest.MonkeyPatch, mutate, message: str) -> None:
    monkeypatch.setattr(PROBE, "validate_source_bundle", lambda value: value)
    report = report_fixture()
    mutate(report)
    with pytest.raises(ValueError, match=message):
        PROBE.validate_report(report)


def test_source_bundle_validates_exact_committed_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    value = source_bundle_fixture()
    monkeypatch.setattr(PROBE, "committed_blob_sha256", lambda path, _commit: value["source_binding_files"][path])
    assert PROBE.validate_source_bundle(value) == value
    changed = copy.deepcopy(value)
    changed["source_binding_paths"] = list(reversed(changed["source_binding_paths"]))
    with pytest.raises(ValueError, match="path order"):
        PROBE.validate_source_bundle(changed)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report: report.update(pose_action_assignment={"class_ids": [0] * 8}), "pose assignment"),
        (lambda report: report.update(finished_at_utc="not-utc"), "finished timestamp"),
        (lambda report: report["manual_inner_loop"].update(action_process_count=37), "manual inner-loop"),
        (lambda report: report["device_readback"].update(runtime_device="cuda:0"), "device readback"),
        (lambda report: report["live_physics_readback"]["max_depenetration_velocity"]["articulations"][0]["links"][0].update(max_depenetration_velocity_m_s=0.75), "max-depenetration"),
    ],
)
def test_validate_report_rejects_pose_time_manual_device_and_max_dep(monkeypatch: pytest.MonkeyPatch, mutate, message: str) -> None:
    monkeypatch.setattr(PROBE, "validate_source_bundle", lambda value: value)
    report = report_fixture()
    mutate(report)
    with pytest.raises(ValueError, match=message):
        PROBE.validate_report(report)


def test_mass_snapshot_and_body_weight_denominator_are_hash_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PROBE, "validate_source_bundle", lambda value: value)
    report = report_fixture()
    report["manual_probe_safety"]["observations"]["non_foot_peak_force_n_per_env"]["0"] += 1.0
    with pytest.raises(ValueError, match="body-weight normalization"):
        PROBE.validate_report(report)
    report = report_fixture()
    report["manual_probe_safety"]["mass_evidence"]["body_names"].reverse()
    with pytest.raises(ValueError, match="body ordering hash"):
        PROBE.validate_report(report)


def test_base_validator_is_not_replaced_and_output_path_patch_is_restored() -> None:
    source = PROBE_PATH.read_text(encoding="utf-8")
    assert "base_probe.validate_report =" not in source
    assert "base_probe.expected_output_relative = lambda" in source
    assert '"expected_output_relative": base_probe.expected_output_relative' in source
    assert "for name, value in saved.items():" in source
    assert "setattr(base_probe, name, value)" in source


def preflight_artifact_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runs = tmp_path / "reports" / "runs"
    runs.mkdir(parents=True)
    preflight_path = runs / "g009_r0_rev19_contact_offset_cpu_preflight_2x2_s42.json"
    source_bundle = source_bundle_fixture()
    bindings = []
    reports = []
    for arm, replicate in (("A", 1), ("A", 2), ("B", 1), ("B", 2)):
        report = report_fixture(arm, "cpu", replicate)
        reports.append(report)
        path = tmp_path / PROBE.expected_output_relative(arm, "cpu", replicate)
        raw = json.dumps(report, sort_keys=True).encode()
        path.write_bytes(raw)
        bindings.append({"path": path.relative_to(tmp_path).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()})
    synthesis_files = {path: hashlib.sha256(path.encode()).hexdigest() for path in PROBE.SYNTHESIS_SOURCE_BINDING_PATHS}
    synthesis_payload = "\n".join(f"{path}:{synthesis_files[path]}" for path in sorted(synthesis_files))
    synthesis_bundle = {"schema_version": 1, "role": "offline_synthesis_implementation", "git_commit": source_bundle["git_commit"], "git_commit_valid": True, "source_binding_paths": list(PROBE.SYNTHESIS_SOURCE_BINDING_PATHS), "source_binding_files": synthesis_files, "source_bundle_sha256": hashlib.sha256(synthesis_payload.encode()).hexdigest(), "all_files_present": True, "missing_files": [], "clean": True, "dirty_source_paths": []}
    monkeypatch.setattr(PROBE, "committed_synthesis_blob_sha256", lambda path, _commit: synthesis_files[path])
    rows = {arm: [] for arm in ("A", "B")}
    for report in reports:
        rows[report["arm"]].append(PROBE.repeatability_row(report, report["feasibility"]))
    repeatability = {f"{arm}.cpu": PROBE.cell_repeatability(rows[arm]) for arm in ("A", "B")}
    mass = reports[0]["manual_probe_safety"]["mass_evidence"]
    created = "2026-08-30T00:00:00Z"
    artifact = {
        "schema_version": PROBE.CPU_PREFLIGHT_SCHEMA_VERSION,
        "evidence_id": "G009-5-E012",
        "goal_id": "g009",
        "stage_id": "R0",
        "revision": "rev19",
        "status": "complete",
        "mode": "cpu_preflight_2x2",
        "input_report_count": 4,
        "input_reports": bindings,
        "integrity": {"passed": True, "hash_bound": True, "unique_execution_ids": True, "exact_slots": ["A.cpu.rep1", "A.cpu.rep2", "B.cpu.rep1", "B.cpu.rep2"], "git_commit": source_bundle["git_commit"], "probe_source_bundle_sha256": source_bundle["source_bundle_sha256"], "synthesis_source_bundle_sha256": synthesis_bundle["source_bundle_sha256"], "mass_tensor_sha256": mass["tensor"]["sha256"], "mass_body_names_sha256": mass["body_names_sha256"]},
        "cpu_preflight": {"passed": True, "raw_pass_probe_valid_safety_pass": True, "within_arm_repeatability_passed": True, "gpu_stage_allowed": True},
        "decision": {"outcome": "gpu_stage_authorized", "selected_lever": None, "third_run_majority_vote_allowed": False, "repeatability": repeatability},
        "governance": PROBE.synthesis_governance(),
        "synthesis_source_bundle": synthesis_bundle,
        "created_at_utc": created,
        "execution": {"execution_id": uuid.uuid4().hex, "started_at_utc": created, "output_path_repo_relative": preflight_path.relative_to(tmp_path).as_posix(), "no_overwrite": True},
    }
    monkeypatch.setattr(PROBE, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(PROBE, "CPU_PREFLIGHT_PATH", preflight_path)
    monkeypatch.setattr(PROBE, "validate_report", lambda report, validate_gpu_preflight=False: report["feasibility"])
    return preflight_path, source_bundle, artifact, reports


def write_preflight_fixture(preflight_path: Path, artifact: dict) -> None:
    preflight_path.write_text(json.dumps(artifact), encoding="utf-8")


def test_cpu_preflight_artifact_binds_exact_four_report_hashes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preflight_path, source_bundle, artifact, _reports = preflight_artifact_fixture(monkeypatch, tmp_path)
    write_preflight_fixture(preflight_path, artifact)
    binding = PROBE.validate_cpu_preflight_artifact(preflight_path, source_bundle)
    assert binding["status"] == "validated_for_gpu"
    first_report = tmp_path / artifact["input_reports"][0]["path"]
    first_report.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="input hash"):
        PROBE.validate_cpu_preflight_artifact(preflight_path, source_bundle)


@pytest.mark.parametrize("kind", ["numeric", "structure"])
def test_cpu_preflight_recomputes_repeatability_from_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str) -> None:
    preflight_path, source_bundle, artifact, reports = preflight_artifact_fixture(monkeypatch, tmp_path)
    report = reports[1]
    events = report["raw_contact_observation"]["events"]
    if kind == "numeric":
        point = next(point for event in events for header in event["headers"] if header["env_index"] == 7 for point in header["contact_points"])
        point["position_w_m"][0] += 1.0
    else:
        event = next(event for event in events if any(header["env_index"] == 7 for header in event["headers"]))
        event["headers"] = [header for header in event["headers"] if header["env_index"] != 7]
    report_path = tmp_path / artifact["input_reports"][1]["path"]
    raw = json.dumps(report, sort_keys=True).encode()
    report_path.write_bytes(raw)
    artifact["input_reports"][1]["sha256"] = hashlib.sha256(raw).hexdigest()
    write_preflight_fixture(preflight_path, artifact)
    with pytest.raises(ValueError, match="not repeatable|decision/repeatability"):
        PROBE.validate_cpu_preflight_artifact(preflight_path, source_bundle)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(synthesis_source_bundle={}), "synthesis source"),
        (lambda value: value["synthesis_source_bundle"].update(clean=False), "complete and clean"),
        (lambda value: value["synthesis_source_bundle"]["source_binding_files"].update({PROBE.SYNTHESIS_SOURCE_BINDING_PATHS[0]: "0" * 64}), "aggregate hash"),
        (lambda value: value["decision"].update(outcome="not_authorized"), "decision/repeatability"),
        (lambda value: value["governance"].update(learned=True), "governance"),
        (lambda value: value.update(created_at_utc="not-utc"), "created timestamp"),
        (lambda value: value["execution"].update(execution_id="bad"), "execution UUID"),
        (lambda value: value["execution"].update(started_at_utc="2026-08-30T00:00:01Z"), "execution binding"),
    ],
)
def test_cpu_preflight_rejects_self_report_and_provenance_tampering(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutate, message: str) -> None:
    preflight_path, source_bundle, artifact, _reports = preflight_artifact_fixture(monkeypatch, tmp_path)
    mutate(artifact)
    write_preflight_fixture(preflight_path, artifact)
    with pytest.raises(ValueError, match=message):
        PROBE.validate_cpu_preflight_artifact(preflight_path, source_bundle)


def test_cpu_preflight_rejects_non_finite_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preflight_path, source_bundle, artifact, _reports = preflight_artifact_fixture(monkeypatch, tmp_path)
    artifact["decision"]["repeatability"]["A.cpu"]["numeric_within_tolerance"] = float("nan")
    write_preflight_fixture(preflight_path, artifact)
    with pytest.raises(ValueError, match="non-finite JSON"):
        PROBE.validate_cpu_preflight_artifact(preflight_path, source_bundle)
