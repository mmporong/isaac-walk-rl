from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
import uuid
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/probe_g009_r0_rev20_terrain_contact_matrix.py"
SPEC = importlib.util.spec_from_file_location("rev20_probe", PATH)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(PROBE)


BODY_NAMES = ["base"] + [f"body_{index}" for index in range(1, 15)] + [f"foot_{index}" for index in range(4)]
ROOTS = [f"/World/envs/env_{index}/Robot/base" for index in range(8)]
NAMESPACES = [root.rsplit("/", 1)[0] for root in ROOTS]


class FakeData:
    def __init__(self, net: torch.Tensor, matrix: torch.Tensor, events: list[str] | None = None) -> None:
        self._net, self._matrix, self.events = net, matrix, events

    @property
    def net_forces_w(self):
        if self.events is not None: self.events.append("net")
        return self._net

    @property
    def force_matrix_w(self):
        if self.events is not None: self.events.append("buffer")
        return self._matrix


class FakeView:
    _g009_test_rigid_contact_view = True
    def __init__(self, direct: torch.Tensor, events: list[str] | None = None, mutate=None) -> None:
        self.direct, self.events, self.mutate = direct, events, mutate
        self.sensor_paths = [f"{namespace}/{body}" for namespace in NAMESPACES for body in BODY_NAMES]
        self.filter_paths = [[PROBE.FILTER_PATHS[0]] for _ in range(152)]
        self.sensor_names = list(BODY_NAMES) * 8; self.filter_names = [["CollisionPlane"] for _ in range(152)]
        self.sensor_count = 152; self.filter_count = 1

    def get_contact_force_matrix(self, _dt):
        if self.events is not None: self.events.append("direct")
        if self.mutate is not None: self.mutate()
        return self.direct


class FakeRobot:
    def __init__(self) -> None:
        self.body_names = list(BODY_NAMES)
        self.root_physx_view = types.SimpleNamespace(prim_paths=list(ROOTS))
        mass = torch.ones((8, 19), dtype=torch.float64)
        limits = torch.tensor([[[-1.0, 1.0]] * 12] * 8)
        self.data = types.SimpleNamespace(default_mass=mass, joint_pos=torch.zeros((8, 12)), joint_pos_limits=limits)


def fake_sensor(force_n: float = 1.0, *, events=None, same_storage=False) -> types.SimpleNamespace:
    net = torch.zeros((8, 19, 3), dtype=torch.float64); net[:, 0, 2] = force_n
    buffer = torch.zeros((8, 19, 1, 3), dtype=torch.float64); buffer[:, 0, 0, 2] = force_n
    direct = buffer.reshape(152, 1, 3) if same_storage else buffer.reshape(152, 1, 3).clone()
    return types.SimpleNamespace(data=FakeData(net, buffer, events), body_names=list(BODY_NAMES), contact_physx_view=FakeView(direct, events))


def test_preregistration_and_filter_injection_are_exact() -> None:
    prereg = PROBE.load_preregistration()
    assert prereg["evidence_id"] == "G009-5-E013"
    cfg = types.SimpleNamespace(scene=types.SimpleNamespace(contact_forces=types.SimpleNamespace(history_length=0)))
    PROBE.inject_terrain_filter(cfg)
    assert cfg.scene.contact_forces.filter_prim_paths_expr == list(PROBE.FILTER_PATHS)
    assert cfg.scene.contact_forces.history_length == 1
    with pytest.raises(ValueError, match="locked"):
        PROBE.inject_terrain_filter(cfg)


def test_startup_capture_env_accepts_event_manager_env_ids_positional_argument() -> None:
    env = object(); env_ids = torch.arange(8)
    setattr(PROBE, "_RUNTIME_ENV", None)
    PROBE.capture_env(env, env_ids, unused="accepted")
    assert getattr(PROBE, "_RUNTIME_ENV") is env
    setattr(PROBE, "_RUNTIME_ENV", None)


def test_ground_material_readback_resolves_physics_purpose_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    class Attr:
        def __init__(self, value): self.value = value
        def Get(self): return self.value
    class Prim:
        def IsValid(self): return True
        def GetAttribute(self, name): return Attr({"physics:staticFriction": 0.8, "physics:dynamicFriction": 0.6, "physxMaterial:frictionCombineMode": "multiply"}[name])
        def GetPath(self): return "/World/PhysicsMaterial"
    class Material:
        def GetPrim(self): return Prim()
    class Binding:
        def __init__(self, _collision): pass
        def ComputeBoundMaterial(self, *, materialPurpose): calls.append(materialPurpose); return Material(), object()
    stage = types.SimpleNamespace(GetPrimAtPath=lambda _path: Prim())
    usd_module = types.ModuleType("omni.usd"); setattr(usd_module, "get_context", lambda: types.SimpleNamespace(get_stage=lambda: stage))
    omni_module = types.ModuleType("omni"); setattr(omni_module, "usd", usd_module)
    pxr_module = types.ModuleType("pxr"); setattr(pxr_module, "UsdShade", types.SimpleNamespace(MaterialBindingAPI=Binding))
    monkeypatch.setitem(sys.modules, "omni", omni_module); monkeypatch.setitem(sys.modules, "omni.usd", usd_module); monkeypatch.setitem(sys.modules, "pxr", pxr_module)
    assert PROBE.ground_material_live_readback()["material_path"] == "/World/PhysicsMaterial"
    assert calls == ["physics"]


def test_lazy_read_order_and_direct_mutation_preserve_earlier_buffer_clone() -> None:
    events: list[str] = []
    net = torch.ones((8, 19, 3)); buffer = torch.ones((8, 19, 1, 3)); direct = torch.zeros((152, 1, 3))
    sensor = types.SimpleNamespace(data=FakeData(net, buffer, events), contact_physx_view=FakeView(direct, events, mutate=lambda: buffer.zero_()))
    cloned_net, cloned_buffer, cloned_direct, independent = PROBE.read_contact_tensors(sensor, 0.005)
    assert events == ["net", "buffer", "direct"]
    assert torch.all(cloned_net == 1) and torch.all(cloned_buffer == 1) and torch.all(cloned_direct == 0)
    assert independent is True


def test_same_tensor_alias_is_rejected_before_clone() -> None:
    assert PROBE.read_contact_tensors(fake_sensor(same_storage=True), 0.005)[3] is False


def test_same_storage_different_offset_view_is_rejected() -> None:
    backing = torch.zeros(8 * 19 * 3 + 1, dtype=torch.float64)
    buffer = backing[:-1].reshape(8, 19, 1, 3)
    direct = backing[1:].reshape(152, 1, 3)
    net = torch.zeros((8, 19, 3), dtype=torch.float64)
    sensor = types.SimpleNamespace(data=FakeData(net, buffer), body_names=list(BODY_NAMES), contact_physx_view=FakeView(direct))
    assert PROBE.read_contact_tensors(sensor, 0.005)[3] is False


def classified_invalid(sensor) -> dict:
    accumulator = PROBE.MatrixSafetyAccumulator(); robot = FakeRobot()
    for step in range(1, 151): accumulator.observe(step, sensor, robot, 0.005, torch)
    snapshot = accumulator.snapshot()
    assert PROBE.recompute_matrix_payload(snapshot)["availability_state"] == "invalid"
    return snapshot


def test_sensor_names_are_152_env_major_repeated_body_chunks() -> None:
    sensor = fake_sensor()
    assert len(sensor.contact_physx_view.sensor_names) == 152
    sensor.contact_physx_view.sensor_names[PROBE.BODY_COUNT] = BODY_NAMES[1]
    snapshot = classified_invalid(sensor)
    assert snapshot["structural_probe_valid"] is False
    assert "view path/count contract" in snapshot["error"]


def test_filter_metadata_requires_152_singleton_rows() -> None:
    sensor = fake_sensor(); sensor.contact_physx_view.filter_paths = list(PROBE.FILTER_PATHS)
    assert "view path/count contract" in classified_invalid(sensor)["error"]
    sensor = fake_sensor(); sensor.contact_physx_view.filter_names[19] = ["WrongPlane"]
    assert "view path/count contract" in classified_invalid(sensor)["error"]


def test_articulation_root_body_derives_robot_namespace() -> None:
    accumulator = PROBE.MatrixSafetyAccumulator(); sensor = fake_sensor(); robot = FakeRobot()
    for step in range(1, 151): accumulator.observe(step, sensor, robot, 0.005, torch)
    valid = accumulator.snapshot(); assert valid["path_order"]["articulation_root_body_paths"] == ROOTS and valid["path_order"]["body_namespace_paths"] == NAMESPACES
    assert valid["path_order"]["logical_filter_paths_sha256"] == PROBE.canonical_sha256(list(PROBE.FILTER_PATHS))
    assert valid["path_order"]["raw_filter_paths_sha256"] != valid["path_order"]["logical_filter_paths_sha256"]
    tampered = json.loads(json.dumps(valid)); tampered["path_order"]["body_namespace_paths"][0] += "/drift"
    with pytest.raises(ValueError, match="serialized matrix checks"): PROBE.recompute_matrix_payload(tampered)
    tampered = json.loads(json.dumps(valid)); tampered["path_order"]["raw_filter_paths_sha256"] = tampered["path_order"]["logical_filter_paths_sha256"]
    with pytest.raises(ValueError, match="serialized matrix checks"): PROBE.recompute_matrix_payload(tampered)


@pytest.mark.parametrize("mutation,message", [
    (lambda view: view.sensor_paths.__setitem__(19, "/World/envs/env_10/Robot/base"), "outside robot root"),
    (lambda view: view.sensor_paths.__setitem__(1, view.sensor_paths[2]), "count contract"),
    (lambda view: view.sensor_paths.__setitem__(1, f"{ROOTS[0]}/{BODY_NAMES[2]}"), "one body leaf"),
])
def test_contiguous_env_major_path_mapping_rejects_prefix_swap_and_duplicate(mutation, message: str) -> None:
    sensor = fake_sensor(); mutation(sensor.contact_physx_view)
    snapshot = classified_invalid(sensor)
    assert snapshot["availability_state"] == "invalid" and snapshot["structural_probe_valid"] is False
    assert message in snapshot["error"]


@pytest.mark.parametrize("shape", [(149, 1, 3), (151, 1, 3)])
def test_wrong_direct_shapes_fail_closed(shape) -> None:
    sensor = fake_sensor(); sensor.contact_physx_view.direct = torch.zeros(shape)
    snapshot = classified_invalid(sensor)
    assert snapshot["availability_state"] == "invalid" and "raw shape" in snapshot["error"]


def test_nan_and_direct_buffer_parity_drift_fail_closed() -> None:
    sensor = fake_sensor(); sensor.contact_physx_view.direct[0, 0, 0] = float("nan")
    snapshot = classified_invalid(sensor)
    assert snapshot["availability_state"] == "invalid" and ("differ" in snapshot["error"] or "finite" in snapshot["error"])
    sensor = fake_sensor(); sensor.contact_physx_view.direct[0, 0, 0] = 2.0
    assert "differ" in classified_invalid(sensor)["error"]


def test_exact_150_same_body_overlap_safety_inclusive_boundaries() -> None:
    robot = FakeRobot()
    robot.data.joint_pos[:] = -1.01  # exactly lower limit - 0.01 is inclusive
    body_weight = float(robot.data.default_mass[0].sum().item() * 9.81)
    sensor = fake_sensor(force_n=15.0 * body_weight)
    accumulator = PROBE.MatrixSafetyAccumulator()
    for step in range(1, 151): accumulator.observe(step, sensor, robot, 0.005, torch)
    snapshot = accumulator.snapshot()
    assert snapshot["passed"] is True
    assert snapshot["checks"]["same_body_positive_force_overlap_8_of_8"] is True
    assert max(snapshot["safety"]["non_foot_peak_force_body_weight_per_env"]) == pytest.approx(15.0)
    with pytest.raises(ValueError, match="step mismatch"):
        accumulator.observe(151, sensor, robot, 0.005, torch)


def test_matrix_validator_recomputes_reshaped_parity_ledger_and_summaries() -> None:
    accumulator = PROBE.MatrixSafetyAccumulator(); sensor = fake_sensor(); robot = FakeRobot()
    for step in range(1, 151): accumulator.observe(step, sensor, robot, 0.005, torch)
    snapshot = accumulator.snapshot()
    assert PROBE.recompute_matrix_payload(snapshot)["availability_state"] == "observed_valid"
    tampered = json.loads(json.dumps(snapshot)); tampered["step_ledger"][0]["direct_matrix_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="serialized matrix checks"):
        PROBE.recompute_matrix_payload(tampered)
    tampered = json.loads(json.dumps(snapshot)); tampered["same_step_overlap"]["all_env_matrix_peak_force_n"] += 1.0
    with pytest.raises(ValueError, match="summary"):
        PROBE.recompute_matrix_payload(tampered)
    tampered = json.loads(json.dumps(snapshot)); tampered["step_ledger"][0]["matrix_body_magnitude_n_by_env"][0][0] = 2.0
    tampered["step_ledger"][0]["matrix_positive_body_indices_by_env"][0] = [0]
    tampered["step_ledger"][0]["all_env_matrix_peak_force_n"] = 2.0
    tampered["same_step_overlap"]["all_env_matrix_peak_force_n"] = 2.0
    with pytest.raises(ValueError, match="magnitude hash"):
        PROBE.recompute_matrix_payload(tampered)
    tampered = json.loads(json.dumps(snapshot)); tampered["step_ledger"][0]["net_body_magnitude_n_by_env"][0][0] = 9.0
    tampered["step_ledger"][0]["net_body_magnitude_sha256"] = PROBE.canonical_sha256({"shape": [8, 19], "values": tampered["step_ledger"][0]["net_body_magnitude_n_by_env"]})
    with pytest.raises(ValueError, match="non-foot peak"):
        PROBE.recompute_matrix_payload(tampered)


def test_joint_position_nan_is_explicitly_recorded_and_recomputed() -> None:
    robot = FakeRobot(); robot.data.joint_pos[0, 0] = float("nan")
    accumulator = PROBE.MatrixSafetyAccumulator(); sensor = fake_sensor()
    for step in range(1, 151): accumulator.observe(step, sensor, robot, 0.005, torch)
    snapshot = accumulator.snapshot()
    assert snapshot["checks"]["finite_joint_position_and_contact_force"] is False
    assert snapshot["step_ledger"][0]["joint_position_finite"] is False
    assert PROBE.recompute_matrix_payload(snapshot)["checks"]["finite_joint_position_and_contact_force"] is False


def test_strict_overlap_threshold_and_149_sample_parity_fail() -> None:
    accumulator = PROBE.MatrixSafetyAccumulator(); sensor = fake_sensor(force_n=PROBE.FORCE_THRESHOLD_N)
    for step in range(1, 150): accumulator.observe(step, sensor, FakeRobot(), 0.005, torch)
    snapshot = accumulator.snapshot()
    assert snapshot["checks"]["exact_150_samples"] is False
    assert snapshot["checks"]["same_body_positive_force_overlap_8_of_8"] is False
    assert snapshot["passed"] is False


def test_baseline_contract_hashes_and_motor_raw_resolved_semantics() -> None:
    prereg = PROBE.load_preregistration(); expected = prereg["baseline_physics"]["expected_snapshot_contracts"]
    values = {name: expected[name]["value"] for name in ("material", "action", "motor", "reset", "timing")}
    tensors = prereg["baseline_physics"]["runtime_readback_contract"]["motor"]["resolved_tensor_expected_sha256"]
    snapshot = PROBE.baseline_snapshot_from_values(prereg, values, tensors)
    assert snapshot["all_match"] is True
    assert values["motor"]["raw_config"]["armature"] is None and values["motor"]["resolved"]["armature"] == 0.0
    bad = json.loads(json.dumps(values)); bad["motor"]["raw_config"]["armature"] = 0.0
    assert PROBE.baseline_snapshot_from_values(prereg, bad, tensors)["all_match"] is False
    bad = json.loads(json.dumps(values)); bad["reset"]["class_ids"][7] = 0
    assert PROBE.baseline_snapshot_from_values(prereg, bad, tensors)["all_match"] is False
    bad = json.loads(json.dumps(values)); bad["timing"]["control_dt_s"] = 0.021
    assert PROBE.baseline_snapshot_from_values(prereg, bad, tensors)["all_match"] is False


def test_capture_parse_cfg_keeps_dc_motor_cfg_distinct_from_runtime_dc_motor() -> None:
    DCMotor = type("DCMotor", (), {})
    DCMotorCfg = type("DCMotorCfg", (), {})
    motor = DCMotorCfg(); setattr(motor, "class_type", DCMotor)
    for name, value in {"armature": None, "damping": 0.5, "effort_limit": 23.5, "effort_limit_sim": None, "friction": 0.0, "joint_names_expr": [".*"], "saturation_effort": 23.5, "stiffness": 25.0, "velocity_limit": 30.0, "velocity_limit_sim": None}.items(): setattr(motor, name, value)
    action = types.SimpleNamespace(alpha=0.2, rescale_to_limits=True, scale=0.7, class_type=type("EMAJointPositionToLimitsActionCfg", (), {}))
    material = types.SimpleNamespace(static_friction=0.8, dynamic_friction=0.6, friction_combine_mode="multiply")
    robot = types.SimpleNamespace(soft_joint_pos_limit_factor=0.9, actuators={"base_legs": motor}, spawn=types.SimpleNamespace(articulation_props=types.SimpleNamespace(solver_position_iteration_count=8, solver_velocity_iteration_count=0), rigid_props=types.SimpleNamespace(max_depenetration_velocity=1.0)))
    cfg = types.SimpleNamespace(actions=types.SimpleNamespace(joint_pos=action), scene=types.SimpleNamespace(robot=robot, terrain=types.SimpleNamespace(physics_material=material)), events=types.SimpleNamespace(reset_base=types.SimpleNamespace(params={"assignment_mode": "stratified", "pose_xy_range": (0.0, 0.0), "yaw_range": (0.0, 0.0)}), physics_material=types.SimpleNamespace(params={"static_friction_range": (1.0, 1.0), "dynamic_friction_range": (1.0, 1.0)})), decimation=4, sim=types.SimpleNamespace(dt=0.005))
    captured = PROBE.capture_parse_cfg_values(cfg)
    assert captured["motor_raw"]["actuator_class"] == "DCMotorCfg"
    assert captured["motor_raw"]["actuator_class"] != getattr(motor, "class_type").__name__ == "DCMotor"


def representative_baseline_payload() -> tuple[dict, dict]:
    prereg = json.loads(json.dumps(PROBE.load_preregistration())); expected = prereg["baseline_physics"]["expected_snapshot_contracts"]
    values = {name: expected[name]["value"] for name in ("material", "action", "motor", "reset", "timing")}
    motor_constants = {"stiffness": 25.0, "damping": 0.5, "armature": 0.0, "friction": 0.0, "effort_limit": 23.5, "velocity_limit": 30.0, "effort_limit_sim": 1e9, "default_joint_armature": 0.0}
    motor_records = {name: PROBE._tensor_record(torch.full((8, 12), constant)) for name, constant in motor_constants.items()}; motor_hashes = {name: record["sha256"] for name, record in motor_records.items()}
    prereg["baseline_physics"]["runtime_readback_contract"]["motor"]["resolved_tensor_expected_sha256"] = motor_hashes
    raw_keys = {"foot_material_all_8x4_exact", "effective_material_all_8x4_exact", "reset_class_ids_exact", "reset_root_velocity_zero", "reset_joint_velocity_zero", "current_root_state_matches_reset_log", "current_joint_state_matches_reset_log", "reset_root_pose_exact_8_env", "reset_folded_joint_state_exact_8_env", "ema_history_equals_reset_joint_position", "zero_action_envs_0_to_3_exact", "hold_action_envs_4_to_7_finite_bounded_unsaturated", "hold_target_envs_4_to_7_equal_folded_state", "root_pose_finite_8x7", "timing_sources_exact", "ground_material_attributes_finite", "live_action_cfg_matches_parse_cfg", "parse_cfg_ground_material_matches_live", "parse_cfg_solver_8_0_depenetration_1", "solver_effort_limits_match", "solver_velocity_limits_match", "default_joint_properties_match"}
    payload = PROBE.baseline_snapshot_from_values(prereg, values, motor_hashes, {key: True for key in raw_keys})
    class_ids = values["reset"]["class_ids"]; definitions = values["reset"]["pose_definitions_in_class_id_order"]
    root_pose = torch.tensor([[0.0, 0.0, definitions[class_id]["root_height_m"], *definitions[class_id]["root_quaternion_wxyz"]] for class_id in class_ids])
    joint_names = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]; folded = values["reset"]["folded_joint_angles_rad"]
    joint_row = [folded["left_hip" if name.startswith(("FL_", "RL_")) else "right_hip"] if name.endswith("_hip_joint") else folded["thigh"] if name.endswith("_thigh_joint") else folded["calf"] for name in joint_names]; joint = torch.tensor([joint_row] * 8)
    hold = torch.zeros((4, 12)); assignment = torch.zeros((8, 12)); assignment[4:] = hold
    payload["action_assignment"] = {"shape": [8, 12], "values": assignment.tolist(), "sha256": PROBE.canonical_sha256({"shape": [8, 12], "values": assignment.tolist()}), "zero_envs": [0, 1, 2, 3], "hold_envs": [4, 5, 6, 7]}
    payload["reset_runtime_evidence"] = {"logged_root_pose": PROBE.rev19.tensor_snapshot(root_pose), "current_root_pose": PROBE.rev19.tensor_snapshot(root_pose), "logged_joint_pos": PROBE.rev19.tensor_snapshot(joint), "current_joint_pos": PROBE.rev19.tensor_snapshot(joint), "ema_previous_targets": PROBE.rev19.tensor_snapshot(joint), "hold_normalized_action": PROBE.rev19.tensor_snapshot(hold), "hold_reachable_target": PROBE.rev19.tensor_snapshot(joint[4:])}
    zero_root_velocity = torch.zeros((8, 6)); zero_joint_velocity = torch.zeros((8, 12)); effort = motor_records["effort_limit_sim"]; velocity_sim = PROBE._tensor_record(torch.full((8, 12), 1e9))
    payload["runtime_observations"] = {
        "material": {"foot": PROBE._tensor_record(torch.ones((8, 4, 2))), "effective": PROBE._tensor_record(torch.tensor([0.8, 0.6]).repeat(8, 4, 1)), "ground": {"static": 0.8, "dynamic": 0.6, "combine": "multiply", "material_path": "/World/PhysicsMaterial"}},
        "action": {"live_cfg": values["action"], "parsed_cfg": values["action"]},
        "motor": {"live_tensors": motor_records, "velocity_limit_sim": velocity_sim, "joint_effort_limits": effort, "joint_velocity_limits": velocity_sim, "default_joint_stiffness": motor_records["stiffness"], "default_joint_damping": motor_records["damping"], "default_joint_friction": motor_records["friction"]},
        "reset": {"class_ids": class_ids, "env_origins": PROBE._tensor_record(torch.zeros((8, 3))), "root_velocity": PROBE._tensor_record(zero_root_velocity), "joint_velocity": PROBE._tensor_record(zero_joint_velocity), "logged_root_velocity": PROBE._tensor_record(zero_root_velocity), "logged_joint_velocity": PROBE._tensor_record(zero_joint_velocity), "joint_names": joint_names, "saturated_mask": PROBE._tensor_record(torch.zeros((4, 12), dtype=torch.bool))},
        "timing": {"physics_dt": 0.005, "cfg_sim_dt": 0.005, "decimation": 4, "step_dt": 0.02, "parsed": values["timing"], "solver_parsed": {"position": 8, "velocity": 0, "max_depenetration_velocity": 1.0}},
    }
    contact = PROBE._tensor_record(torch.zeros((8, 19))); rest = PROBE._tensor_record(torch.zeros((8, 19))); mass = PROBE._tensor_record(torch.ones((8, 19))); names = list(BODY_NAMES)
    prereg["baseline_physics"].update(expected_contact_offset_tensor_sha256=contact["sha256"], expected_rest_offset_tensor_sha256=rest["sha256"], expected_mass_tensor_sha256=mass["sha256"], expected_mass_body_names_sha256=PROBE.canonical_sha256(names), expected_force_body_names_sha256=PROBE.canonical_sha256(names))
    checks = {"contact_offset_tensor_hash": True, "rest_offset_tensor_hash": True, "mass_tensor_hash": True, "mass_body_order_hash": True, "force_body_order_hash": True}
    payload["invariants"] = {"contact_offsets": contact, "rest_offsets": rest, "mass": mass, "mass_body_names": names, "force_body_names": names, "checks": checks}; payload["all_match"] = True
    return payload, prereg


def test_baseline_validator_recomputes_raw_observations_and_rejects_tamper() -> None:
    payload, prereg = representative_baseline_payload(); assert PROBE.validate_baseline_payload(payload, prereg) is True
    missing = json.loads(json.dumps(payload)); del missing["runtime_observations"]["motor"]["velocity_limit_sim"]
    with pytest.raises(ValueError, match="motor runtime observation schema"): PROBE.validate_baseline_payload(missing, prereg)
    tampered = json.loads(json.dumps(payload)); effective = tampered["runtime_observations"]["material"]["effective"]; effective["values"][0][0][0] = 0.9; effective["sha256"] = PROBE.canonical_sha256({"shape": effective["shape"], "values": effective["values"]})
    assert all(tampered["raw_runtime_checks"].values())
    with pytest.raises(ValueError, match="raw runtime checks"): PROBE.validate_baseline_payload(tampered, prereg)


def test_external_source_hashes_are_recomputed(tmp_path: Path) -> None:
    relative = "source/example.py"; path = tmp_path / relative; path.parent.mkdir(parents=True); path.write_bytes(b"authority")
    prereg = {"baseline_physics": {"isaaclab_external_source_binding": {"files": {relative: PROBE.file_sha256(path)}}}}
    assert PROBE.validate_external_sources(tmp_path, prereg)["all_hashes_match"] is True
    path.write_bytes(b"drift")
    with pytest.raises(ValueError, match="hash mismatch"):
        PROBE.validate_external_sources(tmp_path, prereg)


def test_gpu_preflight_validation_happens_before_app_launcher_constructor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[str] = []
    args = argparse.Namespace(device="cuda:0", replicate_index=1, cpu_preflight=Path("preflight"), isaaclab_root=tmp_path)
    execution = {"execution_id": uuid.uuid4().hex, "started_at_utc": "2026-01-01T00:00:00Z", "output_path_repo_relative": PROBE.EXPECTED_PATHS[("cuda:0", 1)], "no_overwrite": True}
    monkeypatch.setattr(PROBE.runtime_probe, "parse_prelaunch_output", lambda _argv: argparse.Namespace(output=tmp_path / "unused.json"))
    monkeypatch.setattr(PROBE.runtime_probe, "prepare_execution", lambda _value: (tmp_path / "unused.json", execution))
    monkeypatch.setattr(PROBE, "parse_args", lambda _argv: args)
    monkeypatch.setattr(PROBE, "prelaunch_validate", lambda _args: events.append("preflight") or {"cpu_preflight_binding": {}})
    monkeypatch.setattr(PROBE, "diagnose", lambda _args, _execution: {"feasibility": {"run_interpretable": True, "availability_state": "observed_valid"}})
    monkeypatch.setattr(PROBE, "validate_report", lambda _report: {"run_interpretable": True})
    monkeypatch.setattr(PROBE.runtime_probe, "_write_json_atomic", lambda *_args: None)

    class FakeLauncher:
        def __init__(self, _args): events.append("launcher"); self.app = types.SimpleNamespace(close=lambda: None)

    isaaclab = types.ModuleType("isaaclab"); app_module = types.ModuleType("isaaclab.app"); setattr(app_module, "AppLauncher", FakeLauncher)
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab); monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    assert PROBE.main([]) == 0
    assert events == ["preflight", "launcher"]


def build_cpu_preflight_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, dict, list[dict]]:
    monkeypatch.setattr(PROBE, "REPO_ROOT", tmp_path)
    preflight_path = tmp_path / "reports/runs/cpu_preflight.json"; preflight_path.parent.mkdir(parents=True)
    monkeypatch.setattr(PROBE, "CPU_PREFLIGHT_PATH", preflight_path)
    source = {"git_commit": "a" * 40, "source_bundle_sha256": "b" * 64}
    monkeypatch.setattr(PROBE, "validate_report", lambda _report: {"run_interpretable": True}); monkeypatch.setattr(PROBE, "live_readback_valid", lambda _report: True)
    reports: list[dict] = []
    bindings: list[dict] = []
    for replicate in (1, 2):
        matrix = {"availability_state": "observed_valid", "structural_probe_valid": True, "safety_valid": True, "path_order": {"sensor_paths_sha256": "1" * 64, "raw_filter_paths_sha256": "2" * 64, "logical_filter_paths_sha256": "4" * 64, "force_body_names_sha256": "3" * 64}, "shapes": {"raw": [152, 1, 3], "reshaped": [8, 19, 1, 3]}, "same_step_overlap": {"per_env_overlap_step_indices": [[1] for _ in range(8)], "source_env_overlap_step_indices": [1], "all_env_matrix_peak_force_n": 1.0, "source_env_matrix_peak_force_n": 1.0, "all_env_matrix_force_integral_n_s": 0.75, "source_env_matrix_force_integral_n_s": 0.75}, "checks": {"all": True}}
        report = {"device": "cpu", "replicate_index": replicate, "execution": {"execution_id": uuid.uuid4().hex}, "terrain_contact_matrix": matrix, "baseline_snapshot": {"all_match": True}, "device_readback": {"gpu_dynamics_matches_device": True}, "external_source_binding": {"all_hashes_match": True}, "source_bundle": source}
        reports.append(report)
        relative = PROBE.EXPECTED_PATHS[("cpu", replicate)]; path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True); raw = json.dumps(report).encode(); path.write_bytes(raw)
        bindings.append({"path": relative, "sha256": PROBE.sha256_bytes(raw)})
    outcome, repeatability = PROBE.recompute_cpu_preflight_decision(reports); assert outcome == "gpu_stage_authorized"
    synthesis_files = {relative: f"{index + 1:064x}" for index, relative in enumerate(PROBE.SYNTHESIS_SOURCE_BINDING_PATHS)}; synthesis_payload = "\n".join(f"{relative}:{synthesis_files[relative]}" for relative in sorted(synthesis_files)); synthesis_sha = PROBE.sha256_bytes(synthesis_payload.encode())
    synthesis = {"schema_version": 1, "git_commit": source["git_commit"], "git_commit_valid": True, "source_binding_paths": list(PROBE.SYNTHESIS_SOURCE_BINDING_PATHS), "source_binding_files": synthesis_files, "source_bundle_sha256": synthesis_sha, "clean": True}
    value = {"schema_version": "g009.r0.rev20.terrain_contact_matrix_cpu_preflight.v1", "evidence_id": "G009-5-E013", "status": "complete", "mode": "cpu_preflight_2x", "input_report_count": 2, "decision": {"outcome": outcome, "third_run_allowed": False, "repeatability": repeatability}, "input_reports": bindings, "integrity": {"passed": True, "hash_bound": True, "unique_report_paths": True, "unique_report_sha256": True, "unique_execution_ids": True, "exact_slots": ["cpu.rep1", "cpu.rep2"], "git_commit": source["git_commit"], "probe_source_bundle_sha256": source["source_bundle_sha256"], "synthesis_source_bundle_sha256": synthesis_sha}, "cpu_preflight": {"passed": True, "required_checks_passed": True, "within_cpu_repeatability_passed": True, "gpu_stage_allowed": True}, "governance": PROBE.governance(), "synthesis_source_bundle": synthesis, "execution": {"execution_id": uuid.uuid4().hex, "started_at_utc": "2026-01-01T00:00:00Z", "output_path_repo_relative": preflight_path.relative_to(tmp_path).as_posix(), "no_overwrite": True}}
    preflight_path.write_text(json.dumps(value), encoding="utf-8")
    return preflight_path, source, reports


def test_cpu_preflight_artifact_recomputes_authorization_and_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path, source, reports = build_cpu_preflight_artifact(monkeypatch, tmp_path)
    assert PROBE.validate_cpu_preflight_artifact(path, source)["status"] == "validated_for_gpu"
    value = json.loads(path.read_text(encoding="utf-8")); value["execution"]["execution_id"] = reports[0]["execution"]["execution_id"]; path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="collision"): PROBE.validate_cpu_preflight_artifact(path, source)


@pytest.mark.parametrize("replacement", [
    lambda item: {"sha256": item["sha256"], "path": item["path"]},
    lambda item: {"path": item["path"]},
    lambda item: {"path": item["path"], "sha256": item["sha256"], "extra": True},
])
def test_cpu_preflight_binding_rejects_reverse_missing_and_extra_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, replacement) -> None:
    path, source, _reports = build_cpu_preflight_artifact(monkeypatch, tmp_path); value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=dict)
    value["input_reports"][0] = replacement(value["input_reports"][0]); path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="key order/schema"): PROBE.validate_cpu_preflight_artifact(path, source)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(unexpected_top_level=True),
    lambda value: value.pop("governance"),
])
def test_cpu_preflight_rejects_extra_and_missing_top_level_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation) -> None:
    path, source, _reports = build_cpu_preflight_artifact(monkeypatch, tmp_path)
    value = json.loads(path.read_text(encoding="utf-8")); mutation(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level schema"):
        PROBE.validate_cpu_preflight_artifact(path, source)


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value.update(evidence_id="wrong"), "identity"),
    (lambda value: value["cpu_preflight"].update(required_checks_passed=False), "status fields"),
    (lambda value: value["decision"].update(third_run_allowed=True), "decision"),
    (lambda value: value["synthesis_source_bundle"].update(source_bundle_sha256="0" * 64), "aggregate"),
    (lambda value: value["execution"].update(output_path_repo_relative="wrong.json"), "execution identity"),
])
def test_cpu_preflight_rejects_identity_semantic_and_source_tamper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation, match: str) -> None:
    path, source, _reports = build_cpu_preflight_artifact(monkeypatch, tmp_path); value = json.loads(path.read_text(encoding="utf-8")); mutation(value); path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=match): PROBE.validate_cpu_preflight_artifact(path, source)


@pytest.mark.parametrize("mutation,match", [
    (lambda value, reports: value["input_reports"][0].update(extra=True), "schema"),
    (lambda value, reports: value["input_reports"].reverse(), "exact inputs"),
    (lambda value, reports: reports.__setitem__(1, json.loads(json.dumps(reports[0]))), "exact inputs"),
    (lambda value, reports: reports[0]["terrain_contact_matrix"].update(availability_state="unavailable"), "do not authorize"),
    (lambda value, reports: reports[1]["terrain_contact_matrix"]["same_step_overlap"].update(all_env_matrix_peak_force_n=2.0), "do not authorize"),
])
def test_cpu_preflight_rejects_forged_authorization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation, match: str) -> None:
    path, source, reports = build_cpu_preflight_artifact(monkeypatch, tmp_path); value = json.loads(path.read_text(encoding="utf-8")); mutation(value, reports)
    for replicate, report in enumerate(reports, 1):
        report_path = tmp_path / PROBE.EXPECTED_PATHS[("cpu", replicate)]; raw = json.dumps(report).encode(); report_path.write_bytes(raw); value["input_reports"][replicate - 1]["sha256"] = PROBE.sha256_bytes(raw)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=match): PROBE.validate_cpu_preflight_artifact(path, source)


def test_invalid_gpu_preflight_never_constructs_app_launcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[str] = []
    args = argparse.Namespace(device="cuda:0", replicate_index=1, cpu_preflight=tmp_path / "bad.json", isaaclab_root=tmp_path)
    execution = {"execution_id": uuid.uuid4().hex, "started_at_utc": "2026-01-01T00:00:00Z", "output_path_repo_relative": PROBE.EXPECTED_PATHS[("cuda:0", 1)], "no_overwrite": True}
    monkeypatch.setattr(PROBE.runtime_probe, "parse_prelaunch_output", lambda _argv: argparse.Namespace(output=tmp_path / "unused.json"))
    monkeypatch.setattr(PROBE.runtime_probe, "prepare_execution", lambda _value: (tmp_path / "unused.json", execution))
    monkeypatch.setattr(PROBE, "parse_args", lambda _argv: args)
    monkeypatch.setattr(PROBE, "prelaunch_validate", lambda _args: (_ for _ in ()).throw(ValueError("invalid immutable preflight")))
    class FakeLauncher:
        def __init__(self, _args): events.append("launcher")
    isaaclab = types.ModuleType("isaaclab"); app_module = types.ModuleType("isaaclab.app"); setattr(app_module, "AppLauncher", FakeLauncher)
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab); monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    assert PROBE.main([]) == 2
    assert events == []


def test_parse_args_requires_headless_before_launcher_construction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class LauncherArgs:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device"); parser.add_argument("--device-explicit", action="store_true"); parser.add_argument("--headless", action="store_true")
    isaaclab = types.ModuleType("isaaclab"); app_module = types.ModuleType("isaaclab.app"); setattr(app_module, "AppLauncher", LauncherArgs)
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab); monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    base = ["--replicate-index", "1", "--output", str(tmp_path / "out.json"), "--device", "cpu", "--device-explicit"]
    with pytest.raises(SystemExit): PROBE.parse_args(base)
    assert PROBE.parse_args([*base, "--headless"]).headless is True


def test_bare_help_is_import_free_and_bypasses_output_preparser(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delitem(sys.modules, "isaaclab", raising=False); monkeypatch.delitem(sys.modules, "isaaclab.app", raising=False)
    monkeypatch.setattr(PROBE.runtime_probe, "parse_prelaunch_output", lambda _argv: (_ for _ in ()).throw(AssertionError("output pre-parser called")))
    assert PROBE.main(["--help"]) == 0
    help_text = capsys.readouterr().out
    for option in ("--task", "--seed", "--replicate-index", "--cpu-preflight", "--isaaclab-root", "--output", "--device", "--headless"): assert option in help_text


def test_runtime_failure_writes_only_noncanonical_failed_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = argparse.Namespace(device="cpu", replicate_index=1, cpu_preflight=None, isaaclab_root=tmp_path)
    canonical = tmp_path / "canonical.json"; failure = tmp_path / "failure.json"
    execution = {"execution_id": uuid.uuid4().hex, "started_at_utc": "2026-01-01T00:00:00Z", "output_path_repo_relative": PROBE.EXPECTED_PATHS[("cpu", 1)], "no_overwrite": True}
    monkeypatch.setattr(PROBE.runtime_probe, "parse_prelaunch_output", lambda _argv: canonical)
    monkeypatch.setattr(PROBE.runtime_probe, "prepare_execution", lambda _value: (canonical, execution))
    monkeypatch.setattr(PROBE, "parse_args", lambda _argv: args)
    monkeypatch.setattr(PROBE, "prelaunch_validate", lambda _args: {"cpu_preflight_binding": {}})
    monkeypatch.setattr(PROBE, "diagnose", lambda *_args: (_ for _ in ()).throw(RuntimeError("injected")))
    monkeypatch.setattr(PROBE, "failed_attempt_path", lambda *_args: failure)
    written: list[Path] = []
    monkeypatch.setattr(PROBE.runtime_probe, "_write_json_atomic", lambda path, _value: written.append(path))
    class FakeLauncher:
        def __init__(self, _args): self.app = types.SimpleNamespace(close=lambda: None)
    isaaclab = types.ModuleType("isaaclab"); app_module = types.ModuleType("isaaclab.app"); setattr(app_module, "AppLauncher", FakeLauncher)
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab); monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    assert PROBE.main([]) == 2
    assert written == [failure]
    assert canonical not in written


def test_callback_fields_cannot_upgrade_matrix_outcome() -> None:
    report = {"terrain_contact_matrix": {"availability_state": "unavailable", "passed": False}, "baseline_snapshot": {"all_match": True}, "device_readback": {"gpu_dynamics_matches_device": True}, "external_source_binding": {"all_hashes_match": True}, "raw_contact_observation": {"callback_count": 999999}}
    assert PROBE.derive_feasibility(report)["availability_state"] == "unavailable"
    assert PROBE.derive_feasibility(report)["probe_valid"] is False
