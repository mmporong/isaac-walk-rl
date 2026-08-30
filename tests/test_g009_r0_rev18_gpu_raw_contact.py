from __future__ import annotations

import copy
import hashlib
import importlib.util
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g009_rev18_raw_contact_test",
    ROOT / "scripts" / "probe_g009_r0_rev18_gpu_raw_contact.py",
)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _source_hash(relative: str) -> str:
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()


def _source_bundle() -> dict:
    files = {path: _source_hash(path) for path in PROBE.SOURCE_BINDING_PATHS}
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return {
        "schema_version": 1,
        "git_commit": "a" * 40,
        "git_commit_valid": True,
        "source_binding_paths": list(PROBE.SOURCE_BINDING_PATHS),
        "source_binding_files": files,
        "source_bundle_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "all_files_present": True,
        "missing_files": [],
        "clean": True,
        "dirty_source_paths": [],
    }


@pytest.fixture(autouse=True)
def _committed_blob_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        PROBE,
        "committed_blob_sha256",
        lambda relative, commit: _source_hash(relative),
    )


class Header:
    actor0 = 1
    actor1 = 2
    collider0 = 3
    collider1 = 4
    contact_data_offset = 0
    num_contact_data = 1
    type = None


class Datum:
    position = [0.0, 0.0, 0.0]
    normal = [0.0, 0.0, 1.0]
    impulse = [0.0, 0.0, 2.0]
    separation = -0.001


def _path(handle: int) -> str:
    return {
        1: "/World/envs/env_7/Robot/base",
        2: "/World/ground",
        3: "/World/envs/env_7/Robot/base/collision",
        4: "/World/ground/mesh",
    }[handle]


def _raw_event(step: int = 1) -> dict:
    return PROBE.copy_contact_callback(
        [Header()],
        [Datum()],
        physics_step=step,
        int_to_path=_path,
        contact_event_types=None,
    )


def _residual() -> dict:
    stats = {
        "position_rms": 0.1,
        "position_max": 0.2,
        "velocity_rms": 0.1,
        "velocity_max": 0.2,
    }
    return {
        "status": "observed",
        "samples": "usd_physx_residual_reporting_api",
        "scene": dict(stats),
        "source_articulation_root": dict(stats),
        "error": None,
    }


def _report(device: str = "cuda:0", replicate_index: int = 1) -> dict:
    rows = [
        {
            "physics_step": step,
            "time_s": step * PROBE.PHYSICS_DT_S,
            "contact_sensor": {
                "net_forces_w_n": [[0.0, 0.0, 1.0]] + [[0.0, 0.0, 0.0]] * 18,
                "force_matrix_w": {
                    "status": "unavailable",
                    "value": None,
                    "error": "not configured",
                },
            },
            "incoming_joint_wrench_b": [[0.0] * 6 for _ in range(19)],
            "solver_residual": _residual(),
        }
        for step in range(1, PROBE.PHYSICS_SUBSTEPS + 1)
    ]
    contract = PROBE.probe_contract(device, replicate_index)
    report = {
        "schema_version": PROBE.SCHEMA_VERSION,
        "goal_id": "g009",
        "stage_id": "R0",
        "experiment_id": "G009-5-E011",
        "revision": "rev18",
        "status": "complete",
        "headless": True,
        "device": device,
        "replicate_index": replicate_index,
        "seed": 42,
        "num_envs": PROBE.NUM_ENVS,
        "source_env_index": PROBE.SOURCE_ENV_INDEX,
        "physics_substeps": PROBE.PHYSICS_SUBSTEPS,
        "physics_dt_s": PROBE.PHYSICS_DT_S,
        "execution": {
            "execution_id": uuid.uuid4().hex,
            "started_at_utc": "2026-08-30T00:00:00Z",
            "output_path_repo_relative": PROBE.expected_output_relative(
                device, replicate_index
            ),
            "no_overwrite": True,
        },
        "source_bundle": _source_bundle(),
        "finished_at_utc": "2026-08-30T00:00:01Z",
        "manual_inner_loop": {
            "control_decimation": 4,
            "action_process_steps": list(range(1, 151, 4)),
            "action_process_count": 38,
            "manager_post_step_executed": False,
            "reward_computed": False,
            "termination_computed": False,
            "trajectory_equivalence_claimed": False,
            "scope": "capability_only",
        },
        "governance": PROBE.governance(),
        "pose_action_assignment": {
            "class_ids": [0, 1, 2, 3, 0, 1, 2, 3]
        },
        "live_physics_readback": {"solver": {}, "max_depenetration_velocity": {}},
        "residual_capability": {"enable_attempted": True},
        "contract": contract,
        "contract_sha256": PROBE.canonical_sha256(contract),
        "predecessor": {
            "path": PROBE.PREDECESSOR_PATH.relative_to(PROBE.REPO_ROOT).as_posix(),
            "sha256": PROBE.PREDECESSOR_SHA256,
        },
        "device_readback": {
            "requested_device": device,
            "runtime_device": device,
            "physics_scene_prim_path": "/physicsScene",
            "gpu_dynamics_enabled": device == "cuda:0",
            "gpu_dynamics_matches_device": True,
            "error": None,
        },
        "physics_step_clock": {
            "source": "subscribe_physics_on_step_events(pre_step=true,order=0)",
            "callback_count": PROBE.PHYSICS_SUBSTEPS,
            "expected_callback_count": PROBE.PHYSICS_SUBSTEPS,
            "observed_dt_s": [PROBE.PHYSICS_DT_S] * PROBE.PHYSICS_SUBSTEPS,
            "passed": True,
        },
        "raw_contact_observation": {
            "authority_scope": PROBE.AUTHORITY_SCOPE,
            "physics_ground_truth_authority": False,
            "subscription_attempted": True,
            "subscription_succeeded": True,
            "subscription_error": None,
            "callback_count": 1,
            "malformed_callback_count": 0,
            "first_callback_error": None,
            "events": [_raw_event()],
        },
        "supporting_telemetry": rows,
    }
    report["feasibility"] = PROBE.derive_feasibility(report)
    return report


def test_contract_is_fixed_to_rev17_b_cell_and_closed_governance() -> None:
    contract = PROBE.probe_contract("cuda:0", 2)

    assert contract["predecessor"]["sha256"] == PROBE.PREDECESSOR_SHA256
    assert contract["controlled_cell"] == {
        "arm": "B",
        "solver_position_iterations": 16,
        "solver_velocity_iterations": 0,
        "seed": 42,
        "num_envs": 8,
        "source_env_index": 7,
        "pose_id": "right_side",
        "action_mode": "reset_pose_hold",
        "device": "cuda:0",
        "replicate_index": 2,
    }
    assert contract["execution"]["physics_substeps"] == 150
    assert contract["execution"]["physics_dt_s"] == pytest.approx(0.005)
    assert PROBE.governance()["ppo_updates"] == 0
    assert PROBE.governance()["gate_execution_allowed"] is False
    assert PROBE.governance()["physics_ground_truth_authority"] is False


def test_callback_copies_absolute_pair_and_full_raw_datum() -> None:
    event = _raw_event(19)

    assert event["physics_step"] == 19
    assert event["robot_ground_header_count"] == 1
    assert event["robot_ground_datum_count"] == 1
    header = event["headers"][0]
    assert header["env_index"] == 7
    assert header["actor0_path"].startswith("/")
    assert header["contact_points"][0] == {
        "position_w_m": [0.0, 0.0, 0.0],
        "normal_w": [0.0, 0.0, 1.0],
        "impulse_n_s": [0.0, 0.0, 2.0],
        "separation_m": -0.001,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("normal", [0.0, 0.0, 2.0], "unit length"),
        ("impulse", [0.0, float("nan"), 1.0], "finite"),
        ("separation", float("inf"), "finite"),
    ],
)
def test_callback_rejects_malformed_raw_data(
    field: str, value: object, message: str
) -> None:
    datum = Datum()
    setattr(datum, field, value)

    with pytest.raises(ValueError, match=message):
        PROBE.copy_contact_callback(
            [Header()],
            [datum],
            physics_step=1,
            int_to_path=_path,
            contact_event_types=None,
        )


def test_accumulator_preserves_malformed_callback_as_fail_closed_evidence() -> None:
    accumulator = PROBE.RawContactAccumulator(_path, lambda: 1, None)
    accumulator.mark_subscription(object())
    bad = Datum()
    bad.normal = [0.0, 0.0, 0.0]

    accumulator([Header()], [bad])
    snapshot = accumulator.snapshot()

    assert snapshot["subscription_succeeded"] is True
    assert snapshot["callback_count"] == 1
    assert snapshot["malformed_callback_count"] == 1
    assert snapshot["events"] == []
    assert "unit length" in snapshot["first_callback_error"]


def test_callback_rejects_out_of_range_slice_even_for_lost_event() -> None:
    header = Header()
    header.contact_data_offset = 2
    header.num_contact_data = 0

    with pytest.raises(ValueError, match="exceeds"):
        PROBE.copy_contact_callback(
            [header],
            [],
            physics_step=1,
            int_to_path=_path,
            contact_event_types=None,
        )


def test_runtime_enum_rejects_missing_contact_event_type() -> None:
    enum = type(
        "ContactEventType",
        (),
        {
            "CONTACT_FOUND": 1,
            "CONTACT_PERSIST": 2,
            "CONTACT_LOST": 3,
        },
    )

    with pytest.raises(ValueError, match="event type is missing"):
        PROBE._event_name(None, enum)
    assert PROBE._event_name(None, None) == "CONTACT_PERSIST"


def test_ground_prefix_boundary_rejects_ground_fake() -> None:
    assert PROBE._is_ground_path("/World/ground") is True
    assert PROBE._is_ground_path("/World/ground/mesh") is True
    assert PROBE._is_ground_path("/World/groundFake") is False

    def fake_ground_path(handle: int) -> str:
        if handle in {2, 4}:
            return "/World/groundFake"
        return _path(handle)

    event = PROBE.copy_contact_callback(
        [Header()],
        [Datum()],
        physics_step=1,
        int_to_path=fake_ground_path,
        contact_event_types=None,
    )
    assert event["robot_ground_header_count"] == 0
    assert event["robot_ground_datum_count"] == 0


class _FakePath:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class _FakePrim:
    def __init__(self, path: str) -> None:
        self.path = _FakePath(path)
        self.stage = object()

    def GetPath(self) -> _FakePath:
        return self.path

    def GetStage(self) -> object:
        return self.stage


class _FakeAttribute:
    def __init__(self, value: object) -> None:
        self.value = value

    def Get(self) -> object:
        return self.value


class _FakeResidualApi:
    def GetPhysxResidualReportingRmsResidualPositionIterationAttr(self):
        return _FakeAttribute(0.1)

    def GetPhysxResidualReportingMaxResidualPositionIterationAttr(self):
        return _FakeAttribute(0.2)

    def GetPhysxResidualReportingRmsResidualVelocityIterationAttr(self):
        return _FakeAttribute(0.3)

    def GetPhysxResidualReportingMaxResidualVelocityIterationAttr(self):
        return _FakeAttribute(0.4)


class _FakeResidualReportingApiType:
    applied: list[str] = []

    @classmethod
    def Apply(cls, prim: _FakePrim) -> object:
        cls.applied.append(str(prim.GetPath()))
        return object()

    @staticmethod
    def Get(stage: object, path: _FakePath) -> _FakeResidualApi:
        assert stage is not None
        assert str(path).startswith("/")
        return _FakeResidualApi()


class _FakeContext:
    def __init__(self) -> None:
        self.enable_calls: list[bool] = []

    def enable_residual_reporting(self, enabled: bool) -> None:
        self.enable_calls.append(enabled)


def test_residual_reader_enables_and_reads_scene_and_source_root() -> None:
    _FakeResidualReportingApiType.applied = []
    schema = type(
        "Schema",
        (),
        {"PhysxResidualReportingAPI": _FakeResidualReportingApiType},
    )
    context = _FakeContext()

    reader = PROBE.ResidualReader(
        context,
        _FakePrim("/physicsScene"),
        _FakePrim("/World/envs/env_7/Robot"),
        schema,
    )
    sample = reader.read()

    assert context.enable_calls == [True]
    assert _FakeResidualReportingApiType.applied == [
        "/physicsScene",
        "/World/envs/env_7/Robot",
    ]
    assert reader.capability["enable_succeeded"] is True
    assert reader.capability["scene"]["status"] == "enabled"
    assert reader.capability["source_articulation_root"]["status"] == "enabled"
    assert sample["status"] == "observed"
    assert sample["scene"]["position_rms"] == pytest.approx(0.1)
    assert sample["source_articulation_root"]["velocity_max"] == pytest.approx(
        0.4
    )


def test_residual_reader_preserves_enable_failure_as_null_capability() -> None:
    class BrokenContext:
        def enable_residual_reporting(self, enabled: bool) -> None:
            assert enabled is True
            raise RuntimeError("unsupported")

    schema = type(
        "Schema",
        (),
        {"PhysxResidualReportingAPI": _FakeResidualReportingApiType},
    )

    reader = PROBE.ResidualReader(
        BrokenContext(),
        _FakePrim("/physicsScene"),
        _FakePrim("/World/envs/env_7/Robot"),
        schema,
    )
    sample = reader.read()

    assert reader.capability["enable_succeeded"] is False
    assert "unsupported" in reader.capability["enable_error"]
    assert sample["status"] == "unavailable"
    assert sample["samples"] is None


@pytest.mark.parametrize(
    ("device", "observed", "expected_match"),
    [
        ("cpu", False, True),
        ("cpu", True, False),
        ("cuda:0", True, True),
        ("cuda:0", False, False),
    ],
)
def test_gpu_dynamics_readback_requires_exact_device_state(
    device: str, observed: bool, expected_match: bool
) -> None:
    class SceneApi:
        def GetEnableGPUDynamicsAttr(self) -> _FakeAttribute:
            return _FakeAttribute(observed)

    class SceneApiType:
        @staticmethod
        def Get(stage: object, path: _FakePath) -> SceneApi:
            assert stage is not None
            assert str(path) == "/physicsScene"
            return SceneApi()

    schema = type("Schema", (), {"PhysxSceneAPI": SceneApiType})

    result = PROBE._gpu_dynamics_readback(
        schema, device, _FakePrim("/physicsScene")
    )

    assert result["gpu_dynamics_enabled"] is observed
    assert result["gpu_dynamics_matches_device"] is expected_match
    assert result["physics_scene_prim_path"] == "/physicsScene"


def test_gpu_dynamics_none_is_not_cpu_false() -> None:
    class SceneApi:
        def GetEnableGPUDynamicsAttr(self) -> _FakeAttribute:
            return _FakeAttribute(None)

    class SceneApiType:
        @staticmethod
        def Get(stage: object, path: _FakePath) -> SceneApi:
            return SceneApi()

    schema = type("Schema", (), {"PhysxSceneAPI": SceneApiType})

    result = PROBE._gpu_dynamics_readback(
        schema, "cpu", _FakePrim("/physicsScene")
    )

    assert result["gpu_dynamics_enabled"] is None
    assert result["gpu_dynamics_matches_device"] is False
    assert "not a bool" in result["error"]


def test_forged_gpu_matches_flag_cannot_override_disabled_gpu_dynamics() -> None:
    report = _report("cuda:0")
    report["device_readback"]["gpu_dynamics_enabled"] = False
    report["device_readback"]["gpu_dynamics_matches_device"] = True
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["checks"]["device_readback_matches"] is False
    assert result["probe_valid"] is False


def test_source_bundle_provenance_hashes_commit_blob_not_windows_worktree_eol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    relative = "bound.txt"
    path = tmp_path / relative
    path.write_bytes(b"line1\r\nline2\r\n")
    committed = hashlib.sha256(b"line1\nline2\n").hexdigest()
    monkeypatch.setattr(PROBE, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(PROBE, "SOURCE_BINDING_PATHS", (relative,))
    monkeypatch.setattr(PROBE, "current_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(PROBE, "source_binding_status", lambda: [])
    monkeypatch.setattr(
        PROBE,
        "committed_blob_sha256",
        lambda source_path, commit: committed,
    )

    bundle = PROBE.source_bundle_provenance()

    assert bundle["clean"] is True
    assert bundle["all_files_present"] is True
    assert bundle["source_binding_files"] == {relative: committed}
    assert bundle["source_binding_files"][relative] != hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_valid_gpu_report_recomputes_raw_and_bundle_status() -> None:
    report = _report()

    result = PROBE.validate_report(report)

    assert result["raw_observation_passed"] is True
    assert result["probe_valid"] is True
    assert result["supporting_bundle_complete"] is True
    assert result["run_feasible"] is True
    assert result["gpu_pair_attribution_available"] is True


def test_proxy_cannot_upgrade_missing_raw_gpu_callback() -> None:
    report = _report()
    report["raw_contact_observation"].update(
        subscription_succeeded=False,
        subscription_error="unsupported",
        callback_count=0,
        events=[],
    )
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["checks"]["force_proxy_complete"] is True
    assert result["checks"]["positive_force_stimulus_present"] is True
    assert result["probe_valid"] is True
    assert result["raw_observation_passed"] is False
    assert result["gpu_pair_attribution_available"] is False
    assert result["run_feasible"] is False


def test_cpu_raw_observation_cannot_substitute_gpu_pair_attribution() -> None:
    report = _report("cpu")

    result = PROBE.validate_report(report)

    assert result["raw_observation_passed"] is True
    assert result["run_feasible"] is True
    assert result["gpu_pair_attribution_available"] is False


def test_missing_force_proxy_invalidates_probe_but_preserves_raw_status() -> None:
    report = _report()
    report["supporting_telemetry"][7]["contact_sensor"]["net_forces_w_n"] = None
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["raw_observation_passed"] is True
    assert result["probe_valid"] is False
    assert result["gpu_pair_attribution_available"] is False


@pytest.mark.parametrize(
    ("field", "bad_value", "check"),
    [
        ("net_forces_w_n", [[0.0, 0.0, 1.0]] * 18, "force_proxy_complete"),
        ("incoming_joint_wrench_b", [[0.0] * 6] * 18, "joint_wrench_complete"),
    ],
)
def test_supporting_tensor_shapes_are_recomputed_exactly(
    field: str, bad_value: list[list[float]], check: str
) -> None:
    report = _report()
    if field == "net_forces_w_n":
        report["supporting_telemetry"][0]["contact_sensor"][field] = bad_value
    else:
        report["supporting_telemetry"][0][field] = bad_value
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["checks"][check] is False
    assert result["run_feasible"] is False


def test_absolute_paths_do_not_replace_source_robot_ground_topology() -> None:
    report = _report()
    header = report["raw_contact_observation"]["events"][0]["headers"][0]
    header["actor1_path"] = "/World/other_surface"
    header["collider1_path"] = "/World/other_surface/mesh"
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["checks"]["absolute_pair_paths"] is True
    assert result["checks"]["source_robot_ground_pair_paths"] is False
    assert result["raw_observation_passed"] is False


def test_raw_and_positive_force_steps_must_overlap() -> None:
    report = _report()
    report["supporting_telemetry"][0]["contact_sensor"][
        "net_forces_w_n"
    ] = [[0.0, 0.0, 0.0] for _ in range(19)]
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["checks"]["positive_force_stimulus_present"] is True
    assert result["checks"]["force_proxy_complete"] is True
    assert result["checks"]["raw_force_step_overlap"] is False
    assert result["probe_valid"] is True
    assert result["raw_observation_passed"] is False


def test_no_positive_force_stimulus_invalidates_probe_independently() -> None:
    report = _report()
    for row in report["supporting_telemetry"]:
        row["contact_sensor"]["net_forces_w_n"] = [
            [0.0, 0.0, 0.0] for _ in range(19)
        ]
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["checks"]["force_proxy_complete"] is True
    assert result["checks"]["positive_force_stimulus_present"] is False
    assert result["probe_valid"] is False
    assert result["raw_observation_passed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_process_steps", list(range(1, 150, 4))[:-1]),
        ("action_process_count", 39),
        ("manager_post_step_executed", True),
        ("reward_computed", True),
        ("termination_computed", True),
        ("trajectory_equivalence_claimed", True),
        ("scope", "trajectory_equivalent"),
    ],
)
def test_manual_inner_loop_contract_is_recomputed(
    field: str, value: object
) -> None:
    report = _report()
    report["manual_inner_loop"][field] = value
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["checks"]["manual_inner_loop_contract"] is False
    assert result["probe_valid"] is False
    assert result["run_feasible"] is False


@pytest.mark.parametrize("field", ["incoming_joint_wrench_b", "solver_residual"])
def test_missing_wrench_or_residual_marks_bundle_incomplete_without_erasing_raw(
    field: str,
) -> None:
    report = _report()
    report["supporting_telemetry"][0][field] = None
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["raw_observation_passed"] is True
    assert result["probe_valid"] is True
    assert result["supporting_bundle_complete"] is False
    assert result["gpu_pair_attribution_available"] is True
    assert result["run_feasible"] is False


def test_serialized_success_is_ignored_and_rejected() -> None:
    report = _report()
    report["raw_contact_observation"]["events"] = []
    report["feasibility"] = {
        **report["feasibility"],
        "raw_observation_passed": True,
        "run_feasible": True,
        "gpu_pair_attribution_available": True,
    }

    derived = PROBE.derive_feasibility(report)
    assert derived["raw_observation_passed"] is False
    assert derived["gpu_pair_attribution_available"] is False
    with pytest.raises(ValueError, match="recomputation|self-report"):
        PROBE.validate_report(report)


@pytest.mark.parametrize("count", [149, 151])
def test_exact_150_step_contract_is_fail_closed(count: int) -> None:
    report = _report()
    report["physics_step_clock"]["callback_count"] = count
    report["physics_step_clock"]["passed"] = False
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["checks"]["exact_150_physics_steps"] is False
    assert result["run_feasible"] is False


@pytest.mark.parametrize(
    "observed_dt",
    [
        [PROBE.PHYSICS_DT_S] * 149,
        [PROBE.PHYSICS_DT_S] * 149 + [float("nan")],
        [PROBE.PHYSICS_DT_S] * 149 + [0.006],
    ],
)
def test_clock_passed_flag_cannot_replace_observed_dt_recomputation(
    observed_dt: list[float],
) -> None:
    report = _report()
    report["physics_step_clock"]["observed_dt_s"] = observed_dt
    report["physics_step_clock"]["passed"] = True
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["checks"]["exact_150_physics_steps"] is False
    assert result["probe_valid"] is False


def test_clock_passed_false_cannot_override_valid_observed_dt() -> None:
    report = _report()
    report["physics_step_clock"]["passed"] = False
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["checks"]["exact_150_physics_steps"] is True
    assert result["probe_valid"] is True


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("callback_count", 2),
        ("robot_ground_header_count", 2),
        ("robot_ground_datum_count", 2),
    ],
)
def test_raw_snapshot_self_report_counts_are_rejected(
    target: str, value: int
) -> None:
    report = _report()
    if target == "callback_count":
        report["raw_contact_observation"][target] = value
    else:
        report["raw_contact_observation"]["events"][0][target] = value
    report["feasibility"] = PROBE.derive_feasibility(report)

    assert report["feasibility"]["checks"]["raw_snapshot_counts_consistent"] is False
    with pytest.raises(ValueError, match="self-report"):
        PROBE.validate_report(report)


def test_validation_rejects_open_governance() -> None:
    report = _report()
    report["governance"]["ppo_updates"] = 1

    with pytest.raises(ValueError, match="governance"):
        PROBE.validate_report(report)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("goal_id", "other", "goal identity"),
        ("stage_id", "S0", "stage identity"),
        ("experiment_id", "G009-5-E010", "experiment identity"),
        ("revision", "rev17", "revision identity"),
        ("seed", 7, "seed"),
        ("num_envs", 4, "num_envs"),
        ("source_env_index", 0, "source env"),
    ],
)
def test_top_level_identity_is_bound_to_contract(
    field: str, value: object, message: str
) -> None:
    report = _report()
    report[field] = value

    with pytest.raises(ValueError, match=message):
        PROBE.validate_report(report)


def test_top_level_field_set_and_pose_assignment_are_exact() -> None:
    extra = _report()
    extra["claimed_success"] = True
    with pytest.raises(ValueError, match="field set"):
        PROBE.validate_report(extra)

    pose = _report()
    pose["pose_action_assignment"]["class_ids"][-1] = 0
    with pytest.raises(ValueError, match="pose/action"):
        PROBE.validate_report(pose)


def test_device_runtime_identity_is_recomputed() -> None:
    report = _report("cuda:0")
    report["device_readback"]["runtime_device"] = "cpu"
    report["feasibility"] = PROBE.derive_feasibility(report)

    result = PROBE.validate_report(report)

    assert result["checks"]["device_readback_matches"] is False
    assert result["probe_valid"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("execution_id", "0" * 32, "UUID4"),
        ("started_at_utc", "2026-08-30T09:00:00+09:00", "UTC"),
        (
            "output_path_repo_relative",
            "reports/runs/arbitrary.json",
            "output binding",
        ),
        ("no_overwrite", False, "no-overwrite"),
    ],
)
def test_execution_metadata_is_recomputed_strictly(
    field: str, value: object, message: str
) -> None:
    report = _report("cuda:0", 2)
    report["execution"][field] = value

    with pytest.raises(ValueError, match=message):
        PROBE.validate_report(report)


def test_execution_metadata_rejects_extra_key() -> None:
    report = _report()
    report["execution"]["claimed_success"] = True

    with pytest.raises(ValueError, match="key set"):
        PROBE.validate_report(report)


def test_source_bundle_rejects_partial_or_reordered_paths() -> None:
    partial = _source_bundle()
    first = PROBE.SOURCE_BINDING_PATHS[0]
    partial["source_binding_paths"] = [first]
    partial["source_binding_files"] = {first: _source_hash(first)}

    with pytest.raises(ValueError, match="path order|key set"):
        PROBE.validate_source_bundle(partial)

    reordered = _source_bundle()
    reordered["source_binding_paths"] = list(
        reversed(reordered["source_binding_paths"])
    )
    with pytest.raises(ValueError, match="path order"):
        PROBE.validate_source_bundle(reordered)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("git_commit_valid", False, "commit identity"),
        ("all_files_present", False, "complete and clean"),
        ("missing_files", ["missing.py"], "complete and clean"),
        ("clean", False, "complete and clean"),
        ("dirty_source_paths", [" M script.py"], "complete and clean"),
        ("source_bundle_sha256", "0" * 64, "aggregate"),
    ],
)
def test_source_bundle_rejects_forged_status_and_digest(
    field: str, value: object, message: str
) -> None:
    bundle = _source_bundle()
    bundle[field] = value

    with pytest.raises(ValueError, match=message):
        PROBE.validate_source_bundle(bundle)


def test_source_bundle_rejects_malformed_and_noncommitted_file_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = PROBE.SOURCE_BINDING_PATHS[0]
    malformed = _source_bundle()
    malformed["source_binding_files"][relative] = "not-a-hash"
    with pytest.raises(ValueError, match="hash format"):
        PROBE.validate_source_bundle(malformed)

    forged = _source_bundle()
    forged["source_binding_files"][relative] = "f" * 64
    files = forged["source_binding_files"]
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    forged["source_bundle_sha256"] = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()
    monkeypatch.setattr(
        PROBE,
        "committed_blob_sha256",
        lambda path, commit: _source_hash(path),
    )
    with pytest.raises(ValueError, match="committed blob mismatch"):
        PROBE.validate_source_bundle(forged)


def test_source_bundle_rejects_extra_schema_key() -> None:
    bundle = _source_bundle()
    bundle["claimed_valid"] = True

    with pytest.raises(ValueError, match="schema"):
        PROBE.validate_source_bundle(bundle)


def test_main_rejects_existing_output_before_app_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(PROBE.runtime_probe, "REPO_ROOT", tmp_path)
    output = tmp_path / "reports" / "runs" / "existing.json"
    output.parent.mkdir(parents=True)
    output.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        PROBE.main(["--output", str(output), "--device", "cpu", "--replicate-index", "1", "--headless"])
    assert output.read_text(encoding="utf-8") == "original"


def test_execution_contract_declares_normal_control_cadence() -> None:
    expected = list(range(1, PROBE.PHYSICS_SUBSTEPS + 1, 4))

    assert len(expected) == 38
    assert expected[0] == 1
    assert expected[-1] == 149


def test_failure_envelope_never_claims_gpu_attribution() -> None:
    args = type(
        "Args",
        (),
        {"headless": True, "device": "cuda:0", "seed": 42, "replicate_index": 2},
    )()
    execution = {
        "execution_id": uuid.uuid4().hex,
        "started_at_utc": "2026-08-30T00:00:00Z",
        "output_path_repo_relative": "reports/runs/failure.json",
        "no_overwrite": True,
    }

    envelope = PROBE.failure_envelope(args, execution, RuntimeError("boom"))

    assert envelope["status"] == "failed_closed"
    assert envelope["gpu_pair_attribution_available"] is False
    assert envelope["governance"] == PROBE.governance()
