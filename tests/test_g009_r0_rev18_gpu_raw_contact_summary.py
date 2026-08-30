from __future__ import annotations

import copy
import hashlib
import importlib.util
import uuid
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g009_rev18_gpu_raw_contact_summary_test",
    ROOT / "scripts" / "summarize_g009_r0_rev18_gpu_raw_contact.py",
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)
PROBE = SUMMARY.probe
ORIGINAL_SYNTHESIS_SOURCE_BUNDLE_PROVENANCE = (
    SUMMARY.synthesis_source_bundle_provenance
)


@pytest.fixture(autouse=True)
def _bind_fixture_source_hashes_to_committed_blobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PROBE,
        "committed_blob_sha256",
        lambda relative_path, _commit: hashlib.sha256(
            relative_path.encode("utf-8")
        ).hexdigest(),
    )
    synthesis_bundle = _synthesis_source_bundle()
    monkeypatch.setattr(
        SUMMARY,
        "committed_synthesis_blob_sha256",
        lambda relative_path, _commit: hashlib.sha256(
            relative_path.encode("utf-8")
        ).hexdigest(),
    )
    monkeypatch.setattr(
        SUMMARY,
        "synthesis_source_bundle_provenance",
        lambda: copy.deepcopy(synthesis_bundle),
    )


def _execution_id(index: int) -> str:
    return uuid.UUID(fields=(index, 1, 0x4000, 0x80, 0, index)).hex


def _source_bundle() -> dict[str, Any]:
    files = {
        path: hashlib.sha256(path.encode("utf-8")).hexdigest()
        for path in PROBE.SOURCE_BINDING_PATHS
    }
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return {
        "schema_version": 1,
        "git_commit": "1" * 40,
        "git_commit_valid": True,
        "source_binding_paths": list(PROBE.SOURCE_BINDING_PATHS),
        "source_binding_files": files,
        "source_bundle_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "all_files_present": True,
        "missing_files": [],
        "clean": True,
        "dirty_source_paths": [],
    }


def _synthesis_source_bundle() -> dict[str, Any]:
    files = {
        path: hashlib.sha256(path.encode("utf-8")).hexdigest()
        for path in SUMMARY.SYNTHESIS_SOURCE_BINDING_PATHS
    }
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return {
        "schema_version": 1,
        "role": "offline_synthesis_implementation",
        "git_commit": "1" * 40,
        "git_commit_valid": True,
        "source_binding_paths": list(SUMMARY.SYNTHESIS_SOURCE_BINDING_PATHS),
        "source_binding_files": files,
        "source_bundle_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "all_files_present": True,
        "missing_files": [],
        "clean": True,
        "dirty_source_paths": [],
    }


def _raw_observation(*, available: bool, variant: str = "base") -> dict[str, Any]:
    if not available:
        subscription_error = None
        callback_count = 0
        if variant == "subscription_failed":
            subscription_error = "RuntimeError: subscription unavailable"
        return {
            "authority_scope": PROBE.AUTHORITY_SCOPE,
            "physics_ground_truth_authority": False,
            "subscription_attempted": True,
            "subscription_succeeded": variant != "subscription_failed",
            "subscription_error": subscription_error,
            "callback_count": callback_count,
            "malformed_callback_count": 0,
            "first_callback_error": None,
            "events": [],
        }
    step = 10 if variant != "different_structure" else 11
    impulse = 0.25
    if variant == "numeric_close":
        impulse += 1.0e-7
    elif variant == "numeric_far":
        impulse += 1.0e-3
    return {
        "authority_scope": PROBE.AUTHORITY_SCOPE,
        "physics_ground_truth_authority": False,
        "subscription_attempted": True,
        "subscription_succeeded": True,
        "subscription_error": None,
        "callback_count": 1,
        "malformed_callback_count": 0,
        "first_callback_error": None,
        "events": [
            {
                "physics_step": step,
                "robot_ground_header_count": 1,
                "robot_ground_datum_count": 1,
                "headers": [
                    {
                        "env_index": PROBE.SOURCE_ENV_INDEX,
                        "event_type": "CONTACT_PERSIST",
                        "actor0_path": "/World/envs/env_7/Robot/base",
                        "actor1_path": "/World/ground/terrain/GroundPlane",
                        "collider0_path": "/World/envs/env_7/Robot/base/collisions",
                        "collider1_path": "/World/ground/terrain/GroundPlane/CollisionPlane",
                        "contact_points": [
                            {
                                "position_w_m": [0.0, 0.0, 0.1],
                                "normal_w": [0.0, 0.0, 1.0],
                                "impulse_n_s": [0.0, 0.0, impulse],
                                "separation_m": -1.0e-4,
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _telemetry(*, residual_complete: bool) -> list[dict[str, Any]]:
    residual = (
        {
            "status": "observed",
            "samples": "usd_physx_residual_reporting_api",
            "scene": {
                "position_rms": 0.1,
                "position_max": 0.2,
                "velocity_rms": 0.3,
                "velocity_max": 0.4,
            },
            "source_articulation_root": {
                "position_rms": 0.1,
                "position_max": 0.2,
                "velocity_rms": 0.3,
                "velocity_max": 0.4,
            },
            "error": None,
        }
        if residual_complete
        else {
            "status": "unavailable",
            "samples": None,
            "scene": None,
            "source_articulation_root": None,
            "error": "residual API unavailable",
        }
    )
    return [
        {
            "physics_step": step,
            "time_s": step * PROBE.PHYSICS_DT_S,
            "contact_sensor": {
                "net_forces_w_n": [[0.0, 0.0, 1.0]]
                + [[0.0, 0.0, 0.0] for _ in range(18)],
                "force_matrix_w": {
                    "status": "unavailable",
                    "value": None,
                    "error": "unavailable by configuration",
                },
            },
            "incoming_joint_wrench_b": [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(19)
            ],
            "solver_residual": copy.deepcopy(residual),
        }
        for step in range(1, PROBE.PHYSICS_SUBSTEPS + 1)
    ]


def make_report(
    device: str,
    replicate_index: int,
    *,
    raw_available: bool = True,
    raw_variant: str = "base",
    residual_complete: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
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
        "finished_at_utc": "2026-08-30T00:00:01.000000Z",
        "manual_inner_loop": {
            "control_decimation": 4,
            "action_process_steps": list(range(1, PROBE.PHYSICS_SUBSTEPS + 1, 4)),
            "action_process_count": 38,
            "manager_post_step_executed": False,
            "reward_computed": False,
            "termination_computed": False,
            "trajectory_equivalence_claimed": False,
            "scope": "capability_only",
        },
        "execution": {
            "execution_id": _execution_id(
                replicate_index + (0 if device == "cpu" else 2)
            ),
            "started_at_utc": "2026-08-30T00:00:00.000000Z",
            "output_path_repo_relative": PROBE.expected_output_relative(
                device, replicate_index
            ),
            "no_overwrite": True,
        },
        "contract": PROBE.probe_contract(device, replicate_index),
        "contract_sha256": PROBE.canonical_sha256(
            PROBE.probe_contract(device, replicate_index)
        ),
        "predecessor": {
            "path": PROBE.PREDECESSOR_PATH.relative_to(PROBE.REPO_ROOT).as_posix(),
            "sha256": PROBE.PREDECESSOR_SHA256,
        },
        "source_bundle": _source_bundle(),
        "governance": PROBE.governance(),
        "pose_action_assignment": {"class_ids": [0, 1, 2, 3, 0, 1, 2, 3]},
        "live_physics_readback": {"solver": {}, "max_depenetration_velocity": {}},
        "device_readback": {
            "requested_device": device,
            "runtime_device": device,
            "physics_scene_prim_path": "/World/physicsScene",
            "gpu_dynamics_enabled": device == "cuda:0",
            "gpu_dynamics_matches_device": True,
            "error": None,
        },
        "residual_capability": {"enable_attempted": True},
        "physics_step_clock": {
            "source": "subscribe_physics_on_step_events(pre_step=true,order=0)",
            "callback_count": PROBE.PHYSICS_SUBSTEPS,
            "expected_callback_count": PROBE.PHYSICS_SUBSTEPS,
            "observed_dt_s": [
                PROBE.PHYSICS_DT_S for _ in range(PROBE.PHYSICS_SUBSTEPS)
            ],
            "passed": True,
        },
        "raw_contact_observation": _raw_observation(
            available=raw_available, variant=raw_variant
        ),
        "supporting_telemetry": _telemetry(residual_complete=residual_complete),
    }
    report["feasibility"] = PROBE.derive_feasibility(report)
    PROBE.validate_report(report)
    return report


def make_entries(**overrides: dict[str, Any]):
    reports = {
        "cpu1": make_report("cpu", 1),
        "cpu2": make_report("cpu", 2),
        "gpu1": make_report("cuda:0", 1),
        "gpu2": make_report("cuda:0", 2),
    }
    reports.update(overrides)
    entries = []
    for index, name in enumerate(("cpu1", "cpu2", "gpu1", "gpu2"), 1):
        path = reports[name]["execution"]["output_path_repo_relative"]
        entries.append(
            (
                reports[name],
                {"path": path, "sha256": f"{index:064x}"},
            )
        )
    return entries


def test_2x2_available_is_hash_bound_and_governance_closed() -> None:
    result = SUMMARY.synthesize_loaded(make_entries())
    assert result["input_report_count"] == 4
    assert result["integrity"]["hash_bound"] is True
    assert result["integrity"]["unique_execution_ids"] is True
    assert result["synthesis_source_bundle"]["role"] == (
        "offline_synthesis_implementation"
    )
    assert result["raw_contact_feasibility"]["outcome"] == (
        "gpu_pair_attribution_available"
    )
    assert result["raw_contact_feasibility"]["gpu_pair_attribution_available"] is True
    assert result["raw_contact_feasibility"]["cross_device_numeric_equality_required"] is False
    assert result["instrumentation_bundle"]["status"] == "complete"
    assert result["decision"]["selected_lever"] is None
    assert result["governance"] == {
        "diagnostic_only": True,
        "selected_lever": None,
        "learned": False,
        "ppo": {"allowed": False, "status": "not_run", "updates": 0},
        "qualification": {"eligible": False, "status": "not_run", "passed": None},
        "gate01": {"allowed": False, "status": "forbidden"},
    }


def test_cpu_cannot_be_substituted_by_gpu() -> None:
    cpu2 = make_report("cpu", 2, raw_available=False)
    result = SUMMARY.synthesize_loaded(make_entries(cpu2=cpu2))
    assert result["decision"]["outcome"] == "probe_invalid"
    assert result["raw_contact_feasibility"]["gpu_pair_attribution_available"] is False


def test_proxy_cannot_upgrade_unavailable_gpu_raw_observation() -> None:
    entries = make_entries(
        gpu1=make_report("cuda:0", 1, raw_available=False),
        gpu2=make_report("cuda:0", 2, raw_available=False),
    )
    result = SUMMARY.synthesize_loaded(entries)
    assert result["instrumentation_bundle"]["status"] == "complete"
    gpu_runs = [
        run
        for run in result["raw_contact_feasibility"]["runs"]
        if run["device"] == "cuda:0"
    ]
    assert all(run["positive_force_stimulus_present"] is True for run in gpu_runs)
    assert all(run["probe_valid"] is True for run in gpu_runs)
    assert result["decision"] == {
        "outcome": "unavailable_on_gpu",
        "next_step": "pre_registered_single_variable_intervention",
        "selected_lever": None,
    }


def test_unavailable_gpu_without_positive_force_stimulus_is_probe_invalid() -> None:
    gpu2 = make_report("cuda:0", 2, raw_available=False)
    for row in gpu2["supporting_telemetry"]:
        row["contact_sensor"]["net_forces_w_n"] = [
            [0.0, 0.0, 0.0] for _ in range(19)
        ]
    gpu2["feasibility"] = PROBE.derive_feasibility(gpu2)
    PROBE.validate_report(gpu2)
    result = SUMMARY.synthesize_loaded(
        make_entries(
            gpu1=make_report("cuda:0", 1, raw_available=False), gpu2=gpu2
        )
    )
    assert result["decision"]["outcome"] == "probe_invalid"


def test_serialized_success_is_recomputed_and_rejected() -> None:
    gpu2 = make_report("cuda:0", 2, raw_available=False)
    gpu2["feasibility"]["raw_observation_passed"] = True
    with pytest.raises(ValueError, match="serialized feasibility"):
        SUMMARY.synthesize_loaded(make_entries(gpu2=gpu2))


def test_duplicate_execution_id_and_raw_hash_are_rejected() -> None:
    entries = make_entries()
    entries[3][0]["execution"]["execution_id"] = entries[2][0]["execution"][
        "execution_id"
    ]
    with pytest.raises(ValueError, match="execution_id values must be unique"):
        SUMMARY.synthesize_loaded(entries)

    entries = make_entries()
    entries[3][1]["sha256"] = entries[2][1]["sha256"]
    with pytest.raises(ValueError, match="duplicate raw report hash"):
        SUMMARY.synthesize_loaded(entries)


def test_one_of_two_gpu_split_is_never_promoted() -> None:
    entries = make_entries(gpu2=make_report("cuda:0", 2, raw_available=False))
    result = SUMMARY.synthesize_loaded(entries)
    assert result["decision"]["outcome"] == "inconclusive_nondeterministic"
    assert result["decision"]["next_step"] == "stop_without_third_run_majority_vote"
    assert result["raw_contact_feasibility"]["third_run_majority_vote_allowed"] is False


def test_only_identical_two_of_two_unavailable_advances_to_intervention() -> None:
    same = make_entries(
        gpu1=make_report("cuda:0", 1, raw_available=False),
        gpu2=make_report("cuda:0", 2, raw_available=False),
    )
    assert SUMMARY.synthesize_loaded(same)["decision"]["outcome"] == (
        "unavailable_on_gpu"
    )

    different = make_entries(
        gpu1=make_report("cuda:0", 1, raw_available=False),
        gpu2=make_report(
            "cuda:0", 2, raw_available=False, raw_variant="subscription_failed"
        ),
    )
    assert SUMMARY.synthesize_loaded(different)["decision"]["outcome"] == (
        "inconclusive_nondeterministic"
    )


def test_raw_feasible_and_partial_residual_bundle_remain_separate() -> None:
    entries = make_entries(
        gpu2=make_report("cuda:0", 2, residual_complete=False)
    )
    result = SUMMARY.synthesize_loaded(entries)
    assert result["decision"]["outcome"] == "gpu_pair_attribution_available"
    assert result["instrumentation_bundle"] == {
        "status": "partial",
        "complete_report_count": 3,
        "required_report_count": 4,
        "independent_of_raw_contact_feasibility": True,
    }


def test_within_device_structure_and_numeric_tolerance_are_enforced() -> None:
    close = make_entries(
        gpu2=make_report("cuda:0", 2, raw_variant="numeric_close")
    )
    assert SUMMARY.synthesize_loaded(close)["decision"]["outcome"] == (
        "gpu_pair_attribution_available"
    )

    far = make_entries(gpu2=make_report("cuda:0", 2, raw_variant="numeric_far"))
    assert SUMMARY.synthesize_loaded(far)["decision"]["outcome"] == (
        "inconclusive_nondeterministic"
    )

    structural = make_entries(
        gpu2=make_report("cuda:0", 2, raw_variant="different_structure")
    )
    assert SUMMARY.synthesize_loaded(structural)["decision"]["outcome"] == (
        "inconclusive_nondeterministic"
    )


def test_duplicate_slot_and_source_bundle_drift_are_rejected() -> None:
    entries = make_entries()
    duplicate_slot = make_report("cpu", 1)
    duplicate_slot["execution"]["execution_id"] = _execution_id(9)
    entries[1] = (duplicate_slot, entries[1][1])
    with pytest.raises(ValueError, match="binding path must match execution output"):
        SUMMARY.synthesize_loaded(entries)

    entries = make_entries()
    entries[3][0]["source_bundle"] = copy.deepcopy(entries[3][0]["source_bundle"])
    entries[3][0]["source_bundle"]["git_commit"] = "9" * 40
    with pytest.raises(ValueError, match="source bundle payload changed"):
        SUMMARY.synthesize_loaded(entries)


def test_binding_path_must_match_report_execution_output() -> None:
    entries = make_entries()
    entries[3][1]["path"] = "reports/runs/g009_r0_rev18_wrong_binding.json"
    with pytest.raises(ValueError, match="binding path must match execution output"):
        SUMMARY.synthesize_loaded(entries)


def test_fake_one_path_raw_source_bundle_is_rejected() -> None:
    entries = make_entries()
    fake = entries[0][0]["source_bundle"]
    fake["source_binding_paths"] = [PROBE.SOURCE_BINDING_PATHS[0]]
    fake["source_binding_files"] = {PROBE.SOURCE_BINDING_PATHS[0]: "a" * 64}
    fake["source_bundle_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="source bundle path order mismatch"):
        SUMMARY.synthesize_loaded(entries)


def test_repeatability_projects_only_controlled_source_env() -> None:
    gpu2 = make_report("cuda:0", 2)
    gpu2["raw_contact_observation"]["events"][0]["headers"].append(
        {
            "env_index": 0,
            "event_type": "CONTACT_FOUND",
            "actor0_path": "/World/envs/env_0/Robot/base",
            "actor1_path": "/World/ground/terrain/GroundPlane",
            "collider0_path": "/World/envs/env_0/Robot/base/collisions",
            "collider1_path": "/World/ground/terrain/GroundPlane/CollisionPlane",
            "contact_points": [
                {
                    "position_w_m": [9.0, 9.0, 9.0],
                    "normal_w": [0.0, 0.0, 1.0],
                    "impulse_n_s": [0.0, 0.0, 99.0],
                    "separation_m": -0.1,
                }
            ],
        }
    )
    gpu2["raw_contact_observation"]["events"][0][
        "robot_ground_header_count"
    ] = 2
    gpu2["raw_contact_observation"]["events"][0][
        "robot_ground_datum_count"
    ] = 2
    gpu2["feasibility"] = PROBE.derive_feasibility(gpu2)
    PROBE.validate_report(gpu2)
    result = SUMMARY.synthesize_loaded(make_entries(gpu2=gpu2))
    assert result["decision"]["outcome"] == "gpu_pair_attribution_available"
    assert result["raw_contact_feasibility"]["repeatability"]["cuda:0"] == {
        "mode": "observed_raw_contact",
        "repeatable": True,
        "structure_exact": True,
        "numeric_within_tolerance": True,
        "unavailable_signature_exact": None,
    }


def test_unavailable_signature_ignores_global_callback_count() -> None:
    gpu1 = make_report("cuda:0", 1, raw_available=False)
    gpu2 = make_report("cuda:0", 2, raw_available=False)
    empty_event = {
        "physics_step": 20,
        "robot_ground_header_count": 0,
        "robot_ground_datum_count": 0,
        "headers": [],
    }
    gpu1["raw_contact_observation"]["events"] = [copy.deepcopy(empty_event)]
    gpu1["raw_contact_observation"]["callback_count"] = 1
    gpu2["raw_contact_observation"]["events"] = [
        copy.deepcopy(empty_event),
        {**copy.deepcopy(empty_event), "physics_step": 21},
    ]
    gpu2["raw_contact_observation"]["callback_count"] = 2
    gpu1["feasibility"] = PROBE.derive_feasibility(gpu1)
    gpu2["feasibility"] = PROBE.derive_feasibility(gpu2)
    PROBE.validate_report(gpu1)
    PROBE.validate_report(gpu2)
    result = SUMMARY.synthesize_loaded(make_entries(gpu1=gpu1, gpu2=gpu2))
    assert result["decision"]["outcome"] == "unavailable_on_gpu"
    signatures = [
        run["unavailable_signature"]
        for run in result["raw_contact_feasibility"]["runs"]
        if run["device"] == "cuda:0"
    ]
    assert all(
        signature["scope"] == "source_env_7_robot_ground_headers_only"
        and "callback_count" not in signature
        for signature in signatures
    )


def test_fake_synthesis_source_bundle_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _synthesis_source_bundle()
    first = SUMMARY.SYNTHESIS_SOURCE_BINDING_PATHS[0]
    bundle["source_binding_files"][first] = "f" * 64
    files = bundle["source_binding_files"]
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    bundle["source_bundle_sha256"] = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="committed blob mismatch"):
        SUMMARY.validate_synthesis_source_bundle(bundle)


def test_synthesis_provenance_uses_commit_blobs_not_windows_worktree_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "4" * 40
    committed_hashes = {
        path: hashlib.sha256(f"{path}\n".encode("utf-8")).hexdigest()
        for path in SUMMARY.SYNTHESIS_SOURCE_BINDING_PATHS
    }

    class Completed:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(args, **_kwargs):
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return Completed(commit)
        if args[:3] == ["git", "status", "--porcelain=v1"]:
            return Completed("")
        raise AssertionError(f"unexpected subprocess call: {args}")

    monkeypatch.setattr(SUMMARY.subprocess, "run", fake_run)
    monkeypatch.setattr(
        SUMMARY,
        "committed_synthesis_blob_sha256",
        lambda relative, observed_commit: (
            committed_hashes[relative]
            if observed_commit == commit
            else (_ for _ in ()).throw(AssertionError("commit changed"))
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("working-tree bytes must not be hashed")
        ),
    )

    bundle = ORIGINAL_SYNTHESIS_SOURCE_BUNDLE_PROVENANCE()

    assert bundle["source_binding_files"] == committed_hashes
    assert bundle["git_commit"] == commit
    assert bundle["clean"] is True


def _main_args(output: Path) -> list[str]:
    args: list[str] = []
    for index in range(4):
        args.extend(["--report", str(SUMMARY.RUNS_DIR / f"unused_{index}.json")])
    args.extend(["--output", str(output)])
    return args


def test_main_refuses_existing_output_and_existing_temp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SUMMARY, "synthesize", lambda _paths: {"status": "test"})
    output = SUMMARY.RUNS_DIR / "g009_r0_rev18_summary_existing_test.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    output.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    try:
        output.write_bytes(b"user-owned")
        with pytest.raises(FileExistsError, match="refusing to overwrite existing report"):
            SUMMARY.main(_main_args(output))
        assert output.read_bytes() == b"user-owned"
        output.unlink()

        temporary.write_bytes(b"user-temp")
        with pytest.raises(FileExistsError, match="temporary report"):
            SUMMARY.main(_main_args(output))
        assert not output.exists()
        assert temporary.read_bytes() == b"user-temp"
    finally:
        output.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def test_main_rolls_back_serialization_and_publish_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(SUMMARY, "synthesize", lambda _paths: {"status": "test"})
    output = SUMMARY.RUNS_DIR / "g009_r0_rev18_summary_rollback_test.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    output.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    original_dumps = SUMMARY.runtime_probe.json.dumps
    try:
        monkeypatch.setattr(
            SUMMARY.runtime_probe.json,
            "dumps",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
        )
        with pytest.raises(OSError, match="write failed"):
            SUMMARY.main(_main_args(output))
        assert not output.exists() and not temporary.exists()

        monkeypatch.setattr(SUMMARY.runtime_probe.json, "dumps", original_dumps)
        monkeypatch.setattr(
            SUMMARY.runtime_probe.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("publish failed")),
        )
        with pytest.raises(OSError, match="publish failed"):
            SUMMARY.main(_main_args(output))
        assert not output.exists() and not temporary.exists()
    finally:
        output.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
