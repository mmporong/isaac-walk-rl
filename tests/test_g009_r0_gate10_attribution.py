from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ATTRIBUTION = _load("attribute_g009_r0_gate10", "scripts/attribute_g009_r0_gate10.py")
GATE01 = _load("attribute_g009_r0_gate01_regression", "scripts/attribute_g009_r0_gate01.py")


def _event() -> dict:
    names = [f"J{index}" for index in range(12)]
    names[0] = "FR_calf_joint"
    position = [-1.021] + [0.0] * 11
    lower = [-1.0] + [-2.0] * 11
    upper = [1.0] + [2.0] * 11
    action = [0.25] * 12
    ring = [
        {
            "iteration": (global_step - 1) // 24,
            "rollout_control_step": (global_step - 1) % 24 + 1,
            "global_action_step": global_step,
            "phase": "post_step_after_manager_reset_for_terminated_envs" if global_step < 31 else "terminal_pre_reset",
            "env_index": 706,
            "episode_control_step": 62 + global_step,
            "sim_step_counter": global_step * 4,
            "action_post_wrapper_clip": action,
            "processed_ema_target_rad": [0.035] * 12,
            "applied_torque_nm": [0.0] * 12,
            "joint_position_rad": position,
            "joint_velocity_rad_s": [0.0] * 12,
            "root_pose": {"position_w_m": [0.0, 0.0, 0.2], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "root_twist": {"linear_velocity_w_m_s": [0.0] * 3, "angular_velocity_w_rad_s": [0.0] * 3},
            "body_force_summary": {
                "body_names": ["base", "FR_calf"],
                "net_forces_w_n": [[0.0, 0.0, 0.0], [0.0, 0.0, 120.0]],
                "total_robot_mass_kg": 120.0 / 9.81,
                "dominant_body": "FR_calf",
                "dominant_force_n": 120.0,
                "dominant_force_bw": 1.0,
                "violated_joint_leg_chain_relation": "same_leg_chain",
            },
        }
        for global_step in range(16, 32)
    ]
    return {
        "iteration": 1,
        "rollout_control_step": 7,
        "global_action_step": 31,
        "episode_control_step": 93,
        "sim_step_counter": 124,
        "env_index": 706,
        "pose_id": 0,
        "pose_name": "prone",
        "action_mode": ATTRIBUTION.ACTION_MODE,
        "joint_names": names,
        "joint_position_rad": position,
        "joint_lower_limit_rad": lower,
        "joint_upper_limit_rad": upper,
        "joint_soft_lower_limit_rad": [-1.0] * 12,
        "joint_soft_upper_limit_rad": [1.0] * 12,
        "joint_attributions": ATTRIBUTION.joint_limit_attributions(
            position=position, lower=lower, upper=upper, joint_names=names, margin_rad=0.01
        ),
        "action_scale": 0.70,
        "ema_alpha": 0.2,
        "ppo_sample_pre_wrapper_clip": action,
        "action_term_raw_post_wrapper_clip": action,
        "pre_ema_scaled_target_rad": [0.175] * 12,
        "ema_previous_target_rad": [0.0] * 12,
        "ema_expected_target_rad": [0.035] * 12,
        "processed_ema_target_rad": [0.035] * 12,
        "joint_velocity_rad_s": [0.0] * 12,
        "applied_torque_nm": [0.0] * 12,
        "root_pose": {"position_w_m": [0.0, 0.0, 0.2], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "root_twist": {"linear_velocity_w_m_s": [0.0] * 3, "angular_velocity_w_rad_s": [0.0] * 3},
        "contact_sensor_history": {
            "sensor_name": "contact_forces",
            "history_length": 3,
            "body_names": ["base", "FR_calf"],
            "net_forces_w_history_n": [[[0.0, 0.0, 0.0], [0.0, 0.0, 120.0]]] * 3,
        },
        "preceding_control_step_ring": ring,
    }


def test_protocol_constants_pin_full_gate10_training() -> None:
    assert ATTRIBUTION.EXPECTED_SEED == 42
    assert ATTRIBUTION.EXPECTED_DEVICE == "cuda:0"
    assert ATTRIBUTION.EXPECTED_NUM_ENVS == 1024
    assert ATTRIBUTION.EXPECTED_ROLLOUT_STEPS == 24
    assert ATTRIBUTION.EXPECTED_ITERATIONS == 10
    assert ATTRIBUTION.EXPECTED_ACT_COUNT == 240
    assert ATTRIBUTION.EXPECTED_UPDATE_COUNT == 10
    assert ATTRIBUTION.RING_BUFFER_STEPS == 16
    assert ATTRIBUTION.EXPECTED_CONTACT_HISTORY_LENGTH == 3
    assert ATTRIBUTION.parse_prelaunch_output(["--help"]).name == "_gate10_attribution_help_only.json"


def test_historical_rev12_training_core_hashes_reject_rev13_source() -> None:
    provenance = ATTRIBUTION.training_core_provenance()
    assert len(ATTRIBUTION.TRAINING_CORE_SHA256) == 10
    assert provenance["expected_files"] == ATTRIBUTION.TRAINING_CORE_SHA256
    assert set(provenance["actual_files"]) == set(ATTRIBUTION.TRAINING_CORE_SHA256)
    assert {
        path
        for path, expected_sha256 in ATTRIBUTION.TRAINING_CORE_SHA256.items()
        if provenance["actual_files"][path] != expected_sha256
    } == {
        "configs/g009_r0.json",
        "src/isaac_walk_g009/recover_contracts.py",
        "src/isaac_walk_g009/recover_env_cfg.py",
    }
    assert provenance["sha256"] != ATTRIBUTION.EXPECTED_TRAINING_CORE_SHA256
    assert provenance["exact_match"] is False


def test_protocol_rejects_any_noncanonical_execution_argument() -> None:
    canonical = {
        "task": ATTRIBUTION.DEFAULT_TASK,
        "seed": 42,
        "num_envs": 1024,
        "rollout_steps": 24,
        "iterations": 10,
        "headless": True,
        "device": "cuda:0",
    }
    ATTRIBUTION.validate_protocol_args(SimpleNamespace(**canonical))
    for key, value in (
        ("seed", 43),
        ("num_envs", 512),
        ("rollout_steps", 23),
        ("iterations", 9),
        ("headless", False),
        ("device", "cpu"),
    ):
        with pytest.raises(ValueError, match="24 steps x 10"):
            ATTRIBUTION.validate_protocol_args(SimpleNamespace(**(canonical | {key: value})))


def test_exact_event_is_accepted_with_three_tuple_multiset() -> None:
    checks = ATTRIBUTION.validate_attribution_result(
        events=[_event()], observed_termination_keys=[(1, 7, 706)], margin_rad=0.01
    )
    assert checks
    assert all(checks.values())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda event: event.update(iteration=2),
        lambda event: event.update(pose_id=3, pose_name="right_side"),
        lambda event: event.update(global_action_step=32),
        lambda event: event.update(joint_attributions=[]),
        lambda event: event["joint_attributions"][0].update(raw_excess_rad=99.0),
        lambda event: event["ppo_sample_pre_wrapper_clip"].__setitem__(0, 2.0),
        lambda event: event["processed_ema_target_rad"].__setitem__(0, 0.03),
        lambda event: (
            event["ema_expected_target_rad"].__setitem__(0, 0.03),
            event["processed_ema_target_rad"].__setitem__(0, 0.03),
        ),
        lambda event: event["contact_sensor_history"].update(history_length=2),
        lambda event: event["contact_sensor_history"]["net_forces_w_history_n"][0].pop(),
        lambda event: event["preceding_control_step_ring"].clear(),
        lambda event: event["preceding_control_step_ring"][0].update(global_action_step=31),
        lambda event: event["preceding_control_step_ring"][0]["processed_ema_target_rad"].pop(),
        lambda event: event["preceding_control_step_ring"][0]["applied_torque_nm"].__setitem__(0, float("nan")),
        lambda event: event["preceding_control_step_ring"][0]["body_force_summary"].update(dominant_force_bw=9.0),
        lambda event: event["root_pose"]["position_w_m"].__setitem__(0, float("nan")),
    ],
)
def test_event_validation_fails_closed(mutation) -> None:
    event = _event()
    mutation(event)
    checks = ATTRIBUTION.validate_attribution_result(
        events=[event], observed_termination_keys=[(1, 7, 706)], margin_rad=0.01
    )
    assert not all(checks.values())


def test_duplicate_or_missing_attribution_fails_multiset_exactness() -> None:
    event = _event()
    duplicate = ATTRIBUTION.validate_attribution_result(
        events=[event, dict(event)], observed_termination_keys=[(1, 7, 706)], margin_rad=0.01
    )
    missing = ATTRIBUTION.validate_attribution_result(
        events=[], observed_termination_keys=[(1, 7, 706)], margin_rad=0.01
    )
    assert duplicate["termination_key_multiset_matches_attribution"] is False
    assert missing["termination_key_multiset_matches_attribution"] is False


def test_global_step_31_requires_full_16_frame_ring() -> None:
    event = _event()
    event["preceding_control_step_ring"] = event["preceding_control_step_ring"][1:]

    checks = ATTRIBUTION.validate_attribution_result(
        events=[event], observed_termination_keys=[(1, 7, 706)], margin_rad=0.01
    )

    assert checks["preceding_16_step_ring_valid"] is False
    assert checks["predicate_recomputed_and_records_valid"] is False


@pytest.mark.parametrize(
    ("case", "mutation"),
    [
        (
            "wrong_sensor_name",
            lambda event: event["contact_sensor_history"].update(sensor_name="fabricated_sensor"),
        ),
        (
            "duplicate_body_names",
            lambda event: event["contact_sensor_history"].update(body_names=["base", "base"]),
        ),
        (
            "ring_body_names_mismatch",
            lambda event: event["preceding_control_step_ring"][0]["body_force_summary"].update(
                body_names=["base", "fabricated_body"]
            ),
        ),
        (
            "fabricated_phase",
            lambda event: event["preceding_control_step_ring"][3].update(phase="terminal_pre_reset"),
        ),
        (
            "terminal_action_mismatch",
            lambda event: event["preceding_control_step_ring"][-1]["action_post_wrapper_clip"].__setitem__(0, 0.5),
        ),
        (
            "terminal_joint_mismatch",
            lambda event: event["preceding_control_step_ring"][-1]["joint_position_rad"].__setitem__(0, -1.03),
        ),
        (
            "terminal_root_mismatch",
            lambda event: event["preceding_control_step_ring"][-1]["root_pose"]["position_w_m"].__setitem__(2, 0.3),
        ),
        (
            "terminal_counter_mismatch",
            lambda event: event["preceding_control_step_ring"][-1].update(sim_step_counter=999),
        ),
        (
            "terminal_ema_target_mismatch",
            lambda event: event["preceding_control_step_ring"][-1]["processed_ema_target_rad"].__setitem__(0, 0.03),
        ),
        (
            "terminal_torque_mismatch",
            lambda event: event["preceding_control_step_ring"][-1]["applied_torque_nm"].__setitem__(0, 1.0),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_contact_ring_lineage_and_terminal_binding_fail_closed(case: str, mutation) -> None:
    del case
    event = _event()
    mutation(event)

    checks = ATTRIBUTION.validate_attribution_result(
        events=[event], observed_termination_keys=[(1, 7, 706)], margin_rad=0.01
    )

    assert not all(checks.values())


def test_hard_series_is_derived_from_iteration_event_counts() -> None:
    series = ATTRIBUTION.hard_series_from_keys([(1, 4, 1), (2, 5, 2), (3, 6, 3)])
    assert series == list(ATTRIBUTION.EXPECTED_HARD_SERIES)


def test_bound_tensorboard_series_requires_exact_order_and_recomputed_summary() -> None:
    samples = [
        {"step": index, "value": value, "wall_time": float(index)}
        for index, value in enumerate(ATTRIBUTION.EXPECTED_BOUND_TENSORBOARD_HARD_SERIES)
    ]
    stored = ATTRIBUTION._scalar_summary(list(ATTRIBUTION.EXPECTED_BOUND_TENSORBOARD_HARD_SERIES))
    assert ATTRIBUTION.validate_bound_hard_series(samples, stored) == list(
        ATTRIBUTION.EXPECTED_BOUND_TENSORBOARD_HARD_SERIES
    )

    swapped = [dict(sample) for sample in samples]
    swapped[0]["value"], swapped[1]["value"] = swapped[1]["value"], swapped[0]["value"]
    with pytest.raises(ValueError, match="value order"):
        ATTRIBUTION.validate_bound_hard_series(swapped, stored)
    with pytest.raises(ValueError, match="stored hard-limit mean"):
        ATTRIBUTION.validate_bound_hard_series(samples, stored | {"mean": 99.0})


def test_historical_identity_requires_every_exact_anchor() -> None:
    canonical = {
        "hard_event_counts": list(ATTRIBUTION.EXPECTED_HARD_EVENT_COUNTS),
        "hard_series": list(ATTRIBUTION.EXPECTED_HARD_SERIES),
        "model_hashes": {0: ATTRIBUTION.EXPECTED_MODEL_0_SHA256, 9: ATTRIBUTION.EXPECTED_MODEL_9_SHA256},
        "act_count": 240,
        "update_count": 10,
        "core_sha256": ATTRIBUTION.EXPECTED_TRAINING_CORE_SHA256,
    }
    assert all(ATTRIBUTION.historical_identity_checks(**canonical).values())
    mutations = [
        {"hard_event_counts": [0] * 10},
        {"hard_series": [0.0] * 10},
        {"model_hashes": {0: "0" * 64, 9: ATTRIBUTION.EXPECTED_MODEL_9_SHA256}},
        {"model_hashes": {0: ATTRIBUTION.EXPECTED_MODEL_0_SHA256, 9: "0" * 64}},
        {"act_count": 239},
        {"update_count": 9},
        {"core_sha256": "0" * 64},
    ]
    for mutation in mutations:
        assert not all(ATTRIBUTION.historical_identity_checks(**(canonical | mutation)).values())


def test_pre_reset_wrapper_calls_original_and_preserves_rng_neutral_contract() -> None:
    calls = []

    class Manager:
        active_terms = []

        def record_pre_reset(self, env_ids, force_export_or_skip=None):
            calls.append((env_ids, force_export_or_skip))

    class Observer:
        collector_state = {"observer_rng_neutral": True}

        def capture(self, env_ids):
            calls.append(("capture", env_ids))

    manager = Manager()
    ATTRIBUTION.install_pre_reset_observer(manager, Observer())
    manager.record_pre_reset([3, 9], False)
    assert manager.active_terms == []
    assert calls == [("capture", [3, 9]), ([3, 9], False)]


def test_diagnostic_and_official_source_provenance_are_explicit() -> None:
    diagnostic = ATTRIBUTION.diagnostic_source_provenance()
    assert set(diagnostic["files"]) == set(ATTRIBUTION.DIAGNOSTIC_SOURCE_PATHS)
    assert len(diagnostic["sha256"]) == 64
    official = ATTRIBUTION.official_runtime_source_provenance()
    assert official == ATTRIBUTION.EXPECTED_OFFICIAL_RUNTIME_SOURCE_SHA256


def test_pre_reset_wrapper_rejects_active_recorder_terms() -> None:
    manager = SimpleNamespace(active_terms=["term"], record_pre_reset=lambda *_: None)
    with pytest.raises(RuntimeError, match="zero active recorder terms"):
        ATTRIBUTION.install_pre_reset_observer(manager, object())


def test_leg_chain_relation_is_explicit() -> None:
    assert ATTRIBUTION.leg_chain_relation("FR_calf_joint", "FR_thigh") == "same_leg_chain"
    assert ATTRIBUTION.leg_chain_relation("FR_calf_joint", "FL_thigh") == "other_leg_chain"
    assert ATTRIBUTION.leg_chain_relation("FR_calf_joint", "base") == "base"


def test_canonical_output_and_atomic_writer_never_overwrite(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ATTRIBUTION.canonical_report_output(tmp_path / "outside.json")
    target = tmp_path / "report.json"
    ATTRIBUTION._write_json_atomic(target, {"gate10_safety_passed": False})
    assert json.loads(target.read_text(encoding="utf-8"))["gate10_safety_passed"] is False
    with pytest.raises(FileExistsError):
        ATTRIBUTION._write_json_atomic(target, {})


def test_gate01_predicate_regression_remains_unchanged() -> None:
    gate01 = GATE01.joint_limit_attributions(
        position=[-1.021], lower=[-1.0], upper=[1.0], joint_names=["FR_calf_joint"], margin_rad=0.01
    )
    gate10 = ATTRIBUTION.joint_limit_attributions(
        position=[-1.021], lower=[-1.0], upper=[1.0], joint_names=["FR_calf_joint"], margin_rad=0.01
    )
    assert gate10 == gate01


def test_source_declares_official_learn_original_update_and_no_load() -> None:
    source = (ROOT / "scripts" / "attribute_g009_r0_gate10.py").read_text(encoding="utf-8")
    assert "runner.learn(num_learning_iterations=args.iterations" in source
    assert "result = original_update(*update_args, **update_kwargs)" in source
    assert "runner.load(" not in source
    assert '"gate10_safety_passed": False' in source
    assert '"learned_policy_qualified": False' in source
