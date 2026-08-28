from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "attribute_g009_r0_gate01", ROOT / "scripts" / "attribute_g009_r0_gate01.py"
)
assert SPEC and SPEC.loader
ATTRIBUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ATTRIBUTION)


def _event() -> dict:
    position = [-1.021, 0.0]
    lower = [-1.0, -2.0]
    upper = [1.0, 2.0]
    names = ["FL_hip_joint", "FL_thigh_joint"]
    return {
        "rollout_control_step": 7,
        "episode_control_step": 314,
        "sim_step_counter": 28,
        "env_index": 19,
        "pose_id": 0,
        "pose_name": "prone",
        "action_mode": ATTRIBUTION.ACTION_MODE,
        "joint_names": names,
        "joint_position_rad": position,
        "joint_lower_limit_rad": lower,
        "joint_upper_limit_rad": upper,
        "ppo_sample_pre_wrapper_clip": [0.5, -0.25],
        "action_term_raw_post_wrapper_clip": [0.5, -0.25],
        "processed_ema_target_rad": [-0.9, 0.1],
        "joint_velocity_rad_s": [-3.0, 1.0],
        "applied_torque_nm": [-12.0, 4.0],
        "joint_attributions": ATTRIBUTION.joint_limit_attributions(
            position=position,
            lower=lower,
            upper=upper,
            joint_names=names,
            margin_rad=0.01,
        ),
    }


def test_joint_limit_attribution_uses_exact_margin_predicate() -> None:
    records = ATTRIBUTION.joint_limit_attributions(
        position=[-1.01, 2.011, 0.0],
        lower=[-1.0, -2.0, -1.0],
        upper=[1.0, 2.0, 1.0],
        joint_names=["inside_at_margin", "above", "inside"],
        margin_rad=0.01,
    )

    assert len(records) == 1
    assert records[0]["joint_index"] == 1
    assert records[0]["joint_name"] == "above"
    assert records[0]["violated_side"] == "upper"
    assert records[0]["raw_excess_rad"] == pytest.approx(0.011)
    assert records[0]["margin_excess_rad"] == pytest.approx(0.001)
    assert records[0]["predicate_recomputed"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": [0.0], "lower": [], "upper": [1.0], "joint_names": ["j"], "margin_rad": 0.01},
        {"position": [float("nan")], "lower": [-1.0], "upper": [1.0], "joint_names": ["j"], "margin_rad": 0.01},
        {"position": [0.0], "lower": [1.0], "upper": [-1.0], "joint_names": ["j"], "margin_rad": 0.01},
        {"position": [0.0], "lower": [-1.0], "upper": [1.0], "joint_names": [""], "margin_rad": 0.01},
    ],
)
def test_joint_limit_attribution_rejects_malformed_state(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ATTRIBUTION.joint_limit_attributions(**kwargs)


def test_validator_accepts_exact_pre_reset_attribution() -> None:
    checks = ATTRIBUTION.validate_attribution_result(
        events=[_event()], observed_termination_keys={(7, 19)}, margin_rad=0.01
    )

    assert checks
    assert all(checks.values())


def test_validator_accepts_wrapper_clamp_for_out_of_range_ppo_sample() -> None:
    event = _event()
    event["ppo_sample_pre_wrapper_clip"][0] = 1.4
    event["action_term_raw_post_wrapper_clip"][0] = 1.0

    checks = ATTRIBUTION.validate_attribution_result(
        events=[event], observed_termination_keys=[(7, 19)], margin_rad=0.01
    )

    assert checks["post_wrapper_action_matches_clamped_ppo_sample"] is True


@pytest.mark.parametrize(
    "mutate, observed",
    [
        (lambda event: event.update(joint_attributions=[]), {(7, 19)}),
        (lambda event: event["joint_attributions"][0].update(raw_excess_rad=99.0), {(7, 19)}),
        (lambda event: event.update(pose_id=4, pose_name=None), {(7, 19)}),
        (lambda event: event["ppo_sample_pre_wrapper_clip"].__setitem__(0, float("nan")), {(7, 19)}),
        (lambda event: event["action_term_raw_post_wrapper_clip"].pop(), {(7, 19)}),
        (lambda event: event["action_term_raw_post_wrapper_clip"].__setitem__(0, 0.49), {(7, 19)}),
        (lambda event: None, {(7, 20)}),
    ],
)
def test_validator_fails_closed_on_invalid_or_missing_attribution(mutate, observed) -> None:
    event = _event()
    mutate(event)

    checks = ATTRIBUTION.validate_attribution_result(
        events=[event], observed_termination_keys=observed, margin_rad=0.01
    )

    assert not all(checks.values())


def test_validator_rejects_duplicate_attribution_for_one_termination() -> None:
    event = _event()
    checks = ATTRIBUTION.validate_attribution_result(
        events=[event, dict(event)], observed_termination_keys={(7, 19)}, margin_rad=0.01
    )

    assert checks["termination_and_attribution_counts_match"] is False


def test_pre_reset_observer_preserves_zero_active_terms_and_original_call() -> None:
    calls: list[tuple[list[int], object]] = []

    class Manager:
        active_terms: list[str] = []

        def record_pre_reset(self, env_ids, force_export_or_skip=None):
            calls.append((env_ids, force_export_or_skip))

    class Observer:
        def __init__(self):
            self.seen = []
            self.collector_state = {"observer_rng_neutral": True}

        def capture(self, env_ids):
            self.seen.append(env_ids)

    manager = Manager()
    observer = Observer()
    ATTRIBUTION.install_pre_reset_observer(manager, observer)
    manager.record_pre_reset([3, 9], force_export_or_skip=False)

    assert manager.active_terms == []
    assert observer.seen == [[3, 9]]
    assert calls == [([3, 9], False)]


def test_pre_reset_observer_rejects_rng_changing_active_terms() -> None:
    class Manager:
        active_terms = ["would_trigger_extra_observation_compute"]

        def record_pre_reset(self, env_ids, force_export_or_skip=None):
            del env_ids, force_export_or_skip

    with pytest.raises(RuntimeError, match="zero active recorder terms"):
        ATTRIBUTION.install_pre_reset_observer(Manager(), object())


def test_not_reproduced_is_not_a_pass() -> None:
    checks = ATTRIBUTION.validate_attribution_result(
        events=[], observed_termination_keys=set(), margin_rad=0.01
    )

    assert checks["hard_joint_limit_reproduced"] is False
    assert checks["termination_attribution_present"] is False
    assert all(checks.values()) is False


def test_training_hard_limit_scalar_binds_exactly_one_event() -> None:
    scaled, count = ATTRIBUTION.expected_training_hard_limit_event_count(
        {"maximum": 0.0416666679084301, "sample_count": 1, "nonzero_sample_count": 1}
    )

    assert scaled == pytest.approx(1.0)
    assert count == 1


def test_runtime_version_contract_is_explicit() -> None:
    assert ATTRIBUTION.EXPECTED_DEVICE == "cuda:0"
    assert ATTRIBUTION.EXPECTED_ISAACLAB_COMMIT == "90b79bb2d44feb8d833f260f2bf37da3487180ba"
    assert ATTRIBUTION.EXPECTED_ISAACLAB_TAG == "v2.1.1"
    assert ATTRIBUTION.EXPECTED_RSL_RL_VERSION == "2.3.3"
    assert len(ATTRIBUTION.EXPECTED_OFFICIAL_RUNTIME_SOURCE_SHA256) == 11
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in ATTRIBUTION.EXPECTED_OFFICIAL_RUNTIME_SOURCE_SHA256.values()
    )


def test_runtime_source_hash_contract_rejects_one_modified_file() -> None:
    actual = dict(ATTRIBUTION.EXPECTED_OFFICIAL_RUNTIME_SOURCE_SHA256)
    assert ATTRIBUTION.official_runtime_source_hashes_pinned(actual) is True

    actual["rsl_rl_on_policy_runner"] = "0" * 64
    assert ATTRIBUTION.official_runtime_source_hashes_pinned(actual) is False


def test_protocol_rejects_noncanonical_device() -> None:
    canonical = {
        "task": ATTRIBUTION.DEFAULT_TASK,
        "seed": ATTRIBUTION.EXPECTED_SEED,
        "num_envs": ATTRIBUTION.EXPECTED_NUM_ENVS,
        "rollout_steps": ATTRIBUTION.EXPECTED_ROLLOUT_STEPS,
        "headless": True,
        "device": ATTRIBUTION.EXPECTED_DEVICE,
    }
    ATTRIBUTION.validate_protocol_args(SimpleNamespace(**canonical))

    for device in ("cpu", "cuda:1"):
        with pytest.raises(ValueError, match="cuda:0"):
            ATTRIBUTION.validate_protocol_args(SimpleNamespace(**(canonical | {"device": device})))


@pytest.mark.parametrize(
    "hard",
    [
        {"maximum": True, "sample_count": 1, "nonzero_sample_count": 1},
        {"maximum": 1 / 24, "sample_count": 2, "nonzero_sample_count": 1},
        {"maximum": 1 / 24, "sample_count": 1, "nonzero_sample_count": 0},
        {"maximum": 0.0, "sample_count": 1, "nonzero_sample_count": 1},
    ],
)
def test_training_hard_limit_scalar_fails_closed(hard: dict) -> None:
    with pytest.raises(ValueError):
        ATTRIBUTION.expected_training_hard_limit_event_count(hard)


def test_canonical_output_rejects_outside_existing_and_temporary(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ATTRIBUTION.canonical_report_output(tmp_path / "outside.json")

    reports = ROOT / "reports" / "runs"
    existing = reports / "test_gate01_existing.json"
    temporary_target = reports / "test_gate01_temporary.json"
    temporary = temporary_target.with_suffix(".json.tmp")
    try:
        existing.write_text("{}", encoding="utf-8")
        with pytest.raises(FileExistsError):
            ATTRIBUTION.canonical_report_output(existing)
        temporary.write_text("{}", encoding="utf-8")
        with pytest.raises(FileExistsError):
            ATTRIBUTION.canonical_report_output(temporary_target)
    finally:
        existing.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def test_atomic_writer_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    ATTRIBUTION._write_json_atomic(target, {"outcome": "attributed"})
    assert json.loads(target.read_text(encoding="utf-8"))["outcome"] == "attributed"

    with pytest.raises(FileExistsError):
        ATTRIBUTION._write_json_atomic(target, {"outcome": "invalid"})
