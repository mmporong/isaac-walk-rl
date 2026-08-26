from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reports" / "runs"
REWARD = RUNS / "g008_reward_contract_s20260826.json"
SUMMARY = RUNS / "g008_road_curriculum_summary_s20260826.json"
VISUAL = RUNS / "g008_road_curriculum_visual_evidence.json"
CAPTURES = (
    RUNS / "g008_road_g0_inherited_capture.json",
    RUNS / "g008_road_g0_turn_air_i2100_capture.json",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_ending_equivalent_hashes(path: Path) -> set[str]:
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    windows = normalized.replace(b"\n", b"\r\n")
    return {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(normalized).hexdigest(),
        hashlib.sha256(windows).hexdigest(),
    }


def _assert_bound_report(metadata: dict) -> dict:
    path = ROOT / metadata["path"]
    assert path.is_file()
    assert _sha256(path) == metadata["sha256"]
    return _load(path)


def test_reward_contract_records_exact_runtime_objective_and_ppo() -> None:
    report = _load(REWARD)
    assert report["schema_version"] == 2
    assert report["status"] == "complete"
    assert report["all_task_contracts_identical"] is True
    assert report["tasks"] == [
        "Isaac-G008-Velocity-Rough-Go2-CommandSuite-v0",
        "Isaac-G008-Velocity-IrregularRoad-Go2-G0-v0",
        "Isaac-G008-Velocity-IrregularRoad-Go2-S1-v0",
    ]

    contract = report["contract"]
    assert contract["physics_dt_s"] == 0.005
    assert contract["decimation"] == 4
    assert contract["step_dt_s"] == 0.02
    assert contract["episode_length_control_steps"] == 1000
    assert contract["reward_aggregation"] == "sum_i(step_dt * weight_i * raw_term_i)"

    terms = {term["name"]: term for term in contract["reward_terms"]}
    assert {name: term["weight"] for name, term in terms.items()} == {
        "track_lin_vel_xy_exp": 1.5,
        "track_ang_vel_z_exp": 0.75,
        "lin_vel_z_l2": -2.0,
        "ang_vel_xy_l2": -0.05,
        "dof_torques_l2": -0.0002,
        "dof_acc_l2": -2.5e-7,
        "action_rate_l2": -0.01,
        "feet_air_time": 0.01,
        "flat_orientation_l2": 0.0,
        "dof_pos_limits": 0.0,
    }
    assert contract["disabled_reward_terms"] == [
        "flat_orientation_l2",
        "dof_pos_limits",
        "undesired_contacts",
    ]
    assert terms["feet_air_time"]["raw_formula"] == (
        "sum_feet((last_air_time - 0.5 s) * first_contact) * "
        "I(||v_cmd_xy|| > 0.1 m/s)"
    )
    assert contract["turn_specific_readback"]["feet_air_time_active_for_pure_yaw"] is False
    assert contract["turn_specific_readback"]["explicit_roll_pitch_angle_penalty_active"] is False
    assert contract["turn_specific_readback"]["explicit_contact_conditioned_foot_slip_penalty_active"] is False

    assert contract["ppo"] == {
        "num_steps_per_env": 24,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "init_noise_std": 1.0,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "initial_learning_rate": 0.001,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "empirical_normalization": False,
    }

    variant = report["reward_variants"][0]
    assert variant["task"] == "Isaac-G008-Velocity-IrregularRoad-Go2-G0-TurnAir-v0"
    assert variant["changed_term"]["name"] == "feet_air_time"
    assert variant["changed_term"]["weight"] == 0.01
    assert variant["changed_term"]["params"]["yaw_command_threshold"] == 0.1
    assert variant["turn_specific_readback"]["feet_air_time_active_for_pure_yaw"] is True
    assert variant["all_non_reward_contract_fields_identical"] is True

    assert report["source_files"]["report_script"]["sha256"] in _line_ending_equivalent_hashes(
        ROOT / "scripts" / "report_g008_reward_contract.py"
    )
    assert report["source_files"]["project_reward_variant"]["sha256"] in _line_ending_equivalent_hashes(
        ROOT / "src" / "isaac_walk_g008" / "rewards.py"
    )


def test_curriculum_summary_binds_training_screening_and_full_evaluations() -> None:
    summary = _load(SUMMARY)
    assert summary["status"] == "complete"
    assert summary["protocol"] == "road_geometry_reward_curriculum_v1"
    _assert_bound_report(summary["reward_contract"])

    assert len(summary["training_runs"]) == 2
    for run in summary["training_runs"]:
        assert run["headless"] is True
        assert run["num_envs"] == 128
        assert run["iterations"] == 300
        assert run["total_transitions"] == 921600
        assert run["optimizer_mini_batch_updates"] == 6000
        training = _assert_bound_report(run["report"])
        assert training["passed"] is True
        assert training["artifacts"]["checkpoint_sha256"] == run["final_checkpoint"]["sha256"]

    screening = summary["screening"]
    assert screening["num_envs"] == 16
    assert screening["horizon_steps"] == 300
    assert screening["terrain_seed"] == 20260828
    for item in screening["reports"]:
        _assert_bound_report(item["report"])

    full = summary["full_evaluation"]
    assert full["num_envs"] == 32
    assert full["horizon_steps"] == 500
    assert full["warmup_steps"] == 50
    assert full["terrain_seeds"] == [20260826, 20260827, 20260828]
    candidates = {item["policy_id"]: item for item in full["candidates"]}
    assert {
        name: (
            item["terrain_seed_pass_count"],
            item["direction_pass_count"],
            item["fall_count"],
        )
        for name, item in candidates.items()
    } == {
        "friction_s1_g0": (2, 11, 1),
        "road_g0_i2100": (0, 9, 0),
        "road_g0_turn_air_i2100": (0, 9, 0),
        "road_g0_i2250": (0, 8, 4),
    }
    assert all(not item["qualified_for_next_friction_stage"] for item in candidates.values())
    for candidate in candidates.values():
        for item in candidate["reports"]:
            _assert_bound_report(item["report"])

    selection = summary["selection"]
    assert selection["policy_id"] == "friction_s1_g0"
    assert selection["checkpoint"]["sha256"] == (
        "40af0a0f80489d705e1e8fdeedd2f765177d3d67bf757709b9195cc2bbeaaee0"
    )
    assert selection["qualified_policy_count"] == 0
    assert selection["proceed_to_friction_f1"] is False
    assert summary["aggregate_source_sha256"] in _line_ending_equivalent_hashes(
        ROOT / "scripts" / "aggregate_g008_road_curriculum.py"
    )


def test_visual_evidence_is_bound_to_public_derivatives_and_local_only_video() -> None:
    visual = _load(VISUAL)
    assert visual["status"] == "complete"
    assert visual["stage"] == "road_reward_curriculum"
    assert visual["composition"]["seed"] == 20260826
    assert visual["composition"]["matched_command_sequence"] is True
    assert visual["composition"]["synchronized_panels"] is True
    assert visual["local_composite"]["git_policy"] == "local_only"
    assert visual["local_composite"]["path"].startswith(
        "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g008\\"
    )
    assert visual["record_source_sha256"] in _line_ending_equivalent_hashes(
        ROOT / "scripts" / "record_g008_road_curriculum.py"
    )
    assert visual["builder_source_sha256"] in _line_ending_equivalent_hashes(
        ROOT / "scripts" / "build_g008_road_curriculum_media.py"
    )

    for metadata in visual["capture_reports"]:
        _assert_bound_report(metadata)
    for capture_path in CAPTURES:
        capture = _load(capture_path)
        assert capture["status"] == "complete"
        assert capture["profile"]["headless"] is True
        assert capture["profile"]["total_steps"] == 900
        assert capture["profile"]["local_video"]["git_policy"] == "local_only"
        assert capture["record_source_sha256"] in _line_ending_equivalent_hashes(
            ROOT / "scripts" / "record_g008_road_curriculum.py"
        )

    gif = visual["public_derivatives"]["gif"]
    sheet = visual["public_derivatives"]["contact_sheet"]
    for metadata in (gif, sheet):
        path = ROOT / metadata["path"]
        assert metadata["git_policy"] == "git_public"
        assert path.stat().st_size == metadata["bytes"]
        assert _sha256(path) == metadata["sha256"]
        assert metadata["bytes"] <= 10 * 1024 * 1024
    assert gif["width"] == 720
    assert gif["height"] == 438
    assert gif["frames"] == 72
    assert (ROOT / gif["path"]).read_bytes().startswith((b"GIF87a", b"GIF89a"))
    assert (ROOT / sheet["path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(ROOT.rglob("*.mp4"))
