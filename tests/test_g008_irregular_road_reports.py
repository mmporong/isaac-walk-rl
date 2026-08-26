from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reports" / "runs"
BASELINE = RUNS / "g008_irregular_road_baseline_friction_s1_e32_h500_s20260826.json"
TRAINED = RUNS / "g008_irregular_road_trained_s1_e32_h500_s20260826.json"
SUMMARY = RUNS / "g008_irregular_road_summary_s20260826.json"
TRAINING = RUNS / "g008_irregular_road_s1_finetune_friction_s1_e64_i300_s20260826.json"
VISUAL = RUNS / "g008_irregular_road_visual_evidence.json"
CAPTURES = (
    RUNS / "g008_irregular_road_baseline_capture.json",
    RUNS / "g008_irregular_road_trained_capture.json",
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


def test_irregular_road_report_binds_nonperiodic_field_and_runtime_contact() -> None:
    report = _load(BASELINE)
    assert report["status"] == "complete"
    assert report["protocol"] == "irregular_road_spatial_friction_height_v1"
    assert report["headless"] is True
    assert report["evaluation_seed"] == 20260826
    assert report["terrain_seed"] == 20260826
    assert report["num_envs"] == 32
    assert report["environments_per_direction"] == 8
    assert report["horizon_steps"] == 500
    assert report["warmup_steps"] == 50
    assert report["checkpoint"]["sha256"] == (
        "40af0a0f80489d705e1e8fdeedd2f765177d3d67bf757709b9195cc2bbeaaee0"
    )

    field = report["road_field"]
    assert field["bounds_m"] == {"x": [-28.0, 28.0], "y": [-28.0, 28.0]}
    assert field["cell_size_m"] == 0.25
    assert field["grid_cells"] == [224, 224]
    assert field["height_range_m"] == 0.08102830499410629
    assert field["local_slope_max_deg"] == 2.6989047527313232
    assert [(item["static"], item["dynamic"]) for item in field["friction_buckets"]] == [
        (0.25, 0.15),
        (0.4, 0.28),
        (0.6, 0.45),
        (0.8, 0.6),
    ]
    assert {item["cell_count"] for item in field["friction_buckets"]} == {12544}

    surface = report["contact_model"]["surface_readback"]
    assert surface["default_ground_collision_exists"] is False
    assert surface["height_scan_has_collision_api"] is False
    assert surface["one_material_binding_per_collision_mesh"] is True
    assert len(surface["collision_meshes"]) == 4
    assert surface["collision_face_total"] == surface["expected_face_total"] == 100352

    directions = report["directions"]
    assert all(direction["field_coverage_pass"] for direction in directions)
    assert all(direction["out_of_field_foot_sample_count"] == 0 for direction in directions)
    assert all(direction["contact_observation_available"] for direction in directions)
    assert all(direction["contact_foot_sample_count"] > 0 for direction in directions)
    assert all(all(count > 0 for count in direction["friction_bucket_foot_sample_counts"]) for direction in directions)
    assert all(
        direction["four_foot_material_diversity"]["all_same_frame_ratio"] > 0
        for direction in directions
    )
    assert any(
        direction["four_foot_material_diversity"]["all_four_distinct_frame_ratio"] > 0
        for direction in directions
    )
    assert any(
        direction["four_foot_material_diversity"]["maximum_simultaneous_bucket_count"] == 4
        for direction in directions
    )
    assert sum(direction["material_transition_count"] for direction in directions) > 0

    assert report["evaluation_source_sha256"] in _line_ending_equivalent_hashes(
        ROOT / "scripts" / "evaluate_g008_irregular_road.py"
    )
    assert report["road_generator_source_sha256"] in _line_ending_equivalent_hashes(
        ROOT / "src" / "isaac_walk_g008" / "irregular_road.py"
    )


def test_training_is_real_ppo_but_final_checkpoint_is_not_adopted() -> None:
    training = _load(TRAINING)
    summary = _load(SUMMARY)
    assert training["passed"] is True
    assert training["exit_code"] == 0
    assert training["headless"] is True
    assert training["num_envs"] == 64
    assert training["max_iterations"] == 300
    assert training["resume"] == {
        "enabled": True,
        "load_run": "2026-08-26_11-37-54_g008_friction_s1_finetune_command_s42_e1024_i300",
        "checkpoint": "model_2097.pt",
    }
    assert training["performance"]["mean_steps_per_second"] == 1132.32
    assert training["performance"]["final_mean_reward"] == 35.84
    assert training["performance"]["final_mean_episode_length"] == 984.0
    assert training["artifacts"]["checkpoint_sha256"] == (
        "1384b92107b776c6c18851abd17d47efc66b9ea42306f6ca354b0b525c7c4486"
    )

    assert summary["status"] == "complete"
    assert summary["protocol"] == "irregular_road_checkpoint_selection_v1"
    assert summary["training"]["iterations"] == 300
    assert summary["training"]["rollout_steps_per_env_iteration"] == 24
    assert summary["training"]["total_transitions"] == 460800
    assert summary["training"]["ppo_learning_epochs"] == 5
    assert summary["training"]["ppo_mini_batches_per_epoch"] == 4
    assert summary["selection"]["policy_id"] == "friction_s1"
    assert summary["selection"]["adopt_dedicated_training_checkpoint"] is False
    assert summary["selection"]["all_directions_gate_pass"] is False
    assert summary["aggregate_source_sha256"] in _line_ending_equivalent_hashes(
        ROOT / "scripts" / "aggregate_g008_irregular_road.py"
    )

    for section in ("training",):
        metadata = summary[section]["report"]
        path = ROOT / metadata["path"]
        assert path.is_file()
        assert _sha256(path) == metadata["sha256"]
    for item in summary["screening"]["reports"]:
        path = ROOT / item["report"]["path"]
        assert path.is_file()
        assert _sha256(path) == item["report"]["sha256"]
    for item in summary["full_evaluations"]:
        path = ROOT / item["report"]["path"]
        assert path.is_file()
        assert _sha256(path) == item["report"]["sha256"]

    baseline = next(item for item in summary["full_evaluations"] if item["policy_id"] == "friction_s1")
    trained = next(item for item in summary["full_evaluations"] if item["policy_id"] == "irregular_road_s1")
    assert baseline["direction_pass_count"] == 3
    assert baseline["fall_count"] == 0
    assert trained["direction_pass_count"] == 2
    assert trained["fall_count"] == 5
    assert _load(TRAINED)["all_directions_gate_pass"] is False


def test_smokes_and_visual_evidence_are_bound_to_public_derivatives() -> None:
    for name in (
        "g008_irregular_road_s1_smoke_e16_i1_s20260826.json",
        "g008_irregular_road_s1_resume_smoke_e64_i1_s20260826.json",
    ):
        assert _load(RUNS / name)["passed"] is True

    visual = _load(VISUAL)
    assert visual["status"] == "complete"
    assert visual["composition"]["matched_command_sequence"] is True
    assert visual["composition"]["synchronized_panels"] is True
    assert visual["local_composite"]["git_policy"] == "local_only"
    assert visual["local_composite"]["path"].startswith(
        "%USERPROFILE%\\IsaacLab\\logs\\visual_evidence\\g008\\"
    )
    assert visual["record_source_sha256"] in _line_ending_equivalent_hashes(
        ROOT / "scripts" / "record_g008_irregular_road.py"
    )
    assert visual["builder_source_sha256"] in _line_ending_equivalent_hashes(
        ROOT / "scripts" / "build_g008_irregular_road_media.py"
    )

    for report_path in CAPTURES:
        report = _load(report_path)
        assert report["status"] == "complete"
        assert report["profile"]["headless"] is True
        assert report["profile"]["local_video"]["git_policy"] == "local_only"
        assert report["record_source_sha256"] in _line_ending_equivalent_hashes(
            ROOT / "scripts" / "record_g008_irregular_road.py"
        )
        assert report["evaluator_source_sha256"] in _line_ending_equivalent_hashes(
            ROOT / "scripts" / "evaluate_g008_irregular_road.py"
        )
        assert report["road_generator_source_sha256"] in _line_ending_equivalent_hashes(
            ROOT / "src" / "isaac_walk_g008" / "irregular_road.py"
        )

    gif = visual["public_derivatives"]["gif"]
    sheet = visual["public_derivatives"]["contact_sheet"]
    for metadata in (gif, sheet):
        path = ROOT / metadata["path"]
        assert metadata["git_policy"] == "git_public"
        assert path.stat().st_size == metadata["bytes"]
        assert _sha256(path) == metadata["sha256"]
        assert metadata["bytes"] <= 10 * 1024 * 1024
    assert (ROOT / gif["path"]).read_bytes().startswith(b"GIF89a")
    assert (ROOT / sheet["path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(ROOT.rglob("*.mp4"))
