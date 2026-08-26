from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "reports" / "runs"
FRICTION_REPORT = RUNS / "g008_periodic_friction_sweep_command_vs_friction_s1_e32_h500_s20260826.json"
MASS_REPORT = RUNS / "g008_link_mass_sensitivity_command_vs_leg_mass_s1_e800_h300_s20260826.json"
CHECKPOINT_HASHES = {
    "command": "53cc09043088bcd53618d2ae1f90c7f2e91d01eab7090cc63922486942b2ed47",
    "friction_s1": "40af0a0f80489d705e1e8fdeedd2f765177d3d67bf757709b9195cc2bbeaaee0",
    "leg_mass_s1": "8976cfff6eee6d1a998c7aa554b23d98b01d3d64da02b43ac3133a9186ae97fa",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _line_ending_equivalent_hashes(path: Path) -> set[str]:
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    windows = normalized.replace(b"\n", b"\r\n")
    return {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(normalized).hexdigest(),
        hashlib.sha256(windows).hexdigest(),
    }


def test_periodic_friction_sweep_is_bound_to_runtime_and_checkpoints() -> None:
    report = _load(FRICTION_REPORT)
    assert report["status"] == "complete"
    assert report["protocol"] == "spatial_periodic_friction_stripes_sweep_v1"
    assert report["headless"] is True
    assert report["num_envs_per_case"] == 32
    assert report["environments_per_policy_direction"] == 4
    assert report["horizon_steps"] == 500
    assert report["warmup_steps"] == 50
    assert report["observation_corruption"] is False
    assert len(report["sweep"]) == 8
    assert report["contact_model"]["stripe_width_m"] == 0.5
    assert report["contact_model"]["minimum_local_stripe_count"] == 48
    assert report["contact_model"]["stripe_width_y_m"] == 4.0
    assert report["contact_model"]["geometry"].startswith("one pre-spawned static triangle mesh")
    assert report["contact_model"]["underlay"]["default_ground_collision_exists"] is False
    assert report["contact_model"]["underlay"]["height_scan_has_collision_api"] is False
    assert report["contact_model"]["field_phase"]["all_environment_origins_on_even_period_cells"] is True
    assert report["fall_detection"]["kinematic_base_height_min_m"] == 0.18
    assert report["fall_detection"]["kinematic_body_up_world_z_min"] == 0.5
    assert len(report["case_reports"]) == 7
    assert len(report["failed_evaluations"]) == 1
    for item in report["case_reports"]:
        raw_path = ROOT / item["path"]
        assert raw_path.is_file()
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == item["sha256"]
        raw = _load(raw_path)
        assert raw["status"] == "complete"
        assert raw["evaluation_source_sha256"] in _line_ending_equivalent_hashes(
            ROOT / "scripts" / "evaluate_g008_periodic_friction.py"
        )
    failed = report["failed_evaluations"][0]
    assert failed["case_id"] == "mixed_010_005"
    assert failed["attempt_count"] == 4
    assert failed["completed_steps_before_native_termination"] == 200
    failure_path = ROOT / failed["path"]
    assert hashlib.sha256(failure_path.read_bytes()).hexdigest() == failed["sha256"]
    failure_report = _load(failure_path)
    assert failure_report["evaluation_source_sha256"] in _line_ending_equivalent_hashes(
        ROOT / "scripts" / "evaluate_g008_periodic_friction.py"
    )
    assert report["evaluation_source_sha256"] in _line_ending_equivalent_hashes(
        ROOT / "scripts" / "evaluate_g008_periodic_friction.py"
    )
    assert report["aggregation_source_sha256"] in _line_ending_equivalent_hashes(
        ROOT / "scripts" / "aggregate_g008_periodic_friction.py"
    )

    policies = {policy["policy_id"]: policy for policy in report["policies"]}
    assert set(policies) == {"command", "friction_s1"}
    for policy_id, policy in policies.items():
        assert policy["checkpoint"]["sha256"] == CHECKPOINT_HASHES[policy_id]
        for case in policy["cases"]:
            assert case["all_directions_field_coverage_pass"] is True
            walk = {direction["id"]: direction for direction in case["directions"]}
            assert walk["forward"]["gate_pass"] is True
            assert walk["backward"]["gate_pass"] is True
            for direction in case["directions"]:
                assert direction["contact_observation_available"] is False
                assert direction["contact_foot_sample_count"] == 0
                assert direction["contact_slip_speed_mean_mps"] is None

    assert sum(
        direction["fall_count"]
        for case in policies["command"]["cases"]
        for direction in case["directions"]
    ) == 0
    assert sum(
        direction["fall_count"]
        for case in policies["friction_s1"]["cases"]
        for direction in case["directions"]
    ) == 1
    command_threshold = policies["command"]["threshold_summary"]
    assert command_threshold["baseline_gate_pass"] is False
    assert command_threshold["baseline_failure"]["failed_directions"] == ["left_turn"]
    assert command_threshold["contiguous_pass_floor"] is None
    friction_threshold = policies["friction_s1"]["threshold_summary"]
    assert friction_threshold["baseline_gate_pass"] is True
    assert friction_threshold["contiguous_pass_floor"] is None
    assert friction_threshold["first_failure"]["case_id"] == "mixed_070_050"
    assert friction_threshold["first_failure"]["failed_directions"] == ["right_turn"]
    assert friction_threshold["lowest_tested_passing"]["case_id"] == "mixed_060_040"
    direction_thresholds = policies["friction_s1"]["direction_thresholds"]
    for direction_id in ("forward", "backward", "left_turn"):
        assert direction_thresholds[direction_id]["contiguous_pass_floor"]["case_id"] == "mixed_020_010"
    assert direction_thresholds["right_turn"]["contiguous_pass_floor"] is None
    assert direction_thresholds["right_turn"]["first_failure"]["case_id"] == "mixed_070_050"


def test_link_mass_report_separates_training_range_from_group_screen() -> None:
    report = _load(MASS_REPORT)
    assert report["status"] == "complete"
    assert report["protocol"] == "controlled_link_group_mass_sensitivity_v1"
    assert report["headless"] is True
    assert report["num_envs"] == 800
    assert report["repetitions_per_policy_case_direction"] == 4
    assert report["horizon_steps"] == 300
    assert len(report["mass_cases"]) == 25
    assert report["interpretation_contract"]["training_range"].endswith("[0.95, 1.05]")
    assert report["evaluation_source_sha256"] in _line_ending_equivalent_hashes(
        ROOT / "scripts" / "evaluate_g008_link_mass_sensitivity.py"
    )
    assert max(item["inertia_scale_absolute_error_max"] for item in report["case_physics"]) <= 1.0e-8

    policies = {policy["policy_id"]: policy for policy in report["policies"]}
    assert set(policies) == {"command", "leg_mass_s1"}
    assert policies["command"]["checkpoint"]["sha256"] == CHECKPOINT_HASHES["command"]
    assert policies["leg_mass_s1"]["checkpoint"]["sha256"] == CHECKPOINT_HASHES["leg_mass_s1"]
    assert policies["command"]["nominal_all_directions_gate_pass"] is True
    assert policies["leg_mass_s1"]["nominal_all_directions_gate_pass"] is False

    for policy in policies.values():
        assert sum(direction["fall_count"] for case in policy["cases"] for direction in case["directions"]) == 0
        for case in policy["cases"]:
            directions = {direction["id"]: direction for direction in case["directions"]}
            assert directions["forward"]["gate_pass"] is True
            assert directions["backward"]["gate_pass"] is True
