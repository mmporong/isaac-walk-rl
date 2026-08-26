import json
import math
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaac_walk_g008.contracts import (
    COMMAND_PRIMITIVE_PROBABILITY,
    COMMAND_PRIMITIVES,
    FRICTION_STAGES,
    LEG_MASS_STAGES,
    command_primitive_by_name,
    friction_stage,
    ideal_planar_acceleration_limit,
    leg_mass_stage,
    scaled_leg_group_mass_kg,
)


def test_command_primitives_cover_requested_directions_and_stand():
    assert {item.name for item in COMMAND_PRIMITIVES} == {
        "forward",
        "backward",
        "left_turn",
        "right_turn",
        "stand",
    }
    assert command_primitive_by_name("forward").velocity_mps_radps == (0.6, 0.0, 0.0)
    assert command_primitive_by_name("backward").velocity_mps_radps[0] < 0.0
    assert command_primitive_by_name("left_turn").velocity_mps_radps[2] > 0.0
    assert command_primitive_by_name("right_turn").velocity_mps_radps[2] < 0.0
    assert math.isclose(sum(item.weight for item in COMMAND_PRIMITIVES), 1.0)
    assert COMMAND_PRIMITIVE_PROBABILITY == pytest.approx(0.8)


def test_friction_ranges_expand_and_keep_nominal_values_in_scope():
    assert [item.stage for item in FRICTION_STAGES] == [1, 2, 3]
    nominal_static, nominal_dynamic = 0.8, 0.6
    for item in FRICTION_STAGES:
        assert item.static_range[0] <= nominal_static <= item.static_range[1]
        assert item.dynamic_range[0] <= nominal_dynamic <= item.dynamic_range[1]
    assert friction_stage(1).static_range == (0.72, 0.88)
    assert friction_stage(3).static_range == (0.5, 1.25)


def test_mass_ranges_are_nested_and_literature_envelope_is_last():
    assert [item.stage for item in LEG_MASS_STAGES] == [1, 2, 3]
    assert leg_mass_stage(1).scale_range == (0.95, 1.05)
    assert leg_mass_stage(3).scale_range == (0.8, 1.2)
    low_1, high_1 = scaled_leg_group_mass_kg(1)
    low_3, high_3 = scaled_leg_group_mass_kg(3)
    assert low_3 < low_1 < high_1 < high_3


def test_coulomb_acceleration_bound_is_dimensionally_consistent():
    assert ideal_planar_acceleration_limit(0.0) == 0.0
    assert ideal_planar_acceleration_limit(0.5) == pytest.approx(4.905)
    with pytest.raises(ValueError):
        ideal_planar_acceleration_limit(-0.1)


def test_manifest_matches_import_light_contracts():
    manifest = json.loads((ROOT / "configs" / "g008_locomotion_dynamics.json").read_text(encoding="utf-8"))
    primitives = manifest["part_1_command_suite"]["primitives"]
    assert [item["id"] for item in primitives] == [item.name for item in COMMAND_PRIMITIVES]
    assert [tuple(item["command"]) for item in primitives] == [item.velocity_mps_radps for item in COMMAND_PRIMITIVES]
    assert manifest["part_1_command_suite"]["primitive_probability"] == COMMAND_PRIMITIVE_PROBABILITY
    assert [tuple(item["static_friction"]) for item in manifest["part_2_friction"]["stages"]] == [
        item.static_range for item in FRICTION_STAGES
    ]
    assert [tuple(item["scale"]) for item in manifest["part_3_leg_link_mass"]["stages"]] == [
        item.scale_range for item in LEG_MASS_STAGES
    ]
    periodic = manifest["part_4_periodic_friction_stress_evaluation"]
    assert periodic["stripe_width_m"] == 0.5
    assert periodic["num_envs_per_case"] == 32
    assert periodic["collision_underlay"] is False
    assert periodic["result"]["lowest_completed_case"] == {
        "static": 0.2,
        "dynamic": 0.1,
    }
    assert periodic["result"]["friction_s1_all_direction_contiguous_pass_floor"] is None
    assert periodic["result"]["mixed_010_005_status"] == "unresolved_after_four_native_terminations"
    assert periodic["visual_evidence"]["local_mp4_only"] is True
    assert periodic["visual_evidence"]["capture_profile"] == "periodic_friction_s1_mu020_010"
    mass_screen = manifest["part_5_link_group_mass_sensitivity"]
    assert mass_screen["groups"] == ["hip", "thigh", "calf", "foot"]
    assert mass_screen["result"]["leg_mass_s1_stage_2_approved"] is False
    assert mass_screen["visual_evidence"]["capture_profiles"] == [
        "link_mass_hip_120",
        "link_mass_thigh_120",
        "link_mass_calf_120",
        "link_mass_foot_120",
    ]
