from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "record_g008_stage_evidence.py"
SPEC = importlib.util.spec_from_file_location("record_g008_stage_evidence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

BUILDER_PATH = ROOT / "scripts" / "build_g008_stage_media.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("build_g008_stage_media", BUILDER_PATH)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
sys.modules[BUILDER_SPEC.name] = BUILDER
BUILDER_SPEC.loader.exec_module(BUILDER)


def test_stage_capture_profiles_cover_periodic_friction_and_all_mass_groups() -> None:
    profiles = MODULE.CAPTURE_PROFILES
    assert len(profiles) == 5
    assert profiles[0].profile_id == "periodic_friction_s1_mu020_010"
    assert profiles[0].stage == "periodic_friction"
    mass_profiles = [profile for profile in profiles if profile.stage == "link_mass"]
    assert [profile.mass_group for profile in mass_profiles] == ["hip", "thigh", "calf", "foot"]
    assert all(profile.mass_factor == 1.2 for profile in mass_profiles)
    assert len({profile.output_name for profile in profiles}) == len(profiles)


def test_profile_lookup_fails_closed() -> None:
    assert MODULE.profile_by_id("link_mass_thigh_120").mass_group == "thigh"
    with pytest.raises(ValueError, match="not found or duplicated"):
        MODULE.profile_by_id("missing")


def test_media_builder_requires_the_recorded_stage_order() -> None:
    assert BUILDER.EXPECTED_PROFILES["periodic_friction"] == (
        "periodic_friction_s1_mu020_010",
    )
    assert BUILDER.EXPECTED_PROFILES["link_mass"] == (
        "link_mass_hip_120",
        "link_mass_thigh_120",
        "link_mass_calf_120",
        "link_mass_foot_120",
    )
    assert set(BUILDER.QUANTITATIVE_REPORTS) == {"periodic_friction", "link_mass"}
