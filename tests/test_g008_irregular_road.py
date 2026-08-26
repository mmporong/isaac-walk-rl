from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaac_walk_g008.irregular_road import (  # noqa: E402
    IrregularRoadSpec,
    build_mesh_arrays,
    field_summary,
    generate_irregular_road,
    sample_cells,
)

EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "evaluate_g008_irregular_road", ROOT / "scripts" / "evaluate_g008_irregular_road.py"
)
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)


@pytest.fixture(scope="module")
def field():
    return generate_irregular_road(
        IrregularRoadSpec(
            x_min_m=-4.0,
            x_max_m=4.0,
            y_min_m=-4.0,
            y_max_m=4.0,
            cell_size_m=0.25,
            seed=20260826,
        )
    )


def test_field_is_deterministic_nonflat_and_uses_every_friction_bucket(field) -> None:
    duplicate = generate_irregular_road(field.spec)
    assert np.array_equal(field.heights_m, duplicate.heights_m)
    assert np.array_equal(field.material_indices, duplicate.material_indices)
    summary = field_summary(field)
    assert summary["height_range_m"] > 0.05
    assert summary["adjacent_height_delta_max_m"] < 0.06
    assert summary["local_slope_max_deg"] < 15.0
    assert [item["index"] for item in summary["friction_buckets"] if item["cell_count"]] == [0, 1, 2, 3]
    assert all(0.20 < item["cell_ratio"] < 0.30 for item in summary["friction_buckets"])


def test_four_foot_footprints_include_same_and_different_material_cases(field) -> None:
    offsets = np.asarray(((0.20, 0.13), (0.20, -0.13), (-0.20, 0.13), (-0.20, -0.13)))
    unique_counts = []
    for x in np.arange(-3.5, 3.51, 0.25):
        for y in np.arange(-3.5, 3.51, 0.25):
            materials, _, _ = sample_cells(field, offsets + np.asarray((x, y)))
            unique_counts.append(len(set(int(value) for value in materials)))
    assert 1 in unique_counts
    assert 4 in unique_counts
    assert set(unique_counts) == {1, 2, 3, 4}


def test_mesh_has_two_triangles_and_one_material_per_cell(field) -> None:
    arrays = build_mesh_arrays(field)
    cell_count = field.material_indices.size
    assert arrays["points"].shape == (33 * 33, 3)
    assert arrays["faces"].shape == (2 * cell_count, 3)
    assert arrays["face_materials"].shape == (2 * cell_count,)
    assert np.array_equal(arrays["face_materials"][0::2], field.material_indices.ravel())
    assert np.array_equal(arrays["face_materials"][1::2], field.material_indices.ravel())


def test_sampling_returns_bilinear_height_and_rejects_out_of_bounds(field) -> None:
    positions = np.asarray(((0.0, 0.0), (1.125, -2.375)))
    materials, heights, slopes = sample_cells(field, positions)
    assert materials.shape == (2,)
    assert heights.shape == (2,)
    assert slopes.shape == (2,)
    with pytest.raises(ValueError, match="outside"):
        sample_cells(field, np.asarray(((99.0, 99.0),)))


def test_invalid_specs_are_rejected() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        generate_irregular_road(IrregularRoadSpec(x_min_m=0.0, x_max_m=1.1, cell_size_m=0.25))
    with pytest.raises(ValueError, match="static >= dynamic"):
        generate_irregular_road(
            IrregularRoadSpec(static_friction=(0.2,), dynamic_friction=(0.3,), colors_rgb=((0.1, 0.1, 0.1),))
        )


def test_diversity_summary_distinguishes_same_and_four_different_frames() -> None:
    summary = EVALUATOR.summarize_diversity_counts({1: 3, 2: 4, 3: 2, 4: 1})
    assert summary["frame_counts"] == {"1": 3, "2": 4, "3": 2, "4": 1}
    assert summary["all_same_frame_ratio"] == pytest.approx(0.3)
    assert summary["all_four_distinct_frame_ratio"] == pytest.approx(0.1)
    assert summary["maximum_simultaneous_bucket_count"] == 4
