from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PACKAGE = ModuleType("isaac_walk_g006")
PACKAGE.__path__ = [str(ROOT / "src" / "isaac_walk_g006")]
sys.modules.setdefault("isaac_walk_g006", PACKAGE)

from isaac_walk_g006.evaluation.difficulty_rough import (  # noqa: E402
    clear_terrain_evidence,
    raw_difficulty_rough_height_field,
    record_height_field_evidence,
)


def cfg(realization: int) -> SimpleNamespace:
    return SimpleNamespace(
        size=(24.0, 24.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        evidence_realization=realization,
    )


def test_raw_field_exact_shape_dtype_seed_and_amplitude() -> None:
    low = raw_difficulty_rough_height_field(0.15, cfg(3))
    repeated = raw_difficulty_rough_height_field(0.15, cfg(3))
    other = raw_difficulty_rough_height_field(0.15, cfg(4))
    assert low.shape == (240, 240)
    assert low.dtype == np.int16
    assert np.array_equal(low, repeated)
    assert not np.array_equal(low, other)
    assert np.max(np.abs(low)) <= int((0.01 + 0.05 * 0.15) / 0.005)


def test_paired_realizations_have_strict_monotonic_metrics_and_unique_hashes() -> None:
    clear_terrain_evidence()
    entries = []
    for realization in range(10):
        for row, difficulty in ((1, 0.15), (4, 0.45), (8, 0.85)):
            local_cfg = cfg(realization)
            raw = raw_difficulty_rough_height_field(difficulty, local_cfg)
            entry = record_height_field_evidence(difficulty, local_cfg, raw)
            entry.update({"row": row, "col": realization, "mesh_sha256": f"{len(entries) + 1:064x}"})
            entries.append(entry)
    assert len({entry["raw_sha256"] for entry in entries}) == 30
    metrics = ("height_rms_m", "height_p90_abs_m", "face_normal_slope_rms_rad", "face_normal_slope_p90_rad")
    for col in range(10):
        by_row = {entry["row"]: entry for entry in entries if entry["col"] == col}
        for metric in metrics:
            assert by_row[1]["metrics"][metric] < by_row[4]["metrics"][metric] < by_row[8]["metrics"][metric]


def test_terrain_hash_tamper_is_rejected() -> None:
    module_path = ROOT / "scripts" / "summarize_g006.py"
    spec = importlib.util.spec_from_file_location("summarize_g006_for_terrain", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entries = []
    for col in range(10):
        for row, scale in ((1, 1.0), (4, 2.0), (8, 3.0)):
            entries.append({
                "row": row,
                "col": col,
                "raw_sha256": f"{len(entries) + 1:064x}",
                "mesh_sha256": f"{len(entries) + 101:064x}",
                "metrics": {
                    "height_rms_m": scale,
                    "height_p90_abs_m": scale,
                    "face_normal_slope_rms_rad": scale,
                    "face_normal_slope_p90_rad": scale,
                },
            })
    module.validate_terrain_evidence({"selected_tiles": entries})
    entries[-1]["raw_sha256"] = entries[0]["raw_sha256"]
    with pytest.raises(module.ValidationError, match="raw terrain hashes"):
        module.validate_terrain_evidence({"selected_tiles": entries})
