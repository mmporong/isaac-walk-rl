"""Terrain generator that preserves evidence after TerrainImporter drops it."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from .difficulty_rough import TERRAIN_EVIDENCE_REGISTRY, clear_terrain_evidence

_GENERATOR_EVIDENCE: dict[int, list[dict[str, Any]]] = {}
_LATEST_GENERATOR_ID: int | None = None


def latest_terrain_evidence() -> list[dict[str, Any]]:
    """Return a defensive copy of evidence from the latest completed generator."""

    if _LATEST_GENERATOR_ID is None:
        return []
    return copy.deepcopy(_GENERATOR_EVIDENCE.get(_LATEST_GENERATOR_ID, []))


try:
    from isaaclab.terrains import TerrainGenerator
except ImportError:  # permits pure unit tests on a non-Sim Python
    class TerrainGenerator:  # type: ignore[no-redef]
        pass


class EvidenceDifficultyTerrainGenerator(TerrainGenerator):
    """Assign stable realization IDs and retain raw terrain evidence."""

    def __init__(self, cfg: Any, device: str = "cpu"):
        global _LATEST_GENERATOR_ID
        self._evidence_realization = 0
        clear_terrain_evidence()  # registry must exist and be empty before super
        super().__init__(cfg=cfg, device=device)
        generator_id = id(self)
        annotated = []
        rows = int(cfg.num_rows)
        for index, item in enumerate(TERRAIN_EVIDENCE_REGISTRY):
            value = copy.deepcopy(item)
            value["row"] = index % rows
            value["col"] = index // rows
            annotated.append(value)
        self.evidence = annotated
        _GENERATOR_EVIDENCE[generator_id] = copy.deepcopy(annotated)
        _LATEST_GENERATOR_ID = generator_id

    def _get_terrain_mesh(self, difficulty: float, cfg: Any):
        # Curriculum generation is column-major.  Reuse the column realization
        # across rows so low/mid/high are paired terrain realizations.
        cfg.evidence_realization = self._evidence_realization // int(self.cfg.num_rows)
        self._evidence_realization += 1
        mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
        if TERRAIN_EVIDENCE_REGISTRY:
            digest = hashlib.sha256()
            digest.update(mesh.vertices.tobytes(order="C"))
            digest.update(mesh.faces.tobytes(order="C"))
            TERRAIN_EVIDENCE_REGISTRY[-1]["mesh_sha256"] = digest.hexdigest()
        return mesh, origin
