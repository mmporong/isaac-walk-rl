"""Repo-local held-out rough terrain and Go2 evaluation task config."""

from __future__ import annotations

from .difficulty_rough import difficulty_rough_terrain
from .terrain_generator import EvidenceDifficultyTerrainGenerator

try:
    from isaaclab.terrains import HfTerrainBaseCfg, TerrainGeneratorCfg
    from isaaclab.utils import configclass
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
except ImportError:  # imported only for pure helper tests outside Isaac Sim
    HfTerrainBaseCfg = TerrainGeneratorCfg = UnitreeGo2RoughEnvCfg = object  # type: ignore[assignment,misc]

    def configclass(cls):  # type: ignore[no-redef]
        return cls


if HfTerrainBaseCfg is not object:
    @configclass
    class HfDifficultyAwareRandomRoughCfg(HfTerrainBaseCfg):
        """Difficulty-aware seeded height-field configuration."""

        function = difficulty_rough_terrain
        evidence_realization: int = 0


    HELD_OUT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
        class_type=EvidenceDifficultyTerrainGenerator,
        seed=20260824,
        curriculum=True,
        size=(24.0, 24.0),
        border_width=0.0,
        num_rows=10,
        num_cols=10,
        color_scheme="none",
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        sub_terrains={"evidence_rough": HfDifficultyAwareRandomRoughCfg(proportion=1.0)},
    )


    @configclass
    class UnitreeGo2EvidenceRoughEnvCfg(UnitreeGo2RoughEnvCfg):
        """Full rough task with only the held-out generator swapped after Go2 post-init."""

        def __post_init__(self):
            super().__post_init__()
            self.scene.terrain.terrain_generator = HELD_OUT_ROUGH_TERRAINS_CFG.copy()
else:
    HfDifficultyAwareRandomRoughCfg = None
    HELD_OUT_ROUGH_TERRAINS_CFG = None
    UnitreeGo2EvidenceRoughEnvCfg = None
