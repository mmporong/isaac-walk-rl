"""Isaac Lab environment configuration for the G009 S0 slope gate."""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaac_walk_g008.env_cfg import G008CommandEnvCfg

from .mdp import reset_root_state_on_slope
from .sim_terrain import G009_SLOPE_FIELD_PRIM, G009_SLOPE_SURFACE_PRIM, SlopeTerrainFieldCfg


@configclass
class G009SlopeWalkEnvCfg(G008CommandEnvCfg):
    """Go2 command task on one deterministic, instrumentable mountain slope."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 4.0
        self.scene.terrain = None
        self.scene.slope_field = AssetBaseCfg(
            prim_path=G009_SLOPE_FIELD_PRIM,
            spawn=SlopeTerrainFieldCfg(),
            collision_group=-1,
        )
        self.scene.height_scanner.mesh_prim_paths = [G009_SLOPE_SURFACE_PRIM]
        self.scene.contact_forces.track_pose = True

        self.curriculum.terrain_levels = None
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.physics_material.params.update(
            {
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "static_friction_range": (1.0, 1.0),
                "dynamic_friction_range": (1.0, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
                "make_consistent": True,
            }
        )
        self.events.reset_base = EventTerm(
            func=reset_root_state_on_slope,
            mode="reset",
            params={
                "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "asset_cfg": SceneEntityCfg("robot"),
                "terrain_cfg_name": "slope_field",
            },
        )


__all__ = ["G009SlopeWalkEnvCfg"]
