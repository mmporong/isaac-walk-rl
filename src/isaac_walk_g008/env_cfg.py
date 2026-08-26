"""G008 command, friction, and independent leg-link mass environments."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import UniformVelocityCommandCfg

from .commands import BalancedVelocityCommandCfg
from .contracts import GO2_LEG_BODY_PATTERN, friction_stage, leg_mass_stage
from .irregular_road import (
    DEFAULT_COLORS_RGB,
    DEFAULT_DYNAMIC_FRICTION,
    DEFAULT_STATIC_FRICTION,
    IRREGULAR_ROAD_MESH_PRIM,
    IRREGULAR_ROAD_PRIM,
    spawn_irregular_road_field,
)
from .rewards import feet_air_time_turn_aware


@configclass
class IrregularRoadFieldCfg(SpawnerCfg):
    """USD spawner parameters for the training-sized irregular road."""

    func: Callable = spawn_irregular_road_field
    x_min_m: float = -28.0
    x_max_m: float = 28.0
    y_min_m: float = -28.0
    y_max_m: float = 28.0
    cell_size_m: float = 0.25
    seed: int = 20260826
    env_spacing_m: float = 4.0
    static_friction: tuple[float, ...] = DEFAULT_STATIC_FRICTION
    dynamic_friction: tuple[float, ...] = DEFAULT_DYNAMIC_FRICTION
    colors_rgb: tuple[tuple[float, float, float], ...] = DEFAULT_COLORS_RGB
    crown_height_m: float = 0.015
    undulation_amplitude_m: float = 0.030
    roughness_amplitude_m: float = 0.012
    pothole_depth_m: float = 0.025


@configclass
class G008CommandEnvCfg(UnitreeGo2RoughEnvCfg):
    """Rough-terrain Go2 task with balanced forward, reverse, and turn commands."""

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None
        self.commands.base_velocity = BalancedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 6.0),
            rel_standing_envs=0.02,
            heading_command=False,
            rel_heading_envs=0.0,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.8, 0.8),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.8, 0.8),
                heading=None,
            ),
            debug_vis=False,
        )
        self.events.physics_material.params["make_consistent"] = True


class _G008FrictionMixin:
    def _configure_friction(self, stage_id: int):
        stage = friction_stage(stage_id)
        self.events.physics_material.params.update(
            {
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "static_friction_range": stage.static_range,
                "dynamic_friction_range": stage.dynamic_range,
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "make_consistent": True,
            }
        )


@configclass
class G008FrictionStage1EnvCfg(_G008FrictionMixin, G008CommandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_friction(1)


@configclass
class G008FrictionStage2EnvCfg(_G008FrictionMixin, G008CommandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_friction(2)


@configclass
class G008FrictionStage3EnvCfg(_G008FrictionMixin, G008CommandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_friction(3)


@configclass
class G008IrregularRoadStage1EnvCfg(G008CommandEnvCfg):
    """Go2 command task on one non-periodic 2-D friction and height field."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 4.0
        self.scene.terrain = None
        self.scene.height_scanner.mesh_prim_paths = [IRREGULAR_ROAD_MESH_PRIM]
        self.scene.irregular_road_field = AssetBaseCfg(
            prim_path=IRREGULAR_ROAD_PRIM,
            spawn=IrregularRoadFieldCfg(),
            collision_group=-1,
        )
        self.curriculum.terrain_levels = None
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
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25),
            "yaw": (-3.14, 3.14),
        }


@configclass
class G008IrregularRoadGeometryEnvCfg(G008IrregularRoadStage1EnvCfg):
    """Road-height isolation stage with one nominal ground material."""

    def __post_init__(self):
        super().__post_init__()
        road_cfg = self.scene.irregular_road_field.spawn
        road_cfg.static_friction = (0.8,)
        road_cfg.dynamic_friction = (0.6,)
        road_cfg.colors_rgb = ((0.45, 0.43, 0.40),)


@configclass
class G008IrregularRoadTurnAirTimeEnvCfg(G008IrregularRoadGeometryEnvCfg):
    """G0 road with one isolated pure-yaw air-time reward correction."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.func = feet_air_time_turn_aware
        self.rewards.feet_air_time.params["yaw_command_threshold"] = 0.1


class _G008LegMassMixin:
    def _configure_leg_mass(self, stage_id: int):
        stage = leg_mass_stage(stage_id)
        self.events.add_leg_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=GO2_LEG_BODY_PATTERN),
                "mass_distribution_params": stage.scale_range,
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )


@configclass
class G008LegMassStage1EnvCfg(_G008LegMassMixin, G008CommandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_leg_mass(1)


@configclass
class G008LegMassStage2EnvCfg(_G008LegMassMixin, G008CommandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_leg_mass(2)


@configclass
class G008LegMassStage3EnvCfg(_G008LegMassMixin, G008CommandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_leg_mass(3)
