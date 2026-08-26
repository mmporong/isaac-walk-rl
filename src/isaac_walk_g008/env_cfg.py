"""G008 command, friction, and independent leg-link mass environments."""

from __future__ import annotations

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import UniformVelocityCommandCfg

from .commands import BalancedVelocityCommandCfg
from .contracts import GO2_LEG_BODY_PATTERN, friction_stage, leg_mass_stage


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
