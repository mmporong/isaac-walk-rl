"""Go2 rough configurations whose normalized difference is only push_robot."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

from . import mdp


@configclass
class G006RoughCommonEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = None
        self.curriculum.g006_state = CurrTerm(func=mdp.log_g006_state)


@configclass
class G006RoughBaselineEnvCfg(G006RoughCommonEnvCfg):
    pass


@configclass
class G006RoughPushCurriculumEnvCfg(G006RoughCommonEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot = EventTerm(
            func=mdp.push_robot_body_xy,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            is_global_time=False,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
