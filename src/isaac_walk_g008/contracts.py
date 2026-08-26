"""Import-light contracts for G008 locomotion and dynamics stages."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose


@dataclass(frozen=True)
class CommandPrimitive:
    name: str
    velocity_mps_radps: tuple[float, float, float]
    weight: float


@dataclass(frozen=True)
class FrictionStage:
    stage: int
    name: str
    static_range: tuple[float, float]
    dynamic_range: tuple[float, float]


@dataclass(frozen=True)
class MassStage:
    stage: int
    name: str
    scale_range: tuple[float, float]


COMMAND_PRIMITIVE_PROBABILITY = 0.80
COMMAND_PRIMITIVES = (
    CommandPrimitive("forward", (0.60, 0.0, 0.0), 0.225),
    CommandPrimitive("backward", (-0.40, 0.0, 0.0), 0.225),
    CommandPrimitive("left_turn", (0.0, 0.0, 0.50), 0.225),
    CommandPrimitive("right_turn", (0.0, 0.0, -0.50), 0.225),
    CommandPrimitive("stand", (0.0, 0.0, 0.0), 0.100),
)

# The nominal Go2 contact material in Isaac Lab 2.1.1 is static=0.8,
# dynamic=0.6. Stage 3 reaches the contact-friction envelope used by
# Tan et al. (0.5 to 1.25) without exposing the first qualification run to it.
FRICTION_STAGES = (
    FrictionStage(1, "narrow", (0.72, 0.88), (0.52, 0.68)),
    FrictionStage(2, "moderate", (0.62, 1.00), (0.42, 0.78)),
    FrictionStage(3, "research_envelope", (0.50, 1.25), (0.30, 1.00)),
)

# 80 to 120 percent is the literature envelope used by Tan et al. and
# reproduced as the mass range in Xie et al. The narrow stage is run first.
LEG_MASS_STAGES = (
    MassStage(1, "narrow", (0.95, 1.05)),
    MassStage(2, "moderate", (0.90, 1.10)),
    MassStage(3, "literature_envelope", (0.80, 1.20)),
)

GO2_LEG_BODY_PATTERN = ".*_(hip|thigh|calf|foot)"
GO2_LEG_BODY_COUNT = 16
GO2_RUNTIME_BODY_MASSES_KG = {
    "base": 6.921,
    "hip_each": 0.678,
    "thigh_each": 1.152,
    "calf_each": 0.154,
    "foot_each": 0.040,
    "head_upper": 0.001,
    "head_lower": 0.001,
}


def command_primitive_by_name(name: str) -> CommandPrimitive:
    for primitive in COMMAND_PRIMITIVES:
        if primitive.name == name:
            return primitive
    raise KeyError(name)


def friction_stage(stage: int) -> FrictionStage:
    for value in FRICTION_STAGES:
        if value.stage == stage:
            return value
    raise KeyError(stage)


def leg_mass_stage(stage: int) -> MassStage:
    for value in LEG_MASS_STAGES:
        if value.stage == stage:
            return value
    raise KeyError(stage)


def ideal_planar_acceleration_limit(mu: float, gravity_mps2: float = 9.81) -> float:
    """Return the Coulomb-friction upper bound |a_xy| <= mu*g."""
    if mu < 0.0 or gravity_mps2 <= 0.0:
        raise ValueError("mu must be non-negative and gravity must be positive")
    return mu * gravity_mps2


def scaled_leg_group_mass_kg(stage: int) -> tuple[float, float]:
    """Return total leg-body mass bounds for the requested independent scale stage."""
    nominal = 4.0 * sum(
        GO2_RUNTIME_BODY_MASSES_KG[key]
        for key in ("hip_each", "thigh_each", "calf_each", "foot_each")
    )
    low, high = leg_mass_stage(stage).scale_range
    return nominal * low, nominal * high


def validate_contracts() -> None:
    if not 0.0 <= COMMAND_PRIMITIVE_PROBABILITY <= 1.0:
        raise ValueError("primitive probability must be in [0, 1]")
    if not isclose(sum(item.weight for item in COMMAND_PRIMITIVES), 1.0, abs_tol=1.0e-12):
        raise ValueError("command primitive weights must sum to one")
    if len({item.name for item in COMMAND_PRIMITIVES}) != len(COMMAND_PRIMITIVES):
        raise ValueError("command primitive names must be unique")
    if tuple(item.stage for item in FRICTION_STAGES) != (1, 2, 3):
        raise ValueError("friction stages must be 1, 2, 3")
    if tuple(item.stage for item in LEG_MASS_STAGES) != (1, 2, 3):
        raise ValueError("mass stages must be 1, 2, 3")
    for item in FRICTION_STAGES:
        if not (0.0 <= item.static_range[0] <= item.static_range[1]):
            raise ValueError(f"invalid static friction stage: {item}")
        if not (0.0 <= item.dynamic_range[0] <= item.dynamic_range[1]):
            raise ValueError(f"invalid dynamic friction stage: {item}")
    for item in LEG_MASS_STAGES:
        if not (0.0 < item.scale_range[0] <= 1.0 <= item.scale_range[1]):
            raise ValueError(f"mass stage must contain the nominal scale: {item}")


validate_contracts()
