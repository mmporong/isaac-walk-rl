"""Front-calf damping extension after the rear-only rev32 smoke."""

from g009_r0_rev32 import configure_damping as configure_rear


def configure_damping(cfg):
    configure_rear(cfg)
    cfg.scene.robot.actuators["base_legs"].damping.update(FL_calf_joint=1.0, FR_calf_joint=1.0)


def actuator_readback(env):
    import torch

    actuator = env.scene["robot"].actuators["base_legs"]
    names = list(actuator.joint_names)
    if len(names) != 12 or len(set(names)) != 12:
        raise RuntimeError("twelve unique joints required")
    for name, values, target in (
        ("damping", actuator.damping, [1.0 if n.endswith("calf_joint") else 0.5 for n in names]),
        ("stiffness", actuator.stiffness, [25.0] * 12),
        ("effort_limit", actuator.effort_limit, [23.5] * 12),
    ):
        expected = torch.tensor(target, device=values.device, dtype=values.dtype).expand_as(values)
        if not torch.isfinite(values).all() or not torch.allclose(values, expected, atol=1e-7, rtol=0):
            raise RuntimeError(f"rev33 live {name} mismatch")
    return {
        "joint_names": names,
        "damping_by_joint": dict(zip(names, actuator.damping[0].detach().cpu().tolist())),
        "stiffness_by_joint": dict(zip(names, actuator.stiffness[0].detach().cpu().tolist())),
        "effort_limit_by_joint": dict(zip(names, actuator.effort_limit[0].detach().cpu().tolist())),
        "environments_checked": int(actuator.damping.shape[0]),
    }
