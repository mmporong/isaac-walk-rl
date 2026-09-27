"""Run the preregistered front-calf extension with the unchanged rev31 observer."""

import sys

from g009_r0_rev32 import REPO_ROOT
from g009_r0_rev33 import actuator_readback, configure_damping


def main():
    import gymnasium as gym
    import bootstrap_train_g009_rev31_attribution as baseline

    custom, _ = baseline.parse_and_strip_custom_args(sys.argv[1:])
    output = REPO_ROOT / "reports/runs" / f"{custom.runtime['run_name']}_intervention.json"
    if output.exists():
        raise FileExistsError(output)
    original_make = gym.make

    def intervention_make(task, **kwargs):
        configure_damping(kwargs["cfg"])
        env = original_make(task, **kwargs)
        before = actuator_readback(env.unwrapped)
        writer = baseline.AttributionReportWriter(output)

        def report():
            after = actuator_readback(env.unwrapped)
            return {
                "protocol": "g009_r0_rev33_front_calf_damping_smoke_v1",
                "diagnostic_only": True, "qualification_eligible": False,
                "candidate_damping_n_m_s_rad": 1.0,
                "runtime": custom.runtime, "actuator_before": before, "actuator_after": after,
                "actuator_readback_stable": before == after,
            }

        baseline.install_close_finalizer(env, writer, report)
        return env

    gym.make = intervention_make
    try:
        return baseline.main()
    finally:
        gym.make = original_make


if __name__ == "__main__":
    raise SystemExit(main())
