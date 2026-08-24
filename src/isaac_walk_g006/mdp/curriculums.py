"""Pure schedule and read-only curriculum logging for G006."""

from __future__ import annotations

import torch


def push_schedule_for_step(common_step_counter: int) -> tuple[int, float, float]:
    if common_step_counter < 12_000:
        return 0, 0.10, 0.25
    if common_step_counter < 24_000:
        return 1, 0.25, 0.50
    return 2, 0.50, 1.00


def _terrain_level_stats(env) -> dict[str, float]:
    levels = getattr(env.scene.terrain, "terrain_levels", None)
    if levels is None or levels.numel() == 0:
        return {
            "mean_level": 0.0,
            "p10_level": 0.0,
            "p50_level": 0.0,
            "p90_level": 0.0,
            "low_fraction": 0.0,
            "mid_fraction": 0.0,
            "high_fraction": 0.0,
        }
    values = levels.float()
    return {
        "mean_level": float(values.mean().item()),
        "p10_level": float(torch.quantile(values, 0.10).item()),
        "p50_level": float(torch.quantile(values, 0.50).item()),
        "p90_level": float(torch.quantile(values, 0.90).item()),
        "low_fraction": float(((values >= 0) & (values <= 2)).float().mean().item()),
        "mid_fraction": float(((values >= 3) & (values <= 6)).float().mean().item()),
        "high_fraction": float(((values >= 7) & (values <= 9)).float().mean().item()),
    }


def log_g006_state(env, env_ids) -> dict[str, float | int]:
    """Return logger state without mutating curriculum, terrain, or event config."""
    result: dict[str, float | int] = _terrain_level_stats(env)
    push_enabled = getattr(env.cfg.events, "push_robot", None) is not None
    stage = push_schedule_for_step(int(env.common_step_counter))[0] if push_enabled else -1
    result["stage"] = stage
    counts = getattr(env, "_g006_push_counts", [0, 0, 0])
    sums = getattr(env, "_g006_push_magnitude_sum", [0.0, 0.0, 0.0])
    minima = getattr(env, "_g006_push_magnitude_min", [float("inf"), float("inf"), float("inf")])
    maxima = getattr(env, "_g006_push_magnitude_max", [0.0, 0.0, 0.0])
    for index in range(3):
        count = int(counts[index]) if push_enabled else 0
        result[f"events_stage_{index}"] = count
        result[f"magnitude_mean_stage_{index}"] = float(sums[index] / count) if count else 0.0
        result[f"magnitude_min_stage_{index}"] = float(minima[index]) if count else 0.0
        result[f"magnitude_max_stage_{index}"] = float(maxima[index]) if count else 0.0
    return result
