"""Register repository-local G006 tasks, then run Isaac Lab's official player."""

from __future__ import annotations

import hashlib
import runpy
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EXPECTED_UPSTREAM_PLAY_SHA256 = "0966feac5a96812fca880e3731e96b001918b57fa372f69e4cf5fdca538bd7bd"
REQUIRED_WINDOWS_KIT_ARGS = (
    "--/app/vulkan=false",
    "--/app/window/hideUi=true",
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def _configure_follow_camera() -> None:
    if "--headless" not in sys.argv or "--video" not in sys.argv:
        return

    import gymnasium as gym

    original_make = gym.make

    def make_with_follow_camera(*args: Any, **kwargs: Any) -> Any:
        env = original_make(*args, **kwargs)
        controller = env.unwrapped.viewport_camera_controller
        if controller is not None:
            controller.update_view_to_asset_root("robot")
            controller.update_view_location(eye=(3.0, 3.0, 2.0), lookat=(0.0, 0.0, 0.3))
        return env

    gym.make = make_with_follow_camera

    overrides = (
        "env.commands.base_velocity.debug_vis=false",
        "env.scene.replicate_physics=false",
        "env.viewer.origin_type=env",
        "env.viewer.eye=[5.0,5.0,3.5]",
        "env.viewer.lookat=[0.0,0.0,0.4]",
    )
    existing_keys = {arg.split("=", 1)[0] for arg in sys.argv if "=" in arg}
    sys.argv.extend(override for override in overrides if override.split("=", 1)[0] not in existing_keys)


def _configure_windows_headless_video() -> None:
    """Use the stable Windows renderer path for headless viewport capture."""
    if sys.platform != "win32" or "--headless" not in sys.argv or "--video" not in sys.argv:
        return

    kit_arg_locations: list[tuple[int, bool]] = []
    for index, argument in enumerate(sys.argv):
        if argument == "--kit_args":
            kit_arg_locations.append((index, True))
        elif argument.startswith("--kit_args="):
            kit_arg_locations.append((index, False))

    if len(kit_arg_locations) > 1:
        raise ValueError("provide --kit_args only once")

    if not kit_arg_locations:
        sys.argv.append(f"--kit_args={' '.join(REQUIRED_WINDOWS_KIT_ARGS)}")
        return

    option_index, separate_value = kit_arg_locations[0]
    if separate_value:
        value_index = option_index + 1
        if value_index >= len(sys.argv):
            raise ValueError("--kit_args requires a value")
        existing_value = sys.argv[value_index]
    else:
        value_index = option_index
        existing_value = sys.argv[option_index].split("=", 1)[1]

    tokens = existing_value.split()
    for required in REQUIRED_WINDOWS_KIT_ARGS:
        key, expected_value = required.split("=", 1)
        configured = [token for token in tokens if token == key or token.startswith(f"{key}=")]
        if any(token != required for token in configured):
            raise ValueError(f"conflicting --kit_args value for {key}; expected {expected_value}")
        if not configured:
            tokens.append(required)

    merged_value = " ".join(tokens)
    if separate_value:
        sys.argv[value_index] = merged_value
    else:
        sys.argv[value_index] = f"--kit_args={merged_value}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    from isaac_walk_g006 import register_tasks

    register_tasks()
    _configure_follow_camera()
    _configure_windows_headless_video()
    upstream = Path.cwd() / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"
    if not upstream.is_file():
        raise FileNotFoundError(f"official play.py not found under Isaac Lab working directory: {upstream}")
    upstream_sha256 = _sha256(upstream)
    if upstream_sha256 != EXPECTED_UPSTREAM_PLAY_SHA256:
        raise RuntimeError(
            "official play.py SHA-256 does not match the pinned Isaac Lab v2.1.1 entry point: "
            f"expected {EXPECTED_UPSTREAM_PLAY_SHA256}, got {upstream_sha256} ({upstream})"
        )
    if str(upstream.parent) not in sys.path:
        sys.path.insert(0, str(upstream.parent))
    runpy.run_path(str(upstream), run_name="__main__")


if __name__ == "__main__":
    main()
