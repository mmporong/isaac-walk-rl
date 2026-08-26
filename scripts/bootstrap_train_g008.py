"""Register G008 tasks, then execute the pinned Isaac Lab RSL-RL trainer."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaac_walk_g008 import register_tasks  # noqa: E402


def main() -> None:
    register_tasks()
    upstream = Path.cwd() / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    if not upstream.is_file():
        raise FileNotFoundError(f"official train.py not found under Isaac Lab working directory: {upstream}")
    if str(upstream.parent) not in sys.path:
        sys.path.insert(0, str(upstream.parent))
    runpy.run_path(str(upstream), run_name="__main__")


if __name__ == "__main__":
    main()
