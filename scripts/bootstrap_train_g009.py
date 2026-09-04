"""Register G009 tasks, then execute the pinned Isaac Lab RSL-RL trainer."""

from __future__ import annotations

import hashlib
import runpy
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaac_walk_g009 import register_tasks  # noqa: E402


EXPECTED_ISAACLAB_COMMIT = "90b79bb2d44feb8d833f260f2bf37da3487180ba"
EXPECTED_TRAIN_SHA256 = "8b995f75ac57ce7403973ff1f3f2715fbff9563ef2cdcdc321a7edc5dd15f5df"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_stdout(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Isaac Lab git command failed: "
            f"args={list(arguments)!r} exit={result.returncode} "
            f"stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}"
        )
    return result.stdout


def validate_upstream(isaaclab_root: Path) -> Path:
    root = isaaclab_root.resolve()
    upstream = root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    if not upstream.is_file():
        raise FileNotFoundError(f"official train.py not found under Isaac Lab working directory: {upstream}")
    commit = _git_stdout(root, "rev-parse", "HEAD").strip()
    if commit != EXPECTED_ISAACLAB_COMMIT:
        raise RuntimeError(
            f"Isaac Lab commit mismatch: expected={EXPECTED_ISAACLAB_COMMIT} actual={commit}"
        )
    tracked_status = _git_stdout(root, "status", "--porcelain=v1", "--untracked-files=no")
    if tracked_status.strip():
        raise RuntimeError("Isaac Lab tracked worktree must be clean")
    actual_sha256 = _file_sha256(upstream)
    if actual_sha256 != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(
            f"official train.py SHA-256 mismatch: expected={EXPECTED_TRAIN_SHA256} actual={actual_sha256}"
        )
    return upstream


def main() -> None:
    register_tasks()
    upstream = validate_upstream(Path.cwd())
    if str(upstream.parent) not in sys.path:
        sys.path.insert(0, str(upstream.parent))
    runpy.run_path(str(upstream), run_name="__main__")


if __name__ == "__main__":
    main()
