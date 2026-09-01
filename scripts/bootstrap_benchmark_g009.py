"""Register G009 tasks, then execute Isaac Lab's pinned RSL-RL benchmark."""

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


EXPECTED_UPSTREAM_SHA256 = "2d5a88b9c07bfb38852082a0b9bf00f4213043b16ce0294776646ab06d351c82"
EXPECTED_ISAACLAB_COMMIT = "90b79bb2d44feb8d833f260f2bf37da3487180ba"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_isaaclab_checkout(root: Path) -> None:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_ISAACLAB_COMMIT:
        raise RuntimeError(
            f"Isaac Lab commit mismatch: expected={EXPECTED_ISAACLAB_COMMIT} actual={commit}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Isaac Lab tracked source is dirty")


def main() -> None:
    """Register the local task before delegating to the official benchmark."""

    register_tasks()
    upstream = Path.cwd() / "scripts" / "benchmarks" / "benchmark_rsl_rl.py"
    if not upstream.is_file():
        raise FileNotFoundError(
            f"official benchmark_rsl_rl.py not found under Isaac Lab working directory: {upstream}"
        )
    actual_sha256 = file_sha256(upstream)
    if actual_sha256 != EXPECTED_UPSTREAM_SHA256:
        raise RuntimeError(
            "official benchmark_rsl_rl.py SHA-256 mismatch: "
            f"expected={EXPECTED_UPSTREAM_SHA256} actual={actual_sha256}"
        )
    validate_isaaclab_checkout(Path.cwd())
    if str(upstream.parent) not in sys.path:
        sys.path.insert(0, str(upstream.parent))
    runpy.run_path(str(upstream), run_name="__main__")


if __name__ == "__main__":
    main()
