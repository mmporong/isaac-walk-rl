"""Run the pinned benchmark and persist matrix-observation runtime telemetry."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bootstrap_benchmark_g009 import main as benchmark_main  # noqa: E402
from isaac_walk_g009.matrix_gate01 import (  # noqa: E402
    MATRIX_CRITIC_OBSERVATION_DIM,
    MATRIX_OBSERVATION_DIM,
    MATRIX_POLICY_OBSERVATION_DIM,
    NOMINAL_BODY_WEIGHT_N,
    ORDERED_BODY_NAMES,
    ORDERED_BODY_NAMES_SHA256,
    TERRAIN_FILTER_PATHS,
    reset_runtime_telemetry,
    runtime_telemetry,
)


def _argument_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError as error:
        raise RuntimeError(f"required argument missing: {name}") from error
    if index + 1 >= len(sys.argv):
        raise RuntimeError(f"required argument value missing: {name}")
    return sys.argv[index + 1]


def telemetry_path(run_name: str) -> Path:
    return Path.home() / "IsaacLab" / "logs" / "harness" / f"{run_name}.matrix_gate01.json"


def write_telemetry(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"matrix telemetry output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"matrix telemetry output already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    run_name = _argument_value("--run_name")
    output = telemetry_path(run_name)
    reset_runtime_telemetry()
    completed = False
    try:
        benchmark_main()
        completed = True
    finally:
        snapshot = runtime_telemetry()
        write_telemetry(
            output,
            {
                "schema_version": "g009.r0.rev25.matrix_gate01_runtime.v1",
                "evidence_id": "G009-5-E018",
                "run_name": run_name,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "repository_commit": subprocess.run(
                    ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "benchmark_completed": completed,
                "terrain_filter_paths": list(TERRAIN_FILTER_PATHS),
                "matrix_observation_dimension": MATRIX_OBSERVATION_DIM,
                "policy_observation_dimension": MATRIX_POLICY_OBSERVATION_DIM,
                "critic_observation_dimension": MATRIX_CRITIC_OBSERVATION_DIM,
                "expected_policy_matrix_slice_from_term_order": [83, 140],
                "raw_authority_frame": "world",
                "policy_projection_frame": "base",
                "nominal_body_weight_n": NOMINAL_BODY_WEIGHT_N,
                "bounding": "elementwise_tanh",
                "ordered_body_names": list(ORDERED_BODY_NAMES),
                "ordered_body_names_sha256": ORDERED_BODY_NAMES_SHA256,
                "runtime": snapshot,
            },
        )


if __name__ == "__main__":
    main()
