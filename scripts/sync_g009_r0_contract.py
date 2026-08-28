#!/usr/bin/env python3
"""Generate or verify the canonical G009 R0 semantic contract manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaac_walk_g009.recover_contracts import canonical_sha256, recover_contract


DEFAULT_OUTPUT = REPO_ROOT / "configs" / "g009_r0.json"


def manifest() -> dict[str, object]:
    contract = recover_contract()
    return {
        "schema_version": 2,
        "goal_id": "g009",
        "stage_id": "R0",
        "contract_sha256": canonical_sha256(contract),
        "contract": contract,
    }


def serialized_manifest() -> str:
    return json.dumps(manifest(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = serialized_manifest()
    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != expected:
            print(f"OUT_OF_DATE: {output}")
            return 1
        print(f"PASS: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(expected, encoding="utf-8")
    temporary.replace(output)
    print(f"WROTE: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
