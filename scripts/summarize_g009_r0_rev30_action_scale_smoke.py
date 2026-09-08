#!/usr/bin/env python3
"""Validate and summarize one completed G009 R0 rev30 action-scale smoke report."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import validate_g009_r0_rev30_action_scale_smoke  # noqa: F401

_IMPLEMENTATION_PATH = SCRIPTS_ROOT / "summarize_g009_r0_rev29_action_scale_smoke.py"
_SPEC = importlib.util.spec_from_file_location(
    "_g009_rev30_action_scale_summary_implementation", _IMPLEMENTATION_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load action-scale summarizer: {_IMPLEMENTATION_PATH}")
_implementation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_implementation)


_implementation.SCHEMA_VERSION = "g009.r0.rev30.action_scale_smoke_summary.v1"
_implementation.REVISION = "rev30"
_implementation.PREVIOUS_REVISION = "rev29"
_implementation.PRELAUNCH_SCHEMA_VERSION = (
    "g009.r0.rev30.action_scale_smoke_prelaunch_validation.v1"
)
_implementation.PREREGISTRATION_RELATIVE_PATH = (
    "configs/g009_r0_rev30_action_scale_smoke.json"
)
_implementation.CANDIDATE_ACTION_SCALE = 0.60
_implementation.HISTORICAL_NOISE_FIELD = "rev29_step49_mean_noise_std"
_implementation.DEFAULT_PREREGISTRATION = (
    SCRIPTS_ROOT.parent / "configs" / "g009_r0_rev30_action_scale_smoke.json"
)
_implementation.EVIDENCE_ID = "G009-5-E023"

SCHEMA_VERSION = _implementation.SCHEMA_VERSION
REVISION = _implementation.REVISION
DEFAULT_PREREGISTRATION = _implementation.DEFAULT_PREREGISTRATION
file_sha256 = _implementation.file_sha256
validate_report = _implementation.validate_report
checkpoint_std_vector = _implementation.checkpoint_std_vector
write_json_no_overwrite = _implementation.write_json_no_overwrite
main = _implementation.main


if __name__ == "__main__":
    raise SystemExit(main())
