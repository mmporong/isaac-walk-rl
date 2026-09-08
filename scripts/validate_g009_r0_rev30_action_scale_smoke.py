#!/usr/bin/env python3
"""Fail-closed prelaunch validation for the G009 R0 rev30 action-scale smoke."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

_IMPLEMENTATION_PATH = SCRIPTS_ROOT / "validate_g009_r0_rev29_action_scale_smoke.py"
_SPEC = importlib.util.spec_from_file_location(
    "_g009_rev30_action_scale_validation_implementation", _IMPLEMENTATION_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load action-scale validator: {_IMPLEMENTATION_PATH}")
_implementation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_implementation)


_implementation.SCHEMA_VERSION = "g009.r0.rev30.action_scale_smoke_preregistration.v1"
_implementation.EVIDENCE_ID = "G009-5-E023"
_implementation.REVISION = "rev30"
_implementation.PREVIOUS_REVISION = "rev29"
_implementation.PREVIOUS_ACTION_SCALE = 0.65
_implementation.CANDIDATE_ACTION_SCALE = 0.60
_implementation.CANONICAL_CONTRACT_ID = "g009_r0_recover_rev30"
_implementation.PRELAUNCH_SCHEMA_VERSION = (
    "g009.r0.rev30.action_scale_smoke_prelaunch_validation.v1"
)
_implementation.DEFAULT_PREREGISTRATION = (
    _implementation.REPO_ROOT / "configs" / "g009_r0_rev30_action_scale_smoke.json"
)
_implementation.EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "cf38974514c595251d40d981548b57699bc17716e99d2d871f4d04f9299506dd"
)
_implementation.EXPECTED_HISTORICAL_SHA256 = {
    "rev27_diagnostic_report": "ffd373d3937558aa71b5afccc70468aff068a28443f728a94343e488046bc315",
    "rev29_training_report": "575c38b015f0d04552bd7276524339a6ec40ed1336903c580ca38b401bce852d",
    "rev29_rejection_report": "9da6ecd5122c36973637a9b0e45f39c911e1287ae3dcc3487d565833156c7160",
}
_implementation.HISTORICAL_NOISE_FIELD = "rev29_step49_mean_noise_std"
_implementation.EXPECTED_HISTORICAL_NOISE = 0.4784904718399048

SCHEMA_VERSION = _implementation.SCHEMA_VERSION
EVIDENCE_ID = _implementation.EVIDENCE_ID
REVISION = _implementation.REVISION
DEFAULT_PREREGISTRATION = _implementation.DEFAULT_PREREGISTRATION
EXPECTED_SOURCE_MANIFEST_SHA256 = _implementation.EXPECTED_SOURCE_MANIFEST_SHA256
file_sha256 = _implementation.file_sha256
canonical_json_sha256 = _implementation.canonical_json_sha256


def load_preregistration(path: Path = DEFAULT_PREREGISTRATION):
    return _implementation.load_preregistration(path)


validate_semantics = _implementation.validate_semantics
validate_canonical_manifest = _implementation.validate_canonical_manifest
validate_historical_evidence = _implementation.validate_historical_evidence
validate_source_state = _implementation.validate_source_state
validate_upstream = _implementation.validate_upstream


def validate(
    path: Path = DEFAULT_PREREGISTRATION,
    isaac_lab_path: Path = Path.home() / "IsaacLab",
):
    return _implementation.validate(path, isaac_lab_path)


main = _implementation.main


if __name__ == "__main__":
    raise SystemExit(main())
