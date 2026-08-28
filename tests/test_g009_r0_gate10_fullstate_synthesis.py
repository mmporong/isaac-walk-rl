from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_g009_r0_gate10_fullstate_synthesis.py"


def _load():
    spec = importlib.util.spec_from_file_location("g009_gate10_fullstate_synthesis", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFY = _load()


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_bound_three_run_synthesis_is_reproducible_and_nonqualified() -> None:
    result = VERIFY.verify(VERIFY.DEFAULT_REPORTS, VERIFY.DEFAULT_SYNTHESIS)
    assert result["status"] == "pass"
    assert result["full_event_payload_sha256"] == (
        "28fd03a57d50738cedff01af51ea5fb2f4f1a9ba9d81ee56d84103de9acb2df2"
    )
    assert result["gate10_safety_passed"] is False
    assert result["learned_policy_qualified"] is False


def test_event_mutation_is_rejected(tmp_path: Path) -> None:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in VERIFY.DEFAULT_REPORTS]
    reports[2]["events"][0]["env_index"] += 1
    paths = tuple(
        _write_json(tmp_path / f"report_{index}.json", report)
        for index, report in enumerate(reports, start=1)
    )
    with pytest.raises(ValueError, match="full event payloads differ"):
        VERIFY.verify(paths, VERIFY.DEFAULT_SYNTHESIS)


def test_stale_synthesis_hash_is_rejected(tmp_path: Path) -> None:
    synthesis = json.loads(VERIFY.DEFAULT_SYNTHESIS.read_text(encoding="utf-8"))
    synthesis["full_event_payload_sha256"] = "0" * 64
    path = _write_json(tmp_path / "synthesis.json", synthesis)
    with pytest.raises(ValueError, match="full event payload hash mismatch"):
        VERIFY.verify(VERIFY.DEFAULT_REPORTS, path)
