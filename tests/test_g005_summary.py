from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_go2_policy as evaluation  # noqa: E402
import summarize_reward_ablation as summary  # noqa: E402


METRICS = summary.METRIC_NAMES


def manifest_fixture() -> dict:
    manifest = {
        "schema_version": 1,
        "experiment_name": "synthetic",
        "seeds": [42, 43, 44],
        "evaluation_protocol": {
            "task": "Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
            "seed": 20260824,
            "num_envs": 260,
            "horizon_steps": 1000,
            "step_dt": 0.02,
            "command_grid_conditions": 26,
            "environments_per_condition": 10,
            "command_grid": {
                "vx_mps": [-1.0, 0.0, 1.0],
                "vy_mps": [-0.5, 0.0, 0.5],
                "yaw_rate_radps": [-0.5, 0.0, 0.5],
                "exclude": [[0.0, 0.0, 0.0]],
                "environments_per_condition": 10,
            },
        },
        "practical_thresholds": {"tracking_relative": 0.05, "energy_relative": 0.05, "fall_absolute": 0.02},
        "variants": [
            {"name": "baseline", "weights": {"torque": -0.0002, "action_rate": -0.01, "feet_air_time": 0.25}},
            {"name": "no_torque", "weights": {"torque": 0.0, "action_rate": -0.01, "feet_air_time": 0.25}},
            {"name": "no_action_rate", "weights": {"torque": -0.0002, "action_rate": 0.0, "feet_air_time": 0.25}},
            {"name": "no_feet_air_time", "weights": {"torque": -0.0002, "action_rate": -0.01, "feet_air_time": 0.0}},
        ],
    }
    manifest["variant_sha256"] = {
        variant["name"]: summary.canonical_sha256(variant) for variant in manifest["variants"]
    }
    return manifest


DENOMINATORS = {
    "sample_count": "active pre-step environment states",
    "fall_trial_rate": "base_contact events / trials_started",
    "survival_rate": "1 - fall_trial_rate",
    "trials_started": "fixed first episodes",
    "fall_timeout_overlap_count": "fall + timeout - reset union",
}


def condition_metrics(value: float) -> dict:
    metrics = {metric: value for metric in METRICS}
    metrics.update({
        "sample_count": 100,
        "first_contact_count": 2,
        "fall_count": 0,
        "timeout_count": 10,
        "reset_count": 10,
        "fall_timeout_overlap_count": 0,
        "trials_started": 10,
        "fall_trial_rate": 0.0,
        "survival_rate": 1.0,
    })
    return metrics


def overall_metrics(value: float) -> dict:
    metrics = condition_metrics(value)
    metrics.update({
        "sample_count": 2600,
        "first_contact_count": 52,
        "timeout_count": 260,
        "reset_count": 260,
        "trials_started": 260,
    })
    return metrics


def evidence_fixture(root: Path) -> tuple[dict, dict, Path]:
    manifest = manifest_fixture()
    manifest_path = root / "g005_reward_ablation.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol_sha = summary.canonical_sha256(manifest["evaluation_protocol"])
    config_sha = summary.canonical_sha256(manifest)
    config_file_sha = summary.file_sha256(manifest_path)
    jobs = []
    for variant_index, variant in enumerate(manifest["variants"]):
        for seed_index, seed in enumerate(manifest["seeds"]):
            checkpoint_path = root / f"{variant['name']}-s{seed}.pt"
            checkpoint_path.write_bytes(f"checkpoint-{variant_index}-{seed}".encode())
            checkpoint_sha = summary.file_sha256(checkpoint_path)
            variant_sha = summary.canonical_sha256(variant)
            value = float(seed_index + 1 + (0 if variant["name"] == "baseline" else 2))
            by_command = [
                {"command": command, **condition_metrics(value)}
                for command in evaluation.command_grid(manifest["evaluation_protocol"]["command_grid"])
            ]
            report = {
                "schema_version": 1,
                "variant": variant["name"],
                "training_seed": seed,
                "evaluation_seed": 20260824,
                "protocol_sha256": protocol_sha,
                "protocol_compliant": True,
                "config_sha256": config_sha,
                "config_file_sha256": config_file_sha,
                "variant_config_sha256": variant_sha,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint": {"reference": str(checkpoint_path), "sha256": checkpoint_sha},
                "task": manifest["evaluation_protocol"]["task"],
                "num_envs": 260,
                "horizon_steps": 1000,
                "step_dt": 0.02,
                "effective_weights": variant["weights"],
                "denominators": DENOMINATORS,
                "metrics": {"overall": overall_metrics(value), "by_command": by_command},
                "runtime_evidence": {
                    "exit_code": 0,
                    "app_close_completed": True,
                    "finalized_after_process_exit": True,
                    "gpu_recovered_to_baseline": True,
                    "process_recovered": True,
                    "gpu_after": {"measurement_complete": True},
                    "fatal_scan": {"measurement_complete": True, "count": 0, "patterns": []},
                },
            }
            report_path = root / f"{variant['name']}-s{seed}.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            jobs.append({
                "id": f"{variant['name']}-s{seed}",
                "variant": variant["name"],
                "seed": seed,
                "run_name": f"g005_production_{variant['name']}_s{seed}",
                "status": "complete",
                "config_sha256": config_sha,
                "config_file_sha256": config_file_sha,
                "protocol_sha256": protocol_sha,
                "variant_config_sha256": variant_sha,
                "overrides": [],
                "training_command": ["pwsh", "run_training.ps1"],
                "training_command_sha256": "1" * 64,
                "evaluation_command": ["python", "evaluate_go2_policy.py"],
                "evaluation_command_sha256": "2" * 64,
                "evaluation_script_sha256": "3" * 64,
                "report_path": f"g005_production_{variant['name']}_s{seed}.json",
                "checkpoint_sha256": checkpoint_sha,
                "evaluation_report_path": report_path.name,
                "attempts": [],
                "hard_blocked": False,
                "updated_at": "2026-08-24T12:00:00+09:00",
            })
    queue = {
        "schema_version": 2,
        "config_path": str(manifest_path),
        "config_sha256": config_sha,
        "config_file_sha256": config_file_sha,
        "protocol_sha256": protocol_sha,
        "mode": "production",
        "status": "complete",
        "lock_owner": {"pid": 1234, "started_at": "2026-08-24T11:00:00+09:00"},
        "created_at": "2026-08-24T11:00:00+09:00",
        "updated_at": "2026-08-24T12:00:00+09:00",
        "jobs": jobs,
    }
    queue_path = root / "queue.json"
    queue_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest, queue, queue_path


def load_first_report(root: Path, queue: dict) -> tuple[Path, dict]:
    path = root / queue["jobs"][0]["evaluation_report_path"]
    return path, json.loads(path.read_text(encoding="utf-8"))


def save_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report), encoding="utf-8")


class EvaluationMetricTests(unittest.TestCase):
    def test_zero_denominators_are_null(self) -> None:
        metrics = evaluation.finalize_accumulator(evaluation._new_accumulator(0))
        self.assertIsNone(metrics["lin_vel_rmse_mps"])
        self.assertIsNone(metrics["feet_air_time_raw_mean"])
        self.assertIsNone(metrics["mean_air_time_at_first_contact_s"])
        self.assertIsNone(metrics["fall_trial_rate"])
        self.assertIsNone(metrics["survival_rate"])

    def test_fall_timeout_overlap_uses_union_reset_count(self) -> None:
        rates = summary.compute_trial_rates(fall_count=1, timeout_count=1, reset_count=1, trials_started=2)
        self.assertEqual(rates["fall_count"], 1)
        self.assertEqual(rates["timeout_count"], 1)
        self.assertEqual(rates["reset_count"], 1)
        self.assertEqual(rates["fall_trial_rate"], 0.5)
        self.assertEqual(rates["survival_rate"], 0.5)

    def test_command_grid_has_26_unique_nonzero_conditions(self) -> None:
        grid = evaluation.command_grid(manifest_fixture()["evaluation_protocol"]["command_grid"])
        self.assertEqual(len(grid), 26)
        self.assertEqual(len({item["id"] for item in grid}), 26)
        self.assertNotIn((0.0, 0.0, 0.0), {(item["vx_mps"], item["vy_mps"], item["yaw_rate_radps"]) for item in grid})

    def test_command_grid_rejects_wrong_exclusion(self) -> None:
        spec = dict(manifest_fixture()["evaluation_protocol"]["command_grid"])
        spec["exclude"] = []
        with self.assertRaisesRegex(ValueError, "exclude"):
            evaluation.command_grid(spec)


class SummaryTests(unittest.TestCase):
    def test_sample_std_and_paired_delta(self) -> None:
        self.assertEqual(summary.sample_std([1.0, 2.0, 3.0]), 1.0)
        with tempfile.TemporaryDirectory() as temp:
            manifest, queue, queue_path = evidence_fixture(Path(temp))
            result = summary.summarize(manifest, queue, queue_path)
        baseline = result["results"]["baseline"]["metrics"]["lin_vel_rmse_mps"]
        paired = result["results"]["no_torque"]["metrics"]["lin_vel_rmse_mps"]["paired_vs_baseline"]
        self.assertEqual(baseline["mean"], 2.0)
        self.assertEqual(baseline["sample_std"], 1.0)
        self.assertEqual(baseline["sample_variance"], 1.0)
        self.assertEqual(paired["deltas_by_seed"], {"42": 2.0, "43": 2.0, "44": 2.0})
        self.assertEqual(paired["mean_delta"], 2.0)
        self.assertEqual(paired["sample_std_delta"], 0.0)
        self.assertEqual(paired["sample_variance_delta"], 0.0)

    def test_actual_queue_schema2_contract_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, queue, queue_path = evidence_fixture(Path(temp))
            self.assertEqual(queue["schema_version"], 2)
            self.assertIn("config_sha256", queue)
            self.assertIn("config_file_sha256", queue)
            self.assertIn("protocol_sha256", queue)
            self.assertNotIn("canonical_config_sha256", queue)
            reports, _, _ = summary.validate_evidence(
                summary._read_json(Path(queue["config_path"])), summary._read_json(queue_path), queue_path
            )
            self.assertEqual(len(reports), 12)

    def test_missing_job_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, queue, queue_path = evidence_fixture(Path(temp))
            queue["jobs"].pop()
            with self.assertRaisesRegex(summary.ValidationError, "completeness"):
                summary.validate_evidence(manifest, queue, queue_path)

    def test_duplicate_job_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, queue, queue_path = evidence_fixture(Path(temp))
            queue["jobs"][-1] = dict(queue["jobs"][0])
            with self.assertRaisesRegex(summary.ValidationError, "duplicate"):
                summary.validate_evidence(manifest, queue, queue_path)

    def test_protocol_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, queue, queue_path = evidence_fixture(Path(temp))
            report_path = Path(temp) / queue["jobs"][0]["evaluation_report_path"]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["protocol_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(summary.ValidationError, "protocol hash mismatch"):
                summary.validate_evidence(manifest, queue, queue_path)

    def test_manifest_unrelated_change_breaks_canonical_config_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, queue, queue_path = evidence_fixture(Path(temp))
            manifest["unrelated_but_hash_bound"] = True
            with self.assertRaisesRegex(summary.ValidationError, "manifest argument differs from queue config file"):
                summary.validate_evidence(manifest, queue, queue_path)

    def test_colluding_wrong_config_hash_is_rejected_against_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, queue, queue_path = evidence_fixture(root)
            wrong = "0" * 64
            queue["config_sha256"] = wrong
            for job in queue["jobs"]:
                report_path = root / job["evaluation_report_path"]
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["config_sha256"] = wrong
                save_report(report_path, report)
            with self.assertRaisesRegex(summary.ValidationError, "queue canonical config hash mismatch"):
                summary.validate_evidence(manifest, queue, queue_path)

    def test_legacy_canonical_config_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, queue, queue_path = evidence_fixture(Path(temp))
            queue["canonical_config_sha256"] = queue["config_sha256"]
            with self.assertRaisesRegex(summary.ValidationError, "legacy canonical_config_sha256"):
                summary.validate_evidence(manifest, queue, queue_path)

    def test_missing_or_wrong_raw_config_hash_is_rejected(self) -> None:
        mutations = (
            ("queue_missing", lambda queue, report: queue.pop("config_file_sha256"), "queue config_file_sha256 mismatch"),
            ("job_wrong", lambda queue, report: queue["jobs"][0].__setitem__("config_file_sha256", "0" * 64), "job config_file_sha256 mismatch"),
            ("report_missing", lambda queue, report: report.pop("config_file_sha256"), "config_file_sha256 mismatch"),
        )
        for label, mutate, pattern in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest, queue, queue_path = evidence_fixture(root)
                path, report = load_first_report(root, queue)
                mutate(queue, report)
                save_report(path, report)
                with self.assertRaisesRegex(summary.ValidationError, pattern):
                    summary.validate_evidence(manifest, queue, queue_path)

    def test_missing_or_wrong_protocol_hash_is_rejected_at_each_level(self) -> None:
        mutations = (
            ("queue_missing", lambda queue, report: queue.pop("protocol_sha256"), "queue protocol hash mismatch"),
            ("job_wrong", lambda queue, report: queue["jobs"][0].__setitem__("protocol_sha256", "0" * 64), "job protocol hash mismatch"),
            ("report_missing", lambda queue, report: report.pop("protocol_sha256"), "protocol hash mismatch"),
        )
        for label, mutate, pattern in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest, queue, queue_path = evidence_fixture(root)
                path, report = load_first_report(root, queue)
                mutate(queue, report)
                save_report(path, report)
                with self.assertRaisesRegex(summary.ValidationError, pattern):
                    summary.validate_evidence(manifest, queue, queue_path)

    def test_tampered_report_config_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, queue, queue_path = evidence_fixture(root)
            path, report = load_first_report(root, queue)
            report["config_sha256"] = "0" * 64
            save_report(path, report)
            with self.assertRaisesRegex(summary.ValidationError, "canonical config hash mismatch"):
                summary.validate_evidence(manifest, queue, queue_path)

    def test_overall_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, queue, queue_path = evidence_fixture(root)
            path, report = load_first_report(root, queue)
            report["metrics"]["overall"]["sample_count"] += 1
            save_report(path, report)
            with self.assertRaisesRegex(summary.ValidationError, "overall sample_count mismatch"):
                summary.validate_evidence(manifest, queue, queue_path)

    def test_per_condition_trial_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, queue, queue_path = evidence_fixture(root)
            path, report = load_first_report(root, queue)
            report["metrics"]["by_command"][0]["trials_started"] = 11
            save_report(path, report)
            with self.assertRaisesRegex(summary.ValidationError, "per-condition trial mismatch"):
                summary.validate_evidence(manifest, queue, queue_path)

    def test_negative_first_contact_counts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, queue, queue_path = evidence_fixture(root)
            path, report = load_first_report(root, queue)
            report["metrics"]["overall"]["first_contact_count"] = -26
            for condition in report["metrics"]["by_command"]:
                condition["first_contact_count"] = -1
            save_report(path, report)
            with self.assertRaisesRegex(summary.ValidationError, "invalid first_contact_count"):
                summary.validate_evidence(manifest, queue, queue_path)

    def test_nonfinite_and_negative_physical_metrics_are_rejected(self) -> None:
        mutations = (
            ("nan_rmse", "lin_vel_rmse_mps", float("nan"), "invalid lin_vel_rmse_mps"),
            ("infinite_power", "absolute_mechanical_power_w", float("inf"), "invalid absolute_mechanical_power_w"),
            ("negative_rmse", "yaw_rate_rmse_radps", -0.1, "negative physical metric yaw_rate_rmse_radps"),
            ("negative_l2", "torque_l2_mean", -0.1, "negative physical metric torque_l2_mean"),
            ("negative_action", "action_rate_l2_mean", -0.1, "negative physical metric action_rate_l2_mean"),
            ("negative_contact_air", "mean_air_time_at_first_contact_s", -0.1, "negative physical metric mean_air_time_at_first_contact_s"),
        )
        for label, metric, value, pattern in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest, queue, queue_path = evidence_fixture(root)
                path, report = load_first_report(root, queue)
                report["metrics"]["overall"][metric] = value
                save_report(path, report)
                with self.assertRaisesRegex(summary.ValidationError, pattern):
                    summary.validate_evidence(manifest, queue, queue_path)

    def test_negative_official_feet_air_time_raw_metric_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, queue, queue_path = evidence_fixture(root)
            path, report = load_first_report(root, queue)
            report["metrics"]["overall"]["feet_air_time_raw_mean"] = -0.5
            for condition in report["metrics"]["by_command"]:
                condition["feet_air_time_raw_mean"] = -0.5
            save_report(path, report)
            reports, _, _ = summary.validate_evidence(manifest, queue, queue_path)
            self.assertEqual(reports[("baseline", 42)]["metrics"]["overall"]["feet_air_time_raw_mean"], -0.5)

    def test_missing_metric_effective_weights_and_protocol_are_rejected(self) -> None:
        mutations = (
            ("metric", lambda report: report["metrics"]["overall"].pop("torque_l2_mean"), "metric torque_l2_mean missing"),
            ("weights", lambda report: report.pop("effective_weights"), "effective weights mismatch"),
            ("protocol", lambda report: report.pop("protocol_sha256"), "protocol hash mismatch"),
            ("denominators", lambda report: report.pop("denominators"), "explicit denominators missing"),
            ("checkpoint", lambda report: report.pop("checkpoint"), "checkpoint reference missing"),
            ("runtime", lambda report: report.pop("runtime_evidence"), "runtime_evidence missing"),
        )
        for label, mutate, pattern in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest, queue, queue_path = evidence_fixture(root)
                path, report = load_first_report(root, queue)
                mutate(report)
                save_report(path, report)
                with self.assertRaisesRegex(summary.ValidationError, pattern):
                    summary.validate_evidence(manifest, queue, queue_path)

    def test_tampered_variant_hash_is_rejected_against_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, queue, queue_path = evidence_fixture(root)
            wrong = "0" * 64
            queue["jobs"][0]["variant_config_sha256"] = wrong
            path, report = load_first_report(root, queue)
            report["variant_config_sha256"] = wrong
            save_report(path, report)
            with self.assertRaisesRegex(summary.ValidationError, "variant_config_sha256 mismatch"):
                summary.validate_evidence(manifest, queue, queue_path)

    def test_checkpoint_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, queue, queue_path = evidence_fixture(Path(temp))
            queue["jobs"][0]["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(summary.ValidationError, "checkpoint_sha256 mismatch"):
                summary.validate_evidence(manifest, queue, queue_path)

    def test_tampered_checkpoint_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, queue, queue_path = evidence_fixture(root)
            path, report = load_first_report(root, queue)
            Path(report["checkpoint"]["reference"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(summary.ValidationError, "checkpoint_sha256 mismatch"):
                summary.validate_evidence(manifest, queue, queue_path)


if __name__ == "__main__":
    unittest.main()
