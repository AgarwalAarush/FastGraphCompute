"""CPU-only tests: no extension imports, CUDA calls or cluster submissions."""
import copy
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from revision_campaign.common import atomic_json, digest, read_json, task_id, verify_artifacts, result_path
from revision_campaign.data import decode_splits, prepare, synthetic
from revision_campaign.launch import jobs, launch
from revision_campaign.planning import blockers, make_plan, verify_plan, write_plan
from revision_campaign.report import report
from revision_campaign.transfer import export_results, import_results, plot_results, read_bundle


class CampaignTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = read_json(Path(__file__).parents[1] / "config.example.json")
        self.cfg["run_root"] = str(Path(self.tmp.name) / "run")

    def tearDown(self):
        self.tmp.cleanup()

    def test_local_resets(self):
        rs, blocks = decode_splits(np.array([[0], [2], [5], [0], [3], [7]]), 12,
                                  "concatenated_local_offsets")
        np.testing.assert_array_equal(rs, [0, 2, 5, 8, 12])
        self.assertEqual(blocks, 2)
        with self.assertRaises(ValueError):
            decode_splits([0, 2, 5, 0, 3, 7], 12, "global")
        with self.assertRaises(ValueError):
            decode_splits([0, 5], 12, "global")

    def test_invalid_splits(self):
        for values in ([0, 0, 2], [-1, 2], [0, 3, 1]):
            with self.assertRaises(ValueError):
                decode_splits(values, 2, "concatenated_local_offsets")

    def test_plan_integrity_and_blockers(self):
        plan = make_plan(self.cfg, "pilot", "synthetic")
        self.assertFalse(plan["blockers"])
        verify_plan(plan)
        changed = copy.deepcopy(plan)
        changed["config"]["full_device_confirmed"] = True
        with self.assertRaises(ValueError):
            verify_plan(changed)
        self.assertTrue(blockers(self.cfg, "all"))
        self.assertEqual(len({task_id(t) for t in plan["tasks"]}), len(plan["tasks"]))

    def test_preview_never_submits(self):
        plan = make_plan(self.cfg, "pilot", "synthetic")
        with patch("revision_campaign.launch.subprocess.check_output") as submit:
            launch(plan, "/tmp/review.json", "slurm", ["prepare"], execute=False)
            submit.assert_not_called()

    def test_submission_requires_inputs_and_gpu_attestation(self):
        plan = make_plan(self.cfg, "pilot", "all")
        with patch("revision_campaign.launch.subprocess.check_output") as submit:
            with self.assertRaises(ValueError):
                launch(plan, "/tmp/review.json", "slurm", ["prepare"], execute=True)
            submit.assert_not_called()

    def test_site_paths_are_arguments(self):
        self.cfg["python_executables"]["core"] = "/path with spaces/bin/python"
        plan = make_plan(self.cfg, "pilot", "synthetic")
        generated = list(jobs(plan, "/plan with spaces.json", ["prepare"], "slurm"))
        self.assertIn("/path with spaces/bin/python", generated[0])
        self.assertIn("/plan with spaces.json", generated[0])
        self.assertIn("--require-complete", generated[-1])

    def test_input_cache_integrity(self):
        case = {"source": "synthetic", "dim": 4, "points": 100, "seed": 7,
                "distribution": "gaussian", "transform": "raw"}
        a = prepare(self.cfg, case)
        b = prepare(self.cfg, case)
        self.assertEqual(a, b)
        path = Path(self.cfg["run_root"]) / "inputs" / task_id(case) / "coords.npy"
        with open(path, "ab") as handle:
            handle.write(b"corrupt")
        with self.assertRaises(ValueError):
            prepare(self.cfg, case)

    def test_synthetic_repeatability(self):
        case = {"dim": 8, "points": 100, "seed": 42, "distribution": "anisotropic_0.90"}
        a, _, _ = synthetic(case)
        b, _, _ = synthetic(case)
        np.testing.assert_array_equal(a, b)

    def test_plan_writes_and_coverage_missing(self):
        plan = make_plan(self.cfg, "pilot", "synthetic")
        path = Path(self.tmp.name) / "plan.json"
        write_plan(plan, path)
        with self.assertRaises(FileExistsError):
            write_plan(plan, path)
        report(plan)
        missing = read_json(Path(self.cfg["run_root"]) / "report/missing.json")
        self.assertEqual(len(missing), len(plan["tasks"]))

    def test_full_scope_requires_learned_manifest(self):
        self.assertTrue(any("learned_manifest" in b for b in make_plan(self.cfg, "pilot", "all")["blockers"]))

    def test_changed_model_file_invalidates_plan(self):
        artifact = Path(self.tmp.name) / "model-owner.py"
        artifact.write_text("# real owner module\n")
        self.cfg["model"]["adapter"] = str(artifact)
        plan = make_plan(self.cfg, "pilot", "synthetic")
        verify_artifacts(plan)
        artifact.write_text("# changed owner module contents\n")
        with self.assertRaises(ValueError):
            verify_artifacts(plan)

    def test_atomic_resume_skips_completed(self):
        from revision_campaign.worker import run
        plan = make_plan(self.cfg, "pilot", "synthetic")
        plan["tasks"] = [plan["tasks"][0]]
        plan.pop("plan_hash")
        plan["plan_hash"] = digest(plan)
        with patch("revision_campaign.worker.preflight", return_value=({}, {})), \
             patch("revision_campaign.worker.run_task", return_value={"metadata": {}}) as execute:
            self.assertEqual(run(plan, "prepare", "core"), 0)
            self.assertEqual(run(plan, "prepare", "core"), 0)
            self.assertEqual(execute.call_count, 1)

    def test_budget_checkpoints_without_claiming_completion(self):
        from revision_campaign.worker import run
        plan = make_plan(self.cfg, "pilot", "synthetic")
        with patch("revision_campaign.worker.preflight", return_value=({}, {})), \
             patch("revision_campaign.worker.run_task") as execute:
            self.assertEqual(run(plan, "prepare", "core", budget_seconds=0, require_complete=True), 75)
            execute.assert_not_called()

    def test_failure_is_recorded_and_not_silently_retried(self):
        from revision_campaign.worker import run
        plan = make_plan(self.cfg, "pilot", "synthetic")
        with patch("revision_campaign.worker.preflight", return_value=({}, {})), \
             patch("revision_campaign.worker.run_task", side_effect=RuntimeError("test failure")):
            with self.assertRaises(RuntimeError):
                run(plan, "prepare", "core")
            self.assertEqual(read_json(result_path(self.cfg, plan["tasks"][0]))["status"], "failed")
            with self.assertRaises(ValueError):
                run(plan, "prepare", "core")


class TransferTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = read_json(Path(__file__).parents[1] / "config.example.json")
        self.cfg["run_root"] = str(self.root / "operator-private-run")

    def tearDown(self):
        self.tmp.cleanup()

    def example_return(self):
        """Small mock results, not a simulated GPU measurement or model claim."""
        plan = make_plan(self.cfg, "pilot", "synthetic")
        benchmark = next(t for t in plan["tasks"] if t["stage"] == "benchmark")
        profile = {**benchmark, "stage": "profile"}
        counters = {**benchmark, "stage": "counters"}
        model = {"stage": "model", "backend": "core", "model_backend": "fgc_pca",
                 "checkpoint": {"path": "/private/secret/checkpoint.pt", "label": "test", "training_stage": "test"}}
        plan["tasks"] += [profile, model, counters]
        plan.pop("plan_hash")
        plan["plan_hash"] = digest(plan)
        common = {"status": "ok", "hardware": {"hardware_hash": "a" * 64, "gpu_name": "MOCK-H100"},
                  "software": {"module_version": "test", "module_file": "/private/secret/module.so",
                               "pip_freeze": ["PRIVATE_OWNER_NOTE"], "packages": {"torch": "test"},
                               "library_source": {"commit": self.cfg["library_commit"], "dirty": False}},
                  "input_metadata": {"shape": [10000, 4], "segment_lengths": [10000], "input_hash": "b" * 64,
                                     "sampled_pca_cumulative_variance": [0.5, 0.7, 0.9, 1.0],
                                     "provenance": {"schema_sha256": "c" * 64, "event_ids": [918273645],
                                                    "features_path": "/private/secret/hits.npy"}}}
        atomic_json(result_path(self.cfg, benchmark), {**common, "task": benchmark, "measurements": [
            {"rep": 0, "quality": {}, "build_ms": 1., "query_ms": 2., "total_ms": 3.,
             "recall": {"distance_recall_rtol_1e_4_atol_1e_10": 1., "pass": True,
                        "reference_coverage": "sampled_queries_all_candidates"}}]})
        atomic_json(result_path(self.cfg, profile), {**common, "task": profile, "diagnostics": {
            "memory": {"peak_delta_bytes": 1024**3}, "timing_eligible": False,
            "trace": "/private/secret/trace.json"}})
        atomic_json(result_path(self.cfg, counters), {"status": "ok", "task": counters,
            "hardware": common["hardware"], "software": common["software"],
            "diagnostics": {"queries": 10000, "candidates": 100000, "timing_eligible": False}})
        atomic_json(result_path(self.cfg, model), {**common, "task": model, "measurements": [
            {"rep": 0, "backend": "fgc_pca", "mode": "inference", "step_ms": [3., 4., 5.],
             "checkpoint": model["checkpoint"], "checkpoint_sha256": "d" * 64,
             "adapter_sha256": "e" * 64, "diagnostic_trace": "/private/secret/model-trace.json"}]})
        archive = self.root / "results.zip"
        export_results(plan, archive)
        return plan, archive

    def test_return_roundtrip_plots_after_operator_files_removed(self):
        plan, archive = self.example_return()
        payload = read_bundle(archive)
        combined = b"\n".join(payload.values()).decode()
        for private in ("/private/secret", "PRIVATE_OWNER_NOTE", "918273645", self.cfg["run_root"]):
            self.assertNotIn(private, combined)
        manifest = json.loads(payload["manifest.json"])
        self.assertEqual(manifest["source_plan_hash"], plan["plan_hash"])
        self.assertEqual(manifest["coverage"]["benchmark/fgc_pca"]["ok"], 1)
        self.assertGreater(manifest["coverage"]["prepare/core"]["missing"], 0)
        inputs = json.loads(payload["inputs.json"])
        self.assertEqual(next(iter(inputs.values()))["provenance"]["schema_sha256"], "c" * 64)
        shutil.rmtree(self.cfg["run_root"])
        received = import_results(archive, "h100-test", root=self.root / "another-machine")
        figures = plot_results(received)
        with open(received / "measurements.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(float(rows[0]["total_ms"]), 3.)
        self.assertEqual(rows[0]["reference_coverage"], "sampled_queries_all_candidates")
        self.assertEqual(len(list(figures.glob("recall-time-*.pdf"))), 1)
        self.assertGreater((figures / "memory.pdf").stat().st_size, 1000)
        self.assertGreater((figures / "model-steps.pdf").stat().st_size, 1000)

    def test_return_rejects_corruption_and_unexpected_members(self):
        _, archive = self.example_return()
        with zipfile.ZipFile(archive) as original:
            contents = {name: original.read(name) for name in original.namelist()}
        for name, change in (("corrupt.zip", {"measurements.csv": b"changed"}),
                             ("unsafe.zip", {"../outside": b"unexpected"})):
            altered = self.root / name
            with zipfile.ZipFile(altered, "w") as target:
                for member, data in {**contents, **change}.items():
                    target.writestr(member, data)
            with self.assertRaises(ValueError):
                import_results(altered, "rejected", root=self.root / "receive")
            self.assertFalse((self.root / "receive/rejected").exists())

    def test_return_refuses_replacement_and_invalid_run_names(self):
        plan, archive = self.example_return()
        with self.assertRaises(FileExistsError):
            export_results(plan, archive)
        received = import_results(archive, "same-name", root=self.root / "receive")
        before = (received / "measurements.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            import_results(archive, "same-name", root=self.root / "receive")
        with self.assertRaises(ValueError):
            import_results(archive, "../escape", root=self.root / "receive")
        self.assertEqual((received / "measurements.csv").read_bytes(), before)

    def test_plot_rejects_modified_aggregates_and_missing_checksums(self):
        _, archive = self.example_return()
        received = import_results(archive, "changed", root=self.root / "receive")
        with open(received / "measurements.csv", "ab") as handle:
            handle.write(b"tampered")
        with self.assertRaises(ValueError):
            plot_results(received)
        received = import_results(archive, "incomplete", root=self.root / "receive")
        lines = (received / "SHA256SUMS").read_text().splitlines()
        (received / "SHA256SUMS").write_text("\n".join(lines[1:]) + "\n")
        with self.assertRaises(ValueError):
            plot_results(received)

    def test_empty_partial_export_cannot_reuse_stale_csv(self):
        plan = make_plan(self.cfg, "pilot", "synthetic")
        directory = Path(self.cfg["run_root"]) / "report"
        directory.mkdir(parents=True)
        (directory / "measurements.csv").write_text("stale rows from a prior report")
        archive = self.root / "partial.zip"
        export_results(plan, archive)
        payload = read_bundle(archive)
        self.assertEqual(payload["measurements.csv"], b"")
        received = import_results(archive, "partial", root=self.root / "receive")
        self.assertEqual(list(plot_results(received).iterdir()), [])


class ReferenceTest(unittest.TestCase):
    def test_tiled_reference_and_tied_neighbors(self):
        import torch
        from revision_campaign.reference import compute_reference, evaluate
        coords = torch.tensor([[0., 0.], [0., 0.], [2., 0.], [100., 1.], [100., 1.], [104., 1.]])
        rs = torch.tensor([0, 3, 6], dtype=torch.int64)
        q, distances, idx = compute_reference(coords, rs, 2, 3, 10)
        np.testing.assert_array_equal(q, np.arange(6))
        np.testing.assert_allclose(distances, [[0, 0], [0, 0], [0, 4], [0, 0], [0, 0], [0, 16]])
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory) / "queries.npy", q)
            np.save(Path(directory) / "distances.npy", distances)
            values = evaluate(coords, rs, torch.from_numpy(idx), torch.from_numpy(distances).float(), directory, 2, exact=True)
            self.assertTrue(values["pass"])
            invalid = torch.from_numpy(idx.copy())
            invalid[0, 0] = 5
            self.assertFalse(evaluate(coords, rs, invalid, None, directory, 2, exact=True)["pass"])
            duplicate = torch.from_numpy(idx.copy())
            duplicate[:, 1] = duplicate[:, 0]
            self.assertFalse(evaluate(coords, rs, duplicate, None, directory, 2, exact=True)["pass"])


if __name__ == "__main__":
    unittest.main()
