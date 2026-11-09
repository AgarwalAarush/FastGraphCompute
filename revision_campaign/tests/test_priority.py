"""Scheduling/coordination tests use synthetic records and a fake clock, never CUDA."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from revision_campaign.common import atomic_json, case_dir, digest, read_json, result_path, task_id
from revision_campaign.coordination import peer_cases
from revision_campaign.planning import make_plan
from revision_campaign.priority import case_key, task_key, units
from revision_campaign.session import session


class PriorityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = read_json(Path(__file__).parents[1] / "config.example.json")
        self.cfg["run_root"] = str(self.root / "run")
        checkpoint = {"path": "/private/preferred.pt", "label": "preferred", "training_stage": "trained"}
        self.cfg["model"]["checkpoints"] = [checkpoint]
        manifest = self.root / "snapshots.json"
        atomic_json(manifest, {"snapshots": [{"coords": "/private/coords.npy", "row_splits": "/private/splits.npy", "dim": 8,
            "metadata": {"checkpoint_sha256": "a" * 64, "adapter_sha256": "b" * 64, "layer": "preferred", "batch_index": 0}}]})
        self.cfg["learned_manifest"] = str(manifest)
        self.plan = make_plan(self.cfg, "focused", "all")

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_tasks_dropped_duplicated_or_run_before_their_prerequisites(self):
        schedule = units(self.plan)
        seen = set()
        actual = []
        for unit in schedule:
            self.assertTrue(set(unit["requires"]).issubset(seen), unit)
            seen.update(unit["tasks"])
            actual.extend(unit["tasks"])
        self.assertEqual(Counter(actual), Counter(task_id(t) for t in self.plan["tasks"]))
        self.assertEqual(len({u["id"] for u in schedule}), len(schedule))

    def test_model_and_real_inputs_precede_bounded_controls_and_breadth(self):
        phases = list(dict.fromkeys(u["phase"] for u in units(self.plan)))
        self.assertEqual(phases[:4], ["checks", "model-core", "learned-core", "events-core"])
        self.assertLess(phases.index("controls-core"), phases.index("breadth"))
        self.assertLess(phases.index("diagnostics-core"), phases.index("breadth"))
        chosen = [u for u in units(self.plan) if u["phase"] == "learned-core" and u["stage"] == "benchmark"]
        self.assertEqual({u["backend"] for u in chosen}, set(self.cfg["backends"]))

    def test_raw_and_standardized_event_controls_remain_paired(self):
        mapping = {task_id(t): t for t in self.plan["tasks"]}
        cases = {mapping[key]["case"]["transform"] for u in units(self.plan) if u["phase"] == "events-core"
                 for key in u["tasks"]}
        self.assertEqual(cases, {"raw", "standardized"})
        controls = {mapping[key]["case"]["distribution"] for u in units(self.plan) if u["phase"] == "controls-core"
                    for key in u["tasks"]}
        self.assertEqual(controls, {"gaussian", "anisotropic_0.90", "anisotropic_0.99"})

    def execute_fake(self, strategy="front", fail_backend=None, time_out=False):
        clock = {"now": 0}
        schedule = {u["id"]: u for u in units(self.plan)}
        mapping = {task_id(t): t for t in self.plan["tasks"]}
        calls = []
        def execute(arguments, *_):
            unit = schedule[arguments[arguments.index("--unit-id") + 1]]
            calls.append((unit["phase"], unit["backend"], clock["now"]))
            clock["now"] += 60
            if unit["backend"] == fail_backend and unit["stage"] == "benchmark":
                return 75 if time_out else 2
            for key in unit["tasks"]:
                atomic_json(result_path(self.cfg, mapping[key]), {"status": "ok"})
            return 0
        with patch("revision_campaign.session.validate_execution"), \
             patch("revision_campaign.session.export_results"), \
             patch("revision_campaign.session.time.monotonic", side_effect=lambda: clock["now"]), \
             patch("revision_campaign.session.run_worker", side_effect=execute):
            code = session(self.plan, self.root / "plan.json", minutes=100, execute=True, strategy=strategy)
        return code, calls

    def test_complement_backfills_front_even_without_any_peer_return(self):
        code, calls = self.execute_fake("complement")
        first = next(c for c in calls if c[0] != "checks")
        self.assertEqual(first[0], "controls-core")
        first_model = next(c for c in calls if c[0] == "model-core")
        self.assertLessEqual(first_model[2], 100 * 60 * .5)  # <=40% middle, plus short gates.
        self.assertTrue(any(c[0] == "learned-core" for c in calls))

    def test_optional_backend_failure_does_not_discard_independent_essential_results(self):
        code, calls = self.execute_fake(fail_backend="cagra_nnd")
        self.assertEqual(code, 1)
        self.assertTrue(any(c[0] == "model-core" for c in calls))
        self.assertTrue(any(c[:2] == ("events-core", "fgc_pca") for c in calls))
        self.assertTrue(any(c[0] == "controls-core" for c in calls))
        self.assertTrue((self.root / "run/unit-failures.json").is_file())

    def test_slow_unit_is_deferred_and_other_comparisons_still_get_a_turn(self):
        code, calls = self.execute_fake(fail_backend="cagra_nnd", time_out=True)
        self.assertIn(code, (0, 75))
        self.assertTrue(any(c[:2] == ("events-core", "fgc_pca") for c in calls))
        summary = read_json(self.root / "run/execution-summary.json")
        self.assertTrue(summary["deferred_units"])
        self.assertFalse(summary["unit_failures"])

    def archive(self, filename, entry):
        from revision_campaign.transfer import FILES
        payload = {key: b"" for key in FILES}
        payload["manifest.json"] = json.dumps({"format": "fastgraph-aggregate-results", "version": 1,
            "source_plan_hash": "test", "comparisons": [entry], "hardware": {"test": {"gpu_name": filename}}}).encode()
        path = self.root / filename
        with zipfile.ZipFile(path, "w") as target:
            for key, data in payload.items():
                target.writestr(key, data)
            target.writestr("SHA256SUMS", "".join(f"{hashlib.sha256(data).hexdigest()}  {key}\n" for key, data in sorted(payload.items())))
        return path

    def peer_example(self):
        case = next(c for c in self.plan["cases"] if c["source"] == "synthetic")
        tasks = [t for t in self.plan["tasks"] if t.get("case") == case and t["stage"] in ("benchmark", "profile", "counters")]
        atomic_json(case_dir(self.cfg, case) / "metadata.json", {"input_hash": "c" * 64})
        entry = {"case_key": case_key(case), "input_hash": "c" * 64, "complete": True,
                 "completed_task_keys": [task_key(self.plan, t, "c" * 64) for t in tasks]}
        return case, entry

    def test_peer_requires_whole_comparison_from_one_hardware_and_exact_input_hash(self):
        case, entry = self.peer_example()
        half = len(entry["completed_task_keys"]) // 2
        a = self.archive("gpu-a.zip", {**entry, "completed_task_keys": entry["completed_task_keys"][:half]})
        b = self.archive("gpu-b.zip", {**entry, "completed_task_keys": entry["completed_task_keys"][half:]})
        self.assertFalse(peer_cases(self.plan, [a, b])[0])  # Never union per-backend rows across GPUs.
        full = self.archive("complete.zip", entry)
        self.assertEqual(peer_cases(self.plan, [full])[0], {task_id(case)})
        atomic_json(case_dir(self.cfg, case) / "metadata.json", {"input_hash": "different"})
        self.assertFalse(peer_cases(self.plan, [full])[0])

    def test_changed_protocol_prevents_peer_skip(self):
        _, entry = self.peer_example()
        full = self.archive("complete.zip", entry)
        changed = copy.deepcopy(self.plan)
        changed["config"]["ann"]["ggnn_tau"] = [0.999]
        self.assertFalse(peer_cases(changed, [full])[0])

    def test_snapshot_identity_uses_provenance_hashes_instead_of_machine_paths(self):
        case = next(c for c in self.plan["cases"] if c["source"] == "learned")
        moved = copy.deepcopy(case)
        moved["coords"] = "/another-machine/snapshot.npy"
        moved["row_splits"] = "/another-machine/rows.npy"
        self.assertEqual(case_key(case), case_key(moved))
        moved["snapshot_metadata"]["checkpoint_sha256"] = "different"
        self.assertNotEqual(case_key(case), case_key(moved))

    def test_worker_unit_does_only_its_selected_tasks_and_enforces_inputs(self):
        from revision_campaign.worker import run
        plan = make_plan(self.cfg, "pilot", "synthetic")
        schedule = units(plan)
        prepare = next(u for u in schedule if u["phase"] == "checks" and u["stage"] == "prepare")
        reference = next(u for u in schedule if u["phase"] == "checks" and u["stage"] == "reference")
        with patch("revision_campaign.worker.preflight", return_value=({}, {})), \
             patch("revision_campaign.worker.run_task", return_value={"metadata": {}}) as worker:
            with self.assertRaises(ValueError):
                run(plan, "reference", "core", unit_id=reference["id"])
            worker.assert_not_called()
            self.assertEqual(run(plan, "prepare", "core", unit_id=prepare["id"]), 0)
            self.assertEqual(worker.call_count, len(prepare["tasks"]))
        untouched = [t for t in plan["tasks"] if t["stage"] == "prepare" and task_id(t) not in prepare["tasks"]]
        self.assertTrue(untouched)
        self.assertTrue(all(not result_path(self.cfg, t).exists() for t in untouched))

    def test_configured_layer_priority_wins_over_capture_hook_order(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["capture_layers"] = ["late", "early"]
        catalog = read_json(cfg["learned_manifest"])
        first = catalog["snapshots"][0]
        first["metadata"]["layer"] = "early"
        late = copy.deepcopy(first)
        late["coords"] = "/private/late.npy"
        late["metadata"]["layer"] = "late"
        catalog["snapshots"].append(late)
        atomic_json(cfg["learned_manifest"], catalog)
        plan = make_plan(cfg, "focused", "all")
        chosen = next(u for u in units(plan) if u["phase"] == "learned-core")
        task = next(t for t in plan["tasks"] if task_id(t) == chosen["tasks"][0])
        self.assertEqual(task["case"]["snapshot_metadata"]["layer"], "late")

    def test_controller_reports_peer_coverage_without_inventing_local_rows(self):
        case, entry = self.peer_example()
        full = self.archive("complete.zip", entry)
        plan = copy.deepcopy(self.plan)
        plan["tasks"] = [t for t in plan["tasks"] if t.get("case") == case]
        plan.pop("plan_hash")
        plan["plan_hash"] = digest(plan)
        with patch("revision_campaign.session.validate_execution"), \
             patch("revision_campaign.session.run_worker") as worker:
            self.assertEqual(session(plan, self.root / "plan.json", execute=True, peer_results=[full]), 0)
            worker.assert_not_called()
        summary = read_json(self.root / "run/execution-summary.json")
        self.assertEqual(summary["status"], "covered-with-peer")
        self.assertEqual(summary["completed_bundles"], 0)
        self.assertEqual(summary["peer_covered_bundles"], len(plan["tasks"]))
        self.assertEqual((self.root / "run/report/measurements.csv").read_text(), "")


if __name__ == "__main__":
    unittest.main()
