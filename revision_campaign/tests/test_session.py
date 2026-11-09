"""Allocation controller tests with mock workers/clock; never launches CUDA."""
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from revision_campaign.common import atomic_json, digest, read_json, result_path
from revision_campaign.planning import make_plan
from revision_campaign.session import run_worker, session
from revision_campaign.transfer import read_bundle


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        cfg = read_json(Path(__file__).parents[1] / "config.example.json")
        cfg["run_root"] = str(self.root / "run")
        self.plan = make_plan(cfg, "pilot", "synthetic")
        self.plan["tasks"] = self.plan["tasks"][:2]  # One prepare and its reference.
        self.plan.pop("plan_hash")
        self.plan["plan_hash"] = digest(self.plan)
        self.cfg = cfg

    def tearDown(self):
        self.temporary.cleanup()

    def save_stage(self, arguments, status="ok"):
        stage = arguments[arguments.index("--stage") + 1]
        for task in self.plan["tasks"]:
            if task["stage"] == stage:
                atomic_json(result_path(self.cfg, task), {"status": status, "metadata": {}})

    def test_preview_never_runs_workers_or_exports(self):
        with patch("revision_campaign.session.run_worker") as worker, \
             patch("revision_campaign.session.export_results") as export:
            self.assertEqual(session(self.plan, self.root / "plan.json"), 0)
            worker.assert_not_called()
            export.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_execution_blocks_missing_inputs_before_starting(self):
        blocked = make_plan(self.cfg, "pilot", "all")
        with patch("revision_campaign.session.run_worker") as worker:
            with self.assertRaises(ValueError):
                session(blocked, self.root / "plan.json", execute=True)
            worker.assert_not_called()

    def test_complete_then_resume_skips_workers_and_keeps_previous_return(self):
        def execute(arguments, *_):
            self.save_stage(arguments)
            return 0
        with patch("revision_campaign.session.validate_execution"), \
             patch("revision_campaign.session.run_worker", side_effect=execute) as worker:
            self.assertEqual(session(self.plan, self.root / "plan.json", execute=True), 0)
            self.assertEqual(worker.call_count, 2)
            self.assertEqual(session(self.plan, self.root / "plan.json", execute=True), 0)
            self.assertEqual(worker.call_count, 2)
        archives = list((self.root / "run/returns").glob("*.zip"))
        self.assertEqual(len(archives), 2)
        for archive in archives:
            coverage = json.loads(read_bundle(archive)["manifest.json"])["coverage"]
            self.assertEqual(coverage["reference/core"]["ok"], 1)

    def test_total_budget_decreases_across_stages_and_partial_work_resumes(self):
        clock = {"now": 0}
        budgets = []
        def execute(arguments, *_):
            budgets.append(int(arguments[arguments.index("--budget-seconds") + 1]))
            if len(budgets) == 1:
                clock["now"] += 900
                self.save_stage(arguments)
                return 0
            clock["now"] += 601
            return 75
        with patch("revision_campaign.session.validate_execution"), \
             patch("revision_campaign.session.time.monotonic", side_effect=lambda: clock["now"]), \
             patch("revision_campaign.session.run_worker", side_effect=execute):
            self.assertEqual(session(self.plan, self.root / "plan.json", minutes=25, execute=True), 75)
        self.assertGreater(budgets[0], 0)
        self.assertLessEqual(budgets[0], 1500)  # A priority phase may impose a smaller soft cap.
        self.assertEqual(budgets[1], 600)  # Time spent earlier is still charged to the ONE deadline.
        records = list((self.root / "run/sessions").glob("*.json"))
        self.assertEqual(read_json(records[0])["status"], "partial")
        self.assertEqual(read_json(records[0])["completed_bundles"], 1)
        def resume(arguments, *_):
            self.assertEqual(arguments[arguments.index("--stage") + 1], "reference")
            self.save_stage(arguments)
            return 0
        with patch("revision_campaign.session.validate_execution"), \
             patch("revision_campaign.session.run_worker", side_effect=resume) as worker:
            self.assertEqual(session(self.plan, self.root / "plan.json", execute=True), 0)
            worker.assert_called_once()

    def test_failure_stops_later_stages_and_exports_failed_coverage(self):
        def fail(arguments, *_):
            self.save_stage(arguments, "failed")
            return 2
        with patch("revision_campaign.session.validate_execution"), \
             patch("revision_campaign.session.run_worker", side_effect=fail) as worker:
            self.assertEqual(session(self.plan, self.root / "plan.json", execute=True), 1)
            worker.assert_called_once()
        archive = next((self.root / "run/returns").glob("*.zip"))
        coverage = json.loads(read_bundle(archive)["manifest.json"])["coverage"]
        self.assertEqual(coverage["prepare/core"]["failed"], 1)
        self.assertEqual(coverage["reference/core"]["missing"], 1)

    def test_worker_exception_still_exports_and_restores_signal_handlers(self):
        previous = {name: signal.getsignal(name) for name in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT)}
        with patch("revision_campaign.session.validate_execution"), \
             patch("revision_campaign.session.run_worker", side_effect=RuntimeError("test exception")):
            with self.assertRaises(RuntimeError):
                session(self.plan, self.root / "plan.json", execute=True)
        self.assertEqual(len(list((self.root / "run/returns").glob("*.zip"))), 1)
        for name, handler in previous.items():
            self.assertEqual(signal.getsignal(name), handler)

    def test_single_overrunning_bundle_is_stopped_at_session_deadline(self):
        alive = {"value": True}
        process = Mock(pid=12345)
        process.poll.side_effect = lambda: None if alive["value"] else -9
        def wait(timeout=None):
            if alive["value"]:
                raise subprocess.TimeoutExpired("mock worker", timeout)
            return -9
        process.wait.side_effect = wait
        with patch("revision_campaign.session.subprocess.Popen", return_value=process), \
             patch("revision_campaign.session.time.monotonic", side_effect=[0, 30]), \
             patch("revision_campaign.session.os.killpg", side_effect=lambda *_: alive.update(value=False)) as kill:
            code = run_worker(["mock worker"], {}, self.root / "worker.log", 20, threading.Event())
        self.assertEqual(code, 75)
        process.send_signal.assert_called_once_with(signal.SIGUSR1)
        kill.assert_called_once_with(12345, signal.SIGKILL)

    def test_invalid_duration_and_changed_plan_are_rejected(self):
        for minutes in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                session(self.plan, self.root / "plan.json", minutes)
        run_root = self.root / "run"
        run_root.mkdir()
        atomic_json(run_root / "session-plan.json", {"plan_hash": "different", "config_hash": "different"})
        with patch("revision_campaign.session.validate_execution"), \
             patch("revision_campaign.session.run_worker") as worker:
            with self.assertRaises(ValueError):
                session(self.plan, self.root / "plan.json", execute=True)
            worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
