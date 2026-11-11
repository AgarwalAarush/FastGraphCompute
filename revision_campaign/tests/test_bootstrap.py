"""CPU checks for owned setup, immutable plans, interrupted installs and safe preview."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from revision_campaign.bootstrap import (FGC_PIN, GGNN_PIN, Setup, SlotExpired, claim_work,
                                         main, pilot_config, preserve_incomplete, save_pilot,
                                         start, time_left_seconds)
from revision_campaign.common import ROOT, atomic_json, read_json
from revision_campaign.planning import blockers


class BootstrapTest(unittest.TestCase):
    @staticmethod
    def fake_sources(_setup, name, arguments):
        if name.startswith("clone-"):
            Path(arguments[-1]).mkdir(parents=True)
        if name == "prepare-counters":
            atomic_json(Path(arguments[-1]) / "fgc-diagnostic-build.json", {"base_commit": FGC_PIN})

    @staticmethod
    def fake_install(_setup, _name, _args, expected):
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.touch()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fgc startup ")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_preview_never_creates_work_or_invokes_dependencies(self):
        work = self.root / "new work"
        result = subprocess.run(["bash", str(ROOT / "revision_campaign/start.sh"),
            "--work", str(work), "--gpu", "H100", "--full-gpu"],
            capture_output=True, text=True, check=True)
        self.assertIn("Without --execute", result.stdout)
        self.assertFalse(work.exists())

    def test_existing_unowned_files_and_changed_identity_are_preserved(self):
        work = self.root / "existing"
        work.mkdir()
        data = work / "owner-data.txt"
        data.write_text("preserve me")
        with self.assertRaises(ValueError):
            claim_work(work, {"gpu": "H100"})
        self.assertEqual(list(work.iterdir()), [data])
        self.assertEqual(data.read_text(), "preserve me")
        owned = self.root / "owned"
        claim_work(owned, {"gpu": "H100"})
        claim_work(owned, {"gpu": "H100"})
        with self.assertRaises(ValueError):
            claim_work(owned, {"gpu": "A100"})
        self.assertEqual(read_json(owned / "bootstrap-owner.json"), {"gpu": "H100"})

    def test_successful_steps_are_cached_and_failed_steps_remain_retryable(self):
        setup = Setup(self.root, dict(os.environ))
        artifact = self.root / "artifact"
        command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('complete')", str(artifact)]
        setup.step("build", command, artifact)
        marker = setup.state / "build.json"
        modified = artifact.stat().st_mtime_ns
        setup.step("build", ["MUST-NOT-BE-EXECUTED"], artifact)
        self.assertEqual(artifact.stat().st_mtime_ns, modified)
        self.assertTrue(marker.exists())
        with self.assertRaises(RuntimeError):
            setup.step("interrupted", [sys.executable, "-c", "raise SystemExit(7)"], artifact)
        self.assertFalse((setup.state / "interrupted.json").exists())
        setup.step("interrupted", command, artifact)
        self.assertTrue((setup.state / "interrupted.json").exists())
        artifact.unlink()
        with self.assertRaises(ValueError):
            setup.step("build", command, artifact)

    def test_generated_pilot_needs_no_model_and_preserves_reviewed_plan(self):
        cfg = pilot_config(self.root, "H100", "")
        path = self.root / "configs/synthetic-pilot.plan.json"
        config_path, plan = save_pilot(cfg, path)
        self.assertEqual((len(plan["cases"]), len(plan["tasks"])), (7, 44))
        self.assertEqual(plan["blockers"], [])
        self.assertTrue(blockers(cfg, "all"))
        self.assertIsNone(cfg["model"]["adapter"])
        self.assertEqual(cfg["python_executables"]["core"], str(self.root / "envs/core/bin/python"))
        original = path.read_bytes()
        self.assertEqual(save_pilot(cfg, path)[1], plan)
        self.assertEqual(path.read_bytes(), original)
        modified = {**cfg, "expected_gpu_regex": "A100"}
        with self.assertRaises(ValueError):
            save_pilot(modified, path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(read_json(config_path), cfg)

    def test_full_gpu_confirmation_is_required_before_creating_work(self):
        work = self.root / "not-created"
        with self.assertRaises(ValueError):
            start(argparse.Namespace(full_gpu=False, work=str(work)))
        self.assertFalse(work.exists())

    def test_interrupted_conda_prefix_requires_explicit_resume(self):
        prefix = self.root / "env"
        (prefix / "conda-meta").mkdir(parents=True)
        (prefix / "conda-meta/history").write_text("owned test prefix\n")
        (prefix / "bin").mkdir()
        fake_python = prefix / "bin/python"
        fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
        fake_python.chmod(0o755)
        tools = self.root / "tools"
        tools.mkdir()
        conda = tools / "conda"
        conda.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FGC_TEST_CONDA_CALLS"\n')
        conda.chmod(0o755)
        calls = self.root / "conda-calls"
        env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
                   FGC_TEST_CONDA_CALLS=str(calls))
        command = ["bash", str(ROOT / "revision_campaign/setup.sh"), "core", str(prefix)]
        rejected = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(rejected.returncode, 2)
        self.assertFalse(calls.exists())
        subprocess.run(command + ["--resume"], env=env, capture_output=True, text=True, check=True)
        self.assertTrue(calls.read_text().startswith("install "))
        self.assertEqual((prefix / "conda-meta/history").read_text(), "owned test prefix\n")

    def test_slurm_remaining_time_formats_and_unlimited(self):
        for text, seconds in (("29:30", 1770), ("00:29:30", 1770),
                              ("2-01:02:03", 176523), ("00:00", 0),
                              ("UNLIMITED", None), ("INVALID", None)):
            with self.subTest(text=text):
                self.assertEqual(time_left_seconds(text), seconds)

    def test_shared_setup_deadline_stops_a_child_and_retains_finished_steps(self):
        setup = Setup(self.root, dict(os.environ), deadline=time.monotonic() + 1)
        first, second = self.root / "first", self.root / "second"
        command = "import sys,time; from pathlib import Path; time.sleep(float(sys.argv[1])); Path(sys.argv[2]).write_text('done')"
        setup.step("first", [sys.executable, "-c", command, "0.2", str(first)], first)
        started = time.monotonic()
        with self.assertRaises(SlotExpired):
            setup.step("second", [sys.executable, "-c", command, "20", str(second)], second)
        self.assertLess(time.monotonic() - started, 3)
        self.assertTrue(first.exists())
        self.assertFalse(second.exists())
        self.assertTrue((setup.state / "first.json").exists())
        self.assertFalse((setup.state / "second.json").exists())
        resumed = Setup(self.root, dict(os.environ))
        resumed.step("first", ["MUST-NOT-RUN"], first)
        resumed.step("second", [sys.executable, "-c", command, "0", str(second)], second)
        self.assertTrue(second.exists())

    def test_incomplete_prefix_backup_preserves_contents_and_rejects_external_paths(self):
        prefix = self.root / "envs/core"
        prefix.mkdir(parents=True)
        (prefix / "partial-package").write_text("preserve this")
        inode = prefix.stat().st_ino
        preserve_incomplete(prefix, self.root)
        backup = next((self.root / "interrupted-setup").iterdir())
        self.assertEqual(backup.stat().st_ino, inode)
        self.assertEqual((backup / "partial-package").read_text(), "preserve this")
        self.assertFalse(prefix.exists())
        with tempfile.TemporaryDirectory() as outside:
            link = self.root / "outside-link"
            link.symlink_to(outside)
            with self.assertRaises(ValueError):
                preserve_incomplete(link, self.root)
            self.assertTrue(link.is_symlink())

    def test_expired_allocation_returns_checkpoint_without_starting_setup(self):
        work = self.root / "work"
        arguments = ["bootstrap", "--work", str(work), "--full-gpu", "--execute"]
        with patch.object(sys, "argv", arguments), \
             patch("revision_campaign.bootstrap.allocation_seconds", return_value=0), \
             patch("revision_campaign.bootstrap.subprocess.Popen") as execute:
            self.assertEqual(main(), 75)
            execute.assert_not_called()
        record = read_json(work / "startup-status.json")
        self.assertEqual(record["status"], "checkpoint")
        self.assertFalse(record["scientific_completion_claimed"])

    def test_twenty_minutes_of_setup_leave_only_five_minutes_for_pilot(self):
        work = self.root / "budget-work"
        clock = {"now": 0}
        def step(setup, name, _args, expected):
            if name == "environment-core":
                clock["now"] += 1200
            self.fake_install(setup, name, _args, expected)
        args = argparse.Namespace(full_gpu=True, minutes=25, slot_minutes=30,
                                  work=str(work), gpu="H100", site_env=None)
        process = Mock()
        process.wait.return_value = 75
        with patch("revision_campaign.bootstrap.time.monotonic", side_effect=lambda: clock["now"]), \
             patch("revision_campaign.bootstrap.allocation_seconds", return_value=None), \
             patch("revision_campaign.bootstrap.git_identity", side_effect=lambda p:
                   {"commit": GGNN_PIN if p.name == "ggnn" else FGC_PIN, "dirty": False}), \
             patch.object(Setup, "step", step), patch.object(Setup, "run", self.fake_sources), \
             patch("revision_campaign.bootstrap.subprocess.Popen", return_value=process) as execute:
            self.assertEqual(start(args), 75)
        command = execute.call_args.args[0]
        self.assertEqual(float(command[command.index("--minutes") + 1]), 5)
        self.assertLessEqual(process.wait.call_args.kwargs["timeout"], 600)

    def test_cpu_setup_does_not_confirm_a_gpu_or_start_gpu_workers(self):
        work = self.root / "cpu-setup"
        args = argparse.Namespace(full_gpu=False, setup_only=True, minutes=25, slot_minutes=30,
                                  work=str(work), gpu="H100", site_env=None)
        with patch("revision_campaign.bootstrap.allocation_seconds", return_value=None), \
             patch("revision_campaign.bootstrap.git_identity", side_effect=lambda p:
                   {"commit": GGNN_PIN if p.name == "ggnn" else FGC_PIN, "dirty": False}), \
             patch.object(Setup, "step", self.fake_install), patch.object(Setup, "run", self.fake_sources), \
             patch("revision_campaign.bootstrap.subprocess.Popen") as execute:
            self.assertEqual(start(args), 0)
            execute.assert_not_called()
        self.assertFalse(read_json(work / "configs/setup.config.json")["full_device_confirmed"])
        self.assertFalse((work / "configs/synthetic-pilot.plan.json").exists())
        self.assertEqual(read_json(work / "startup-status.json")["status"], "setup-ready")


if __name__ == "__main__":
    unittest.main()
