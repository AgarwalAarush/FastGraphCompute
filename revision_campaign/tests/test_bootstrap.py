"""CPU checks for owned setup, immutable plans, interrupted installs and safe preview."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from revision_campaign.bootstrap import Setup, claim_work, pilot_config, save_pilot, start
from revision_campaign.common import ROOT, read_json
from revision_campaign.planning import blockers


class BootstrapTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
