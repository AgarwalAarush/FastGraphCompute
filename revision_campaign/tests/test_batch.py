"""Exercise copied Slurm scripts from a spool directory without requesting GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from revision_campaign.common import ROOT, atomic_json, read_json
from revision_campaign.planning import make_plan, write_plan


class BatchScriptTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fgc batch ")
        self.root = Path(self.temporary.name)
        self.spool = self.root / "spool"
        self.spool.mkdir()
        self.environment = dict(os.environ, FGC_CAMPAIGN_ROOT=str(ROOT),
                                SLURM_SUBMIT_DIR=str(self.spool), SLURM_JOB_ID="cpu-test")

    def tearDown(self):
        self.temporary.cleanup()

    def copied(self, name):
        target = self.spool / name
        shutil.copy2(ROOT / "revision_campaign" / name, target)
        return target

    def test_worker_wrapper_preserves_root_arguments_and_site_environment_when_spooled(self):
        site = self.root / "site environment.sh"
        site.write_text("export FGC_BATCH_TEST='site loaded'\n")
        self.environment["FGC_SITE_ENV"] = str(site)
        result = subprocess.run(["bash", str(self.copied("job.sh")), sys.executable, "-c",
            "import json,os,sys; print(json.dumps([os.getcwd(),sys.argv[1],os.environ['FGC_BATCH_TEST']]))",
            "argument with spaces"], cwd=self.spool, env=self.environment,
            capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [str(ROOT), "argument with spaces", "site loaded"])

    def test_spooled_local_wrapper_can_find_an_already_complete_run(self):
        run = self.root / "run"
        atomic_json(run / "execution-summary.json", {"status": "complete"})
        atomic_json(run / "session-plan.json", {"plan_hash": "test-plan"})
        plan = self.root / "plan.json"
        atomic_json(plan, {"plan_hash": "test-plan", "config": {"run_root": str(run)}})
        self.environment.pop("FGC_CAMPAIGN_ROOT")
        self.environment.update(SLURM_SUBMIT_DIR=str(ROOT), CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="100")
        result = subprocess.run(["bash", str(self.copied("local_slurm_job.sh")), str(plan),
            "unused-calibration.json", "1", sys.executable], cwd=self.spool,
            env=self.environment, capture_output=True, text=True, check=True)
        self.assertIn("already completed", result.stdout)

    def test_spooled_collection_exports_honest_empty_coverage(self):
        cfg = read_json(ROOT / "revision_campaign/config.example.json")
        cfg["run_root"] = str(self.root / "run")
        plan = self.root / "plan.json"
        write_plan(make_plan(cfg, "pilot", "synthetic"), plan)
        subprocess.run(["bash", str(self.copied("local_collect.sh")), str(plan), sys.executable],
            cwd=self.spool, env=self.environment, capture_output=True, text=True, check=True)
        self.assertTrue((Path(cfg["run_root"]) / "returns/local-final-cpu-test.zip").is_file())
        coverage = (Path(cfg["run_root"]) / "report/COVERAGE.md").read_text()
        self.assertIn("missing", coverage.lower())

    def test_submission_sets_explicit_root_and_bounded_dependencies(self):
        binary = self.root / "bin"
        binary.mkdir()
        calls = self.root / "calls.json"
        stub = binary / "sbatch"
        stub.write_text(f"#!{sys.executable}\n" +
            "import json,os,sys\nfrom pathlib import Path\n" +
            "p=Path(os.environ['FGC_TEST_CALLS'])\n" +
            "calls=json.loads(p.read_text()) if p.exists() else []\n" +
            "calls.append({'args':sys.argv[1:],'root':os.environ.get('FGC_CAMPAIGN_ROOT')})\n" +
            "p.write_text(json.dumps(calls))\nprint(1000+len(calls))\n")
        stub.chmod(0o755)
        plan = self.root / "plan.json"
        atomic_json(plan, {"config": {"run_root": str(self.root / "run")}})
        self.environment.update(PATH=str(binary) + os.pathsep + os.environ["PATH"],
                                FGC_TEST_CALLS=str(calls), FGC_LOCAL_LAST_SLOT_MINUTES="95")
        subprocess.run(["bash", str(ROOT / "revision_campaign/submit_complement.sh"), str(plan),
            "calibration.json", sys.executable, "999"], cwd=self.spool,
            env=self.environment, capture_output=True, text=True, check=True)
        submitted = read_json(calls)
        self.assertEqual(len(submitted), 6)
        for call in submitted:
            self.assertEqual(call["root"], str(ROOT))
            self.assertIn("--chdir=" + str(ROOT), call["args"])
            self.assertIn("--no-requeue", call["args"])
            self.assertIn("--kill-on-invalid-dep=yes", call["args"])
        self.assertIn("--dependency=afterok:999", submitted[0]["args"])
        self.assertIn("--dependency=afterok:1001:1002", submitted[2]["args"])
        self.assertIn("--time=01:35:00", submitted[4]["args"])


if __name__ == "__main__":
    unittest.main()
