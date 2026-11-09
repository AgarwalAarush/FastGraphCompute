"""Historical timing gate logic with mock subprocess outputs; no GPU used."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from revision_campaign.calibrate import calibrate
from revision_campaign.common import atomic_json, read_json


class CalibrationTest(unittest.TestCase):
    def test_absolute_drift_gate_handles_pass_slowdown_and_unexpected_speedup(self):
        for elapsed, expected in ((1000, True), (1200, False), (500, False)):
            with self.subTest(elapsed=elapsed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                previous = root / "previous.json"
                atomic_json(previous, {"candidate": {"revision": "pinned", "median_ms": 1000},
                    "points": 5000000, "dimensions": 8, "k": 40, "max_bin_dims": 3})
                result = Mock(stdout='RESULT_JSON=' + json.dumps({"gpu": "NVIDIA A100", "time_ms": elapsed}))
                with patch.dict(os.environ, {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100", "SLURM_JOB_ID": "test"}), \
                     patch("revision_campaign.calibrate.git_identity", return_value={"commit": "pinned", "dirty": False}), \
                     patch("revision_campaign.calibrate.subprocess.run", return_value=result):
                    code = calibrate(root, previous, root / "new.json")
                self.assertEqual(code, 0 if expected else 2)
                self.assertEqual(read_json(root / "new.json")["pass"], expected)

    def test_partial_mps_is_rejected_before_measurements(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50", "SLURM_JOB_ID": "test"}), \
             patch("revision_campaign.calibrate.subprocess.run") as execute:
            with self.assertRaises(ValueError):
                calibrate(directory, "/not/read", Path(directory) / "result.json")
            execute.assert_not_called()

    def test_historical_case_mismatch_cannot_pass_a_drift_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "wrong.json", {"points": 1000, "dimensions": 8, "k": 40, "max_bin_dims": 3})
            with patch.dict(os.environ, {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100", "SLURM_JOB_ID": "test"}), \
                 patch("revision_campaign.calibrate.subprocess.run") as execute:
                with self.assertRaises(ValueError):
                    calibrate(root, root / "wrong.json", root / "new.json")
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
