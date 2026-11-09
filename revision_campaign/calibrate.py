"""Quick full-A100 drift gate using the unchanged historical headline checker."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from .common import ROOT, atomic_json, file_digest, git_identity, read_json, stamp


def calibrate(source, historical, output, repetitions=3, tolerance_percent=10):
    if Path(output).exists():
        raise FileExistsError("Refusing to replace a calibration record")
    if os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") != "100" or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run calibration in a Slurm mps:100 allocation")
    previous = read_json(historical)
    if tuple(previous.get(k) for k in ("points", "dimensions", "k", "max_bin_dims")) != (5000000, 8, 40, 3):
        raise ValueError("Historical summary must describe the checker's 5M/d8/K40/three-bin-dimension case")
    identity = git_identity(source)
    if identity != {"commit": previous["candidate"]["revision"], "dirty": False}:
        raise ValueError("Calibration needs the clean historical candidate source commit")
    if repetitions < 3 or tolerance_percent <= 0:
        raise ValueError("Use at least three repetitions and a positive drift threshold")
    script = ROOT / "reproduction/verification/compare_fgc_release_speed.py"
    rows = []
    for rep in range(1, repetitions + 1):
        result = subprocess.run([sys.executable, str(script), "--child", "--source-root", str(source),
                                 "--label", "current-A100", "--rep", str(rep)],
                                capture_output=True, text=True, check=True)
        row = next(json.loads(line.removeprefix("RESULT_JSON=")) for line in result.stdout.splitlines()
                   if line.startswith("RESULT_JSON="))
        if "A100" not in row["gpu"]:
            raise ValueError("Historical drift comparison requires an A100")
        rows.append(row)
        print(f"Calibration {rep}/{repetitions}: {row['time_ms']:.3f} ms on {row['gpu']}", flush=True)
    median = statistics.median(row["time_ms"] for row in rows)
    reference = previous["candidate"]["median_ms"]
    drift = 100 * (median / reference - 1)
    summary = {"pass": abs(drift) <= tolerance_percent, "created": stamp(), "median_ms": median,
               "historical_median_ms": reference, "drift_percent": drift,
               "criterion": f"absolute median drift <= {tolerance_percent}%", "repetitions": repetitions,
               "points": previous["points"], "dimensions": previous["dimensions"], "k": previous["k"],
               "max_bin_dims": previous["max_bin_dims"], "source": identity,
               "historical_sha256": file_digest(historical), "checker_sha256": file_digest(script),
               "binaries": {p.name: file_digest(p) for p in (Path(source) / "fastgraphcompute/extensions").glob("*.so")},
               "mps_percent": 100, "job_id": os.environ["SLURM_JOB_ID"], "measurements": rows}
    summary["node"] = os.environ.get("SLURMD_NODENAME")
    summary["historical_file"] = str(Path(historical).resolve())
    atomic_json(output, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tolerance-percent", type=float, default=10)
    args = parser.parse_args()
    raise SystemExit(calibrate(args.source, args.historical, args.out, args.repetitions, args.tolerance_percent))
