#!/usr/bin/env python3
"""Compare two FastGraphCompute source releases on the paper headline case.

Each measurement runs in a fresh child process so differently built PyTorch
extensions never share an operator registry. The parent alternates AB/BA order,
records every timing, and evaluates a predeclared median-regression threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


POINTS = 5_000_000
DIMENSIONS = 8
K = 40
MAX_BIN_DIMS = 3
WARMUP_POINTS = 4096
WARMUP_K = 16
INPUT_PROTOCOL = "preloaded_torch_cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--baseline-label", default="v1.1.0")
    parser.add_argument("--candidate-label", default="v1.2.0")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--max-regression-pct", type=float, default=5.0)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--rep", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def child_main(args: argparse.Namespace) -> int:
    if args.source_root is None or args.label is None or args.rep is None:
        raise ValueError("child mode requires --source-root, --label, and --rep")

    source_root = args.source_root.resolve()
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(scripts_dir))

    import torch
    from npy_loader import load_rec_hit_features
    import fastgraphcompute
    from fastgraphcompute import binned_select_knn_pca

    imported_from = Path(fastgraphcompute.__file__).resolve()
    if source_root not in imported_from.parents:
        raise RuntimeError(
            f"loaded FastGraphCompute from {imported_from}, expected {source_root}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(42)
    warm = torch.randn(WARMUP_POINTS, DIMENSIONS, device="cuda")
    warm_splits = torch.tensor([0, WARMUP_POINTS], device="cuda", dtype=torch.int64)
    binned_select_knn_pca(
        WARMUP_K, warm, warm_splits, max_bin_dims=MAX_BIN_DIMS
    )
    torch.cuda.synchronize()
    del warm, warm_splits
    torch.cuda.empty_cache()

    data_np = load_rec_hit_features(POINTS, DIMENSIONS)
    data = torch.from_numpy(data_np).to(device="cuda", dtype=torch.float32)
    del data_np
    row_splits = torch.tensor([0, POINTS], device="cuda", dtype=torch.int64)
    torch.cuda.synchronize()

    start = time.perf_counter()
    indices, distances = binned_select_knn_pca(
        K, data, row_splits, max_bin_dims=MAX_BIN_DIMS
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    result = {
        "label": args.label,
        "rep": args.rep,
        "time_ms": elapsed_ms,
        "reported_version": fastgraphcompute.__version__,
        "source_root": str(source_root),
        "imported_from": str(imported_from),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "output_shape": list(indices.shape),
        "distance_shape": list(distances.shape),
    }
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


def run_child(root: Path, label: str, rep: int) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--source-root",
        str(root.resolve()),
        "--label",
        label,
        "--rep",
        str(rep),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"{label} repetition {rep} failed")
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line.removeprefix("RESULT_JSON="))
    raise RuntimeError(f"{label} repetition {rep} produced no result")


def median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def parent_main(args: argparse.Namespace) -> int:
    required = (args.baseline_root, args.candidate_root, args.output_csv, args.summary_json)
    if any(value is None for value in required):
        raise ValueError(
            "parent mode requires both release roots, --output-csv, and --summary-json"
        )
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists() or args.summary_json.exists():
        raise FileExistsError("refusing to overwrite an existing comparison result")

    releases = {
        "A": (args.baseline_root, args.baseline_label),
        "B": (args.candidate_root, args.candidate_label),
    }
    schedule: list[tuple[str, int]] = []
    for rep in range(1, args.repetitions + 1):
        order = ("A", "B") if rep % 2 else ("B", "A")
        schedule.extend((key, rep) for key in order)

    fieldnames = [
        "sequence", "label", "revision", "reported_version", "rep", "time_ms",
        "points", "dim", "k", "max_bin_dims", "input_protocol", "gpu",
        "torch_version", "cuda_version", "slurm_job_id", "source_root",
        "imported_from", "output_shape", "distance_shape",
    ]
    rows: list[dict] = []
    revisions = {
        label: git_revision(root) for root, label in releases.values()
    }
    with args.output_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sequence, (key, rep) in enumerate(schedule, start=1):
            root, label = releases[key]
            print(f"[{sequence}/{len(schedule)}] {label} rep={rep}", flush=True)
            result = run_child(root, label, rep)
            row = {
                **result,
                "sequence": sequence,
                "revision": revisions[label],
                "points": POINTS,
                "dim": DIMENSIONS,
                "k": K,
                "max_bin_dims": MAX_BIN_DIMS,
                "input_protocol": INPUT_PROTOCOL,
                "output_shape": json.dumps(result["output_shape"]),
                "distance_shape": json.dumps(result["distance_shape"]),
            }
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(f"  {result['time_ms']:.3f} ms", flush=True)

    timings = {
        label: [float(row["time_ms"]) for row in rows if row["label"] == label]
        for label in (args.baseline_label, args.candidate_label)
    }
    baseline_median = statistics.median(timings[args.baseline_label])
    candidate_median = statistics.median(timings[args.candidate_label])
    regression_pct = (candidate_median / baseline_median - 1.0) * 100.0
    paired_ratios = []
    for rep in range(1, args.repetitions + 1):
        baseline = next(
            float(row["time_ms"]) for row in rows
            if row["label"] == args.baseline_label and int(row["rep"]) == rep
        )
        candidate = next(
            float(row["time_ms"]) for row in rows
            if row["label"] == args.candidate_label and int(row["rep"]) == rep
        )
        paired_ratios.append(candidate / baseline)

    passed = regression_pct <= args.max_regression_pct
    summary = {
        "pass": passed,
        "criterion": (
            f"candidate median slowdown <= {args.max_regression_pct:.1f}%"
        ),
        "baseline": {
            "label": args.baseline_label,
            "revision": revisions[args.baseline_label],
            "median_ms": baseline_median,
            "mad_ms": median_absolute_deviation(timings[args.baseline_label]),
            "timings_ms": timings[args.baseline_label],
        },
        "candidate": {
            "label": args.candidate_label,
            "revision": revisions[args.candidate_label],
            "median_ms": candidate_median,
            "mad_ms": median_absolute_deviation(timings[args.candidate_label]),
            "timings_ms": timings[args.candidate_label],
        },
        "candidate_regression_pct": regression_pct,
        "median_paired_ratio": statistics.median(paired_ratios),
        "repetitions": args.repetitions,
        "points": POINTS,
        "dimensions": DIMENSIONS,
        "k": K,
        "max_bin_dims": MAX_BIN_DIMS,
        "input_protocol": INPUT_PROTOCOL,
        "output_csv": str(args.output_csv.resolve()),
    }
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def main() -> int:
    args = parse_args()
    return child_main(args) if args.child else parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
