"""Run task bundles with immutable identities and atomic per-task commits."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import signal
import time
import traceback

from .common import (atomic_json, attach_source, case_dir, digest, read_json,
                     result_path, run_lock, stamp, task_id)


def preflight(cfg, backend):
    import os
    import torch
    from .backends import Backend, metadata
    from .hardware import inspect_gpu, persist_identity
    library = attach_source(cfg)
    for flag in ("FGC_DISABLE_PCA", "FGC_FORCE_GLOBAL"):
        if os.environ.get(flag, "0") not in ("", "0"):
            raise ValueError(f"Unset {flag}: the campaign measures the released default dispatch")
    hardware = inspect_gpu(cfg)
    software = metadata(backend)
    software["library_source"] = library
    if backend.startswith("fgc"):
        import fastgraphcompute
        if Path(cfg["library_source"]).resolve() not in Path(fastgraphcompute.__file__).resolve().parents:
            raise RuntimeError("Wrong fastgraphcompute import")
    if backend != "core":
        torch.manual_seed(17)
        x = torch.randn(8192, 8, device="cuda", dtype=torch.float32)
        rs = torch.tensor([0, 4096, 8192], device="cuda", dtype=torch.int64)
        b = Backend(backend, x, rs, 16)
        try:
            b.build(cfg)
            idx, distances = b.query(b.qualities(cfg)[0])
            torch.cuda.synchronize()
            if idx.shape != (8192, 16) or not bool(((idx >= 0) & (idx < 8192)).all()):
                raise RuntimeError("GPU backend smoke returned invalid indices")
            if not bool((idx[:4096] < 4096).all() and (idx[4096:] >= 4096).all()):
                raise RuntimeError("Backend crosses event boundaries")
        finally:
            b.close()
    persist_identity(cfg, backend, hardware, software)
    return hardware, software


def benchmark(cfg, task, x, rs, meta):
    import numpy as np
    import torch
    from .backends import Backend, timed
    from .reference import ensure_reference, evaluate
    reference_path = ensure_reference(cfg, task["case"], task, x, rs, meta)
    if min(meta["segment_lengths"]) < task["k"]:
        raise ValueError("Segment is smaller than K")
    rows = []
    for rep in range(-task["warmups"], task["repetitions"]):
        torch.manual_seed(100 + max(rep, 0))
        backend = Backend(task["backend"], x, rs, task["k"], task["bin_dims"], task.get("pca_subsample", 50000))
        qualities = backend.qualities(cfg)
        if not qualities:
            raise ValueError("No supported query-quality settings for this K")
        try:
            _, build_ms = timed(lambda: backend.build(cfg))
            for quality in qualities:
                outputs, query_ms = timed(lambda: backend.query(quality))
                idx, distances = outputs
                if rep >= 0:
                    checks = evaluate(x, rs, idx, distances if task["backend"].startswith("fgc") else None,
                                      reference_path, task["k"], exact=task["backend"].startswith("fgc"))
                    rows.append({"rep": rep, "quality": quality, "build_ms": build_ms,
                                 "query_ms": query_ms, "total_ms": build_ms + query_ms,
                                 "recall": checks, "build_reused_across_query_settings": len(qualities) > 1,
                                 "output_contract": "GPU int64 global event-local indices and float32 squared distances",
                                 "input_protocol": "preloaded_torch_cuda_zero_copy"})
                del idx, distances, outputs
        finally:
            backend.close()
            del backend
            gc.collect()
            torch.cuda.empty_cache()
    return rows


def run_task(cfg, task):
    import torch
    from .data import load_cuda, prepare
    if task["stage"] == "prepare":
        return {"metadata": prepare(cfg, task["case"])}
    if task["stage"] == "model":
        from .model import benchmark_model
        return {"measurements": benchmark_model(cfg, task)}
    x, rs, meta = load_cuda(cfg, task["case"])
    try:
        if task["stage"] == "reference":
            from .reference import ensure_reference
            path = ensure_reference(cfg, task["case"], task, x, rs, meta)
            return {"metadata": read_json(path / "metadata.json"), "reference_path": str(path)}
        if task["stage"] == "benchmark":
            return {"measurements": benchmark(cfg, task, x, rs, meta), "input_metadata": meta}
        if task["stage"] == "profile":
            from .diagnostics import profile
            return {"diagnostics": profile(cfg, task, x, rs,
                      Path(cfg["run_root"]) / "diagnostics" / task_id(task)), "input_metadata": meta}
        if task["stage"] == "counters":
            from .instrument import run_counters
            return {"diagnostics": run_counters(cfg, task, x, rs)}
        raise ValueError("Unknown stage")
    finally:
        del x, rs
        gc.collect()
        torch.cuda.empty_cache()


def run(plan, stage, backend, budget_seconds=12000, require_complete=False, retry_failed=False, unit_id=None):
    from .planning import verify_plan
    verify_plan(plan)
    from .common import verify_artifacts
    verify_artifacts(plan)
    cfg = plan["config"]
    if plan["blockers"]:
        raise ValueError("Plan has unresolved blockers: " + "; ".join(plan["blockers"]))
    tasks = [t for t in plan["tasks"] if t["stage"] == stage and t["backend"] == backend]
    if unit_id:
        from .priority import unit_tasks
        unit, tasks = unit_tasks(plan, unit_id)
        if (unit["stage"], unit["backend"]) != (stage, backend):
            raise ValueError("Work unit does not match the requested stage/backend")
        for dependency in unit["requires"]:
            matching = next(t for t in plan["tasks"] if task_id(t) == dependency)
            path = result_path(cfg, matching)
            if not path.exists() or read_json(path)["status"] != "ok":
                raise ValueError("Work-unit prerequisite is incomplete; use the priority session runner")
    if not tasks:
        raise ValueError(f"No tasks for {stage}/{backend}")
    stop_requested = False
    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    with run_lock(cfg):
        identity_path = Path(cfg["run_root"]) / "campaign.json"
        identity = {"config_hash": plan["config_hash"], "schema_version": 1}
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("run_root belongs to a different config; choose a new directory")
        atomic_json(identity_path, identity)
        hardware, software = preflight(cfg, backend)
        deadline = time.monotonic() + budget_seconds
        for number, task in enumerate(tasks):
            path = result_path(cfg, task)
            if path.exists():
                saved = read_json(path)
                if saved["status"] == "ok":
                    continue
                if not retry_failed:
                    raise ValueError(f"Failed task {path}; investigate then explicitly --retry-failed")
                # Preserve failed evidence rather than replacing it without an audit trail.
                path.rename(path.with_suffix(f".failed-{time.time_ns()}.json"))
            if stop_requested or time.monotonic() >= deadline:
                print("Checkpointed between tasks; invoke the same worker to resume", flush=True)
                return 75 if require_complete else 0
            print(f"{stage}/{backend} {number + 1}/{len(tasks)} {task_id(task)}", flush=True)
            record = {"task": task, "plan_hash": plan["plan_hash"], "config_hash": plan["config_hash"],
                      "hardware": hardware, "software": software, "started": stamp()}
            started = time.monotonic()
            try:
                record.update(run_task(cfg, task))
                if stage == "benchmark" and any(not row["recall"]["pass"] for row in record["measurements"]):
                    raise RuntimeError("Correctness/structural check failed; inspect saved measurements")
                record["status"] = "ok"
            except Exception:
                record["status"] = "failed"
                record["error"] = traceback.format_exc()
                record["elapsed_seconds"] = time.monotonic() - started
                atomic_json(path, record)
                raise
            record["elapsed_seconds"] = time.monotonic() - started
            record["finished"] = stamp()
            atomic_json(path, record)
    return 0
