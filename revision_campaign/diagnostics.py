"""Isolated memory/profiler passes; never used as clean timing results."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .backends import Backend, timed


def profile(cfg, task, x, rs, output_dir):
    import torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    uuid = str(getattr(props, "uuid", ""))
    if not uuid:
        raise RuntimeError("Cannot map CUDA-visible device to its NVML UUID")
    monitor_files = [output_dir / name for name in ("nvml.ready", "nvml.stop", "nvml.json")]
    for path in monitor_files:
        path.unlink(missing_ok=True)
    backend = Backend(task["backend"], x, rs, task["k"], task["bin_dims"])
    # Warm up outside profiling and memory measurements, then discard index/output state.
    backend.build(cfg)
    warm = backend.query(backend.qualities(cfg)[0])
    torch.cuda.synchronize()
    del warm
    backend.close()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    process = subprocess.Popen([sys.executable, "-m", "revision_campaign.nvml_monitor",
                                *(str(p) for p in monitor_files), uuid])
    try:
        deadline = time.monotonic() + 20
        while not monitor_files[0].exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Independent NVML sampler did not start; install nvidia-ml-py")
            time.sleep(0.02)
        backend.build(cfg)
        outputs = backend.query(backend.qualities(cfg)[0])
        torch.cuda.synchronize()
        memory = {"torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "outputs_held_at_end_sample": True}
        monitor_files[1].touch()
        process.wait(timeout=20)
        if process.returncode:
            raise RuntimeError("NVML sampler failed")
        memory.update(json.loads(monitor_files[2].read_text()))
        # Includes live outputs/index/allocator state. NVML is device-wide and sampled.
        del outputs
        backend.close()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as profiler:
            with torch.profiler.record_function("FGC_campaign/build"):
                backend.build(cfg)
            with torch.profiler.record_function("FGC_campaign/query"):
                outputs = backend.query(backend.qualities(cfg)[0])
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(output_dir / "trace.json"))
        (output_dir / "operators.txt").write_text(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))
        details = {"memory": memory, "trace": str(output_dir / "trace.json"),
                   "timing_eligible": False, "counters_status": "separate diagnostic build required"}
        if task["backend"] == "fgc_pca" and x.shape[1] > task["bin_dims"]:
            module = importlib.import_module("fastgraphcompute.extensions.binned_select_knn")
            v, fit = timed(lambda: module._compute_pca_projection(x.detach(), task["bin_dims"], 50000))
            y, projection = timed(lambda: x.detach() @ v)
            _, search = timed(lambda: module.binned_select_knn(task["k"], x, rs,
                                                               max_bin_dims=task["bin_dims"], bin_coords=y))
            error = (v.double().T @ v.double() - torch.eye(v.shape[1], device=x.device)).abs().max().item()
            details["isolated_stages"] = {"pca_fit_ms": fit, "projection_ms": projection,
                                           "bin_sort_search_ms": search, "orthogonality_max_error": error,
                                           "note": "diagnostic decomposition, not the clean wrapper latency"}
        if task["backend"].startswith("fgc"):
            details["gradient_check"] = gradient_check(task, x, rs)
        return details
    finally:
        monitor_files[1].touch()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        backend.close()


def gradient_check(task, x, rs):
    import torch
    from fastgraphcompute import binned_select_knn, binned_select_knn_pca
    fn = binned_select_knn_pca if task["backend"] == "fgc_pca" else binned_select_knn
    # This diagnostic intentionally uses a bounded prefix, not an event-performance result.
    coords = x[:min(len(x), 8192)].detach().clone().requires_grad_(True)
    splits = torch.tensor([0, len(coords)], device=x.device, dtype=torch.int64)
    previous = None
    max_repeat_difference = 0.0
    max_reference_error = 0.0
    for _ in range(3):
        torch.manual_seed(914)
        coords.grad = None
        idx, d2 = fn(task["k"], coords, splits, max_bin_dims=task["bin_dims"])
        d2.mean().backward()
        gradient = coords.grad.detach().clone()
        reference_coords = coords.detach().clone().requires_grad_(True)
        direct = (reference_coords[:, None, :] - reference_coords[idx.detach()]).square().sum(-1)
        direct.mean().backward()
        max_reference_error = max(max_reference_error, (gradient - reference_coords.grad).abs().max().item())
        if previous is not None:
            max_repeat_difference = max(max_repeat_difference, (gradient - previous).abs().max().item())
        previous = gradient
    return {"queries": len(coords), "max_repeat_absolute_difference": max_repeat_difference,
            "max_fixed_index_autograd_absolute_error": max_reference_error,
            "loss": "mean selected squared distance", "selection_seed_fixed": True,
            "bitwise_determinism_guaranteed": False}
