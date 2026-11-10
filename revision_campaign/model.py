"""Owner-supplied trained models: coordinate capture and meaningful model steps."""
from __future__ import annotations

from pathlib import Path
import copy
import math

from .backends import timed
from .common import adapter, attach_source, atomic_json, digest, file_digest, task_id


def validate_model_config(cfg):
    model = cfg.get("model", {})
    if not model.get("trained_checkpoint_confirmed"):
        raise ValueError("Confirm checkpoint provenance; random/zero-loss substitute models are rejected")
    if not model.get("checkpoints") or not model.get("capture_layers"):
        raise ValueError("Supply checkpoints and the actual coordinate-module names to capture")
    for checkpoint in model["checkpoints"]:
        if not Path(checkpoint["path"]).is_file() or not checkpoint.get("training_stage"):
            raise ValueError("Each checkpoint needs an existing path, label and training_stage")


def capture(cfg):
    import numpy as np
    import torch
    validate_model_config(cfg)
    attach_source(cfg)
    owner = adapter(cfg)
    catalog = {"snapshots": []}
    root = Path(cfg["run_root"]) / "learned"
    root.mkdir(parents=True, exist_ok=True)
    for checkpoint in cfg["model"]["checkpoints"]:
        checkpoint_hash = file_digest(checkpoint["path"])
        model = owner.load(checkpoint, "cuda", cfg)
        model.eval()
        captured = {}
        handles = []
        modules = dict(model.named_modules())
        for name in cfg["model"]["capture_layers"]:
            if name not in modules:
                raise ValueError(f"Coordinate module not found: {name}")
            def hook(_module, _args, output, layer_name=name):
                if not isinstance(output, torch.Tensor) or output.ndim != 2:
                    raise ValueError("Coordinate hooks must return an actual (N,d) coordinate tensor")
                captured[layer_name] = output.detach().float().cpu().numpy()
            handles.append(modules[name].register_forward_hook(hook))
        try:
            for batch_index, batch in enumerate(owner.batches(cfg, "capture", "cuda")):
                if batch_index >= cfg["model"]["capture_batches"]:
                    break
                captured.clear()
                with torch.no_grad():
                    owner.forward(model, batch)
                if set(captured) != set(cfg["model"]["capture_layers"]):
                    raise ValueError("Not every requested coordinate module ran in this forward pass")
                rs = batch["row_splits"].detach().cpu().numpy()
                for layer, coords in captured.items():
                    if len(coords) != int(rs[-1]):
                        raise ValueError("Captured layer changes node count; adapter must export matching row splits")
                    identity = digest({"checkpoint": checkpoint_hash, "layer": layer, "batch": batch_index})[:24]
                    directory = root / identity
                    directory.mkdir(exist_ok=True)
                    np.save(directory / "coords.npy", coords, allow_pickle=False)
                    np.save(directory / "row_splits.npy", rs, allow_pickle=False)
                    catalog["snapshots"].append({"coords": str(directory / "coords.npy"),
                        "row_splits": str(directory / "row_splits.npy"), "dim": coords.shape[1],
                        "metadata": {"trained_checkpoint": True, "checkpoint": checkpoint,
                                     "checkpoint_sha256": checkpoint_hash,
                                     "adapter_sha256": file_digest(cfg["model"]["adapter"]),
                                     "layer": layer, "batch_index": batch_index,
                                     "batch_provenance": batch.get("provenance", {})}})
        finally:
            for handle in handles:
                handle.remove()
        del model
        torch.cuda.empty_cache()
    if not catalog["snapshots"]:
        raise ValueError("Model adapter produced no captured coordinates")
    path = root / "manifest.json"
    atomic_json(path, catalog)
    return str(path)


def benchmark_model(cfg, task):
    import torch
    validate_model_config(cfg)
    owner = adapter(cfg)
    rows = []
    checkpoint_hash = file_digest(task["checkpoint"]["path"])
    for rep in range(task["repetitions"]):
        torch.manual_seed(42 + rep)
        model = owner.load(task["checkpoint"], "cuda", cfg)
        owner.set_backend(model, task["model_backend"], cfg)
        initial = copy.deepcopy(model.state_dict())
        for mode in ("inference", "forward_backward"):
            model.load_state_dict(initial, strict=True)
            torch.manual_seed(42 + rep)
            model.train(mode == "forward_backward")
            optimizer = owner.optimizer(model, cfg) if mode == "forward_backward" else None
            if mode == "forward_backward" and optimizer is None:
                raise ValueError("Supply the actual model optimizer")
            measurements, event_counts = [], []
            for step, batch in enumerate(owner.batches(cfg, "benchmark", "cuda")):
                if step >= task["steps"] + task["warmups"]:
                    break
                def run():
                    if mode == "inference":
                        with torch.no_grad():
                            return owner.forward(model, batch)
                    model.zero_grad(set_to_none=True)
                    output = owner.forward(model, batch)
                    loss = owner.loss(output, batch, cfg)
                    if loss.ndim != 0:
                        raise ValueError("Adapter must supply a scalar task loss")
                    loss.backward()
                    optimizer.step()
                    return loss.detach()
                output, elapsed = timed(run)
                if mode == "forward_backward" and (not bool(torch.isfinite(output)) or float(output) == 0):
                    raise ValueError("Task loss must be finite and nonzero (checked outside the timer)")
                if step >= task["warmups"]:
                    measurements.append(elapsed)
                    event_counts.append(len(batch["row_splits"]) - 1)
                del output
            if len(measurements) != task["steps"]:
                raise ValueError("Insufficient batches; model adapter must provide enough measured steps")
            rows.append({"mode": mode, "rep": rep, "step_ms": measurements,
                         "events_per_step": event_counts, "optimizer_step_included": mode == "forward_backward",
                         "backend": task["model_backend"], "checkpoint": task["checkpoint"],
                         "checkpoint_sha256": checkpoint_hash,
                         "adapter_sha256": file_digest(cfg["model"]["adapter"]),
                         "protocol": "restored checkpoint per mode; inference or forward+task_loss+backward+optimizer"})
            directory = Path(cfg["run_root"]) / "diagnostics" / task_id(task)
            directory.mkdir(parents=True, exist_ok=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True, profile_memory=True) as profiler:
                with torch.profiler.record_function("FGC_campaign/complete_model_step"):
                    diagnostic_output = run()
                torch.cuda.synchronize()
            trace_path = directory / f"{mode}-rep{rep}.json"
            profiler.export_chrome_trace(str(trace_path))
            rows[-1]["diagnostic_trace"] = str(trace_path)
            rows[-1]["profiler_pass_included_in_step_ms"] = False
            del diagnostic_output
        del model, initial
        torch.cuda.empty_cache()
    return rows
