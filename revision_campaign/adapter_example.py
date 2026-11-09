"""Runnable bridge to the model owner's project; no fabricated model/loss.

Set model.owner_options.project_module to an importable module that implements
the functions documented in MODEL_ADAPTER.md. The checkpoint architecture,
meaningful loss and data preparation remain in that actual project.
"""
import importlib


def project(cfg):
    name = cfg["model"]["owner_options"].get("project_module")
    if not name:
        raise ValueError("Set model.owner_options.project_module to the real training project")
    return importlib.import_module(name)


def load(checkpoint, device, cfg):
    return project(cfg).load_for_fastgraph_benchmark(checkpoint, device, cfg)


def batches(cfg, purpose, device):
    return project(cfg).fastgraph_benchmark_batches(cfg, purpose, device)


def forward(model, batch):
    return model(batch["features"], batch["row_splits"])


def loss(output, batch, cfg):
    return project(cfg).fastgraph_benchmark_loss(output, batch, cfg)


def optimizer(model, cfg):
    return project(cfg).fastgraph_benchmark_optimizer(model, cfg)


def set_backend(model, name, cfg):
    if name not in ("fgc_pca", "fgc_axis"):
        return project(cfg).fastgraph_set_backend(model, name, cfg)
    found = 0
    for layer in model.modules():
        if hasattr(layer, "use_pca") and hasattr(layer, "space_transformations"):
            layer.use_pca = name == "fgc_pca"
            found += 1
    if not found:
        raise ValueError("No compatible GravNetOp layers; provide the owner's set_backend implementation")
