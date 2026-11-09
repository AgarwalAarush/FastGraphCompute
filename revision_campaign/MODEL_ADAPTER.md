# Real trained-model interface

The FastGraph repository does not contain the team's reconstruction model,
its verified event/label pipeline, or trained checkpoints. These must be
supplied by the model owner. The pipeline is implemented around the following
interface; no toy network or zero-valued loss is accepted as a substitute.

Set `model.adapter` to an absolute Python file implementing:

```python
def load(checkpoint, device, cfg):
    """Return the real architecture with this checkpoint loaded strictly."""

def batches(cfg, purpose, device):
    """Yield deterministic real batches; purpose is capture or benchmark.

    Each batch is a dict with CUDA float features, CUDA int64 row_splits,
    any task labels required by loss(), and a JSON-serializable provenance
    dict naming event IDs / source / preprocessing. The sequence and shapes
    must match across backend comparisons. Yield enough batches for warmup
    plus every requested measured step, cycling a documented cohort if needed.
    """

def forward(model, batch):
    """Run the actual model and return its task outputs."""

def loss(output, batch, cfg):
    """Return the real, finite, nonzero scalar reconstruction task loss."""

def optimizer(model, cfg):
    """Return the actual training optimizer configured for this model."""

def set_backend(model, name, cfg):
    """Select this model's neighbor backend, preserving other architecture,
    parameters, self-edge policy, metric, device and gradient semantics.
    """
```

`adapter_example.py` is a functional bridge for models using the released
`GravNetOp`. Its `set_backend` toggles the supported axis/PCA path; its other
functions dispatch to the owner's module, selected by
`model.owner_options.project_module`. That module must provide:

- `load_for_fastgraph_benchmark(checkpoint, device, cfg)`
- `fastgraph_benchmark_batches(cfg, purpose, device)`
- `fastgraph_benchmark_loss(output, batch, cfg)`
- `fastgraph_benchmark_optimizer(model, cfg)`
- `fastgraph_set_backend(model, name, cfg)` for additional comparators.

Alternatively, implement the six adapter functions directly in your own
file. Keep owner code importable through the trusted `site_env`/PYTHONPATH.
Use only checkpoints you trust and the training project's intended loader;
strictly restore matching architecture/weights and report the file SHA-256.

Example config entries:

```json
{
  "model": {
    "adapter": "/project/fgc-gpu-campaign/revision_campaign/adapter_example.py",
    "trained_checkpoint_confirmed": true,
    "checkpoints": [
      {"label": "early", "training_stage": "epoch 1", "path": "/private/model/epoch1.pt"},
      {"label": "final", "training_stage": "epoch 50", "path": "/private/model/epoch50.pt"}
    ],
    "capture_layers": ["gravnet_layers.0.space_transformations", "gravnet_layers.3.space_transformations"],
    "capture_batches": 3,
    "backends": ["fgc_pca", "fgc_axis"],
    "owner_options": {"project_module": "my_reconstruction_project.benchmark_bridge"}
  }
}
```

Replace layer paths with the actual coordinate-producing modules. Capture
uses forward hooks on their output, not a random projection or post-hoc
dimension slice. Every captured `(N,d)` output must match the batch's event
boundaries. For architectures that downsample nodes, export layer-specific
coordinates and row splits yourself in the catalog format below.

```json
{
  "snapshots": [{
    "coords": "/private/snapshot/coords.npy",
    "row_splits": "/private/snapshot/row_splits.npy",
    "dim": 4,
    "metadata": {
      "trained_checkpoint": true,
      "checkpoint_sha256": "actual file hash",
      "training_stage": "final",
      "layer": "actual layer name",
      "batch_index": 0,
      "event_ids": [100, 101]
    }
  }]
}
```

Only genuine checkpoint stages and dimensions may be labeled trained. The
model benchmark restores the checkpoint at each repetition and mode, uses
seeded batches, and includes optimizer.step in measured training steps.
Warmups are excluded from reported times, and a separate extra step exports
a CUDA profiler trace. The adapter should mark graph-search operations with
`torch.profiler.record_function("model_neighbor_search")` if the model's
own operator names do not identify them. Traces provide the attribution;
the pipeline does not assume that every GravNet module's time is search.

When comparing a baseline such as FAISS inside a differentiable model,
preserve the same continuous squared-distance gradients by recomputing
distances from the selected indices in Torch, as appropriate. Simply
returning a non-differentiable external distance array would change the
training computation and is not a valid backend swap.

Full convergence/physics-accuracy comparisons are separate from these
throughput measurements. If retraining is required to obtain checkpoints,
first measure real step time and obtain approval for that additional budget.
