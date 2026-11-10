# FastGraph GPU campaign — detailed setup and operation

Start with [HANDOFF.md](HANDOFF.md) for the short checklist; the repository map
is in [OPERATOR_DETAILS.md](OPERATOR_DETAILS.md).
Its `start.sh` command automates fresh synthetic-pilot setup. Use the manual
instructions below for site customization and real event/model production.
This reference supplies the installation, configuration, preflight and resume
details. Read RUN_PLAN.md for scientific approval decisions.

## 1. What the runner needs

- Linux x86-64, an NVIDIA GPU (H100 expected; an A100 can use a separate
  configuration), a working driver, CUDA **12.4 toolkit including nvcc**,
  GCC supported by that toolkit, Conda, git and a build-capable compute node.
  The CUDA version printed by nvidia-smi is driver capability; verify nvcc
  separately. If the site uses a different supported stack, record/review
  the change and pass backend preflight before measuring.
- A whole H100 for timing and memory comparisons, not a MIG slice or a
  partially shared GPU. All methods in each comparison use that GPU class.
- About 64 GiB CPU RAM for builds/runs and at least 100 GiB free private
  scratch as an initial allowance. The restricted feature file is about
  43 GiB; cached cohorts, embeddings, diagnostic outputs and traces need
  additional space. Inspect generated input counts and available capacity
  before the full grid. The code memory-maps the source file rather than
  loading its entire contents into RAM.
- Approved access to the HGCAL feature/offset files and any checkpoints. Ask
  the owners for the feature schema and confirmation of the event mapping.
  Do not upload restricted data or cached coordinates to a public repo.
- For learned/model work: the real model project, dependencies, trained
  checkpoint(s), batches/labels, loss, optimizer and adapter described in
  MODEL_ADAPTER.md. These cannot be inferred from the kNN library.

**Falcon:** follow its local rule to execute Python, builds and tests inside
Slurm allocations. CPU-only tests do not require a GPU. No Falcon home paths,
partition names, QoS names or MPS settings are hardcoded into the portable runner.

## 2. Obtain the branch or the code ZIP

```bash
git clone --branch codex/gpu-operator-handoff https://github.com/AgarwalAarush/FastGraphCompute.git fgc-gpu-campaign
cd fgc-gpu-campaign
```

If you cannot access the repository, unzip
`FastGraph-GPU-campaign-handoff.zip` into a new directory and use that
directory as the code root. It contains the `revision_campaign/` package,
tests, example config and these instructions. It contains no restricted data.

Set paths for YOUR machine; these are examples, not literal directories:

```bash
export FGC_CODE=/absolute/path/to/fgc-gpu-campaign
export FGC_WORK=/absolute/path/to/private/fastgraph-work
mkdir -p "$FGC_WORK/src" "$FGC_WORK/envs" "$FGC_WORK/configs"
cd "$FGC_CODE"
```

Keep the operator checkout above separate from the clean pinned library.
Clone the pinned library and GGNN sources into NEW directories:

```bash
git clone https://github.com/AgarwalAarush/FastGraphCompute.git "$FGC_WORK/src/fgc"
git -C "$FGC_WORK/src/fgc" checkout --detach 76a869cc8758a8a846b18e93934e2b6078d83279
git clone https://github.com/cgtuebingen/ggnn.git "$FGC_WORK/src/ggnn"
git -C "$FGC_WORK/src/ggnn" checkout --detach 2d03b36362bc824e37099092eabb95172ec18b64
```

Do not copy A100 `.so` binaries to the H100. Rebuild against the recipient's
PyTorch/CUDA ABI and `sm_90`. The clean pinned checkout is checked at runtime.

## 3. Create environments and build on a compute node

Load the site's CUDA 12.4/compiler modules first. If needed, create a local
shell file that loads those modules and put its absolute path in `site_env`
in the config; job.sh sources it before invoking the selected interpreter.

These setup scripts create NEW prefixes and refuse to overwrite existing
environments. They use PyTorch 2.5.1 CUDA-12.4 wheels, NumPy 1.26.4, FAISS GPU
1.12.0, and a pinned cuVS 25.02 Conda generation. Those are portable setup
recipes, not a claim that every Conda solver/driver combination was installed
and GPU-tested here. Stop on dependency errors; do not silently upgrade all
libraries or replace FAISS GPU with its CPU-only package.

```bash
export TORCH_CUDA_ARCH_LIST=9.0
bash revision_campaign/setup.sh core "$FGC_WORK/envs/core"
bash revision_campaign/setup.sh faiss "$FGC_WORK/envs/faiss"
bash revision_campaign/setup.sh cuvs "$FGC_WORK/envs/cuvs"
export GGNN_SOURCE="$FGC_WORK/src/ggnn"
bash revision_campaign/setup.sh ggnn "$FGC_WORK/envs/ggnn"
export FGC_CORE_PYTHON="$FGC_WORK/envs/core/bin/python"
bash revision_campaign/setup.sh build-fgc "$FGC_WORK/src/fgc"
```

FastGraph extensions build serially (`build_ext --inplace -j 1`) to avoid
the shared Ninja build-file race. The default two compile workers also
limit RAM use. GGNN source selection and its CUDA architecture are pinned.
If setting up on A100 instead, use `TORCH_CUDA_ARCH_LIST=8.0` and a separate
run configuration with `expected_gpu_regex` set to `A100`.

Source documentation checked for these recipes:
[PyTorch previous releases](https://docs.pytorch.org/get-started/previous-versions/),
[NVIDIA CUDA-X installation](https://docs.rapids.ai/install/),
[FAISS installation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md),
and the [pinned GGNN source](https://github.com/cgtuebingen/ggnn/tree/2d03b36362bc824e37099092eabb95172ec18b64).
Record the actual package resolutions; preflight saves versions/module paths.

## 4. Fill a private configuration

```bash
cp revision_campaign/config.example.json "$FGC_WORK/configs/h100-pilot.json"
export FGC_CONFIG="$FGC_WORK/configs/h100-pilot.json"
```

Edit these fields with a text editor:

| Field | Required value |
| --- | --- |
| run_root | A NEW absolute private output path, e.g. fastgraph-work/runs/h100-pilot-synthetic |
| library_source | Absolute path to the clean pinned src/fgc checkout |
| instrumented_library_source | NEW separate diagnostic source path, e.g. src/fgc-counters |
| python_executables.core/faiss/cuvs/ggnn | Absolute interpreter paths from step 3 |
| expected_gpu_regex | H100, or A100 in a different run configuration |
| full_device_confirmed | true only after arranging a full-device reservation |
| site_env | Optional absolute trusted module-loading shell file |
| detector.* | Actual approved data paths, offset layout, schema and owner-confirmation note |
| model.* | Actual adapter/checkpoints/layers and owner options for learned/model scope |
| slurm.* | Site partition/account/QoS/GRES, CPU/RAM/walltime; null optional fields are omitted |

For the known shared HGCAL artifact, the proposed offset layout is
`concatenated_local_offsets`: each reset to zero starts another block; every
block contributes its local event lengths to a cumulative global vector.
The decoder rejects invalid resets, zero-length events and mismatched totals.
The owner must confirm the semantic mapping. Only then set
`detector.event_layout_confirmed=true` and provide the written note.

A feature schema can contain an object such as
`{"columns":[{"index":0,"name":"OWNER_DEFINED","unit":"OWNER_DEFINED","preprocessing":"OWNER_DEFINED"}]}`,
with entries for all columns. Explicitly mark unavailable definitions as
unknown; do not guess physical positions. The schema/note file hashes are
recorded with the input artifacts.

For a Falcon full-GPU run, site fields are normally `partition=work`,
`qos=heavy`, `gres=mps:100`, with a four-hour walltime. On another site's
H100 cluster the GRES may be `gpu:1` or `gpu:h100:1`; the local operator must
use the site's syntax. Do not copy Falcon's MPS settings to an H100 site
without confirming its scheduler configuration.

## 5. CPU checks, then real CUDA preflight

On a compute node, with the code directory as the working directory:

```bash
"$FGC_CORE_PYTHON" -m unittest discover -s revision_campaign/tests -v
"$FGC_CORE_PYTHON" -m revision_campaign.cli audit-data --config "$FGC_CONFIG"
"$FGC_CORE_PYTHON" -m revision_campaign.cli preflight --config "$FGC_CONFIG" --backend fgc_pca
"$FGC_CORE_PYTHON" -m revision_campaign.cli preflight --config "$FGC_CONFIG" --backend fgc_axis
"$FGC_WORK/envs/faiss/bin/python" -m revision_campaign.cli preflight --config "$FGC_CONFIG" --backend faiss
"$FGC_WORK/envs/cuvs/bin/python" -m revision_campaign.cli preflight --config "$FGC_CONFIG" --backend cuvs_bf
"$FGC_WORK/envs/cuvs/bin/python" -m revision_campaign.cli preflight --config "$FGC_CONFIG" --backend cagra_nnd
"$FGC_WORK/envs/ggnn/bin/python" -m revision_campaign.cli preflight --config "$FGC_CONFIG" --backend ggnn
```

Skip only `audit-data` if running synthetic scope without detector access.
CUDA preflight performs actual small segmented builds/queries, validates
index range and event locality, checks imports, and records device/driver,
compute capability, memory, Torch/CUDA and package versions. It rejects a
wrong GPU class, a dirty/unpinned library and common MIG configurations.
The full-device reservation still needs operator confirmation; code cannot
prove that no other user shares bandwidth. Any failed comparator blocks the
comparison until fixed or explicitly removed from a newly reviewed plan.

## 6. Prepare the diagnostic counter build

```bash
"$FGC_CORE_PYTHON" -m revision_campaign.cli prepare-instrumented-source \
  --config "$FGC_CONFIG" --destination "$FGC_WORK/src/fgc-counters"
bash revision_campaign/setup.sh build-counters "$FGC_WORK/src/fgc-counters"
```

This clones the pinned source into the NEW directory and adds counters for
actual candidate distance evaluations and visited valid bins. It never edits
the timed release checkout. The counters run in a separate interpreter and
check that returned distance multisets agree with the release. The diagnostic
binary adds synchronization/counter overhead and is **ineligible for timing**.
CUDA traces and NVML monitoring likewise run separately from clean timings.

## 7. Generate a reviewable pilot plan and preview commands

```bash
"$FGC_CORE_PYTHON" -m revision_campaign.cli plan --config "$FGC_CONFIG" \
  --profile pilot --scope synthetic --out "$FGC_WORK/configs/pilot.plan.json"
"$FGC_CORE_PYTHON" -m revision_campaign.cli launch \
  --plan "$FGC_WORK/configs/pilot.plan.json" --scheduler slurm
```

The planner writes both JSON and a matching Markdown inventory. Review the
`.plan.md` and the printed commands. They do not submit jobs. If a required
path/model/event confirmation is absent, the appropriate plan is marked
BLOCKED. `--execute` refuses such plans.

**After the author approves this exact plan**, submit:

```bash
"$FGC_CORE_PYTHON" -m revision_campaign.cli launch \
  --plan "$FGC_WORK/configs/pilot.plan.json" --scheduler slurm --execute
```

For a reserved standalone H100 without Slurm, replace `slurm` with `local`.
Keep the same dry-run/review step first. Local execution is serialized and
stops nonzero if a stage cannot finish in its configured time budget; use the
resume command below or increase a reviewed budget.

## 8. Run detector and learned/model production scopes

For the single-script route inside an existing allocation, follow the explicit
pilot/production commands and time budget in [HANDOFF.md](HANDOFF.md). The
`launch --scheduler slurm` commands below are the alternative scheduled route.

Make separate configs with NEW run roots for pilot, focused and full work.
Do not edit a running plan or reuse a run root with a changed config.

After owner confirmation, generate `--scope detector` plans for actual events.
For learned data, implement the real adapter first, then on the reserved GPU:

```bash
"$FGC_CORE_PYTHON" -m revision_campaign.cli capture --config "$FGC_CONFIG"
```

Capture prints the saved `learned/manifest.json` path. Put that path into
`learned_manifest` in the config for the learned/all-scope production run.
The capture file paths must remain accessible to the run. Do not delete or
change them. Generate a new reviewed plan:

```bash
"$FGC_CORE_PYTHON" -m revision_campaign.cli plan --config "$FGC_CONFIG" \
  --profile focused --scope all --out "$FGC_WORK/configs/production.plan.json"
"$FGC_CORE_PYTHON" -m revision_campaign.cli launch \
  --plan "$FGC_WORK/configs/production.plan.json" --scheduler slurm
```

Review the exact bundle count and input/model flags, use pilot runtimes to
check the budget, then add `--execute` to the SAME previewed command when
authorized. The all-scope production config needs real model inputs as well
as exported snapshots; snapshots alone cannot provide model step timings.

The `full` grid is optional extra breadth and can exceed the earlier
70-A100-hour reservation estimate. It must be costed from the pilot rather
than assumed equivalent on H100. Never mix new H100 latencies with old A100
latencies to compute a cross-method speedup.

## 9. Resume, failures and cancellation

Each completed bundle is committed atomically. Completed results are skipped
when the same plan/stage/backend is run again. Input and reference caches
are hashed; corrupt caches are rejected. A run lock prevents overlapping
workers using this output directory.

For example, resume an incomplete CAGRA stage INSIDE the intended allocation:

```bash
"$FGC_WORK/envs/cuvs/bin/python" -m revision_campaign.cli run \
  --plan "$FGC_WORK/configs/production.plan.json" --stage benchmark \
  --backend cagra_nnd --budget-seconds 12000 --require-complete
```

Exit 75 means the final worker chunk stopped before completing all bundles.
Resume it, or review/submit more chunks. A failed check/backend writes a
failed JSON with the traceback and stops the worker. Diagnose it first. Use
`--retry-failed` only after review; the previous failed JSON is preserved.
If fixing a failure changes library code or package versions, start a NEW
run root and plan instead of mixing implementations.

The Slurm launcher records every submitted ID immediately in
`run_root/submission.json`. It refuses to submit the same launch again.
Use those IDs to inspect/cancel the intended jobs if needed. There are no
EXIT-triggered resubmission traps. Dependent jobs can remain pending after a
failed final chunk; cancel or resubmit the reviewed remainder explicitly.
No hidden monitor will restart a canceled campaign.

## 10. Return results and preserve the audit record

Follow [RETURN_RESULTS.md](RETURN_RESULTS.md) to export a checksummed aggregate
ZIP, import it into the authors' performance repository, and regenerate PDF
graphs without the operator's paths or GPU. To inspect the full private report
on the operator's machine (CPU is sufficient):

```bash
"$FGC_CORE_PYTHON" -m revision_campaign.cli report \
  --plan "$FGC_WORK/configs/production.plan.json" --plots
```

Keep the following audit artifacts in approved private storage; supply them
privately if the authors need to investigate a result:

- The exact approved JSON/Markdown plan and run config.
- `environment/`, `results/`, `report/`, logs and `submission.json`.
- `inputs/*/metadata.json`, reference metadata, learned snapshot manifest,
  model/checkpoint identities, schema and data-owner provenance note.
- Relevant diagnostic traces/counter logs as requested. They can contain
  private paths and model structure. Raw cached coordinates and reference
  tensors should stay in approved storage unless explicitly requested.

`report/COVERAGE.md` and missing/failed JSON lists are the completion criteria
for the chosen plan. `measurements.csv` preserves input, K, bin/PCA-sample
setting, GPU identity, repetition, quality, recall, build/query time and
reference coverage. Plot PDFs are new matched-input recall/time curves,
memory and model-step summaries, not replacements for old paper figures.
Detailed raw JSONs retain spectra, counters, numerical/gradient checks and
stage/profiler information for additional figures.

Do not report successful synthetic-only work as a completed learned-model
revision. Likewise, sampled-query checks are not a full-graph exactness
certificate, and NVML sampling is not a guarantee of observing every
instantaneous allocation. These limitations are printed in the report.
