# Return measurements and regenerate graphs

The destination is **fgc-performance**, locally `/home/export/aarusha/Performance`
for Aarush. Keep each H100 campaign under a new `revision_results/<run-name>/`.
Historical A100 data stay in `reproduction/results/`.
The operator uses the public packet and sends the ZIP; access to Aarush's
private results repository is unnecessary.

## GPU operator: export and send

If using [run_all.sh](QUICK_RUN.md), the returned ZIP is generated automatically
under `run_root/returns/`; send the printed path's file. The manual export below
is for the older scheduled workflow or an additional export.

After checking the chosen plan's coverage, from the same campaign checkout:

```bash
"$FGC_CORE_PYTHON" -m revision_campaign.cli export-results \
  --plan "$FGC_WORK/configs/production.plan.json" \
  --out "$FGC_WORK/h100-focused-results.zip"
```

Use your actual approved plan filename. The command runs on CPU, packages partial
runs honestly, and refuses to overwrite an existing archive. Send the ZIP through
the authors' approved private file-transfer route. State what failed or remains
missing; export success does not certify scientific completion.

The ZIP includes aggregate timing/recall CSV, model and diagnostic JSON, input
hashes/spectra, software/hardware identity, plan/code hashes, coverage counts and
checksums. It excludes raw coordinates, checkpoints, event IDs, full private
configurations, absolute source paths, raw result JSON and profiler traces.
Keep the full run folder, approved plan/config, logs, environment records and
data/model provenance in approved private storage for follow-up analysis.

## Aarush: import and plot

Clone/switch to branch `codex/reviewer-gpu-campaign` of `fgc-performance`. Use
the campaign core environment, or a CPU Python environment with NumPy and
Matplotlib. From that repository root:

```bash
python -m revision_campaign.cli import-results \
  --archive /absolute/path/to/h100-focused-results.zip --name h100-focused-return
python -m revision_campaign.cli plot-results \
  --directory revision_results/h100-focused-return
```

On Falcon, prefix each Python command with a CPU-only allocation, for example
`srun --partition=work --cpus-per-task=2 --mem=8G --time=00:05:00`, and use
`/home/export/aarusha/conda-envs/fgc-fast/bin/python`. No GPU, original operator
paths, raw hits or checkpoint weights are needed for these plots.

```text
revision_results/h100-focused-return/
  manifest.json         # plan/code, device/software and coverage identity
  measurements.csv      # timings, recall, input/K/backend/settings per row
  inputs.json           # input hashes, shapes, sampled PCA spectra
  diagnostics.json      # memory, PCA stages, counters, gradient checks
  model.json            # complete model-step times and checkpoint hashes
  COVERAGE.md           # completed, failed and missing bundle counts
  SHA256SUMS
  figures/              # regenerated recall/time, memory and model-step PDFs
```

Import verifies checksums and refuses duplicate run names; choose a new name
for another return. Plotting also checks the imported aggregates. Figures are
produced only for available evidence. Additional manuscript-specific figures
may need analysis of the retained private audit artifacts.

## Review before publishing

Returned run directories are Git-ignored by default. Review `COVERAGE.md`,
hardware, correctness fields and scientific labels before selectively adding
approved derived files to a results branch in **fgc-performance**. Move reviewed
paper figures into **fastgraph-paper** only when the authors choose them. Avoid
copying H100 rows over A100 CSVs or using cross-device latencies for speedups.
