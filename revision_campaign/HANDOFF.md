# FastGraph GPU operator handoff

**This is the single entry point to forward to the GPU operator or their coding
agent.** It covers getting the code, setup, the run command and returning results.
No access to the original chat is needed.

**The 25-minute command below is a pilot, not the production dataset.** The
same script runs production when given `production.plan.json`. Increasing the
time limit or repeating `pilot.plan.json` only runs more of that pilot.

## Time to reserve

- **Pilot:** try a 25-minute session inside a 30-minute debug allocation,
  after installation/builds. Completion on H100 has not yet been measured.
- **Production:** the earlier planning estimate for representative reviewer
  coverage was **32–66 additional hours occupying one full A100; reserve about
  70 hours**, assuming usable trained checkpoints. This is neither a measured
  runtime of the generated grid nor an H100 estimate. Refine the H100 budget
  using initial measurements, including representative production-size cases.
- Setup/builds, queue waiting and training from scratch are additional. The
  optional denser `full` profile may exceed that budget. Some individual tasks
  may need more than 25 minutes; use longer allocations so they can complete.

The small synthetic pilot checks compatibility; its timings alone cannot
predict large-event or learned-model production cost.

## If only ten GPU-hours are available

**The runner now enforces scientific priorities**, rather than finishing every
reference and every row for one backend before the next. It collects:

1. Numerical gates, then complete-model inference/training-step comparisons at
   the first configured checkpoint. Put the preferred trained checkpoint first.
2. Matched comparisons on the first captured coordinate layer/batch, then
   single/small-batch real events with paired raw/standardized controls.
3. Bounded Gaussian/anisotropic controls and memory/pruning diagnostics.
4. Additional checkpoints, layers, batches, seeds, dimensions, sizes and ablations.

All original timing repetitions, references, ANN quality settings and remaining
grid tasks are retained; bounded synthetic memory/counter cases are added so
diagnostics need not wait for the largest sweep. The generated plan lists the
actual queue units. Each return reports completed/failed/missing work by priority.
This aims for useful partial results within ten hours; H100 completion is not
guaranteed. Some requested inputs may still be missing.

Soft time shares give each priority phase a turn. A slow unit is deferred, with
one more attempt before breadth; default unit cap is 30 minutes (`--unit-minutes`).
Critical numerical, input or exact-FGC failures stop the run. Optional model or
external-comparator failures remain recorded while independent evidence proceeds;
they must be resolved before claiming full coverage. Exit 75 means partial work,
not failure. For an existing ten-hour allocation use `--minutes 595`; for a
four-hour allocation use 235 as below.

**Aarush's complementary A100 queue:** [LOCAL_COMPLEMENT.md](LOCAL_COMPLEMENT.md)
uses `run_complement.sh`, beginning in the middle for at most 40% of the session
then backfilling high-priority gaps. It accepts peer result ZIPs to avoid whole
completed comparisons without combining A100/H100 timing rows.

Update the branch and regenerate a reviewed plan in a new run root before
using this policy. Keep any already-running campaign on its original code.

**Handing this to an agent?** Give it this page and this instruction:

> Clone `AgarwalAarush/FastGraphCompute`, branch `codex/gpu-operator-handoff`.
> Follow `revision_campaign/AGENT_HANDOFF.md` and the setup linked here. Prepare
> and preview the pilot, then prepare the production plan described below,
> flagging missing inputs. Run each approved plan with `run_all.sh` inside my
> GPU allocations and return the printed ZIPs. Pilot completion is only the
> first step; report the remaining production coverage explicitly.

## Where everything lives

| Repository | Purpose |
| --- | --- |
| [Public operator packet](https://github.com/AgarwalAarush/FastGraphCompute/tree/codex/gpu-operator-handoff) — branch `codex/gpu-operator-handoff` | This package: scripts, tests and human/agent instructions; no GitHub sign-in required |
| [FastGraphCompute](https://github.com/AgarwalAarush/FastGraphCompute/tree/76a869cc8758a8a846b18e93934e2b6078d83279) — pinned commit `76a869c` | PCA-enabled GPU library; build this source on the recipient GPU stack |
| Aarush's `fgc-performance` and `fastgraph-paper` checkouts | Author-managed results/graphs and manuscript; operator access is unnecessary |

Both operator checkouts come from the public repository. Start here:

```bash
git clone --branch codex/gpu-operator-handoff https://github.com/AgarwalAarush/FastGraphCompute.git fgc-gpu-campaign
cd fgc-gpu-campaign
```

Keep this operator checkout separate from the clean library checkout at the
pinned commit. [SETUP.md](SETUP.md) creates that second checkout and builds it.

Alternatively, extract `FastGraph-GPU-campaign-handoff.zip` into a new directory.
It contains this package and instructions, without datasets or checkpoints.
For an existing clone, update this branch before generating a new plan; keep
the code for an already running campaign unchanged.

## Prepare once

1. Read [RUN_PLAN.md](RUN_PLAN.md) for scope, reuse and budget decisions.
2. Follow [SETUP.md, steps 1–6](SETUP.md) for site requirements, pinned source
   checkouts, environments, H100 builds, private configuration, tests and CUDA
   preflight. It defines `FGC_WORK`, `FGC_CORE_PYTHON` and `FGC_CONFIG` used below.
3. Supply approved detector files, the feature schema and written event-layout
   confirmation. Learned/model work also needs the owner's real adapter,
   trained checkpoints, batches, loss and optimizer: [MODEL_ADAPTER.md](MODEL_ADAPTER.md).
4. If using a coding agent, give it [AGENT_HANDOFF.md](AGENT_HANDOFF.md).

H100 runtime preflight is still required. The package has CPU checks and sm_90
compile validation; the recipient environment and real model must be validated
on their machine. Keep new H100 results separate from historical A100 results.

## Pilot: check the environment first

Inside the code directory, after setup:

```bash
"$FGC_CORE_PYTHON" -m revision_campaign.cli plan --config "$FGC_CONFIG" \
  --profile pilot --scope synthetic --out "$FGC_WORK/configs/pilot.plan.json"
bash revision_campaign/run_all.sh \
  --plan "$FGC_WORK/configs/pilot.plan.json" --minutes 25
```

These commands generate a JSON/Markdown plan and preview execution. After
Aarush approves that exact plan, run this inside the reserved full-GPU allocation:

```bash
bash revision_campaign/run_all.sh \
  --plan "$FGC_WORK/configs/pilot.plan.json" --minutes 25 --execute
```

The script runs every unfinished stage in order using a **25-minute total work
budget**, exports a results ZIP automatically, and prints its path. Leave five
extra minutes for checkpointing/export. Repeat the identical command in a new
allocation to resume saved work; inspect failures before continuing.

## Production: all configured results

Use a **new private config and run root**, separate from the pilot. For the
commands below, create `$FGC_WORK/configs/h100-production.json` from your site
config, set its new `run_root`, and provide the confirmed detector files/schema,
trained-model adapter, checkpoints, coordinate-layer names, batches, loss and
optimizer. See [SETUP.md, step 8](SETUP.md) and [MODEL_ADAPTER.md](MODEL_ADAPTER.md).

First capture the trained model's real coordinates once, inside a GPU allocation:

```bash
export FGC_PRODUCTION_CONFIG="$FGC_WORK/configs/h100-production.json"
"$FGC_CORE_PYTHON" -m revision_campaign.cli capture --config "$FGC_PRODUCTION_CONFIG"
```

Set `learned_manifest` in that config to the absolute manifest path printed by
capture. Retain the captured files. Capture is separate from `run_all.sh` and
must be completed before planning learned/model production.

Generate the all-scope production plan and preview its execution:

```bash
"$FGC_CORE_PYTHON" -m revision_campaign.cli plan --config "$FGC_PRODUCTION_CONFIG" \
  --profile focused --scope all --out "$FGC_WORK/configs/production.plan.json"
bash revision_campaign/run_all.sh \
  --plan "$FGC_WORK/configs/production.plan.json" --minutes 235
```

Review `production.plan.md`: resolve every blocker and approve the case count,
grid and budget with Aarush. Then, **inside an existing four-hour full-GPU
allocation**, execute:

```bash
bash revision_campaign/run_all.sh \
  --plan "$FGC_WORK/configs/production.plan.json" --minutes 235 --execute
```

This executes preparation, references, synthetic/detector/learned-coordinate
benchmarks, diagnostics/counters and model inference/training-step measurements
for the approved plan. `--scope all` selects these input scopes; `focused`
selects a representative grid. The optional `full` profile adds a denser grid
and ablations; choose it before planning only after a separate scope/cost review,
using a new config/run root. Neither profile retrains a model to convergence.

**235 minutes is one session's work limit, not the total production estimate.**
Use your actual allocation duration minus at least five minutes for export.
Repeat the identical production command/plan in subsequent allocations until
all its bundles finish. A task killed mid-bundle restarts that bundle; it
does not resume inside it, so repeated 25-minute sessions can stall on a long task.
The script does not request or resubmit allocations automatically.

Local completion means exit 0 and no missing/failed bundles or input blockers in
`run_root/report/COVERAGE.md` for the agreed production scope. Exit 75 is partial
progress; inspect failures before retrying any error. A completed pilot or a
successful ZIP export alone does not establish production completion.
If peer ZIPs are used, `covered-with-peer` means evidence is distributed between
machines; local missing counts stay explicit, and each comparison retains its GPU.

## Return results and generate graphs

**Operator:** send the ZIP printed by `run_all.sh` privately to Aarush. It is
under `run_root/returns/`; partial returns include explicit missing/failed coverage.

**Aarush:** in your `Performance/` checkout of
`fgc-performance`, using a Python environment with NumPy and Matplotlib:

```bash
python -m revision_campaign.cli import-results \
  --archive /absolute/path/to/returned-results.zip --name h100-focused-return
python -m revision_campaign.cli plot-results \
  --directory revision_results/h100-focused-return
```

The imported CSV/JSON and coverage report go into `revision_results/<name>/`;
PDF graphs go into its `figures/` directory. This takes no GPU or raw dataset.
On Falcon, run Python inside a CPU Slurm allocation. Returned runs are ignored
by Git until author review. [RETURN_RESULTS.md](RETURN_RESULTS.md) has the brief
return checklist and explains which audit artifacts remain with the operator.
