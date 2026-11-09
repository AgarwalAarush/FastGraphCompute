# One script for an allocated GPU

Entry point: **`revision_campaign/run_all.sh`** on branch
`codex/gpu-operator-handoff` of public `AgarwalAarush/FastGraphCompute`.

Do the one-time [setup and builds](SETUP.md) first, including the private site
config, clean pinned library, backend environments and counter build. Setup
time is additional; the script does not install environments or obtain data.
It runs inside an existing full-GPU allocation and never submits another job.

The actual queue now follows the scientific priorities in [HANDOFF.md](HANDOFF.md).
Use [LOCAL_COMPLEMENT.md](LOCAL_COMPLEMENT.md) for the complementary A100 worker.
Regenerate plans after updating this code; measurement/settings coverage stays
explicit in the plan and every results return.

**The pilot and production are different plans.** `--minutes` limits a session;
it does not enlarge the experiment grid or turn the pilot into production.

## First 30-minute debug allocation

From the repository root with `FGC_CORE_PYTHON`, `FGC_CONFIG` and `FGC_WORK`
defined by setup, generate a small synthetic pilot plan:

```bash
"$FGC_CORE_PYTHON" -m revision_campaign.cli plan --config "$FGC_CONFIG" \
  --profile pilot --scope synthetic --out "$FGC_WORK/configs/pilot.plan.json"
```

Review its generated Markdown. Inside the reserved GPU allocation, the one
command to execute all its stages and package the results is:

```bash
bash revision_campaign/run_all.sh \
  --plan "$FGC_WORK/configs/pilot.plan.json" --minutes 25 --execute
```

Omit `--execute` to preview. **25 minutes is the total session work budget**,
shared by all stages/backends. Leave at least five additional minutes for
checkpointing and export. An overlong in-flight bundle may be stopped; completed
bundles remain saved. The script runs each unfinished prerequisite in order,
using its configured Python environment and GPU preflight.

It prints a ZIP path under `run_root/returns/` on completion, budget stop or
worker failure. Send that ZIP to Aarush. Private worker logs are under
`run_root/logs/`; each session has a record under `run_root/sessions/`.

| Exit | Meaning | Next step |
| --- | --- | --- |
| 0 | Local completion, or separately reported whole-case peer coverage | Return the ZIP and inspect local/peer coverage |
| 75 | Session budget/signal stopped an incomplete plan | Return partial results; repeat the identical command in the next allocation |
| 1 or another error | Failure or invalid setup | Inspect the printed error/log; resolve it before continuing |

Do not treat the synthetic pilot as the complete reviewer dataset. Its runtime
is not yet measured on H100, so finishing even the pilot in 30 minutes is not
guaranteed. This first return checks the environment and informs the real budget.

## Production

Follow the complete **[production procedure in HANDOFF.md](HANDOFF.md)**: create
a separate config/run root, supply real inputs, capture coordinates, and generate
the reviewed `focused --scope all` production plan. Then, for an existing
four-hour allocation:

```bash
bash revision_campaign/run_all.sh \
  --plan "$FGC_WORK/configs/production.plan.json" --minutes 235 --execute
```

Repeat the identical command/plan in later allocations to resume until all
production bundles complete. `235` is a session limit, not a completion estimate;
choose a limit at least five minutes shorter than the actual allocation. Long
individual bundles need a longer allocation because they restart if interrupted.

The planning estimate was **32–66 additional A100 GPU-hours, reserving about 70**
for representative coverage with existing checkpoints. H100 duration is unmeasured;
the optional denser `full` grid can exceed this estimate. Setup/builds, queue wait
and training from scratch are additional. A 25-minute pilot does not collect all
production results.

[Results return and CPU graph generation](RETURN_RESULTS.md) works with either
partial or complete returned ZIPs. No manual collection of result files is needed.
