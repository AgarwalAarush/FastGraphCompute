# Complementary local A100 run

The remote operator uses `run_all.sh` to collect front-priority evidence. Use
**`run_complement.sh`** locally: after numerical gates, it spends **at most 40%**
of a session on controls, diagnostics and middle/later cases, then falls back
to unfinished front-priority work. It does not assume the remote operator
finishes anything. Soft phase shares and a default 30-minute unit cap prevent
one slow comparison from consuming the whole session; deferred work gets one
more opportunity before broad sweeps. Completed atomic bundles are retained.

## Run inside a reserved GPU allocation

Prepare a separate A100 config/run root and generate the reviewed plan using
the same code revision as the other operator. A100 and H100 each run all methods
for their assigned cases; do not put one method on each GPU and combine timings.

```bash
bash revision_campaign/run_complement.sh \
  --plan /absolute/path/local.plan.json --minutes 235 --execute
```

The example uses an existing four-hour allocation. For a ten-hour allocation,
use `--minutes 595`. On Falcon, request **`--qos=heavy --gres=mps:100`**; its
four-hour QoS limit means using bounded sequential allocations instead of
requesting ten hours in one job. Setup/compilation should use CPU allocations.

If remote returns are available at startup:

```bash
bash revision_campaign/run_complement.sh \
  --plan /absolute/path/local.plan.json --minutes 235 --execute \
  --peer-results /absolute/path/remote-results.zip
```

Whole cases are skipped only when **one checksummed peer return** covers every
local benchmark/diagnostic setting for that case, with the identical input hash
and protocol. Local preparation may still be needed to establish the hash.
Fragments from different GPUs are never combined to manufacture a completed
comparison. Numerical gates always run locally; model steps are not automatically
skipped using another machine's model timings. Older ZIPs without the new
comparison coverage cannot be used for automatic skipping.

Returns are read at session startup, not fetched from another person. Supply new
ZIPs in the next invocation. Local missing rows remain missing; peer evidence
is recorded separately in the exported manifest and coverage report.

## Drift gate and the prepared Falcon campaign

`calibrate.py` invokes the unchanged historical FGC checker: first five million
HGCAL rows, d=8, K=40, three PCA bin dimensions, GPU-resident input. Three fresh
process measurements must have a median within **±10%** of the saved v1.2.0
median (9,940.165 ms). This is a timing check on the old unsegmented stress case;
it does not confirm event provenance or replace new correctness references.

```bash
python -m revision_campaign.calibrate --source /absolute/path/pinned-fgc \
  --historical /absolute/path/historical-summary.json --out /private/new-calibration.json
```

Calibration requires an actual Slurm `mps:100` A100 allocation. Production is
gated on its saved pass result; a change of compute node triggers a recheck and
that time is subtracted from the allocation's work budget. The timed release
source and its binaries remain unchanged. Diagnostic counters use a separate
sm_80 build.

The bounded Falcon submission script is `submit_complement.sh`. Invoke it from
a CPU allocation after the source builds and plan checks pass:

```bash
bash revision_campaign/submit_complement.sh /absolute/path/local.plan.json \
  /private/new-calibration.json /absolute/path/core/bin/python CALIBRATION_JOB_ID
```

Replace `CALIBRATION_JOB_ID` with the numeric submitted drift-check job ID, or
omit it if calibration already completed successfully. The submission records
each ID immediately and refuses to overwrite its manifest. GPU slots are
serialized with CPU plotting/export between them. Cancellation or a critical
failure prevents later GPU slots from starting; no canceled job resubmits itself.

The prepared local run directory is
`/home/export/aarusha/tmp/fgc-complement-20261109/`. A ten-hour maximum allocation
allowance consists of the initial 20-minute calibration slot, then 4h + 4h +
1h40m GPU slots (235/235/95-minute work limits). Jobs request `mps:100`, run
serially, and have no resubmission trap. CPU-only collection runs between slots
to export results and generate graphs; a critical numerical/input failure stops
the remaining GPU chain. Optional comparator failures stay visible while
independent work continues. All submitted IDs are recorded in that private
directory; cancel all remaining IDs to stop the entire bounded campaign.

Only synthetic controls and PCA/bin/sampling ablations are locally ready at
preparation time. Real event/model work still needs owner-confirmed metadata
and the actual trained-model pipeline. Missing inputs are not replaced with
fake learned coordinates. A separate real-input plan can use the same strategy
once those inputs are supplied.

Private results: `results/report/`, `results/returns/`, `results/sessions/` and
`results/logs/`. Code is frozen in a separate snapshot for the queued jobs;
later edits to the public branch do not invalidate that queued plan. Imported
remote and local graphs remain labeled by GPU class. Do not overwrite the
historical `reproduction/results/` or paper figures before author review.
