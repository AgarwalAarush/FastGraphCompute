# Instructions for the receiving coding agent

Work in public `AgarwalAarush/FastGraphCompute`, branch
`codex/gpu-operator-handoff`. This branch contains the operator packet. Clone
the GPU library separately at commit
`76a869cc8758a8a846b18e93934e2b6078d83279`; this pin contains PCA. Follow SETUP.md
for both checkouts. Aarush maintains returned results in `fgc-performance`
and the manuscript in `fastgraph-paper`; neither private checkout is needed
by the operator.

Read the host's AGENTS.md/local rules, then [HANDOFF.md](HANDOFF.md),
[RUN_PLAN.md](RUN_PLAN.md), [SETUP.md](SETUP.md), [MODEL_ADAPTER.md](MODEL_ADAPTER.md)
and [VALIDATION.md](VALIDATION.md). Use private scratch and new environments;
keep `reproduction/` and existing author work intact.

Your assignment:

Execution uses the priority policy in HANDOFF.md: gates, preferred checkpoint
model steps, learned/event matched cases, bounded controls/diagnostics, breadth.
Put the preferred trained checkpoint/layer first before capture/planning. Use
LOCAL_COMPLEMENT.md for the complementary local strategy. Never fill a missing
method's timing with a result from another GPU class.

1. Resolve the site's CUDA/compiler/Conda/Slurm details and approved input
   paths. Rebuild the pinned sources for H100 (`TORCH_CUDA_ARCH_LIST=9.0`).
   Record actual package versions; never reuse an A100 binary blindly.
2. Run CPU checks and actual CUDA preflight for every selected backend inside
   the appropriate allocation. On Falcon, even Python tests need a CPU Slurm
   allocation. A compiler check alone cannot certify H100 execution.
3. Obtain the real feature schema, written event-layout confirmation, and
   trained-model adapter/checkpoints from the owner. Report missing inputs
   explicitly. Random coordinates or zero-loss models cannot substitute.
4. Generate the exact pilot plan and launch preview for Aarush to review.
   Do not submit experiments until Aarush approves that plan. Review the
   measured pilot budget before generating/launching production. Changed code,
   inputs or configuration require a new reviewed plan and new run root.
   For an already allocated GPU, [QUICK_RUN.md](QUICK_RUN.md) gives the single
   `run_all.sh` entry point, a total session budget and automatic results export.
   Continue with the explicit production procedure in [HANDOFF.md](HANDOFF.md):
   a separate config/run root, coordinate capture, `focused --scope all`, and
   `production.plan.json`. Repeating the pilot never expands it into production.
   The 32–66-hour planning range is in full-A100 hours, not measured H100 time;
   review representative production timings before sizing the recipient runs.
5. Check coverage, correctness and failures. Resume only the same plan; retain
   failed evidence. Keep profiled/counter timings out of clean latency curves
   and keep H100 comparisons separate from historical A100 measurements.
6. Follow [RETURN_RESULTS.md](RETURN_RESULTS.md): export aggregate results,
   retain private audit artifacts, and tell Aarush the plan hash, coverage,
   actual resource use, environment changes and any unresolved failures.

Do not publish raw hits, cached coordinates, weights or private traces to Git.
Do not merge into main or submit the paper. Preserve the library pin unless an
explicitly reviewed change requires a new campaign. If fixing package/API
compatibility, commit the fix on a work branch and document it before timing.

Human-supplied values to request if absent: approved data/checkpoint locations,
model project and environment, event confirmation, scheduler allocation details,
and authorization for the exact generated GPU plan. The code cannot infer these.
