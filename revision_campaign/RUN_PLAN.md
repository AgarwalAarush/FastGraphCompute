# FastGraph revision run plan — review before submission

Date: November 4, 2026; public handoff checked November 10, 2026.
Operator branch: `codex/gpu-operator-handoff` in public
`AgarwalAarush/FastGraphCompute`. Author maintenance remains on
`codex/reviewer-gpu-campaign` in `fgc-performance`.

## What is ready to review

This is an executable experiment harness and an operator handoff. It is not
a claim that the missing model/data-owner inputs have already been supplied,
nor that H100 kernels have been exercised here. The local synthetic A100 pilot
has since passed; see VALIDATION.md for its scope and the required recipient
checks. The source is kept on a branch for author review and later merge.

The exact earlier explanation is saved separately in Downloads as
`FastGraph-reviewer-requests-and-GPU-budget.md` and its matching PDF. The
historical A100 results and manuscript are not changed by this branch.

## Coverage by reviewer request

| Request | Executable stage | Existing material reused | New evidence |
| --- | --- | --- | --- |
| Event-preserving benchmarks | prepare + reference + benchmark, detector scope | Original feature memmap, APIs, timing protocol | Actual event cohorts, multiple batch sizes, segment-local comparisons |
| Learned latent coordinates | capture, then a new plan with learned_manifest | Trained model/checkpoints supplied by owner; benchmark adapters | Real per-layer/per-checkpoint coordinate snapshots, reused across analyses |
| Scaling and feature controls | prepare + benchmark, synthetic/detector scopes | Raw-feature and Gaussian curves, axis/PCA ablation | Standardized cohorts, fixed-trace covariance controls, seeds, selected bin/PCA-sample ablations |
| Numerical correctness | reference + benchmark | Existing proof and direct-distance checker design | FP64 references, full small cases, documented large-input query samples, ties/duplicates/scales/boundaries |
| Matched quality | benchmark | Backend adapters and prior parameter choices | Recall and build+query time for the identical outputs/settings; shared ANN builds across search settings |
| End-to-end model cost | model | GravNet implementation and actual owner model | Inference and forward+real loss+backward+optimizer steps, repeated from restored checkpoints |
| Peak memory | profile | Output-size knowledge and original resident-memory plot | Independent NVML sampler, absolute sampled peak and deltas, PyTorch peak allocations |
| Why pruning works/fails | prepare + profile + counters | Gaussian failures, covariance generators | PCA spectra, isolated stages/CUDA traces, actual candidate and valid-bin counts from a separate diagnostic build |
| Reproducibility and writing | report, provenance records | All original CSVs/figures/releases | Explicit completed/missing/failed coverage and matched-input plots; no new paper edits |

## Inputs still requiring the team

1. **Event provenance.** The available offset file structurally decodes to
   35 blocks × 200 events, ending at 1,158,960,407 feature rows in total.
   Older project notes call the offsets stale. Obtain a written confirmation
   that block order and offsets map to this exact feature matrix; structural
   agreement alone is insufficient. Then set `event_layout_confirmed=true`.
2. **Feature schema.** Supply a JSON description of all stored columns,
   units and preprocessing. Unknown definitions must be labeled unknown,
   not inferred from numeric position. Standardization changes the metric;
   every transformed case gets a separate reference cache.
3. **Checkpoint/model adapter.** Supply real early/intermediate/final
   checkpoints if available, actual model code, batch preparation, coordinate
   module paths, task loss and optimizer. The integration interface is in
   MODEL_ADAPTER.md. There is no random-model substitute. Only available
   checkpoint stages may be claimed.
4. **Recipient site.** Confirm GPU type/VRAM, Linux architecture, CUDA toolkit
   and driver, Slurm account/partition/GRES, storage location, and approved
   access to restricted data. H100 is configured as the expected class, not
   assumed to have been tested or to deliver a particular speedup.

Synthetic work can run independently. Detector and learned/model scopes are
explicitly gated until their respective inputs exist. An all-scope plan with
missing inputs cannot be executed. A partial plan is labeled partial in its
coverage report.

## Proposed execution order

1. Review this file and fill a private config copied from config.example.json.
2. Install environments/build the pinned source on the recipient compute node.
3. Run CPU tests, inspect data metadata, and run one CUDA preflight per backend.
4. Generate a **pilot** synthetic plan and preview all submission commands.
5. With author approval, execute the pilot and inspect failures and actual
   task durations. Correct environment/API problems before production.
6. Once event metadata is confirmed, run a detector pilot using complete events.
7. Capture actual learned coordinates ONCE. Add the returned manifest to a
   new config/run root and generate the learned/all-scope review plan.
8. Use the pilot durations and exact generated bundle counts to approve a
   **focused** production plan. The `full` profile is a denser optional
   study; selected ablation cases sweep bin dimensions and PCA sample sizes.
9. Execute the predeclared priority queue: numerical gates, preferred-checkpoint
   model steps, learned/event matched comparisons, bounded controls/diagnostics,
   additional models and broad sweeps. Each case gets its own prerequisites;
   references for the entire grid no longer delay the first useful comparison.
   Inspect COVERAGE.md, correctness failures, raw metadata and plots.
10. Update the paper and point-by-point responses only after evidence review.

All full-device workers are serialized. The session runner uses soft phase
shares and a 30-minute default work-unit cap, defers slow work and retries it
once before breadth. The complementary strategy begins with middle/control
work for at most 40% of a session, then backfills front-priority gaps. Unit
failures and missing work remain explicit; original measurement settings and
all grid tasks are retained. See LOCAL_COMPLEMENT.md for coordination and the
full-A100 drift gate. The alternative four-hour scheduled worker allocations
stop scheduling new bundles after 12,000 seconds. Atomic completed bundles
are skipped on resume. The last chunk of a stage must finish all its bundles
or exits nonzero, preventing the dependent chain from claiming completion.
The scheduler never auto-resubmits on EXIT or cancellation.

## What the profiles mean

| Profile | Purpose | Replications and controls |
| --- | --- | --- |
| pilot | Check environment, schema and actual GPU runtime | 10k synthetic points, d=4/8, one measured repetition, bounded reference sample; includes 4,096-point fully checked adversarial inputs |
| focused | Representative evidence around the target regime | d=4/8/10, 100k/1M synthetic points, isotropic/90%/99% top-three variance, two data seeds, event batches 1/4/16, three cohorts, K=40, three repetitions |
| full | Additional breadth after pilot/author review | Three seeds, five cohorts, K=10/40/100, extra 50% covariance case; bin dimensions 2–5 and PCA samples 2,000/50,000 only at selected ablation cells |

All small cases up to 8,192 rows receive a full reference check. Larger
cases use 32/512/1,024 sampled queries for pilot/focused/full, plus a query
at every segment start, each searched against ALL eligible segment points.
To fully validate a representative event, increase the selected plan's
`full_reference_limit` above its event size before generating that plan.
Ground-truth cost grows rapidly; do not silently equate a query sample with
full-graph validation. The checker saves exact policies and input hashes.

## Budget interpretation

The preceding 32–66 A100-hour estimate (about 70 hours reserved) was a
planning range for representative coverage with checkpoints available. It
was not a measurement of this grid or an H100 prediction. The full grid can
exceed that range, particularly on high-dimensional isotropic inputs and
large K/bin dimensions. Generate/review counts, run the pilot, then reduce
the grid or reserve more time using measured durations. Stage chunk counts
bound submission slots, not expected consumption; jobs skip completed work
and usually finish before their maximum wall time.

Training checkpoints from scratch and independently retraining models to
compare physics accuracy are outside this budget. Model-stage traces and
step times measure an existing trained model. They do not establish a new
reconstruction-accuracy result.

## Review checklist before giving the execution instruction

- Input provenance and model dependencies are resolved for the chosen scope.
- The generated plan hash, input case list, K/seed/bin/PCA-sample grids and
  actual task counts are understood and approved.
- Pilot correctness and H100 imports/kernels pass in every backend environment.
- A new private run root has adequate space; the operator has approved data access.
- Full-device reservation and scheduler settings match the recipient site.
- The planned stage chunks and time budget are based on the pilot.
- Reports keep prior A100 and new H100 results separate.

After review, execute the previewed command with `--execute` added: either
`run_all.sh` inside an existing allocation (the pilot and production commands
are in [HANDOFF.md](HANDOFF.md)), or `launch --scheduler slurm` to submit the
configured job chain. The session time limit does not change a pilot into a
production plan. The next operator should not need to infer these decisions
from the original chat.
