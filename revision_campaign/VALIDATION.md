# Validation status — November 10, 2026

## Public handoff recheck

- All 47 CPU tests pass, including executing copies of the worker, local-run
  and collection scripts from an unrelated Slurm spool directory. The launch
  paths now carry the campaign root explicitly. Actual CPU Slurm job 64773
  also loaded the CLI successfully through the spooled worker wrapper.
- All six selected backends passed actual segmented CUDA preflight on the
  local A100 at `mps:100`: FGC PCA/axis, FAISS GPU, cuVS brute force, CAGRA
  NN-Descent and GGNN. cuVS output arrays now pass through a CuPy GPU view
  before DLPack conversion; RAFT's CUDA-array-interface-only type is supported.
  These are local A100 environments, not the recipient's H100 stack.
- Both documented FAISS-1.12.0 and cuVS-25.02 Conda recipes resolved in fresh
  dry-run solves with Python 3.11, CUDA 12.4 and NumPy 1.26.4. This verifies
  dependency resolution; it does not certify installation or H100 execution.
- The full-device A100 historical timing check passed: median 9,932.549 ms
  versus 9,940.165 ms previously (0.077% faster), three fresh processes on the
  unchanged pinned release. The first production wrapper failed before any
  measurements; its logs are retained and its blocked successors canceled.
- The actual synthetic A100 pilot completed all 44 bundles across seven
  cases: numerical references/checks, all six benchmark backends, six memory/
  profiler passes and both diagnostic counter variants. NVML recorded samples,
  and instrumented distances matched the released solver. Repeating the same
  command completed without rerunning any workers. Importing its real results
  ZIP into a separate directory generated eight PDFs. These small pilot
  measurements do not establish production-scale or H100 performance.

The real trained-model checkpoint/adapter and confirmed event metadata remain
missing. Those production scopes cannot be completed from this code alone.

## Completed here

- CPU unit tests exercise local-offset block decoding, invalid/mismatched
  event boundaries, deterministic synthetic data, input-cache corruption,
  immutable plans, input-change detection, dry-run submission behavior,
  missing-input/full-device gates, argument paths with spaces, atomic resume,
  worker budget checkpoints, failure retention, and FP64 reference/tie/index
  checks. The suite now contains 43 tests and runs without any CUDA operations.
- New result-return tests export aggregate measurements, remove the operator's
  original run directory, import on a different path, and regenerate timing,
  memory and model-step PDFs. They verify private-path/event-ID exclusion,
  retained plan/coverage identity, corrupt/unexpected archive rejection,
  duplicate-run protection, plot-time checksum checks and empty partial runs.
  All 20 tests passed in a CPU-only Slurm allocation on November 7.
- The single-allocation controller adds eight CPU tests covering a shared
  session budget across stages, checkpoint resume, automatic partial/complete
  result export, prerequisite failure stops, signal-handler restoration,
  hard termination of an overlong bundle, immutable session plans and preview
  behavior. The 28-test suite passes with mock GPU workers. This validates
  orchestration; no H100 session duration or runtime result is claimed.
- Priority/complement tests verify full task preservation, topological
  prerequisites, early real-model/event comparisons, raw/standardized pairing,
  owner checkpoint/layer preferences, slow-unit deferral, independent progress
  after optional comparator failures, and complement-to-front fallback without
  any peer return. Worker unit selection and prerequisite enforcement pass.
- Peer-return tests require matching input/protocol hashes and an entire case
  from one return; cross-GPU row fragments cannot manufacture completion.
  Coordinated coverage remains separate from local measurement rows. The drift
  gate is CPU-tested for acceptable drift, slowdown, unexpected speedup and
  rejection of partial-MPS allocations and mismatched historical workloads.
  All 43 tests passed in CPU-only Slurm.
- Python compileall syntax checks and bash syntax checks pass.
- The focused all-scope template generates 95 input cases and 788 task
  bundles before real learned snapshots/model tasks are supplied. It reports
  eight missing-input/provenance blockers and previews commands without
  invoking sbatch. Bundle counts are not GPU-hour measurements.
- The separate diagnostic checkout was created from pinned FastGraph commit
  `76a869cc8758a8a846b18e93934e2b6078d83279`, and all five extensions built
  successfully with `TORCH_CUDA_ARCH_LIST=9.0` using CUDA 12.1 and the existing
  Torch 2.5.0 environment. The instrumented search kernel compiled and linked.
  This was a CPU-only Slurm allocation with no GPU resource requested.
- All five isolated diagnostic extensions also built for `sm_80` on November 9
  using CUDA 12.1, Torch 2.5.0 and Conda GCC 11.4 in a CPU-only allocation. The
  system GCC 13 is incompatible with that CUDA toolkit; no unsupported-compiler
  override was used. The timed release checkout/binaries were not changed.
- The saved explanation's Markdown was rendered with pandoc/pdfLaTeX to a
  four-page PDF; all pages were visually inspected together for clipping.
- Historical `reproduction/results/`, figures and manuscript files are not
  modified. Existing untracked `reproduction/run/` files are not included.

## Required on the recipient GPU before production

- Install/resolve the provided environment recipes on the recipient site.
  The new CUDA-12.4/PyTorch-2.5.1/cuVS-25.02 setup recipes were not installed
  here and are not a frozen copy of the old A100 environments.
- Run CUDA backend preflight and the synthetic/event pilot on the actual
  H100. A successful sm_90 compilation does not prove H100 runtime behavior,
  external backend API compatibility, or numerical correctness at scale.
- Verify the independent NVML sampler's UUID mapping, sample count and
  device-wide baseline on that machine. Sampling can miss brief transients.
- Run/validate diagnostic counters against the clean release distance sets.
  Instrumented binaries and profiler timings are excluded from latency plots.
- Supply/validate the real model adapter, checkpoints, task loss, optimizer,
  event batches and coordinate-layer paths. No learned/model results exist
  until those inputs are available and their stages complete.
- Confirm event metadata with its owner. Structural agreement of offset
  totals is not sufficient evidence of event semantics.

Initial preparation through November 8 used only CPU test/build allocations.
On November 9 the user authorized a full-`mps:100` A100 timing-drift check and
a bounded complementary control/ablation campaign; live job IDs/status and
results are kept in the private directory specified in LOCAL_COMPLEMENT.md.
Submission is not evidence of completion. No H100 run or trained-model result
is claimed by these CPU tests/builds.
