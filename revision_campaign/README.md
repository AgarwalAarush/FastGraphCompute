# FastGraph reviewer GPU campaign

Public operator branch: `codex/gpu-operator-handoff` in
`AgarwalAarush/FastGraphCompute`. Author maintenance/results remain in
`fgc-performance` on `codex/reviewer-gpu-campaign`. No GPU experiment is launched
by importing this package, generating a plan, or previewing launch commands.

Start with the short [operator handoff](HANDOFF.md). It maps the repositories
and covers obtaining code, setup, plan approval, execution and returning data.

| Need | Instructions |
| --- | --- |
| One script inside a short GPU allocation | [QUICK_RUN.md](QUICK_RUN.md) |
| Complement the remote operator, calibrate A100 timings and collect local graphs | [LOCAL_COMPLEMENT.md](LOCAL_COMPLEMENT.md) |
| Review scope, reuse and GPU budget | [RUN_PLAN.md](RUN_PLAN.md) |
| Install, configure, preflight and resume | [SETUP.md](SETUP.md) |
| Give the work to another coding agent | [AGENT_HANDOFF.md](AGENT_HANDOFF.md) |
| Connect a real trained model | [MODEL_ADAPTER.md](MODEL_ADAPTER.md) |
| Return results to this repo and regenerate graphs | [RETURN_RESULTS.md](RETURN_RESULTS.md) |
| See what has actually been validated | [VALIDATION.md](VALIDATION.md) |

The pipeline provides:

- Confirmed event-local raw/standardized HGCAL cohorts and controlled
  synthetic covariance distributions.
- Private, hashed input caches and tiled FP64 direct-distance references
  shared across K values and backend configurations.
- FastGraph PCA/axis, FAISS GPU, cuVS brute force, CAGRA NN-Descent, and
  GGNN with matching GPU-resident inputs and event-local searches.
- Query-quality sweeps on reused ANN builds, recording construction plus
  query time and recall from the same outputs/configuration.
- FP32 distance/index validation, adversarial numerical inputs, gradient
  checks, PCA sampling/bin-dimension controls, PCA spectra, independent
  NVML sampling, CUDA traces, and isolated search-counter builds.
- Real coordinate capture and complete model inference/training-step
  measurements through the actual model owner's adapter.
- Dry-run-first Slurm/local launches, atomic result commits, explicit
  missing/failure reports, and same-input timing/recall plots.
- Checksummed aggregate export/import and CPU-only plotting under
  `revision_results/<run-name>/`, independent of the operator's paths.

Existing author results, figures, source tags, and old run files are
read-only historical evidence and are not included in the public operator packet.
All new results go to the configured NEW
private `run_root`; restricted inputs/checkpoints/caches are never committed
or included in the code handoff ZIP.

The package is prepared for H100 (`TORCH_CUDA_ARCH_LIST=9.0`) and A100
(`8.0`) builds. H100 runtime validation is a required operator preflight;
compile-only and CPU tests cannot certify it. Keep all comparator timings on
the same GPU class and software stack. No A100/H100 speedup scaling is assumed.

This package is based on the supplied summary of nine visible Array reviews.
The original numbered reviews and Reviewer 18's attachment still need an
author audit; the package does not invent their contents.
