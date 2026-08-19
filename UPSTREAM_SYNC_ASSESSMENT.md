# Upstream Sync Assessment

## Scope

This document compares the PCA paper-release branch with
`jkiesele/FastGraphCompute` upstream and records the recommended integration
order after the paper release.

## Baseline

- Upstream main: `03486a8` (2026-05-31)
- PCA paper-release: `7e84da8` (2026-08-14)
- Common base: `f951f1e` (2025-12-08)

## Integrated In v1.2.0

1. Current-stream kernel launches from `198bab3`.
2. Active-stream row-split transfer ordering from `357d521`.
3. CUDA device guards from `8a91943`.
4. CUDA placement and same-device validation from `6238965`.
5. Safe Object Condensation bounds, synchronization, and optional-output
   handling from `b7c27a0`, with regression tests.

These changes were integrated manually because direct cherry-picks conflict
with the PCA-specific local/global dispatch, PCA bin-coordinate handling,
autograd-memory changes, and fused scatter path.

## Deferred Changes

- `ca4b267`, `f5eabbd`, and `23f76ba` add and refine a pure-PyTorch reference
  kNN implementation. Useful for debugging, but defer until its API and dtype
  contract are specified.
- `ff6f98d` changes upstream package versioning. Decide a single coherent
  package version for this fork rather than importing it independently.

## Hardware And Multi-GPU Findings

The PCA fork already requests `sm_90` in `setup.py`; upstream adds no H100
compilation support. Its stream and guard work improves correctness on H100
but needs validation there. Upstream also adds no distributed or data-parallel
implementation. `CUDAGuard` enables correct single-process multi-device use;
it is not multi-GPU partitioning.

## Validation Status

Completed on an A100 with CUDA 12.1 and PyTorch 2.5:

- Default and non-default CUDA stream execution.
- Mixed CPU/CUDA input rejection.
- PCA forward/backward exactness against the reference path.
- Object Condensation maximum-size and `calc_m_not=False` cases.
- A compile-only build of every CUDA extension for `sm_90`.

The two-device `cuda:1` regression test is included but skips when fewer than
two GPUs are visible. An H100 runtime smoke test is also still pending because
the validation cluster exposes A100 GPUs only. Neither limitation blocks the
A100 paper release, but both should be completed before claiming runtime
validation on those hardware configurations.
