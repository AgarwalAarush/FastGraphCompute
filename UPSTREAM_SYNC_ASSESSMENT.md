# Upstream Sync Assessment

## Scope

This document compares the PCA paper-release branch with
`jkiesele/FastGraphCompute` upstream and records the recommended integration
order after the paper release.

## Baseline

- Upstream main: `03486a8` (2026-05-31)
- PCA paper-release: `7e84da8` (2026-08-14)
- Common base: `f951f1e` (2025-12-08)

## Required Before The Next Production Release

1. Port current-stream kernel launches from `198bab3`.
2. Port active-stream row-split transfer ordering from `357d521`.
3. Port CUDA device guards from `8a91943`.
4. Port CUDA placement and same-device validation from `6238965`.
5. Port the safe Object Condensation bounds and synchronization fixes from
   `b7c27a0`, with regression tests.

These commits must be integrated manually: direct cherry-picks conflict with
the PCA-specific local/global dispatch, PCA bin-coordinate handling,
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

## Required Validation After Porting

- Default and non-default CUDA streams.
- `cuda:1` inputs while `cuda:0` is current.
- Mixed-device and CPU-input rejection.
- PCA forward/backward exactness against the reference path.
- Object Condensation maximum-size and `calc_m_not=False` cases.
- H100 build and smoke test, followed by paper exactness/performance smokes.
