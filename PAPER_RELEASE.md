# PCA-FastGraph Paper Release

The `pca-fgc-paper-v1.1.0` tag is the clean public release associated with
the FastGraph paper. The earlier `v1.0.1` tag remains immutable as the
original paper audit record.

## Public Components

- Library and tests: this repository.
- Canonical results and figure generator:
  `AgarwalAarush/fgc-performance/reproduction/`.
- Manuscript and reproduction guide: `AgarwalAarush/fastgraph-paper`.
- Pinned d=3 comparison dependency: `AgarwalAarush/clover-knn`.

The PCA path is opt-in through `GravNetOp(..., use_pca=True)` in eager mode.
It evaluates candidate distances in the original coordinate space and is
exact under the paper's distance convention.

For the pending post-paper hardening work, see
[UPSTREAM_SYNC_ASSESSMENT.md](UPSTREAM_SYNC_ASSESSMENT.md).
