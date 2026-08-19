# PCA-FastGraph Paper Release

The `pca-fgc-paper-v1.2.0` tag is the hardened public release associated
with the FastGraph paper. It reports package version `1.2.0` and includes
current-stream launches, CUDA device guards and validation, and Object
Condensation bounds fixes. The earlier `v1.0.1` and `v1.1.0` tags remain
immutable audit records.

## Public Components

- Library and tests: this repository.
- Canonical results and figure generator:
  `AgarwalAarush/fgc-performance/reproduction/`.
- Manuscript and reproduction guide: `AgarwalAarush/fastgraph-paper`.
- Pinned d=3 comparison dependency: `AgarwalAarush/clover-knn`.

The PCA path is opt-in through `GravNetOp(..., use_pca=True)` in eager mode.
It evaluates candidate distances in the original coordinate space and is
exact under the paper's distance convention.

The upstream changes incorporated into this release are recorded in
[UPSTREAM_SYNC_ASSESSMENT.md](UPSTREAM_SYNC_ASSESSMENT.md).
