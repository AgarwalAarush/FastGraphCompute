"""Shared loader that maps benchmark requests onto rhe2 shareddata .npy files.

Used by all generate_gpu_*.py and the CPU benchmarker. Single source of truth
for where the input data lives, what the dimensional cap is, and how rows are
sampled.

The data lives at /home/export/rhe2/shareddata/recHitFeatures.npy as a flat
(N, F) float32 memmap. As of writing: N = 1,158,960,407 and F = 10. Override
the directory with the SHAREDDATA_DIR env var. The 10-column cap is enforced
strictly — see Performance/CLAUDE.md for the rationale.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

DEFAULT_SHAREDDATA_DIR = "/home/export/rhe2/shareddata"
REC_HIT_FEATURES_FILENAME = "recHitFeatures.npy"

# Hard cap. recHitFeatures.npy has 10 columns; benchmarks asking for more must
# either be trimmed or the loader extended (see CLAUDE.md, "Adding more dims").
MAX_DIMS = 10

# Cached memmap so repeated calls inside one process don't re-open the file.
_MMAP_CACHE: dict = {}


def _shareddata_dir() -> str:
    return os.environ.get("SHAREDDATA_DIR", DEFAULT_SHAREDDATA_DIR)


def _get_mmap(shareddata_dir: str) -> np.ndarray:
    """Return a cached read-only memmap of recHitFeatures.npy."""
    key = os.path.abspath(shareddata_dir)
    cached = _MMAP_CACHE.get(key)
    if cached is not None:
        return cached
    path = os.path.join(shareddata_dir, REC_HIT_FEATURES_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"recHitFeatures.npy not found at {path!r}. Set SHAREDDATA_DIR to "
            f"override the default ({DEFAULT_SHAREDDATA_DIR!r})."
        )
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(
            f"expected recHitFeatures.npy to be 2D (N, F); got shape {arr.shape}"
        )
    _MMAP_CACHE[key] = arr
    return arr


def feature_shape() -> Tuple[int, int]:
    """Return (N, F) of the underlying recHitFeatures memmap without copying."""
    arr = _get_mmap(_shareddata_dir())
    return int(arr.shape[0]), int(arr.shape[1])


def load_rec_hit_features(
    n_points: int,
    n_dims: int,
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Return a contiguous (n_points, n_dims) array sampled from recHitFeatures.

    Sampling is deterministic: the first ``n_points`` rows and the first
    ``n_dims`` columns of the memmap are taken. This is intentionally cheap
    (no shuffle, no copy beyond the slice) so benchmark timing reflects only
    the algorithm under test, not data loading variance.

    Raises ValueError if n_dims exceeds MAX_DIMS or n_points exceeds the file's
    row count — callers should not silently truncate.
    """
    if n_points < 0 or n_dims < 0:
        raise ValueError(f"n_points and n_dims must be >= 0; got {n_points}, {n_dims}")
    if n_dims > MAX_DIMS:
        raise ValueError(
            f"requested n_dims={n_dims} exceeds MAX_DIMS={MAX_DIMS}. "
            f"recHitFeatures.npy only has {MAX_DIMS} feature columns. "
            f"Trim your benchmark sweep or extend npy_loader.py to splice in "
            f"additional truth tensors (see CLAUDE.md)."
        )
    arr = _get_mmap(_shareddata_dir())
    total_rows = int(arr.shape[0])
    if n_points > total_rows:
        raise ValueError(
            f"requested n_points={n_points} exceeds available rows "
            f"({total_rows}) in recHitFeatures.npy"
        )
    if n_points == 0:
        # Return a real empty array of the right shape so callers' "empty"
        # branches still get a typed buffer.
        return np.empty((0, n_dims), dtype=dtype)
    sliced = arr[:n_points, :n_dims]
    # Force a contiguous copy in the requested dtype. Without this, downstream
    # libraries (faiss, torch, cupy) often complain about mmap-backed strides.
    return np.ascontiguousarray(sliced, dtype=dtype)


def _smoke_test(argv: list[str]) -> int:
    """python -m npy_loader [n_points [n_dims]] -> prints shape/dtype/stats."""
    n_points = int(argv[1]) if len(argv) > 1 else 1000
    n_dims = int(argv[2]) if len(argv) > 2 else 5
    print(f"npy_loader smoke test: load_rec_hit_features({n_points}, {n_dims})")
    print(f"  SHAREDDATA_DIR = {_shareddata_dir()}")
    full_n, full_f = feature_shape()
    print(f"  underlying recHitFeatures shape: ({full_n}, {full_f})")
    arr = load_rec_hit_features(n_points, n_dims)
    print(f"  returned: shape={arr.shape}, dtype={arr.dtype}, "
          f"contiguous={arr.flags['C_CONTIGUOUS']}, nbytes={arr.nbytes}")
    if arr.size:
        print(f"  stats: min={float(arr.min()):+.4g} "
              f"max={float(arr.max()):+.4g} "
              f"mean={float(arr.mean()):+.4g}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_smoke_test(sys.argv))
