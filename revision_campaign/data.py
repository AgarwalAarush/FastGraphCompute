"""Read-only memmap inputs, explicit local-offset decoding, private caches."""
from __future__ import annotations

from pathlib import Path
import os

from .common import atomic_json, case_dir, digest, file_digest, read_json


def decode_splits(values, n_rows, layout):
    import numpy as np
    a = np.asarray(values)
    if a.dtype.kind not in "iu" or a.ndim not in (1, 2) or (a.ndim == 2 and a.shape[1] != 1):
        raise ValueError("row_splits must be an integer vector or (N,1) array")
    a = a.reshape(-1).astype(np.int64)
    if len(a) < 2 or a[0] != 0 or (a < 0).any():
        raise ValueError("Invalid row_splits")
    lengths = []
    blocks = 1
    for previous, current in zip(a[:-1], a[1:]):
        if current < previous:
            if layout != "concatenated_local_offsets" or current != 0:
                raise ValueError("Offsets reset: select confirmed concatenated_local_offsets layout")
            blocks += 1
        else:
            lengths.append(int(current - previous))
    rs = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    if rs[-1] != n_rows or (np.diff(rs) <= 0).any():
        raise ValueError(f"Decoded offsets end at {rs[-1]}, expected {n_rows}; zero/negative events rejected")
    return rs, blocks


def audit_detector(cfg):
    import numpy as np
    data = cfg["detector"]
    x = np.load(data["features"], mmap_mode="r", allow_pickle=False)
    if x.ndim != 2 or x.dtype != np.float32:
        raise ValueError("Expected an (N,d) float32 feature memmap")
    rs, blocks = decode_splits(np.load(data["row_splits"], allow_pickle=False), len(x), data["layout"])
    lengths = np.diff(rs)
    return {"shape": list(x.shape), "blocks": blocks, "events": len(lengths),
            "min_hits": int(lengths.min()), "median_hits": float(np.median(lengths)),
            "max_hits": int(lengths.max()), "metadata_confirmed": data.get("event_layout_confirmed", False)}


def synthetic(case):
    import numpy as np
    rng = np.random.default_rng(case["seed"])
    n, d = case["points"], case["dim"]
    x = rng.standard_normal((n, d)).astype(np.float32)
    kind = case["distribution"]
    if kind.startswith("anisotropic_"):
        fraction = float(kind.split("_")[1])
        # Unit trace for every d. Top three axes capture the configured fraction.
        eigenvalues = np.full(d, (1 - fraction) / max(d - 3, 1))
        eigenvalues[:3] = fraction / 3
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        x = ((x * np.sqrt(eigenvalues)) @ q.T).astype(np.float32)
    elif kind == "duplicates":
        x[1::2] = x[::2][:len(x[1::2])]
    elif kind == "near_ties":
        x[:] = 0
        x[:, 0] = np.linspace(-1, 1, n, dtype=np.float32)
        x[:, 1] = np.nextafter(x[:, 0], np.float32(np.inf))
    elif kind == "large_offset":
        x = (x * 0.01 + 10000).astype(np.float32)
    elif kind == "extreme_scale":
        x *= np.geomspace(1e-4, 1e4, d).astype(np.float32)
    elif kind == "bin_boundary":
        x[:] = rng.integers(-3, 4, (n, d))
        x[::2] = np.nextafter(x[::2], np.float32(np.inf))
    elif kind != "gaussian":
        raise ValueError(f"Unsupported distribution {kind}")
    return x, np.array([0, n], dtype=np.int64), {"generator": kind, "seed": case["seed"]}


def prepare(cfg, case):
    import numpy as np
    directory = case_dir(cfg, case)
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        meta = read_json(metadata_path)
        for name in ("coords.npy", "row_splits.npy"):
            if file_digest(directory / name) != meta["hashes"][name]:
                raise ValueError(f"Cached input corrupted: {directory / name}")
        return meta
    directory.mkdir(parents=True, exist_ok=True)
    if case["source"] in ("synthetic", "adversarial"):
        x, rs, provenance = synthetic(case)
    elif case["source"] == "detector":
        data = cfg["detector"]
        if not data.get("event_layout_confirmed"):
            raise ValueError("Event provenance is unconfirmed")
        x_source = np.load(data["features"], mmap_mode="r", allow_pickle=False)
        rs_source, blocks = decode_splits(np.load(data["row_splits"], allow_pickle=False),
                                         len(x_source), data["layout"])
        rng = np.random.default_rng(case["seed"] + case["cohort"] * 1009)
        # Same events across raw/standardized and all dimensional prefixes.
        event_ids = np.sort(rng.choice(len(rs_source) - 1, case["events"], replace=False))
        columns = data.get("columns", list(range(x_source.shape[1])))[:case["dim"]]
        if len(columns) != case["dim"]:
            raise ValueError("Insufficient documented feature columns")
        segments = [np.asarray(x_source[rs_source[i]:rs_source[i + 1], :][:, columns]) for i in event_ids]
        x = np.ascontiguousarray(np.concatenate(segments), dtype=np.float32)
        rs = np.concatenate(([0], np.cumsum([len(s) for s in segments], dtype=np.int64)))
        provenance = {"event_ids": event_ids.tolist(), "offset_blocks": blocks,
                      "feature_columns": columns, "features_path": data["features"],
                      "features_size": Path(data["features"]).stat().st_size,
                      "offsets_sha256": file_digest(data["row_splits"]),
                      "schema_sha256": file_digest(data["feature_schema"]),
                      "owner_note_sha256": file_digest(data["provenance_note"]),
                      "dimension_interpretation": "feature-prefix: dimension and feature identity both change"}
    elif case["source"] == "learned":
        x = np.array(np.load(case["coords"], mmap_mode="r", allow_pickle=False), dtype=np.float32, copy=True)
        rs, _ = decode_splits(np.load(case["row_splits"], allow_pickle=False), len(x), "global")
        provenance = dict(case["snapshot_metadata"])
        if provenance.get("trained_checkpoint") is not True:
            raise ValueError("Learned snapshots require trained-checkpoint provenance")
    else:
        raise ValueError("Unknown input source")
    if x.ndim != 2 or x.shape[1] != case["dim"] or not np.isfinite(x).all():
        raise ValueError("Coordinates must be finite (N,d)")
    if case["transform"] == "standardized":
        mean = x.mean(axis=0, dtype=np.float64)
        scale = x.std(axis=0, dtype=np.float64)
        scale[scale == 0] = 1
        x = ((x - mean) / scale).astype(np.float32)
        provenance["normalization"] = {"mean": mean.tolist(), "scale": scale.tolist(),
                                       "fit_scope": "same selected benchmark cohort; no model training"}
    sample = x[::max(1, len(x) // 50000)][:50000].astype(np.float64)
    covariance = np.atleast_2d(np.cov(sample, rowvar=False))
    eig = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0)
    fractions = np.cumsum(eig) / max(float(eig.sum()), np.finfo(float).tiny)
    for name, value in (("coords.npy", x), ("row_splits.npy", rs)):
        temporary = directory / (name + ".pending")
        with open(temporary, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
        os.replace(temporary, directory / name)
    meta = {"case": case, "shape": list(x.shape), "segment_lengths": np.diff(rs).tolist(),
            "provenance": provenance, "sampled_pca_cumulative_variance": fractions.tolist(),
            "pca_sample_count": len(sample),
            "hashes": {name: file_digest(directory / name) for name in ("coords.npy", "row_splits.npy")}}
    meta["input_hash"] = digest(meta["hashes"])
    atomic_json(metadata_path, meta)
    return meta


def load_cuda(cfg, case):
    import numpy as np
    import torch
    directory = case_dir(cfg, case)
    meta = prepare(cfg, case)
    x = torch.from_numpy(np.array(np.load(directory / "coords.npy", mmap_mode="r", allow_pickle=False), copy=True)).cuda()
    rs = torch.from_numpy(np.load(directory / "row_splits.npy", allow_pickle=False)).cuda()
    return x, rs, meta
