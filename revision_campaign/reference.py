"""Bounded-memory FP64 references; cache once for every K and backend."""
from __future__ import annotations

from pathlib import Path
import os

from .common import atomic_json, case_dir, digest, file_digest, read_json


def reference_dir(cfg, case, max_k, query_count, full_limit):
    policy = dict(max_k=max_k, query_count=query_count, full_limit=full_limit,
                  arithmetic="fp64_direct_subtraction", version=1)
    return case_dir(cfg, case) / ("reference-" + digest(policy)[:16])


def compute_reference(x, rs, k, query_count, full_limit, seed=42):
    import numpy as np
    import torch
    bounds = rs.cpu().tolist()
    if min(b - a for a, b in zip(bounds[:-1], bounds[1:])) < k:
        raise ValueError("Every segment must contain at least max(K) points")
    n = len(x)
    if n <= full_limit:
        qids = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        qids = np.unique(np.concatenate((rng.choice(n, min(query_count, n), replace=False),
                                         np.array(bounds[:-1], dtype=np.int64))))
    q = torch.from_numpy(qids).to(device=x.device)
    out_dist = torch.empty((len(q), k), dtype=torch.float64, device=x.device)
    out_idx = torch.empty((len(q), k), dtype=torch.int64, device=x.device)
    for start, end in zip(bounds[:-1], bounds[1:]):
        positions = torch.where((q >= start) & (q < end))[0]
        for offset in range(0, len(positions), 16):
            rows = positions[offset:offset + 16]
            queries = x[q[rows]].double()
            best_d = torch.full((len(rows), k), float("inf"), dtype=torch.float64, device=x.device)
            best_i = torch.zeros_like(best_d, dtype=torch.int64)
            for base in range(start, end, 16384):
                stop = min(base + 16384, end)
                distances = (queries[:, None, :] - x[base:stop].double()[None, :, :]).square().sum(-1)
                ids = torch.arange(base, stop, device=x.device).expand(len(rows), -1)
                merged_d = torch.cat((best_d, distances), dim=1)
                merged_i = torch.cat((best_i, ids), dim=1)
                best_d, order = merged_d.topk(k, dim=1, largest=False, sorted=True)
                best_i = merged_i.gather(1, order)
            out_dist[rows], out_idx[rows] = best_d, best_i
    return q.cpu().numpy(), out_dist.cpu().numpy(), out_idx.cpu().numpy()


def ensure_reference(cfg, case, task, x, rs, meta):
    import numpy as np
    directory = reference_dir(cfg, case, task["max_k"], task["query_count"], task["full_limit"])
    marker = directory / "metadata.json"
    if marker.exists():
        saved = read_json(marker)
        if saved["input_hash"] != meta["input_hash"]:
            raise ValueError("Reference input identity mismatch")
        for name, expected in saved["hashes"].items():
            if file_digest(directory / name) != expected:
                raise ValueError("Corrupt reference cache")
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    q, distances, indices = compute_reference(x, rs, task["max_k"], task["query_count"],
                                              task["full_limit"], case["seed"])
    for name, array in (("queries.npy", q), ("distances.npy", distances), ("indices.npy", indices)):
        temporary = directory / (name + ".pending")
        with open(temporary, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, directory / name)
    atomic_json(marker, {"input_hash": meta["input_hash"], "max_k": task["max_k"],
                         "queries": len(q), "query_coverage": "full" if len(q) == len(x) else "sampled_queries_all_candidates",
                         "arithmetic": "float64_direct_coordinate_subtraction",
                         "hashes": {name: file_digest(directory / name) for name in
                                    ("queries.npy", "distances.npy", "indices.npy")}})
    return directory


def evaluate(x, rs, indices, distances, reference_path, k, exact=False):
    import numpy as np
    import torch
    q = torch.from_numpy(np.load(Path(reference_path) / "queries.npy", allow_pickle=False)).to(x.device)
    truth = torch.from_numpy(np.load(Path(reference_path) / "distances.npy", allow_pickle=False)[:, :k]).to(x.device)
    predicted = indices[q].long()
    bounds = rs.cpu().tolist()
    all_structural_ok = True
    for start, end in zip(bounds[:-1], bounds[1:]):
        for offset in range(start, end, 65536):
            block = indices[offset:min(offset + 65536, end)]
            in_segment = ((block >= start) & (block < end)).all()
            sorted_indices = block.sort(1).values
            no_duplicates = (sorted_indices[:, 1:] != sorted_indices[:, :-1]).all()
            all_structural_ok = all_structural_ok and bool((in_segment & no_duplicates).item())
    valid = (predicted >= 0) & (predicted < len(x))
    for start, end in zip(bounds[:-1], bounds[1:]):
        mask = (q >= start) & (q < end)
        valid[mask] &= (predicted[mask] >= start) & (predicted[mask] < end)
    ordered = predicted.sort(1).values
    unique = (ordered[:, 1:] != ordered[:, :-1]).all(1)
    safe = predicted.clamp(0, len(x) - 1)
    recomputed = (x[safe].double() - x[q].double()[:, None, :]).square().sum(-1)
    within = recomputed <= truth[:, -1:] * (1 + 1e-4) + 1e-10
    # Invalid/duplicate rows cannot receive full recall credit.
    recall = (within & valid & unique[:, None]).double().mean().item()
    sorted_d = recomputed.sort(1).values
    strict = torch.isclose(sorted_d, truth, rtol=1e-5, atol=1e-7).all().item()
    returned_ok = None
    if distances is not None:
        returned_ok = torch.isclose(distances[q].double(), recomputed, rtol=1e-5, atol=1e-6).all().item()
    result = {"distance_recall_rtol_1e_4_atol_1e_10": recall,
              "all_query_structural_checks_pass": all_structural_ok,
              "valid_segment_local_indices": bool(valid.all().item()),
              "unique_indices": bool(unique.all().item()), "tested_queries": len(q),
              "reference_coverage": "full" if len(q) == len(x) else "sampled_queries_all_candidates",
              "distance_multiset_rtol_1e_5_atol_1e_7": bool(strict),
              "returned_distance_check": returned_ok,
              "max_absolute_distance_error": float((sorted_d - truth).abs().max().item())}
    result["pass"] = bool(all_structural_ok and valid.all().item() and unique.all().item()
                          and (not exact or (strict and returned_ok is not False)))
    return result
