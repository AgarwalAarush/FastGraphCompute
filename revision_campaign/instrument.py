"""Create an isolated diagnostic FGC build; preserve the timed release checkout."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

from .common import atomic_json, command, file_digest, git_identity, read_json


def prepare_instrumented(cfg, destination):
    source = Path(cfg["library_source"])
    if git_identity(source) != {"commit": cfg["library_commit"], "dirty": False}:
        raise ValueError("Expected clean pinned release checkout")
    dest = Path(destination).resolve()
    if dest.exists():
        raise FileExistsError("Use a NEW diagnostic checkout path")
    subprocess.run(["git", "clone", "--no-hardlinks", str(source), str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--detach", cfg["library_commit"]], check=True)
    path = dest / "fastgraphcompute/extensions/binned_select_knn_cuda_kernel.cu"
    text = path.read_text()
    def replace(old, new, expected):
        nonlocal text
        if text.count(old) != expected:
            raise ValueError(f"Pinned kernel does not match instrumentation anchor: {old[:70]}")
        text = text.replace(old, new)
    replace('#include <cstdlib>', '#include <cstdlib>\n#include <cstdio>\n__device__ unsigned long long fgc_profile_totals[2];', 1)
    replace('int64_t nfilled=1;', 'unsigned long long profile_candidates=0, profile_bins=0;\n    int64_t nfilled=1;', 2)
    replace('int64_t start_vertex = d_bin_boundaries[idx];',
            '++profile_bins;\n            int64_t start_vertex = d_bin_boundaries[idx];', 2)
    replace('float distsq = 0;', '++profile_candidates;\n                float distsq = 0;', 2)
    replace('        distance++;\n    }',
            '        distance++;\n    }\n    atomicAdd(&fgc_profile_totals[0], profile_candidates);\n    atomicAdd(&fgc_profile_totals[1], profile_bins);', 2)
    replace('    const bool use_local =',
            '    void* profile_address = nullptr;\n    C10_CUDA_CHECK(cudaGetSymbolAddress(&profile_address, fgc_profile_totals));\n    C10_CUDA_CHECK(cudaMemsetAsync(profile_address, 0, 2*sizeof(unsigned long long), stream.stream()));\n    const bool use_local =', 1)
    replace('    return std::make_tuple(indices, distances);',
            '    unsigned long long counters[2] = {0,0};\n'
            '    C10_CUDA_CHECK(cudaMemcpyFromSymbolAsync(counters, fgc_profile_totals, sizeof(counters), 0, cudaMemcpyDeviceToHost, stream.stream()));\n'
            '    C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));\n'
            '    std::printf("FGC_COUNTERS queries=%lld candidates=%llu visited_bins=%llu\\n", (long long)n_vert, counters[0], counters[1]);\n'
            '    std::fflush(stdout);\n    return std::make_tuple(indices, distances);', 1)
    path.write_text(text)
    atomic_json(dest / "fgc-diagnostic-build.json", {"base_commit": cfg["library_commit"],
                "kernel_sha256": file_digest(path), "timing_eligible": False,
                "counters": ["candidate distance evaluations excluding self", "valid visited bins including empty bins"]})
    return dest


def run_counters(cfg, task, x, rs):
    """Separate child interpreter so the diagnostic extension cannot replace the timed one."""
    import numpy as np
    from .common import case_dir, task_id
    diagnostic = Path(cfg.get("instrumented_library_source") or "")
    marker = diagnostic / "fgc-diagnostic-build.json"
    if not marker.is_file():
        raise ValueError("Build diagnostic source first; see HANDOFF.md")
    info = read_json(marker)
    if info["base_commit"] != cfg["library_commit"]:
        raise ValueError("Diagnostic build base does not match timed release")
    directory = Path(cfg["run_root"]) / "diagnostics" / task_id(task)
    directory.mkdir(parents=True, exist_ok=True)
    input_path = case_dir(cfg, task["case"])
    env = dict(os.environ, PYTHONPATH=str(diagnostic))
    script = (
        "import numpy as np, torch, sys; "
        "from fastgraphcompute import binned_select_knn_pca, binned_select_knn; "
        "torch.backends.cuda.matmul.allow_tf32=False; "
        "torch.manual_seed(42); "
        "x=torch.from_numpy(np.load(sys.argv[1])).cuda(); "
        "rs=torch.from_numpy(np.load(sys.argv[2])).cuda(); "
        "fn=binned_select_knn_pca if sys.argv[5]=='fgc_pca' else binned_select_knn; "
        "idx,d=fn(int(sys.argv[3]),x,rs,max_bin_dims=int(sys.argv[4])); "
        "torch.cuda.synchronize(); "
        "np.save(sys.argv[6],idx.cpu().numpy()); np.save(sys.argv[7],d.cpu().numpy())"
    )
    output = subprocess.check_output([sys.executable, "-c", script, str(input_path / "coords.npy"),
             str(input_path / "row_splits.npy"), str(task["k"]), str(task["bin_dims"]), task["backend"],
             str(directory / "indices.npy"), str(directory / "distances.npy")], cwd=diagnostic, env=env, text=True)
    (directory / "counters.log").write_text(output)
    match = re.search(r"FGC_COUNTERS queries=(\d+) candidates=(\d+) visited_bins=(\d+)", output)
    if not match:
        raise RuntimeError("Diagnostic binary did not expose counters; rebuild the instrumented extension")
    from .backends import Backend
    import torch
    normal = Backend(task["backend"], x, rs, task["k"], task["bin_dims"])
    torch.manual_seed(42)
    idx, d2 = normal.query({})
    instrumented_dist = torch.from_numpy(np.load(directory / "distances.npy")).to(x.device)
    same = torch.allclose(d2.sort(1).values, instrumented_dist.sort(1).values, rtol=1e-5, atol=1e-6)
    if not same:
        raise RuntimeError("Instrumentation changes returned distances")
    n, candidates, bins = map(int, match.groups())
    return {"queries": n, "candidates": candidates, "visited_bins": bins,
            "candidates_per_query": candidates / n, "visited_bins_per_query": bins / n,
            "matches_release_distance_multiset": same, "timing_eligible": False,
            "source": info, "log": str(directory / "counters.log")}
