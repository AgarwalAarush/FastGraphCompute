"""Common GPU-resident inputs and event-local build/query adapters."""
from __future__ import annotations

import time


def timed(fn):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000


class Backend:
    def __init__(self, name, x, rs, k, bin_dims=3, pca_subsample=50000):
        import torch
        self.name, self.x, self.rs, self.k, self.bin_dims = name, x, rs, k, bin_dims
        self.pca_subsample = pca_subsample
        self.bounds = rs.cpu().tolist()
        self.parts = [x[a:b] for a, b in zip(self.bounds[:-1], self.bounds[1:])]
        self.indexes = []
        self.resources = []
        self.arrays = []
        if name in ("cuvs_bf", "cagra_nnd"):
            import cupy as cp
            self.arrays = [cp.from_dlpack(part) for part in self.parts]
            if any(int(view.data.ptr) != part.data_ptr() for view, part in zip(self.arrays, self.parts)):
                raise RuntimeError("cuVS input bridge copied coordinates")
        if name == "faiss":
            import faiss.contrib.torch_utils  # noqa: F401
        torch.cuda.synchronize()

    def qualities(self, cfg):
        if self.name == "cagra_nnd":
            # The same graph build is reused only across SEARCH settings.
            return [{"itopk_size": n} for n in cfg["ann"]["cagra_itopk"] if n >= self.k]
        if self.name == "ggnn":
            return [{"tau_query": q, "max_iterations": cfg["ann"]["ggnn_max_iterations"]}
                    for q in cfg["ann"]["ggnn_tau"]]
        return [{}]

    def build(self, cfg):
        if self.name.startswith("fgc"):
            return
        if self.name == "faiss":
            import faiss
            for part in self.parts:
                resources = faiss.StandardGpuResources()
                cpu_index = faiss.IndexFlatL2(part.shape[1])
                index = faiss.index_cpu_to_gpu(resources, part.device.index or 0, cpu_index)
                index.add(part)
                self.resources.append(resources)
                self.indexes.append(index)
        elif self.name == "cuvs_bf":
            from cuvs.neighbors import brute_force
            self.indexes = [brute_force.build(part, metric="sqeuclidean") for part in self.arrays]
        elif self.name == "cagra_nnd":
            from cuvs.neighbors import cagra
            params = cagra.IndexParams(metric="sqeuclidean", build_algo="nn_descent")
            self.indexes = [cagra.build(params, part) for part in self.arrays]
        elif self.name == "ggnn":
            import ggnn
            ggnn.set_log_level(0)
            for part in self.parts:
                index = ggnn.GGNN()
                index.set_base(part)
                index.set_return_results_on_gpu(True)
                index.build(k_build=cfg["ann"]["ggnn_k_build"], tau_build=cfg["ann"]["ggnn_tau_build"],
                            measure=ggnn.DistanceMeasure.Euclidean)
                self.indexes.append(index)
        else:
            raise ValueError(f"Unsupported backend {self.name}")

    def query(self, quality):
        import torch
        if self.name.startswith("fgc"):
            from fastgraphcompute import binned_select_knn, binned_select_knn_pca
            fn = binned_select_knn_pca if self.name == "fgc_pca" else binned_select_knn
            kwargs = {"max_bin_dims": self.bin_dims}
            if self.name == "fgc_pca":
                kwargs["pca_subsample"] = self.pca_subsample
            return fn(self.k, self.x, self.rs, **kwargs)
        output_idx, output_dist = [], []
        for offset, part, index in zip(self.bounds[:-1], self.parts, self.indexes):
            if self.name == "faiss":
                distances, indices = index.search(part, self.k)
            elif self.name == "cuvs_bf":
                from cuvs.neighbors import brute_force
                array = self.arrays[len(output_idx)]
                distances, indices = brute_force.search(index, array, self.k)
            elif self.name == "cagra_nnd":
                from cuvs.neighbors import cagra
                array = self.arrays[len(output_idx)]
                distances, indices = cagra.search(cagra.SearchParams(**quality), index, array, self.k)
            elif self.name == "ggnn":
                import ggnn
                indices, distances = index.query(part, self.k, quality["tau_query"],
                                                quality["max_iterations"], ggnn.DistanceMeasure.Euclidean)
            if not isinstance(indices, torch.Tensor):
                indices = torch.from_dlpack(indices)
            output_idx.append(indices.to(device=self.x.device, dtype=torch.int64) + offset)
            if not isinstance(distances, torch.Tensor):
                distances = torch.from_dlpack(distances)
            output_dist.append(distances.to(device=self.x.device, dtype=torch.float32))
        return torch.cat(output_idx), torch.cat(output_dist)

    def close(self):
        self.indexes.clear()
        self.resources.clear()


def metadata(name):
    import importlib
    import importlib.metadata
    import sys
    import subprocess
    from pathlib import Path
    from .common import file_digest
    package = {"fgc_pca": "fastgraphcompute", "fgc_axis": "fastgraphcompute", "faiss": "faiss",
               "cuvs_bf": "cuvs", "cagra_nnd": "cuvs", "ggnn": "ggnn", "core": "torch"}[name]
    module = importlib.import_module(package)
    versions = {}
    for dist in ("torch", "numpy", "cuvs-cu12", "libcuvs-cu12", "cupy-cuda12x", "faiss-gpu", "ggnn"):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    binaries = {}
    if package == "fastgraphcompute":
        binaries = {str(p): file_digest(p) for p in (Path(module.__file__).parent / "extensions").glob("*.so")}
    return {"python_executable": sys.executable, "module_file": getattr(module, "__file__", None),
            "module_version": getattr(module, "__version__", None), "packages": versions,
            "extension_sha256": binaries,
            "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()}
