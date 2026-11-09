"""Check device class and persist hardware/software identity per backend."""
from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess

from .common import atomic_json, digest, read_json


def inspect_gpu(cfg):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required; execute inside the intended allocation")
    if not cfg.get("full_device_confirmed"):
        raise ValueError("Confirm full_device_confirmed after reserving a whole GPU for timings")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    name = props.name
    if not re.search(cfg["expected_gpu_regex"], name):
        raise ValueError(f"Device {name!r} does not match expected_gpu_regex")
    if not cfg.get("allow_mig", False) and "MIG" in name.upper():
        raise ValueError("MIG device rejected for this full-device campaign")
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    identity = {"gpu_name": name, "compute_capability": [props.major, props.minor],
                "total_memory_bytes": props.total_memory, "driver_versions": sorted(set(driver.splitlines())),
                "mig_allowed": cfg.get("allow_mig", False)}
    if props.major == 9 and not cfg.get("allow_mig", False) and props.total_memory < 65 * 1024**3:
        raise ValueError("Hopper device has less than 65 GiB: verify full H100 versus a MIG slice")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {**identity, "hardware_hash": digest(identity), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "tf32": False}


def persist_identity(cfg, backend, hardware, software):
    path = Path(cfg["run_root"]) / "environment" / (backend + ".json")
    stable = {"hardware_hash": hardware["hardware_hash"], "torch": hardware["torch"],
              "torch_cuda": hardware["torch_cuda"], "software": software}
    if path.exists() and read_json(path) != stable:
        raise ValueError(f"Hardware or software changed within a run: {path}; start a new run_root")
    atomic_json(path, stable)
