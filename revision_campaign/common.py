"""Standard-library-only configuration, provenance and atomic artifacts."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def git_identity(path):
    return {"commit": command(["git", "-C", str(path), "rev-parse", "HEAD"]),
            "dirty": bool(command(["git", "-C", str(path), "status", "--porcelain",
                                    "--untracked-files=no"]))}


def config(path):
    result = read_json(path)
    if result.get("schema_version") != VERSION:
        raise ValueError("Unsupported configuration schema")
    for key in ("run_root", "library_source"):
        if result.get(key):
            result[key] = str(Path(os.path.expandvars(result[key])).expanduser().resolve())
    if not result.get("run_root"):
        raise ValueError("Set run_root to a NEW private output directory")
    run_root = Path(result["run_root"])
    if run_root in (ROOT, ROOT / "reproduction") or ROOT / "reproduction" in run_root.parents:
        raise ValueError("Output must be outside the historical reproduction/ tree")
    return result


def adapter(cfg):
    path = cfg.get("model", {}).get("adapter")
    if not path or not Path(path).is_file():
        raise ValueError("A real model adapter is required; see MODEL_ADAPTER.md")
    spec = importlib.util.spec_from_file_location("fgc_owner_model_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("load", "batches", "forward", "loss", "set_backend", "optimizer"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"Model adapter must implement {name}()")
    return module


def task_id(task):
    return digest(task)[:24]


def case_dir(cfg, case):
    return Path(cfg["run_root"]) / "inputs" / task_id(case)


def result_path(cfg, task):
    return Path(cfg["run_root"]) / "results" / task["stage"] / (task_id(task) + ".json")


@contextlib.contextmanager
def run_lock(cfg):
    """Prevent concurrent processes from contending for a benchmark device/run."""
    import fcntl
    path = Path(cfg["run_root"]) / ".worker.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another worker owns this run; serialize the campaign") from exc
        yield


def attach_source(cfg):
    """Avoid accidentally loading an old site-packages or another FGC checkout."""
    import sys
    source = Path(cfg["library_source"]).resolve()
    identity = git_identity(source)
    if identity["commit"] != cfg["library_commit"] or identity["dirty"]:
        raise ValueError(f"FGC source must be clean at {cfg['library_commit']}: {identity}")
    sys.path.insert(0, str(source))
    return identity


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def artifact_stats(cfg):
    paths = [cfg.get("learned_manifest"), cfg.get("model", {}).get("adapter")]
    paths += [c.get("path") for c in cfg.get("model", {}).get("checkpoints", [])]
    paths += [cfg.get("detector", {}).get(k) for k in ("features", "row_splits", "feature_schema", "provenance_note")]
    if cfg.get("learned_manifest") and Path(cfg["learned_manifest"]).is_file():
        for snapshot in read_json(cfg["learned_manifest"])["snapshots"]:
            paths += [snapshot["coords"], snapshot["row_splits"]]
    result = {}
    for path in paths:
        if path and Path(path).is_file():
            stat = Path(path).stat()
            result[str(Path(path).resolve())] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def verify_artifacts(plan):
    if artifact_stats(plan["config"]) != plan["artifact_stats"]:
        raise ValueError("An input/checkpoint/schema changed after planning; generate/review a new plan")


def harness_files():
    base = ROOT / "revision_campaign"
    return {str(path.relative_to(base)): file_digest(path)
            for path in sorted(base.rglob("*"))
            if path.is_file() and path.suffix in (".py", ".sh")
            and not any(part in ("local", "generated", "__pycache__") for part in path.relative_to(base).parts)}
