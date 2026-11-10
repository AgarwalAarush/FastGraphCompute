"""Owned, resumable setup and a bounded synthetic pilot; no scheduler submission."""
from __future__ import annotations

import argparse
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, atomic_json, digest, file_digest, git_identity, harness_files, read_json
from .planning import make_plan, verify_plan, write_plan


FGC_PIN = "76a869cc8758a8a846b18e93934e2b6078d83279"
GGNN_PIN = "2d03b36362bc824e37099092eabb95172ec18b64"


def claim_work(work, identity):
    """Never adopt or overwrite an unrelated work directory."""
    marker = work / "bootstrap-owner.json"
    if marker.exists():
        if read_json(marker) != identity:
            raise ValueError("Setup code/GPU/site changed: use a NEW --work directory; existing work is preserved")
    else:
        if work.exists() and any(work.iterdir()):
            raise ValueError("--work must be NEW or empty; this directory has no bootstrap ownership record")
        work.mkdir(parents=True, mode=0o700, exist_ok=True)
        atomic_json(marker, identity)


class Setup:
    def __init__(self, work, environment):
        self.work, self.environment = work, environment
        self.state, self.logs = work / "setup-state", work / "setup-logs"
        self.state.mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)

    def run(self, name, args):
        log = self.logs / f"{name}-{time.time_ns()}.log"
        print(f"{name} ... log: {log}", flush=True)
        with open(log, "w") as output:
            result = subprocess.run(args, cwd=ROOT, env=self.environment,
                                    stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            print("\n".join(log.read_text(errors="replace").splitlines()[-20:]), file=sys.stderr)
            raise RuntimeError(f"{name} failed (exit {result.returncode}); fix the error and repeat the SAME command. Log: {log}")

    def step(self, name, args, expected):
        marker = self.state / (name + ".json")
        if marker.exists():
            if not expected.exists():
                raise ValueError(f"Completed setup artifact is missing: {expected}; use a NEW --work directory")
            print(f"{name}: already prepared", flush=True)
            return
        self.run(name, args)
        if not expected.exists():
            raise RuntimeError(f"{name} did not produce {expected}")
        atomic_json(marker, {"complete": True})


def pilot_config(work, gpu, site_env):
    cfg = read_json(ROOT / "revision_campaign/config.example.json")
    cfg.update(run_root=str(work / "runs/synthetic-pilot"), library_source=str(work / "src/fgc"),
               instrumented_library_source=str(work / "src/fgc-counters"), library_commit=FGC_PIN,
               expected_gpu_regex=gpu, full_device_confirmed=True, site_env=site_env)
    cfg["python_executables"] = {name: str(work / "envs" / name / "bin/python")
                                 for name in ("core", "faiss", "cuvs", "ggnn")}
    return cfg


def save_pilot(cfg, path):
    """Preserve the original immutable config/plan across allocation boundaries."""
    config_path = path.parent / "synthetic-pilot.config.json"
    if config_path.exists():
        if read_json(config_path) != cfg:
            raise ValueError("The generated pilot config was edited; keep it intact or use a NEW --work")
    else:
        atomic_json(config_path, cfg)
    if path.exists():
        plan = read_json(path)
        verify_plan(plan)
        if plan["config"] != cfg or plan["profile"] != "pilot" or plan["scope"] != "synthetic":
            raise ValueError("Existing plan differs from the documented synthetic pilot")
    else:
        plan = make_plan(cfg, "pilot", "synthetic")
        write_plan(plan, path)
    return config_path, plan


def start(args):
    if not args.full_gpu:
        raise ValueError("Reserve a full GPU, then explicitly confirm it with --full-gpu")
    if not math.isfinite(args.minutes) or args.minutes <= 0:
        raise ValueError("--minutes must be finite and positive")
    work = Path(args.work).expanduser().resolve()
    if work == ROOT or ROOT in work.parents:
        raise ValueError("Keep --work outside the operator checkout")
    site = str(Path(args.site_env).expanduser().resolve()) if args.site_env else ""
    identity = {"harness_hash": digest(harness_files()), "gpu": args.gpu,
                "requirements_sha256": file_digest(ROOT / "revision_campaign/requirements.txt"),
                "config_template_sha256": file_digest(ROOT / "revision_campaign/config.example.json"),
                "site_env": site, "site_env_sha256": file_digest(site) if site else None,
                "torch_version": os.environ.get("FGC_TORCH_VERSION", "2.5.1"),
                "torch_index": os.environ.get("FGC_TORCH_INDEX", "https://download.pytorch.org/whl/cu124")}
    claim_work(work, identity)
    with open(work / ".bootstrap.lock", "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another startup command owns this work directory") from exc
        env = dict(os.environ, TORCH_CUDA_ARCH_LIST="9.0" if args.gpu == "H100" else "8.0",
                   FGC_CAMPAIGN_ROOT=str(ROOT), FGC_SITE_ENV=site, GGNN_SOURCE=str(work / "src/ggnn"))
        setup = Setup(work, env)
        for directory in ("src", "envs", "configs"):
            (work / directory).mkdir(exist_ok=True)
        for name, repository, pin in (("fgc", "AgarwalAarush/FastGraphCompute", FGC_PIN),
                                       ("ggnn", "cgtuebingen/ggnn", GGNN_PIN)):
            source = work / "src" / name
            if not source.exists():
                setup.run("clone-" + name, ["git", "clone", "https://github.com/" + repository + ".git", str(source)])
            if git_identity(source)["dirty"]:
                raise ValueError(f"Preserving modified source: {source}; use a NEW --work directory")
            if git_identity(source)["commit"] != pin:
                setup.run("pin-" + name, ["git", "-C", str(source), "checkout", "--detach", pin])
        recipe = str(ROOT / "revision_campaign/setup.sh")
        for name in ("core", "faiss", "cuvs", "ggnn"):
            prefix = work / "envs" / name
            command = ["bash", recipe, name, str(prefix)]
            if prefix.exists():
                command.append("--resume")
            setup.step("environment-" + name, command, prefix / "bin/python")
        cfg = pilot_config(work, args.gpu, site)
        core = cfg["python_executables"]["core"]
        env["FGC_CORE_PYTHON"] = core
        setup.step("build-fgc", ["bash", recipe, "build-fgc", cfg["library_source"]],
                   work / "src/fgc/fastgraphcompute/extensions/binned_knn_ops.so")
        config_path = work / "configs/synthetic-pilot.config.json"
        # Instrumentation needs configuration, while the immutable plan is saved after all builds.
        if config_path.exists() and read_json(config_path) != cfg:
            raise ValueError("Generated pilot config was edited; use a NEW --work directory")
        if not config_path.exists():
            atomic_json(config_path, cfg)
        diagnostic = work / "src/fgc-counters"
        if not diagnostic.exists():
            setup.run("prepare-counters", [core, "-m", "revision_campaign.cli", "prepare-instrumented-source",
                      "--config", str(config_path), "--destination", str(diagnostic)])
        diagnostic_marker = diagnostic / "fgc-diagnostic-build.json"
        if not diagnostic_marker.is_file() or read_json(diagnostic_marker)["base_commit"] != FGC_PIN:
            raise ValueError("Incomplete diagnostic checkout: preserve it and use a NEW --work directory")
        setup.step("build-counters", ["bash", recipe, "build-counters", str(diagnostic)],
                   diagnostic / "fastgraphcompute/extensions/binned_knn_ops.so")
        setup.step("cpu-tests", [core, "-m", "unittest", "discover", "-s", "revision_campaign/tests", "-v"],
                   work / "envs/core/bin/python")
        plan_path = work / "configs/synthetic-pilot.plan.json"
        _, plan = save_pilot(cfg, plan_path)
        print(f"Synthetic pilot: {len(plan['cases'])} cases, {len(plan['tasks'])} bundles. Plan: {plan_path}", flush=True)
        print("This covers the synthetic pilot. Real event/model production needs its own supplied inputs/plan.", flush=True)
        for backend in cfg["backends"]:
            key = "core" if backend.startswith("fgc") else "cuvs" if backend in ("cuvs_bf", "cagra_nnd") else backend
            setup.run("preflight-" + backend, [cfg["python_executables"][key], "-m", "revision_campaign.cli",
                      "preflight", "--config", str(config_path), "--backend", backend])
        print(f"Setup/preflight complete. Starting {args.minutes:g}-minute pilot work budget; leave five extra minutes for return/export.", flush=True)
        return subprocess.run(["bash", str(ROOT / "revision_campaign/run_all.sh"), "--plan", str(plan_path),
                               "--minutes", str(args.minutes), "--execute"], cwd=ROOT, env=env).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--gpu", choices=("A100", "H100"), default="H100")
    parser.add_argument("--full-gpu", action="store_true")
    parser.add_argument("--minutes", type=float, default=25)
    parser.add_argument("--site-env")
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    try:
        return start(args)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"STARTUP STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
