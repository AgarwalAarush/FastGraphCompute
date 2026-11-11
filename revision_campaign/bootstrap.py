"""Owned, resumable setup and a bounded synthetic pilot; no scheduler submission."""
from __future__ import annotations

import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .common import ROOT, atomic_json, digest, file_digest, git_identity, harness_files, read_json
from .planning import make_plan, verify_plan, write_plan


FGC_PIN = "76a869cc8758a8a846b18e93934e2b6078d83279"
GGNN_PIN = "2d03b36362bc824e37099092eabb95172ec18b64"


class SlotExpired(RuntimeError):
    """A saved setup checkpoint, not a completed experiment or failed measurement."""


def time_left_seconds(value):
    """Slurm %L: MM:SS, HH:MM:SS or D-HH:MM:SS."""
    try:
        days, clock = value.split("-", 1) if "-" in value else ("0", value)
        fields = [int(part) for part in clock.split(":")]
        if len(fields) not in (2, 3) or min(fields) < 0 or int(days) < 0:
            return None
        seconds = 0
        for part in fields:
            seconds = seconds * 60 + part
        return int(days) * 86400 + seconds
    except ValueError:
        return None


def allocation_seconds():
    job = os.environ.get("SLURM_JOB_ID")
    if not job or not shutil.which("squeue"):
        return None
    try:
        result = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"],
                                capture_output=True, text=True, check=True, timeout=5)
        return time_left_seconds(result.stdout.strip().splitlines()[0])
    except (subprocess.SubprocessError, OSError, IndexError):
        return None


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def preserve_incomplete(path, work):
    """Retain an interrupted owned prefix/clone before recreating it."""
    if work.resolve() not in path.resolve().parents:
        raise ValueError(f"Refusing to move a path outside owned work: {path}")
    before = path.stat()
    directory = work / "interrupted-setup"
    directory.mkdir(exist_ok=True)
    backup = directory / f"{path.name}-{time.time_ns()}"
    path.rename(backup)
    after = backup.stat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or path.exists():
        raise RuntimeError("Interrupted setup backup could not be verified")
    print(f"Preserved interrupted setup: {backup}", flush=True)


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
    def __init__(self, work, environment, deadline=None):
        self.work, self.environment = work, environment
        self.deadline = deadline
        self.state, self.logs = work / "setup-state", work / "setup-logs"
        self.state.mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)

    def remaining(self):
        left = self.deadline - time.monotonic() if self.deadline is not None else float("inf")
        if left <= 0:
            raise SlotExpired("The shared setup/check/pilot work budget is used up")
        return left

    def run(self, name, args):
        left = self.remaining()
        log = self.logs / f"{name}-{time.time_ns()}.log"
        print(f"{name} ... log: {log}", flush=True)
        with open(log, "w") as output:
            process = subprocess.Popen(args, cwd=ROOT, env=self.environment,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=left if math.isfinite(left) else None)
            except subprocess.TimeoutExpired as exc:
                stop_group(process)
                raise SlotExpired(f"Saved setup during {name}; log: {log}") from exc
            except BaseException:
                stop_group(process)
                raise
        if code:
            print("\n".join(log.read_text(errors="replace").splitlines()[-20:]), file=sys.stderr)
            raise RuntimeError(f"{name} failed (exit {code}); fix the error and repeat the SAME command. Log: {log}")

    def step(self, name, args, expected):
        self.remaining()
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
    started = time.monotonic()
    setup_only = getattr(args, "setup_only", False)
    if not args.full_gpu and not setup_only:
        raise ValueError("Reserve a full GPU, then explicitly confirm it with --full-gpu")
    if not math.isfinite(args.minutes) or args.minutes <= 0:
        raise ValueError("--minutes must be finite and positive")
    if not math.isfinite(args.slot_minutes) or args.slot_minutes <= 5:
        raise ValueError("--slot-minutes must exceed the five-minute checkpoint/export reserve")
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
    slot_end = started + args.slot_minutes * 60
    allocation_left = allocation_seconds()
    if allocation_left is not None:
        slot_end = min(slot_end, time.monotonic() + allocation_left)
    deadline = slot_end - 300
    print(f"One shared {max(0, (slot_end - started) / 60):.1f}-minute slot for setup/checks/pilot, including five minutes reserved for checkpoint/export.", flush=True)
    with open(work / ".bootstrap.lock", "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another startup command owns this work directory") from exc
        env = dict(os.environ, TORCH_CUDA_ARCH_LIST="9.0" if args.gpu == "H100" else "8.0",
                   FGC_CAMPAIGN_ROOT=str(ROOT), FGC_SITE_ENV=site, GGNN_SOURCE=str(work / "src/ggnn"))
        if setup_only:
            env.setdefault("CONDA_OVERRIDE_CUDA", "12.4")  # Solver target only; actual GPU preflight remains mandatory.
        setup = Setup(work, env, deadline=deadline)
        for directory in ("src", "envs", "configs"):
            (work / directory).mkdir(exist_ok=True)
        for name, repository, pin in (("fgc", "AgarwalAarush/FastGraphCompute", FGC_PIN),
                                       ("ggnn", "cgtuebingen/ggnn", GGNN_PIN)):
            source = work / "src" / name
            if source.exists():
                try:
                    git_identity(source)
                except subprocess.CalledProcessError:
                    preserve_incomplete(source, work)
            if not source.exists():
                setup.run("clone-" + name, ["git", "clone", "https://github.com/" + repository + ".git", str(source)])
            if git_identity(source)["dirty"]:
                raise ValueError(f"Preserving modified source: {source}; use a NEW --work directory")
            if git_identity(source)["commit"] != pin:
                setup.run("pin-" + name, ["git", "-C", str(source), "checkout", "--detach", pin])
        recipe = str(ROOT / "revision_campaign/setup.sh")
        for name in ("core", "faiss", "cuvs", "ggnn"):
            prefix = work / "envs" / name
            if prefix.exists() and not (setup.state / ("environment-" + name + ".json")).exists() \
                    and not (prefix / "conda-meta/history").is_file():
                preserve_incomplete(prefix, work)
            command = ["bash", recipe, name, str(prefix)]
            if prefix.exists():
                command.append("--resume")
            setup.step("environment-" + name, command, prefix / "bin/python")
        cfg = pilot_config(work, args.gpu, site)
        core = cfg["python_executables"]["core"]
        env["FGC_CORE_PYTHON"] = core
        setup.step("build-fgc", ["bash", recipe, "build-fgc", cfg["library_source"]],
                   work / "src/fgc/fastgraphcompute/extensions/binned_knn_ops.so")
        config_path = work / "configs/setup.config.json"
        setup_cfg = {**cfg, "full_device_confirmed": False}
        # Instrumentation needs configuration, while the immutable plan is saved after all builds.
        if config_path.exists() and read_json(config_path) != setup_cfg:
            raise ValueError("Generated pilot config was edited; use a NEW --work directory")
        if not config_path.exists():
            atomic_json(config_path, setup_cfg)
        diagnostic = work / "src/fgc-counters"
        if diagnostic.exists() and not (diagnostic / "fgc-diagnostic-build.json").is_file():
            preserve_incomplete(diagnostic, work)
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
        if setup_only:
            atomic_json(work / "startup-status.json", {"status": "setup-ready", "gpu_measurements_run": False})
            print("SETUP READY. In a 30-minute full-GPU allocation, repeat with the SAME --work and --full-gpu, without --setup-only.", flush=True)
            return 0
        plan_path = work / "configs/synthetic-pilot.plan.json"
        config_path, plan = save_pilot(cfg, plan_path)
        print(f"Synthetic pilot: {len(plan['cases'])} cases, {len(plan['tasks'])} bundles. Plan: {plan_path}", flush=True)
        print("This covers the synthetic pilot. Real event/model production needs its own supplied inputs/plan.", flush=True)
        for backend in cfg["backends"]:
            key = "core" if backend.startswith("fgc") else "cuvs" if backend in ("cuvs_bf", "cagra_nnd") else backend
            setup.run("preflight-" + backend, [cfg["python_executables"][key], "-m", "revision_campaign.cli",
                      "preflight", "--config", str(config_path), "--backend", backend])
        minutes = min(args.minutes, setup.remaining() / 60)
        print(f"Setup/preflight complete. Pilot receives the remaining {minutes:.2f} work minutes of THIS slot.", flush=True)
        process = subprocess.Popen(["bash", str(ROOT / "revision_campaign/run_all.sh"), "--plan", str(plan_path),
                                    "--minutes", str(minutes), "--execute"], cwd=ROOT, env=env, start_new_session=True)
        try:
            return process.wait(timeout=max(1, slot_end - time.monotonic() - 10))
        except subprocess.TimeoutExpired as exc:
            stop_group(process)
            raise SlotExpired("Pilot reached the overall slot deadline; completed bundles remain saved") from exc
        except BaseException:
            stop_group(process)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--gpu", choices=("A100", "H100"), default="H100")
    parser.add_argument("--full-gpu", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--minutes", type=float, default=25)
    parser.add_argument("--slot-minutes", type=float, default=30)
    parser.add_argument("--site-env")
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    try:
        return start(args)
    except SlotExpired as exc:
        work = Path(args.work).expanduser().resolve()
        atomic_json(work / "startup-status.json", {"status": "checkpoint", "message": str(exc),
                    "slot_minutes": args.slot_minutes, "scientific_completion_claimed": False})
        print(f"SLOT CHECKPOINT: {exc}. Repeat the SAME command in the next allocation.", flush=True)
        archives = sorted((work / "runs/synthetic-pilot/returns").glob("*.zip"), key=lambda p: p.stat().st_mtime_ns)
        if archives:
            print(f"Latest available results ZIP: {archives[-1]}", flush=True)
        else:
            print("No results ZIP yet: this slot saved setup progress; the pilot has not returned measurements.", flush=True)
        return 75
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"STARTUP STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
