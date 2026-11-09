"""Dry-run-first portable launch; serialized, bounded checkpoint chains."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess

from .common import ROOT, atomic_json, git_identity, read_json, verify_artifacts
from .planning import verify_plan


def jobs(plan, plan_path, stages, scheduler):
    cfg = plan["config"]
    from .priority import units
    ordered = [u for u in units(plan) if u["stage"] in stages]
    site = cfg["slurm"]
    for unit in ordered:
        stage, backend = unit["stage"], unit["backend"]
        python = cfg["python_executables"].get(backend, cfg["python_executables"]["core"])
        if backend in ("cagra_nnd", "cuvs_bf"):
            python = cfg["python_executables"]["cuvs"]
        if backend.startswith("fgc"):
            python = cfg["python_executables"]["core"]
        chunks = cfg["stage_chunks"].get(stage, 1) if scheduler == "slurm" else 1
        for chunk in range(chunks):
            arguments = [python, "-m", "revision_campaign.cli", "run", "--plan", str(Path(plan_path).resolve()),
                         "--stage", stage, "--backend", backend, "--budget-seconds", str(site["worker_seconds"])]
            arguments += ["--unit-id", unit["id"]]
            if chunk == chunks - 1:
                arguments.append("--require-complete")
            if scheduler == "slurm":
                name = f"fgcr_{stage}_{backend}_{unit['id'][:8]}_{chunk}"
                args = ["sbatch", "--parsable", "--job-name", name, "--nodes", "1", "--ntasks", "1",
                        "--cpus-per-task", str(site["cpus"]), "--mem", site["memory"], "--time", site["walltime"],
                        "--gres", site["gres"], "--signal", "B:USR1@180",
                        "--output", str(Path(cfg["run_root"]) / "logs" / (name + "_%j.out")),
                        "--error", str(Path(cfg["run_root"]) / "logs" / (name + "_%j.err"))]
                for key in ("partition", "account", "qos", "constraint"):
                    if site.get(key):
                        args += ["--" + key, str(site[key])]
                args += [str(ROOT / "revision_campaign/job.sh"), *arguments]
            else:
                args = [str(ROOT / "revision_campaign/job.sh"), *arguments]
            yield args


def validate_execution(plan):
    """Shared execution gates for scheduled jobs and an existing allocation."""
    cfg = plan["config"]
    if plan["blockers"]:
        raise ValueError("Resolve plan blockers before execution: " + "; ".join(plan["blockers"]))
    if not cfg["full_device_confirmed"]:
        raise ValueError("A full-device reservation must be confirmed in the config")
    verify_artifacts(plan)
    if git_identity(cfg["library_source"]) != {"commit": cfg["library_commit"], "dirty": False}:
        raise ValueError("Pinned FGC checkout is missing, dirty or at a different commit")
    for key, executable in cfg["python_executables"].items():
        if not Path(executable).is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"Set an existing executable for python_executables.{key}")


def launch(plan, plan_path, scheduler, stages, execute=False):
    verify_plan(plan)
    if len(stages) != len(set(stages)):
        raise ValueError("Stage names must be unique")
    cfg = plan["config"]
    if execute:
        validate_execution(plan)
    elif plan["blockers"]:
        print("BLOCKED: " + "; ".join(plan["blockers"]))
    root = Path(cfg["run_root"])
    manifest = root / "submission.json"
    if execute and manifest.exists():
        raise FileExistsError("Submission already recorded; resume individual workers or review a new launch")
    previous = None
    submitted = []
    for original in jobs(plan, plan_path, stages, scheduler):
        args = list(original)
        if scheduler == "slurm" and previous:
            args[1:1] = ["--dependency", "afterok:" + previous]
        print(shlex.join(args), flush=True)
        if not execute:
            if scheduler == "slurm":
                previous = "PREVIOUS_JOB_ID"
            continue
        (root / "logs").mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, FGC_SITE_ENV=cfg.get("site_env", ""))
        if scheduler == "slurm":
            value = subprocess.check_output(args, text=True, env=env).strip()
            previous = value.split(";")[0]
            if not previous.isdigit():
                raise RuntimeError(f"Unexpected sbatch response: {value}")
            submitted.append({"job_id": previous, "args": args})
            # Persist after EACH submission: a later failure cannot erase earlier job identities.
            atomic_json(manifest, {"plan_hash": plan["plan_hash"], "jobs": submitted})
        else:
            subprocess.run(args, check=True, cwd=ROOT, env=env)
    if not execute:
        print("DRY RUN ONLY. Add --execute after reviewing the plan and input blockers.")
