"""One allocation: resume every planned stage, then export completed evidence."""
from __future__ import annotations

import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .common import ROOT, atomic_json, read_json, result_path, task_id
from .launch import jobs, launch, validate_execution
from .planning import verify_plan
from .transfer import export_results
from .priority import units, PHASES, FRACTIONS
from .coordination import load_peers, peer_cases


STAGES = ["prepare", "reference", "benchmark", "profile", "counters", "model"]


def completed(cfg, task):
    path = result_path(cfg, task)
    return path.is_file() and read_json(path)["status"] == "ok"


def stop_worker(process):
    """Ask for an atomic checkpoint; kill the worker group if a bundle overruns."""
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGUSR1)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_worker(arguments, environment, log, deadline, stopped):
    with open(log, "w") as output:
        process = subprocess.Popen(arguments, cwd=ROOT, env=environment, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if stopped.is_set() or remaining <= 0:
                    stop_worker(process)
                    return 75
                try:
                    return process.wait(timeout=min(remaining, 1))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            stop_worker(process)


def session(plan, plan_path, minutes=25, execute=False, strategy="front", unit_minutes=30, peer_results=()):
    """No sbatch here. The caller supplies the reserved GPU and reviewed plan."""
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("minutes must be finite and positive")
    if not math.isfinite(unit_minutes) or unit_minutes <= 0 or strategy not in ("front", "complement"):
        raise ValueError("Use a positive unit time cap and front/complement strategy")
    verify_plan(plan)
    cfg = plan["config"]
    if not execute:
        launch(plan, plan_path, "local", STAGES, execute=False)
        print(f"Session work limit: {minutes:g} minutes TOTAL, then export a partial/complete results ZIP.")
        print("Run inside an existing allocation. Leave at least five extra minutes for checkpoint/export.")
        print(f"Strategy: {strategy}; slow-unit cap: {unit_minutes:g} minutes. Complement starts in the middle for <=40%, then falls back to the front.")
        return 0
    started = time.monotonic()
    validate_execution(plan)
    root = Path(cfg["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".session.lock", "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another session owns this run") from exc
        identity_path = root / "session-plan.json"
        identity = {"plan_hash": plan["plan_hash"], "config_hash": plan["config_hash"]}
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("This run has a different session plan; use its original plan or a new run_root")
        atomic_json(identity_path, identity)
        session_id = str(time.time_ns())
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        archive = root / "returns" / f"results-{session_id}.zip"
        record_path = root / "sessions" / f"session-{session_id}.json"
        record = {**identity, "minutes": minutes, "strategy": strategy, "workers": [], "status": "running",
                  "unit_failures": {}, "deferred_units": []}
        atomic_json(record_path, record)
        deadline = started + minutes * 60
        stopped = threading.Event()
        previous_signals = {}
        for name in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            previous_signals[name] = signal.signal(name, lambda _signum, _frame: stopped.set())
        code = 0
        try:
            environment = dict(os.environ, FGC_SITE_ENV=cfg.get("site_env", ""), FGC_CAMPAIGN_ROOT=str(ROOT))
            schedule = units(plan)
            mapping = {task_id(t): t for t in plan["tasks"]}
            commands = {args[args.index("--unit-id") + 1]: args
                        for args in jobs(plan, plan_path, STAGES, "local")}
            loaded_peers = load_peers(peer_results)
            covered, evidence = peer_cases(plan, loaded=loaded_peers)
            attempts, disabled = {}, set()
            failure_path = root / "unit-failures.json"
            failures = read_json(failure_path) if failure_path.exists() else {}
            fatal = False

            def is_done(unit):
                return all(completed(cfg, mapping[key]) for key in unit["tasks"])

            def elsewhere(unit):
                return bool(unit["case_ids"]) and all(key in covered for key in unit["case_ids"])

            def ready(unit):
                return all(completed(cfg, mapping[key]) for key in unit["requires"])

            def wave(selected, until, fair=False):
                nonlocal fatal, covered
                for index, unit in enumerate(selected):
                    if fatal or stopped.is_set() or time.monotonic() >= min(until, deadline):
                        break
                    if is_done(unit) or elsewhere(unit) or attempts.get(unit["id"], 0) >= 2:
                        continue
                    if unit["id"] in failures or unit["backend"] in disabled:
                        if unit["critical"]:
                            fatal = True
                        continue
                    if not ready(unit):
                        continue  # Its missing prerequisite is retried before breadth, never fabricated.
                    left = min(until, deadline) - time.monotonic()
                    pending = sum(not is_done(u) and not elsewhere(u) for u in selected[index:])
                    share = max(60, left / max(1, pending)) if fair else left
                    limit = min(left, share, unit_minutes * 60)
                    if limit < 1:
                        break
                    args = list(commands[unit["id"]])
                    args[args.index("--budget-seconds") + 1] = str(max(1, int(limit)))
                    attempts[unit["id"]] = attempts.get(unit["id"], 0) + 1
                    log = logs / f"session-{session_id}-{unit['id']}-{attempts[unit['id']]}.log"
                    print(f"{unit['phase']}: {unit['stage']}/{unit['backend']}; limit {limit/60:.1f} min; log: {log}", flush=True)
                    value = run_worker(args, environment, log, min(deadline, time.monotonic() + limit), stopped)
                    record["workers"].append({"phase": unit["phase"], "unit": unit["id"],
                        "stage": unit["stage"], "backend": unit["backend"], "exit_code": value, "log": str(log)})
                    if value == 75:
                        record["deferred_units"].append(unit["id"])
                    elif value:
                        failures[unit["id"]] = {"phase": unit["phase"], "backend": unit["backend"], "exit_code": value}
                        atomic_json(failure_path, failures)
                        if unit["critical"]:
                            fatal = True
                        elif unit["stage"] in ("benchmark", "profile", "counters") and unit["backend"] in ("faiss", "cuvs_bf", "cagra_nnd", "ggnn"):
                            disabled.add(unit["backend"])
                    if unit["stage"] == "prepare" and loaded_peers[0]:
                        covered, _ = peer_cases(plan, loaded=loaded_peers)
                    atomic_json(record_path, record)

            # Shared scientific/numerical gates always run first, including on the complement worker.
            gates = [u for u in schedule if u["phase"] == "checks"]
            wave(gates, min(deadline, time.monotonic() + min(900, minutes * 60 * .2)), fair=True)
            if strategy == "complement":
                breadth_cases = list(dict.fromkeys(k for u in schedule if u["phase"] == "breadth" for k in u["case_ids"]))
                tail = set(breadth_cases[len(breadth_cases)//2:])
                control_cases = {k for u in schedule if u["phase"] == "controls-core" for k in u["case_ids"]}
                middle = [u for u in schedule if u["phase"] in ("controls-core", "diagnostics-core") or
                          (u["phase"] == "breadth" and any(k in tail | control_cases for k in u["case_ids"]))]
                wave(middle, min(deadline, time.monotonic() + minutes * 60 * .4), fair=True)
            for phase in PHASES[1:-1]:
                wave([u for u in schedule if u["phase"] == phase],
                     min(deadline, time.monotonic() + minutes * 60 * FRACTIONS[phase]), fair=True)
            # A second pass gives unfinished essential work a chance before the expanded grid.
            wave([u for u in schedule if u["phase"] != "breadth"], deadline, fair=True)
            wave([u for u in schedule if u["phase"] == "breadth"], deadline)
            done = sum(completed(cfg, task) for task in plan["tasks"])
            peer_count = sum(not completed(cfg, t) and "case" in t and task_id(t["case"]) in covered for t in plan["tasks"])
            failed = any(result_path(cfg, task).is_file() and read_json(result_path(cfg, task))["status"] == "failed"
                         for task in plan["tasks"])
            outstanding_failures = {u["id"]: failures[u["id"]] for u in schedule if u["id"] in failures and not is_done(u)}
            if failed or fatal or outstanding_failures:
                record["status"] = "failed"
                code = 1
            elif done + peer_count != len(plan["tasks"]):
                record["status"] = "partial"
                code = 75
            else:
                record["status"] = "complete" if not peer_count else "covered-with-peer"
                code = 0
            record["completed_bundles"] = done
            record["total_bundles"] = len(plan["tasks"])
            record["peer_covered_bundles"] = peer_count
            record["peer_covered_cases"] = sorted(covered)
            record["peer_evidence"] = evidence
            record["unit_failures"] = outstanding_failures
            record["fatal"] = fatal
            unit_by_id = {u["id"]: u for u in schedule}
            record["deferred_units"] = list(dict.fromkeys(key for key in record["deferred_units"]
                if not is_done(unit_by_id[key]) and not elsewhere(unit_by_id[key])))
            return code
        finally:
            try:
                # Do not claim whole-plan completion after timeout, cancellation or error.
                if record["status"] == "running":
                    record["status"] = "interrupted"
                atomic_json(root / "execution-summary.json", {key: record[key] for key in
                    ("strategy", "status", "completed_bundles", "total_bundles", "peer_covered_bundles",
                     "peer_covered_cases", "peer_evidence", "unit_failures", "deferred_units", "fatal") if key in record})
                export_results(plan, archive)
                record["archive"] = str(archive)
                print(f"{record['status'].upper()}: send {archive}. Re-run the SAME command to resume; inspect failures first.", flush=True)
            finally:
                record["elapsed_seconds"] = time.monotonic() - started
                atomic_json(record_path, record)
                for name, handler in previous_signals.items():
                    signal.signal(name, handler)
