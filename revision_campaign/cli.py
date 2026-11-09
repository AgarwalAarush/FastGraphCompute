"""python -m revision_campaign.cli --help (does not import torch to plan)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .common import config, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Generate review JSON/Markdown; no GPU use or submission")
    plan.add_argument("--config", required=True)
    plan.add_argument("--out", required=True)
    plan.add_argument("--profile", choices=("pilot", "focused", "full"), default="focused")
    plan.add_argument("--scope", choices=("synthetic", "detector", "learned", "all"), default="all")
    audit = sub.add_parser("audit-data", help="CPU-only bounded memmap/header metadata audit")
    audit.add_argument("--config", required=True)
    check = sub.add_parser("preflight", help="Perform actual CUDA import/kernel smoke checks")
    check.add_argument("--config", required=True)
    check.add_argument("--backend", required=True)
    capture = sub.add_parser("capture", help="Export real coordinates using the owner's model adapter")
    capture.add_argument("--config", required=True)
    instrument = sub.add_parser("prepare-instrumented-source", help="Clone/patch a NEW private diagnostic source tree")
    instrument.add_argument("--config", required=True)
    instrument.add_argument("--destination", required=True)
    run = sub.add_parser("run", help="Execute or resume one stage/backend, inside an allocation")
    run.add_argument("--plan", required=True)
    run.add_argument("--stage", required=True, choices=("prepare", "reference", "benchmark", "profile", "counters", "model"))
    run.add_argument("--backend", required=True)
    run.add_argument("--budget-seconds", type=int, default=12000)
    run.add_argument("--require-complete", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument("--unit-id")
    launch = sub.add_parser("launch", help="Print commands by default; --execute explicitly submits/runs")
    launch.add_argument("--plan", required=True)
    launch.add_argument("--scheduler", choices=("slurm", "local"), default="slurm")
    launch.add_argument("--stages", nargs="+", default=["prepare", "reference", "benchmark", "profile", "counters", "model"])
    launch.add_argument("--execute", action="store_true")
    session = sub.add_parser("session", help="Run/resume all stages inside one existing allocation, then export results")
    session.add_argument("--plan", required=True)
    session.add_argument("--minutes", type=float, default=25)
    session.add_argument("--execute", action="store_true")
    session.add_argument("--strategy", choices=("front", "complement"), default="front")
    session.add_argument("--unit-minutes", type=float, default=30)
    session.add_argument("--peer-results", nargs="*", default=[])
    report = sub.add_parser("report", help="Summarize new results and explicit missing coverage")
    report.add_argument("--plan", required=True)
    report.add_argument("--plots", action="store_true")
    export = sub.add_parser("export-results", help="Package portable aggregate results; exclude raw data/checkpoints")
    export.add_argument("--plan", required=True)
    export.add_argument("--out", required=True)
    receive = sub.add_parser("import-results", help="Verify and import a returned archive into revision_results/")
    receive.add_argument("--archive", required=True)
    receive.add_argument("--name", required=True)
    plot = sub.add_parser("plot-results", help="Replot returned aggregates without CUDA or the original machine paths")
    plot.add_argument("--directory", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        from .planning import make_plan, write_plan
        value = make_plan(config(args.config), args.profile, args.scope)
        write_plan(value, args.out)
        print(f"{args.out}: {len(value['tasks'])} bundles, {len(value['blockers'])} input blockers; nothing submitted")
    elif args.command == "audit-data":
        from .data import audit_detector
        import json
        print(json.dumps(audit_detector(config(args.config)), indent=2))
    elif args.command == "preflight":
        from .worker import preflight
        import json
        print(json.dumps(preflight(config(args.config), args.backend), indent=2))
    elif args.command == "capture":
        from .common import attach_source, adapter
        from .hardware import inspect_gpu
        from .model import capture as export
        cfg = config(args.config)
        attach_source(cfg)
        inspect_gpu(cfg)
        print(export(cfg))
    elif args.command == "prepare-instrumented-source":
        from .instrument import prepare_instrumented
        print(prepare_instrumented(config(args.config), args.destination))
    elif args.command == "run":
        from .worker import run
        return run(read_json(args.plan), args.stage, args.backend, args.budget_seconds,
                   args.require_complete, args.retry_failed, args.unit_id)
    elif args.command == "launch":
        from .launch import launch
        launch(read_json(args.plan), args.plan, args.scheduler, args.stages, args.execute)
    elif args.command == "session":
        from .session import session
        return session(read_json(args.plan), args.plan, args.minutes, args.execute,
                       args.strategy, args.unit_minutes, args.peer_results)
    elif args.command == "report":
        from .report import report
        report(read_json(args.plan), args.plots)
    elif args.command == "export-results":
        from .transfer import export_results
        export_results(read_json(args.plan), args.out)
    elif args.command == "import-results":
        from .transfer import import_results
        import_results(args.archive, args.name)
    elif args.command == "plot-results":
        from .transfer import plot_results
        plot_results(args.directory)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
