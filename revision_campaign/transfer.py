"""Portable aggregate result exchange; never exports raw hits or model weights."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

from .common import ROOT, read_json, result_path, stamp, task_id
from .planning import verify_plan


FILES = {"manifest.json", "measurements.csv", "model.json", "diagnostics.json", "inputs.json", "COVERAGE.md"}
INTEGER_FIELDS = ("dim", "points", "events", "k", "bin_dims", "pca_subsample", "rep")
FLOAT_FIELDS = ("build_ms", "query_ms", "total_ms", "recall")


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def case_summary(case):
    keys = ("source", "dim", "points", "events", "cohort", "seed", "transform", "distribution", "correctness_only")
    return {key: case[key] for key in keys if key in case}


def export_results(plan, output):
    """Export aggregate tables, provenance hashes and coverage, including partial runs."""
    from .report import report
    verify_plan(plan)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to replace a result archive")
    report(plan)
    cfg = plan["config"]
    directory = Path(cfg["run_root"]) / "report"
    rows = []
    if (directory / "measurements.csv").exists():
        with open(directory / "measurements.csv", newline="") as handle:
            for row in csv.DictReader(handle):
                # Keep the result identity, not its original machine-specific filename.
                row["result_file"] = "results/benchmark/" + Path(row["result_file"]).name
                rows.append(row)
    csv_buffer = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts, hardware, software, inputs, models, diagnostics = {}, {}, {}, {}, [], []
    for task in plan["tasks"]:
        group = task["stage"] + "/" + task["backend"]
        counts.setdefault(group, {"ok": 0, "failed": 0, "missing": 0})
        path = result_path(cfg, task)
        if not path.exists():
            counts[group]["missing"] += 1
            continue
        value = read_json(path)
        counts[group][value["status"]] += 1
        if value["status"] != "ok":
            continue  # Full tracebacks stay in private storage on the GPU host.
        gpu = value.get("hardware", {})
        if gpu:
            hardware[gpu["hardware_hash"]] = {key: gpu[key] for key in
                ("hardware_hash", "gpu_name", "compute_capability", "total_memory_bytes", "driver_versions", "torch", "torch_cuda", "tf32") if key in gpu}
        env = value.get("software", {})
        software[task["backend"]] = {key: env[key] for key in ("module_version", "packages", "library_source") if key in env}
        if "case" in task:
            case_id = task_id(task["case"])
            entry = case_summary(task["case"])
            meta = value.get("input_metadata", value.get("metadata", {}))
            entry.update({key: meta[key] for key in ("input_hash", "shape", "sampled_pca_cumulative_variance", "pca_sample_count") if key in meta})
            provenance = meta.get("provenance", {})
            kept_provenance = {key: provenance[key] for key in
                ("offset_blocks", "feature_columns", "offsets_sha256", "schema_sha256", "owner_note_sha256",
                 "dimension_interpretation", "trained_checkpoint", "checkpoint_sha256", "adapter_sha256", "layer", "batch_index")
                if key in provenance}
            if kept_provenance:
                entry["provenance"] = kept_provenance
            inputs.setdefault(case_id, {}).update(entry)
        if task["stage"] == "model":
            for measure in value["measurements"]:
                kept = {key: measure[key] for key in
                        ("mode", "rep", "step_ms", "events_per_step", "optimizer_step_included", "backend",
                         "checkpoint_sha256", "adapter_sha256", "protocol", "profiler_pass_included_in_step_ms") if key in measure}
                kept["checkpoint"] = {key: measure["checkpoint"][key] for key in ("label", "training_stage")
                                      if key in measure.get("checkpoint", {})}
                models.append(kept)
        if task["stage"] in ("profile", "counters"):
            d = value["diagnostics"]
            kept = {key: d[key] for key in ("memory", "isolated_stages", "gradient_check", "queries", "candidates", "visited_bins",
                     "candidates_per_query", "visited_bins_per_query", "matches_release_distance_multiset", "timing_eligible") if key in d}
            kept["task"] = {"stage": task["stage"], "backend": task["backend"], "case": case_summary(task["case"]),
                            "k": task["k"], "bin_dims": task["bin_dims"]}
            kept["case_id"] = task_id(task["case"])
            diagnostics.append(kept)
    manifest = {"format": "fastgraph-aggregate-results", "version": 1, "created": stamp(),
                "source_plan_hash": plan["plan_hash"], "config_hash": plan["config_hash"],
                "harness_files": plan["harness_files"], "profile": plan["profile"], "scope": plan["scope"],
                "coverage": counts, "hardware": hardware, "software": software,
                "input_blockers": plan["blockers"],
                "publication_status": "author review required before committing or publishing these derived results",
                "excluded": ["raw coordinates", "hits", "checkpoint weights", "event IDs", "owner notes", "full run config",
                             "absolute source paths", "profiler traces", "raw result JSONs", "reference tensors"]}
    from .priority import coverage as priority_coverage
    from .coordination import comparisons
    manifest["priority_coverage"] = priority_coverage(plan)
    manifest["comparisons"] = comparisons(plan)
    summary = Path(cfg["run_root"]) / "execution-summary.json"
    if summary.exists():
        manifest["execution_summary"] = read_json(summary)
    coverage = ["# Returned result coverage", "", f"Source plan: `{plan['plan_hash']}`", "",
                f"Profile: {plan['profile']}; scope: {plan['scope']}", "",
                "| Stage/backend | Complete | Failed | Missing |", "|---|---:|---:|---:|"]
    coverage += [f"| {key} | {v['ok']} | {v['failed']} | {v['missing']} |" for key, v in sorted(counts.items())]
    coverage += ["", "## Scientific priority coverage", "", "| Phase | Complete | Failed | Missing |", "|---|---:|---:|---:|"]
    coverage += [f"| {key} | {v['ok']} | {v['failed']} | {v['missing']} |" for key, v in manifest["priority_coverage"].items()]
    if "execution_summary" in manifest:
        state = manifest["execution_summary"]
        coverage += ["", f"Strategy: {state['strategy']}; status: {state['status']}; peer-covered bundles: {state.get('peer_covered_bundles', 0)}.",
                     "Peer results count only as separate whole-case evidence; local missing rows stay missing.",
                     f"Deferred units: {len(set(state.get('deferred_units', [])))}; unresolved unit failures: {len(state.get('unit_failures', {}))}."]
    coverage += ["", "Partial runs remain partial. Review failed/missing stages before claiming coverage.",
                 "Large-input recall uses the recorded query sample; it is not a full-graph certificate.",
                 "H100 and historical A100 results must remain separate. Original audit artifacts stay with the operator.",
                 "Derived results still require author review before Git publication.", ""]
    payload = {"manifest.json": json_bytes(manifest), "measurements.csv": csv_buffer.getvalue().encode(),
               "model.json": json_bytes(models), "diagnostics.json": json_bytes(diagnostics),
               "inputs.json": json_bytes(inputs), "COVERAGE.md": "\n".join(coverage).encode()}
    checksums = "".join(f"{hashlib.sha256(content).hexdigest()}  {name}\n" for name, content in sorted(payload.items()))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing an existing returned run.
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, content in payload.items():
            archive.writestr(name, content)
        archive.writestr("SHA256SUMS", checksums)
    print(str(output))


def validate_payload(payload, checksum_text):
    checksums = {}
    for line in checksum_text.splitlines():
        checksum, name = line.split("  ", 1)
        if name in checksums or not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise ValueError("Invalid checksum manifest")
        checksums[name] = checksum
    if set(checksums) != FILES or set(payload) != FILES:
        raise ValueError("Incomplete checksum manifest")
    for name, data in payload.items():
        if hashlib.sha256(data).hexdigest() != checksums[name]:
            raise ValueError(f"Checksum mismatch: {name}")
    manifest = json.loads(payload["manifest.json"])
    if manifest.get("format") != "fastgraph-aggregate-results" or manifest.get("version") != 1:
        raise ValueError("Unsupported results format")


def read_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != FILES | {"SHA256SUMS"}:
            raise ValueError("Unexpected or duplicate archive members; use export-results")
        if any(i.file_size > 512 * 1024**2 for i in archive.infolist()):
            raise ValueError("Aggregate archive member exceeds the 512 MiB limit")
        payload = {name: archive.read(name) for name in FILES}
        validate_payload(payload, archive.read("SHA256SUMS").decode())
    return payload


def import_results(archive, name, root=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name):
        raise ValueError("Use a simple run name, e.g. h100-focused-2026-11-10")
    payload = read_bundle(archive)
    root = Path(root) if root is not None else ROOT / "revision_results"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / name
    if destination.exists():
        raise FileExistsError(f"Run already exists: {destination}; choose a new name")
    temporary = Path(tempfile.mkdtemp(prefix=".import-", dir=root))
    try:
        for filename, contents in payload.items():
            (temporary / filename).write_bytes(contents)
        checksums = "".join(f"{hashlib.sha256(data).hexdigest()}  {filename}\n" for filename, data in sorted(payload.items()))
        (temporary / "SHA256SUMS").write_text(checksums)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(str(destination))
    return destination


def plot_results(directory):
    """Replot aggregates on a different machine: no original paths/plan/CUDA needed."""
    from .report import make_plots
    directory = Path(directory)
    payload = {name: (directory / name).read_bytes() for name in FILES}
    validate_payload(payload, (directory / "SHA256SUMS").read_text())
    rows = []
    with open(directory / "measurements.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in INTEGER_FIELDS:
                row[key] = int(row[key])
            for key in FLOAT_FIELDS:
                row[key] = float(row[key])
            rows.append(row)
    plots = directory / "figures"
    plots.mkdir(exist_ok=True)
    make_plots(plots, rows, read_json(directory / "diagnostics.json"), read_json(directory / "model.json"))
    print(str(plots))
    return plots
