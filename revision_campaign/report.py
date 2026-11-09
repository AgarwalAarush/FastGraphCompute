"""Tables, plots and missing-evidence report; never rewrites historical figures."""
from __future__ import annotations

import csv
from pathlib import Path
import statistics

from .common import atomic_json, read_json, result_path, task_id
from .planning import verify_plan


def report(plan, plots=False):
    verify_plan(plan)
    cfg = plan["config"]
    directory = Path(cfg["run_root"]) / "report"
    directory.mkdir(parents=True, exist_ok=True)
    counts, missing, rows, failures, model_rows, diagnostic_rows = {}, [], [], [], [], []
    for task in plan["tasks"]:
        path = result_path(cfg, task)
        group = task["stage"] + "/" + task["backend"]
        counts.setdefault(group, {"ok": 0, "failed": 0, "missing": 0})
        if not path.exists():
            counts[group]["missing"] += 1
            missing.append({"id": task_id(task), "task": task})
            continue
        value = read_json(path)
        counts[group][value["status"]] += 1
        if value["status"] != "ok":
            failures.append({"path": str(path), "error": value.get("error")})
            continue
        if task["stage"] == "benchmark":
            for measure in value["measurements"]:
                rows.append({"case_id": task_id(task["case"]), "source": task["case"]["source"],
                             "distribution": task["case"].get("distribution", ""),
                             "transform": task["case"]["transform"], "dim": task["case"]["dim"],
                             "points": value["input_metadata"]["shape"][0],
                             "events": len(value["input_metadata"]["segment_lengths"]),
                             "backend": task["backend"], "k": task["k"], "bin_dims": task["bin_dims"],
                             "pca_subsample": task.get("pca_subsample", 50000),
                             "rep": measure["rep"], "quality": str(sorted(measure["quality"].items())),
                             "build_ms": measure["build_ms"], "query_ms": measure["query_ms"],
                             "total_ms": measure["total_ms"],
                             "recall": measure["recall"]["distance_recall_rtol_1e_4_atol_1e_10"],
                             "reference_coverage": measure["recall"]["reference_coverage"],
                             "check_pass": measure["recall"]["pass"],
                             "gpu_name": value["hardware"]["gpu_name"],
                             "hardware_hash": value["hardware"]["hardware_hash"],
                             "input_hash": value["input_metadata"]["input_hash"], "result_file": str(path)})
        elif task["stage"] == "model":
            model_rows.extend(value["measurements"])
        elif task["stage"] in ("profile", "counters"):
            diagnostic_rows.append({"task": task, **value["diagnostics"]})
    with open(directory / "measurements.csv", "w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(directory / "missing.json", missing)
    atomic_json(directory / "failures.json", failures)
    atomic_json(directory / "model.json", model_rows)
    atomic_json(directory / "diagnostics.json", diagnostic_rows)
    lines = ["# Campaign evidence coverage", "", f"Plan: `{plan['plan_hash']}`", "",
             "| Stage/backend | Complete | Failed | Missing |", "|---|---:|---:|---:|"]
    lines += [f"| {key} | {v['ok']} | {v['failed']} | {v['missing']} |" for key, v in sorted(counts.items())]
    from .priority import coverage
    lines += ["", "## Scientific priority coverage", "", "| Phase | Complete | Failed | Missing |", "|---|---:|---:|---:|"]
    lines += [f"| {key} | {v['ok']} | {v['failed']} | {v['missing']} |" for key, v in coverage(plan).items()]
    summary = Path(cfg["run_root"]) / "execution-summary.json"
    if summary.exists():
        state = read_json(summary)
        lines += ["", f"Strategy: {state['strategy']}; status: {state['status']}; peer-covered bundles: {state.get('peer_covered_bundles', 0)}.",
                  "Peer-covered cases have separate GPU evidence. They remain missing in LOCAL counts; no cross-GPU rows are combined.",
                  f"Deferred work units: {len(set(state.get('deferred_units', [])))}; unresolved unit failures: {len(state.get('unit_failures', {}))}."]
    lines += ["", "## Scope and limitations", "",
              f"Requested scope: {plan['scope']}. A synthetic-only plan cannot satisfy event or learned-model requests.",
              "Real event labels require the owner's offset confirmation; structural totals are insufficient.",
              "FP64 references cover the documented queries; sampled queries are not a full-graph certificate.",
              "NVML peaks are device-wide samples, not an analytical bound on instantaneous allocations.",
              "Diagnostic builds and profilers are excluded from clean timing curves.",
              "Old A100 measurements remain unchanged; no mixed-hardware speedup is computed.",
              "Reviewer 18 and the original numbered reviews still require an author audit.", ""]
    if plan["blockers"]:
        lines += ["## Input blockers", "", *["- " + b for b in plan["blockers"]], ""]
    if not missing and not failures and not plan["blockers"]:
        lines += ["All bundles in THIS plan completed. Review scope above before claiming reviewer coverage.", ""]
    (directory / "COVERAGE.md").write_text("\n".join(lines))
    if plots and rows:
        make_plots(directory, rows, diagnostic_rows, model_rows)
    print(str(directory / "COVERAGE.md"))


def make_plots(directory, rows, diagnostics, model_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Preserve input identity, K and hardware in each matched curve. No cross-grid averages.
    groups = {}
    for row in rows:
        key = (row["case_id"], row["k"], row["hardware_hash"])
        groups.setdefault(key, []).append(row)
    for (case_id, k, hardware), group in groups.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        settings = {}
        for row in group:
            settings.setdefault((row["backend"], row["quality"], row["bin_dims"], row["pca_subsample"]), []).append(row)
        for (backend, quality, bins, sample), values in sorted(settings.items()):
            times = [r["total_ms"] for r in values]
            quality_values = [r["recall"] for r in values]
            ax.scatter(statistics.mean(quality_values), statistics.median(times), label=f"{backend}, bins={bins}, PCA n={sample}, {quality}")
        ax.set(xlabel="Distance-based recall (documented query sample)", ylabel="Build + all-query search (ms)", yscale="log",
               title=f"{group[0]['gpu_name']}: case {case_id}, K={k}")
        ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1, 1))
        fig.savefig(directory / f"recall-time-{case_id}-k{k}-{hardware[:8]}.pdf", bbox_inches="tight")
        plt.close(fig)
    if diagnostics:
        values = [d for d in diagnostics if "memory" in d]
        if values:
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = [d["task"]["backend"] + "/" + d.get("case_id", task_id(d["task"]["case"]))[:8] for d in values]
            ax.bar(range(len(values)), [d["memory"]["peak_delta_bytes"] / 1024**3 for d in values])
            ax.set_xticks(range(len(values)), labels, rotation=90, fontsize=6)
            ax.set_ylabel("Sampled NVML peak delta (GiB)")
            fig.savefig(directory / "memory.pdf", bbox_inches="tight")
            plt.close(fig)
    if model_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(model_rows)), [statistics.median(r["step_ms"]) for r in model_rows])
        ax.set_xticks(range(len(model_rows)), [r["backend"] + "/" + r["mode"] for r in model_rows], rotation=90, fontsize=6)
        ax.set_ylabel("Complete model step (ms)")
        fig.savefig(directory / "model-steps.pdf", bbox_inches="tight")
        plt.close(fig)
