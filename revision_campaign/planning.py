"""Pure-stdlib planner: runnable even on a login node without torch/numpy."""
from __future__ import annotations

from itertools import product
from pathlib import Path

from .common import VERSION, artifact_stats, atomic_json, digest, harness_files, read_json, task_id


BACKENDS = ("fgc_pca", "fgc_axis", "faiss", "cuvs_bf", "cagra_nnd", "ggnn")


def blockers(cfg, scope):
    issues = []
    if scope in ("detector", "all"):
        data = cfg.get("detector", {})
        for key in ("features", "row_splits", "provenance_note", "feature_schema"):
            if not data.get(key) or not Path(data[key]).is_file():
                issues.append(f"detector.{key}: supply an existing file")
        if not data.get("event_layout_confirmed"):
            issues.append("detector.event_layout_confirmed: data-owner confirmation is required")
    if scope in ("learned", "all"):
        if not cfg.get("learned_manifest") or not Path(cfg["learned_manifest"]).is_file():
            issues.append("learned_manifest: capture real coordinates first")
        model = cfg.get("model", {})
        if not model.get("adapter") or not Path(model["adapter"]).is_file():
            issues.append("model.adapter: supply the real trained-model/data/loss adapter")
        if not model.get("trained_checkpoint_confirmed") or not model.get("checkpoints"):
            issues.append("model.checkpoints: trained checkpoint provenance is required for model throughput")
    return issues


def make_plan(cfg, profile="focused", scope="synthetic"):
    if profile not in ("pilot", "focused", "full") or scope not in ("synthetic", "detector", "learned", "all"):
        raise ValueError("Unknown profile/scope")
    grid = cfg["grids"][profile]
    cases = []
    if scope in ("synthetic", "all"):
        for dim, n, distribution, seed in product(grid["dims"], grid["points"],
                                                  grid["distributions"], grid["seeds"]):
            cases.append(dict(source="synthetic", dim=dim, points=n,
                              distribution=distribution, seed=seed, transform="raw"))
        # Small, fully checked floating-point stress inputs.
        for kind in ("duplicates", "near_ties", "large_offset", "extreme_scale", "bin_boundary"):
            cases.append(dict(source="adversarial", dim=8, points=4096,
                              distribution=kind, seed=42, transform="raw", correctness_only=True))
    if scope in ("detector", "all"):
        for dim, events, cohort, transform in product(grid["dims"], grid["event_batches"],
                                                     range(grid["cohorts"]), ("raw", "standardized")):
            cases.append(dict(source="detector", dim=dim, events=events, cohort=cohort,
                              seed=grid["seeds"][cohort % len(grid["seeds"])], transform=transform))
    if scope in ("learned", "all") and cfg.get("learned_manifest") and Path(cfg["learned_manifest"]).is_file():
        catalog = read_json(cfg["learned_manifest"])
        for entry in catalog["snapshots"]:
            cases.append(dict(source="learned", coords=entry["coords"], row_splits=entry["row_splits"],
                              snapshot_metadata=entry["metadata"], dim=entry["dim"], seed=42, transform="raw"))
    unique = {task_id(case): case for case in cases}
    cases = list(unique.values())
    ks = grid["neighbors"]
    tasks = []
    selected_backends = cfg.get("backends", list(BACKENDS))
    if not selected_backends or any(b not in BACKENDS for b in selected_backends):
        raise ValueError("Select supported backends")
    for case in cases:
        tasks.append(dict(stage="prepare", backend="core", case=case))
        tasks.append(dict(stage="reference", backend="core", case=case, max_k=max(ks),
                          query_count=grid["reference_queries"], full_limit=grid["full_reference_limit"]))
        for backend in selected_backends:
            ablation_cell = ((case["source"] == "synthetic" and case["points"] == min(grid["points"])
                              and case["seed"] == grid["seeds"][0])
                             or (case["source"] == "detector" and case["events"] == 1 and case["cohort"] == 0))
            bin_dims = grid["bin_dims"] if backend.startswith("fgc") and ablation_cell else [3]
            if case.get("correctness_only") and backend not in ("fgc_pca", "fgc_axis"):
                continue
            subsamples = grid.get("pca_subsamples", [50000]) if backend == "fgc_pca" and ablation_cell else [50000]
            for k, bins, sample in product(ks, bin_dims, subsamples):
                tasks.append(dict(stage="benchmark", backend=backend, case=case, k=k,
                                  bin_dims=min(bins, case["dim"]), repetitions=grid["repetitions"],
                                  pca_subsample=sample,
                                  query_count=grid["reference_queries"], full_limit=grid["full_reference_limit"],
                                  max_k=max(ks), warmups=grid["warmups"]))
        diagnostic_cell = (case["dim"] == 8 and
                           ((case["source"] == "synthetic" and case["seed"] == grid["seeds"][0]
                             and case["points"] in (min(grid["points"]), max(grid["points"]))
                             and case["distribution"] in ("gaussian", "anisotropic_0.99"))
                            or (case["source"] == "detector" and case["events"] == 1
                                and case["cohort"] == 0)
                            )) or (case["source"] == "learned" and
                                   case["snapshot_metadata"].get("batch_index", 0) == 0)
        if not case.get("correctness_only") and diagnostic_cell:
            for backend in selected_backends:
                tasks.append(dict(stage="profile", backend=backend, case=case, k=max(ks), bin_dims=3))
            for backend in ("fgc_pca", "fgc_axis"):
                if backend in selected_backends:
                    tasks.append(dict(stage="counters", backend=backend, case=case, k=max(ks), bin_dims=3))
    if scope in ("learned", "all"):
        for checkpoint in cfg.get("model", {}).get("checkpoints", []):
            for backend in cfg.get("model", {}).get("backends", ["fgc_pca", "fgc_axis"]):
                tasks.append(dict(stage="model", backend="core", model_backend=backend,
                                  checkpoint=checkpoint, warmups=grid["warmups"],
                                  steps=grid["model_steps"], repetitions=grid["repetitions"]))
    # Duplicate grid entries must not cause redundant computations.
    tasks = list({task_id(task): task for task in tasks}.values())
    result = dict(schema_version=VERSION, profile=profile, scope=scope, config=cfg,
                  config_hash=digest(cfg), artifact_stats=artifact_stats(cfg), harness_files=harness_files(),
                  blockers=blockers(cfg, scope), cases=cases, tasks=tasks)
    from .priority import POLICY
    result["execution_policy"] = POLICY
    result["plan_hash"] = digest(result)
    return result


def verify_plan(plan):
    original = dict(plan)
    expected = original.pop("plan_hash")
    if expected != digest(original) or plan["config_hash"] != digest(plan["config"]):
        raise ValueError("Plan modified after generation; generate/review a new plan")
    if plan["harness_files"] != harness_files():
        raise ValueError("Experiment scripts changed after review; generate/review a new plan")


def write_plan(plan, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Refusing to replace an existing review plan")
    atomic_json(path, plan)
    counts = {}
    for task in plan["tasks"]:
        key = task["stage"] + "/" + task["backend"]
        counts[key] = counts.get(key, 0) + 1
    lines = ["# Generated run plan", "", f"Plan hash: `{plan['plan_hash']}`", "",
             f"Profile: {plan['profile']}; scope: {plan['scope']}", "",
             f"Input cases: {len(plan['cases'])}; task bundles: {len(plan['tasks'])}", "",
             "Each benchmark bundle contains all configured repetitions and ANN query settings.", "",
             "## Input blockers", ""]
    lines += [f"- {issue}" for issue in plan["blockers"]] or ["None detected by the file-level planner; GPU preflight is still required."]
    lines += ["", "## Work inventory", "", "| Stage/backend | Bundles |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in sorted(counts.items())]
    from .priority import units
    schedule = units(plan)
    lines += ["", "## Priority queue (measurement settings unchanged)", "",
              "Preferred checkpoint and coordinate layer go FIRST in the owner config/catalog.",
              "Front strategy: checks, model, learned/event comparisons, bounded controls/diagnostics, breadth.",
              "Complement strategy: checks, middle/control work for at most 40% of the session, then front fallback.",
              "Soft per-phase shares and per-unit caps defer slow work; missed units remain incomplete.", "",
              "| Order | Phase | Stage/backend | Bundles | Unit |", "|---:|---|---|---:|---|"]
    lines += [f"| {i+1} | {u['phase']} | {u['stage']}/{u['backend']} | {len(u['tasks'])} | `{u['id']}` |"
              for i, u in enumerate(schedule)]
    lines += ["", "H100 timings are new measurements; historical A100 curves stay separate.",
              "A clean plan does not certify model provenance, data semantics, or GPU execution.",
              "Read RUN_PLAN.md and HANDOFF.md before executing. Nothing has been submitted by this command.", ""]
    path.with_suffix(".md").write_text("\n".join(lines))
