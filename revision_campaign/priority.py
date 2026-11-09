"""Predeclared scientific priorities; preserves every original measurement task."""
from __future__ import annotations

from collections import defaultdict

from .common import digest, result_path, read_json, task_id


POLICY = "physics-first-v1"
BACKENDS = ("core", "fgc_pca", "fgc_axis", "faiss", "cuvs_bf", "cagra_nnd", "ggnn")
STAGES = ("prepare", "reference", "benchmark", "profile", "counters", "model")
PHASES = ("checks", "model-core", "learned-core", "events-core", "controls-core",
          "diagnostics-core", "other-models", "breadth")
FRACTIONS = {"checks": .2, "model-core": .2, "learned-core": .2, "events-core": .2,
             "controls-core": .1, "diagnostics-core": .1, "other-models": .1}


def case_key(case):
    public = {k: v for k, v in case.items() if k not in ("coords", "row_splits", "snapshot_metadata")}
    if "snapshot_metadata" in case:
        meta = case["snapshot_metadata"]
        public["snapshot"] = {k: meta[k] for k in
            ("checkpoint_sha256", "adapter_sha256", "layer", "batch_index") if k in meta}
    return digest(public)


def task_key(plan, task, input_hash):
    return digest({"case": case_key(task["case"]), "input_hash": input_hash,
                   "task": {k: v for k, v in task.items() if k != "case"},
                   "library": plan["config"]["library_commit"], "ann": plan["config"]["ann"],
                   "protocol_sources": {key: plan["harness_files"].get(key) for key in
                                        ("backends.py", "reference.py", "data.py", "worker.py")},
                   "policy": POLICY})


def units(plan):
    """Small ordered units: finish a matched case before opening a broad sweep."""
    tasks, cfg = plan["tasks"], plan["config"]
    by_case = defaultdict(list)
    for task in tasks:
        if "case" in task:
            by_case[task_id(task["case"])].append(task)
    cases = {key: values[0]["case"] for key, values in by_case.items()}
    selected, result = set(), []
    guards = [t for t in tasks if t.get("case", {}).get("correctness_only")]
    guard_ids = [task_id(t) for t in guards if t["stage"] == "benchmark"]
    inputs = {key: [task_id(t) for t in values if t["stage"] in ("prepare", "reference")]
              for key, values in by_case.items()}

    def add(phase, values):
        groups = defaultdict(list)
        for t in values:
            if task_id(t) not in selected:
                groups[t["stage"], t["backend"]].append(t)
        for (stage, backend), group in sorted(groups.items(), key=lambda x:
                (STAGES.index(x[0][0]), BACKENDS.index(x[0][1]))):
            ids = [task_id(t) for t in group]
            needs = [] if phase == "checks" else list(guard_ids)
            for t in group:
                if "case" not in t:
                    continue
                case_id = task_id(t["case"])
                needs += [task_id(p) for p in by_case[case_id]
                          if p["stage"] == "prepare" and stage != "prepare"]
                if stage not in ("prepare", "reference"):
                    needs += inputs[case_id]
                if stage in ("profile", "counters"):
                    needs += [task_id(p) for p in central(case_id)
                              if p["stage"] == "benchmark" and p["backend"] == backend]
            needs = list(dict.fromkeys(n for n in needs if n not in ids))
            item = {"phase": phase, "stage": stage, "backend": backend, "tasks": ids,
                    "case_ids": list(dict.fromkeys(task_id(t["case"]) for t in group if "case" in t)),
                    "requires": needs, "critical": phase == "checks" or stage in ("prepare", "reference")
                    or (stage == "benchmark" and backend.startswith("fgc"))}
            item["id"] = digest(item)[:20]
            result.append(item)
            selected.update(ids)

    def central(case_id):
        values = by_case[case_id]
        chosen = [t for t in values if t["stage"] in ("prepare", "reference")]
        for backend in BACKENDS:
            candidates = [t for t in values if t["stage"] == "benchmark" and t["backend"] == backend]
            if candidates:
                chosen.append(min(candidates, key=lambda t: (abs(t["k"] - 40),
                    abs(t["bin_dims"] - min(3, t["case"]["dim"])), abs(t.get("pca_subsample", 50000) - 50000))))
        return chosen

    add("checks", guards)
    models = [t for t in tasks if t["stage"] == "model"]
    if models:
        preferred = models[0]["checkpoint"]
        add("model-core", [t for t in models if t["checkpoint"] == preferred])
    learned = [key for key, case in cases.items() if case["source"] == "learned"]
    checkpoint_order = {c["path"]: i for i, c in enumerate(cfg.get("model", {}).get("checkpoints", []))}
    layer_order = {layer: i for i, layer in enumerate(cfg.get("model", {}).get("capture_layers", []))}
    learned.sort(key=lambda key: (
        checkpoint_order.get(cases[key]["snapshot_metadata"].get("checkpoint", {}).get("path"), len(checkpoint_order)),
        layer_order.get(cases[key]["snapshot_metadata"].get("layer"), len(layer_order)),
        cases[key]["snapshot_metadata"].get("batch_index", 0)))
    # Honor explicit owner preferences even if forward-hook execution changes catalog order.
    core_learned = learned[:1]
    for key in core_learned:
        add("learned-core", central(key))
    detector = [key for key, case in cases.items() if case["source"] == "detector"]
    core_events = []
    if detector:
        dim = min({cases[k]["dim"] for k in detector}, key=lambda d: (abs(d - 8), d))
        batches = sorted({cases[k]["events"] for k in detector})[:2]
        cohort = min(cases[k]["cohort"] for k in detector)
        core_events = sorted([k for k in detector if cases[k]["dim"] == dim and
                              cases[k]["events"] in batches and cases[k]["cohort"] == cohort],
                             key=lambda k: (cases[k]["events"], cases[k]["transform"] != "raw"))
        for key in core_events:
            add("events-core", central(key))
    synthetic = [key for key, case in cases.items() if case["source"] == "synthetic"]
    if synthetic:
        dim = min({cases[k]["dim"] for k in synthetic}, key=lambda d: (abs(d - 8), d))
        size = min(cases[k]["points"] for k in synthetic)
        seed = cfg["grids"][plan["profile"]]["seeds"][0]
        for distribution in ("gaussian", "anisotropic_0.90", "anisotropic_0.99"):
            for key in synthetic:
                c = cases[key]
                if (c["dim"], c["points"], c["seed"], c["distribution"]) == (dim, size, seed, distribution):
                    add("controls-core", central(key))
    bounded_diagnostics = [key for key in synthetic if cases[key]["points"] ==
                           min(cases[k]["points"] for k in synthetic) and
                           any(t["stage"] == "profile" for t in by_case[key])]
    for key in core_learned + core_events[:2] + bounded_diagnostics:
        add("diagnostics-core", [t for t in by_case[key] if t["stage"] in ("profile", "counters")])
    add("other-models", models)
    # Remaining real inputs precede broad synthetic scaling. Within a case, keep all baselines together.
    for key in sorted(cases, key=lambda k: (0 if cases[k]["source"] == "learned" else
                    1 if cases[k]["source"] == "detector" else 2,
                    cases[k].get("points", cases[k].get("events", 0)), k)):
        add("breadth", by_case[key])
    if selected != {task_id(t) for t in tasks}:
        raise ValueError("Priority schedule must cover every measurement exactly once")
    return result


def unit_tasks(plan, unit_id):
    unit = next((u for u in units(plan) if u["id"] == unit_id), None)
    if unit is None:
        raise ValueError("Unknown work unit; regenerate the plan with this code")
    mapping = {task_id(t): t for t in plan["tasks"]}
    return unit, [mapping[key] for key in unit["tasks"]]


def coverage(plan):
    phases = {}
    mapping = {task_id(t): t for t in plan["tasks"]}
    for unit in units(plan):
        counts = phases.setdefault(unit["phase"], {"ok": 0, "failed": 0, "missing": 0})
        for key in unit["tasks"]:
            task = mapping[key]
            path = result_path(plan["config"], task)
            counts[read_json(path)["status"] if path.exists() else "missing"] += 1
    return phases
