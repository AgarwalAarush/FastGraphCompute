"""Coordinate whole comparisons; never substitute another GPU's individual rows."""
from collections import defaultdict
from pathlib import Path

from .common import case_dir, file_digest, read_json, result_path, task_id
from .priority import case_key, task_key


def comparisons(plan):
    grouped = defaultdict(list)
    for task in plan["tasks"]:
        if "case" in task and task["stage"] in ("benchmark", "profile", "counters"):
            grouped[task_id(task["case"])].append(task)
    entries = []
    for tasks in grouped.values():
        case = tasks[0]["case"]
        meta = case_dir(plan["config"], case) / "metadata.json"
        if not meta.is_file():
            continue
        input_hash = read_json(meta)["input_hash"]
        done = [t for t in tasks if result_path(plan["config"], t).is_file() and
                read_json(result_path(plan["config"], t))["status"] == "ok"]
        entries.append({"case_key": case_key(case), "input_hash": input_hash,
                        "complete": len(done) == len(tasks), "expected_bundles": len(tasks),
                        "completed_task_keys": [task_key(plan, t, input_hash) for t in done]})
    return entries


def load_peers(archives):
    """Skip only fully covered cases from ONE archive, after local input-hash verification."""
    from .transfer import read_bundle
    import json
    entries = []
    evidence = []
    for path in archives:
        manifest = json.loads(read_bundle(path)["manifest.json"])
        entries.extend(e for e in manifest.get("comparisons", []) if e["complete"])
        evidence.append({"archive_sha256": file_digest(path), "source_plan_hash": manifest["source_plan_hash"],
                         "hardware": manifest.get("hardware", {})})
    return entries, evidence


def peer_cases(plan, archives=(), loaded=None):
    entries, evidence = load_peers(archives) if loaded is None else loaded
    covered = []
    grouped = defaultdict(list)
    for task in plan["tasks"]:
        if "case" in task and task["stage"] in ("benchmark", "profile", "counters"):
            grouped[task_id(task["case"])].append(task)
    for key, tasks in grouped.items():
        case = tasks[0]["case"]
        if case.get("correctness_only"):
            continue  # Always validate this machine's numerical gates.
        meta = case_dir(plan["config"], case) / "metadata.json"
        if not meta.is_file():
            continue
        input_hash = read_json(meta)["input_hash"]
        expected = {task_key(plan, t, input_hash) for t in tasks}
        if any(e["case_key"] == case_key(case) and e["input_hash"] == input_hash and
               expected.issubset(e["completed_task_keys"]) for e in entries):
            covered.append(key)
    return set(covered), evidence
