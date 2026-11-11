# evaluation/experiment_registry.py
"""
Local experiment registry — indexes every result JSON this project has
produced (results/**/*.json, sumo/training_logs/*.json) into one queryable
file, instead of leaving ~150 timestamped JSON files across a dozen
directories discoverable only by manually browsing the filesystem.

Design-review finding (category 9 "Training pipeline"): "no experiment
tracker (W&B/MLflow/TensorBoard) — results live only as timestamped JSON
files a human must locate and read." A full external experiment-tracking
service is disproportionate for this project's scale and would add a
network dependency/account requirement this project has never needed —
this registry is the right-sized answer: no new service, no account, just
a single generated index over what already exists, kept in the repo
itself and regenerable at any time.

Usage:
    python -m evaluation.experiment_registry              # rebuild + print summary table
    python -m evaluation.experiment_registry --json-only   # rebuild, no table

Output:
    results/experiment_registry.json — one entry per result file, with:
      file, category (inferred from directory), environment (from the
      file's own "environment" tag when present), mtime, size_bytes, and
      a best-effort one-line `summary` extracted from whichever of a
      dozen different known schemas the file happens to use (every
      script in this project saves a differently-shaped JSON — this is
      explicitly a tolerant, best-effort extractor, not a schema
      validator).
"""
import argparse
import json
import os
import time

BACKEND_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIRS = [
    os.path.join(BACKEND_ROOT, "results"),
    os.path.join(BACKEND_ROOT, "sumo", "training_logs"),
]
REGISTRY_PATH = os.path.join(BACKEND_ROOT, "results", "experiment_registry.json")


def _infer_category(path):
    rel = os.path.relpath(path, BACKEND_ROOT).replace("\\", "/")
    parts = rel.split("/")
    if parts[0] == "sumo":
        return "sumo_training_logs"
    if len(parts) >= 3:
        return parts[1]  # results/<category>/file.json
    return "results_root"  # results/file.json — no subdirectory, not a category name


def _summarize(data):
    """Best-effort one-line description, trying known field names from
    every result-producing script in this project in priority order.
    Returns None (not an exception) for shapes this doesn't recognize —
    the registry entry is still useful without a summary line."""
    if not isinstance(data, dict):
        return None

    parts = []

    if "environment" in data:
        parts.append(f"env={data['environment']}")
    elif "config" in data and isinstance(data["config"], dict) and "seeds" in data["config"]:
        pass  # environment sometimes only in nested config; skip rather than guess

    if "agent" in data:
        parts.append(f"agent={data['agent']}")
    if "seed" in data:
        parts.append(f"seed={data['seed']}")
    if "seeds" in data:
        parts.append(
            f"n_seeds={len(data['seeds']) if isinstance(data['seeds'], list) else data['seeds']}"
        )
    elif "config" in data and isinstance(data.get("config"), dict) and "seeds" in data["config"]:
        seeds = data["config"]["seeds"]
        parts.append(f"n_seeds={len(seeds) if isinstance(seeds, list) else seeds}")
    if "episodes" in data:
        parts.append(f"episodes={data['episodes']}")
    if "pattern" in data:
        parts.append(f"pattern={data['pattern']}")

    for key in (
        "final_avg_reward_last200",
        "final_emergency_accuracy_pct",
        "n_flips",
        "n_total_checks",
    ):
        if key in data:
            parts.append(f"{key}={data[key]}")

    if "summary" in data and isinstance(data["summary"], dict):
        s = data["summary"]
        for key in ("reward_mean_of_means", "emergency_accuracy_mean", "n_runs"):
            if key in s:
                parts.append(f"{key}={s[key]}")

    if not parts and isinstance(data, dict):
        # Last resort: just list the top-level keys so the entry is still
        # navigable even for a completely unrecognized schema.
        parts.append("keys=" + ",".join(list(data.keys())[:6]))

    return " | ".join(parts) if parts else None


def build_registry():
    entries = []
    for results_dir in RESULTS_DIRS:
        if not os.path.isdir(results_dir):
            continue
        for root, _dirs, files in os.walk(results_dir):
            for fname in files:
                if not fname.endswith(".json") or fname == "experiment_registry.json":
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    entries.append(
                        {
                            "file": os.path.relpath(fpath, BACKEND_ROOT).replace("\\", "/"),
                            "category": _infer_category(fpath),
                            "error": f"could not parse: {e}",
                        }
                    )
                    continue

                stat = os.stat(fpath)
                entries.append(
                    {
                        "file": os.path.relpath(fpath, BACKEND_ROOT).replace("\\", "/"),
                        "category": _infer_category(fpath),
                        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                        "size_bytes": stat.st_size,
                        "summary": _summarize(data),
                    }
                )

    entries.sort(key=lambda e: e.get("mtime", ""), reverse=True)

    registry = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_entries": len(entries),
        "categories": sorted({e["category"] for e in entries}),
        "entries": entries,
    }
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    return registry


def print_table(registry):
    print("=" * 100)
    print(
        f"EXPERIMENT REGISTRY — {registry['n_entries']} result files across "
        f"{len(registry['categories'])} categories"
    )
    print("=" * 100)
    for category in registry["categories"]:
        cat_entries = [e for e in registry["entries"] if e["category"] == category]
        print(f"\n{category} ({len(cat_entries)} files):")
        for e in cat_entries[:5]:
            summary = e.get("summary") or e.get("error") or ""
            print(f"  {e.get('mtime', '?'):19s}  {e['file']:60.60s}  {summary}")
        if len(cat_entries) > 5:
            print(f"  ... and {len(cat_entries) - 5} more (see {REGISTRY_PATH})")
    print(f"\nFull index: {REGISTRY_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the local experiment registry")
    parser.add_argument("--json-only", action="store_true", help="Skip printing the summary table")
    args = parser.parse_args()

    reg = build_registry()
    if not args.json_only:
        print_table(reg)
