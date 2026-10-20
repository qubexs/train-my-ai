"""Bucket coverage report for the 10M plan (datasets/config/buckets.json).
Usage:
  python datasets/scripts/buckets.py [--json] [--bucket NAME]
Prints target vs actual rows per plan bucket, remaining and coverage %.
Stdlib only; read-only (never writes).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATASETS_DIR, load_config, stack_files, read_rows, check_row

BUCKETS_FILE = DATASETS_DIR / "config" / "buckets.json"


def load_buckets():
    return json.loads(BUCKETS_FILE.read_text(encoding="utf-8"))


def validate_mapping(cfg, plan):
    """Every config stack in exactly one bucket -> (missing, duplicated)."""
    stacks = set(cfg["stacks"])
    seen = [s for b in plan["buckets"].values() for s in b["stacks"]]
    missing = sorted(stacks - set(seen))
    dupes = sorted({s for s in seen if seen.count(s) > 1})
    unknown = sorted(set(seen) - stacks)
    return missing, dupes, unknown


def main():
    as_json = "--json" in sys.argv
    only = ""
    if "--bucket" in sys.argv:
        i = sys.argv.index("--bucket")
        if i + 1 < len(sys.argv):
            only = sys.argv[i + 1]

    cfg = load_config()
    plan = load_buckets()
    files = stack_files(cfg)

    per_stack = {}
    for name, path in files.items():
        per_stack[name] = sum(1 for r in read_rows(path) if check_row(r))

    missing, dupes, unknown = validate_mapping(cfg, plan)
    if missing or dupes or unknown:
        print(f"WARN: stacks not in any bucket: {missing or '-'}")
        print(f"WARN: stacks in multiple buckets: {dupes or '-'}")
        print(f"WARN: unknown stacks in buckets.json: {unknown or '-'}")

    out, total_rows, total_target = {}, 0, 0
    for name, b in plan["buckets"].items():
        if only and name != only:
            continue
        rows = sum(per_stack.get(s, 0) for s in b["stacks"])
        target = b["target"]
        total_rows += rows
        total_target += target
        out[name] = {"label": b["label"], "target": target, "rows": rows,
                     "remaining": max(0, target - rows),
                     "pct": round(100.0 * rows / target, 2) if target else 0.0,
                     "stacks": b["stacks"]}

    if as_json:
        print(json.dumps({"buckets": out, "total_rows": total_rows,
                          "total_target": total_target}, indent=1))
        return

    print(f"{'bucket':14} {'rows':>9} {'target':>9} {'remaining':>9}  {'cov%':>6}")
    print("-" * 55)
    for name, d in sorted(out.items(), key=lambda kv: -kv[1]["remaining"]):
        print(f"{name:14} {d['rows']:>9} {d['target']:>9} {d['remaining']:>9}  {d['pct']:>6.2f}")
    print("-" * 55)
    pct = round(100.0 * total_rows / total_target, 2) if total_target else 0.0
    print(f"{'TOTAL':14} {total_rows:>9} {total_target:>9} "
          f"{max(0, total_target - total_rows):>9}  {pct:>6.2f}")


if __name__ == "__main__":
    main()
