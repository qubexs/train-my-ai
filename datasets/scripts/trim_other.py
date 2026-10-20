"""Trim excess rows from the 'other' bucket (github-code imports), largest first.

Removes rows whose key8 appears in the ledger with source codeparrot/github-code,
until the bucket total drops to the target. Trimmed rows are appended to
datasets/_quarantine.jsonl (quarantine-not-delete).

Usage: python datasets/scripts/trim_other.py [--dry]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATASETS_DIR, read_rows, dump_row
from import_hf import key8

BUCKETS = json.loads((DATASETS_DIR / "config" / "buckets.json")
                     .read_text(encoding="utf-8"))
OTHER = BUCKETS["buckets"]["other"]
TARGET = OTHER["target"]
STACKS = OTHER["stacks"]
LEDGER = DATASETS_DIR / "manifests" / "ledger.jsonl"
QUARANTINE = DATASETS_DIR / "_quarantine.jsonl"
dry = "--dry" in sys.argv


def count_rows(path):
    return sum(1 for _ in open(path, encoding="utf-8"))


def main():
    sizes = {}
    for s in STACKS:
        p = DATASETS_DIR / f"{s}.jsonl"
        if p.exists():
            sizes[s] = count_rows(p)
    total = sum(sizes.values())
    needed = total - TARGET
    print(f"other total={total} target={TARGET} remove={needed}")
    if needed <= 0:
        return

    ghc_keys = set()
    n_led = 0
    with LEDGER.open(encoding="utf-8") as f:
        for line in f:
            if '"codeparrot/github-code"' not in line:
                continue
            n_led += 1
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            ghc_keys.add(rec.get("k"))
    print(f"ghc ledger entries: {n_led} keys: {len(ghc_keys)}")

    order = sorted(sizes, key=lambda s: -sizes[s])
    removed_total = 0
    for s in order:
        if needed <= 0:
            break
        path = DATASETS_DIR / f"{s}.jsonl"
        keep, removed = [], []
        for line in open(path, encoding="utf-8"):
            t = line.strip()
            if not t:
                continue
            try:
                r = json.loads(t)
            except ValueError:
                keep.append(line)
                continue
            k = key8(r["instruction"].strip(), r["output"].strip())
            if k in ghc_keys and len(removed) < needed:
                removed.append(line)
            else:
                keep.append(line)
        if not removed:
            continue
        print(f"{s}: rows={sizes[s]} removing={len(removed)} "
              f"keeping={len(keep)}")
        if not dry:
            tmp = path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8", newline="\n") as f:
                f.writelines(keep)
            tmp.replace(path)
            with QUARANTINE.open("a", encoding="utf-8") as f:
                f.writelines(removed)
        removed_total += len(removed)
        needed -= len(removed)
    print(f"removed total={removed_total} remaining_needed={needed}")


if __name__ == "__main__":
    main()
