"""Merge ALL config stacks -> ONE datasets/all.jsonl (single-file generalist set).

Usage: python datasets/scripts/merge_one.py [--force]
Run from the repo root. Writes datasets/all.jsonl (git-ignored, generated).

Rules (same as merge_all.py, minus the eval split — the training script does
its own deterministic 1% holdout):
- dedupe on (instruction, output) across ALL stacks (a row trains once)
- datasets/fable_agent_ms.jsonl (terjemahan Fable agent) digabung jika wujud
- row order shuffled deterministically (seed 77) so no stack dominates
- aborts if datasets/all.jsonl exists unless --force is given
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import stack_files, read_rows, check_row, dump_row

DATASETS_DIR = Path(__file__).resolve().parent.parent
DEST = DATASETS_DIR / "all.jsonl"

force = "--force" in sys.argv
if DEST.exists() and not force:
    sys.exit(f"{DEST} sudah wujud — guna --force untuk tulis semula "
             "(fail sumber lain tidak disentuh).")

seen: set = set()
rows = []
dupes = 0
bad = 0
per_stack = {}
for stack, path in sorted(stack_files().items()):
    n = 0
    for r in read_rows(path):
        if not check_row(r):
            bad += 1
            continue
        q = r["instruction"].strip()
        a = r["output"].strip()
        if (q, a) in seen:
            dupes += 1
            continue
        seen.add((q, a))
        rows.append({"instruction": q, "input": (r.get("input") or "").strip(),
                     "output": a})
        n += 1
    per_stack[stack] = n

FABLE = DATASETS_DIR / "fable_agent_ms.jsonl"
if FABLE.exists():
    n, bad_f = 0, 0
    for line in FABLE.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad_f += 1
            continue
        if not check_row(r):
            bad_f += 1
            continue
        q = r["instruction"].strip()
        a = r["output"].strip()
        if (q, a) in seen:
            dupes += 1
            continue
        seen.add((q, a))
        rows.append({"instruction": q, "input": (r.get("input") or "").strip(),
                     "output": a})
        n += 1
    per_stack["fable (ms)"] = n
    if bad_f:
        print(f"fable: {bad_f} baris rosak dilangkau")
else:
    print("fable_agent_ms.jsonl: tiada — langkau (jalankan colab_mt.ipynb dahulu)")

random.Random(77).shuffle(rows)

with DEST.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(dump_row(r) + "\n")

print(f"rows={len(rows)} dupes={dupes} bad={bad} stacks={len(per_stack)}")
for s, n in sorted(per_stack.items()):
    print(f"  {s:<14} +{n}")
print(f"wrote {DEST}")
