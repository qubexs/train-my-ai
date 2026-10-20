"""Merge ALL stacks -> ONE generalist corpus for xcoder-best (0.5B generalist).

Usage: python datasets/scripts/merge_all.py [--eval 0.05]
Run from the repo root. Writes (git-ignored, like other merge outputs):
  finetune/stacks/all/dataset.jsonl   (train)
  finetune/stacks/all/eval.jsonl      (deterministic held-out, md5 split)

Rules (same as merge_stacks, globalized):
- dedupe on (instruction, output) across ALL stacks (a row trains once)
- md5 hash split is deterministic: same row always lands on the same side,
  so eval never leaks into train, stable across re-runs
- row order shuffled deterministically (seed 77) so no stack dominates
  the tail of an epoch
"""
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import stack_files, read_rows, check_row, dump_row

DATASETS_DIR = Path(__file__).resolve().parent.parent
ROOT = DATASETS_DIR.parent

eval_frac = 0.05
for a in sys.argv[1:]:
    if a.startswith("--eval="):
        eval_frac = float(a.split("=", 1)[1])
    elif a.replace(".", "", 1).isdigit() and 0 < float(a) < 1:
        eval_frac = float(a)

ethresh = int(eval_frac * 1000)
seen: set = set()
train, evrows = [], []
per_stack = {}
for stack, path in sorted(stack_files().items()):
    n = 0
    for r in read_rows(path):
        if not check_row(r):
            continue
        q = r["instruction"].strip()
        a = r["output"].strip()
        if (q, a) in seen:
            continue
        seen.add((q, a))
        row = {"instruction": q, "input": (r.get("input") or "").strip(),
               "output": a}
        if ethresh > 0 and int(hashlib.md5(
                f"{q}\n{a}".encode("utf-8")).hexdigest(), 16) % 1000 < ethresh:
            evrows.append(row)
        else:
            train.append(row)
            n += 1
    per_stack[stack] = n

rng = random.Random(77)
rng.shuffle(train)
rng2 = random.Random(78)
rng2.shuffle(evrows)

dest = ROOT / "finetune" / "stacks" / "all"
dest.mkdir(parents=True, exist_ok=True)
with (dest / "dataset.jsonl").open("w", encoding="utf-8") as f:
    for r in train:
        f.write(dump_row(r) + "\n")
with (dest / "eval.jsonl").open("w", encoding="utf-8") as f:
    for r in evrows:
        f.write(dump_row(r) + "\n")

ident = sum(1 for r in train
            if "xcoder" in (r["instruction"] + " " + r["output"]).lower())
print(f"train={len(train)} eval={len(evrows)} unique={len(seen)} "
      f"identity_rows(train)={ident}")
for s, n in sorted(per_stack.items()):
    print(f"  {s:<12} +{n}")
print(f"wrote {dest / 'dataset.jsonl'} + {dest / 'eval.jsonl'}")
