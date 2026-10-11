"""Merge datasets/<stack>.jsonl -> finetune/stacks/<stack>/dataset.jsonl (dedupe).
Usage: python datasets/scripts/merge.py [stack] [--eval 0.05]
Run from the repo root. Overwrites finetune/stacks/<stack>/dataset.jsonl.
--eval F writes deterministic held-out finetune/stacks/<stack>/eval.jsonl (md5 split).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "app"))
from store import merge_stacks

only, eval_frac = None, 0.0
for a in sys.argv[1:]:
    if a == "--eval":
        continue
    elif a.startswith("--eval="):
        eval_frac = float(a.split("=", 1)[1])
    elif a.replace(".", "", 1).isdigit() and 0 < float(a) < 1:
        eval_frac = float(a)
    elif only is None:
        only = a.lower()
out, evout = merge_stacks(root=ROOT, stack=only, eval_frac=eval_frac)
total = 0
for stack, n in sorted(out.items()):
    total += n
    ev = f" (+{evout.get(stack, 0)} eval)" if stack in evout else ""
    print(f"{stack}: {n} rows -> finetune/stacks/{stack}/dataset.jsonl{ev}")
print(f"TOTAL: {total} rows")
