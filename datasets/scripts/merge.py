"""Merge datasets/<stack>.jsonl -> finetune/stacks/<stack>/dataset.jsonl (dedupe).
Usage: python datasets/scripts/merge.py [stack]
Run from the repo root. Overwrites finetune/stacks/<stack>/dataset.jsonl.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "app"))
from store import merge_stacks

only = sys.argv[1].lower() if len(sys.argv) > 1 else None
out = merge_stacks(root=ROOT, stack=only)
total = 0
for stack, n in sorted(out.items()):
    total += n
    print(f"{stack}: {n} rows -> finetune/stacks/{stack}/dataset.jsonl")
print(f"TOTAL: {total} rows")
