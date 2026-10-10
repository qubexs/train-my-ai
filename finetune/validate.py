"""Validate finetune/stacks/<stack>/dataset.jsonl files. Usage:
  python finetune/validate.py            # all stacks
  python finetune/validate.py docker     # one stack
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check(p: Path):
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert all("instruction" in r and "output" in r for r in rows), f"bad row in {p}"
    n_id = sum(1 for r in rows if "XCoder" in r.get("output", "")
               or "Coder 77" in r.get("output", ""))
    return len(rows), n_id


def main():
    only = sys.argv[1].lower() if len(sys.argv) > 1 else None
    files = sorted((ROOT / "stacks").glob("*/dataset.jsonl"))
    if only:
        files = [f for f in files if only in f.parts[-2].lower()]
    if not files:
        raise SystemExit("no dataset files found under finetune/stacks/")
    total = 0
    for f in files:
        try:
            n, n_id = check(f)
        except (AssertionError, ValueError):
            print(f"{f.parent.name}: SKIP (bukan format Alpaca) — "
                  f"tukar: py finetune/convert_messages.py {f.parent.name}")
            continue
        total += n
        print(f"stacks/{f.parent.name}: {n} rows OK, identity rows: {n_id}")
    print(f"TOTAL: {total} rows")


if __name__ == "__main__":
    main()
