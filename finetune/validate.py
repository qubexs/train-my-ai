"""Validate finetune/<domain>/dataset.jsonl files. Usage:
  python finetune/validate.py            # all domains
  python finetune/validate.py docker     # one domain
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
    # new layout finetune/<domain>/dataset.jsonl (+ legacy flat finetune/dataset_*.jsonl)
    files = sorted(ROOT.glob("*/dataset.jsonl")) + sorted(ROOT.glob("dataset_*.jsonl"))
    if only:
        files = [f for f in files if only in f.parts[-2].lower() or only in f.stem.lower()]
    if not files:
        raise SystemExit("no dataset files found under finetune/")
    total = 0
    for f in files:
        try:
            n, n_id = check(f)
        except (AssertionError, ValueError) as e:
            label = f.parent.name + "/" + f.name if f.parent != ROOT else f.name
            print(f"{label}: SKIP (bukan format Alpaca: instruction/output) — "
                  f"tukar dahulu: py finetune/convert_messages.py {label}")
            continue
        total += n
        label = f.parent.name + "/" + f.name if f.parent != ROOT else f.name
        print(f"{label}: {n} rows OK, identity rows: {n_id}")
    print(f"TOTAL: {total} rows")


if __name__ == "__main__":
    main()
