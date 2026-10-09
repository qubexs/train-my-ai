"""Validate datasets/<stack>.jsonl files. Usage:
  python datasets/scripts/validate.py [stack]
Exit code 1 on any bad row.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import load_config, stack_files, read_rows, check_row

only = sys.argv[1].lower() if len(sys.argv) > 1 else None
cfg = load_config()
total, bad = 0, 0
for name, path in sorted(stack_files(cfg).items()):
    if only and only not in (name, cfg["stacks"][name]["domain"]):
        continue
    rows = read_rows(path)
    ok = [r for r in rows if check_row(r)]
    bad_n = len(rows) - len(ok)
    bad += bad_n
    n_id = sum(1 for r in ok if "XCoder" in r.get("output", "")
               or "Coder 77" in r.get("output", ""))
    total += len(ok)
    flag = f"  <-- {bad_n} BAD ROWS" if bad_n else ""
    print(f"{name}.jsonl: {len(ok)} rows OK, identity: {n_id}{flag}")
print(f"TOTAL: {total} rows")
sys.exit(1 if bad else 0)
