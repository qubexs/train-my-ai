"""Statistics over datasets/<stack>.jsonl. Usage:
  python datasets/scripts/statistics.py [--json]
"""
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _common import load_config, stack_files, read_rows, check_row

as_json = "--json" in sys.argv
cfg = load_config()
per_stack, per_domain = {}, {}
for name, path in sorted(stack_files(cfg).items()):
    rows = [r for r in read_rows(path) if check_row(r)]
    dom = cfg["stacks"][name]["domain"]
    li = sum(len(r["instruction"]) for r in rows)
    lo = sum(len(r["output"]) for r in rows)
    st = {"rows": len(rows),
          "avg_instruction_chars": round(li / len(rows), 1) if rows else 0,
          "avg_output_chars": round(lo / len(rows), 1) if rows else 0,
          "identity_rows": sum(1 for r in rows if "XCoder" in r.get("output", "")
                               or "Coder 77" in r.get("output", ""))}
    per_stack[name] = st
    d = per_domain.setdefault(dom, {"rows": 0})
    d["rows"] += st["rows"]
out = {"stacks": per_stack, "domains": per_domain,
       "total_rows": sum(s["rows"] for s in per_stack.values())}
if as_json:
    print(json.dumps(out, indent=1))
else:
    for name, st in per_stack.items():
        print(f"{name:12} rows={st['rows']:4}  avg_q={st['avg_instruction_chars']:6}  "
              f"avg_a={st['avg_output_chars']:7}  id={st['identity_rows']}")
    print("-" * 60)
    for dom, d in sorted(per_domain.items()):
        print(f"[{dom:8}] rows={d['rows']}")
    print(f"TOTAL rows={out['total_rows']}")
