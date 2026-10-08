import json
rows = [json.loads(l) for l in open("finetune/dataset.jsonl", encoding="utf-8")]
assert all("instruction" in r and "output" in r for r in rows)
n_id = sum(1 for r in rows if "Coder 77" in r["output"])
print(f"{len(rows)} rows OK, identity rows: {n_id}")
