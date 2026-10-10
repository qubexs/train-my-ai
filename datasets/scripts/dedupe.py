"""Cross-file dedupe for datasets/*.jsonl (stdlib only).
Usage:
  python datasets/scripts/dedupe.py           # report only
  python datasets/scripts/dedupe.py --fix     # remove dupes (keep first file A-Z),
                                              # drop general.jsonl versions on conflict
                                              # (stack owner wins)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import read_rows

DATASETS_DIR = Path(__file__).resolve().parent.parent
FIX = "--fix" in sys.argv


def load_all():
    data = {}
    for f in sorted(DATASETS_DIR.glob("*.jsonl")):
        rows = []
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            q, a = (r.get("instruction") or "").strip(), (r.get("output") or "").strip()
            if q and a:
                rows.append({"instruction": q, "input": (r.get("input") or "").strip(),
                             "output": a})
        data[f.name] = rows
    return data


def save(data):
    for name, rows in sorted(data.items()):
        p = DATASETS_DIR / name
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    data = load_all()
    # 1. exact dupes across files: keep first file A-Z
    seen = {}
    removed = 0
    for name in sorted(data):
        keep = []
        for r in data[name]:
            key = (r["instruction"], r["output"])
            if key in seen:
                removed += 1
                print(f"dupe: {name} <- sudah ada di {seen[key]} :: {r['instruction'][:60]}")
                continue
            seen[key] = name
            keep.append(r)
        data[name] = keep
    print(f"exact cross-file dupes: {removed}")
    # 2. same question, different answers: stack owner wins over general.jsonl
    owners = defaultdict(set)
    for name, rows in data.items():
        for r in rows:
            owners[r["instruction"]].add(name)
    fixed = 0
    for q, files in sorted(owners.items()):
        if len(files) < 2:
            continue
        answers = {(r["output"]) for n in files for r in data[n] if r["instruction"] == q}
        if len(answers) < 2:
            continue
        if "general.jsonl" in files and len(files) > 1:
            before = len(data["general.jsonl"])
            data["general.jsonl"] = [r for r in data["general.jsonl"] if r["instruction"] != q]
            fixed += before - len(data["general.jsonl"])
            print(f"conflict -> general dibuang: {q[:60]} (kekal di {sorted(files - {'general.jsonl'})})")
        else:
            print(f"conflict MANUAL: {q[:60]} @ {sorted(files)}")
    print(f"general conflict rows: {fixed}")
    if FIX:
        save(data)
        print("written.")
    else:
        print("(dry-run; tambah --fix untuk tulis)")


if __name__ == "__main__":
    main()
