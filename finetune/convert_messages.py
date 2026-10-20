"""Convert messages-format jsonl -> Alpaca + split by domain.
Usage: py finetune/convert_messages.py finetune/linuxdocker/dataset.jsonl [--move]
Reads {"messages":[{"role":"user",...},{"role":"assistant",...}]} rows,
writes {"instruction","input","output"} into finetune/stacks/<stack>/dataset.jsonl
(domain auto-detected: linux/docker/web/data/python/identity/general).
--move deletes the source file after a successful convert.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from experts import detect

ROOT = Path(__file__).resolve().parent


def convert(src: Path):
    counts = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        msgs = r.get("messages") or []
        users = [m.get("content", "") for m in msgs if m.get("role") == "user"]
        assists = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
        q = (users[0] if users else "").strip()
        a = "\n".join(s.strip() for s in assists if s.strip()).strip()
        if not q or not a:
            continue
        dom = detect(q + " " + a)
        dest = ROOT / dom / "dataset.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"instruction": q, "input": "", "output": a},
                               ensure_ascii=False) + "\n")
        counts[dom] = counts.get(dom, 0) + 1
    return counts


def main():
    if len(sys.argv) < 2:
        raise SystemExit("guna: py finetune/convert_messages.py <fail.jsonl> [--move]")
    src = Path(sys.argv[1])
    if not src.exists():
        raise SystemExit(f"tiada: {src}")
    counts = convert(src)
    total = sum(counts.values())
    for dom, n in sorted(counts.items()):
        print(f"{dom}/dataset.jsonl: +{n} rows")
    print(f"TOTAL: {total} rows converted")
    if "--move" in sys.argv:
        src.unlink()
        print(f"deleted {src}")


if __name__ == "__main__":
    main()
