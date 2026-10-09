"""LLM distillation runner — generate stack rows at 100k scale (stdlib only).
Usage:
  python datasets/scripts/distill.py --stack docker --n 500 --base-url http://127.0.0.1:8080/v1
  python datasets/scripts/distill.py --stack sql --n 500 --base-url https://api.openai.com/v1 --api-key $KEY --model gpt-4o-mini
Flow: topics file -> prompt teacher -> parse jsonl -> dedupe -> validate -> append.
Topics: datasets/seeds/<stack>.txt (one topic per line; created on first run
from existing instructions if missing). Resume-safe: skips known instructions.
Needs a teacher model: local llama-server (free, slow) or hosted API (fast, paid).
Review output with validate.py + statistics.py before merging.
"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import read_rows

SCRIPTS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPTS_DIR.parent
SEEDS_DIR = DATASETS_DIR / "seeds"

SYSTEM = ("You write training rows for XCoder, a tiny CPU coding assistant. "
          "Output ONLY one JSON line per row: "
          "{\"instruction\": \"...\", \"input\": \"\", \"output\": \"...\"}. "
          "Answers: short, correct, Malay or English to match the question. "
          "Never mention Qwen/Alibaba. No markdown fences around the JSON lines.")

TOPIC_PROMPT = ("Write {k} diverse Alpaca rows about {stack} on the topic: {topic}. "
                "Vary question style (how/what/why/write/fix/compare). "
                "One JSON line per row, no other text.")


def call_teacher(base_url, api_key, model, system, user, timeout=300):
    body = json.dumps({"model": model,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}],
                       "temperature": 0.7, "max_tokens": 4000}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]


def parse_rows(blob):
    rows = []
    for line in blob.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        q, a = (r.get("instruction") or "").strip(), (r.get("output") or "").strip()
        if q and a:
            rows.append({"instruction": q, "input": (r.get("input") or "").strip(), "output": a})
    return rows


def ensure_seeds(stack):
    SEEDS_DIR.mkdir(exist_ok=True)
    f = SEEDS_DIR / f"{stack}.txt"
    if not f.exists():
        topics = sorted({r["instruction"] for r in
                         read_rows(DATASETS_DIR / f"{stack}.jsonl") if r.get("instruction")})
        f.write_text("\n".join(topics) + "\n", encoding="utf-8")
        print(f"seed topics written: {f} ({len(topics)} topics — edit/extend it)")
    return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--n", type=int, default=200, help="target new rows")
    ap.add_argument("--per-topic", type=int, default=10)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="xcoder")
    a = ap.parse_args()

    dest = DATASETS_DIR / f"{a.stack}.jsonl"
    known = {(r.get("instruction", ""), r.get("output", "")) for r in read_rows(dest)}
    topics = ensure_seeds(a.stack)
    if not topics:
        raise SystemExit(f"no topics — edit {SEEDS_DIR / (a.stack + '.txt')}")
    added, ti = 0, 0
    with dest.open("a", encoding="utf-8") as f:
        while added < a.n:
            topic = topics[ti % len(topics)]
            ti += 1
            try:
                blob = call_teacher(a.base_url, a.api_key, a.model, SYSTEM,
                                    TOPIC_PROMPT.format(k=a.per_topic, stack=a.stack, topic=topic))
            except Exception as e:
                print(f"teacher error (topic {ti}): {e} — stopping, progress kept")
                break
            fresh = 0
            for r in parse_rows(blob):
                if (r["instruction"], r["output"]) in known:
                    continue
                known.add((r["instruction"], r["output"]))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                added += 1
                fresh += 1
            print(f"topic {ti}/{len(topics)}: +{fresh} (total +{added}/{a.n})")
    print(f"DONE: +{added} rows -> {dest}; run validate.py {a.stack}")


if __name__ == "__main__":
    main()
