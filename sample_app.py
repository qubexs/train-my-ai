"""XCoder sample app (Python stdlib only, CPU 0.5B).
Usage:
  python sample_app.py --lang ms --ask "Apakah fungsi tambah?"
  python sample_app.py --lang en --code py --task "write tambah(a,b) with example"
"""
import argparse
import json
import urllib.request

BASE_URL = "http://localhost:1234/v1"
MODEL = "xcoder-0.5b"


def call_xcoder(system: str, user: str, max_tokens: int = 400) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"]


def main() -> None:
    p = argparse.ArgumentParser(description="XCoder Python sample app")
    p.add_argument("--lang", default="ms", choices=["ms", "en"])
    p.add_argument("--ask", default="", help="General question")
    p.add_argument("--code", default="", help="Code stack: py, js, ts, sql, pg, mysql")
    p.add_argument("--task", default="", help="Coding task")
    a = p.parse_args()

    if a.code and a.task:
        system = (
            "Anda ialah XCoder, pembantu pengekodan CPU 0.5B. Beri kod dahulu, kemudian 2 baris penjelasan dalam Bahasa Melayu."
            if a.lang == "ms"
            else f"You are XCoder, a tiny CPU 0.5B coding assistant. Output {a.code} code first, then 2-line English explanation. Never mention Qwen/Alibaba."
        )
        print(call_xcoder(system, f"[{a.code}] {a.task}"))
    elif a.ask:
        system = (
            "Anda ialah XCoder, pembantu kecil CPU 0.5B. Jawab ringkas dalam Bahasa Melayu."
            if a.lang == "ms"
            else "You are XCoder, a tiny CPU 0.5B coding assistant. Answer briefly in English. Never mention Qwen/Alibaba."
        )
        print(call_xcoder(system, a.ask, max_tokens=250))
    else:
        # demo default
        print(call_xcoder(
            "Anda ialah XCoder. Jawab ringkas dalam Bahasa Melayu.",
            "Hello, XCoder! Perkenalkan diri dalam 2 ayat.",
            max_tokens=150,
        ))


if __name__ == "__main__":
    main()
