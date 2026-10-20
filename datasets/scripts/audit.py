"""Deep audit for datasets/<stack>.jsonl (10M-plan quality gate).
Usage:
  python datasets/scripts/audit.py                    # report semua stack (read-only)
  python datasets/scripts/audit.py docker             # satu stack
  python datasets/scripts/audit.py --fix              # pindah baris rosak ke _quarantine.jsonl
  python datasets/scripts/audit.py --fix --samples 200 --strict
  python datasets/scripts/audit.py --report           # tulis datasets/audit/report.json
Flags: --fix, --strict, --json, --report [PATH], --samples N, --max-q N, --max-a N,
       --min-q N, --min-a N, --skip-near, --global (cross-stack exact dup, lebih perlahan).

Checks:
- structural: check_row + input mesti string
- bounds: instruction 5..max-q, output min-a..max-a (aksara; default min 5)
- scrub: qwen/alibaba/tongyi/dashscope dll -> flag KECUALI dalam ayat penafian
  ("Bukan...", "not...", "never...")
- dup: exact (instruction, output) per stack -> dibuang dgn --fix; near-dup
  (simhash64, jarak Hamming <=3) dan same-answer paraphrases -> REPORT sahaja
  (variasi phrasing satu jawapan memang disengajakan)
- retag: detect_stack(instruction) vs nama fail -> report sahaja
- lang: heuristik ms/en/mixed pada instruction
Exit 1 bila ada baris structural rosak; --strict -> exit 1 bila apa-apa flag.
Jangan guna --fix tanpa backup — baris DIPINDAH (bukan dipadam) supaya boleh
re-tag semula kemudian.
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATASETS_DIR, load_config, stack_files, read_rows, check_row, dump_row

ROOT = DATASETS_DIR.parent
QUARANTINE = DATASETS_DIR / "_quarantine.jsonl"
SAMPLES_DIR = DATASETS_DIR / "audit" / "samples"
REPORT_DEFAULT = DATASETS_DIR / "audit" / "report.json"

FORBIDDEN = ["qwen", "alibaba", "tongyi", "dashscope", "通义",
             "as an ai language model", "as a large language model"]
DENIAL_MARKERS = ["bukan", "not ", "never", "tiada", "tidak ", "jangan",
                  "no. i am", "no, i am", "i am xcoder", "saya xcoder"]
MS_MARKERS = ["apa", "apakah", "bagaimana", "terangkan", "saya", "tidak",
              "boleh", "gunakan", "dalam", "untuk", "mengapa", "kenapa",
              "bila", "adakah", "contoh", "senarai", "buat", "tulis", "cara",
              "ialah", "dengan", "jangan", "huraikan", "silakan"]
EN_MARKERS = ["what", "how", "why", "when", "which", "write", "explain",
              "show", "give", "list", "does", "difference", "between",
              "using", "the ", "should", "can i"]
TOKEN_RE = re.compile(r"[a-z0-9_]+")


def scrub_hits(text):
    """Istilah terlarang sebaris TANPA penafian -> senarai istilah.
    Baris identiti (soalan sebut Qwen, jawapan 'Bukan. Saya XCoder') dibenarkan."""
    low = text.lower()
    return [t for t in FORBIDDEN
            if t in low and not any(m in low for m in DENIAL_MARKERS)]


def detect_lang(text):
    t = " " + text.lower() + " "
    ms = sum(1 for m in MS_MARKERS if f" {m}" in t)
    en = sum(1 for m in EN_MARKERS if f" {m}" in t)
    if ms and en:
        return "mixed"
    if ms:
        return "ms"
    if en:
        return "en"
    return "unknown"


_hcache = {}


def _tok_hash(tok):
    h = _hcache.get(tok)
    if h is None:
        h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"),
                                           digest_size=8).digest(), "big")
        _hcache[tok] = h
    return h


def simhash64(text, max_tokens=400):
    toks = list(dict.fromkeys(TOKEN_RE.findall(text)))[:max_tokens]
    if not toks:
        return 0
    bits = [0] * 64
    for t in toks:
        h = _tok_hash(t)
        for i in range(64):
            bits[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, v in enumerate(bits):
        if v > 0:
            out |= 1 << i
    return out


def band_keys(h):
    return [(i, (h >> (16 * i)) & 0xFFFF) for i in range(4)]


def popcount(x):
    return bin(x).count("1")


def load_detector():
    try:
        sys.path.insert(0, str(ROOT / "app"))
        from experts import detect_stack
        return detect_stack
    except Exception:
        return None


def audit_stack(name, rows, args, detect):
    m = {k: 0 for k in ("rows", "bad_structural", "too_short_q", "too_long_q",
                        "too_short_a", "too_long_a", "scrub_hits", "dup_exact",
                        "near_dup_pairs", "same_answer_variants",
                        "retag_mismatch", "identity_rows")}
    m["rows"] = len(rows)
    langs = {"ms": 0, "en": 0, "mixed": 0, "unknown": 0}
    kept, quarantined = [], []
    seen_exact, seen_answer = {}, {}
    band_map = {}
    hashes = []

    for idx, r in enumerate(rows):
        reason = None
        if not check_row(r) or not isinstance(r.get("input", ""), str):
            m["bad_structural"] += 1
            reason = "structural"
        else:
            q, a = r["instruction"].strip(), r["output"].strip()
            if len(q) < args.min_q:
                m["too_short_q"] += 1
                reason = "bounds"
            elif len(q) > args.max_q:
                m["too_long_q"] += 1
                reason = "bounds"
            elif len(a) < args.min_a:
                m["too_short_a"] += 1
                reason = "bounds"
            elif len(a) > args.max_a:
                m["too_long_a"] += 1
                reason = "bounds"
            elif scrub_hits(q + "\n" + a):
                m["scrub_hits"] += 1
                reason = "scrub"
            else:
                key = (q, a)
                if key in seen_exact:
                    m["dup_exact"] += 1
                    reason = "dup_exact"
                else:
                    seen_exact[key] = idx

        if reason:
            quarantined.append((r, reason))
            continue

        r = {"instruction": r["instruction"].strip(),
             "input": (r.get("input") or "").strip(), "output": r["output"].strip()}
        q, a = r["instruction"], r["output"]
        if "XCoder" in a or "Coder 77" in a:
            m["identity_rows"] += 1
        langs[detect_lang(q)] += 1
        prev = seen_answer.get(a)
        if prev is not None and prev != q:
            m["same_answer_variants"] += 1
        else:
            seen_answer.setdefault(a, q)
        if detect and detect(q) not in (name, "general"):
            m["retag_mismatch"] += 1
        if not args.skip_near:
            h = simhash64(q + "\n" + a)
            found = False
            for bkey in band_keys(h):
                if found:
                    break
                for j in band_map.get(bkey, ()):
                    if popcount(h ^ hashes[j]) <= 3:
                        m["near_dup_pairs"] += 1
                        found = True
                        break
            for bkey in band_keys(h):
                band_map.setdefault(bkey, []).append(len(kept))
            hashes.append(h)
        kept.append(r)

    m["kept"] = len(kept)
    m["quarantined"] = len(quarantined)
    m["langs"] = langs
    return kept, quarantined, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stack", nargs="?", default=None)
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--report", nargs="?", const=str(REPORT_DEFAULT), default=None)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--max-q", type=int, default=2000)
    ap.add_argument("--max-a", type=int, default=12000)
    ap.add_argument("--min-q", type=int, default=5)
    ap.add_argument("--min-a", type=int, default=5)
    ap.add_argument("--skip-near", action="store_true")
    ap.add_argument("--global", dest="global_dup", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    files = stack_files(cfg)
    only = args.stack.lower() if args.stack else None
    detect = load_detector()
    if detect is None:
        print("WARN: app/experts.py tidak dapat diimport — retag check dilangkau")

    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "flags": {"max_q": args.max_q, "max_a": args.max_a,
                        "min_q": args.min_q, "min_a": args.min_a,
                        "skip_near": args.skip_near, "fix": args.fix},
              "stacks": {}}
    totals = {k: 0 for k in ("rows", "kept", "quarantined", "bad_structural",
                             "too_short_q", "too_long_q", "too_short_a",
                             "too_long_a", "scrub_hits", "dup_exact",
                             "near_dup_pairs", "same_answer_variants",
                             "retag_mismatch", "identity_rows")}
    total_langs = {"ms": 0, "en": 0, "mixed": 0, "unknown": 0}
    cross = {}
    cross_dupes = 0

    for name, path in sorted(files.items()):
        if only and only not in (name, cfg["stacks"][name]["domain"]):
            continue
        rows = read_rows(path)
        if not rows:
            continue
        kept, quarantined, m = audit_stack(name, rows, args, detect)
        report["stacks"][name] = m
        for k in totals:
            if k in m:
                totals[k] += m[k]
        for k, v in m["langs"].items():
            total_langs[k] += v
        if args.global_dup:
            for r in kept:
                key = (r["instruction"], r["output"])
                prev = cross.get(key)
                if prev is not None and prev != name:
                    cross_dupes += 1
                else:
                    cross.setdefault(key, name)

        if args.fix and quarantined:
            with QUARANTINE.open("a", encoding="utf-8") as f:
                for r, reason in quarantined:
                    row = dict(r) if isinstance(r, dict) else {"_raw": str(r)[:2000]}
                    row["_reason"] = reason
                    row["_stack"] = name
                    f.write(dump_row(row) + "\n")
            if len(kept) < len(rows):
                tmp = path.with_suffix(".jsonl.tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    for r in kept:
                        f.write(dump_row(r) + "\n")
                tmp.replace(path)

        if args.samples > 0 and kept:
            SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
            picks = random.Random(77).sample(kept, min(args.samples, len(kept)))
            with (SAMPLES_DIR / f"{name}.txt").open("w", encoding="utf-8") as f:
                for i, r in enumerate(picks, 1):
                    f.write(f"### {name} sample {i}/{len(picks)}\n")
                    f.write(f"Q: {r['instruction']}\n")
                    if r["input"]:
                        f.write(f"I: {r['input']}\n")
                    f.write(f"A: {r['output']}\n\n")

        flag = m["bad_structural"] + m["too_short_q"] + m["too_long_q"] + \
            m["too_short_a"] + m["too_long_a"] + m["scrub_hits"] + m["dup_exact"]
        extra = ""
        if flag:
            extra = (f"  <-- flag={flag} (struct={m['bad_structural']} "
                     f"bounds={m['too_short_q'] + m['too_long_q'] + m['too_short_a'] + m['too_long_a']} "
                     f"scrub={m['scrub_hits']} dup={m['dup_exact']})")
        print(f"{name:12} rows={m['rows']:5} kept={m['kept']:5} "
              f"near={m['near_dup_pairs']:4} variants={m['same_answer_variants']:5} "
              f"retag={m['retag_mismatch']:4} lang={m['langs']}{extra}")

    report["totals"] = totals
    report["langs"] = total_langs
    if args.global_dup:
        report["cross_stack_dupes"] = cross_dupes
    print("-" * 60)
    print(f"TOTAL rows={totals['rows']} kept={totals['kept']} "
          f"quarantined={totals['quarantined']} near={totals['near_dup_pairs']} "
          f"variants={totals['same_answer_variants']} "
          f"retag={totals['retag_mismatch']}")
    print(f"LANG {total_langs}")
    if args.global_dup:
        print(f"cross-stack exact dupes: {cross_dupes}")

    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                      encoding="utf-8")
        print(f"report: {rp}")
    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))

    if totals["bad_structural"]:
        sys.exit(1)
    if args.strict and (totals["quarantined"] or totals["near_dup_pairs"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
