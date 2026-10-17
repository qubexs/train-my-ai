"""Import jejak agen luaran (Fable 5k — format OpenAI tool-calling) ke baris
XCoder {instruction, input, output}.

Peta penukaran (boleh dilaras melalui bendera):
  <thought>  -> FIKIR: (lalai) atau [berfikir] (--think-label berfikir)
  <final_answer> -> JAWAPAN AKHIR:
  tool_calls -> blok ```tool:<nama> (hanya bila hujah ASLI, bukan placeholder)
  role tool   -> dipadatkan dalam konteks: "[Tool: nama] -> hasil"
  jejak 213KB -> konteks berjendela (--ctx-chars), jawapan dipotong (--max-final)

Nota: ~99% hujah tool_calls dalam Fable 5k telah disanitasi jadi placeholder
(<file_path>, <bash_command>). Baris sedemikian tidak menghasilkan sampel
"sasaran tool" — hanya sampel konteks->jawapan akhir (yang asli).

Subcommands:
  convert   : Fable jsonl -> datasets/fable_agent.jsonl
  translate : terjemah medan ke Bahasa Melayu Malaysia melalui server
              OpenAI-compatible tempatan (llama-server), berkabus + boleh sambung.

Contoh:
  python datasets/scripts/import_fable.py convert --src train.jsonl
  python datasets/scripts/import_fable.py convert --src train.jsonl --limit 500 --out datasets/fable_500.jsonl
  python datasets/scripts/import_fable.py translate --in datasets/fable_agent.jsonl \
      --server http://127.0.0.1:8080 --model gemma --workers 2
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

THOUGHT_RE = re.compile(r"<thought>\s*(.*?)\s*</thought>", re.S)
FINAL_RE = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.S)
PLACEHOLDER_RE = re.compile(
    r"<(?:file_path|directory_path|file_content|bash_command|shell_command|"
    r"shell-command|search_query|query|path|command|content|question|email|"
    r"name|id|arg|args|input|output|text|code|url|value|key|pattern|target|"
    r"source|dest|destination)>", re.I)
TAG_RE = re.compile(r"</?(?:thought|final_answer)>", re.I)
WS_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{3,}")

TOOL_MAP = {"bash": "bash", "read_file": "read", "write_file": "write",
            "list_dir": "list", "search": "rag"}

MT_SYS = ("Anda penterjemah teknikal. Terjemah teks pengguna ke Bahasa Melayu "
          "Malaysia (BUKAN Indonesia: guna 'komuniti', 'kod', 'boleh', "
          "'bagaimana', 'fail', 'paparan'). KEKALKAN kod, nama fail, jalan "
          "fail, arahan shell, dan format markdown. KEKALKAN token @@FK@@ dan "
          "@@JA@@ tepat seperti asal. Jangan tambah, ringkas, atau buang "
          "kandungan. Output terjemahan sahaja tanpa mukadimah.")

LABEL_MASKS = [("FIKIR:", "@@FK@@"), ("JAWAPAN AKHIR:", "@@JA@@")]


def mask_labels(text):
    for label, token in LABEL_MASKS:
        text = text.replace(label, token)
    return text


def unmask_labels(text):
    text = re.sub(r"@+\s*F\s*K\s*@+", "FIKIR:", text)
    text = re.sub(r"@+\s*J\s*A\s*@+", "JAWAPAN AKHIR:", text)
    for label, token in LABEL_MASKS:
        text = text.replace(token, label)
    return text

UNICODE_MAP = {"\u2192": "->", "\u2190": "<-", "\u2194": "<->", "\u00d7": "x",
               "\u2139": "(i)", "\u2714": "OK", "\u00b7": ".", "\u2018": "'",
               "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2265": ">=",
               "\u2264": "<=", "\u2260": "!=", "\u2212": "-", "\u00bc": "1/4",
               "\u00ab": "<<", "\u00bb": ">>", "\u2028": "\n", "\u2029": "\n",
               "\ufffd": "", "\u0307": "", "\u00e8": "e", "\u00df": "ss",
               "\u0130": "I", "\u0131": "i"}


def normalize_unicode(text):
    for ch, rep in UNICODE_MAP.items():
        text = text.replace(ch, rep)
    return text


def prose_segments(text):
    """Pecahan (is_code, seg) pada pagar ``` — hanya prosa diterjemah."""
    parts = re.split(r"(```.*?```)", text or "", flags=re.S)
    out = []
    for i, s in enumerate(parts):
        if not s:
            continue
        out.append((i % 2 == 1, s))
    return out


def clean_text(s):
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = WS_RE.sub(" ", s)
    return BLANK_RE.sub("\n\n", s).strip()


def one_line(s, n):
    s = clean_text(s).replace("\n", " ")
    return s if len(s) <= n else s[:n].rstrip() + " ..."


def thought_of(content):
    m = THOUGHT_RE.search(content or "")
    return clean_text(m.group(1)) if m else ""


def final_of(content):
    content = content or ""
    m = FINAL_RE.search(content)
    if m:
        return clean_text(m.group(1))
    if "<final_answer>" in content.lower():
        return clean_text(re.split(r"<final_answer>", content, flags=re.I)[-1])
    return ""


def real_args(call):
    """dict hujah jika ASLI (tiada placeholder), else None."""
    try:
        obj = json.loads(call.get("function", {}).get("arguments") or "")
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or not obj:
        return None
    for v in obj.values():
        if isinstance(v, str) and PLACEHOLDER_RE.search(v):
            return None
    return obj


def render_call_block(name, args, max_body=800):
    if name == "bash":
        body = args.get("command", "")
    elif name in ("read_file", "list_dir"):
        body = args.get("path", "")
    elif name == "search":
        body = args.get("query", "")
    elif name == "write_file":
        body = f"{args.get('path', '')} {args.get('content', '')}"
    else:
        return None
    body = clean_text(str(body))
    if not body:
        return None
    if len(body) > max_body:
        body = body[:max_body].rstrip() + "\n[...dipenggal]"
    return f"```tool:{TOOL_MAP[name]}\n{body}\n```"


def compact_result(s, head=260, tail=80):
    s = clean_text(s)
    if len(s) <= head + tail + 30:
        return s
    return s[:head].rstrip() + "\n[...]\n" + s[-tail:].lstrip()


def walk(messages):
    """Senaraikan langkah assistant: {think, calls, results, final}."""
    steps, pending = [], {}
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            content = m.get("content") or ""
            calls = []
            for c in m.get("tool_calls") or []:
                fn = c.get("function", {})
                calls.append((fn.get("name") or "?", real_args(c), c.get("id")))
            steps.append({"think": thought_of(content), "calls": calls,
                          "results": [], "final": final_of(content),
                          "plain": clean_text(TAG_RE.sub("", content))})
            for _, _, cid in calls:
                if cid:
                    pending[cid] = len(steps) - 1
        elif role == "tool":
            s = pending.get(m.get("tool_call_id"))
            if s is not None:
                steps[s]["results"].append(
                    (m.get("name") or "?", m.get("content") or ""))
    return steps


def render_step(i, st):
    lines = []
    if st["think"]:
        lines.append(f"[Langkah {i}] [berfikir] {one_line(st['think'], 300)}")
    else:
        lines.append(f"[Langkah {i}]")
    for name, args, _ in st["calls"]:
        arg_hint = ""
        if args:
            hint = args.get("command") or args.get("path") or args.get("query") or ""
            if hint:
                arg_hint = f" {one_line(str(hint), 80)}"
        lines.append(f"[Tool: {name}]{arg_hint}")
    for name, res in st["results"]:
        lines.append(f"[Tool: {name}] -> {compact_result(res)}")
    return "\n".join(lines)


def render_context(steps, budget, skip_last=0):
    steps = steps[:len(steps) - skip_last] if skip_last else steps
    rendered, used = [], 0
    for i in range(len(steps) - 1, -1, -1):
        blk = render_step(i + 1, steps[i])
        if used + len(blk) > budget and rendered:
            rendered.append(f"[... {i + 1} langkah awal diringkaskan ...]")
            break
        rendered.append(blk)
        used += len(blk)
    return "\n\n".join(reversed(rendered))


def convert_row(row, a, seen_final, seen_tool):
    msgs = row.get("messages") or []
    user = next((m.get("content") or "" for m in msgs if m.get("role") == "user"), "")
    if not user.strip():
        return [], "skip:no_user"
    steps = walk(msgs)
    if not steps:
        return [], "skip:no_steps"

    task = clean_text(user)
    if len(task) > a.task_chars:
        task = task[:a.task_chars].rstrip() + "\n[...]"

    out = []
    n_tool = 0
    for i, st in enumerate(steps):
        if n_tool >= a.max_tool_samples:
            break
        blocks = [b for b in (render_call_block(n, args)
                              for n, args, _ in st["calls"] if args)
                  if b]
        if not blocks:
            continue
        head = f"FIKIR: {clean_text(st['think'])}\n\n" if st["think"] else ""
        r = {"instruction": task,
             "input": render_context(steps, a.ctx_chars, skip_last=len(steps) - i),
             "output": head + "\n\n".join(blocks)}
        key = hashlib.sha1((r["input"] + "\x00" + r["output"]).encode("utf-8")).hexdigest()
        if key not in seen_tool:
            seen_tool.add(key)
            out.append(r)
        n_tool += 1

    final = ""
    for st in reversed(steps):
        if st["final"]:
            final = st["final"]
            break
    if not final and msgs and msgs[-1].get("role") == "assistant":
        plain = steps[-1]["plain"]
        if plain and not steps[-1]["think"] and not steps[-1]["calls"]:
            final = plain

    if final and len(final) >= a.min_final:
        if len(final) > a.max_final:
            final = final[:a.max_final].rstrip() + "\n[...dipenggal]"
        think = ""
        for st in reversed(steps):
            if st["think"]:
                think = st["think"]
                break
        if think and a.think_label == "berfikir":
            head = f"[berfikir]\n{clean_text(think)}\n\n"
        elif think:
            head = f"FIKIR: {clean_text(think)}\n\n"
        else:
            head = ""
        r = {"instruction": task,
             "input": render_context(steps, a.ctx_chars),
             "output": f"{head}JAWAPAN AKHIR:\n{final}"}
        key = hashlib.sha1(r["output"].encode("utf-8")).hexdigest()
        if key not in seen_final:
            seen_final.add(key)
            out.insert(0, r)

    if not out:
        if not final:
            return [], "skip:no_final"
        return [], "skip:short_final"
    return out, "ok"


def cmd_convert(a):
    src = Path(a.src)
    if not src.is_file():
        print(f"src tidak wujud: {src}")
        return 1
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"rows": 0, "final": 0, "tool": 0}
    skips = {}
    seen_final, seen_tool = set(), set()
    with src.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["rows"] += 1
            if a.limit and stats["rows"] > a.limit:
                break
            try:
                row = json.loads(line)
            except ValueError:
                skips["skip:bad_json"] = skips.get("skip:bad_json", 0) + 1
                continue
            try:
                rows, why = convert_row(row, a, seen_final, seen_tool)
            except Exception as exc:  # baris luaran — jangan matikan seluruh larian
                skips["skip:error"] = skips.get("skip:error", 0) + 1
                if a.verbose:
                    print(f"  ralat baris {stats['rows']}: {exc}")
                continue
            if not rows:
                skips[why] = skips.get(why, 0) + 1
                continue
            n_final = sum(1 for r in rows if "JAWAPAN AKHIR" in r["output"])
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            stats["final"] += n_final
            stats["tool"] += len(rows) - n_final
    n_out = stats["final"] + stats["tool"]
    print(f"convert: {stats['rows']} baris masuk -> {n_out} baris keluar "
          f"({stats['final']} jawapan akhir + {stats['tool']} sasaran tool)")
    for k, v in sorted(skips.items()):
        print(f"  {k}: {v}")
    print(f"ditulis: {out_path}")
    return 0


def http_translate(server, model, text, timeout):
    body = json.dumps({
        "model": model, "temperature": 0.1, "max_tokens": 3000,
        "messages": [{"role": "system", "content": MT_SYS},
                     {"role": "user", "content": text}]}).encode("utf-8")
    req = urllib.request.Request(server.rstrip("/") + "/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def translate_long(server, model, text, timeout, chunk=3000):
    if len(text) <= chunk:
        return http_translate(server, model, text, timeout)
    parts, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) > chunk and cur:
            parts.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return "\n\n".join(http_translate(server, model, p, timeout) for p in parts)


def cmd_translate(a):
    in_path, out_path = Path(a.src_in), Path(a.out)
    cache_path = Path(a.cache)
    cache = {}
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            try:
                o = json.loads(line)
                cache[o["h"]] = o["t"]
            except (ValueError, KeyError):
                pass
    print(f"cache: {len(cache)} entri")
    rows = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]

    jobs = {}
    for r in rows:
        texts = [r["instruction"]] + ([r["output"]] if a.translate_output else [])
        for text in texts:
            for is_code, seg in prose_segments(mask_labels(text)):
                if is_code or not seg.strip():
                    continue
                h = hashlib.sha1(seg.encode("utf-8")).hexdigest()
                if h not in cache:
                    jobs[h] = seg
    print(f"segmen unik perlu terjemah: {len(jobs)}")

    def work(item):
        h, seg = item
        try:
            return h, translate_long(a.server, a.model, seg, a.timeout), None
        except Exception as exc:
            return h, None, str(exc)

    done, err = 0, 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cfile = cache_path.open("a", encoding="utf-8")
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for h, tr, e in pool.map(work, jobs.items()):
            if e:
                err += 1
                if a.verbose:
                    print(f"  ralat {h[:8]}: {e[:100]}")
            else:
                cache[h] = tr
                done += 1
                cfile.write(json.dumps({"h": h, "t": tr},
                                       ensure_ascii=False) + "\n")
                cfile.flush()
            if (done + err) % 25 == 0 and (done + err):
                print(f"  progres: {done} ok, {err} gagal", flush=True)
    cfile.close()
    if not jobs:
        print("  (semua segmen sudah dalam cache)")

    def rebuild(text):
        if not text:
            return text
        out = []
        for is_code, seg in prose_segments(mask_labels(text)):
            if is_code or not seg.strip():
                out.append(seg)
                continue
            lead = seg[:len(seg) - len(seg.lstrip())]
            trail = seg[len(seg.rstrip()):]
            h = hashlib.sha1(seg.encode("utf-8")).hexdigest()
            out.append(lead + cache.get(h, seg) + trail)
        return unmask_labels("".join(out))

    for r in rows:
        r["instruction"] = rebuild(r["instruction"])
        if a.translate_output:
            r["output"] = rebuild(r["output"])

    with out_path.open("w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"translate: {done} segmen baharu, {err} gagal -> {out_path}")
    return 1 if err and not done else 0


def cmd_clean(a):
    rows = [json.loads(l) for l in Path(a.src_in).open(encoding="utf-8")
            if l.strip()]
    changed = 0
    out_path = Path(a.out)
    with out_path.open("w", encoding="utf-8") as out:
        for r in rows:
            before = (r.get("instruction", ""), r.get("input", ""),
                      r.get("output", ""))
            r["instruction"] = normalize_unicode(before[0])
            r["input"] = normalize_unicode(before[1])
            r["output"] = normalize_unicode(before[2])
            if (r["instruction"], r["input"], r["output"]) != before:
                changed += 1
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"clean: {len(rows)} baris, {changed} dinormalisasi -> {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="Fable jsonl -> XCoder jsonl")
    c.add_argument("--src", required=True)
    c.add_argument("--out", default="datasets/fable_agent.jsonl")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--min-final", type=int, default=80)
    c.add_argument("--max-final", type=int, default=4000)
    c.add_argument("--task-chars", type=int, default=1200)
    c.add_argument("--ctx-chars", type=int, default=3000)
    c.add_argument("--max-tool-samples", type=int, default=10)
    c.add_argument("--think-label", choices=["fikir", "berfikir"], default="fikir")
    c.add_argument("--verbose", action="store_true")
    c.set_defaults(fn=cmd_convert)

    t = sub.add_parser("translate", help="terjemah ke Bahasa Melayu Malaysia")
    t.add_argument("--in", dest="src_in", default="datasets/fable_agent.jsonl")
    t.add_argument("--out", default="datasets/fable_agent_ms.jsonl")
    t.add_argument("--server", default="http://127.0.0.1:8080")
    t.add_argument("--model", default="gemma")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--timeout", type=int, default=180)
    t.add_argument("--cache", default="datasets/.fable_mt_cache.jsonl")
    t.add_argument("--translate-output", action="store_true",
                   help="terjemah juga prosa dalam output (kod dikecualikan)")
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--verbose", action="store_true")
    t.set_defaults(fn=cmd_translate)

    k = sub.add_parser("clean", help="normalisasi aksara unicode ke ascii")
    k.add_argument("--in", dest="src_in", default="datasets/fable_agent_ms.jsonl")
    k.add_argument("--out", default="datasets/fable_agent_ms.jsonl")
    k.set_defaults(fn=cmd_clean)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
