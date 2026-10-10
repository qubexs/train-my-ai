"""Agentic loop: prompt model to emit tool blocks, execute, feed back. Works with 0.5B."""
import re

from tools import DISPATCH, TOOL_SPEC

TOOL_RE = re.compile(r"```tool:(\w+)\s*(.*?)\s*```", re.S)

CLEAN_PATS = [r"qwen2\.5-0\.5b-instruct", r"\bqwen2\.5\b", r"\bqwen\b",
              r"\banthropic\b", r"\bclaude\b", r"\bmeta ai\b", r"\bllama\b",
              r"\balibaba cloud\b", r"\balibaba\b", r"\btongyi\b", r"\bqianwen\b",
              r"\bcoder\s*77\b", r"\bezcodex(?:-0\.5b)?\b"]


def sanitize(text):
    """Remove role-play echo (Pengguna:/Pembantu: lines), instruction echo,
    and collapse repeats."""
    lines, out, prev, repeats = (text or "").splitlines(), [], None, 0
    for ln in lines:
        s = ln.strip()
        if re.match(r"^(pengguna|pembantu|jawapan sebelum|soalan|user|assistant)\s*:"
                     r"|^hasil tool\b.*?:",
                    s, re.IGNORECASE):
            continue
        if re.match(r"^jawab soalan terakhir|^answer the last question"
                    r"|^jangan sebut|^never mention",
                    s, re.IGNORECASE):
            continue
        ln = re.sub(r"\s*\.{0,3}\s*\(truncated\)\.?", "", ln, flags=re.IGNORECASE)
        s = ln.strip()
        if not s:
            continue
        if s == prev:
            repeats += 1
            if repeats >= 2:
                continue
        else:
            repeats = 0
        prev = s
        out.append(ln)
    text = "\n".join(out).strip()
    return re.sub(r"\s*\(truncated\)\.?\s*$", "", text, flags=re.IGNORECASE).strip()


def clean(text, name):
    t = malay_fix(text)
    for pat in CLEAN_PATS:
        t = re.sub(pat, name, t, flags=re.IGNORECASE)
    t = re.sub(r"an?\s+ai\s+language\s+model\s+created\s+by\s+[^.,;\n]+",
               f"coding assistant, I am {name}", t, flags=re.IGNORECASE)
    return t.strip()


def system_prompt(name, lang, tools_on=True):
    base = ("Anda ialah {n}, pembantu pengekodan CPU kecil 0.5B. Jawab ringkas dalam Bahasa Melayu. "
            "Jika ditanya siapa anda / siapa cipta anda, jawab tepat seperti ini: Saya {n}, dibina untuk pengekodan. "
            "Jangan sebut Qwen, Alibaba, Tongyi. Anda offline tanpa internet — "
            "jika ditanya berita terkini atau data live, jawab jujur tiada akses, jangan reka. "
            "Jangan ulang semula soalan pengguna; terus jawab."
            if lang == "ms" else
            "You are {n}, a tiny CPU 0.5B coding assistant. Answer briefly in English. "
            "If asked who you are / who created you, answer exactly: I am {n}, built for coding. "
            "Never mention Qwen, Alibaba, Tongyi. You are fully offline with no internet — "
            "if asked for latest news or live data, say honestly you have no access, never invent. "
            "Never repeat the user's question back; answer directly.")
    base = base.format(n=name)
    if lang == "ms":
        base += ("\n\nGuna Bahasa Melayu Malaysia (BUKAN Indonesia): tulis 'komuniti' bukan 'komunitas', "
                 "'kod' bukan 'kode', 'boleh' bukan 'bisa', 'bagaimana' bukan 'gimana'.")
    base += ("\n\nSebelum menjawab, tulis fikiran ringkas (2-4 baris) dahulu, "
             "kemudian tulis JAWAPAN AKHIR berasingan."
             if lang == "ms" else
             "\n\nBefore answering, write brief thinking (2-4 lines) first, "
             "then write a separate FINAL ANSWER.")
    if tools_on:
        base += "\n\n" + TOOL_SPEC
    return base


def split_thinking(text, lang="ms"):
    """Pisah fikiran vs jawapan akhir. Returns (thinking|None, answer)."""
    mark = "jawapan akhir" if lang == "ms" else "final answer"
    parts = re.split(mark, text or "", flags=re.IGNORECASE, maxsplit=1)
    if len(parts) == 2 and parts[1].strip():
        return sanitize(parts[0]), parts[1]
    return None, text


def extract_tool_call(text):
    m = TOOL_RE.search(text or "")
    if not m:
        return None
    return m.group(1).lower(), m.group(2).strip()


def strip_tool_blocks(text):
    return TOOL_RE.sub("", text or "").strip()


# Bahasa Melayu Malaysia fixer (prose only — never inside ``` code).
MS_FIX = [
    (r"\bkomunitas\b", "komuniti"),
    (r"\bKomunitas\b", "Komuniti"),
    (r"\bkode\b", "kod"),
    (r"\bKode\b", "Kod"),
    (r"\bbisa\b", "boleh"),
    (r"\bBisa\b", "Boleh"),
    (r"\bgimana\b", "bagaimana"),
    (r"\bGimana\b", "Bagaimana"),
    (r"\bnggak\b", "tidak"),
    (r"\bngga\b", "tidak"),
    (r"\bdoang\b", "sahaja"),
    (r"\bbanget\b", "sangat"),
    (r"\bngomong\b", "bercakap"),
    (r"\bngerti\b", "faham"),
    (r"\byg\b", "yang"),
]


def malay_fix(text):
    parts = re.split(r"(```.*?```)", text or "", flags=re.DOTALL)
    for i in range(0, len(parts), 2):
        for pat, rep in MS_FIX:
            parts[i] = re.sub(pat, rep, parts[i])
    return "".join(parts)


def run_agent(backend, question, name="XCoder", lang="ms", history=None,
              tools_on=True, max_steps=3, tool_ctx=None, on_tool=None,
              expert_ctx="", on_think=None):
    """Returns (final_answer, tps, tools_trace). expert_ctx grounds the model
    on the real active expert + registry so it never invents model names.
    Thinking trace is parsed and shown via on_think (always on)."""
    history = history or []
    tool_ctx = tool_ctx or {}
    messages = list(history[-8:]) + [{"role": "user", "content": question}]
    system = system_prompt(name, lang, tools_on)
    if expert_ctx:
        system += "\n\n" + expert_ctx
    # Fikiran dahulu (panggilan khas, sentiasa ditunjuk): apa yang AI akan buat.
    if on_think is not None:
        try:
            tsys = ("Tulis fikiran ringkas 2-4 baris: apa yang pengguna mahu dan bagaimana anda akan jawab. "
                    "Jangan jawab lagi, tulis fikiran sahaja."
                    if lang == "ms" else
                    "Write brief thinking, 2-4 lines: what the user wants and how you will answer. "
                    "Do not answer yet, thinking only.")
            if expert_ctx:
                tsys += "\n\n" + expert_ctx
            traw, _ = backend.chat(tsys, messages[-2:], max_tokens=150)
            thinking = sanitize(strip_tool_blocks(traw))
            if thinking:
                on_think(thinking)
        except Exception:
            pass
    trace, tps = [], None

    def finish(raw_text):
        """Clean + auto-continue (maks 2) jika jawapan terpotong."""
        nonlocal tps
        _, answer = split_thinking(raw_text, lang)
        ans = clean(sanitize(answer), name)
        for _ in range(2):
            if not re.search(r"\(truncated\)", raw_text, re.IGNORECASE):
                break
            raw_text = re.sub(r"\s*\(truncated\)\.?", "", raw_text, flags=re.IGNORECASE)
            try:
                crow, ctps = backend.chat(
                    system, messages + [{"role": "assistant", "content": ans},
                                        {"role": "user",
                                         "content": ("Sambung jawapan tergantung di atas dari tepat mana "
                                                     "ia berhenti. Jangan ulang dari awal."
                                                     if lang == "ms" else
                                                     "Continue the cut-off answer above from exactly where it "
                                                     "stopped. Do not restart.")}])
            except Exception:
                break
            if ctps:
                tps = ctps
            _, more = split_thinking(crow, lang)
            more = re.sub(r"^\s*(sambungan|jawapan akhir|final answer)\s*[:\-]?\s*",
                          "", more, flags=re.IGNORECASE)
            ans = (ans + "\n" + clean(sanitize(more), name)).strip()
            raw_text = crow
        return ans

    for _ in range(max_steps + 1):
        raw, tps = backend.chat(system, messages)
        call = extract_tool_call(raw) if tools_on else None
        if not call:
            return finish(raw), tps, trace
        tname, targ = call
        fn = DISPATCH.get(tname)
        result = fn(targ, tool_ctx) if fn else f"ERROR: unknown tool {tname}"
        trace.append({"tool": tname, "arg": targ[:300], "result": result[:800]})
        if on_tool:
            on_tool(tname, targ, result)
        messages = messages + [{"role": "assistant", "content": strip_tool_blocks(raw) or "(using tool)"},
                               {"role": "tool", "name": tname, "content": result}]
    # max steps hit: one final pass without tools
    plain = system_prompt(name, lang, False)
    if expert_ctx:
        plain += "\n\n" + expert_ctx
    raw, tps = backend.chat(plain, messages)
    return finish(strip_tool_blocks(raw)), tps, trace
