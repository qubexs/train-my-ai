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
                     r"|^hasil tool\b.*?:"
                     r"|^<\|\s*im_(start|end)\s*\|>",
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


def system_prompt(name, lang, tools_on=True, size="0.5B"):
    base = ("Anda ialah {n}, pembantu pengekodan CPU {s}. Jawab ringkas dalam Bahasa Melayu. "
            "Jika ditanya siapa anda / siapa cipta anda, jawab tepat seperti ini: Saya {n}, dibina untuk pengekodan. "
            "Jangan sebut Qwen, Alibaba, Tongyi, Gemma, Google, DeepMind. Anda offline tanpa internet — "
            "jika ditanya berita terkini atau data live, jawab jujur tiada akses, jangan reka. "
            "Jangan ulang semula soalan pengguna; terus jawab."
            if lang == "ms" else
            "You are {n}, a CPU coding assistant {s}. Answer briefly in English. "
            "If asked who you are / who created you, answer exactly: I am {n}, built for coding. "
            "Never mention Qwen, Alibaba, Tongyi, Gemma, Google, DeepMind. You are fully offline with no internet — "
            "if asked for latest news or live data, say honestly you have no access, never invent. "
            "Never repeat the user's question back; answer directly.")
    base = base.format(n=name, s=size)
    if lang == "ms":
        base += ("\n\nGuna Bahasa Melayu Malaysia (BUKAN Indonesia): tulis 'komuniti' bukan 'komunitas', "
                 "'kod' bukan 'kode', 'boleh' bukan 'bisa', 'bagaimana' bukan 'gimana'.")
    base += ("\n\nFormat wajib setiap jawapan:\nFIKIR: <fikiran ringkas 1-2 baris>\n"
             "JAWAPAN AKHIR: <jawapan penuh>"
             if lang == "ms" else
             "\n\nMandatory format for every answer:\nTHINKING: <brief 1-2 lines>\n"
             "FINAL ANSWER: <full answer>")
    if tools_on:
        base += "\n\n" + TOOL_SPEC
    return base


def split_thinking(text, lang="ms"):
    """Pisah fikiran vs jawapan akhir. Returns (thinking|None, answer)."""
    mark = "jawapan akhir" if lang == "ms" else "final answer"
    parts = re.split(mark, text or "", flags=re.IGNORECASE, maxsplit=1)
    if len(parts) == 2 and parts[1].strip():
        think = sanitize(parts[0])
        think = re.sub(r"^(fikir|thinking)\s*:\s*", "", think, flags=re.IGNORECASE)
        answer = re.sub(r"^[\s:\-]+", "", parts[1])
        return think or None, answer
    return None, text


def extract_tool_calls(text):
    """All tool blocks in order: [(name, arg)]."""
    return [(m.group(1).lower(), m.group(2).strip())
            for m in TOOL_RE.finditer(text or "")]


def extract_tool_call(text):
    calls = extract_tool_calls(text)
    return calls[0] if calls else None


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
              expert_ctx="", on_think=None, size="0.5B"):
    """Returns (final_answer, tps, tools_trace). expert_ctx grounds the model
    on the real active expert + registry so it never invents model names.
    Thinking trace is parsed and shown via on_think (always on)."""
    history = history or []
    tool_ctx = tool_ctx or {}
    messages = list(history[-4:]) + [{"role": "user", "content": question}]
    system = system_prompt(name, lang, tools_on, size)
    if expert_ctx:
        system += "\n\n" + expert_ctx
    trace, tps = [], None

    def show_think(raw_text):
        """Papar fikiran inline (JAWAPAN AKHIR dipisah) — tiada panggilan tambahan."""
        if on_think is None:
            return
        try:
            think, _ = split_thinking(raw_text, lang)
            think = malay_fix(think) if think else ""
            if think:
                on_think(think)
        except Exception:
            pass

    def ask(sysmsg, msgs, max_tokens=None):
        """chat_ex when available (text, tps, finish); else chat (finish unknown)."""
        ex = getattr(backend, "chat_ex", None)
        if ex is not None:
            return ex(sysmsg, msgs, max_tokens)
        raw, tps = backend.chat(sysmsg, msgs, max_tokens)
        return raw, tps, "stop"

    def finish(raw_text, hit_length=False):
        """Clean + auto-continue (maks 2) jika jawapan terpotong.
        hit_length comes from backend finish_reason; (truncated) marker is fallback."""
        nonlocal tps
        _, answer = split_thinking(raw_text, lang)
        ans = clean(sanitize(answer), name)
        for _ in range(2):
            cut = hit_length or re.search(r"\(truncated\)", raw_text, re.IGNORECASE)
            if not cut:
                break
            hit_length = False
            raw_text = re.sub(r"\s*\(truncated\)\.?", "", raw_text, flags=re.IGNORECASE)
            try:
                crow, ctps, _ = ask(
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

    budget = max_steps + 1  # total tool executions allowed
    for _ in range(max_steps + 1):
        raw, tps, fin = ask(system, messages)
        calls = extract_tool_calls(raw) if tools_on else []
        if not calls:
            show_think(raw)
            return finish(raw, fin == "length"), tps, trace
        messages = messages + [{"role": "assistant",
                                "content": strip_tool_blocks(raw) or "(using tool)"}]
        for tname, targ in calls:
            if budget <= 0:
                break
            budget -= 1
            fn = DISPATCH.get(tname)
            result = fn(targ, tool_ctx) if fn else f"ERROR: unknown tool {tname}"
            trace.append({"tool": tname, "arg": targ[:300], "result": result[:800]})
            if on_tool:
                on_tool(tname, targ, result)
            messages = messages + [{"role": "tool", "name": tname, "content": result}]
        if budget <= 0:
            break
    # max steps hit: one final pass without tools
    plain = system_prompt(name, lang, False)
    if expert_ctx:
        plain += "\n\n" + expert_ctx
    raw, tps, fin = ask(plain, messages)
    show_think(raw)
    return finish(strip_tool_blocks(raw), fin == "length"), tps, trace
