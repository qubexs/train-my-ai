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
        if re.match(r"^jawab soalan terakhir|^answer the last question",
                    s, re.IGNORECASE):
            continue
        if s == prev:
            repeats += 1
            if repeats >= 2:
                continue
        else:
            repeats = 0
        prev = s
        out.append(ln)
    return "\n".join(out).strip()


def clean(text, name):
    t = text
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
    if tools_on:
        base += "\n\n" + TOOL_SPEC
    return base


def extract_tool_call(text):
    m = TOOL_RE.search(text or "")
    if not m:
        return None
    return m.group(1).lower(), m.group(2).strip()


def strip_tool_blocks(text):
    return TOOL_RE.sub("", text or "").strip()


def run_agent(backend, question, name="XCoder", lang="ms", history=None,
              tools_on=True, max_steps=3, tool_ctx=None, on_tool=None,
              expert_ctx=""):
    """Returns (final_answer, tps, tools_trace). expert_ctx grounds the model
    on the real active expert + registry so it never invents model names."""
    history = history or []
    tool_ctx = tool_ctx or {}
    messages = list(history[-8:]) + [{"role": "user", "content": question}]
    system = system_prompt(name, lang, tools_on)
    if expert_ctx:
        system += "\n\n" + expert_ctx
    trace, tps = [], None
    for _ in range(max_steps + 1):
        raw, tps = backend.chat(system, messages)
        call = extract_tool_call(raw) if tools_on else None
        if not call:
            return clean(sanitize(raw), name), tps, trace
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
    return clean(sanitize(strip_tool_blocks(raw)), name), tps, trace
