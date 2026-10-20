"""Agentic loop: prompt model to emit tool blocks, execute, feed back. Works with 0.5B."""
import re

from tools import DISPATCH, TOOL_SPEC, maybe_learn

TOOL_RE = re.compile(r"```tool:(\w+)\s*(.*?)\s*```", re.S)

# Aksara CJK (Cina/Jepun/Korea) — model asas Qwen kadang bocor ke sini;
# khalayak repo ini Melayu + English sahaja, jadi baris beraksara CJK dibuang.
CJK_RE = re.compile("[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af\U00020000-\U0002a6df]")

FENCE_OPEN_RE = re.compile(r"^```[\w+#-]*\s*$")

CLEAN_PATS = [r"qwen2\.5-0\.5b-instruct", r"\bqwen2\.5\b", r"\bqwen3\.5\b",
              r"\bqwen3\b", r"\bqwen\b",
              r"\banthropic\b", r"\bclaude\b", r"\bmeta ai\b", r"\bllama\b",
              r"\balibaba cloud\b", r"\balibaba\b", r"\btongyi\b", r"\bqianwen\b",
              r"\bcoder\s*77\b", r"\bezcodex(?:-0\.5b)?\b"]


def sanitize(text):
    """Remove role-play echo (Pengguna:/Pembantu: lines), instruction echo,
    system-prompt echo (0.5B models repeat system as answer),
    and collapse repeats."""
    # Model penaakulan (Qwen3.5): buang kesan berfikir, ambil jawapan akhir.
    # Ada "[End thinking]" -> ambil selepas yang TERAKHIR (walau tanpa Start).
    # Ada "[Start thinking]" tanpa penutup -> llama.cpp membalut output; kandungan
    # selepasnya ialah jawapan sebenar. Buang penanda sahaja, JANGAN kosongkan
    # (dulu ini memadam jawapan/kod yang betul -> fallback "mengulang arahan").
    t = text or ""
    if "</think>" in t:
        t = t.split("</think>")[-1]
    elif "[End thinking]" in t:
        t = t.split("[End thinking]")[-1]
    elif "<think>" in t:
        t = t.split("<think>", 1)[1]
    elif "[Start thinking]" in t:
        t = t.split("[Start thinking]", 1)[1]
    text = t
    # Substrings (lowercase) that only appear in system prompt / instructions.
    # If a model echoes them, drop that line — real answers never need them.
    SYS_ECHO = (
        "anda ialah",
        "you are xcoder",
        "you are, a cpu",  # safety for variant punctuation
        "a cpu coding assistant",
        "pembantu pengekodan cpu",
        "pembantu kecil cpu",
        "jawab ringkas dalam bahasa melayu",
        "answer briefly in english",
        "jika ditanya siapa anda",
        "siapa cipta anda",
        "if asked who you are",
        "who created you",
        "jawab tepat seperti ini",
        "answer exactly",
        "dibina untuk pengekodan",
        "built for coding",
        "jangan sebut",
        "never mention",
        "aksara cina",
        "chinese/japanese/korean",
        "anda offline",
        "fully offline",
        "tanpa internet",
        "no internet",
        "anda boleh guna tool",
        "you can use tool",
        "tool:search",
        "tool:fetch",
        "sentiasa sebut sumber",
        "always cite sources",
        "maklumat terkini dari internet",
        "up-to-date information",
        "tiada akses",
        "no access to news",
        "berita terkini",
        "data live",
        "latest news",
        "live data",
        "jangan reka",
        "never invent",
        "jangan ulang semula",
        "never repeat the user's",
        "terus jawab",
        "answer directly",
        "guna bahasa melayu malaysia",
        "bukan indonesia",
        "bukan 'komunitas'",
        'bukan "komunitas"',
        "format wajib setiap jawapan",
        "mandatory format for every answer",
        "max 3 tool steps",
        "only use tools when needed",
        "fikir section",
        "jawapan akhir section",
        "thinking section",
        "final answer section",
        "<fikiran ringkas",
        "<brief 1-2 lines",
        "<jawapan penuh",
        "<full answer",
    )
    lines, out, prev, repeats = (text or "").splitlines(), [], None, 0
    sys_removed = 0
    # Buang pasangan fence kosong (```js terus ```) — 0.5B kerap buka-tutup tanpa isi.
    # Pasangan berjujukan: penutup blok berisi tidak boleh dimakan sebagai pembuka.
    _drop, _open, _seen = set(), None, False
    for _i, _ln in enumerate(lines):
        _s = _ln.strip()
        if _s == "```" or FENCE_OPEN_RE.match(_s):
            if _open is None:
                _open, _seen = _i, False
            else:
                if not _seen:
                    _drop.add(_open)
                    _drop.add(_i)
                _open = None
        elif _s:
            _seen = True
    lines = [ln for _i, ln in enumerate(lines) if _i not in _drop]
    for ln in lines:
        s = ln.strip()
        if CJK_RE.search(s):
            continue
        if re.match(r"^(pengguna|pembantu|jawapan sebelum|soalan|user|assistant)\s*:"
                     r"|^hasil tool\b.*?:"
                     r"|^<\|\s*im_(start|end)\s*\|>",
                     s, re.IGNORECASE):
            continue
        if re.match(r"^jawab soalan terakhir|^answer the last question"
                    r"|^jangan sebut|^never mention",
                    s, re.IGNORECASE):
            continue
        low = s.lower()
        # Jawapan identiti yang SAH (cth "Saya XCoder, dibina untuk pengekodan")
        # mengandungi frasa prompt sistem — jangan padam.
        identity_ans = ("saya xcoder", "saya adalah xcoder", "nama saya xcoder",
                        "i am xcoder", "i'm xcoder", "my name is xcoder")
        if any(k in low for k in SYS_ECHO) and not any(k in low for k in identity_ans):
            sys_removed += 1
            continue
        # Broken tool template echo (0.5B invents syntax): ```bash:rag, placeholder args.
        # Real tool calls have real args — placeholders like <question> are never valid.
        if "```" in s and (
                "```bash:" in low
                or "<question>" in low
                or "<shell command>" in low
                or "<shell-command>" in low
                or re.search(r"```tool:\w+\s*<", s, re.IGNORECASE)):
            continue
        ln = re.sub(r"\s*\.{0,3}\s*\(truncated\)\.?", "", ln, flags=re.IGNORECASE)
        s = ln.strip()
        if not s:
            continue
        if s == prev:
            repeats += 1
            if repeats >= 1:
                continue
        else:
            repeats = 0
        prev = s
        out.append(ln)
    # Greeting echo: "Halo! Bagaimana saya boleh membantu..." is valid ONLY as
    # a standalone reply to hai/hello. If it is glued to other content
    # (tool examples, tutorial, code), strip it — it is hallucination.
    # If system echo was removed and only greeting remains, it was part of
    # the echo (T1 case) — return empty so finish() shows fallback.
    if len(out) > 1:
        greet = [l for l in out if "bagaimana saya boleh membantu" in l.lower()
                 or "how can i help you" in l.lower()]
        if greet:
            out = [l for l in out if l not in greet]
    elif len(out) == 1 and sys_removed > 0 and (
            "bagaimana saya boleh membantu" in out[0].lower()
            or "how can i help you" in out[0].lower()):
        return ""
    # Trailing template fragment: model stops mid-template, e.g. real answer +
    # "Tutupi dengan:" with nothing after. Strip those dangling lines.
    while out and re.match(r"^\s*tutupi dengan\s*:?\s*$", out[-1], re.IGNORECASE):
        out.pop()
    text = "\n".join(out).strip()
    return re.sub(r"\s*\(truncated\)\.?\s*$", "", text, flags=re.IGNORECASE).strip()


def clean(text, name):
    t = malay_fix(text)
    for pat in CLEAN_PATS:
        t = re.sub(pat, name, t, flags=re.IGNORECASE)
    t = re.sub(r"an?\s+ai\s+language\s+model\s+created\s+by\s+[^.,;\n]+",
               f"coding assistant, I am {name}", t, flags=re.IGNORECASE)
    return t.strip()


# Prompt sepadan latihan SFT (finetune/train_unsloth.py). Fine-tune xcoder-*
# dilatih dengan sistem ringkas ini + jawapan terus (tiada FIKIR/JAWAPAN AKHIR
# atau tool spec); prompt app yang panjang menyebabkan ia meracau/gagal.
SFT_SYSTEM = ("You are XCoder, a tiny CPU coding assistant. "
              "Answer briefly in the user's language (Bahasa Melayu or English). "
              "Never mention Qwen/Alibaba.")


def is_own_model(backend):
    """Model yang dilatih/dikemas app ini (xcoder-*, bukan Qwen asas)."""
    m = str(getattr(backend, "model_path", None)
            or getattr(backend, "model", "") or "").lower()
    return "xcoder-" in m and "qwen" not in m


def is_sft_model(backend):
    """True bila prompt ringkas (SFT_SYSTEM) perlu digunakan. Selain model yang
    dilatih app ini (xcoder-*), model asas/instruct (cth Qwen3.5 base) juga tidak
    dilatih untuk format panjang FIKIR/JAWAPAN AKHIR — prompt panjang buat ia
    mengulang system prompt dan meracau, jadi ia guna prompt ringkas."""
    m = str(getattr(backend, "model_path", None)
            or getattr(backend, "model", "") or "").lower()
    if is_own_model(backend):
        return True
    return ("qwen" in m or "base" in m) and "think" not in m


def system_prompt(name, lang, tools_on=True, size="0.5B", online=False, compact=False):
    if compact:
        return SFT_SYSTEM
    if lang == "ms":
        net = ("Anda boleh guna tool:search <soalan> untuk maklumat terkini dari internet, "
               "dan tool:fetch <url> untuk baca laman. Sentiasa sebut sumber."
               if online else
               "Anda offline tanpa internet — jika ditanya berita terkini atau data live, "
               "jawab jujur tiada akses, jangan reka.")
    else:
        net = ("You can use tool:search <query> for up-to-date information from the internet, "
               "and tool:fetch <url> to read a page. Always cite sources."
               if online else
               "You are fully offline with no internet — if asked for latest news or live data, "
               "say honestly you have no access, never invent.")
    base = ("Anda ialah {n}, pembantu pengekodan CPU {s}. Jawab ringkas dalam Bahasa Melayu. "
            "Jangan sesekali keluarkan aksara Cina/Jepun/Korea — hanya Bahasa Melayu "
            "(atau English jika diminta). "
            "Jika ditanya siapa anda / siapa cipta anda, jawab tepat seperti ini: Saya {n}, dibina untuk pengekodan. "
            "Jangan sebut Qwen, Alibaba, Tongyi, Gemma, Google, DeepMind. "
            "{net} Jangan ulang semula soalan pengguna; terus jawab."
            if lang == "ms" else
            "You are {n}, a CPU coding assistant {s}. Answer briefly in English. "
            "Never output Chinese/Japanese/Korean characters — English only. "
            "If asked who you are / who created you, answer exactly: I am {n}, built for coding. "
            "Never mention Qwen, Alibaba, Tongyi, Gemma, Google, DeepMind. "
            "{net} Never repeat the user's question back; answer directly.")
    base = base.format(n=name, s=size, net=net)
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
    if lang == "ms":
        base += "\nJika anda model penaakulan, berfikir RINGKAS (maksimum 5 langkah) — jangan meluah panjang."
    else:
        base += "\nIf you are a reasoning model, think BRIEFLY (max 5 steps) — never ramble."
    return base


def split_thinking(text, lang="ms"):
    """Pisah fikiran vs jawapan akhir. Returns (thinking|None, answer).
    Belah pada kemunculan TERAKHIR (model penaakulan kadang bincang format
    dahulu sebelum tulis jawapan sebenar)."""
    mark = "jawapan akhir" if lang == "ms" else "final answer"
    hits = list(re.finditer(mark, text or "", flags=re.IGNORECASE))
    if hits and hits[-1].end() < len(text or ""):
        think = sanitize((text or "")[:hits[-1].start()])
        think = re.sub(r"^(fikir|thinking)\s*:\s*", "", think, flags=re.IGNORECASE)
        answer = re.sub(r"^[\s:\-]+", "", (text or "")[hits[-1].end():])
        if answer.strip():
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
    (r"\bmencoba\b", "mencuba"),
    (r"\bMencoba\b", "Mencuba"),
    (r"\bcoba\b", "cuba"),
    (r"\bCoba\b", "Cuba"),
    (r"\bkalo\b", "kalau"),
    (r"\bKalo\b", "Kalau"),
    (r"\bpakai\b", "guna"),
    (r"\bPakai\b", "Guna"),
    (r"\bkabar\b", "khabar"),
    (r"\bKabar\b", "Khabar"),
    (r"\bkabarmu\b", "khabarmu"),
    (r"\bKabarmu\b", "Khabarmu"),
    (r"\bnggak\b", "tidak"),
    (r"\bngga\b", "tidak"),
    (r"\bdoang\b", "sahaja"),
    (r"\bbanget\b", "sangat"),
    (r"\bngomong\b", "bercakap"),
    (r"\bngerti\b", "faham"),
    (r"\bmembagi\b", "berkongsi"),
    (r"\bMembagi\b", "Berkongsi"),
    (r"\bmenjelajahi\b", "meneroka"),
    (r"\bMenjelajahi\b", "Meneroka"),
    (r"\bmengeksplor\w*\b", "meneroka"),
    (r"\bsukses\b", "berjaya"),
    (r"\bSukses\b", "Berjaya"),
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
              expert_ctx="", on_think=None, size="0.5B", max_tokens=None,
              online=False):
    """Returns (final_answer, tps, tools_trace). expert_ctx grounds the model
    on the real active expert + registry so it never invents model names.
    Thinking trace is parsed and shown via on_think (always on).
    online: prompt selari + hasil search/fetch auto-disimpan ke docs/learned.
    max_tokens: bajet token (None = lalai backend; model penaakulan perlu lebih)."""
    history = history or []
    tool_ctx = tool_ctx or {}
    messages = list(history[-4:]) + [{"role": "user", "content": question}]
    compact = is_sft_model(backend)
    system = system_prompt(name, lang, tools_on, size, online=online, compact=compact)
    if expert_ctx and not compact:
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
        if not ans or re.match(r"^\s*tutupi dengan\s*:?\s*$", ans, re.IGNORECASE):
            # Model only echoed system prompt (0.5B repeat bug) — jangan papar kosong.
            ans = ("Maaf, model mengulang arahan sistem. Sila tanya semula dengan lebih spesifik, "
                   "cth: `tulis fungsi tambah(a,b) python`."
                   if lang == "ms" else
                   "Sorry, the model echoed its system prompt. Please retry with a more specific prompt.")
        for _ in range(2):
            cut = hit_length or re.search(r"\(truncated\)", raw_text, re.IGNORECASE)
            # Auto-continue HANYA untuk model kita (xcoder-*) DAN blok kod
            # terpotong (fence ``` ganjil). Model asas meracau bila disuntik
            # arahan "Sambung jawapan tergantung..." (gelung 0.8B).
            if (not cut or not is_own_model(backend)
                    or ans.count("```") % 2 == 0):
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
                                                     "stopped. Do not restart.")}], max_tokens)
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
    learned = False  # auto-learn: sekali sahaja per giliran
    for _ in range(max_steps + 1):
        raw, tps, fin = ask(system, messages, max_tokens)
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
            if (online and not learned
                    and tname in ("search", "fetch", "web_fetch", "web_extractor")
                    and not result.startswith("ERROR")):
                srcs = [targ.strip()] if tname in ("fetch", "web_fetch", "web_extractor") else None
                p = maybe_learn(tool_ctx, targ, result, sources=srcs, lang=lang)
                if p is not None:
                    learned = True
                    note = f"docs/learned/{p.name}"
                    trace.append({"tool": "learn", "arg": targ[:300], "result": note})
                    if on_tool:
                        on_tool("learn", note, "disimpan + masuk log latihan")
            messages = messages + [{"role": "tool", "name": tname, "content": result}]
        if budget <= 0:
            break
    # max steps hit: one final pass without tools
    plain = system_prompt(name, lang, False, online=online, compact=compact)
    if expert_ctx and not compact:
        plain += "\n\n" + expert_ctx
    raw, tps, fin = ask(plain, messages, max_tokens)
    show_think(raw)
    return finish(strip_tool_blocks(raw), fin == "length"), tps, trace
