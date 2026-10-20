"""Safe tools: list/read/write/edit/run/bash/rag/search/fetch/learn — stdlib only, workspace-jailed."""
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd().resolve()
DOCS_DIR = ROOT / "docs"
APP_DIR = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
           else Path(__file__).resolve().parent)

TOOL_SPEC = """Tools (emit one block per call; several calls per turn allowed):
```tool:list <dir>
```tool:read <file>
```tool:write <file> <content...>
```tool:edit <file> <old> => <new>
```tool:run [py|js] <code>
```tool:bash <shell command>
```tool:rag <question>
```tool:search <soalan>
```tool:fetch <url>
```tool:web_fetch <url>
```tool:web_extractor <url>
```tool:code_interpreter [py|js] <code>
```tool:learn <topik> => <teks>
```tool:models
Rules: only use a tool when you NEED a file/command/fact. Otherwise answer directly.
Use tool:search for current/online info; its result is auto-saved to knowledge.
To answer what you are expert in, call tool:models and list ONLY those names.
Never invent model names. After a tool result arrives, continue reasoning.
Max 3 tool steps per question. Paths must stay inside the workspace.
"""


def _safe(p: str) -> Path:
    r = (ROOT / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
    if r != ROOT and ROOT not in r.parents:
        raise ValueError(f"Blocked: outside workspace: {p}")
    return r


def tool_list(arg="docs") -> str:
    d = _safe(arg or ".")
    if not d.exists():
        return f"ERROR: not found: {arg}"
    out = []
    for p in sorted(d.iterdir())[:100]:
        out.append(f"{p.name}{'/' if p.is_dir() else ''}")
    return "\n".join(out) or "(empty)"


def tool_read(arg) -> str:
    f = _safe(arg)
    if not f.is_file():
        return f"ERROR: not a file: {arg}"
    if f.stat().st_size > 30000:
        return "ERROR: file too large (>30KB)"
    return f.read_text(encoding="utf-8", errors="replace")[:8000]


def _need_confirm(ctx, msg):
    return bool(ctx and not ctx.get("allow_all") and ctx.get("confirm")
                and not ctx["confirm"](msg))


def tool_write(arg, ctx=None) -> str:
    # arg: "<file> <content...>"
    parts = arg.split(None, 1)
    if not parts:
        return "ERROR: usage: tool:write <file> <content>"
    f = _safe(parts[0])
    content = parts[1] if len(parts) > 1 else ""
    if _need_confirm(ctx, f"Tulis fail {parts[0]} ({len(content)} aksara)? [y/N] "):
        return "SKIPPED by user"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return f"OK wrote {len(content)} chars to {parts[0]}"


def tool_edit(arg, ctx=None) -> str:
    # arg: "<file> <old> => <new>" (old text must not contain "=>"; split on FIRST "=>")
    if "=>" not in arg:
        return "ERROR: usage: tool:edit <file> <old> => <new>"
    head, new = arg.split("=>", 1)
    fp, _, old = head.partition(" ")
    fp, old, new = fp.strip(), old.strip(), new.strip()
    if not fp or not old:
        return "ERROR: usage: tool:edit <file> <old> => <new>"
    f = _safe(fp)
    if not f.is_file():
        return f"ERROR: not a file: {fp}"
    t = f.read_text(encoding="utf-8")
    n = t.count(old)
    if n == 0:
        return "ERROR: old text not found"
    if _need_confirm(ctx, f"Ubah fail {fp}? [y/N] "):
        return "SKIPPED by user"
    f.write_text(t.replace(old, new, 1), encoding="utf-8")
    extra = f" ({n} matches, replaced first - be more specific if wrong one)" if n > 1 else ""
    return f"OK edited {fp}{extra}"


def _python_exe():
    """Interpreter for python snippets. Frozen exe has no python — use PATH."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    import shutil
    for c in ("py", "python", "python3"):
        p = shutil.which(c)
        if p:
            return p
    return None


def tool_run(arg, timeout=8) -> str:
    # Optional language prefix: "py <code>" or "js <code>". Else heuristic vote.
    lang, code = None, (arg or "").strip()
    for prefix, name in (("py ", "py"), ("py\n", "py"), ("py\t", "py"),
                         ("js ", "js"), ("js\n", "js"), ("js\t", "js")):
        if code.startswith(prefix):
            lang, code = name, code[len(prefix):].strip()
            break
    if lang is None:
        js_hints = ("console.", "=>", "function ", "const ", "let ", "var ",
                    "require(", "process.", "===")
        py_hints = ("import ", "print(", "def ", "self", "None", "True", "False",
                    "elif ", "lambda ")
        js = sum(1 for h in js_hints if h in code)
        py = sum(1 for h in py_hints if h in code)
        lang = "js" if js > py else "py"
    if lang == "py":
        exe = _python_exe()
        if exe is None:
            return "ERROR: no python on PATH (frozen exe cannot run python code)"
        cmd = [exe, "-c", code]
    else:
        cmd = ["node", "-e", code]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        out = (p.stdout or "") + (("\nSTDERR:\n" + p.stderr) if p.stderr else "")
        return (out.strip() or "(no output)")[:4000]
    except subprocess.TimeoutExpired:
        return "ERROR: timeout"
    except FileNotFoundError:
        return f"ERROR: {cmd[0]} not found"


def tool_bash(arg, timeout=10, allow_all=False, confirm=None) -> str:
    dangerous = ["rm -rf", "mkfs", "format", ":(){", "shutdown", "del /f/s/q"]
    if any(d in arg for d in dangerous):
        return "ERROR: blocked dangerous command"
    if not allow_all and confirm and not confirm(f"Run shell? `{arg}` [y/N] "):
        return "SKIPPED by user"
    try:
        p = subprocess.run(arg, shell=True, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout,
                           cwd=str(ROOT))
        out = (p.stdout or "") + (f"\n[exit={p.returncode}]\n" if p.returncode else "")
        out += (p.stderr or "")
        return (out.strip() or "(no output)")[:4000]
    except subprocess.TimeoutExpired:
        return "ERROR: timeout"


def _tokenize(s):
    import re
    return [t for t in re.sub(r"[^a-z0-9_]+", " ", s.lower()).split() if t]


# --- RAG: BM25 over overlapping chunks, index cached by dir signature ---
RAG_EXTS = (".md", ".txt", ".js", ".ts", ".py")
RAG_CHUNK = 500
RAG_OVERLAP = 100
BM25_K1 = 1.2
BM25_B = 0.75
_RAG_CACHE = {"sig": None, "chunks": [], "df": {}, "avgdl": 0.0}


def _rag_index():
    """Build (or reuse) BM25 index. Sig = file names+sizes+mtimes; rebuild on change.
    Rekursif: docs/learned/*.md ikut masuk (pengetahuan terkumpul)."""
    from collections import Counter
    files = [f for f in sorted(DOCS_DIR.rglob("*"))
             if f.is_file() and f.suffix.lower() in RAG_EXTS]
    sig = [(f.relative_to(DOCS_DIR).as_posix(), f.stat().st_size,
            f.stat().st_mtime_ns) for f in files]
    if sig == _RAG_CACHE["sig"]:
        return _RAG_CACHE
    chunks, df = [], {}
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        step = RAG_CHUNK - RAG_OVERLAP
        for start in range(0, len(text), step):
            c = text[start:start + RAG_CHUNK].strip()
            if not c:
                continue
            toks = Counter(_tokenize(c))
            if toks:
                chunks.append((f.relative_to(DOCS_DIR).as_posix(), c, toks))
                for t in toks:
                    df[t] = df.get(t, 0) + 1
    avgdl = (sum(sum(c[2].values()) for c in chunks) / len(chunks)) if chunks else 0.0
    _RAG_CACHE.update(sig=sig, chunks=chunks, df=df, avgdl=avgdl)
    return _RAG_CACHE


def _bm25(qtoks, toks, df, N, avgdl):
    import math
    doclen = sum(toks.values())
    score = 0.0
    for t in set(qtoks):
        tf = toks.get(t, 0)
        if not tf:
            continue
        idf = math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1)
        denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * doclen / max(1e-9, avgdl))
        score += idf * (tf * (BM25_K1 + 1)) / denom
    return score


def tool_rag(arg, k=2) -> str:
    if not DOCS_DIR.is_dir():
        return "ERROR: docs/ not found"
    idx = _rag_index()
    chunks, df, avgdl = idx["chunks"], idx["df"], idx["avgdl"]
    if not chunks:
        return "(no relevant docs)"
    qtoks = _tokenize(arg)
    scored = sorted(((_bm25(qtoks, toks, df, len(chunks), avgdl), n, t)
                     for n, t, toks in chunks), reverse=True)
    ranked = [(n, t) for s, n, t in scored[:max(1, k)] if s > 0]
    if not ranked:
        return "(no relevant docs)"
    return "\n\n".join(f"[{n}]\n{t}" for n, t in ranked)


# --- Learn: auto-simpan pengetahuan web ke docs/learned (RAG) + knowledge.db ---
LEARN_DIR = DOCS_DIR / "learned"
LEARN_MIN_CHARS = 200
LEARN_MAX_BYTES = 8192
LEARN_SESSION_CAP = 10


def _slug(s, n=40):
    t = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return t[:n].rstrip("-") or "nota"


def learn_save(query, text, sources=None, ctx=None):
    """Tulis nota ke docs/learned/<ts>-<slug>.md + knowledge.db (dedup sha256).
    Returns Path nota, atau None kalau dilangkau (pendek/dup/bajet habis)."""
    text = (text or "").strip()
    if len(text) < LEARN_MIN_CHARS:
        return None
    if ctx is not None:
        if "learn_budget" not in ctx:
            ctx["learn_budget"] = LEARN_SESSION_CAP
        if ctx["learn_budget"] <= 0:
            return None
    h = hashlib.sha256(re.sub(r"\s+", " ", text).encode("utf-8")).hexdigest()
    try:
        from store import learned_has
        if learned_has(h):
            return None
    except Exception:
        pass
    LEARN_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = LEARN_DIR / f"{ts}-{_slug(query)}.md"
    n = 1
    while path.exists():
        path = LEARN_DIR / f"{ts}-{_slug(query)}-{n}.md"
        n += 1
    sources = [s for s in (sources or []) if s][:5]
    head = [f"# {(query or 'nota').strip()[:200]}",
            f"Sumber: {', '.join(sources) if sources else '-'}",
            f"Tarikh: {time.strftime('%Y-%m-%d %H:%M')}", ""]
    path.write_text("\n".join(head) + text[:LEARN_MAX_BYTES] + "\n",
                    encoding="utf-8")
    try:
        from store import log_learned
        log_learned(h, (query or "").strip()[:200],
                    (query or "nota").strip()[:200] or "nota",
                    path.read_text(encoding="utf-8"),
                    sources[0] if sources else "", path.name)
    except Exception:
        pass
    if ctx is not None:
        ctx["learn_budget"] = ctx.get("learn_budget", LEARN_SESSION_CAP) - 1
    return path


def maybe_learn(ctx, query, result, sources=None, lang="ms"):
    """Auto-simpan hasil tool web (search/fetch) ke docs/learned + training log.
    Returns Path nota atau None. Hormati ctx['autolearn'] dan bajet sesi."""
    if ctx is not None and not ctx.get("autolearn", True):
        return None
    res = (result or "").strip()
    if not res or res.startswith(("ERROR:", "(no results", "(laman kosong")):
        return None
    if sources is None:
        m = re.search(r"^(?:SUMBER|SOURCES):\s*(.+)$", res, re.M)
        sources = [u.strip() for u in m.group(1).split(",") if u.strip()] if m else []
    path = learn_save(query, res, sources=sources, ctx=ctx)
    if path is None:
        return None
    try:
        from store import log_training
        log_training(query, res[:1000], lang=lang, kind="learn",
                     tools_trace=[{"tool": "web", "arg": (query or "")[:120],
                                   "result": f"docs/learned/{path.name}"}])
    except Exception:
        pass
    return path


def tool_search(arg, ctx=None) -> str:
    from websearch import search as _search
    q = (arg or "").strip()
    if not q:
        return "ERROR: usage: tool:search <soalan>"
    try:
        return _search(q, lang=(ctx or {}).get("lang", "ms"))
    except Exception as e:
        return f"ERROR: carian gagal: {e}"


def tool_fetch(arg, ctx=None) -> str:
    from websearch import fetch_text
    url = (arg or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return "ERROR: usage: tool:fetch <http(s)://url>"
    return fetch_text(url)


def tool_code_interpreter(arg, ctx=None) -> str:
    """Alias gaya OpenAI: jalankan kod dalam kotak pasir (sama seperti tool:run)."""
    return tool_run(arg)


def tool_web_fetch(arg, ctx=None) -> str:
    """Alias: ambil teks bersih dari URL (sama seperti tool:fetch)."""
    return tool_fetch(arg, ctx)


def tool_web_extractor(arg, ctx=None) -> str:
    """Ekstrak title/description/pautan/teks dari URL (HTML -> struktur)."""
    from websearch import extract_page
    url = (arg or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return "ERROR: usage: tool:web_extractor <http(s)://url>"
    return extract_page(url)


def tool_learn(arg, ctx=None) -> str:
    arg = (arg or "").strip()
    if not arg:
        return "ERROR: usage: tool:learn <topik> => <teks>"
    topic, _, text = arg.partition("=>")
    if not text.strip():
        topic, text = arg[:120], arg
    p = maybe_learn(ctx or {}, topic.strip(), text.strip(),
                    lang=(ctx or {}).get("lang", "ms"))
    return f"OK disimpan: docs/learned/{p.name}" if p else \
        "SKIPPED (pendek/dup/bajet habis)"


def tool_models(arg="") -> str:
    """Ground truth: which expert models exist + their domains."""
    from experts import scan, resolve_layout
    _, models_dir = resolve_layout(APP_DIR)
    rows = []
    for m in scan(models_dir):
        rows.append(f"{m['name']} [{','.join(m.get('domains', []))}]")
    return "\n".join(rows) or "(no models)"


DISPATCH = {
    "list": lambda a, ctx: tool_list(a),
    "read": lambda a, ctx: tool_read(a),
    "write": lambda a, ctx: tool_write(a, ctx),
    "edit": lambda a, ctx: tool_edit(a, ctx),
    "run": lambda a, ctx: tool_run(a),
    "bash": lambda a, ctx: tool_bash(a, allow_all=ctx.get("allow_all"), confirm=ctx.get("confirm")),
    "rag": lambda a, ctx: tool_rag(a, k=ctx.get("rag_k", 2)),
    "search": lambda a, ctx: tool_search(a, ctx),
    "fetch": lambda a, ctx: tool_fetch(a, ctx),
    "web_fetch": lambda a, ctx: tool_web_fetch(a, ctx),
    "web_extractor": lambda a, ctx: tool_web_extractor(a, ctx),
    "code_interpreter": lambda a, ctx: tool_code_interpreter(a, ctx),
    "learn": lambda a, ctx: tool_learn(a, ctx),
    "models": lambda a, ctx: tool_models(a),
}
