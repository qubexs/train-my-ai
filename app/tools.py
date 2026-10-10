"""Safe tools: list/read/write/edit/run/bash/rag — stdlib only, workspace-jailed."""
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
DOCS_DIR = ROOT / "docs"
APP_DIR = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
           else Path(__file__).resolve().parent)

TOOL_SPEC = """Tools (call ONE per turn with this exact block):
```tool:list <dir>
```tool:read <file>
```tool:write <file> <content...>
```tool:edit <file> <old> => <new>
```tool:run <python-or-node -e code>
```tool:bash <shell command>
```tool:rag <question>
```tool:models
Rules: only use a tool when you NEED a file/command/fact. Otherwise answer directly.
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
    # arg: "<file> <old> => <new>"
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
    if old not in t:
        return "ERROR: old text not found"
    if _need_confirm(ctx, f"Ubah fail {fp}? [y/N] "):
        return "SKIPPED by user"
    f.write_text(t.replace(old, new, 1), encoding="utf-8")
    return f"OK edited {fp}"


def tool_run(arg, timeout=8) -> str:
    exe = sys.executable if ("import " in arg or "print(" in arg) else "node"
    try:
        p = subprocess.run([exe, "-e", arg], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        out = (p.stdout or "") + (("\nSTDERR:\n" + p.stderr) if p.stderr else "")
        return (out.strip() or "(no output)")[:4000]
    except subprocess.TimeoutExpired:
        return "ERROR: timeout"
    except FileNotFoundError:
        return f"ERROR: {exe} not found"


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


def tool_rag(arg, k=2) -> str:
    if not DOCS_DIR.is_dir():
        return "ERROR: docs/ not found"
    chunks = []
    for f in sorted(DOCS_DIR.iterdir()):
        if f.suffix.lower() not in (".md", ".txt", ".js", ".ts", ".py"):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i in range(0, len(text), 500):
            c = text[i:i + 500].strip()
            if c:
                chunks.append((f.name, c))
    q = set(_tokenize(arg))

    def score(c):
        return len(q & set(_tokenize(c[1])))

    ranked = sorted(chunks, key=score, reverse=True)[: max(1, k)]
    if not ranked or score(ranked[0]) == 0:
        return "(no relevant docs)"
    return "\n\n".join(f"[{n}]\n{t}" for n, t in ranked)


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
    "models": lambda a, ctx: tool_models(a),
}
