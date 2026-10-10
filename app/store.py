"""Sessions + training capture — jsonl, stdlib only."""
import json
import sys
import time
from pathlib import Path

from experts import detect as detect_domain
from experts import detect_stack, DOMAINS, STACK2DOMAIN

APP_DIR = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
           else Path(__file__).resolve().parent)
SESSIONS_DIR = APP_DIR / "sessions"
DATA_DIR = APP_DIR / "data"
SESSIONS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def session_path(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "default"
    return SESSIONS_DIR / f"{safe}.jsonl"


def load_session(name: str):
    p = session_path(name)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return [r for r in rows if r.get("role") in ("user", "assistant", "tool")]


def save_turn(name: str, messages, keep_last=60):
    p = session_path(name)
    with p.open("a", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    # trim: session file stays bounded, context belongs to session file
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
        if len(lines) > keep_last:
            p.write_text("\n".join(lines[-keep_last:]) + "\n", encoding="utf-8")
    except OSError:
        pass


def clear_session(name: str):
    p = session_path(name)
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


def context_stats(name: str, memory_turns: int):
    file_turns = len(load_session(name))
    chars = sum(len((m.get("content") or "")) for m in load_session(name)[-20:])
    return {"session": name, "memory": memory_turns, "file": file_turns,
            "approx_tokens": chars // 4}


def list_sessions():
    return sorted(p.stem for p in SESSIONS_DIR.glob("*.jsonl"))


def log_training(question, answer, lang="ms", kind="chat", tools_trace=None,
                 rating=None, domain=None):
    """Append one SFT row + raw trace for later pretrain/finetune export."""
    row = {
        "instruction": question,
        "input": "",
        "output": answer,
        "meta": {"lang": lang, "kind": kind, "ts": int(time.time()),
                 "tools": tools_trace or [], "rating": rating,
                 "domain": domain or detect_domain(question)},
    }
    p = DATA_DIR / "training.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def _match(flt, stack):
    if flt == stack:
        return True
    return flt in DOMAINS and STACK2DOMAIN.get(stack) == flt


def export_stacks(dest_dir="datasets", domain=None, min_len=2):
    """Split training.jsonl into datasets/<stack>.jsonl (dedupe).
    domain filter: stack name (docker) or broad domain (web/data/linux...)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    src = DATA_DIR / "training.jsonl"
    if not src.exists():
        return {}, "no training.jsonl yet"
    flt = (domain or "").strip().lower() or None
    seen = {}
    added = {}
    for line in src.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        q = (r.get("instruction") or "").strip()
        a = (r.get("output") or "").strip()
        if len(q) < min_len or len(a) < min_len:
            continue
        if (r.get("meta") or {}).get("rating") == "bad":
            continue
        stack = detect_stack(q + " " + a)
        if flt and not _match(flt, stack):
            continue
        dest = dest_dir / f"{stack}.jsonl"
        if stack not in seen:
            seen[stack] = set()
            if dest.exists():
                for l in dest.read_text(encoding="utf-8").splitlines():
                    try:
                        e = json.loads(l)
                        seen[stack].add((e.get("instruction", ""), e.get("output", "")))
                    except ValueError:
                        pass
        if (q, a) in seen[stack]:
            continue
        seen[stack].add((q, a))
        with dest.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"instruction": q, "input": r.get("input", ""),
                                "output": a}, ensure_ascii=False) + "\n")
        added[stack] = added.get(stack, 0) + 1
    return added, str(dest_dir)


def merge_stacks(root=".", domain=None):
    """Combine datasets/<stack>.jsonl -> finetune/<domain>/dataset.jsonl (dedupe)."""
    root = Path(root)
    cfg = json.loads((root / "datasets" / "config" / "dataset.json").read_text(encoding="utf-8"))
    out = {}
    for dom, stacks in sorted(cfg["domains"].items()):
        if domain and domain != dom:
            continue
        seen, merged = set(), []
        for s in stacks:
            p = root / "datasets" / f"{s}.jsonl"
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                q, a = (r.get("instruction") or "").strip(), (r.get("output") or "").strip()
                if not q or not a or (q, a) in seen:
                    continue
                seen.add((q, a))
                merged.append({"instruction": q, "input": (r.get("input") or "").strip(),
                               "output": a})
        dest = root / "finetune" / dom / "dataset.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out[dom] = len(merged)
    # per-stack copies (validated, training-ready): finetune/<stack>/dataset.jsonl
    for s in sorted(cfg["stacks"]):
        src = root / "datasets" / f"{s}.jsonl"
        rows = []
        if src.exists():
            seen = set()
            for line in src.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                q, a = (r.get("instruction") or "").strip(), (r.get("output") or "").strip()
                if not q or not a or (q, a) in seen:
                    continue
                seen.add((q, a))
                rows.append({"instruction": q, "input": (r.get("input") or "").strip(),
                             "output": a})
        dest = root / "finetune" / s / "dataset.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def _row_domain(r):
    d = (r.get("meta") or {}).get("domain")
    if d:
        return d
    return detect_domain((r.get("instruction") or "") + " " + (r.get("output") or ""))


def export_sft(dest="finetune/general/dataset.jsonl", min_len=2, domain=None):
    src = DATA_DIR / "training.jsonl"
    if not src.exists():
        return 0, "no training.jsonl yet — chat first, then export"
    seen, added = set(), 0
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    if dest_p.exists():
        for line in dest_p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                seen.add((r.get("instruction", ""), r.get("output", ""))[:2])
            except ValueError:
                pass
    with src.open(encoding="utf-8") as fin, dest_p.open("a", encoding="utf-8") as fout:
        for line in fin:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            q, a = (r.get("instruction") or "").strip(), (r.get("output") or "").strip()
            if len(q) < min_len or len(a) < min_len:
                continue
            if (r.get("meta") or {}).get("rating") == "bad":
                continue
            if domain and _row_domain(r) != domain:
                continue
            if domain and _row_domain(r) != domain:
                continue
            key = (q, a)
            if key in seen:
                continue
            seen.add(key)
            fout.write(json.dumps({"instruction": q, "input": r.get("input", ""),
                                   "output": a}, ensure_ascii=False) + "\n")
            added += 1
    return added, str(dest_p)


def export_pretrain_corpus(dest="finetune/general/corpus.txt", domain=None):
    src = DATA_DIR / "training.jsonl"
    if not src.exists():
        return 0, "no training.jsonl yet"
    n = 0
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as fin, out.open("a", encoding="utf-8") as fout:
        for line in fin:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if domain and _row_domain(r) != domain:
                continue
            if (r.get("meta") or {}).get("rating") == "bad":
                continue
            q, a = r.get("instruction", ""), r.get("output", "")
            fout.write(f"### User:\n{q}\n\n### Assistant:\n{a}\n\n")
            n += 1
    # also dump docs/*.md for continued pretraining (general corpus only)
    if not domain:
        for md in list(Path.cwd().glob("docs/*.md"))[:20]:
            try:
                out.open("a", encoding="utf-8").write("\n" + md.read_text(encoding="utf-8") + "\n")
            except OSError:
                pass
    return n, str(out)
