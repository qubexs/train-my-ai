"""Sessions + training capture - SQLite (db.py), stdlib only.

Dua DB di app/data/:
  * chat.db      - sesi/chat  (load_session/save_turn/clear_session/list_sessions)
  * knowledge.db - learn + dataset on-progress + train_state (rv)
JSONL kekal sebagai EKSPORT sahaja (export_*), untuk pipeline latihan.
Data lama (sessions/*.jsonl, data/training.jsonl, data/train_state.json,
docs/learned/.index.jsonl) dipindahkan automatik pada import kali pertama.
"""
import json
import sys
import time
from pathlib import Path

import db
from experts import detect as detect_domain
from experts import detect_stack, STACK2DOMAIN, LEGACY_GROUPS

APP_DIR = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
           else Path(__file__).resolve().parent)
SESSIONS_DIR = APP_DIR / "sessions"
DATA_DIR = APP_DIR / "data"
SESSIONS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

CHAT_DB = DATA_DIR / "chat.db"
KNOW_DB = DATA_DIR / "knowledge.db"

# Laluan legacy (migrasi sahaja)
TRAIN_STATE = DATA_DIR / "train_state.json"


def session_path(name: str) -> Path:
    """Laluan JSONL lama (compat/migrasi sahaja)."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "default"
    return SESSIONS_DIR / f"{safe}.jsonl"


# --------------------------------------------------------------- sessions ----
def load_session(name: str):
    return db.get_messages(CHAT_DB, name)


def save_turn(name: str, messages, keep_last=60):
    db.add_messages(CHAT_DB, name, messages, int(time.time()))
    db.trim_session(CHAT_DB, name, keep_last)


def clear_session(name: str):
    db.clear_session(CHAT_DB, name)


def context_stats(name: str, memory_turns: int):
    rows = db.get_messages(CHAT_DB, name)
    chars = sum(len(m.get("content") or "") for m in rows[-20:])
    return {"session": name, "memory": memory_turns, "file": len(rows),
            "approx_tokens": chars // 4}


def list_sessions():
    return db.list_sessions(CHAT_DB)


# --------------------------------------------------------------- training ----
def log_training(question, answer, lang="ms", kind="chat", tools_trace=None,
                 rating=None, domain=None):
    """Tambah satu baris SFT + trace ke dataset (knowledge.db)."""
    row = {
        "instruction": question,
        "input": "",
        "output": answer,
        "lang": lang,
        "kind": kind,
        "ts": int(time.time()),
        "tools": tools_trace or [],
        "rating": rating,
        "domain": domain or detect_domain(question),
    }
    db.dataset_add(KNOW_DB, row)
    return KNOW_DB


def dataset_count(consumed=None):
    return db.dataset_count(KNOW_DB, consumed)


def _match(flt, stack):
    if flt == stack:
        return True
    # legacy broad filters (web/data/...) still match member stacks
    return stack in LEGACY_GROUPS.get(flt, []) or STACK2DOMAIN.get(stack) == flt


def export_stacks(dest_dir="datasets", domain=None, min_len=2):
    """Split dataset -> datasets/<stack>.jsonl (dedupe).
    domain filter: stack name (docker) or broad domain (web/data/linux...)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows = db.dataset_rows(KNOW_DB, consumed=None, min_len=min_len)
    if not rows:
        return {}, "no dataset yet"
    flt = (domain or "").strip().lower() or None
    seen = {}
    added = {}
    for r in rows:
        q, a = r["instruction"], r["output"]
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


def merge_stacks(root=".", stack=None, eval_frac=0.0):
    """Combine datasets/<stack>.jsonl -> finetune/stacks/<stack>/dataset.jsonl (dedupe).
    Stack = domain (single tier, no broad merges).
    eval_frac>0 also writes a deterministic held-out finetune/stacks/<stack>/eval.jsonl
    (md5 hash split, stable across re-runs). Returns (train_counts, eval_counts)."""
    import hashlib
    root = Path(root)
    cfg = json.loads((root / "datasets" / "config" / "dataset.json").read_text(encoding="utf-8"))
    out, evout = {}, {}
    ethresh = int(eval_frac * 1000)
    for s in sorted(cfg["stacks"]):
        if stack and stack != s:
            continue
        src = root / "datasets" / f"{s}.jsonl"
        rows, evrows = [], []
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
                row = {"instruction": q, "input": (r.get("input") or "").strip(),
                       "output": a}
                if ethresh > 0 and int(hashlib.md5(
                        f"{q}\n{a}".encode("utf-8")).hexdigest(), 16) % 1000 < ethresh:
                    evrows.append(row)
                else:
                    rows.append(row)
        dest = root / "finetune" / "stacks" / s / "dataset.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out[s] = len(rows)
        if ethresh > 0:
            evdest = root / "finetune" / "stacks" / s / "eval.jsonl"
            with evdest.open("w", encoding="utf-8") as f:
                for r in evrows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            evout[s] = len(evrows)
    return out, evout


def _row_domain(r):
    return r.get("domain") or detect_domain(
        (r.get("instruction") or "") + " " + (r.get("output") or ""))


def export_sft(dest="finetune/stacks/general/dataset.jsonl", min_len=2, domain=None):
    rows = db.dataset_rows(KNOW_DB, consumed=None, min_len=min_len)
    if not rows:
        return 0, "no dataset yet - chat first, then export"
    seen, added = set(), 0
    dest_p = Path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    if dest_p.exists():
        for line in dest_p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                seen.add((r.get("instruction", ""), r.get("output", "")))
            except ValueError:
                pass
    with dest_p.open("a", encoding="utf-8") as fout:
        for r in rows:
            q, a = r["instruction"], r["output"]
            if domain and _row_domain(r) != domain:
                continue
            if (q, a) in seen:
                continue
            seen.add((q, a))
            fout.write(json.dumps({"instruction": q, "input": r.get("input", ""),
                                   "output": a}, ensure_ascii=False) + "\n")
            added += 1
    return added, str(dest_p)


def export_pretrain_corpus(dest="finetune/stacks/general/corpus.txt", domain=None):
    rows = db.dataset_rows(KNOW_DB, consumed=None)
    if not rows:
        return 0, "no dataset yet"
    n = 0
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fout:
        for r in rows:
            if domain and _row_domain(r) != domain:
                continue
            q, a = r.get("instruction", ""), r.get("output", "")
            fout.write(f"### User:\n{q}\n\n### Assistant:\n{a}\n\n")
            n += 1
    # also dump docs/*.md + docs/learned/*.md for continued pretraining
    # (general corpus only; learned notes first - fresh knowledge)
    if not domain:
        root = Path.cwd()
        learned = sorted((root / "docs" / "learned").glob("*.md"))
        top = sorted(root.glob("docs/*.md"))
        for md in (learned + top)[:300]:
            try:
                with out.open("a", encoding="utf-8") as fh:
                    fh.write("\n" + md.read_text(encoding="utf-8") + "\n")
            except OSError:
                pass
    return n, str(out)


# --- Delta retrain state: disimpan dalam knowledge.db (kv) ---
DELTA_MIN_ROWS = 200
DELTA_MIN_NOTES = 50
DELTA_FAIL_COOLDOWN = 24 * 3600
_KV_TRAIN = "train_state"


def train_state():
    st = db.kv_get(KNOW_DB, _KV_TRAIN) or {}
    if not isinstance(st, dict):
        st = {}
    st.setdefault("version", 2)
    st.setdefault("lines_consumed", 0)
    st.setdefault("learned_notes", 0)
    st.setdefault("last_train_ts", 0)
    st.setdefault("last_failed_ts", 0)
    st.setdefault("delta_rounds", 0)
    return st


def save_train_state(st):
    db.kv_set(KNOW_DB, _KV_TRAIN, st)
    return st


def learned_count():
    return db.learned_count(KNOW_DB)


def pending_since_train(min_rows=DELTA_MIN_ROWS, min_notes=DELTA_MIN_NOTES):
    st = train_state()
    total = db.dataset_count(KNOW_DB)
    rows = db.dataset_count(KNOW_DB, consumed=0)
    notes = max(0, learned_count() - st["learned_notes"])
    return {"rows": rows, "notes": notes, "total_lines": total,
            "ready": rows >= min_rows or notes >= min_notes, "state": st}


def mark_trained():
    st = train_state()
    db.mark_consumed(KNOW_DB)
    st["lines_consumed"] = db.dataset_count(KNOW_DB)
    st["learned_notes"] = learned_count()
    st["last_train_ts"] = int(time.time())
    st["delta_rounds"] = st.get("delta_rounds", 0) + 1
    return save_train_state(st)


def mark_failed():
    st = train_state()
    st["last_failed_ts"] = int(time.time())
    return save_train_state(st)


def build_delta_file(dest=None, replay_pct=2, replay_cap=2000):
    """Baris belum dilatih (consumed=0) + sampel replay (anti-lupa) dari all.jsonl.
    Deterministik (md5 band), dedup. Returns (Path|None, n_delta, n_replay, note)."""
    import hashlib
    rows = db.dataset_rows(KNOW_DB, consumed=0)
    if not rows:
        return None, 0, 0, "tiada baris baharu"
    seen = db._all_pairs(KNOW_DB)  # elak replay pendua (semua baris)
    replay = []
    root = Path.cwd()
    for cand in (root / "datasets" / "all.jsonl",
                 root / "drive_upload" / "DriveXcoder" / "xcoder" / "all.jsonl"):
        if not cand.is_file():
            continue
        band = max(1, int(replay_pct * 10))
        try:
            with cand.open(encoding="utf-8") as f:
                for line in f:
                    if len(replay) >= replay_cap:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    q = (r.get("instruction") or "").strip()
                    a = (r.get("output") or "").strip()
                    if len(q) < 2 or len(a) < 2 or (q, a) in seen:
                        continue
                    h = int(hashlib.md5(f"{q}\n{a}".encode("utf-8")).hexdigest(), 16)
                    if h % 1000 < band:
                        replay.append({"instruction": q,
                                       "input": (r.get("input") or ""),
                                       "output": a})
        except OSError:
            pass
        if replay:
            break
    dest = (Path(dest) if dest else
            root / "build" / "train" / f"delta-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for r in rows + replay:
            f.write(json.dumps({"instruction": r["instruction"],
                                "input": r.get("input", ""),
                                "output": r["output"]}, ensure_ascii=False) + "\n")
    return dest, len(rows), len(replay), ""


# --------------------------------------------------------- learned notes ----
def log_learned(sha256, query, title, body, url=None, file=None, ts=None):
    """Simpan nota learned ke knowledge.db. Returns True jika baharu."""
    return db.learned_add(KNOW_DB, sha256, query, title, body, url, file,
                          ts or int(time.time()))


def learned_has(sha256):
    return db.learned_has(KNOW_DB, sha256)


def list_learned(limit=5):
    return db.learned_recent(KNOW_DB, limit)


# ------------------------------------------------------------- migration ----
def _migrate_legacy():
    """Pindah data JSONL lama ke SQLite (sekali sahaja; selamat gagal)."""
    legacy_st = {}
    try:
        legacy_st = json.loads(TRAIN_STATE.read_text(encoding="utf-8"))
        if not isinstance(legacy_st, dict):
            legacy_st = {}
    except (OSError, ValueError):
        pass
    try:
        if not db.has_messages(CHAT_DB):
            for p in sorted(SESSIONS_DIR.glob("*.jsonl")):
                msgs = []
                for line in p.read_text(encoding="utf-8").splitlines():
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("role") in ("user", "assistant", "tool"):
                        msgs.append(r)
                if msgs:
                    db.add_messages(CHAT_DB, p.stem, msgs, int(time.time()))
    except Exception:
        pass
    try:
        if db.dataset_count(KNOW_DB) == 0:
            src = DATA_DIR / "training.jsonl"
            imported = 0
            if src.is_file():
                for line in src.open(encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    meta = r.get("meta") or {}
                    db.dataset_add(KNOW_DB, {
                        "instruction": r.get("instruction", ""),
                        "input": r.get("input", ""), "output": r.get("output", ""),
                        "lang": meta.get("lang"), "kind": meta.get("kind"),
                        "domain": meta.get("domain"), "rating": meta.get("rating"),
                        "tools": meta.get("tools") or [], "ts": meta.get("ts")})
                    imported += 1
            consumed = int(legacy_st.get("lines_consumed") or 0)
            if imported and consumed > 0:
                db.mark_consumed_upto(KNOW_DB, min(consumed, imported))
    except Exception:
        pass
    try:
        if db.learned_count(KNOW_DB) == 0:
            for base in (Path.cwd() / "docs" / "learned",
                         APP_DIR.parent / "docs" / "learned"):
                idx = base / ".index.jsonl"
                if not idx.is_file():
                    continue
                for line in idx.read_text(encoding="utf-8").splitlines():
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    body = ""
                    fp = base / (r.get("file") or "")
                    if fp.is_file():
                        try:
                            body = fp.read_text(encoding="utf-8", errors="replace")
                        except OSError:
                            pass
                    db.learned_add(KNOW_DB, r.get("sha256") or "", r.get("query"),
                                   (r.get("query") or "")[:120], body, r.get("url"),
                                   r.get("file"), r.get("ts") or int(time.time()))
                break
    except Exception:
        pass
    try:
        if db.kv_get(KNOW_DB, _KV_TRAIN) is None and TRAIN_STATE.is_file():
            st = json.loads(TRAIN_STATE.read_text(encoding="utf-8"))
            if isinstance(st, dict):
                db.kv_set(KNOW_DB, _KV_TRAIN, st)
    except Exception:
        pass


_migrate_legacy()
