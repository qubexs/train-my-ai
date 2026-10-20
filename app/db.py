"""SQLite storage for XCoder - stdlib only, dua pangkalan data berasingan.

  * chat.db      - mesej sesi/chat (menggantikan app/sessions/*.jsonl)
  * knowledge.db - nota learned + dataset on-progress untuk latihan
                   (menggantikan app/data/training.jsonl, train_state.json,
                    docs/learned/.index.jsonl)

JSONL kekal sebagai EKSPORT untuk pipeline latihan (lihat store.export_*).
Setiap sambungan memastikan skema wujud (CREATE IF NOT EXISTS) supaya selamat
digunakan dengan laluan fail yang berbeza (ujian boleh monkeypatch laluan).
"""
import json
import sqlite3
from pathlib import Path

_CHAT_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session TEXT NOT NULL,
  ts INTEGER NOT NULL,
  role TEXT NOT NULL,
  content TEXT,
  extra TEXT
);
CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session, id);
"""

_KNOW_SCHEMA = """
CREATE TABLE IF NOT EXISTS learned (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sha256 TEXT UNIQUE,
  query TEXT,
  title TEXT,
  body TEXT,
  url TEXT,
  file TEXT,
  ts INTEGER
);
CREATE TABLE IF NOT EXISTS dataset (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  instruction TEXT,
  input TEXT,
  output TEXT,
  lang TEXT,
  kind TEXT,
  domain TEXT,
  rating TEXT,
  tools TEXT,
  ts INTEGER,
  consumed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ds_consumed ON dataset(consumed, id);
CREATE TABLE IF NOT EXISTS kv (
  k TEXT PRIMARY KEY,
  v TEXT
);
"""


def _connect(path, schema):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cx = sqlite3.connect(str(p), timeout=10)
    cx.row_factory = sqlite3.Row
    try:
        cx.execute("PRAGMA journal_mode=WAL")
        cx.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        pass
    cx.executescript(schema)
    return cx


def chat(path):
    return _connect(path, _CHAT_SCHEMA)


def know(path):
    return _connect(path, _KNOW_SCHEMA)


# ----------------------------------------------------------------- chat ----
def add_messages(path, session, messages, ts):
    cx = chat(path)
    try:
        for m in messages:
            extra = {k: v for k, v in m.items() if k not in ("role", "content")}
            cx.execute(
                "INSERT INTO messages(session, ts, role, content, extra) VALUES(?,?,?,?,?)",
                (session, int(ts), m.get("role", "assistant"), m.get("content") or "",
                 json.dumps(extra, ensure_ascii=False) if extra else None))
        cx.commit()
    finally:
        cx.close()


def _rows_to_messages(rows):
    out = []
    for r in rows:
        m = {"role": r["role"], "content": r["content"]}
        if r["extra"]:
            try:
                m.update(json.loads(r["extra"]))
            except ValueError:
                pass
        if m.get("role") in ("user", "assistant", "tool"):
            out.append(m)
    return out


def get_messages(path, session, limit=None):
    cx = chat(path)
    try:
        if limit:
            rows = cx.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE session=? ORDER BY id DESC LIMIT ?) "
                "ORDER BY id", (session, int(limit))).fetchall()
        else:
            rows = cx.execute(
                "SELECT * FROM messages WHERE session=? ORDER BY id", (session,)).fetchall()
    finally:
        cx.close()
    return _rows_to_messages(rows)


def trim_session(path, session, keep_last):
    cx = chat(path)
    try:
        cx.execute(
            "DELETE FROM messages WHERE session=? AND id NOT IN "
            "(SELECT id FROM messages WHERE session=? ORDER BY id DESC LIMIT ?)",
            (session, session, int(keep_last)))
        cx.commit()
    finally:
        cx.close()


def clear_session(path, session):
    cx = chat(path)
    try:
        cx.execute("DELETE FROM messages WHERE session=?", (session,))
        cx.commit()
    finally:
        cx.close()


def count_session(path, session):
    cx = chat(path)
    try:
        return cx.execute("SELECT COUNT(*) FROM messages WHERE session=?",
                          (session,)).fetchone()[0]
    finally:
        cx.close()


def list_sessions(path):
    cx = chat(path)
    try:
        return [r[0] for r in cx.execute(
            "SELECT DISTINCT session FROM messages ORDER BY session")]
    finally:
        cx.close()


def has_messages(path):
    cx = chat(path)
    try:
        return cx.execute("SELECT 1 FROM messages LIMIT 1").fetchone() is not None
    finally:
        cx.close()


# ---------------------------------------------------- knowledge: learned ----
def learned_add(path, sha256, query, title, body, url, file, ts):
    cx = know(path)
    try:
        try:
            cx.execute(
                "INSERT INTO learned(sha256,query,title,body,url,file,ts) "
                "VALUES(?,?,?,?,?,?,?)",
                (sha256, query, title, body, url, file, int(ts)))
            cx.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    finally:
        cx.close()


def learned_has(path, sha256):
    cx = know(path)
    try:
        return cx.execute("SELECT 1 FROM learned WHERE sha256=?",
                          (sha256,)).fetchone() is not None
    finally:
        cx.close()


def learned_count(path):
    cx = know(path)
    try:
        return cx.execute("SELECT COUNT(*) FROM learned").fetchone()[0]
    finally:
        cx.close()


def learned_recent(path, limit=5):
    cx = know(path)
    try:
        rows = cx.execute(
            "SELECT * FROM (SELECT * FROM learned ORDER BY id DESC LIMIT ?) ORDER BY id",
            (int(limit),)).fetchall()
    finally:
        cx.close()
    return [{"sha256": r["sha256"], "query": r["query"], "title": r["title"],
             "url": r["url"], "file": r["file"], "ts": r["ts"]} for r in rows]


# ---------------------------------------------------- knowledge: dataset ----
def dataset_add(path, row):
    cx = know(path)
    try:
        cx.execute(
            "INSERT INTO dataset(instruction,input,output,lang,kind,domain,rating,tools,ts,consumed)"
            " VALUES(?,?,?,?,?,?,?,?,?,0)",
            (row.get("instruction", ""), row.get("input", ""), row.get("output", ""),
             row.get("lang"), row.get("kind"), row.get("domain"), row.get("rating"),
             json.dumps(row.get("tools") or [], ensure_ascii=False), row.get("ts")))
        cx.commit()
    finally:
        cx.close()


def dataset_rows(path, consumed=None, min_len=2):
    cx = know(path)
    try:
        if consumed is None:
            rows = cx.execute("SELECT * FROM dataset ORDER BY id").fetchall()
        else:
            rows = cx.execute("SELECT * FROM dataset WHERE consumed=? ORDER BY id",
                              (int(consumed),)).fetchall()
    finally:
        cx.close()
    out = []
    for r in rows:
        if r["rating"] == "bad":
            continue
        q = (r["instruction"] or "").strip()
        a = (r["output"] or "").strip()
        if len(q) < min_len or len(a) < min_len:
            continue
        try:
            tools = json.loads(r["tools"]) if r["tools"] else []
        except ValueError:
            tools = []
        out.append({"instruction": q, "input": r["input"] or "", "output": a,
                    "lang": r["lang"], "kind": r["kind"], "domain": r["domain"],
                    "rating": r["rating"], "tools": tools, "ts": r["ts"],
                    "consumed": r["consumed"]})
    return out


def _all_pairs(path, min_len=2):
    cx = know(path)
    try:
        rows = cx.execute("SELECT instruction, output FROM dataset").fetchall()
    finally:
        cx.close()
    pairs = set()
    for r in rows:
        q = (r["instruction"] or "").strip()
        a = (r["output"] or "").strip()
        if len(q) >= min_len and len(a) >= min_len:
            pairs.add((q, a))
    return pairs


def dataset_count(path, consumed=None):
    cx = know(path)
    try:
        if consumed is None:
            return cx.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
        return cx.execute("SELECT COUNT(*) FROM dataset WHERE consumed=?",
                          (int(consumed),)).fetchone()[0]
    finally:
        cx.close()


def mark_consumed(path):
    cx = know(path)
    try:
        cx.execute("UPDATE dataset SET consumed=1 WHERE consumed=0")
        cx.commit()
    finally:
        cx.close()


def mark_consumed_upto(path, upto_id):
    cx = know(path)
    try:
        cx.execute("UPDATE dataset SET consumed=1 WHERE id<=?", (int(upto_id),))
        cx.commit()
    finally:
        cx.close()


# ---------------------------------------------------- knowledge: kv ----
def kv_get(path, k, default=None):
    cx = know(path)
    try:
        r = cx.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    finally:
        cx.close()
    if r is None:
        return default
    try:
        return json.loads(r[0])
    except ValueError:
        return default


def kv_set(path, k, v):
    cx = know(path)
    try:
        cx.execute(
            "INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (k, json.dumps(v, ensure_ascii=False)))
        cx.commit()
    finally:
        cx.close()
