r"""HF import runner (P2) — sumber lesen-bersih -> datasets/<stack>.jsonl + ledger.

Guna .venv-gpu (perlu `datasets`, `huggingface_hub`, `pyarrow`):
  .\.venv-gpu\Scripts\python.exe datasets/scripts/import_hf.py --source oci --limit 20000
  .\.venv-gpu\Scripts\python.exe datasets/scripts/import_hf.py --source commitpackft --langs python,javascript,php
  .\.venv-gpu\Scripts\python.exe datasets/scripts/import_hf.py --source all --cap python=1000000 --cap algorithms=150000
  .\.venv-gpu\Scripts\python.exe datasets/scripts/import_hf.py --source gretel --dry-run
  .\.venv-gpu\Scripts\python.exe datasets/scripts/import_hf.py --source github-code \
      --cap javascript=1250000 --cap php=1160000 --ghc-shards 400 --ghc-workers 3

Flags:
  --source NAME   (boleh ulang; all = kesemua) oci|commitpackft|taco|gretel|magicoder-oss|magicoder-evol
  --limit N       maks baris baharu per sumber (default 0 = tiada had)
  --cap STACK=N   siling baris baharu per stack utk run ini (boleh ulang)
  --langs a,b,c   had bahasa utk commitpackft
  --no-oci-tests  jangan jana baris 'testing' dari unit_tests OCI
  --no-scrub      langkau scrub Qwen/Alibaba (JANGAN guna utk final)
  --dry-run       proses + kira + tunjuk sampel, tiada tulis

Provenance: datasets/manifests/ledger.jsonl — sha256-8(instruction\\noutput),
source, license, stack, bucket, id. Resume-safe: hash yang sudah ada dalam
ledger atau fail stack dilangkau. Lesen whitelist sahaja (NC/SA/GPL dikecualikan).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATASETS_DIR, load_config, read_rows, check_row, dump_row
from audit import scrub_hits

RAW = DATASETS_DIR / "raw"
LEDGER = DATASETS_DIR / "manifests" / "ledger.jsonl"
MIN_OUT = 10
MAX_Q = 16000
MAX_A = 16000

WHITELIST = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc",
             "cc0-1.0", "unlicense", "cc-by-4.0"}

LANG2STACK = {
    "python": "python", "javascript": "javascript", "typescript": "typescript",
    "php": "php", "shell": "bash", "bash": "bash", "sh": "bash",
    "html": "html", "css": "css", "scss": "css", "sql": "sql",
    "java": "java", "c": "c", "cpp": "cpp", "c++": "cpp", "c-sharp": "csharp",
    "csharp": "csharp", "cs": "csharp", "go": "go", "rust": "rust",
    "ruby": "ruby", "perl": "perl", "lua": "lua", "dart": "dart",
    "kotlin": "kotlin", "swift": "swift", "pascal": "pascal", "cobol": "cobol",
    "visual-basic": "visualbasic", "ts": "typescript", "js": "javascript",
}

ALGO_HINTS = ["algorithm", "complexity", "big o", "binary search", "linked list",
              "hash map", "dynamic programming", "memoization", "recursion",
              "quicksort", "mergesort", "sorting", "graph", "traversal",
              "backtracking", "greedy", "heap", "stack", "queue", "tree",
              "trie", "subarray", "subsequence", "matrix", "strings",
              "maximum", "minimum", "optimization", "permutation", "subset"]


def key8(q, a):
    return hashlib.blake2b((q + "\n" + a).encode("utf-8"), digest_size=8).hexdigest()


def load_detector():
    try:
        sys.path.insert(0, str(DATASETS_DIR.parent / "app"))
        from experts import detect_stack
        return detect_stack
    except Exception:
        return None


class Sink:
    def __init__(self, cfg, buckets, caps, dry_run=False):
        self.cfg = cfg
        self.caps = caps
        self.dry_run = dry_run
        self.counts = {}
        self.seen = {}
        self.handles = {}
        self.samples = {}
        self.total_added = 0
        self.total_skipped = 0
        self.stack2bucket = {}
        for bname, b in buckets["buckets"].items():
            for s in b["stacks"]:
                self.stack2bucket[s] = bname
        if not dry_run:
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            self.ledger = LEDGER.open("a", encoding="utf-8")

    def _load_seen(self, stack):
        if stack in self.seen:
            return self.seen[stack]
        seen = set()
        path = DATASETS_DIR / self.cfg["stacks"][stack]["file"]
        for r in read_rows(path):
            if check_row(r):
                seen.add(key8(r["instruction"].strip(), r["output"].strip()))
        self.seen[stack] = seen
        return seen

    def add(self, stack, q, inp, a, source, license, rid):
        q = (q or "").strip()
        a = (a or "").strip()
        inp = (inp or "").strip()
        if not q or not a or len(a) < MIN_OUT:
            self.total_skipped += 1
            return False
        if len(q) > MAX_Q:
            q = q[:MAX_Q].rstrip()
        if len(a) > MAX_A:
            a = a[:MAX_A].rstrip()
        cap = self.caps.get(stack)
        if cap is not None and self.counts.get(stack, 0) >= cap:
            self.total_skipped += 1
            return False
        seen = self._load_seen(stack)
        k = key8(q, a)
        if k in seen:
            self.total_skipped += 1
            return False
        if not self.dry_run and scrub_hits(q + "\n" + a):
            self.total_skipped += 1
            return False
        seen.add(k)
        self.counts[stack] = self.counts.get(stack, 0) + 1
        self.total_added += 1
        if len(self.samples.setdefault(stack, [])) < 3:
            self.samples[stack].append((q, a))
        if self.dry_run:
            return True
        f = self.handles.get(stack)
        if f is None:
            path = DATASETS_DIR / self.cfg["stacks"][stack]["file"]
            f = path.open("a", encoding="utf-8")
            self.handles[stack] = f
        f.write(dump_row({"instruction": q, "input": inp, "output": a}) + "\n")
        self.ledger.write(json.dumps(
            {"k": k, "source": source, "license": license, "stack": stack,
             "bucket": self.stack2bucket.get(stack, "?"), "id": rid,
             "ts": time.strftime("%Y-%m-%d")}, ensure_ascii=False) + "\n")
        if self.total_added % 200 == 0:
            self.flush()
        return True

    def flush(self):
        for f in self.handles.values():
            f.flush()
        if not self.dry_run:
            self.ledger.flush()

    def caps_reached(self, stacks):
        need = [s for s in stacks if s in self.caps]
        return bool(need) and all(self.counts.get(s, 0) >= self.caps[s]
                                  for s in need)

    def close(self):
        self.flush()
        for f in self.handles.values():
            f.close()
        if not self.dry_run:
            self.ledger.close()


def pick(r, *names):
    for n in names:
        v = r.get(n)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def rows_oci(sink, detect, args):
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq
    files = sorted(f for f in list_repo_files("nvidia/OpenCodeInstruct",
                                              repo_type="dataset")
                   if f.endswith(".parquet"))
    cols = ["id", "input", "output", "llm_judgement", "unit_tests",
            "tests_execution_status", "average_test_score"]
    n = 0
    for fname in files:
        if sink.caps_reached(("python", "algorithms", "testing")):
            break
        if args.limit and sink.total_added >= args.limit:
            break
        try:
            path = hf_hub_download("nvidia/OpenCodeInstruct", fname,
                                   repo_type="dataset",
                                   cache_dir=str(RAW / "hf-cache"))
            pf = pq.ParquetFile(path)
        except Exception as e:
            print(f"  oci/{fname}: skip ({e})")
            continue
        for batch in pf.iter_batches(batch_size=2000, columns=cols):
            if sink.caps_reached(("python", "algorithms", "testing")):
                break
            for r in batch.to_pylist():
                n += 1
                try:
                    score = float(r.get("average_test_score") or 0)
                except (TypeError, ValueError):
                    score = 0.0
                if score < 0.8:
                    continue
                judge = r.get("llm_judgement") or ""
                if isinstance(judge, str):
                    try:
                        judge = json.loads(judge)
                    except ValueError:
                        judge = {}
                req = (judge.get("requirement_conformance") or {}).get("score", 0)
                try:
                    req = float(req)
                except (TypeError, ValueError):
                    req = 0
                if req < 4:
                    continue
                q = pick(r, "input")
                a = pick(r, "output")
                stack = "python"
                if detect and detect(q[:600]) == "algorithms":
                    stack = "algorithms"
                elif any(h in q[:600].lower() for h in ALGO_HINTS):
                    stack = "algorithms"
                sink.add(stack, q, "", a, "nvidia/OpenCodeInstruct", "cc-by-4.0",
                         str(r.get("id")))
                if args.oci_tests:
                    tests = r.get("unit_tests") or []
                    statuses = r.get("tests_execution_status") or []
                    if isinstance(tests, str):
                        try:
                            tests = json.loads(tests)
                        except ValueError:
                            tests = []
                    if isinstance(statuses, str):
                        try:
                            statuses = json.loads(statuses)
                        except ValueError:
                            statuses = []
                    if tests and statuses and all(s == "pass" for s in statuses):
                        body = "\n".join(t.strip() for t in tests
                                         if isinstance(t, str))
                        if body:
                            tq = ("Write assert-style unit tests for this task:"
                                  "\n\n" + q[:1200])
                            sink.add("testing", tq, "", body,
                                     "nvidia/OpenCodeInstruct", "cc-by-4.0",
                                     str(r.get("id")) + "#tests")
            if n % 50000 < 2000:
                sink.flush()
            if n % 100000 == 0:
                print(f"  oci: scanned={n} added={sink.total_added} "
                      f"skipped={sink.total_skipped}")


def rows_commitpackft(sink, detect, args):
    from huggingface_hub import hf_hub_download
    langs = [l.strip() for l in (args.langs or "python,javascript,typescript,php,shell,html,css,sql,java,go,rust,ruby,c,c#,c++").split(",")]
    for lang in langs:
        try:
            path = hf_hub_download("bigcode/commitpackft", f"data/{lang}/data.jsonl",
                                   repo_type="dataset",
                                   cache_dir=str(RAW / "hf-cache"))
        except Exception as e:
            print(f"  commitpackft/{lang}: skip ({e})")
            continue
        n = 0
        for line in open(path, encoding="utf-8"):
            if args.limit and sink.total_added >= args.limit:
                return
            n += 1
            if n % 20000 == 0:
                print(f"  commitpackft/{lang}: scanned={n} added={sink.total_added}")
            try:
                r = json.loads(line)
            except ValueError:
                continue
            lic = (r.get("license") or "").lower()
            if lic not in WHITELIST:
                continue
            msg = (r.get("message") or r.get("subject") or "").strip()
            old = r.get("old_contents") or ""
            new = r.get("new_contents") or ""
            if not (10 <= len(msg) <= 400) or len(old) > 8000 or len(new) > 12000:
                continue
            if len(new) < 20 or new == old or "\x00" in new:
                continue
            low = msg.lower()
            if any(w in low for w in ("fix", "bug", "error", "issue", "crash")):
                stack = "debugging"
            elif any(w in low for w in ("refactor", "rename", "cleanup", "clean up",
                                        "simplify", "reorganize", "reorganise")):
                stack = "refactoring"
            else:
                stack = LANG2STACK.get(lang)
            if not stack or stack not in sink.cfg["stacks"]:
                continue
            sink.add(stack, msg, old, new, "bigcode/commitpackft", lic,
                     str(r.get("commit")))


def rows_taco(sink, detect, args):
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq
    files = [f for f in list_repo_files("BAAI/TACO", repo_type="dataset")
             if f.endswith(".parquet") and f.startswith("ALL/")]
    if not files:
        print("  taco: tiada parquet dalam repo")
        return
    for fname in files:
        if args.limit and sink.total_added >= args.limit:
            return
        if sink.caps_reached(("algorithms",)):
            return
        try:
            path = hf_hub_download("BAAI/TACO", fname, repo_type="dataset",
                                   cache_dir=str(RAW / "hf-cache"))
            t = pq.read_table(path, columns=["question", "solutions", "name"])
        except Exception as e:
            print(f"  taco/{fname}: skip ({e})")
            continue
        for r in t.to_pylist():
            if args.limit and sink.total_added >= args.limit:
                return
            q = (r.get("question") or "").strip()
            sols = r.get("solutions") or "[]"
            try:
                sols = json.loads(sols) if isinstance(sols, str) else sols
            except ValueError:
                continue
            sols = [s for s in sols if isinstance(s, str) and s.strip()]
            if not sols or not q:
                continue
            best = min(sols, key=len)
            sink.add("algorithms", "Solve this programming problem:\n\n" + q[:6000],
                     "", best, "BAAI/TACO", "apache-2.0", r.get("name"))


def rows_gretel(sink, detect, args):
    from datasets import load_dataset
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train", streaming=True)
    for r in ds:
        if args.limit and sink.total_added >= args.limit:
            break
        sink.add("sql", pick(r, "sql_prompt"), pick(r, "sql_context"),
                 pick(r, "sql"), "gretelai/synthetic_text_to_sql", "apache-2.0",
                 str(r.get("id")))


def rows_magicoder(sink, detect, args, name, license_):
    from datasets import load_dataset
    ds = load_dataset(name, split="train", streaming=True)
    for r in ds:
        if args.limit and sink.total_added >= args.limit:
            break
        q = pick(r, "problem", "instruction")
        a = pick(r, "solution", "response", "output")
        lang = (r.get("lang") or "").strip().lower()
        stack = LANG2STACK.get(lang)
        if not stack and detect:
            stack = detect(q[:600])
        if not stack or stack not in sink.cfg["stacks"]:
            stack = "general"
        sink.add(stack, q, "", a, name, license_, str(r.get("index") or r.get("id") or ""))


GHC_EXT2STACK = {
    ".js": "javascript", ".ts": "typescript", ".tsx": "react",
    ".php": "php", ".php3": "php", ".php4": "php", ".php5": "php",
    ".phps": "php", ".phpt": "php", ".py": "python",
    ".html": "html", ".htm": "html", ".css": "css",
    ".md": "docs", ".markdown": "docs",
    ".sh": "bash", ".bash": "bash", ".command": "bash", ".zsh": "bash",
    ".sql": "sql", ".pl": "perl", ".pm": "perl", ".pm6": "perl",
    ".lua": "lua", ".rb": "ruby", ".rs": "rust", ".go": "go",
    ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".cc": "cpp", ".hh": "cpp", ".cs": "csharp", ".vb": "visualbasic",
}
GHC_LABEL = {"javascript": "JavaScript", "typescript": "TypeScript",
             "react": "React TSX", "php": "PHP", "python": "Python",
             "html": "HTML", "css": "CSS", "docs": "Markdown",
             "bash": "shell script", "sql": "SQL", "docker": "Dockerfile",
             "c": "C", "cpp": "C++", "csharp": "C#", "go": "Go", "rust": "Rust",
             "ruby": "Ruby", "perl": "Perl", "lua": "Lua", "scala": "Scala",
             "julia": "Julia", "haskell": "Haskell", "visualbasic": "Visual Basic",
             "git": "Git config", "linux": "Linux config", "mysql": "MySQL",
             "postgresql": "PostgreSQL", "sqlite": "SQLite", "mariadb": "MariaDB",
             "mongodb": "MongoDB", "redis": "Redis", "flask": "Flask",
             "django": "Django", "nodejs": "Node.js", "spring": "Spring",
             "api": "API"}
GHC_SKIP_DIRS = ("node_modules/", "/vendor/", "vendor/", "/dist/", "/build/",
                 "/third_party/", "/external/", "/bower_components/")
GHC_TEST_MARK = ("/test", "/tests", ".test.", ".spec.", "_test.", "test_",
                 "/spec", "/specs")
GHC_COMMENT_PREFIXES = ("//", "#", "/*", "*", "<!--", "--", ";", "%%")
GHC_INTENT_SKIP = ("copyright", "licence", "license", "spdx",
                   "all rights reserved", "permission is hereby",
                   "distributed under", "http://", "https://", "@author",
                   "generated", "eslint", "tslint", "jshint", "noqa",
                   "file:", "module:", "created by", "modified by",
                   "revision:", "coding:")
GHC_TEST_STACKS = ("javascript", "typescript", "react", "php", "python")
GHC_BASE2STACK = {".gitignore": "git", ".gitattributes": "git",
                  ".gitmodules": "git", ".editorconfig": "git",
                  ".dockerignore": "docker"}
GHC_DEVEXT = (".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf",
              ".service", ".timer")
GHC_DEVMARK = ("apiversion:", "kind:", "ansible", "systemd", "[unit]",
               "nginx", "helm", "kubernetes", "kustomize", "docker:")


def ghc_content_stack(stack, low):
    if stack in ("python", "api", "flask", "django"):
        if "from flask" in low or "import flask" in low:
            return "flask"
        if "from django" in low or "import django" in low:
            return "django"
        if "fastapi" in low or "from rest_framework" in low:
            return "api"
    if stack in ("javascript", "typescript", "react", "nodejs"):
        if any(m in low for m in ("require('express')", 'require("express")',
                                  "from 'express'", 'from "express"',
                                  "from 'koa'", 'from "koa"', "@nestjs")):
            return "nodejs"
    if stack == "java" and "org.springframework" in low:
        return "spring"
    if stack == "sql":
        if "engine=innodb" in low or "auto_increment" in low:
            return "mysql"
        if "engine=myisam" in low:
            return "mariadb"
        if any(m in low for m in ("plpgsql", "pg_catalog", "create extension",
                                  "serial primary key", "serial not null")):
            return "postgresql"
        if "autoincrement" in low or "pragma " in low:
            return "sqlite"
    if stack in ("javascript", "typescript", "nodejs", "python", "api"):
        if "mongoose" in low or "mongoclient" in low or "pymongo" in low:
            return "mongodb"
        if "redis" in low and any(m in low for m in (
                "createclient", "import redis", "from redis", "redisclient",
                "require('redis')", 'require("redis")')):
            return "redis"
    return stack


def ghc_intent(content):
    parts = []
    for ln in content.splitlines()[:40]:
        s = ln.strip()
        if not s or s.startswith(("#!", "<?", "<!")):
            continue
        for pref in GHC_COMMENT_PREFIXES:
            if s.startswith(pref):
                text = s[len(pref):].strip().strip("*/->").strip()
                low = text.lower()
                if len(text) >= 15 and not any(k in low for k in GHC_INTENT_SKIP):
                    parts.append(text[:110])
                break
        if len(parts) >= 2:
            break
    return " ".join(parts)[:160]


def ghc_to_row(r):
    lic = (r.get("license") or "").lower()
    if lic not in WHITELIST:
        return None
    p = r.get("path") or ""
    low = p.lower()
    base = p.rsplit("/", 1)[-1]
    content = r.get("content") or ""
    if content.startswith("\ufeff"):
        content = content[1:]
    if any(k in low for k in GHC_SKIP_DIRS) or ".min." in low \
            or low.endswith(".map"):
        return None
    if base in GHC_BASE2STACK:
        stack = GHC_BASE2STACK[base]
    elif base == "Dockerfile" or base.lower().endswith(".dockerfile"):
        stack = "docker"
    elif base == "Makefile":
        stack = "bash"
    else:
        ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
        stack = GHC_EXT2STACK.get("." + ext)
        if not stack and ("." + ext) in GHC_DEVEXT:
            lowc = content[:800].lower()
            if any(m in lowc for m in GHC_DEVMARK):
                stack = "linux"
    if not stack:
        return None
    stack = ghc_content_stack(stack, content[:1200].lower())
    if any(k in low for k in GHC_TEST_MARK) and stack in GHC_TEST_STACKS:
        stack = "testing"
    if "\x00" in content or content.count("\n") < 3:
        return None
    output = content.rstrip()
    intent = ghc_intent(content)
    label = GHC_LABEL.get(stack, stack)
    if stack == "git":
        instr = f"Write a `{base}` file for a repository"
        if intent:
            instr += f": {intent}"
    elif stack == "docs":
        instr = f"Tulis dokumentasi: {intent or base}"
    elif intent:
        instr = f"Write {label} code for `{base}`: {intent}"
    else:
        instr = f"Write a complete {label} file named `{base}`"
    return (stack, instr, output, lic, p[:120])


def rows_github_code(sink, detect, args):
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    import gc
    import os
    import pyarrow.parquet as pq

    total_shards = 1126
    span = args.ghc_shards or total_shards
    all_shards = [f"data/train-{i:05d}-of-{total_shards:05d}.parquet"
                  for i in range(total_shards)]
    shards = all_shards[args.ghc_start:args.ghc_start + span]
    maxc = args.ghc_max_chars
    capcheck = sorted(sink.caps) or None
    stop = {"flag": False}
    valid = set(sink.cfg["stacks"])
    url_base = ("https://huggingface.co/datasets/codeparrot/github-code"
                "/resolve/main/")
    tmp_dir = RAW / "ghc"

    def fetch(fname):
        import subprocess
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dest = tmp_dir / fname.replace("/", "_")
        if not dest.exists():
            part = Path(str(dest) + ".part")
            r = subprocess.run(
                ["curl.exe", "-fL", "-sS", "--retry", "6", "--retry-delay", "3",
                 "--retry-all-errors", "--max-time", "1200", "-o", str(part),
                 url_base + fname],
                capture_output=True, text=True)
            if r.returncode != 0 or not part.exists():
                part.unlink(missing_ok=True)
                raise RuntimeError(f"curl rc={r.returncode} {r.stderr[:200]}")
            os.replace(part, dest)
        batch_iter = pq.ParquetFile(dest).iter_batches(
            batch_size=4096, columns=["path", "license", "size", "content"])
        rows = []
        for batch in batch_iter:
            for r in batch.to_pylist():
                try:
                    size = int(r.get("size") or 0)
                except (TypeError, ValueError):
                    continue
                if size < 100 or size > maxc:
                    continue
                row = ghc_to_row(r)
                if row and row[0] in valid:
                    rows.append(row)
            del batch
        del batch_iter
        return dest, rows

    def unlink(path):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.ghc_workers)) as ex:
        it = iter(shards)
        inflight = {}
        for _ in range(max(1, args.ghc_workers)):
            try:
                f = next(it)
            except StopIteration:
                break
            inflight[ex.submit(fetch, f)] = f
        while inflight:
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                fname = inflight.pop(fut)
                done += 1
                try:
                    path, rows = fut.result()
                    for stack, instr, output, lic, rid in rows:
                        sink.add(stack, instr, "", output,
                                 "codeparrot/github-code", lic, rid)
                    unlink(path)
                except Exception as e:
                    import traceback
                    where = " | ".join(
                        f"{f.name}:{f.lineno}"
                        for f in traceback.extract_tb(e.__traceback__)[-3:])
                    print(f"  ghc/{fname}: skip ({e!r}) at {where}")
                if stop["flag"]:
                    continue
                if capcheck and sink.caps_reached(capcheck):
                    stop["flag"] = True
                if args.limit and sink.total_added >= args.limit:
                    stop["flag"] = True
                if done % 5 == 0 or stop["flag"]:
                    print(f"  ghc {done}/{len(shards)} shards: "
                          f"added={sink.total_added} skipped={sink.total_skipped}")
                if not stop["flag"]:
                    try:
                        nxt = next(it)
                        inflight[ex.submit(fetch, nxt)] = nxt
                    except StopIteration:
                        pass
            gc.collect()
    if stop["flag"]:
        print(f"  ghc: berhenti awal — caps penuh (shards done={done})")


def rows_sqlctx(sink, detect, args):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("b-mc2/sql-create-context",
                           "sql_create_context_v4.json",
                           repo_type="dataset",
                           cache_dir=str(RAW / "hf-cache"))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for r in data:
        q = (r.get("question") or "").strip()
        ctx = (r.get("context") or "").strip()
        a = (r.get("answer") or "").strip()
        if not q or not a:
            continue
        instr = f"Given this schema:\n{ctx}\n\n{q}" if ctx else q
        sink.add("sql", instr, "", a, "b-mc2/sql-create-context",
                 "cc-by-4.0", q[:60])


def rows_gretel_sql(sink, detect, args):
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq
    files = sorted(f for f in list_repo_files("gretelai/synthetic_text_to_sql",
                                              repo_type="dataset")
                   if f.endswith(".parquet"))
    for fname in files:
        try:
            path = hf_hub_download("gretelai/synthetic_text_to_sql", fname,
                                   repo_type="dataset",
                                   cache_dir=str(RAW / "hf-cache"))
        except Exception as e:
            print(f"  gretelsql/{fname}: skip ({e})")
            continue
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1000,
                                     columns=["id", "sql_prompt",
                                              "sql_context", "sql"]):
            for r in batch.to_pylist():
                q = (r.get("sql_prompt") or "").strip()
                ctx = (r.get("sql_context") or "").strip()
                a = (r.get("sql") or "").strip()
                if not q or not a:
                    continue
                instr = f"{q}\n\nSchema:\n{ctx}" if ctx else q
                sink.add("sql", instr, "", a, "gretelai/synthetic_text_to_sql",
                         "apache-2.0", str(r.get("id")))


def rows_sqale(sink, detect, args):
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq
    files = sorted(f for f in list_repo_files("trl-lab/SQaLe-text-to-SQL-dataset",
                                              repo_type="dataset")
                   if f.endswith(".parquet"))
    for fname in files:
        if args.limit and sink.total_added >= args.limit:
            break
        try:
            path = hf_hub_download("trl-lab/SQaLe-text-to-SQL-dataset", fname,
                                   repo_type="dataset",
                                   cache_dir=str(RAW / "hf-cache"))
        except Exception as e:
            print(f"  sqale/{fname}: skip ({e})")
            continue
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1000,
                                     columns=["schema", "question", "query"]):
            for r in batch.to_pylist():
                q = (r.get("question") or "").strip()
                schema = (r.get("schema") or "").strip()
                a = (r.get("query") or "").strip()
                if not q or not a:
                    continue
                instr = f"{q}\n\nSchema:\n{schema}" if schema else q
                sink.add("sql", instr, "", a,
                         "trl-lab/SQaLe-text-to-SQL-dataset", "mit", q[:60])


def rows_apigen(sink, detect, args):
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq
    files = sorted(f for f in list_repo_files("argilla/apigen-function-calling",
                                              repo_type="dataset")
                   if f.endswith(".parquet"))
    for fname in files:
        try:
            path = hf_hub_download("argilla/apigen-function-calling", fname,
                                   repo_type="dataset",
                                   cache_dir=str(RAW / "hf-cache"))
        except Exception as e:
            print(f"  apigen/{fname}: skip ({e})")
            continue
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=500,
                                     columns=["query", "tools", "answers"]):
            for r in batch.to_pylist():
                q = (r.get("query") or "").strip()
                tools = (r.get("tools") or "").strip()
                a = (r.get("answers") or "").strip()
                if not q or not a:
                    continue
                if tools:
                    q = f"{q}\n\nAvailable tools:\n{tools[:1500]}"
                sink.add("api", q, "", a[:6000],
                         "argilla/apigen-function-calling", "cc-by-4.0",
                         q[:60])


def rows_glaive_fc(sink, detect, args):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("glaiveai/glaive-function-calling-v2",
                           "glaive-function-calling-v2.json",
                           repo_type="dataset",
                           cache_dir=str(RAW / "hf-cache"))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for r in data:
        system = (r.get("system") or "").strip()
        chat = r.get("chat") or ""
        if "ASSISTANT:" not in chat:
            continue
        user = chat.split("ASSISTANT:", 1)[0].split("USER:")[-1].strip()
        a = chat.split("ASSISTANT:", 1)[1].strip()
        if not user or not a:
            continue
        q = f"{system[:1200]}\n\n{user}" if system else user
        sink.add("api", q, "", a[:6000],
                 "glaiveai/glaive-function-calling-v2", "apache-2.0",
                 user[:60])


def rows_xlam(sink, detect, args):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("lockon/xlam-function-calling-60k",
                           "xlam_function_calling_60k.json",
                           repo_type="dataset",
                           cache_dir=str(RAW / "hf-cache"))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for r in data:
        q = (r.get("query") or "").strip()
        a = r.get("answers")
        tools = r.get("tools")
        if not q:
            continue
        if isinstance(a, (list, dict)):
            a = json.dumps(a, ensure_ascii=False)
        a = (a or "").strip()
        if not a:
            continue
        if tools:
            tl = tools if isinstance(tools, str) else json.dumps(
                tools, ensure_ascii=False)
            q = f"{q}\n\nAvailable tools:\n{tl[:1500]}"
        sink.add("api", q, "", a[:6000], "lockon/xlam-function-calling-60k",
                 "cc-by-4.0", q[:60])


HANDLERS = {
    "oci": (rows_oci, None),
    "commitpackft": (rows_commitpackft, None),
    "taco": (rows_taco, None),
    "gretel": (rows_gretel, None),
    "github-code": (rows_github_code, None),
    "sql-create-context": (rows_sqlctx, None),
    "gretel-sql": (rows_gretel_sql, None),
    "sqale": (rows_sqale, None),
    "apigen": (rows_apigen, None),
    "glaive-fc": (rows_glaive_fc, None),
    "xlam": (rows_xlam, None),
    "magicoder-oss": (lambda s, d, a: rows_magicoder(s, d, a, "ise-uiuc/Magicoder-OSS-Instruct-75K", "mit"), None),
    "magicoder-evol": (lambda s, d, a: rows_magicoder(s, d, a, "ise-uiuc/Magicoder-Evol-Instruct-110K", "apache-2.0"), None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap", action="append", default=[])
    ap.add_argument("--langs", default="")
    ap.add_argument("--no-oci-tests", dest="oci_tests", action="store_false")
    ap.add_argument("--no-scrub", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ghc-workers", type=int, default=3)
    ap.add_argument("--ghc-max-chars", type=int, default=10000)
    ap.add_argument("--ghc-shards", type=int, default=0,
                    help="had atas shard github-code (0=semua 1126)")
    ap.add_argument("--ghc-start", type=int, default=0,
                    help="skip first N shards (resume)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    sources = args.source or ["all"]
    if "all" in sources:
        sources = ["oci", "commitpackft", "taco", "gretel",
                   "magicoder-oss", "magicoder-evol"]
    caps = {}
    for c in args.cap:
        stack, _, n = c.partition("=")
        caps[stack] = int(n)

    cfg = load_config()
    buckets = json.loads((DATASETS_DIR / "config" / "buckets.json")
                         .read_text(encoding="utf-8"))
    detect = load_detector()
    if detect is None:
        print("WARN: detect_stack tidak tersedia (stack fallback mudah)")
    sink = Sink(cfg, buckets, caps, dry_run=args.dry_run)
    t0 = time.time()
    for src in sources:
        handler = HANDLERS.get(src)
        if not handler:
            print(f"unknown source: {src}")
            continue
        print(f"== {src} (limit={args.limit or 'unlimited'}) ==")
        handler[0](sink, detect, args)
        print(f"   -> added={sink.total_added} skipped={sink.total_skipped} "
              f"({time.time() - t0:.0f}s)")
    sink.close()
    print("-" * 50)
    for stack, n in sorted(sink.counts.items(), key=lambda kv: -kv[1]):
        print(f"  {stack:<12} +{n}")
    print(f"TOTAL added={sink.total_added} skipped={sink.total_skipped} "
          f"({time.time() - t0:.0f}s)")
    if args.dry_run:
        for stack, ss in sorted(sink.samples.items()):
            print(f"\n--- sample {stack} ---")
            for q, a in ss[:2]:
                print("Q:", q[:130].replace("\n", " "))
                print("A:", a[:130].replace("\n", " "))


if __name__ == "__main__":
    main()
