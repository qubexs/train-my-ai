"""XCoder - opencode-style agentic CLI (stdlib only).

Backends: llama-cli subprocess (default, 0 RAM idle) | --backend lmstudio (http://localhost:1234/v1)
Tools: list/read/write/edit/run/bash/rag with workspace jail + confirm.
Sessions: app/sessions/<name>.jsonl | Training flywheel: app/data/training.jsonl
Train: `py app/ezcodex.py train --mode sft|pretrain --export-only` (+ Colab scripts in finetune/)
Build exe: `py app/build_exe.py` (PyInstaller, win/linux/macos)

Compat: old flags --lang/--name/--setup still work. Chat is now agentic.
"""
import argparse
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

APP_DIR = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
           else Path(__file__).resolve().parent)
sys.path.insert(0, str(APP_DIR))

__version__ = "0.4.22"

LLAMA_TAG = "b11491"
LLAMA_BASE = f"https://github.com/ggerganov/llama.cpp/releases/download/{LLAMA_TAG}"
HF_MODEL_URL = ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
                "qwen2.5-0.5b-instruct-q4_k_m.gguf")
MODEL_SIZE = 491400032

from backends import LlamaCliBackend, LmStudioBackend
from agent import run_agent, sanitize
from experts import (GENERAL, add as add_model, detect as detect_domain,
                     detect_stack, fmt_size, resolve as resolve_model,
                     scan as scan_models,
                     STACK2DOMAIN, infer_name_domains, resolve_layout)
from store import load_session, save_turn, list_sessions, log_training, export_sft, export_pretrain_corpus
from store import clear_session, context_stats

BIN_DIR, MODELS_DIR = resolve_layout(APP_DIR)


def asset_name():
    s, m = platform.system(), platform.machine().lower()
    if s == "Windows":
        return f"llama-{LLAMA_TAG}-bin-win-cpu-x64.zip"
    if s == "Linux":
        return f"llama-{LLAMA_TAG}-bin-ubuntu-x64.tar.gz"
    if s == "Darwin":
        arm = "arm64" in m or m == "aarch64"
        return f"llama-{LLAMA_TAG}-bin-macos-{'arm64' if arm else 'x64'}.tar.gz"
    raise SystemExit(f"OS tidak disokong: {s}")


def download(url, dest, expect=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ezcodex-standalone"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0)) or expect or 1
        got, t0 = 0, time.time()
        while True:
            b = r.read(1024 * 1024)
            if not b:
                break
            f.write(b)
            got += len(b)
            el = max(0.1, time.time() - t0)
            print(f"\r  {got / 1e6:.0f}/{total / 1e6:.0f} MB ({got / el / 1e6:.1f} MB/s)",
                  end="", flush=True)
    print()


def ensure_cli():
    name = "llama-cli.exe" if platform.system() == "Windows" else "llama-cli"
    hit = list(BIN_DIR.rglob(name))
    if hit:
        return hit[0]
    asset = asset_name()
    print(f"[ezcodex] downloading engine {asset} ...")
    with tempfile.TemporaryDirectory() as tmp:
        arc = Path(tmp) / asset
        download(f"{LLAMA_BASE}/{asset}", arc)
        if asset.endswith(".zip"):
            with zipfile.ZipFile(arc) as z:
                z.extractall(BIN_DIR)
        else:
            with tarfile.open(arc, "r:gz") as t:
                t.extractall(BIN_DIR)
    hit = list(BIN_DIR.rglob(name))
    if not hit:
        raise SystemExit("llama-cli not found after extract")
    if platform.system() != "Windows":
        hit[0].chmod(hit[0].stat().st_mode | stat.S_IEXEC)
    return hit[0]


def ensure_model():
    dest = MODELS_DIR / HF_MODEL_URL.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size == MODEL_SIZE:
        return dest
    print("[ezcodex] downloading model (~491MB, once only) ...")
    download(HF_MODEL_URL, dest, expect=MODEL_SIZE)
    return dest


HELP = """Commands:
/help                 this help
/lang ms|en           switch language (clears turn cache, keeps session file)
/tools on|off         enable/disable agentic tool loop
/plan <tugas>         numbered build plan (maks 6 langkah), then build file by file
(fikiran AI sentiasa ditunjuk automatik sebelum jawapan — tiada arahan khas)
/backend llama|lmstudio|server  tukar enjin inferens
/server               status resident llama-server (backend server sahaja)
/model <nama|path|id>  tukar model pakar (satu aktif pada satu masa)
/model add <url|fail> [--name N] [--domains a,b]  daftar model baharu
/model import [fail] [--from dir]  auto-import GGUF terbaru + aktifkan
/models               senarai semua model pakar + domain
/models check         audit GGUF (hash, sah, duplikat, registry)
/consult <soalan>     runding 2 pakar, general sintesis jawapan penuh
/route on|off         auto-tukar pakar ikut domain (lalai: sentiasa on)
/session <name>       switch session file (sessions/<name>.jsonl)
/sessions             list sessions
/context              show context usage of current session
/clear                clear in-memory history AND session file
/good | /bad          rate last answer -> training log (flywheel)
/train-log            show training.jsonl count
/export sft|corpus    pecah ke datasets/<stack> + gabung finetune/ (atau corpus.txt)
/cwd                  print workspace root
/keluar|/exit|/quit   exit
Inline tool (when /tools on): model may emit ```tool:read docs/js-basics.md``` etc.
Tools: list read write edit run bash rag (workspace-jailed, bash asks confirm).
"""


def cuda_dir():
    return APP_DIR / "bin-cuda"


def ensure_cuda():
    """Download llama.cpp CUDA build (GTX 1070+) into bin-cuda/. Returns dir."""
    import json as _json
    d = cuda_dir()
    if list(d.rglob("llama-server.exe")) + list(d.rglob("llama-server")):
        return d
    api = f"https://api.github.com/repos/ggerganov/llama.cpp/releases/tags/{LLAMA_TAG}"
    req = urllib.request.Request(api, headers={"User-Agent": "ezcodex-standalone"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rel = _json.loads(r.read().decode())
    names = [a["name"] for a in rel.get("assets", [])]
    lower = [(n, n.lower()) for n in names]
    pack = next((n for n, l in lower if l.startswith("llama-") and "-cuda-12" in l
                 and "win" in l and "x64" in l and l.endswith(".zip")), None)
    # matching CUDA runtime (cublas/cudart DLLs ship separately)
    rt = next((n for n, l in lower if l.startswith("cudart-llama-") and "-cuda-12" in l
               and "win" in l and "x64" in l and l.endswith(".zip")), None)
    if not pack:
        raise SystemExit(f"tiada asset llama CUDA untuk {LLAMA_TAG}.")
    print(f"[ezcodex] downloading CUDA engine {pack} ...")
    with tempfile.TemporaryDirectory() as tmp:
        for hit in [x for x in (pack, rt) if x]:
            arc = Path(tmp) / hit
            if not arc.exists():
                download(f"{LLAMA_BASE}/{hit}", arc)
            with zipfile.ZipFile(arc) as z:
                z.extractall(d)
    if not (list(d.rglob("llama-server.exe")) + list(d.rglob("llama-server"))):
        raise SystemExit("llama-server tidak ditemui selepas extract CUDA")
    return d


def build_parser():
    ap = argparse.ArgumentParser(description="XCoder agentic CLI (opencode-style, stdlib only)")
    ap.add_argument("--lang", default="ms", choices=["ms", "en"])
    ap.add_argument("--name", default="XCoder")
    ap.add_argument("--setup", action="store_true", help="download engine + model only, then exit")
    ap.add_argument("--backend", default="llama", choices=["llama", "lmstudio", "server", "auto"])
    ap.add_argument("--lm-url", default="http://localhost:1234/v1")
    ap.add_argument("--lm-model", default="ezcodex-0.5b")
    ap.add_argument("--port", type=int, default=8080, help="llama-server port (backend server)")
    ap.add_argument("--threads", type=int, default=4, help="llama-server CPU threads")
    ap.add_argument("--ctx", type=int, default=4096, help="llama-server context size")
    ap.add_argument("--idle-timeout", type=int, default=180, help="server unload after N idle secs (0=never)")
    ap.add_argument("--gpu", default="off", choices=["off", "cuda"],
                    help="server/llama GPU offload (cuda = GTX 1070+, bin-cuda)")
    ap.add_argument("--model", default="", help="override GGUF path (llama) or model id (lmstudio)")
    ap.add_argument("--session", default="default")
    ap.add_argument("--no-tools", action="store_true", help="disable agentic tool loop")
    ap.add_argument("--allow-all", action="store_true", help="skip bash confirm (dangerous)")
    ap.add_argument("--max-tokens", type=int, default=500, help="max tokens per answer (code: 800-1000)")
    ap.add_argument("--autoroute", default=True, action=argparse.BooleanOptionalAction,
                    help="auto-switch expert model by question domain (default on)")
    ap.add_argument("--tui", default=True, action=argparse.BooleanOptionalAction,
                    help="fullscreen opencode-style view (default on when TTY)")
    ap.add_argument("--once", default="", help="one-shot prompt, print answer and exit (for scripts)")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train", help="training flywheel: export SFT dataset / pretrain corpus")
    t.add_argument("--mode", choices=["sft", "pretrain", "both"], default="both")
    t.add_argument("--domain", default="", help="export domain sahaja cth: linux,docker,web,data,php,python")
    t.add_argument("--export-only", action="store_true", help="only export, do not print Colab next-steps")
    return ap


_MANAGER = None


def get_manager(args):
    global _MANAGER
    if _MANAGER is None:
        from server import ModelManager
        flavor = "cuda" if args.gpu == "cuda" else "cpu"
        bindir = ensure_cuda() if flavor == "cuda" else BIN_DIR
        _MANAGER = ModelManager(bindir, port=args.port, ctx=args.ctx,
                                threads=args.threads, idle_timeout=args.idle_timeout,
                                flavor=flavor, ngl=99 if flavor == "cuda" else 0)
    return _MANAGER


def make_backend(args, cli_path=None, model_path=None):
    if args.backend == "lmstudio":
        return LmStudioBackend(args.lm_url, args.model or args.lm_model,
                               max_tokens=args.max_tokens)
    if args.backend == "server":
        from backends import ServerBackend
        model = Path(args.model) if args.model else (model_path or ensure_model())
        mgr = get_manager(args)
        already, secs = mgr.load(model)
        if not already:
            print(f"[server] model dimuat dalam {secs:.1f}s (resident, unload selepas {args.idle_timeout}s idle)")
        return ServerBackend(mgr, model="xcoder", max_tokens=args.max_tokens)
    cli = cli_path or ensure_cli()
    model = Path(args.model) if args.model else (model_path or ensure_model())
    return LlamaCliBackend(cli, model, n_predict=args.max_tokens)


def match_cur(model_path_str):
    for m in scan_models(MODELS_DIR):
        if m.get("file") and str(model_path_str or "").replace("\\", "/").endswith(m["file"]):
            return {"name": m["name"], "domains": m.get("domains", [GENERAL])}
    return {"name": "base", "domains": [GENERAL]}


def pick_expert(domain, stack):
    """Pilih pakar terbaik: utamakan nama sepadan stack (xcoder-nodejs untuk
    soalan nodejs), elak pakar rawak dalam domain sama."""
    cands = [m for m in scan_models(MODELS_DIR)
             if m.get("file") and (MODELS_DIR / m["file"]).is_file()
             and domain in m.get("domains", [])]
    if not cands:
        return None
    if stack != GENERAL:
        for m in cands:
            if stack in m["name"].lower():
                return m
    return cands[0]


def expert_context(cur):
    names = ", ".join(f"{m['name']} [{','.join(m.get('domains', []))}]"
                      for m in scan_models(MODELS_DIR))
    return (f"Anda sedang berjalan sebagai {cur['name']} (kepakaran: {','.join(cur.get('domains', []))}). "
            f"Pakar lain yang wujud: {names}. "
            f"Bantu pengguna TERUS dengan jawapan dan kod — anda sendiri pakarnya, "
            f"tugas anda menyiapkan kerja, bukan mengarah ke orang lain. "
            f"Jika ditanya senarai model, berikan nama dari senarai ini sahaja.")


SELF_KEYS = ("pakar", "expert", "kepakaran", "senarai", "list", "boleh buat",
             "what can you")

OFFLINE_KEYS = ("berita", "news", "terkini", "semasa", "cuaca", "weather",
                "live", "breaking")


def answer_offline(q, lang):
    """Soalan data-live/berita - jawab tepat tanpa model (model offline)."""
    t = q.lower()
    if not any(k in t for k in OFFLINE_KEYS):
        return None
    if lang == "ms":
        return ("Saya offline sepenuhnya tanpa internet - tiada akses berita, "
                "cuaca atau data live. Saya hanya boleh bantu pengekodan "
                "(kod, SQL, Linux, Docker) setakat pengetahuan sedia ada.")
    return ("I am fully offline with no internet - no access to news, weather "
            "or live data. I can only help with coding (code, SQL, Linux, "
            "Docker) from existing knowledge.")


def answer_self(q, cur, name, lang):
    """Jawapan tepat dari registry - model 0.5B tidak perlu berimaginasi."""
    t = q.lower()
    if not any(k in t for k in SELF_KEYS):
        return None
    if not any(k in t for k in ("anda", "kamu", "awak", "you")):
        return None
    names = ", ".join(
        f"{m['name']} [{','.join(m.get('domains', []))}]"
        f"{' (belum dilatih)' if not (m.get('file') and (MODELS_DIR / m['file']).is_file()) else ''}"
        for m in scan_models(MODELS_DIR))
    if lang == "ms":
        return (f"Saya {name}, model aktif: {cur['name']} "
                f"(pakar: {','.join(cur.get('domains', []))}). "
                f"Senarai model: {names}. Tukar dengan /model <nama>.")
    return (f"I am {name}, active model: {cur['name']} "
            f"(expert: {','.join(cur.get('domains', []))}). "
            f"Model list: {names}. Switch with /model <name>.")


def local_action(q, lang):
    """Arahan jelas yang CLI boleh buat terus tanpa model (tepat, tiada halusinasi)."""
    m = re.match(r"\s*(buatkan|buat|bina|create)\s+(folder|direktori|directory|fail|file)\s+"
                 r"[\"']?([^\"'\s]+)[\"']?", q, re.IGNORECASE)
    if not m:
        return None
    kind, dirname = m.group(2).lower(), m.group(3).strip().lstrip("/\\")
    if not dirname or dirname in (".", ".."):
        return None
    target = (Path.cwd() / dirname).resolve()
    if target != Path.cwd().resolve() and Path.cwd().resolve() not in target.parents:
        if lang == "ms":
            return (f"Blocked: '{dirname}' di luar workspace. Guna nama folder biasa, cth: buat folder test.")
        return f"Blocked: '{dirname}' outside workspace. Use a plain folder name, e.g. make folder test."
    try:
        if kind in ("folder", "direktori", "directory"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=True)
    except Exception as e:
        return f"Ralat: {e}"
    if lang == "ms":
        return (f"OK {'folder' if kind in ('folder', 'direktori', 'directory') else 'fail'} '{dirname}' siap. "
                f"Nak saya bina website blog Node.js + PostgreSQL di dalamnya fail demi fail? "
                f"Cth: `tulis package.json untuk blog`.")
    return (f"OK {'folder' if kind in ('folder', 'direktori', 'directory') else 'file'} '{dirname}' ready. "
            f"Want the Node.js + PostgreSQL blog scaffolded in it file by file?")


PLAN_TEMPLATES = {
    "web": ["Buat folder projek + `npm init -y`, pasang express + pg",
            "Tulis server + pool PostgreSQL (sambungan env)",
            "Tulis laluan CRUD: senarai, baca satu, cipta",
            "Tulis paparan ringkas (HTML hantar/baca)",
            "Tulis Dockerfile + compose (app + db + healthcheck)",
            "Uji setiap fail (`node --check`, curl) + /good"],
    "data": ["Lakar skema (tabel + kunci + hubungan)",
             "Tulis migrasi SQL cipta tabel",
             "Tulisbenih data contoh (seed)",
             "Tulis query CRUD + uji di psql",
             "Sandar/restore + /good"],
    "linux": ["Kenal pasti tugas + arahan terlibat (man --help)",
              "Cuba arahan selamat dahulu (dry-run, --help)",
              "Tulis skrip .sh + chmod +x",
              "Uji dalam folder sandbox + /good"],
}


def plan_fallback(task, lang, dom):
    steps = PLAN_TEMPLATES.get(dom) or ["Pecah tugas kepada fail terkecil",
                                        "Siapkan satu fail, uji, /good",
                                        "Ulang langkah seterusnya",
                                        "Gabung + uji hujung-ke-hujung"]
    if lang == "ms":
        return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    en = {"Buat": "Create", "Tulis": "Write", "Uji": "Test"}
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def scrub_history(h):
    """Drop degenerate turns (role-play echoes, hallucinated status lines)."""
    out = []
    for m in h:
        c = m.get("content", "") or ""
        if m.get("role") == "assistant" and (
                "Pengguna:" in c or "Pembantu:" in c
                or "Current use of models is" in c
                or "model aktif:" in c):
            continue
        out.append(m)
    return out


# commands also accepted without leading slash ("route on" -> "/route on")
BARE_CMDS = ("help", "route", "tools", "backend", "session", "sessions",
             "clear", "context", "server", "plan", "export", "cwd",
             "keluar", "exit", "quit", "good", "bad", "train-log")


def do_sft_export(flt):
    """training.jsonl -> datasets/<stack>.jsonl -> finetune/<domain>/. Returns text."""
    from store import export_stacks, merge_stacks
    added, _ = export_stacks(domain=flt)
    total = sum(added.values())
    mdom = STACK2DOMAIN.get(flt, flt) if flt else None
    merged = merge_stacks(domain=mdom)
    parts = [f"{s}+{n}" for s, n in sorted(added.items())]
    mg = ", ".join(f"{d}={n}" for d, n in sorted(merged.items()))
    return (f"SFT: {total} new rows -> datasets/ [{', '.join(parts) or 'none'}]; "
            f"merged [{mg}]")


def repl(args):
    cli_path = model_path = None
    if args.backend in ("llama", "auto"):
        cli_path, model_path = ensure_cli(), ensure_model()
    elif args.backend == "server":
        model_path = ensure_model()
    backend = make_backend(args, cli_path, model_path)
    name, lang = args.name, args.lang
    tools_on = not args.no_tools
    auto_route = args.autoroute
    session = args.session
    history = scrub_history(load_session(session)[-20:])
    cur = match_cur(model_path)
    tool_ctx = {"allow_all": args.allow_all,
                "confirm": lambda msg: input(msg).strip().lower() in ("y", "yes"),
                "rag_k": 2}
    print(f"{name} v{__version__} ready [{backend.kind}, model={cur['name']}, tools={'on' if tools_on else 'off'}, "
          f"route={'on' if auto_route else 'off'}, session={session}]. Taip /help\n")

    def on_tool(tname, targ, result):
        print(f"[tool:{tname}] {targ[:120]}")
        print((result[:600] + ("..." if len(result) > 600 else "")) + "\n")

    def on_think(thinking):
        print("\x1b[2m" + f"[berfikir]\n{thinking[:800]}" + "\x1b[0m")

    width = shutil.get_terminal_size((80, 20)).columns
    last_qa = [None, None, None]  # (question, answer, domain) for /good /bad
    while True:
        try:
            q = input("anda> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if not q.startswith("/") and q.split() and q.split()[0].lower() in BARE_CMDS:
            q = "/" + q
            print(f"(dibaca sebagai {q.split()[0]})")
        if q in ("/keluar", "/exit", "/quit"):
            break
        if q == "/help":
            print(HELP)
            continue
        if q.startswith("/lang"):
            lang = "en" if q.split()[-1] == "en" else "ms"
            print(f"Bahasa: {lang}")
            continue
        if q.startswith("/tools"):
            tools_on = q.split()[-1] != "off"
            print(f"tools={'on' if tools_on else 'off'}")
            continue
        if q == "/plan" or q.startswith("/plan "):
            task = q[len("/plan"):].strip() or "tugas semasa"
            psys = ("Anda perancang projek. Balas dengan senarai bernombor TEPAT 6 langkah ke bawah "
                    "(setiap langkah: satu fail atau satu arahan + satu ayat tujuan). "
                    "Tiada penerangan panjang."
                    if lang == "ms" else
                    "You are a project planner. Reply with EXACTLY 6 numbered steps or fewer "
                    "(each: one file or one command + one purpose sentence). No long prose.")
            print(f"... merancang ...")
            try:
                praw, _ = backend.chat(psys, history[-4:] + [{"role": "user", "content": task}],
                                       max_tokens=300)
            except Exception as e:
                print(f"Ralat: {e}")
                continue
            plan = sanitize(praw)
            import re as _re
            n_steps = len(_re.findall(r"^\s*\d+\s*[.)]", plan, _re.M))
            if (not _re.search(r"^\s*\d+\s*[.)]", plan, _re.M) or n_steps > 8
                    or _re.search(r"tepat\s+\d+", plan, _re.I)):
                plan = plan_fallback(task, lang, detect_domain(task))
                print("(rancangan templat — model tidak patuh format langkah)")
            history += [{"role": "user", "content": "/plan " + task},
                        {"role": "assistant", "content": plan}]
            save_turn(session, history[-2:])
            print(f"\nRancangan {task}:\n{plan}\nBina satu langkah satu masa — sebut 'langkah 1', dll.")
            continue
        if q.startswith("/backend"):
            b = q.split()[-1]
            if b in ("llama", "auto"):
                cli_path, model_path = ensure_cli(), ensure_model()
                backend = LlamaCliBackend(cli_path, model_path, n_predict=args.max_tokens)
            elif b == "lmstudio":
                backend = LmStudioBackend(args.lm_url, args.lm_model, max_tokens=args.max_tokens)
            elif b == "server":
                from backends import ServerBackend
                model_path = ensure_model()
                mgr = get_manager(args)
                already, secs = mgr.load(model_path)
                if not already:
                    print(f"[server] model dimuat dalam {secs:.1f}s")
                backend = ServerBackend(mgr, model="xcoder", max_tokens=args.max_tokens)
            else:
                print("usage: /backend llama|lmstudio|server"); continue
            print(f"backend={backend.kind}")
            continue
        if q == "/server":
            if backend.kind != "server":
                print("backend bukan server. /backend server dahulu.")
                continue
            st = backend.manager.status()
            print(f"server {st['url']} alive={st['alive']} healthy={st['healthy']} "
                  f"model={st['model']} flavor={st.get('flavor')} ngl={st.get('ngl')} "
                  f"idle_timeout={st['idle_timeout']}s")
            continue
        if q == "/models":
            for m in scan_models(MODELS_DIR):
                f = MODELS_DIR / m.get("file", "")
                has = bool(m.get("file")) and f.exists()
                mark = "*" if m.get("name") == cur.get("name") else " "
                real = "" if has else " (fail tiada)"
                print(f"{mark} {m['name']} [{','.join(m.get('domains', []))}] "
                      f"{fmt_size(f) if has else 'tiada'}{real}")
            print("* = aktif (satu pada satu masa). /model <nama> tukar. /route on = auto.")
            continue
        if q == "/models check":
            from experts import audit_models
            rep = audit_models(
                MODELS_DIR,
                progress=lambda n, i, t: print(f"\r  hash {i}/{t} {n[:40]}", end="", flush=True))
            print()
            for it in rep["models"]:
                flag = "OK " if it["magic_ok"] and it["registered"] else "!! "
                print(f"{flag} {it['file']} {it['mb']}MB sha={it['sha']} "
                      f"model={it['registered'] or '(tidak didaftar)'}")
            if rep["duplicates"]:
                for h, names in rep["duplicates"].items():
                    print(f"DUPLIKAT sha={h}: {', '.join(names)}")
            else:
                print("tiada duplikat.")
            if rep["registry_missing"]:
                print(f"fail tiada (dalam registry): {', '.join(rep['registry_missing'])}")
            continue
        if q == "/model" or q.startswith("/model "):
            rest = q[len("/model"):].strip()
            if not rest:
                print(f"aktif: {cur['name']} [{','.join(cur['domains'])}]")
                continue
            if rest == "import" or rest.startswith("import "):
                from experts import auto_import as auto_import_model
                itoks = rest[len("import"):].strip().split()
                explicit = next((t for t in itoks if not t.startswith("--")), "")
                from_dir = None
                for i, t in enumerate(itoks):
                    if t == "--from" and i + 1 < len(itoks):
                        from_dir = itoks[i + 1]
                if explicit and Path(explicit).expanduser().is_file():
                    nm, doms = infer_name_domains(Path(explicit).name)
                    for i, t in enumerate(itoks):
                        if t == "--name" and i + 1 < len(itoks):
                            nm = itoks[i + 1]
                        if t == "--domains" and i + 1 < len(itoks):
                            doms = [d.strip() for d in itoks[i + 1].split(",") if d.strip()]
                    e = add_model(MODELS_DIR, str(Path(explicit).expanduser()),
                                  name=nm, domains=doms)
                    note = f"import {explicit} -> {e['name']}"
                else:
                    e, note = auto_import_model(MODELS_DIR, from_dir)
                    if e is None:
                        print(note)
                        continue
                p2 = MODELS_DIR / e["file"]
                if backend.kind == "server":
                    already, secs = backend.manager.load(p2)
                    backend.model = e["name"]
                    if not already:
                        print(f"[server] dimuat dalam {secs:.1f}s")
                elif backend.kind != "lmstudio":
                    backend = LlamaCliBackend(cli_path, p2, n_predict=args.max_tokens)
                cur = {"name": e["name"], "domains": e.get("domains", [GENERAL])}
                print(f"{note}; aktif: {cur['name']} [{','.join(cur['domains'])}]")
                continue
            if rest.startswith("add "):
                toks = rest[4:].split()
                src = toks[0] if toks else ""
                nm, doms = None, [GENERAL]
                for i, t in enumerate(toks):
                    if t == "--name" and i + 1 < len(toks):
                        nm = toks[i + 1]
                    if t == "--domains" and i + 1 < len(toks):
                        doms = [d.strip() for d in toks[i + 1].split(",") if d.strip()]
                if not src:
                    print("guna: /model add <url|fail> [--name N] [--domains a,b]")
                    continue
                try:
                    e = add_model(MODELS_DIR, src, name=nm, domains=doms)
                    print(f"OK daftar {e['name']} [{','.join(e['domains'])}]")
                except Exception as ex:
                    print(f"Ralat: {ex}")
                continue
            p, note = resolve_model(MODELS_DIR, rest)
            if p is None:
                print(note)
                continue
            if backend.kind == "lmstudio":
                backend = LmStudioBackend(args.lm_url, rest, max_tokens=args.max_tokens)
            elif backend.kind == "server":
                already, secs = backend.manager.load(p)
                backend.model = rest
                if not already:
                    print(f"[server] Loading model... dimuat dalam {secs:.1f}s")
            else:
                backend = LlamaCliBackend(cli_path, p, n_predict=args.max_tokens)
            hit = next((x for x in scan_models(MODELS_DIR)
                        if x["name"].lower() == rest.lower()), None)
            cur = {"name": hit["name"] if hit else rest,
                   "domains": hit.get("domains", [GENERAL]) if hit else [GENERAL]}
            print(f"model={cur['name']} [{','.join(cur['domains'])}] {note}".rstrip())
            continue
        if q == "/consult" or q.startswith("/consult "):
            ctask = q[len("/consult"):].strip()
            if not ctask:
                print("guna: /consult <soalan>  (runding 2 pakar, general sintesis penuh)")
                continue
            cdom = detect_domain(ctask)
            cands = [m for m in scan_models(MODELS_DIR)
                     if m.get("file") and (MODELS_DIR / m["file"]).is_file()
                     and cdom in m.get("domains", [])][:2]
            if not cands:
                print(f"tiada pakar {cdom} berfail — latih dahulu.")
                continue
            cli = cli_path or ensure_cli()
            from agent import clean as _clean
            parts = []
            for m in cands:
                print(f"[consult:{m['name']}] bertanya...")
                try:
                    tmp = LlamaCliBackend(cli, MODELS_DIR / m["file"], n_predict=300)
                    esys = (f"Anda pakar {','.join(m.get('domains', []))}. Jawab ringkas."
                            if lang == "ms" else
                            f"You are a {','.join(m.get('domains', []))} expert. Answer briefly.")
                    rraw, _ = tmp.chat(esys, [{"role": "user", "content": ctask}])
                    rans = _clean(sanitize(rraw), name)
                except Exception as e:
                    rans = f"(ralat: {e})"
                parts.append((m["name"], rans))
                print(f"[{m['name']}]: {rans[:400]}\n")
            gfile = next((MODELS_DIR / m["file"] for m in scan_models(MODELS_DIR)
                          if m.get("name") in ("xcoder-general", "base") and m.get("file")
                          and (MODELS_DIR / m["file"]).is_file()), None)
            if gfile is None:
                print("tiada model general berfail untuk sintesis.")
                continue
            ssys = ("Gabung jawapan pakar di bawah menjadi SATU jawapan akhir ringkas dalam Bahasa Melayu. "
                    "Buang ulangan; jika bercanggah pilih yang paling betul. Terus jawab, tanpa ulas."
                    if lang == "ms" else
                    "Merge the expert answers below into ONE brief final answer in English. "
                    "Drop repetition; on conflict pick the most correct. Answer directly.")
            suser = ctask + "\n\n" + "\n\n".join(f"[{n}]: {a}" for n, a in parts)
            try:
                sraw, stps = LlamaCliBackend(cli, gfile, n_predict=args.max_tokens).chat(ssys, [{"role": "user",
                                                                                             "content": suser}])
                sans = _clean(sanitize(sraw), name)
            except Exception as e:
                print(f"Ralat sintesis: {e}")
                continue
            history += [{"role": "user", "content": ctask}, {"role": "assistant", "content": sans}]
            save_turn(session, history[-2:])
            log_training(ctask, sans, lang=lang, kind="consult",
                         tools_trace=[{"tool": "consult", "arg": n, "result": a[:300]} for n, a in parts],
                         domain=cdom)
            last_qa = [ctask, sans, cdom]
            print(f"\n{name} (sintesis general)> {sans}")
            if stps:
                print(f"{stps:.1f} T/s")
            print()
            continue
        if q.startswith("/route"):
            auto_route = q.split()[-1] != "off" if len(q.split()) > 1 else True
            print(f"route={'on' if auto_route else 'off'} (auto-tukar pakar ikut domain)")
            continue
        if q == "/sessions":
            print("\n".join(list_sessions()) or "(no sessions)")
            continue
        if q.startswith("/session ") or q == "/session":
            session = q.split(None, 1)[1] if len(q.split()) > 1 else "default"
            history = scrub_history(load_session(session)[-20:])
            last_qa = [None, None, None]
            print(f"session={session} ({len(history)} turns loaded)")
            continue
        if q == "/context":
            st = context_stats(session, len(history))
            print(f"session={st['session']} memory={st['memory']} file={st['file']} "
                  f"~{st['approx_tokens']} tokens (model ctx 2048 llama-cli / 4096 server)")
            continue
        if q == "/clear":
            history.clear()
            last_qa = [None, None, None]
            clear_session(session)
            print("Sejarah dipadam (memori + fail session).")
            continue
        if q in ("/good", "/bad"):
            if last_qa[0] and last_qa[2]:
                log_training(last_qa[0], last_qa[1], lang=lang, kind="rated",
                             tools_trace=[], rating=q[1:], domain=last_qa[2])
                print("Logged ke training.jsonl")
            else:
                print("Nothing to rate yet.")
            continue
        if q == "/train-log":
            from store import DATA_DIR
            p = DATA_DIR / "training.jsonl"
            n = sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0
            print(f"{p} : {n} rows")
            continue
        if q.startswith("/export"):
            mode = q.split()[-1] if len(q.split()) > 1 else "sft"
            if mode in ("sft", "both"):
                print(do_sft_export(None))
            if mode in ("corpus", "pretrain", "both"):
                print(export_pretrain_corpus())
            continue
        if q == "/cwd":
            print(Path.cwd())
            continue
        if q.startswith("/"):
            print("Perintah tidak dikenali. Taip /help untuk senarai.")
            continue

        print(f"... {name} berfikir ({backend.kind}/{cur['name']}) ...")
        dom = detect_domain(q)
        self_ans = answer_self(q, cur, name, lang)
        off_ans = answer_offline(q, lang) if self_ans is None else None
        if self_ans is not None:
            ans, tps, trace = self_ans, None, [{"tool": "models", "arg": "", "result": "registry"}]
        elif off_ans is not None:
            ans, tps, trace = off_ans, None, [{"tool": "local", "arg": q[:100], "result": "offline-note"}]
        else:
            loc = local_action(q, lang)
            if loc is not None:
                ans, tps, trace = loc, None, [{"tool": "local", "arg": q[:100], "result": "mkdir"}]
            else:
                if auto_route and backend.kind in ("llama-cli", "llama", "server") and dom != GENERAL \
                        and dom not in cur.get("domains", []):
                    cand = pick_expert(dom, detect_stack(q))
                    if cand:
                        if backend.kind == "server":
                            print(f"-> route: {cand['name']} [{dom}] Loading model...")
                            already, secs = backend.manager.load(MODELS_DIR / cand["file"])
                            backend.model = cand["name"]
                            if not already:
                                print(f"[server] dimuat dalam {secs:.1f}s")
                        else:
                            backend = LlamaCliBackend(cli_path, MODELS_DIR / cand["file"],
                                                      n_predict=args.max_tokens)
                        cur = {"name": cand["name"], "domains": cand.get("domains", [dom])}
                        print(f"-> route: {cur['name']} [{dom}] (satu model aktif)")
                    else:
                        print(f"(tiada pakar {dom} - jawab dengan {cur['name']}; "
                              f"latih: train --mode sft --domain {dom})")
                try:
                    ans, tps, trace = run_agent(backend, q, name=name, lang=lang, history=history,
                                                tools_on=tools_on, tool_ctx=tool_ctx, on_tool=on_tool,
                                                expert_ctx=expert_context(cur), on_think=on_think)
                except Exception as e:
                    print(f"Ralat: {e}")
                    continue
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        save_turn(session, history[-2:])
        log_training(q, ans, lang=lang, kind="chat", tools_trace=trace, domain=dom)
        last_qa = [q, ans, dom]
        print(f"\n{name}> {ans}")
        if tps:
            s = f"{tps:.1f} T/s"
            print(" " * max(0, width - len(s)) + f"\x1b[2m{s}\x1b[0m")
        print()

    if backend.kind == "server":
        backend.manager.stop()
        print("[server] stopped, RAM dilepaskan.")


def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.setup:
        cli = ensure_cli()
        model = ensure_model()
        print(f"OK engine: {cli}\nOK model : {model} ({model.stat().st_size} bytes)")
        return
    if args.cmd == "train":
        flt = (args.domain or "").strip().lower() or None
        ddir = STACK2DOMAIN.get(flt, flt) if flt else "general"
        corp_dest = f"finetune/{ddir}/corpus.txt"
        if args.mode in ("sft", "both"):
            print(do_sft_export(flt) + "; validate: python datasets/scripts/validate.py")
        if args.mode in ("pretrain", "both"):
            n, where = export_pretrain_corpus(dest=corp_dest, domain=flt)
            print(f"Pretrain corpus: {n} turns -> {where}")
        if not args.export_only:
            print("Next: Colab T4 -> finetune/train_unsloth.py (LoRA/SFT) or app/train_pretrain.py "
                  "(continued pretrain), export GGUF.")
            if flt:
                print(f"Next: import GGUF pakar -> /model add <fail> --name xcoder-{flt} "
                      f"--domains {flt}. See step.md.")
            else:
                print("Import: /model add <fail> --name xcoder-base --domains general. See step.md.")
        return
    if args.once:
        backend = make_backend(args)
        cur = match_cur(args.model)
        self_ans = answer_self(args.once, cur, args.name, args.lang)
        if self_ans is not None:
            print(self_ans)
            return
        off_ans = answer_offline(args.once, args.lang)
        if off_ans is not None:
            print(off_ans)
            return
        loc = local_action(args.once, args.lang)
        if loc is not None:
            print(loc)
            return
        ans, tps, _ = run_agent(backend, args.once, name=args.name, lang=args.lang,
                                tools_on=not args.no_tools,
                                tool_ctx={"allow_all": True, "rag_k": 2},
                                expert_ctx=expert_context(cur),
                                on_think=lambda t: print("\x1b[2m[berfikir]\n" + t[:800] + "\x1b[0m"))
        print(ans)
        if backend.kind == "server":
            backend.manager.stop()
        return
    if args.tui and sys.stdin.isatty() and sys.stdout.isatty():
        from tui import run_tui
        title = f" XCoder {__version__} [{args.backend}] lang={args.lang} "
        status = " PgUp/PgDn skrol | Up/Dn sejarah | Ctrl-D keluar | /help perintah "
        run_tui(lambda: repl(args), title, status)
        return
    repl(args)


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        raise SystemExit("Perlukan Python 3.9+")
    main()
