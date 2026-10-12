"""XCoder - opencode-style agentic CLI (stdlib only).

Backends: resident server (default interaktif, 0 reload) | llama-cli (--once/skrip) | lmstudio
Tools: list/read/write/edit/run/bash/rag with workspace jail + confirm.
Sessions: app/sessions/<name>.jsonl | Training flywheel: app/data/training.jsonl
Train: `py app/ezcodex.py train --mode sft|pretrain --export-only` (+ Colab scripts in finetune/)
Build exe: `py app/build_exe.py` (PyInstaller, win/linux/macos)

Compat: old flags --lang/--name/--setup still work. Chat is now agentic.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

APP_DIR = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
           else Path(__file__).resolve().parent)
sys.path.insert(0, str(APP_DIR))

__version__ = "0.4.46"

from fetch import (LLAMA_TAG, LLAMA_BASE, HF_MODEL_URL, MODEL_SIZE,
                   asset_name, download, verify, sha256_file,
                   ensure_cli as _ensure_cli, ensure_model as _ensure_model,
                   cuda_dir as _cuda_dir, ensure_cuda as _ensure_cuda)

from backends import LlamaCliBackend, LmStudioBackend
from agent import run_agent, sanitize
from experts import (GENERAL, add as add_model, detect as detect_domain,
                     detect_stack, fmt_size, resolve as resolve_model,
                     scan as scan_models, record_speed,
                     infer_name_domains, resolve_layout)
from store import load_session, save_turn, list_sessions, log_training, export_sft, export_pretrain_corpus
from store import clear_session, context_stats

BIN_DIR, MODELS_DIR = resolve_layout(APP_DIR)


def ensure_cli():
    return _ensure_cli(BIN_DIR)


def ensure_model():
    return _ensure_model(MODELS_DIR)


HELP = """Commands:
/help                 this help
/lang ms|en           switch language (clears turn cache, keeps session file)
/tools on|off         enable/disable agentic tool loop
/plan <tugas>         numbered build plan (maks 6 langkah), then build file by file
/bina <tugas>         laksana plan automatik langkah demi langkah (maks 8, tukar pakar setiap langkah)
(fikiran AI sentiasa ditunjuk automatik sebelum jawapan — tiada arahan khas)
/backend llama|lmstudio|server|auto  tukar enjin (auto=server interaktif, llama --once)
/server               status resident llama-server (backend server sahaja)
/model <nama|path|id>  tukar model pakar (satu aktif pada satu masa)
/model add <url|fail> [--name N] [--domains a,b]  daftar model baharu
/model import [fail] [--from dir]  auto-import GGUF terbaru + aktifkan
/models               senarai semua model pakar + domain
/models check         audit GGUF (hash, sah, duplikat, registry)
  /consult <soalan>     runding 2 pakar, pengurus sintesis jawapan penuh
  /semua <soalan>       semua model jawab, setiap satu result sendiri
  /kerja <soalan>       pakar bekerjasama, gemma semak + putuskan
/route on|off         auto-tukar pakar ikut domain (lalai: sentiasa on)
/session <name>       switch session file (sessions/<name>.jsonl)
/sessions             list sessions
/new [nama]           session baharu segar (kosongkan konteks)
/context              show context usage of current session
/clear                clear in-memory history AND session file
/good | /bad          rate last answer -> training log (flywheel)
/train-log            show training.jsonl count
/export sft|corpus    pecah ke datasets/<stack> + gabung finetune/ (atau corpus.txt)
/sambung | /continue   sambung jawapan terpotong terakhir
/cwd                  print workspace root
/keluar|/exit|/quit   exit
Inline tool (when /tools on): model may emit ```tool:read docs/js-basics.md``` etc.
Tools: list read write edit run bash rag (workspace-jailed, bash asks confirm).
"""


def cuda_dir():
    return _cuda_dir(APP_DIR)


def ensure_cuda():
    return _ensure_cuda(APP_DIR)


def build_parser():
    ap = argparse.ArgumentParser(description="XCoder agentic CLI (opencode-style, stdlib only)")
    ap.add_argument("--lang", default="ms", choices=["ms", "en"])
    ap.add_argument("--name", default="XCoder")
    ap.add_argument("--setup", action="store_true", help="download engine + model only, then exit")
    ap.add_argument("--backend", default="auto", choices=["llama", "lmstudio", "server", "auto"],
                    help="auto=server jika interaktif, llama jika --once/pipe")
    ap.add_argument("--lm-url", default="http://localhost:1234/v1")
    ap.add_argument("--lm-model", default="ezcodex-0.5b")
    ap.add_argument("--port", type=int, default=8080, help="llama-server port (backend server)")
    ap.add_argument("--threads", type=int, default=0, help="llama-server CPU threads (0=auto)")
    ap.add_argument("--ctx", type=int, default=0, help="llama-server context size (0=auto ikut RAM)")
    ap.add_argument("--idle-timeout", type=int, default=180, help="server unload after N idle secs (0=never)")
    ap.add_argument("--gpu", default="off", choices=["off", "cuda"],
                    help="server/llama GPU offload (cuda = GTX 1070+, bin-cuda)")
    ap.add_argument("--model", default="", help="override GGUF path (llama) or model id (lmstudio)")
    ap.add_argument("--session", default="default")
    ap.add_argument("--no-tools", action="store_true", help="disable agentic tool loop")
    ap.add_argument("--allow-all", action="store_true", help="skip bash confirm (dangerous)")
    ap.add_argument("--max-tokens", type=int, default=0, help="max tokens per answer (0=auto ikut RAM, kod: 800-1000)")
    ap.add_argument("--autoroute", default=True, action=argparse.BooleanOptionalAction,
                    help="auto-switch expert model by question domain (default on)")
    ap.add_argument("--tui", default=True, action=argparse.BooleanOptionalAction,
                    help="fullscreen opencode-style view (default on when TTY)")
    ap.add_argument("--once", default="", help="one-shot prompt, print answer and exit (for scripts)")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train", help="training flywheel: export SFT dataset / pretrain corpus")
    t.add_argument("--mode", choices=["sft", "pretrain", "both"], default="both")
    t.add_argument("--domain", default="", help="export stack sahaja cth: docker,linux,web,sql,python,php")
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


def resolve_backend_name(args):
    """auto -> server jika REPL interaktif (KV-cache + tanpa reload),
    llama jika --once/pipe (0 RAM resident)."""
    if getattr(args, "backend", "auto") != "auto":
        return args.backend
    if getattr(args, "once", ""):
        return "llama"
    try:
        if sys.stdin.isatty():
            return "server"
    except Exception:
        pass
    return "llama"


def fmt_speed(backend, tps):
    """Pecahan timing: [prefill Xs | decode Ys -> N T/s] jika backend lapor timings."""
    tm = getattr(backend, "last_timings", None)
    if isinstance(tm, dict) and tm.get("predicted_ms"):
        pre = (tm.get("prompt_ms") or 0) / 1000
        dec = (tm.get("predicted_ms") or 1) / 1000
        pn = tm.get("prompt_n", "?")
        dn = tm.get("predicted_n", "?")
        return f"[prefill {pre:.1f}s/{pn} | decode {dec:.1f}s/{dn} -> {tps:.1f} T/s]"
    return f"{tps:.1f} T/s"


def make_backend(args, cli_path=None, model_path=None):
    name = resolve_backend_name(args)
    if name == "lmstudio":
        return LmStudioBackend(args.lm_url, args.model or args.lm_model,
                               max_tokens=args.max_tokens)
    if name == "server":
        from backends import ServerBackend
        ensure_cli()  # pastikan enjin (termasuk llama-server) wujud
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

MANAGER_NAME = "gemma4-e4b"

TUTORIAL_KEYS = ("tutorial", "dari awal", "zero to", "langkah demi langkah",
                 "step by step", "teruskan", "buat website", "buat blog",
                 "bina website", "bina blog", "belajar ", "panduan lengkap",
                 "complete guide", "website untuk", "blog untuk",
                 "website", "blog", "custom")


def manager_entry():
    """Entri registry pengurus (gemma) jika berfail, else None."""
    for m in scan_models(MODELS_DIR):
        if m.get("name") == MANAGER_NAME and m.get("file") \
                and (MODELS_DIR / m["file"]).is_file():
            return m
    return None


def manager_file():
    """Fail model pengurus (gemma) jika wujud, else None."""
    m = manager_entry()
    return MODELS_DIR / m["file"] if m else None


def pick_manager_open(q):
    """Soalan terbuka/tutorial/rentas-stack -> pengurus (karangan), else None."""
    if any(k in q.lower() for k in TUTORIAL_KEYS):
        return manager_entry()
    try:
        from experts import count_stacks as _cs
        if _cs(q) >= 2:
            return manager_entry()
    except Exception:
        pass
    return None

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

# English aliases for Malay commands
ALIASES = {"/continue": "/sambung"}


def do_sft_export(flt):
    """training.jsonl -> datasets/<stack>.jsonl -> finetune/stacks/<stack>/. Returns text."""
    from store import export_stacks, merge_stacks
    added, _ = export_stacks(domain=flt)
    total = sum(added.values())
    merged, _ev = merge_stacks(stack=flt)
    parts = [f"{s}+{n}" for s, n in sorted(added.items())]
    mg = ", ".join(f"{d}={n}" for d, n in sorted(merged.items()))
    return (f"SFT: {total} new rows -> datasets/ [{', '.join(parts) or 'none'}]; "
            f"merged [{mg}]")


def print_speed_recap(rows):
    """Ringkasan kelajuan semua model terlibat: [(nama, tps, saat)]."""
    rows = [(n, t, s) for n, t, s in rows if t]
    if not rows:
        return
    print("\n--- kelajuan ---")
    for n, t, s in rows:
        print(f"  {n:<18} {t:>6.1f} T/s ({s:.0f}s)")
    print()


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
    # Fresh session on every app start (default): jangan sambung default.jsonl lama.
    # Explicit --session <nama> masih dihormati; /new dan /session masih boleh tukar.
    if session == "default":
        import time as _ts
        session = f"sesi-{_ts.strftime('%H%M%S')}"
    history = scrub_history(load_session(session)[-20:])
    cur = match_cur(model_path)
    short = (name[:2].lower() if name and len(name) >= 2 else (name or "xc").lower())
    tool_ctx = {"allow_all": args.allow_all,
                "confirm": lambda msg: input(msg).strip().lower() in ("y", "yes"),
                "rag_k": 2}
    print(f"{name} v{__version__} ready [{backend.kind}, model={cur['name']}, tools={'on' if tools_on else 'off'}, "
          f"route={'on' if auto_route else 'off'}, session={session}]. Taip /help\n")
    try:
        from tools import _rag_index
        _rag_index()  # prewarm cache RAG semasa startup, bukan semasa tool dipanggil
    except Exception:
        pass

    def on_tool(tname, targ, result):
        print(f"[tool:{tname}] {targ[:120]}")
        print((result[:600] + ("..." if len(result) > 600 else "")) + "\n")

    def on_think(thinking):
        import re as _re
        thinking = _re.sub(r"(?i)\b(gemma|google deepmind|google)\b", name, thinking)
        print("\x1b[2m" + f"[berfikir]\n{thinking[:800]}" + "\x1b[0m")

    width = shutil.get_terminal_size((80, 20)).columns
    last_qa = [None, None, None]  # (question, answer, domain) for /good /bad

    def switch_to(cand, dom):
        """Tukar pakar (satu aktif pada satu masa)."""
        nonlocal backend, cur
        if backend.kind == "server":
            print(f"-> route: {cand['name']} [{dom}] Loading model...")
            already, secs = backend.manager.load(MODELS_DIR / cand["file"])
            backend.model = cand["name"]
            if not already:
                print(f"[server] dimuat dalam {secs:.1f}s")
        elif backend.kind == "lmstudio":
            print(f"(lmstudio: tukar manual /model {cand['name']})")
            return
        else:
            backend = LlamaCliBackend(cli_path, MODELS_DIR / cand["file"],
                                      n_predict=args.max_tokens)
        cur = {"name": cand["name"], "domains": cand.get("domains", [dom])}
        print(f"-> route: {cur['name']} [{dom}] (satu model aktif)")

    def make_plan(task):
        """Rancangan bernombor (pengurus gemma jika ada, else model aktif)."""
        psys = ("Anda perancang projek. Balas dengan senarai bernombor TEPAT 6 langkah ke bawah "
                "(setiap langkah: satu fail atau satu arahan + satu ayat tujuan). "
                "Tiada penerangan panjang."
                if lang == "ms" else
                "You are a project planner. Reply with EXACTLY 6 numbered steps or fewer "
                "(each: one file or one command + one purpose sentence). No long prose.")
        mfile = manager_file()
        if mfile is not None:
            print(f"... {MANAGER_NAME} merancang (pengurus) ...")
            pbe = LlamaCliBackend(cli_path, mfile, n_predict=400)
        else:
            print(f"... merancang ...")
            pbe = backend
        praw, _ = pbe.chat(psys, history[-4:] + [{"role": "user", "content": task}],
                           max_tokens=400)
        plan = sanitize(praw)
        import re as _re
        n_steps = len(_re.findall(r"^\s*\d+\s*[.)]", plan, _re.M))
        if (not _re.search(r"^\s*\d+\s*[.)]", plan, _re.M) or n_steps > 8
                or _re.search(r"tepat\s+\d+", plan, _re.I)):
            plan = plan_fallback(task, lang, detect_domain(task))
            print("(rancangan templat — model tidak patuh format langkah)")
        return plan
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
        _head = q.split()[0] if q.split() else ""
        if _head in ALIASES:
            q = ALIASES[_head] + q[len(_head):]
            print(f"(alias Inggeris: {_head} -> {ALIASES[_head]})")
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
            try:
                plan = make_plan(task)
            except Exception as e:
                print(f"Ralat: {e}")
                continue
            history += [{"role": "user", "content": "/plan " + task},
                        {"role": "assistant", "content": plan}]
            save_turn(session, history[-2:])
            print(f"\nRancangan {task}:\n{plan}\nBina satu langkah satu masa — sebut 'langkah 1', dll.")
            continue
        if q == "/bina" or q.startswith("/bina "):
            btask = q[len("/bina"):].strip()
            if not btask:
                print("guna: /bina <tugas projek>  (cth: /bina blog gitar node.js postgresql)")
                continue
            try:
                plan = make_plan(btask)
            except Exception as e:
                print(f"Ralat: {e}")
                continue
            print(f"\nRancangan {btask}:\n{plan}\n")
            import re as _r2
            steps = [m.strip() for m in _r2.findall(r"^\s*\d+\s*[.)]\s*(.+)$", plan, _r2.M)][:8]
            if not steps:
                steps = [btask]
            for i, st in enumerate(steps, 1):
                import time as _t
                t0 = _t.time()
                print(f"\n=== Langkah {i}/{len(steps)} [{_t.strftime('%H:%M:%S')}] "
                      f"{st[:100]} ===")
                sq = (f"Langkah {i}/{len(steps)} projek [{btask}]: {st}. "
                      f"Lengkapkan langkah ini sahaja (kod/arahan penuh).")
                sdom = detect_domain(sq)
                if auto_route and backend.kind in ("llama-cli", "llama", "server") \
                        and sdom != GENERAL and sdom not in cur.get("domains", []):
                    cand = pick_expert(sdom, detect_stack(sq))
                    if cand:
                        switch_to(cand, sdom)
                    else:
                        print(f"(tiada pakar {sdom} — terus dengan {cur['name']})")
                print(f"{short} berfikir ({cur['name']}) ...")
                try:
                    msize = "4.5B" if "gemma" in cur.get("name", "") else "0.5B"
                    ans, tps, trace = run_agent(backend, sq, name=name, lang=lang, history=history,
                                                tools_on=tools_on, tool_ctx=tool_ctx, on_tool=on_tool,
                                                expert_ctx=expert_context(cur) if auto_route else "",
                                                on_think=on_think,
                                                size=msize)
                except Exception as e:
                    print(f"Ralat langkah {i}: {e} — berhenti.")
                    break
                history += [{"role": "user", "content": sq}, {"role": "assistant", "content": ans}]
                save_turn(session, history[-2:])
                log_training(sq, ans, lang=lang, kind="bina", tools_trace=trace, domain=sdom)
                last_qa = [sq, ans, sdom]
                el = _t.time() - t0
                if tps:
                    record_speed(MODELS_DIR, cur.get("name", ""), tps)
                print(f"{short}> {ans}")
                if tps:
                    s = fmt_speed(backend, tps) + f" ({el:.0f}s langkah)"
                    print(" " * max(0, width - len(s)) + f"\x1b[2m{s}\x1b[0m")
                print()
            print(f"Siap {len(steps)} langkah. Semak fail, /good yang betul.")
            continue
        if q.startswith("/backend"):
            b = q.split()[-1]
            if b == "auto":
                b = resolve_backend_name(args)
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
                spd = f" {m['last_tps']} T/s" if m.get("last_tps") else ""
                print(f"{mark} {m['name']} [{','.join(m.get('domains', []))}] "
                      f"{fmt_size(f) if has else 'tiada'}{spd}{real}")
            print("* = aktif (satu pada satu masa). /model <nama> tukar. /route on = auto.")
            continue
        if q == "/models check":
            from experts import audit_models
            rep = audit_models(
                MODELS_DIR,
                progress=lambda n, i, t: print(f"\r  hash {i}/{t} {n[:40]}", end="", flush=True))
            print()
            for it in rep["models"]:
                hk = it.get("hash_ok")
                hs = "pin-ok" if hk is True else ("PIN-GAGAL" if hk is False else "tanpa-pin")
                flag = "OK " if it["magic_ok"] and it["registered"] and hk is not False else "!! "
                print(f"{flag} {it['file']} {it['mb']}MB sha={it['sha']} [{hs}] "
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
                print("guna: /consult <soalan>  (runding 2 pakar, pengurus sintesis penuh)")
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
            crows = []
            for m in cands:
                print(f"[consult:{m['name']}] bertanya...")
                try:
                    tmp = LlamaCliBackend(cli, MODELS_DIR / m["file"], n_predict=300)
                    esys = (f"Anda pakar {','.join(m.get('domains', []))}. Jawab ringkas."
                            if lang == "ms" else
                            f"You are a {','.join(m.get('domains', []))} expert. Answer briefly.")
                    import time as _t0
                    _s0 = _t0.time()
                    rraw, _ctps = tmp.chat(esys, [{"role": "user", "content": ctask}])
                    _cel = _t0.time() - _s0
                    rans = _clean(sanitize(rraw), name)
                except Exception as e:
                    rans = f"(ralat: {e})"
                    _ctps, _cel = None, 0
                parts.append((m["name"], rans))
                crows.append((m["name"], _ctps, _cel))
                print(f"[{m['name']}]: {rans[:400]}\n")
            mfile = manager_file()
            if mfile is not None:
                gfile, gname = mfile, MANAGER_NAME + " (pengurus)"
            else:
                gfile = next((MODELS_DIR / m["file"] for m in scan_models(MODELS_DIR)
                              if m.get("name") in ("xcoder-general", "base") and m.get("file")
                              and (MODELS_DIR / m["file"]).is_file()), None)
                gname = "general"
            if gfile is None:
                print("tiada model general berfail untuk sintesis.")
                continue
            print(f"[sintesis {gname}] menggabung...")
            ssys = ("Gabung jawapan pakar di bawah menjadi SATU jawapan akhir ringkas dalam Bahasa Melayu. "
                    "Buang ulangan; jika bercanggah pilih yang paling betul. Terus jawab, tanpa ulas."
                    if lang == "ms" else
                    "Merge the expert answers below into ONE brief final answer in English. "
                    "Drop repetition; on conflict pick the most correct. Answer directly.")
            suser = ctask + "\n\n" + "\n\n".join(f"[{n}]: {a}" for n, a in parts)
            try:
                _s1 = _t0.time()
                sraw, stps = LlamaCliBackend(cli, gfile, n_predict=args.max_tokens).chat(ssys, [{"role": "user",
                                                                                             "content": suser}])
                _sel = _t0.time() - _s1
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
            print(f"\n{name} (sintesis {gname})> {sans}")
            if stps:
                print(f"{stps:.1f} T/s")
                crows.append((gname, stps, _sel))
            print_speed_recap(crows)
            print()
            continue
        if q == "/semua" or q.startswith("/semua "):
            stask = q[len("/semua"):].strip()
            if not stask:
                print("guna: /semua <soalan>  (semua model jawab, result sendiri-sendiri)")
                continue
            cands = [m for m in scan_models(MODELS_DIR)
                     if m.get("file") and (MODELS_DIR / m["file"]).is_file()]
            if not cands:
                print("tiada model berfail.")
                continue
            print(f"{len(cands)} model akan jawab bergilir (~1-2 min setiap satu, Ctrl-C henti).")
            cli = cli_path or ensure_cli()
            from agent import clean as _clean2
            sdom = detect_domain(stask)
            srows = []
            for m in cands:
                hdr = f"=== [{m['name']}] [{','.join(m.get('domains', []))}] ==="
                print(f"\n{hdr}")
                try:
                    tmp = LlamaCliBackend(cli, MODELS_DIR / m["file"], n_predict=250)
                    esys = (f"Anda pakar {','.join(m.get('domains', []))}. Jawab ringkas."
                            if lang == "ms" else
                            f"You are a {','.join(m.get('domains', []))} expert. Answer briefly.")
                    import time as _t2
                    t0 = _t2.time()
                    rraw, rtps = tmp.chat(esys, [{"role": "user", "content": stask}])
                    rans = _clean2(sanitize(rraw), name)
                except KeyboardInterrupt:
                    print("(dihenti — model berbaki dilangkau)")
                    break
                except Exception as e:
                    print(f"(ralat: {e})")
                    continue
                el = _t2.time() - t0
                print(rans)
                if rtps:
                    print(f"\x1b[2m[{m['name']}] {rtps:.1f} T/s ({el:.0f}s)\x1b[0m")
                    record_speed(MODELS_DIR, m["name"], rtps)
                srows.append((m["name"], rtps, el))
                log_training(stask, rans, lang=lang, kind="semua",
                             domain=",".join(m.get("domains", [])))
            print_speed_recap(srows)
            print("Siap /semua. /good mana yang betul untuk latih semula.")
            continue
        if q == "/kerja" or q.startswith("/kerja "):
            wtask = q[len("/kerja"):].strip()
            if not wtask:
                print("guna: /kerja <soalan>  (pakar jawab, gemma semak + putuskan)")
                continue
            from experts import matching_stacks as _ms
            stacks = _ms(wtask) or [detect_domain(wtask)]
            cands = [m for m in scan_models(MODELS_DIR)
                     if m.get("file") and (MODELS_DIR / m["file"]).is_file()
                     and m.get("name") != MANAGER_NAME
                     and any(d in stacks for d in m.get("domains", []))]
            # utamakan nama sepadan stack, had 4 pakar (masa CPU)
            cands.sort(key=lambda m: 0 if any(s in m["name"] for s in stacks) else 1)
            cands = cands[:4]
            if not cands:
                print("tiada pakar berfail untuk soalan ini.")
                continue
            mfile = manager_file()
            if mfile is None:
                print(f"pengurus {MANAGER_NAME} tiada — muat turun dahulu.")
                continue
            print(f"{len(cands)} pakar {[m['name'] for m in cands]} akan jawab, "
                  f"{MANAGER_NAME} semak + putuskan.")
            cli = cli_path or ensure_cli()
            from agent import clean as _clean3
            parts = []
            wrows = []
            for m in cands:
                print(f"\n=== [{m['name']}] ===")
                try:
                    tmp = LlamaCliBackend(cli, MODELS_DIR / m["file"], n_predict=250)
                    esys = (f"Anda pakar {','.join(m.get('domains', []))}. Jawab ringkas."
                            if lang == "ms" else
                            f"You are a {','.join(m.get('domains', []))} expert. Answer briefly.")
                    import time as _t3
                    t0 = _t3.time()
                    rraw, rtps = tmp.chat(esys, [{"role": "user", "content": wtask}])
                    rans = _clean3(sanitize(rraw), name)
                except KeyboardInterrupt:
                    print("(dihenti — pakar berbaki dilangkau)")
                    break
                except Exception as e:
                    print(f"(ralat: {e})")
                    continue
                el = _t3.time() - t0
                print(rans)
                if rtps:
                    print(f"\x1b[2m[{m['name']}] {rtps:.1f} T/s ({el:.0f}s)\x1b[0m")
                    record_speed(MODELS_DIR, m["name"], rtps)
                wrows.append((m["name"], rtps, el))
                parts.append((m["name"], rans))
                log_training(wtask, rans, lang=lang, kind="kerja",
                             domain=",".join(m.get("domains", [])))
            if not parts:
                print("tiada jawapan pakar untuk disemak.")
                continue
            vsys = ("Anda pengurus dan penyemak. Di bawah jawapan beberapa pakar untuk soalan pengguna. "
                    "Untuk SETIAP pakar beri: skor /10 + satu ayat (betul? salah di mana?). "
                    "Kemudian tulis JAWAPAN AKHIR yang digabung dan diperbetul, dalam Bahasa Melayu. "
                    "Format: baris `Nama: skor/10 - ulasan` setiap pakar, baris kosong, `KEPUTUSAN:` + jawapan. "
                    "Jangan tulis proses berfikir; terus ke format."
                    if lang == "ms" else
                    "You are the manager and verifier. Below are answers from several experts. "
                    "For EACH expert give: score /10 + one sentence (correct? where wrong?). "
                    "Then write the merged corrected FINAL ANSWER in English. "
                    "Format: `Name: score/10 - review` per expert, blank line, `VERDICT:` + answer. "
                    "No thinking process; go straight to the format.")
            vuser = wtask + "\n\n" + "\n\n".join(f"[{n}]: {a}" for n, a in parts)
            print(f"\n[{MANAGER_NAME} menyemak...]")
            try:
                _v0 = _t3.time()
                vraw, vtps = LlamaCliBackend(cli, mfile,
                                             n_predict=args.max_tokens).chat(vsys, [{"role": "user",
                                                                                    "content": vuser}])
                _vel = _t3.time() - _v0
                verdict = _clean3(sanitize(vraw), name)
            except Exception as e:
                print(f"Ralat semakan: {e}")
                continue
            history += [{"role": "user", "content": wtask},
                        {"role": "assistant", "content": verdict}]
            save_turn(session, history[-2:])
            log_training(wtask, verdict, lang=lang, kind="kerja-verdict",
                         tools_trace=[{"tool": "kerja", "arg": n, "result": a[:200]}
                                      for n, a in parts])
            last_qa = [wtask, verdict, detect_domain(wtask)]
            if vtps:
                record_speed(MODELS_DIR, MANAGER_NAME, vtps)
            print(f"\n{name} (keputusan {MANAGER_NAME})> {verdict}")
            if vtps:
                print(f"\x1b[2m[{MANAGER_NAME}] {vtps:.1f} T/s ({_vel:.0f}s)\x1b[0m")
                wrows.append((MANAGER_NAME, vtps, _vel))
            print_speed_recap(wrows)
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
        if q == "/new" or q.startswith("/new "):
            import time as _t
            session = q.split(None, 1)[1] if len(q.split()) > 1 else f"sesi-{_t.strftime('%H%M')}"
            history = []
            last_qa = [None, None, None]
            print(f"session baharu: {session} (konteks kosong)")
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
        if q == "/sambung":
            if not last_qa[0]:
                print("Tiada apa untuk disambung lagi.")
                continue
            q = ("Sambung jawapan tergantung anda di atas dari tepat mana ia berhenti."
                 if lang == "ms" else
                 "Continue your cut-off answer above from exactly where it stopped.")
        if q.startswith("/"):
            print("Perintah tidak dikenali. Taip /help untuk senarai.")
            continue

        # Compact thinking line: xc berfikir (base) ... + animated dots (classic TTY only).
        # TUI viewport cannot do \r overwrite — static line there.
        import threading as _th
        import time as _tm
        _is_tui = getattr(sys.stdout, "__class__", object).__name__ == "TuiStdout"
        _stop = _th.Event()
        _th_obj = None
        if not _is_tui:
            def _spin(_s=short, _m=cur.get("name", ""), _ev=_stop):
                _pats = [".  ", ".. ", "..."]
                _i = 0
                while not _ev.is_set():
                    try:
                        sys.stdout.write(f"\r{_s} berfikir ({_m}) {_pats[_i % 3]}")
                        sys.stdout.flush()
                    except Exception:
                        break
                    _i += 1
                    _ev.wait(0.3)
            _th_obj = _th.Thread(target=_spin, daemon=True)
            _th_obj.start()
        else:
            # Leading \n separates from "> hai" partial in TUI viewport
            # (view.add concatenates partial without newline otherwise).
            print(f"\n{short} berfikir ({cur['name']}) ...")
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
                if auto_route and backend.kind in ("llama-cli", "llama", "server"):
                    mg = pick_manager_open(q)
                    if mg is not None and cur.get("name") != MANAGER_NAME:
                        switch_to(mg, GENERAL)
                        print(f"(soalan terbuka — pengurus {MANAGER_NAME} mengambil alih)")
                    elif dom != GENERAL and dom not in cur.get("domains", []):
                        cand = pick_expert(dom, detect_stack(q))
                        if cand:
                            switch_to(cand, dom)
                        else:
                            print(f"(tiada pakar {dom} — jawab dengan {cur['name']}; "
                                  f"latih: train --mode sft --domain {dom})")
                try:
                    msize = "4.5B" if "gemma" in cur.get("name", "") else "0.5B"
                    ans, tps, trace = run_agent(backend, q, name=name, lang=lang, history=history,
                                                tools_on=tools_on, tool_ctx=tool_ctx, on_tool=on_tool,
                                                expert_ctx=expert_context(cur) if auto_route else "",
                                                on_think=on_think,
                                                size=msize)
                except Exception as e:
                    _stop.set()
                    if _th_obj is not None:
                        try:
                            sys.stdout.write("\r" + " " * 60 + "\r")
                        except Exception:
                            pass
                    print(f"Ralat: {e}")
                    continue
        # stop spinner, clear line
        _stop.set()
        if _th_obj is not None:
            try:
                sys.stdout.write("\r" + " " * 60 + "\r")
            except Exception:
                pass
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        save_turn(session, history[-2:])
        log_training(q, ans, lang=lang, kind="chat", tools_trace=trace, domain=dom)
        last_qa = [q, ans, dom]
        if tps:
            record_speed(MODELS_DIR, cur.get("name", ""), tps)
        print(f"{short}> {ans}")
        if tps:
            s = f"({cur['name']} | {tps:.1f} T/s)"
            print(f"\x1b[2m{s}\x1b[0m")
        else:
            s = f"({cur['name']})"
            print(f"\x1b[2m{s}\x1b[0m")
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
    from hardware import auto_budget, cpu_count
    ram, auto_tok, auto_ctx, auto_thr = auto_budget()
    if not args.max_tokens:
        args.max_tokens = auto_tok
    if not args.ctx:
        args.ctx = auto_ctx
    if not args.threads:
        args.threads = auto_thr
    if args.ctx > 32768:  # Qwen2.5-0.5B hard ceiling (native 32k)
        print(f"[auto] ctx {args.ctx} melebihi siling model 32768 — dihadkan.")
        args.ctx = 32768
    if args.max_tokens > args.ctx:
        print(f"[auto] max_tokens {args.max_tokens} > ctx — dihadkan kepada ctx.")
        args.max_tokens = args.ctx
    if args.cmd != "train" and not args.once:
        print(f"[auto] RAM {ram:.1f}GB CPU {cpu_count()} -> backend={resolve_backend_name(args)} "
              f"max_tokens={args.max_tokens} ctx={args.ctx} threads={args.threads} (tetapkan manual untuk ubah)")
    if args.cmd == "train":
        flt = (args.domain or "").strip().lower() or None
        corp_dest = f"finetune/stacks/{flt or 'general'}/corpus.txt"
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
                      f"--domains {flt}. See STEP.md.")
            else:
                print("Import: /model add <fail> --name xcoder-base --domains general. See STEP.md.")
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
                                expert_ctx=expert_context(cur) if args.autoroute else "",
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
