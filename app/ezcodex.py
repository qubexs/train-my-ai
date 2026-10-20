"""XCoder - opencode-style agentic CLI (stdlib only).

Backends: resident server (default interaktif, 0 reload) | llama-cli (--once/skrip) | lmstudio
Tools: list/read/write/edit/run/bash/rag with workspace jail + confirm.
Sessions: app/data/chat.db | Training flywheel: app/data/knowledge.db (SQLite)
Train: `py app/ezcodex.py train --mode sft|pretrain --export-only` (+ Colab scripts in finetune/)
Build exe: `py app/build_exe.py` (PyInstaller, win/linux/macos)

Compat: old flags --lang/--name/--setup still work. Chat is now agentic.
"""
import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

APP_DIR = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
           else Path(__file__).resolve().parent)
sys.path.insert(0, str(APP_DIR))

__version__ = "0.4.52"

from fetch import (LLAMA_TAG, LLAMA_BASE, HF_MODEL_URL, MODEL_SIZE,
                   asset_name, download, verify, sha256_file,
                   ensure_cli as _ensure_cli, ensure_model as _ensure_model,
                   cuda_dir as _cuda_dir, ensure_cuda as _ensure_cuda,
                   vulkan_dir as _vulkan_dir, ensure_vulkan as _ensure_vulkan)

from backends import LlamaCliBackend, LmStudioBackend
from agent import run_agent, sanitize
from experts import (GENERAL, add as add_model, detect as detect_domain,
                     detect_stack, fmt_size, gguf_ctx, resolve as resolve_model,
                     scan as scan_models, record_speed,
                     infer_name_domains, resolve_layout,
                     swap_pending_models)
from store import load_session, save_turn, list_sessions, log_training, export_sft, export_pretrain_corpus
from store import clear_session, context_stats
from store import (train_state, pending_since_train, build_delta_file,
                   mark_trained, mark_failed)

BIN_DIR, MODELS_DIR = resolve_layout(APP_DIR)

# Solo: app guna model ini sahaja (tiada relay/tukar pakar). Muat turun
# automatik dari HF_MODEL_URL (fetch.py) jika tiada. Set None untuk mod lama.
ONLY_MODEL = "xcoder-gen1-0.8b-q4_k_m.gguf"


def ensure_cli():
    return _ensure_cli(BIN_DIR)


def ensure_model():
    return _ensure_model(MODELS_DIR)


def only_entry():
    """Entri registry model solo (ONLY_MODEL) jika ada fail, else None."""
    if not ONLY_MODEL:
        return None
    for m in scan_models(MODELS_DIR):
        f = m.get("file") or ""
        if f == ONLY_MODEL and f and (MODELS_DIR / f).is_file():
            return m
    return None


HELP = """Commands:
/help                 this help
/lang ms|en           switch language (clears turn cache, keeps session file)
/tools on|off         enable/disable agentic tool loop (Tab = suis Build/Plan)
/online on|off        carian internet + auto-belajar ke docs/learned (lalai: on)
/search <soalan>      cari web sekarang + simpan nota pengetahuan (tanpa model)
/learned              bilangan nota pembelajaran + 5 terkini
/plan <tugas>         numbered build plan (maks 6 langkah), then build file by file
/bina <tugas>         laksana plan automatik langkah demi langkah (maks 8, tukar pakar setiap langkah)
(fikiran AI sentiasa ditunjuk automatik sebelum jawapan — tiada arahan khas)
/backend llama|lmstudio|server|auto  tukar enjin (auto=server interaktif, llama --once)
/server               status resident llama-server (backend server sahaja)
/update               semak git + model lokal/HF baharu, tawar pull/daftar/muat turun
/model <nama|path|id>  tukar model pakar (satu aktif pada satu masa)
/model add <url|fail> [--name N] [--domains a,b]  daftar model baharu
/model import [fail] [--from dir]  auto-import GGUF terbaru + aktifkan
/models               senarai semua model pakar + domain
/models check         audit GGUF (hash, sah, duplikat, registry)
(Solo: xcoder-gen1-think jawab semua soalan — tiada relay/tukar pakar automatik.)
/route on|off         auto-tukar pakar ikut domain (dimatikan dalam mod solo)
/session <name>       switch session (chat.db)
/sessions             list sessions
/new [nama]           session baharu segar (kosongkan konteks)
/context              show context usage of current session
/clear                clear in-memory history AND session db
/good | /bad          rate last answer -> dataset (knowledge.db)
/train-log            show dataset count (knowledge.db)
/export sft|corpus    pecah ke datasets/<stack> + gabung finetune/ (atau corpus.txt)
/retrain [status|full]  latih delta model dgn data baharu (auto pada 200 baris/50 nota)
/sambung | /continue   sambung jawapan terpotong terakhir
/cwd                  print workspace root
/keys                 diagnostik kekunci (paparkan nama setiap tekanan, Esc keluar)
/keluar|/exit|/quit   exit
Inline tool (when /tools on): model may emit ```tool:read docs/js-basics.md``` etc.
Tools: list read write edit run bash rag (workspace-jailed, bash asks confirm).
"""


def cuda_dir():
    return _cuda_dir(APP_DIR)


def ensure_cuda():
    return _ensure_cuda(APP_DIR)


def vulkan_dir():
    return _vulkan_dir(APP_DIR)


def ensure_vulkan():
    return _ensure_vulkan(APP_DIR)


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
    ap.add_argument("--gpu", default="off", choices=["off", "cuda", "vulkan", "auto"],
                    help="server/llama GPU offload (auto: NVIDIA->cuda, Intel/AMD->vulkan)")
    ap.add_argument("--model", default="", help="override GGUF path (llama) or model id (lmstudio)")
    ap.add_argument("--session", default="default")
    ap.add_argument("--no-tools", action="store_true", help="disable agentic tool loop")
    ap.add_argument("--allow-all", action="store_true", help="skip bash confirm (dangerous)")
    ap.add_argument("--offline", action="store_true",
                    help="mod offline: matikan carian internet + auto-belajar")
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

# Dicapai TUI: repl() daftar closure toggle di sini supaya TAB berfungsi.
TUI_TOGGLE = {"fn": None}


def _tui_on_tab():
    fn = TUI_TOGGLE.get("fn")
    return fn() if fn else None


def resolve_gpu(args):
    """Flavor backend: 'cpu' | 'cuda' | 'vulkan'. 'auto' = kesan perkakasan."""
    g = getattr(args, "gpu", "off")
    if g == "auto":
        from hardware import recommended_gpu
        g = recommended_gpu()
    return g if g in ("cuda", "vulkan") else "cpu"


def get_manager(args):
    global _MANAGER
    if _MANAGER is None:
        from server import ModelManager
        flavor = resolve_gpu(args)
        if flavor == "cuda":
            bindir = ensure_cuda()
        elif flavor == "vulkan":
            bindir = ensure_vulkan()
        else:
            bindir = BIN_DIR
        _MANAGER = ModelManager(bindir, port=args.port, ctx=args.ctx,
                                threads=args.threads, idle_timeout=args.idle_timeout,
                                flavor=flavor,
                                ngl=99 if flavor in ("cuda", "vulkan") else 0)
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
        return ServerBackend(mgr, model="xcoder", max_tokens=args.max_tokens,
                             model_path=str(model))
    cli = cli_path or ensure_cli()
    model = Path(args.model) if args.model else (model_path or ensure_model())
    return LlamaCliBackend(cli, model, ctx=args.ctx, n_predict=args.max_tokens)


def ctx_for(args, path):
    """ctx selamat untuk model: min(args.ctx, native GGUF); args.ctx tanpa metadata."""
    native, _ = gguf_ctx(path) if Path(path).is_file() else (0, 0)
    return min(args.ctx, native) if native else args.ctx


def model_size(name, file=""):
    """Label saiz untuk system prompt, dari nama + fail model (base kini 1.5B)."""
    n = ((name or "") + " " + (file or "")).lower()
    if "gemma" in n or "4.5b" in n:
        return "4.5B"
    if "0.8b" in n:
        return "0.8B"
    if "3.5" in n or re.search(r"(?<![\d.])2b", n):
        return "2B"
    if "1.5b" in n:
        return "1.5B"
    return "0.5B"


def is_thinking(name, file=""):
    """Kesan model penaakulan (Qwen3.5 kes berfikir panjang, perlu bajet 2x)."""
    n = ((name or "") + " " + (file or "")).lower()
    return ("3.5" in n or "0.8b" in n or "think" in n or "reason" in n or "qwq" in n)


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


def pick_base():
    """Model utama (base) — tempat kembali bila soalan general.
    Utamakan 'base' berfail, kemudian 'xcoder-general', kemudian mana-mana general berfail."""
    cands = [m for m in scan_models(MODELS_DIR)
             if m.get("file") and (MODELS_DIR / m["file"]).is_file()
             and GENERAL in m.get("domains", [])]
    if not cands:
        return None
    for prefer in ("xcoder-gen1-think", "xcoder-best", "base", "xcoder-general"):
        for m in cands:
            if m.get("name") == prefer:
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
             "keluar", "exit", "quit", "good", "bad", "train-log",
             "online", "search", "learned", "retrain")

# English aliases for Malay commands
ALIASES = {"/continue": "/sambung"}


def do_sft_export(flt):
    """knowledge.db -> datasets/<stack>.jsonl -> finetune/stacks/<stack>/. Returns text."""
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


def reveal(prefix, text, enabled=True):
    """Cetak jawapan dengan kesan menaip (typewriter) — visual seperti streaming.
    Output paip/skrip cetak sekaligus (tiada lengah). Berfungsi di REPL klasik
    dan TUI sebab kedua-duanya melalui sys.stdout.write."""
    import re as _re
    import time as _t
    _is_tui = getattr(sys.stdout, "__class__", object).__name__ == "TuiStdout"
    try:
        _tty = bool(sys.stdout.isatty())
    except Exception:
        _tty = False
    sys.stdout.write(prefix)
    sys.stdout.flush()
    if enabled and text and (_is_tui or _tty):
        step = max(24, len(text) // 60 or 1)
        buf = ""
        for w in _re.split(r"(\s+)", text):
            buf += w
            if len(buf) >= step:
                sys.stdout.write(buf)
                sys.stdout.flush()
                buf = ""
                _t.sleep(0.03)
        if buf:
            sys.stdout.write(buf)
            sys.stdout.flush()
    else:
        sys.stdout.write(text or "")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def repl(args, shared=None):
    # Aktifkan GGUF baharu (*.gguf.new) yg menunggu - hanya bila fail lama TIDAK
    # dikunci (proses lain / model sedang digunakan = langkau, cuba lagi nanti).
    try:
        for _n in swap_pending_models(MODELS_DIR):
            print(f"[model] GGUF baharu diaktifkan: {_n}")
    except Exception:
        pass
    cli_path = model_path = None
    if args.backend in ("llama", "auto"):
        cli_path, model_path = ensure_cli(), ensure_model()
    elif args.backend == "server":
        model_path = ensure_model()
    backend = make_backend(args, cli_path, model_path)
    name, lang = args.name, args.lang
    tools_on = not args.no_tools
    online = not args.offline
    auto_route = args.autoroute
    session = args.session
    # Fresh session on every app start (default): jangan sambung default.jsonl lama.
    # Explicit --session <nama> masih dihormati; /new dan /session masih boleh tukar.
    if session == "default":
        import time as _ts
        session = f"sesi-{_ts.strftime('%H%M%S')}"
    history = scrub_history(load_session(session)[-20:])
    cur = match_cur(model_path)
    solo = only_entry() if not args.model else None
    if solo is not None:
        auto_route = False  # solo: tiada relay/auto-tukar pakar
        cur = {"name": solo["name"], "domains": solo.get("domains", [GENERAL])}
    short = (name[:2].lower() if name and len(name) >= 2 else (name or "xc").lower())
    tool_ctx = {"allow_all": args.allow_all,
                "confirm": lambda msg: input(msg).strip().lower() in ("y", "yes"),
                "rag_k": 2, "lang": lang, "autolearn": True}
    print(f"{name} v{__version__} ready [{backend.kind}, model={cur['name']}, tools={'on' if tools_on else 'off'}, "
          f"online={'on' if online else 'off'}, "
          f"route={'on' if auto_route else 'off'}{', solo' if solo else ''}, "
          f"ctx={args.ctx}, session={session}]. Taip /help\n")
    try:
        from tools import _rag_index
        _rag_index()  # prewarm cache RAG semasa startup, bukan semasa tool dipanggil
    except Exception:
        pass
    ROOT = APP_DIR.parent if (APP_DIR / "ezcodex.py").exists() else APP_DIR
    try:
        from updatecheck import summary as _upd_summary
        _banner = _upd_summary(ROOT, MODELS_DIR)
        if _banner:
            print(_banner + "\n")
    except Exception:
        pass

    # --- Auto delta-retrain: latar belakang, elak ganggu proses lain ---
    gpu_cache = None

    def gpu_python():
        for cand in (ROOT / ".venv-gpu" / "Scripts" / "python.exe",
                     ROOT / ".venv-gpu" / "bin" / "python"):
            if cand.is_file():
                return cand
        return None

    def gpu_ok(exe):
        nonlocal gpu_cache
        if gpu_cache is None:
            try:
                r = subprocess.run([str(exe), "-c",
                                    "import torch;print(torch.cuda.is_available())"],
                                   capture_output=True, text=True, timeout=300)
                gpu_cache = "True" in (r.stdout or "")
            except Exception:
                gpu_cache = False
        return gpu_cache

    train_job = {"proc": None, "log": None}

    def start_retrain(force=False):
        proc = train_job.get("proc")
        if proc is not None and proc.poll() is None:
            print("[retrain] latihan latar sedang berjalan.")
            return
        p = pending_since_train()
        if not force and not p["ready"]:
            return
        st = train_state()
        if not force and st["last_failed_ts"] and \
                time.time() - st["last_failed_ts"] < 86400:
            print("[retrain] gagal <24j lepas - data kekal, cuba lagi kemudian.")
            return
        exe = gpu_python()
        if exe is None:
            print("[retrain] tiada .venv-gpu - data kekal dalam knowledge.db "
                  "(delta auto dilangkau; lihat README: delta retrain).")
            return
        master = ROOT / "finetune" / "masters" / "xcoder-gen-0.8b"
        if not (master / "adapter_model.safetensors").is_file():
            print(f"[retrain] master adapter tiada di {master} - salin "
                  f"drive_upload\\DriveXcoder\\xcoder\\v2\\checkpoint-* ke situ "
                  f"sekali sahaja (README: delta retrain).")
            return
        if not gpu_ok(exe):
            print("[retrain] tiada GPU (CUDA) - data kekal dalam knowledge.db.")
            return
        dest, nd, nr, note = build_delta_file()
        if dest is None:
            print(f"[retrain] tiada data delta ({note}).")
            return
        out_dir = ROOT / "build" / "train" / \
            f"xcoder-gen-0.8b-delta-{time.strftime('%Y%m%d-%H%M%S')}"
        logp = ROOT / "finetune" / ".train_run.log"
        logp.parent.mkdir(parents=True, exist_ok=True)
        log = logp.open("a", encoding="utf-8")
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                 if hasattr(subprocess, "DETACHED_PROCESS") else 0)
        cmd = [str(exe), str(ROOT / "finetune" / "train_delta_qwen35.py"),
               "--data", str(dest), "--master", str(master), "--out", str(out_dir)]
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log,
                                stderr=subprocess.STDOUT, creationflags=flags)
        train_job["proc"], train_job["log"] = proc, log
        print(f"[retrain] latihan latar bermula: delta={nd}, replay={nr} "
              f"(log: finetune\\.train_run.log) - chat tetap guna model semasa.")

        def watch():
            rc = proc.wait()
            try:
                log.close()
            except Exception:
                pass
            if rc == 0:
                try:
                    st2 = mark_trained()
                    sw = swap_pending_models(MODELS_DIR)
                    if sw:
                        msg = (f"[retrain] SIAP (round {st2['delta_rounds']}): "
                               f"model baharu aktif: {', '.join(sw)}. "
                               f"Backend server yg sudah memuat model lama: mula "
                               f"semula app utk muat GGUF baharu.")
                    else:
                        msg = (f"[retrain] SIAP (round {st2['delta_rounds']}); "
                               f"GGUF .new sedia - diaktifkan automatik bila model "
                               f"tidak digunakan (mula semula app).")
                except Exception as e:
                    msg = f"[retrain] siap tetapi ralat susulan: {e}"
            elif rc == 2:
                msg = "[retrain] tiada GPU - data kekal dalam knowledge.db."
            elif rc == 3:
                msg = "[retrain] latihan lain sedang berjalan - cuba lagi nanti."
            else:
                try:
                    mark_failed()
                except Exception:
                    pass
                msg = "[retrain] gagal - semak finetune\\.train_run.log (data kekal)."
            print("\n" + msg + "\n")
        threading.Thread(target=watch, daemon=True).start()

    start_retrain()

    def ask_yn(msg):
        try:
            return input(msg + " [y/N] ").strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def on_tool(tname, targ, result):
        print(f"[tool:{tname}] {targ[:120]}")
        print((result[:600] + ("..." if len(result) > 600 else "")) + "\n")

    def on_think(thinking):
        import re as _re
        thinking = _re.sub(r"(?i)\b(gemma|google deepmind|google)\b", name, thinking)
        print("\x1b[2m" + f"[berfikir]\n{thinking[:800]}" + "\x1b[0m")

    width = shutil.get_terminal_size((80, 20)).columns
    last_qa = [None, None, None]  # (question, answer, domain) for /good /bad

    def toggle_tools():
        """Flip tools_on untuk TAB TUI. Returns status text."""
        nonlocal tools_on
        tools_on = not tools_on
        if shared is not None:
            shared["tools"] = tools_on
        return f"[tools {'on' if tools_on else 'off'}]"
    TUI_TOGGLE["fn"] = toggle_tools

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
            _mm = MODELS_DIR / cand["file"]
            backend = LlamaCliBackend(cli_path, _mm, ctx=ctx_for(args, _mm),
                                      n_predict=args.max_tokens)
        cur = {"name": cand["name"], "domains": cand.get("domains", [dom])}
        print(f"-> route: {cur['name']} [{dom}] (satu model aktif)")

    def cur_size():
        """Label saiz model aktif (registry fail, bukan tekaan nama)."""
        hit = next((x for x in scan_models(MODELS_DIR)
                    if x.get("name") == cur.get("name")), {})
        return model_size(cur.get("name", ""), hit.get("file", ""))

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
            pbe = LlamaCliBackend(cli_path, mfile, ctx=ctx_for(args, mfile), n_predict=400)
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
        # Tab (TUI) togol shared["tools"] — selaras senyap, bar bawah sudah tunjuk mod.
        if shared is not None and shared.get("tools", tools_on) != tools_on:
            tools_on = shared["tools"]
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
            tool_ctx["lang"] = lang
            print(f"Bahasa: {lang}")
            continue
        if q.startswith("/tools"):
            tools_on = q.split()[-1] != "off"
            if shared is not None:
                shared["tools"] = tools_on
            print(f"tools={'on' if tools_on else 'off'}")
            continue
        if q.startswith("/online"):
            online = q.split()[-1] != "off"
            tool_ctx["autolearn"] = online
            print(f"online={'on' if online else 'off'}")
            continue
        if q == "/search" or q.startswith("/search "):
            sq = q[len("/search"):].strip()
            if not sq:
                print("guna: /search <soalan>")
                continue
            from websearch import search as _ws
            from tools import maybe_learn as _ml
            print("... mencari ...")
            try:
                res = _ws(sq, lang=lang)
            except Exception as e:
                print(f"Ralat carian: {e}")
                continue
            print(res)
            p = _ml(tool_ctx, sq, res, lang=lang)
            if p:
                print(f"[learned] docs/learned/{p.name}  (/export sft utk latihan)")
            continue
        if q == "/learned":
            from store import list_learned, learned_count
            print(f"{learned_count()} nota (RAG docs/learned + knowledge.db)")
            for r in list_learned(5):
                print(f"  {r.get('file') or '?'}  <- {r.get('url') or '-'}")
            continue
        if q == "/retrain" or q.startswith("/retrain "):
            arg = q[len("/retrain"):].strip().lower()
            if arg == "status":
                p = pending_since_train()
                st = train_state()
                print(f"delta sedia: {p['rows']} baris + {p['notes']} nota "
                      f"(auto pada 200/50) | rounds={st['delta_rounds']}")
                if st["last_train_ts"]:
                    print("  latihan terakhir: " +
                          time.strftime("%Y-%m-%d %H:%M",
                                        time.localtime(st["last_train_ts"])))
                news = [n.name for n in MODELS_DIR.glob("*.gguf.new")]
                if news:
                    print("  GGUF menunggu aktivasi: " + ", ".join(news))
                if train_job["proc"] is not None and train_job["proc"].poll() is None:
                    print("  latihan sedang berjalan (latar)")
                continue
            if arg.startswith("full"):
                print("full re-baseline dijalankan manual (GPU keras): "
                      ".venv-gpu\\Scripts\\python.exe finetune\\train_local_qwen35.py "
                      "--data all  - delta automatik guna train_delta_qwen35.py")
                continue
            p = pending_since_train()
            if not p["ready"]:
                print(f"paksa latihan delta: {p['rows']} baris + {p['notes']} "
                      f"nota sedia (/retrain status utk butiran)")
            start_retrain(force=True)
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
                if auto_route and backend.kind in ("llama-cli", "llama", "server"):
                    if sdom == GENERAL and (GENERAL not in cur.get("domains", [])
                                            or cur.get("name") == MANAGER_NAME):
                        _bb = pick_base()
                        if _bb is not None and _bb.get("name") != cur.get("name"):
                            switch_to(_bb, GENERAL)
                            print(f"-> kembali: {_bb['name']} [general] (base utama)")
                    elif sdom != GENERAL and sdom not in cur.get("domains", []):
                        cand = pick_expert(sdom, detect_stack(sq))
                        if cand:
                            switch_to(cand, sdom)
                        else:
                            print(f"(tiada pakar {sdom} — terus dengan {cur['name']})")
                print(f"{short} berfikir ({cur['name']}) ...")
                try:
                    msize = cur_size()
                    ans, tps, trace = run_agent(backend, sq, name=name, lang=lang, history=history,
                                                tools_on=tools_on, tool_ctx=tool_ctx, on_tool=on_tool,
                                                expert_ctx=expert_context(cur) if auto_route else "",
                                                on_think=on_think,
                                                size=msize, online=online)
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
                reveal(f"{short}> ", ans)
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
                backend = LlamaCliBackend(cli_path, model_path,
                                          ctx=ctx_for(args, model_path),
                                          n_predict=args.max_tokens)
            elif b == "lmstudio":
                backend = LmStudioBackend(args.lm_url, args.lm_model, max_tokens=args.max_tokens)
            elif b == "server":
                from backends import ServerBackend
                model_path = ensure_model()
                mgr = get_manager(args)
                already, secs = mgr.load(model_path)
                if not already:
                    print(f"[server] model dimuat dalam {secs:.1f}s")
                backend = ServerBackend(mgr, model="xcoder", max_tokens=args.max_tokens,
                                        model_path=str(model_path))
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
        if q == "/update":
            from updatecheck import (git_pull_ff, git_state, hf_new_files,
                                     local_new_models)
            print("--- git ---")
            gs = git_state(ROOT)
            if not gs["repo"]:
                print("bukan repo git (exe diagih tanpa .git) — langkau git.")
            else:
                print(f"dirty={gs['dirty']} ahead={gs['ahead']} behind={gs['behind']}")
                if gs["behind"] and not gs["dirty"]:
                    if ask_yn(f"pull {gs['behind']} komit (fast-forward)?"):
                        ok, msg = git_pull_ff(ROOT)
                        print(("OK: " if ok else "GAGAL: ") + msg)
                elif gs["behind"]:
                    print("remote mendahului — commit/stash dahulu, kemudian /update semula.")
                elif not gs["dirty"] and not gs["ahead"]:
                    print("git terkini.")
            print("--- model lokal baharu ---")
            new_local = local_new_models(MODELS_DIR)
            if not new_local:
                print("tiada GGUF tidak berdaftar.")
            else:
                for f in new_local:
                    print(f"  {f}")
                if ask_yn(f"daftar {len(new_local)} model?"):
                    for f in new_local:
                        try:
                            e = add_model(MODELS_DIR, str(MODELS_DIR / f))
                            print(f"OK {e['name']} [{','.join(e.get('domains', []))}]")
                        except SystemExit as ex:
                            print(f"TOLAK {f}: {ex}")
            print("--- HF baharu (myxcoder/xcoder) ---")
            new_hf, note = hf_new_files(MODELS_DIR, force=True)
            print(f"sumber: {note}")
            if not new_hf:
                print("tiada fail baharu di HF.")
            else:
                from fetch import download as _dl
                base = "https://huggingface.co/myxcoder/xcoder/resolve/main"
                show = new_hf[:20]
                for f in show:
                    print(f"  {f}")
                if len(new_hf) > len(show):
                    print(f"  ... +{len(new_hf) - len(show)} lagi")
                if ask_yn(f"muat turun {len(new_hf)} fail?"):
                    for f in new_hf:
                        try:
                            dest = MODELS_DIR / f.rsplit("/", 1)[-1]
                            _dl(f"{base}/{f}", dest)
                            e = add_model(MODELS_DIR, str(dest))
                            print(f"OK {e['name']}")
                        except SystemExit as ex:
                            print(f"GAGAL {f}: {ex}")
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
                    backend = LlamaCliBackend(cli_path, p2, ctx=ctx_for(args, p2),
                                              n_predict=args.max_tokens)
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
                backend = LlamaCliBackend(cli_path, p, ctx=ctx_for(args, p),
                                          n_predict=args.max_tokens)
            hit = next((x for x in scan_models(MODELS_DIR)
                        if x["name"].lower() == rest.lower()), None)
            cur = {"name": hit["name"] if hit else rest,
                   "domains": hit.get("domains", [GENERAL]) if hit else [GENERAL]}
            print(f"model={cur['name']} [{','.join(cur['domains'])}] {note}".rstrip())
            continue
        if q.startswith("/route"):
            if only_entry() is not None:
                auto_route = False
                print(f"solo: route dimatikan (hanya {ONLY_MODEL})")
                continue
            auto_route = q.split()[-1] != "off" if len(q.split()) > 1 else True
            print(f"route={'on' if auto_route else 'off'} (auto-tukar pakar ikut domain)")
            continue
        if q.startswith("/solo"):
            print(f"solo: {ONLY_MODEL or '(off)'} — tetapan kekal dalam kod (app/ezcodex.py)")
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
                  f"~{st['approx_tokens']} tokens (model ctx {args.ctx})")
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
                print("Logged ke knowledge.db")
            else:
                print("Nothing to rate yet.")
            continue
        if q == "/train-log":
            from store import KNOW_DB, dataset_count
            print(f"{KNOW_DB} : {dataset_count()} rows "
                  f"({dataset_count(consumed=0)} belum dilatih)")
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
        if q == "/keys":
            print("Tekan kekunci (Esc keluar):")
            try:
                import os as _os
                if _os.name == "nt":
                    from tui import WindowsReader as _R
                    rdr = _R()
                    while True:
                        k = rdr.read_key()
                        print(f"  {k!r}")
                        if k == "esc":
                            break
                else:
                    from tui import PosixReader as _R
                    with _R() as rdr:
                        while True:
                            k = rdr.read_key()
                            print(f"  {k!r}")
                            if k == "esc":
                                break
            except Exception as e:
                print(f"(tidak dapat baca kekunci: {e})")
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
            # Dot beranimasi: tulis-semula baris ini selagi model berfikir.
            _tpre = f"{short} berfikir ({cur['name']}) "
            print(f"\n{_tpre}...")
            _tui = getattr(sys.stdout, "ui", None)
            _tidx = len(_tui.view.lines) - 1 if _tui is not None else -1
            if _tui is not None and _tidx >= 0:
                def _tspin(_u=_tui, _idx=_tidx, _pre=_tpre, _ev=_stop):
                    _pats = ["... ", "..  ", ".   ", "..  "]
                    _i = 0
                    while not _ev.is_set():
                        try:
                            if _u.view.lines[_idx].startswith(_pre):
                                _u.view.update_line(_idx, _pre + _pats[_i % 4])
                                _u.draw()
                        except Exception:
                            break
                        _i += 1
                        _ev.wait(0.3)
                _th_obj = _th.Thread(target=_tspin, daemon=True)
                _th_obj.start()
        dom = detect_domain(q)
        already_shown, log_dom = False, dom
        self_ans = answer_self(q, cur, name, lang)
        off_ans = answer_offline(q, lang) if (self_ans is None and not online) else None
        if self_ans is not None:
            ans, tps, trace = self_ans, None, [{"tool": "models", "arg": "", "result": "registry"}]
        elif off_ans is not None:
            ans, tps, trace = off_ans, None, [{"tool": "local", "arg": q[:100], "result": "offline-note"}]
        else:
            loc = local_action(q, lang)
            if loc is not None:
                ans, tps, trace = loc, None, [{"tool": "local", "arg": q[:100], "result": "mkdir"}]
            else:
                already_shown = False
                log_dom = dom
                if auto_route and backend.kind in ("llama-cli", "llama", "server"):
                    mg = pick_manager_open(q)
                    if mg is not None:
                        if cur.get("name") != mg.get("name"):
                            switch_to(mg, GENERAL)
                            print(f"(soalan terbuka — pengurus {MANAGER_NAME} mengambil alih)")
                        msize = cur_size()
                        try:
                            ans, tps, trace = run_agent(backend, q, name=name, lang=lang, history=history,
                                                        tools_on=tools_on, tool_ctx=tool_ctx, on_tool=on_tool,
                                                        expert_ctx=expert_context(cur) if auto_route else "",
                                                        on_think=on_think,
                                                        size=msize, online=online)
                        except Exception as e:
                            _stop.set()
                            if _th_obj is not None:
                                try:
                                    sys.stdout.write("\r" + " " * 60 + "\r")
                                except Exception:
                                    pass
                            print(f"Ralat: {e}")
                            continue
                    else:
                        # LALAI: base jawab dulu, pakar berkaitan sambung bergilir
                        # hingga lengkap (unload automatik bila switch_to).
                        from experts import matching_stacks as _dms
                        need = [s for s in _dms(q) if s != GENERAL]
                        if not need and dom != GENERAL:
                            need = [dom]
                        cands = [m for m in scan_models(MODELS_DIR)
                                 if m.get("file") and (MODELS_DIR / m["file"]).is_file()
                                 and m.get("name") != MANAGER_NAME
                                 and any(d in need for d in m.get("domains", []))]

                        def _rkey(m):
                            doms = m.get("domains", [])
                            hit = next((d for d in doms if d in need), None)
                            return (0 if any(s in m["name"] for s in need) else 1,
                                    len(doms),
                                    need.index(hit) if hit is not None else 999)
                        cands.sort(key=_rkey)
                        legs, covered = [], set()
                        for m in cands:
                            new = {d for d in m.get("domains", []) if d in need} - covered
                            if new:
                                legs.append(m)
                                covered |= new
                            if len(legs) >= 3 or covered >= set(need):
                                break
                        # spinner awal sudah selesai tugas — hentikan sebelum streaming kaki.
                        _stop.set()
                        if not _is_tui and _th_obj is not None:
                            try:
                                sys.stdout.write("\r" + " " * 60 + "\r")
                            except Exception:
                                pass
                        base = pick_base()
                        if base is None:
                            print("(tiada model base berfail — jawab dengan model semasa)")
                            msize = cur_size()
                            try:
                                ans, tps, trace = run_agent(backend, q, name=name, lang=lang, history=history,
                                                            tools_on=tools_on, tool_ctx=tool_ctx, on_tool=on_tool,
                                                            expert_ctx=expert_context(cur) if auto_route else "",
                                                            on_think=on_think, size=msize, online=online)
                            except Exception as e:
                                print(f"Ralat: {e}")
                                continue
                        else:
                            if cur.get("name") != base["name"]:
                                switch_to(base, GENERAL)
                            import time as _t5
                            rrows = []
                            acc = []
                            try:
                                _b0 = _t5.time()
                                _bbig = (args.max_tokens * 2 if is_thinking(
                                    base.get("name", ""), base.get("file", "")) else None)
                                bans, btps, btrace = run_agent(
                                    backend, q, name=name, lang=lang, history=history,
                                    tools_on=tools_on, tool_ctx=tool_ctx, on_tool=on_tool,
                                    expert_ctx=expert_context(cur) if auto_route else "",
                                    on_think=on_think,
                                    size=model_size(base.get("name", ""), base.get("file", "")),
                                    max_tokens=_bbig, online=online)
                            except KeyboardInterrupt:
                                print("(dihenti)")
                                continue
                            except Exception as e:
                                print(f"Ralat: {e}")
                                continue
                            _bel = _t5.time() - _b0
                            acc.append("[base]:\n" + bans)
                            reveal(f"\n{name} (base)> ", bans)
                            if btps:
                                record_speed(MODELS_DIR, base["name"], btps)
                                print(f"\x1b[2m[{base['name']}] {btps:.1f} T/s\x1b[0m")
                            rrows.append((base["name"], btps, _bel))
                            log_training(q, bans, lang=lang, kind="chat-base",
                                         tools_trace=btrace, domain=GENERAL)
                            for m in legs:
                                switch_to(m, ",".join(m.get("domains", [])))
                                sq = (f"Soalan asal pengguna (WAJIB dijawab): {q}\n\nJawapan terkumpul setakat ini (konteks sahaja):\n"
                                      f"{chr(10).join(acc)}\n\nAnda pakar {','.join(m.get('domains', []))}. "
                                      f"HANYA tulis bahagian BARU yang belum diliputi (maksimum ~40 baris, kod penuh + ringkas). "
                                      f"DILARANG mengulang atau menulis semula jawapan dari awal. "
                                      f"Jika yang terkumpul menyeleweng dari soalan asal, "
                                      f"ABAIKAN ia dan jawab soalan asal. Jangan ulang apa yang sudah betul."
                                      if lang == "ms" else
                                      f"Original user question (MUST answer): {q}\n\nAccumulated answer so far (context only):\n"
                                      f"{chr(10).join(acc)}\n\nYou are the {','.join(m.get('domains', []))} expert. "
                                      f"Write ONLY the NEW not-yet-covered part (max ~40 lines, full code + brief). "
                                      f"FORBIDDEN to repeat or rewrite the answer from the start. "
                                      f"If the accumulated text drifted off-topic, IGNORE it and answer the original. "
                                      f"Do not repeat what is already correct.")
                                print(f"{short} berfikir ({m['name']}) ...")
                                try:
                                    _s0 = _t5.time()
                                    _lbig = (args.max_tokens * 2 if is_thinking(
                                        m.get("name", ""), m.get("file", "")) else None)
                                    eraw, etps, etrace = run_agent(
                                        backend, sq, name=name, lang=lang, history=history,
                                        tools_on=False, tool_ctx=tool_ctx, on_tool=None,
                                        expert_ctx=expert_context(cur) if auto_route else "",
                                        on_think=on_think,
                                        size=model_size(m.get("name", ""), m.get("file", "")),
                                        max_tokens=_lbig)
                                except KeyboardInterrupt:
                                    print("(dihenti — pakar berbaki dilangkau)")
                                    break
                                except Exception as e:
                                    print(f"(ralat {m['name']}: {e} — dilangkau)")
                                    continue
                                _eel = _t5.time() - _s0
                                acc.append(f"[{m['name']}]:\n" + eraw)
                                reveal(f"\n{name} ({m['name']} sambung)> ", eraw)
                                if etps:
                                    record_speed(MODELS_DIR, m["name"], etps)
                                    print(f"\x1b[2m[{m['name']}] {etps:.1f} T/s\x1b[0m")
                                rrows.append((m["name"], etps, _eel))
                                log_training(sq, eraw, lang=lang, kind="chat-leg",
                                             domain=",".join(m.get("domains", [])))
                                # kembali ke base: KUMPUL + SUSUN (collector/organizer)
                                switch_to(base, GENERAL)
                                oq = (f"Soalan asal pengguna: {q}\n\nJawapan terkumpul setakat ini:\n"
                                      f"{chr(10).join(acc)}\n\nAnda base XCoder (pengumpul/penyusun). "
                                      f"SUSUN SEMULA jawapan di atas jadi satu jawapan lengkap dan kemas: "
                                      f"buang ulangan, betulkan percanggahan (utamakan pakar), kekal ringkas. "
                                      f"Jawab penuh dalam Bahasa Melayu."
                                      if lang == "ms" else
                                      f"Original user question: {q}\n\nAccumulated answer so far:\n"
                                      f"{chr(10).join(acc)}\n\nYou are the base XCoder (collector/organizer). "
                                      f"REORGANIZE the above into one complete tidy answer: drop repetition, "
                                      f"fix conflicts (prefer the expert), stay brief. Answer fully in English.")
                                print(f"{short} berfikir (base) ...")
                                try:
                                    _o0 = _t5.time()
                                    merged, otps, otrace = run_agent(
                                        backend, oq, name=name, lang=lang, history=history,
                                        tools_on=False, tool_ctx=tool_ctx, on_tool=None,
                                        expert_ctx=expert_context(cur) if auto_route else "",
                                        on_think=on_think,
                                        size=model_size(base.get("name", ""), base.get("file", "")),
                                        max_tokens=_bbig)
                                except KeyboardInterrupt:
                                    print("(dihenti — pakar berbaki dilangkau)")
                                    break
                                except Exception as e:
                                    print(f"(ralat kumpul: {e} — kekal susunan sedia ada)")
                                    continue
                                _oel = _t5.time() - _o0
                                acc = [merged]
                                reveal(f"\n{name} (base)> ", merged)
                                if otps:
                                    record_speed(MODELS_DIR, base["name"], otps)
                                    print(f"\x1b[2m[base] {otps:.1f} T/s\x1b[0m")
                                rrows.append((base["name"], otps, _oel))
                                log_training(oq, merged, lang=lang, kind="chat-organize",
                                             domain=GENERAL)
                            ans = "\n\n".join(acc) if legs else bans
                            trace = btrace
                            already_shown = True
                            log_dom = ",".join(need) if need else GENERAL
                            if legs:
                                if cur.get("name") != base["name"]:
                                    switch_to(base, GENERAL)
                                print_speed_recap(rrows)
                            tps = None  # kelajuan sudah dipapar live + recap; elak duplikat di ekor
                else:
                    msize = cur_size()
                    try:
                        ans, tps, trace = run_agent(backend, q, name=name, lang=lang, history=history,
                                                    tools_on=tools_on, tool_ctx=tool_ctx, on_tool=on_tool,
                                                    expert_ctx=expert_context(cur) if auto_route else "",
                                                    on_think=on_think,
                                                    size=msize, online=online)
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
        log_training(q, ans, lang=lang, kind="chat", tools_trace=trace, domain=log_dom)
        last_qa = [q, ans, log_dom]
        if tps:
            record_speed(MODELS_DIR, cur.get("name", ""), tps)
        if not already_shown:
            reveal(f"{short}> ", ans)
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
        from hardware import report
        rep = report()
        gpu_s = ", ".join(f"{g['name']} [{g['vendor']}]" for g in rep["gpus"]) or "tiada"
        print(f"RAM {rep['ram_gb']}GB  CPU cores {rep['cpu']}  GPU {gpu_s}")
        print(f"Backend GPU disyorkan: {rep['recommended_gpu']} "
              f"(--gpu auto|cuda|vulkan|off)")
        return
    from hardware import auto_budget, cpu_count
    ram, auto_tok, auto_ctx, auto_thr = auto_budget()
    if not args.max_tokens:
        args.max_tokens = auto_tok
    if not args.threads:
        args.threads = auto_thr
    # Max ctx: guna n_ctx_train model (GGUF) — auto hadkan ikut RAM utk KV
    # (f16, ~25% RAM; ~1MB/1k token untuk 0.8B hybrid). Manual --ctx dihadkan
    # kepada native. Model tanpa metadata kekal siling lama 32768.
    hint = Path(args.model) if args.model else (MODELS_DIR / ONLY_MODEL)
    native, kv_bpt = gguf_ctx(hint) if hint.is_file() else (0, 0)
    if native:
        if not args.ctx:
            args.ctx = native
            if kv_bpt:
                budget = max(0.25 * ram * 1e9, 512e6)
                args.ctx = max(4096, min(native, int(budget / kv_bpt) // 4096 * 4096))
        else:
            args.ctx = min(args.ctx, native)
        if args.ctx < native:
            cap = f" (RAM had {int(0.25 * ram * 1e9 / 1e6)}MB KV)" if kv_bpt else ""
            print(f"[auto] ctx {args.ctx} daripada native {native}{cap}.")
    else:
        if not args.ctx or args.ctx > 32768:  # siling lama tanpa metadata
            if args.ctx > 32768:
                print(f"[auto] ctx {args.ctx} melebihi siling 32768 — dihadkan.")
            args.ctx = min(args.ctx or auto_ctx, 32768)
    if args.max_tokens > args.ctx:
        print(f"[auto] max_tokens {args.max_tokens} > ctx — dihadkan kepada ctx.")
        args.max_tokens = args.ctx
    if args.cmd != "train" and not args.once:
        from hardware import report
        rep = report()
        gpu_s = ", ".join(g["name"] for g in rep["gpus"]) or "tiada"
        print(f"[auto] RAM {rep['ram_gb']}GB CPU {cpu_count()} GPU {gpu_s} "
              f"-> backend={resolve_backend_name(args)} accel={rep['recommended_gpu']} "
              f"max_tokens={args.max_tokens} ctx={args.ctx} threads={args.threads} "
              f"(tetapkan manual untuk ubah)")
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
        off_ans = answer_offline(args.once, args.lang) if not args.offline else None
        if off_ans is not None:
            print(off_ans)
            return
        loc = local_action(args.once, args.lang)
        if loc is not None:
            print(loc)
            return
        ans, tps, _ = run_agent(backend, args.once, name=args.name, lang=args.lang,
                                tools_on=not args.no_tools,
                                tool_ctx={"allow_all": True, "rag_k": 2,
                                          "lang": args.lang, "autolearn": True},
                                expert_ctx=expert_context(cur) if args.autoroute else "",
                                online=not args.offline,
                                on_think=lambda t: print("\x1b[2m[berfikir]\n" + t[:800] + "\x1b[0m"))
        print(ans)
        if backend.kind == "server":
            backend.manager.stop()
        return
    if args.tui and sys.stdin.isatty() and sys.stdout.isatty():
        from tui import run_tui
        title = f" XCoder {__version__} [{args.backend}] lang={args.lang} "
        shared = {"tools": not args.no_tools}

        def status_fn(ui):
            # Bar bawah hidup: mod Plan/Build + (Done.) bila tiada lagi kandungan bawah.
            mode = "Build" if shared["tools"] else "Plan"
            s = (f" {mode} | PgUp/PgDn skrol | Up/Dn sejarah | Ctrl-D keluar ")
            if ui.view.scroll == 0:
                s += "(Done.)"
            return s

        run_tui(lambda: repl(args, shared), title, status_fn,
                on_tab=_tui_on_tab, shared=shared)
        return
    repl(args)


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        raise SystemExit("Perlukan Python 3.9+")
    main()
