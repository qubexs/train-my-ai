"""XCoder / XCoder-0.5B standalone app - NO LM Studio, NO Ollama, NO pip install.

Everything lives under the working folder:
  app/xcoder.py       this CLI (Python stdlib only)
  app/bin/            llama-cli binary, auto-downloaded on first run
  app/models/         GGUF model, auto-downloaded on first run (~491MB)

Run from the working folder:
  Windows:    py app\\xcoder.py [--lang ms|en] [--name "XCoder"]
  Linux/Mac:  python3 app/xcoder.py [--lang ms|en] [--name "XCoder"]

Each answer spawns llama-cli, then the process exits: 0 RAM between turns.
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

APP_DIR = Path(__file__).resolve().parent
BIN_DIR = APP_DIR / "bin"
MODELS_DIR = APP_DIR / "models"

LLAMA_TAG = "b11491"
LLAMA_BASE = f"https://github.com/ggerganov/llama.cpp/releases/download/{LLAMA_TAG}"
HF_MODEL_URL = ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
                "qwen2.5-0.5b-instruct-q4_k_m.gguf")
MODEL_SIZE = 491400032


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
    req = urllib.request.Request(url, headers={"User-Agent": "xcoder-standalone"})
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
            print(f"\r  {got / 1e6:.0f}/{total / 1e6:.0f} MB "
                  f"({got / el / 1e6:.1f} MB/s)", end="", flush=True)
    print()


def ensure_cli():
    name = "llama-cli.exe" if platform.system() == "Windows" else "llama-cli"
    hit = list(BIN_DIR.rglob(name))
    if hit:
        return hit[0]
    asset = asset_name()
    print(f"[xcoder] downloading engine {asset} ...")
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
    print("[xcoder] downloading model (~491MB, once only) ...")
    download(HF_MODEL_URL, dest, expect=MODEL_SIZE)
    return dest


def clean(text, name):
    t = text
    for pat in [r"qwen2\.5-0\.5b-instruct", r"\bqwen2\.5\b", r"\bqwen\b",
                r"\banthropic\b", r"\bclaude\b", r"\bmeta ai\b", r"\bllama\b",
                r"\balibaba cloud\b", r"\balibaba\b", r"\btongyi\b", r"\bqianwen\b"]:
        t = re.sub(pat, name, t, flags=re.IGNORECASE)
    t = re.sub(r"an?\s+ai\s+language\s+model\s+created\s+by\s+[^.,;\n]+",
               f"coding assistant, I am {name}", t, flags=re.IGNORECASE)
    t = re.sub(r"designed\s+by\s+[^.,;\n]+", f"built by {name}", t, flags=re.IGNORECASE)
    t = re.sub(r"created\s+by\s+[^.,;\n]+", f"built by {name}", t, flags=re.IGNORECASE)
    return t.strip()


# Labels the model emits when it role-plays extra turns instead of answering.
# Generation is stopped there (llama-cli --reverse-prompt in ask()) AND any
# leftovers are trimmed here, so a 0.5B model can't snowball old turns.
TURN_LABELS = ("pengguna:", "soalan baru:", "perbualan sebelum:",
               "konteks:", "context:", "soalan:", "question:")


def strip_turns(text, name):
    """Keep only the direct answer: drop prompt echoes, self-labels
    ('XCoder: ...') and fabricated follow-up turns ('Pengguna: ...')."""
    mine = name.lower() + ":"
    kept = []
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("> "):  # llama-cli prompt echo
            s = s[2:].strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith(TURN_LABELS) or low.startswith(mine):
            if kept:
                break  # fabricated next turn -> answer ends here
            if low.startswith(mine):
                s = s[len(name) + 1:].strip()  # "XCoder: Hai" -> "Hai"
                if s:
                    kept.append(s)
            continue  # leading prompt/question echo -> skip line
        kept.append(s)
    return "\n".join(kept).strip()


def build_prompt(history, q, lang):
    """Current question first, minimal context (last exchange only),
    explicit 'do not repeat' framing so small models answer directly."""
    if lang == "ms":
        ctx = ("Konteks (rujukan sahaja, JANGAN ulang atau tulis semula):\n"
               + "\n".join(history[-2:]) + "\n\n") if history else ""
        return ctx + "Soalan: " + q
    ctx = ("Context (reference only, do NOT repeat or rewrite):\n"
           + "\n".join(history[-2:]) + "\n\n") if history else ""
    return ctx + "Question: " + q


def ask(cli, model, system, prompt, name):
    """One subprocess per answer. Returns (text, tps or None)."""
    p = subprocess.run(
        [str(cli), "-m", str(model), "-c", "2048", "-n", "400",
         "--temp", "0.3", "--log-disable", "-st",
         "--no-display-prompt", "-sys", system, "-p", prompt,
         "-rs", "Pengguna:", "-rs", "Soalan baru:",
         "-rs", "Perbualan sebelum:"],
        capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout)[-1000:])
    blob = p.stdout
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", blob)
    tps = float(m.group(1)) if m else None
    # answer = lines after echoed "> prompt" up to "[ Prompt:" stats
    lines = blob.splitlines()
    start = next((i + 1 for i, l in enumerate(lines) if l.startswith("> ")), 0)
    end = next((i for i, l in enumerate(lines) if l.startswith("[ Prompt:")), len(lines))
    text = "\n".join(lines[start:end]).strip() or blob.strip()
    return strip_turns(text, name), tps


def main():
    ap = argparse.ArgumentParser(description="XCoder standalone (engine built-in)")
    ap.add_argument("--lang", default="ms", choices=["ms", "en"])
    ap.add_argument("--name", default="XCoder")
    ap.add_argument("--setup", action="store_true",
                    help="download engine + model only, then exit")
    a = ap.parse_args()

    if a.setup:
        cli = ensure_cli()
        model = ensure_model()
        print(f"OK engine: {cli}")
        print(f"OK model : {model} ({model.stat().st_size} bytes)")
        return

    if platform.system() == "Windows" and os.path.exists("NUL"):
        pass  # working folder is current dir; all paths below are relative
    cli = ensure_cli()
    model = ensure_model()
    print(f"{a.name} ready [standalone, engine built-in]. Taip /lang ms|en, /clear, /keluar\n")

    width = shutil.get_terminal_size((80, 20)).columns
    history, lang = [], a.lang
    while True:
        try:
            q = input("anda> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/keluar", "/exit", "/quit"):
            break
        if q.startswith("/lang"):
            lang = "en" if q.split()[-1] == "en" else "ms"
            history.clear()
            print(f"Bahasa: {lang}")
            continue
        if q == "/clear":
            history.clear()
            print("Sejarah dipadam.")
            continue
        if lang == "ms":
            system = (f"Anda ialah {a.name}, pembantu pengekodan CPU kecil 0.5B. "
                      f"Jawab ringkas dalam Bahasa Melayu. Jawab HANYA soalan terakhir. "
                      f"Jangan ulang perbualan atau tulis giliran baharu seperti 'Pengguna:'. "
                      f"Jika ditanya siapa anda, jawab HANYA: "
                      f"Saya {a.name}, dibina untuk pengekodan. Jangan sebut Qwen, Alibaba, Tongyi.")
        else:
            system = (f"You are {a.name}, a tiny CPU 0.5B coding assistant. Answer briefly in English. "
                      f"Answer ONLY the last question. "
                      f"Do not repeat the conversation or start new turns like 'Pengguna:'. "
                      f"If asked who you are, answer ONLY: I am {a.name}, built for coding. "
                      f"Never mention Qwen, Alibaba, Tongyi.")
        print(f"... {a.name} berfikir (CPU) ...")
        try:
            raw, tps = ask(cli, model, system, build_prompt(history, q, lang), a.name)
        except Exception as e:
            print(f"Ralat: {e}")
            continue
        ans = clean(raw, a.name)
        if not ans:
            print("Maaf, tiada jawapan. Cuba lagi dengan ayat lebih ringkas.")
            continue
        history += [f"Pengguna: {q}", f"{a.name}: {ans}"]
        del history[:-8]  # bound growth: last 4 exchanges max
        print(f"\n{a.name}> {ans}")
        if tps:
            s = f"{tps:.1f} T/s"
            print(" " * max(0, width - len(s)) + f"\x1b[2m{s}\x1b[0m")
        print()


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        raise SystemExit("Perlukan Python 3.9+")
    main()
