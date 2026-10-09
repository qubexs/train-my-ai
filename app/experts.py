"""Multi-model pakar: setiap model kecil ada knowledge berbeza, satu aktif pada satu masa.

Layout: <models_dir>/models.json (registry) + fail *.gguf.
Alir pakar baharu: chat -> /good -> `train --mode sft --domain docker`
  -> finetune/docker/dataset.jsonl -> Colab LoRA -> GGUF -> `/model add <fail>`.
Stdlib sahaja.
"""
import json
import shutil
import time
import urllib.request
from pathlib import Path

GENERAL = "general"

# Fine-grained stacks (order matters: specific before generic on ties).
STACKS = {
    "typescript": ["typescript", ".ts", "interface ", ": number", ": string",
                   "enum ", " tsx", "generic "],
    "javascript": ["javascript", ".js", "console.log", "function ", "=>",
                   "const ", "let ", " js "],
    "nodejs": ["nodejs", "node.js", "node:", "express", "npm", "package.json",
               "hono", "fastify", "nextjs"],
    "html": ["html", ".html", "<div", "<html", "tag html"],
    "tailwind": ["tailwind"],
    "css": ["css", ".css", "flexbox", "stylesheet", "grid "],
    "postgresql": ["postgresql", "postgres", "psql", "supabase", "prisma"],
    "mysql": ["mysql", "mariadb", "phpmyadmin"],
    "sql": ["sql", "query", "select ", "join ", "jadual", "table"],
    "laravel": ["laravel", "eloquent", "blade", "migration"],
    "php": ["php", ".php", "composer", "artisan", "wordpress", "symfony"],
    "bash": ["bash", "shell", "skrip shell", ".sh", "chmod +x", "terminal"],
    "docker": ["docker", "kontena", "container", "dockerfile", "compose",
               "kubernetes", "k8s", "docker image", "image docker"],
    "linux": ["linux", "ubuntu", "debian", "grep", "chmod", "chown", "ssh",
              "cron", "systemd", "arahan linux", "perintah linux"],
    "python": ["python", "pip install", "pandas", "flask", "django",
               "skrip python"],
}

STACK2DOMAIN = {
    "typescript": "web", "javascript": "web", "nodejs": "web", "html": "web",
    "tailwind": "web", "css": "web",
    "postgresql": "data", "mysql": "data", "sql": "data",
    "laravel": "php", "php": "php",
    "bash": "linux", "docker": "linux", "linux": "linux",
    "python": "python",
}


def detect_stack(text):
    """Kesan stack halus -> nama stack atau 'general'."""
    t = " " + (text or "").lower() + " "
    best, best_n = GENERAL, 0
    for stack, keys in STACKS.items():
        n = sum(1 for k in keys if k in t)
        if n > best_n:
            best, best_n = stack, n
    return best

DOMAINS = {
    "identity": ["siapa anda", "siapa cipta", "siapa bina", "siapa yang buat",
                 "who are you", "who created you", "who made you",
                 "coder 77", "ezcodex", "xcoder", "pakar", "kepakaran", "expert",
                 "senarai program", "list program", "boleh buat apa",
                 "what can you do"],
    "linux": ["linux", "bash", "shell", "ubuntu", "debian", "grep", "chmod",
              "chown", "ssh", "terminal", "cron", "systemd", "arahan linux",
              "perintah linux", "skrip shell", "shell script"],
    "docker": ["docker", "kontena", "container", "dockerfile", "docker-compose",
               "compose", "kubernetes", "k8s", "docker image", "image docker"],
    "web": ["node", "nodejs", "node.js", "express", "typescript", "javascript",
            "npm", "react", "vue", "angular", "html", "css", "frontend",
            "website", "laman web", "blog", "hono", "fastify", "nextjs",
            "vite", "tailwind"],
    "data": ["sql", "postgres", "postgresql", "mysql", "sqlite", "mongodb",
             "database", "pangkalan data", "query", "select ", "prisma",
             "supabase", "jadual", "table"],
    "python": ["python", "pip install", "pandas", "flask", "django",
               "skrip python"],
    "php": ["php", "laravel", "composer", "symfony", "wordpress", "artisan",
            "skrip php"],
}


def detect(text):
    """Kesan domain soalan -> nama domain atau 'general'."""
    t = " " + (text or "").lower() + " "
    best, best_n = GENERAL, 0
    for dom, keys in DOMAINS.items():
        n = sum(1 for k in keys if k in t)
        if n > best_n:
            best, best_n = dom, n
    return best


def _reg_file(models_dir):
    return Path(models_dir) / "models.json"


def load_registry(models_dir):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    f = _reg_file(models_dir)
    if f.exists():
        try:
            reg = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(reg.get("models"), list):
                return reg
        except ValueError:
            pass
    return {"models": []}


def save_registry(models_dir, reg):
    _reg_file(models_dir).write_text(
        json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")


def scan(models_dir):
    """Gabung registry + fail *.gguf atas cakera. Daftar asas jika kosong."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reg = load_registry(models_dir)
    known_files = {m.get("file") for m in reg["models"]}
    has_base = any(m.get("name") == "base" for m in reg["models"])
    for g in sorted(models_dir.glob("*.gguf")):
        if g.name not in known_files:
            if "qwen" in g.stem.lower() and not has_base:
                name = "base"
                has_base = True
            else:
                name = g.stem
            reg["models"].append({"name": name, "file": g.name, "url": "",
                                  "domains": [GENERAL], "desc": "auto-dikesan",
                                  "trained": True})
    if not reg["models"]:
        reg["models"].append({"name": "base", "file": "", "url": "",
                              "domains": [GENERAL],
                              "desc": "model asas (belum dimuat turun)",
                              "trained": False})
    # Phase 1 lineup: 4 XCoder specialists, fallback to base until trained.
    seeds = [("xcoder-general", [GENERAL]), ("xcoder-js", ["web"]),
             ("xcoder-sql", ["data"]), ("xcoder-linux", ["linux"]),
             ("xcoder-php", ["php"]), ("xcoder-python", ["python"])]
    for sname, sdom in seeds:
        if not any(m.get("name") == sname for m in reg["models"]):
            reg["models"].append({"name": sname, "file": "", "url": "",
                                  "domains": sdom, "fallback": "base",
                                  "desc": "pakar (belum dilatih)", "trained": False})
    save_registry(models_dir, reg)
    return reg["models"]


def find(models, name):
    q = (name or "").lower()
    for m in models:
        if (m.get("name", "").lower() == q
                or Path(m.get("file", "")).stem.lower() == q
                or m.get("file", "").lower() == q):
            return m
    return None


def _download(url, dest, expect=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ezcodex-models"})
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
    return dest


def resolve(models_dir, name, _depth=0):
    """Cari model mengikut nama -> (Path|None, nota). Auto-muat turun jika ada URL."""
    models_dir = Path(models_dir)
    m = find(scan(models_dir), name)
    if not m:
        return None, f"model '{name}' tidak ditemui. /models untuk senarai."
    f = models_dir / m.get("file", "")
    if m.get("file") and f.exists():
        return f, ""
    if m.get("url"):
        try:
            dest = models_dir / (m.get("file") or (m["name"] + ".gguf"))
            _download(m["url"], dest)
            return dest, f"dimuat turun {dest.name}"
        except Exception as e:
            return None, f"gagal muat turun: {e}"
    fb = m.get("fallback")
    if fb and _depth < 3:
        p, note = resolve(models_dir, fb, _depth + 1)
        extra = f"({m['name']} belum dilatih -> guna {fb})"
        return p, (note + " " + extra).strip()
    return None, f"fail {m.get('file') or m['name']} tiada. /model add <url|fail> untuk daftar."


def add(models_dir, src, name=None, domains=None):
    """Daftar model baharu dari URL (muat turun) atau fail tempatan (salin)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    domains = [d for d in (domains or [GENERAL]) if d] or [GENERAL]
    if Path(src).exists():
        src_p = Path(src)
        dest = models_dir / src_p.name
        if src_p.resolve() != dest.resolve():
            shutil.copy(src_p, dest)
        url = ""
    else:
        tail = src.rsplit("/", 1)[-1].split("?")[0] or "model.gguf"
        fname = (name + ".gguf") if name and not name.endswith(".gguf") else tail
        dest = models_dir / fname
        _download(src, dest)
        url = src
    entry = {"name": name or dest.stem, "file": dest.name, "url": url,
             "domains": domains, "desc": "pakar " + ",".join(domains),
             "trained": True}
    reg = load_registry(models_dir)
    reg["models"] = [m for m in reg["models"] if m.get("name") != entry["name"]]
    reg["models"].append(entry)
    save_registry(models_dir, reg)
    return entry


def fmt_size(path):
    try:
        return f"{Path(path).stat().st_size / 1e6:.0f}MB"
    except OSError:
        return "tiada"
