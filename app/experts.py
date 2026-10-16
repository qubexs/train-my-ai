"""Multi-model pakar: setiap model kecil ada knowledge berbeza, satu aktif pada satu masa.

Layout: <models_dir>/models.json (registry) + fail *.gguf.
Alir pakar baharu: chat -> /good -> `train --mode sft --domain docker`
  -> finetune/stacks/docker/dataset.jsonl -> Colab LoRA -> GGUF -> `/model add <fail>`.
Stdlib sahaja.
"""
import json
import shutil
import time
import urllib.request
from pathlib import Path

GENERAL = "general"


def resolve_layout(app_dir):
    """Unify split layouts: prefer repo-root bin/models when populated,
    else app/ layout. Frozen exe: exe-dir layout (unchanged)."""
    app_dir = Path(app_dir)
    root = app_dir.parent
    rb, rm = root / "bin", root / "models"
    ab, am = app_dir / "bin", app_dir / "models"

    def has_gguf(d):
        return d.is_dir() and any(d.glob("*.gguf"))

    def has_cli(d):
        return d.is_dir() and bool(list(d.rglob("llama-cli.exe"))
                                   + list(d.rglob("llama-cli")))
    bins = rb if has_cli(rb) else ab
    models = rm if has_gguf(rm) else am
    return bins, models

# xcoder-<expert>-0.5b-<quant>.gguf -> default domains for auto-registration.
# Nota: stack = domain (tiada lapisan broad) — setiap stack petakan ke dirinya sendiri.
EXPERT_DOMAIN_MAP = {
    "general": [GENERAL], "base": [GENERAL],
    "js": ["javascript"], "javascript": ["javascript"],
    "typescript": ["typescript"], "ts": ["typescript"],
    "nodejs": ["nodejs"], "node": ["nodejs"],
    "html": ["html"], "css": ["css"], "tailwind": ["tailwind"],
    "web": ["web"],
    "sql": ["sql"], "postgresql": ["postgresql"], "postgres": ["postgresql"],
    "mysql": ["mysql"], "data": ["data"],
    "linux": ["linux"], "docker": ["docker"], "bash": ["bash"],
    "git": ["git"],
    "react": ["react"],
    "mongodb": ["mongodb"], "mongo": ["mongodb"],
    "redis": ["redis"],
    "python": ["python"], "php": ["php"], "laravel": ["laravel"],
}


def infer_name_domains(filename):
    """xcoder-docker-0.5b-q4_k_m.gguf -> (xcoder-docker, [docker])."""
    import re
    stem = Path(filename).stem.lower()
    m = re.match(r"^(xcoder|coder77|ezcodex)-([a-z0-9]+?)(?:-0\.5b)?(?:-|$)", stem)
    if m:
        expert = m.group(2)
        return f"xcoder-{expert}", EXPERT_DOMAIN_MAP.get(expert, [GENERAL])
    return stem, [GENERAL]


def auto_import(models_dir, src_dir=None):
    """Copy newest *.gguf from Downloads (or src_dir) into models_dir + register.
    Returns (entry|None, message). Skips files already registered."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(src_dir) if src_dir else Path.home() / "Downloads"
    cands = sorted(src_dir.glob("*.gguf"), key=lambda p: p.stat().st_mtime,
                   reverse=True) if src_dir.is_dir() else []
    if not cands:
        return None, f"tiada *.gguf dalam {src_dir}"
    known = {m.get("file") for m in scan(models_dir)}
    for cand in cands:
        if cand.name in known or (models_dir / cand.name).exists():
            continue
        name, doms = infer_name_domains(cand.name)
        entry = add(models_dir, str(cand), name=name, domains=doms)
        return entry, f"auto-import {cand.name} -> {entry['name']} [{','.join(doms)}]"
    return None, "semua GGUF dalam Downloads sudah didaftar"

# Fine-grained stacks (order matters: specific before generic on ties;
# nodejs before javascript because "package.json" contains ".js").
STACKS = {
    "typescript": ["typescript", ".ts", "interface ", ": number", ": string",
                   "enum ", " tsx", "generic "],
    "nodejs": ["nodejs", "node.js", "node:", "express", "npm", "package.json",
               "hono", "fastify", "nextjs", "server js", "js server",
               "backend js", "rest api", "endpoint"],
    "javascript": ["javascript", ".js", "console.log", "function ", "=>",
                   "const ", "let ", " js "],
    "html": ["html", ".html", "<div", "<html", "tag html"],
    "tailwind": ["tailwind"],
    "css": ["css", ".css", "flexbox", "stylesheet", "grid "],
    "postgresql": ["postgresql", "postgres", "psql", "supabase", "prisma"],
    "mysql": ["mysql", "mariadb", "phpmyadmin"],
    "sql": ["sql", "query", "select ", "join ", "jadual", "table"],
    "laravel": ["laravel", "eloquent", "blade", "migration"],
    "php": ["php", ".php", "composer", "artisan", "wordpress", "symfony"],
    "react": ["react", "jsx", "usestate", "useeffect", "component react",
              "next.js", "vite react"],
    "mongodb": ["mongodb", "mongo", "mongosh", "objectid", "aggregation pipeline",
                "find({", "mongoose"],
    "redis": ["redis", "setex", "hgetall", "pub/sub", "ioredis", "cache redis",
              "redis-cli"],
    "git": ["git commit", "git push", "git pull", "git clone", "github",
            "pull request", "merge conflict", "gitignore", "rebase", "git log"],
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
    "bash": "linux", "docker": "linux", "linux": "linux", "git": "linux",
    "mongodb": "data", "redis": "data",
    "react": "web",
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
    if best != GENERAL:
        return best
    # fallback: perkataan web umum -> stack wakil (blog/website tiada milik stack khusus)
    for stack, keys in [("nodejs", ["website", "laman web", "blog", "web app", "aplikasi web"]),
                        ("sql", ["database", "pangkalan data", "query", "data"]),
                        ("docker", ["kontena", "container"]),
                        ("linux", ["terminal", "arahan", "server linux"]),
                        ("python", ["skrip", "aturcara"]),
                        ("php", ["laman php", "wordpress"])]:
        if any(k in t for k in keys):
            return stack
    return GENERAL

def count_stacks(text):
    """Bilangan stack berbeza yang disebut — >=2 ertinya soalan rentas-stack."""
    return len(matching_stacks(text))


def matching_stacks(text):
    """Senarai stack yang disebut dalam teks."""
    t = " " + (text or "").lower() + " "
    return [s for s, keys in STACKS.items() if any(k in t for k in keys)]


IDENTITY_KEYS = ["siapa anda", "siapa cipta", "siapa bina", "siapa yang buat",
                 "who are you", "who created you", "who made you",
                 "coder 77", "ezcodex", "xcoder", "pakar", "kepakaran", "expert",
                 "senarai program", "list program", "boleh buat apa",
                 "what can you do"]

# stack = domain (16 pakar, tiada lapisan broad). Legacy broad tags below
# are auto-expanded to member stacks when loading old registries.
LEGACY_GROUPS = {
    "web": ["javascript", "typescript", "nodejs", "html", "css", "tailwind"],
    "data": ["sql", "postgresql", "mysql"],
    "linux": ["linux", "bash", "docker"],
    "python": ["python"],
    "php": ["php", "laravel"],
    "general": [GENERAL],
}


def detect(text):
    """Kesan stack soalan -> nama stack (stack = domain) atau 'general'."""
    t = " " + (text or "").lower() + " "
    if any(k in t for k in IDENTITY_KEYS):
        return GENERAL
    return detect_stack(text)


def file_hash(path, chunk=8 * 1024 * 1024):
    """sha256 hex penuh. Raises OSError jika tidak boleh dibaca."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def _gguf_ok(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


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
    # migrate: drop stale broad seeds, expand broad tags on real entries
    reg["models"] = [m for m in reg["models"]
                     if not (m.get("fallback") and not m.get("file")
                             and m.get("name") in ("xcoder-js", "xcoder-sql", "xcoder-linux",
                                                   "xcoder-php", "xcoder-python", "xcoder-general"))]
    for m in reg["models"]:
        doms = m.get("domains", [])
        if len(doms) == 1 and doms[0] in ("web", "data"):
            m["domains"] = list(LEGACY_GROUPS[doms[0]])
    known_files = {m.get("file") for m in reg["models"]}
    has_base = any(m.get("name") == "base" for m in reg["models"])
    for g in sorted(models_dir.glob("*.gguf")):
        if g.name not in known_files:
            if "qwen" in g.stem.lower() and not has_base:
                reg["models"].append({"name": "base", "file": g.name, "url": "",
                                      "domains": [GENERAL], "desc": "auto-dikesan",
                                      "trained": True})
                has_base = True
                continue
            name, doms = infer_name_domains(g.name)
            ph = next((m for m in reg["models"]
                       if m.get("name") == name and not m.get("file")), None)
            if ph:
                # seed placeholder + real file arrived = complete registration
                ph["file"] = g.name
                ph["domains"] = doms or ph.get("domains", [GENERAL])
                ph["desc"] = "auto-dikesan"
                ph["trained"] = True
                continue
            if any(m.get("name") == name for m in reg["models"]):
                name = g.stem
            reg["models"].append({"name": name, "file": g.name, "url": "",
                                  "domains": doms, "desc": "auto-dikesan",
                                  "trained": True})
    if not reg["models"]:
        reg["models"].append({"name": "base", "file": "", "url": "",
                              "domains": [GENERAL],
                              "desc": "model asas (belum dimuat turun)",
                              "trained": False})
    # 20 stack seeds (stack = domain), fallback to base until trained.
    seeds = [(f"xcoder-{s}", [s]) for s in
             ["general", "javascript", "typescript", "nodejs", "html", "css",
              "tailwind", "sql", "postgresql", "mysql", "linux", "docker",
              "bash", "python", "php", "laravel",
              "git", "react", "mongodb", "redis"]]
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


def _download(url, dest, expect=None, sha256=None, tries=2):
    """Muat turun + sahkan saiz/sha256. Cuba sekali lagi, gagal dengan jelas."""
    dest = Path(dest)
    for attempt in range(1, tries + 1):
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
        ok, reason = True, "ok"
        if expect is not None and dest.stat().st_size != expect:
            ok, reason = False, "saiz tidak sepadan"
        elif sha256 is not None and file_hash(dest) != sha256:
            ok, reason = False, "sha256 tidak sepadan"
        if ok and not _gguf_ok(dest):
            ok, reason = False, "bukan fail GGUF"
        if ok:
            return dest
        print(f"[models] semakan gagal ({reason}), cubaan {attempt}/{tries}")
        try:
            dest.unlink()
        except OSError:
            pass
    raise SystemExit(f"gagal muat turun disahkan: {url} ({reason})")


def verify_pinned(models_dir, entry):
    """Semak fail model lawan pin registry (sha256/saiz). Backfill pin jika tiada.
    Returns (ok, reason)."""
    f = Path(models_dir) / (entry.get("file") or "")
    if not entry.get("file") or not f.exists():
        return False, "tiada fail"
    if not _gguf_ok(f):
        return False, "bukan fail GGUF"
    pin, size = entry.get("sha256"), entry.get("bytes")
    if pin:
        if file_hash(f) != pin:
            return False, "sha256 tidak sepadan (fail rosak atau diganti)"
        return True, "ok"
    try:
        digest, nbytes = file_hash(f), f.stat().st_size
        entry["sha256"], entry["bytes"] = digest, nbytes
        reg = load_registry(models_dir)
        for m in reg["models"]:
            if m.get("name") == entry.get("name") and m.get("file") == entry.get("file"):
                m["sha256"], m["bytes"] = digest, nbytes
                break
        save_registry(models_dir, reg)
    except OSError:
        pass
    return True, "ok (pin baharu direkod)"


def resolve(models_dir, name, _depth=0):
    """Cari model mengikut nama -> (Path|None, nota). Auto-muat turun jika ada URL."""
    models_dir = Path(models_dir)
    m = find(scan(models_dir), name)
    if not m:
        return None, f"model '{name}' tidak ditemui. /models untuk senarai."
    f = models_dir / m.get("file", "")
    if m.get("file") and f.exists():
        ok, reason = verify_pinned(models_dir, m)
        if ok:
            return f, ""
        return None, f"model '{name}' gagal semakan: {reason}. /model add semula."
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
    if not _gguf_ok(dest):
        try:
            if not Path(src).exists():
                dest.unlink()
        except OSError:
            pass
        raise SystemExit(f"daftar gagal: {dest.name} bukan fail GGUF")
    entry = {"name": name or dest.stem, "file": dest.name, "url": url,
             "domains": domains, "desc": "pakar " + ",".join(domains),
             "trained": True, "sha256": file_hash(dest),
             "bytes": dest.stat().st_size}
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


def record_speed(models_dir, name, tps):
    """Simpan kelajuan terakhir model (untuk paparan /models)."""
    import time as _t
    try:
        reg = load_registry(models_dir)
        for m in reg["models"]:
            if m.get("name") == name:
                m["last_tps"] = round(float(tps), 1)
                m["measured_at"] = int(_t.time())
                break
        save_registry(models_dir, reg)
    except Exception:
        pass


def audit_models(models_dir, progress=None):
    """Semak setiap GGUF: magic GGUF, saiz, sha256, duplikat, sepadan registry.
    Returns dict(report). progress(name, done, total) callback optional."""
    import hashlib
    models_dir = Path(models_dir)
    files = sorted(models_dir.glob("*.gguf"))
    reg = {m.get("file"): m.get("name") for m in scan(models_dir)}
    pins = {m.get("file"): m.get("sha256") for m in scan(models_dir)}
    items, by_hash = [], {}
    for i, f in enumerate(files):
        if progress:
            progress(f.name, i + 1, len(files))
        try:
            magic = f.open("rb").read(4) == b"GGUF"
        except OSError:
            magic = False
        h = hashlib.sha256()
        try:
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            digest = h.hexdigest()[:16]
            full = h.hexdigest()
        except OSError:
            digest, full = "Ralat-baca", ""
        by_hash.setdefault(digest, []).append(f.name)
        pin = pins.get(f.name)
        items.append({"file": f.name, "mb": round(f.stat().st_size / 1e6),
                      "magic_ok": magic, "sha": digest,
                      "hash_ok": (full == pin) if (pin and full) else None,
                      "registered": reg.get(f.name)})
    dupes = {h: n for h, n in by_hash.items() if len(n) > 1}
    reg_files = {m.get("file") for m in scan(models_dir) if m.get("file")}
    missing = sorted(r for r in reg_files if not (models_dir / r).exists())
    return {"models": items, "duplicates": dupes, "registry_missing": missing,
            "count": len(items)}
