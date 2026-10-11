"""Fetch: download engine + models with checksum verification. Stdlib only.

Split out of ezcodex.py (Item 4+). All functions take explicit dirs (no globals).
"""
import platform
import stat
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

LLAMA_TAG = "b11491"
LLAMA_BASE = f"https://github.com/ggerganov/llama.cpp/releases/download/{LLAMA_TAG}"
HF_MODEL_URL = ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
                "qwen2.5-0.5b-instruct-q4_k_m.gguf")
MODEL_SIZE = 491400032
# Pinned checksums (sha256). None = verify size/presence only.
PINNED = {
    "qwen2.5-0.5b-instruct-q4_k_m.gguf": {
        "size": MODEL_SIZE,
        "sha256": "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
    },
}


def sha256_file(path, chunk=16 * 1024 * 1024):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def verify(path, size=None, sha256=None):
    """(ok, reason) for an on-disk file."""
    p = Path(path)
    if not p.is_file():
        return False, "tiada fail"
    if size is not None and p.stat().st_size != size:
        return False, (f"saiz {p.stat().st_size} != jangka {size} "
                       "(muat turun rosak/tidak lengkap)")
    if sha256 is not None and sha256_file(p) != sha256:
        return False, "sha256 tidak sepadan (fail rosak atau diganti)"
    return True, "ok"


def download(url, dest, expect=None, sha256=None, tries=2):
    """Download + verify (size/sha256). Retry once, fail loudly on mismatch."""
    dest = Path(dest)
    for attempt in range(1, tries + 1):
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
                print(f"\r  {got / 1e6:.0f}/{total / 1e6:.0f} MB "
                      f"({got / el / 1e6:.1f} MB/s)", end="", flush=True)
        print()
        ok, reason = verify(dest, expect, sha256)
        if ok:
            return dest
        print(f"[fetch] semakan gagal ({reason}), cubaan {attempt}/{tries}")
        try:
            dest.unlink()
        except OSError:
            pass
    raise SystemExit(f"gagal muat turun disahkan: {url} ({reason})")


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


def _extract(arc, dest_dir):
    if arc.suffix == ".zip":
        with zipfile.ZipFile(arc) as z:
            z.extractall(dest_dir)
    else:
        with tarfile.open(arc, "r:gz") as t:
            t.extractall(dest_dir)


def ensure_cli(bin_dir):
    """llama-cli sedia (muat turun + sahkan jika tiada). Returns path."""
    bin_dir = Path(bin_dir)
    name = "llama-cli.exe" if platform.system() == "Windows" else "llama-cli"
    hit = list(bin_dir.rglob(name))
    if hit:
        return hit[0]
    asset = asset_name()
    print(f"[ezcodex] downloading engine {asset} ...")
    with tempfile.TemporaryDirectory() as tmp:
        arc = Path(tmp) / asset
        download(f"{LLAMA_BASE}/{asset}", arc)
        _extract(arc, bin_dir)
    hit = list(bin_dir.rglob(name))
    if not hit:
        raise SystemExit("llama-cli not found after extract")
    if platform.system() != "Windows":
        hit[0].chmod(hit[0].stat().st_mode | stat.S_IEXEC)
    return hit[0]


def ensure_model(models_dir):
    """Base GGUF sedia + disahkan (saiz + sha256 pinned). Returns path."""
    models_dir = Path(models_dir)
    dest = models_dir / HF_MODEL_URL.rsplit("/", 1)[-1]
    pin = PINNED.get(dest.name, {})
    ok, _ = verify(dest, pin.get("size", MODEL_SIZE), pin.get("sha256"))
    if ok:
        return dest
    if dest.exists():
        print("[ezcodex] model sedia ada gagal semakan — muat turun semula ...")
        try:
            dest.unlink()
        except OSError:
            pass
    else:
        print("[ezcodex] downloading model (~491MB, once only) ...")
    download(HF_MODEL_URL, dest, expect=pin.get("size", MODEL_SIZE),
             sha256=pin.get("sha256"))
    return dest


def cuda_dir(app_dir):
    return Path(app_dir) / "bin-cuda"


def ensure_cuda(app_dir):
    """Download llama.cpp CUDA build (GTX 1070+) into bin-cuda/. Returns dir."""
    import json as _json
    d = cuda_dir(app_dir)
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
    rt = next((n for n, l in lower if l.startswith("cudart-llama-") and "-cuda-12" in l
               and "win" in l and "x64" in l and l.endswith(".zip")), None)
    if not pack:
        raise SystemExit(f"tiada asset llama CUDA untuk {LLAMA_TAG}.")
    print(f"[ezcodex] downloading CUDA engine {pack} ...")
    with tempfile.TemporaryDirectory() as tmp:
        for asset in ([pack] + ([rt] if rt else [])):
            arc = Path(tmp) / asset
            download(f"{LLAMA_BASE}/{asset}", arc)
            _extract(arc, d)
    if not (list(d.rglob("llama-server.exe")) + list(d.rglob("llama-server"))):
        raise SystemExit("llama-server tidak ditemui selepas extract CUDA")
    return d
