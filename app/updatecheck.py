"""Update check: git divergence + unregistered local GGUFs + new HF files. Stdlib only.

Read-only by default: git_state() uses status/ls-remote (never fetch/pull),
hf_files() caches the file list in memory. Callers perform actions explicitly.
"""
import subprocess
import time
import urllib.request
from pathlib import Path

HF_API = "https://huggingface.co/api/models/tokkilejen/xcoder"
_HF_CACHE = {"at": 0.0, "files": []}
HF_CACHE_SECS = 3600


def _git(args, cwd, timeout=15):
    try:
        p = subprocess.run(["git"] + args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout,
                           cwd=str(cwd))
        return p.returncode, (p.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""


def git_state(root):
    """(is_repo, dirty, ahead, behind). behind via ls-remote (no fetch)."""
    root = str(root)
    rc, _ = _git(["rev-parse", "--git-dir"], root)
    if rc != 0:
        return {"repo": False, "dirty": False, "ahead": 0, "behind": 0}
    _, porcelain = _git(["status", "--porcelain=v1", "--untracked-files=no"], root)
    dirty = bool(porcelain.strip())
    _, local = _git(["rev-parse", "HEAD"], root)
    _, remote = _git(["ls-remote", "origin", "HEAD"], root)
    remote = (remote.split() or [""])[0]
    ahead = behind = 0
    if local and remote and local != remote:
        _, base = _git(["merge-base", "HEAD", remote], root)
        if base:
            _, a = _git(["rev-list", "--count", f"{base}..HEAD"], root)
            _, b = _git(["rev-list", "--count", f"{base}..{remote}"], root)
            ahead = int(a or 0)
            behind = int(b or 0)
        else:
            behind = 1  # sejarah tidak berkait — anggap remote mendahului
    return {"repo": True, "dirty": dirty, "ahead": ahead, "behind": behind}


def git_pull_ff(root, timeout=120):
    """Fast-forward pull only. Returns (ok, message). Never stash/merge."""
    st = git_state(root)
    if not st["repo"]:
        return False, "bukan repo git"
    if st["dirty"]:
        return False, "working tree kotor — commit/stash dahulu (tidak auto-stash)"
    rc, out = _git(["pull", "--ff-only"], root, timeout=timeout)
    if rc != 0:
        return False, f"pull gagal: {out[-200:] or 'ralat'}"
    return True, out[-200:] or "sudah terkini"


def local_new_models(models_dir):
    """GGUF atas cakera tiada dalam registry. Returns [filenames]."""
    from experts import load_registry
    models_dir = Path(models_dir)
    reg_files = {m.get("file") for m in load_registry(models_dir).get("models", [])
                 if m.get("file")}
    return sorted(f.name for f in models_dir.glob("*.gguf")
                  if f.name not in reg_files)


def hf_files(force=False, timeout=15):
    """Senarai rfilename di HF (cache sejam). Gagal rangkaian -> cache lama/senarai kosong."""
    import json
    now = time.time()
    if not force and now - _HF_CACHE["at"] < HF_CACHE_SECS and _HF_CACHE["files"]:
        return _HF_CACHE["files"], "cache"
    try:
        req = urllib.request.Request(HF_API, headers={"User-Agent": "ezcodex-update"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        files = [s.get("rfilename", "") for s in data.get("siblings", []) if s.get("rfilename")]
        _HF_CACHE.update(at=now, files=files)
        return files, "live"
    except Exception as e:
        if _HF_CACHE["files"]:
            return _HF_CACHE["files"], f"cache (rangkaian: {e})"
        return [], f"ralat rangkaian: {e}"


def hf_new_files(models_dir, force=False):
    """Fail GGUF HF tiada di lokal. Returns (names, source_note)."""
    files, note = hf_files(force=force)
    local = {f.name for f in Path(models_dir).glob("*") if f.is_file()}
    return [f for f in files
            if f.lower().endswith(".gguf") and f.rsplit("/", 1)[-1] not in local], note


def summary(root, models_dir, include_hf=False):
    """One-line banner text or '' when everything current. Never raises."""
    try:
        parts = []
        gs = git_state(root)
        if gs["repo"] and (gs["dirty"] or gs["ahead"] or gs["behind"]):
            g = []
            if gs["dirty"]:
                g.append("kotor")
            if gs["behind"]:
                g.append(f"behind {gs['behind']}")
            if gs["ahead"]:
                g.append(f"ahead {gs['ahead']}")
            parts.append("git: " + ", ".join(g))
        try:
            n_local = local_new_models(models_dir)
            if n_local:
                parts.append(f"model lokal baharu: {len(n_local)}")
        except Exception:
            pass
        if include_hf:
            try:
                n_hf, _ = hf_new_files(models_dir)
                if n_hf:
                    parts.append(f"HF baharu: {len(n_hf)}")
            except Exception:
                pass
        if not parts:
            return ""
        return "[update] " + " | ".join(parts) + " — /update untuk tindakan"
    except Exception:
        return ""
