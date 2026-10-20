"""updatecheck.py: git state, local/HF model diff. Tmp dirs + stub HTTP only."""
import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import updatecheck
from updatecheck import git_state, hf_new_files, local_new_models, summary

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(not GIT, reason="git tiada")


def _git(cwd, *args):
    subprocess.run([GIT] + list(args), capture_output=True, cwd=str(cwd), check=True)


@needs_git
def test_git_state_clean_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-q", "--allow-empty", "-m", "x")
    st = git_state(tmp_path)
    assert st == {"repo": True, "dirty": False, "ahead": 0, "behind": 0}
    assert summary(tmp_path, tmp_path / "models") == ""


@needs_git
def test_git_state_dirty(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-q", "--allow-empty", "-m", "x")
    (tmp_path / "f.txt").write_text("x")
    _git(tmp_path, "add", "f.txt")
    st = git_state(tmp_path)
    assert st["repo"] is True and st["dirty"] is True


def test_git_state_not_repo(tmp_path):
    assert git_state(tmp_path)["repo"] is False


def test_local_new_models(tmp_path):
    md = tmp_path / "models"
    md.mkdir()
    (md / "a.gguf").write_bytes(b"GGUF" + b"\x00" * 10)
    assert local_new_models(md) == ["a.gguf"]
    (md / "models.json").write_text(
        json.dumps({"models": [{"name": "a", "file": "a.gguf"}]}))
    assert local_new_models(md) == []


def test_hf_new_files_stub():
    payload = {"siblings": [{"rfilename": "have.gguf"},
                            {"rfilename": "missing.gguf"},
                            {"rfilename": "README.md"}]}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        updatecheck.HF_API = f"http://127.0.0.1:{srv.server_port}/api"
        updatecheck._HF_CACHE.update(at=0.0, files=[])
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            (Path(td) / "have.gguf").write_bytes(b"x")
            new, note = hf_new_files(td, force=True)
            assert new == ["missing.gguf"] and note == "live"
            # cache reuse tanpa force
            new2, note2 = hf_new_files(td)
            assert new2 == ["missing.gguf"] and note2 == "cache"
    finally:
        updatecheck.HF_API = "https://huggingface.co/api/models/myxcoder/xcoder"
        updatecheck._HF_CACHE.update(at=0.0, files=[])
        srv.shutdown()
