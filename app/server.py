"""llama-server process manager — one resident GGUF, idle auto-unload, switch lock.

Phase 1 runtime: single model process owned by the app.
Load -> warm up -> generate -> idle timeout -> unload.
Switching models = clean stop + start (never two models resident). Stdlib only.
"""
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def find_server(bin_dir, flavor="cpu"):
    if flavor == "cuda":
        bin_dir = Path(bin_dir).parent / "bin-cuda"
    cands = list(Path(bin_dir).rglob("llama-server.exe")) + \
        list(Path(bin_dir).rglob("llama-server"))
    cands = [c for c in cands if c.is_file()]
    if not cands:
        raise SystemExit(f"llama-server ({flavor}) not found in {bin_dir} — run with --setup first")
    return str(cands[0])


class ModelManager:
    def __init__(self, bin_dir, host="127.0.0.1", port=8080, ctx=4096,
                 threads=4, idle_timeout=180, flavor="cpu", ngl=0):
        self.flavor = flavor
        self.ngl = ngl if flavor == "cuda" else 0
        self.exe = find_server(bin_dir, flavor)
        self.host, self.port = host, port
        self.ctx, self.threads = ctx, threads
        self.idle_timeout = idle_timeout
        self.url = f"http://{host}:{port}"
        self._proc = None
        self._model = None
        self._lock = threading.RLock()
        self._timer = None

    # --- lifecycle ---
    def _alive(self):
        return self._proc is not None and self._proc.poll() is None

    def _healthy(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def _reset_timer(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self.idle_timeout > 0 and self._alive():
            self._timer = threading.Timer(self.idle_timeout, self._idle_stop)
            self._timer.daemon = True
            self._timer.start()

    def _idle_stop(self):
        with self._lock:
            self.stop()

    def load(self, gguf_path):
        """Switch to gguf_path. Returns (already_loaded, seconds)."""
        gguf_path = str(gguf_path)
        with self._lock:
            if self._model == gguf_path and self._alive() and self._healthy():
                self._reset_timer()
                return True, 0.0
            self.stop()
            t0 = time.time()
            cmd = [self.exe, "-m", gguf_path, "--host", self.host,
                   "--port", str(self.port), "-c", str(self.ctx),
                   "-b", "128", "-t", str(self.threads), "--parallel", "1"]
            if self.ngl:
                cmd += ["-ngl", str(self.ngl)]
            cmd += ["--log-disable"]
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.STDOUT)
            self._model = gguf_path
            for _ in range(120):
                if not self._alive():
                    raise RuntimeError("llama-server exited during load")
                if self._healthy():
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("llama-server health timeout")
            self._warmup()
            secs = time.time() - t0
            self._reset_timer()
            return False, secs

    def _warmup(self):
        """One tiny generation so the first real request is fast (non-fatal)."""
        try:
            body = json.dumps({"model": "xcoder",
                               "messages": [{"role": "user", "content": "OK"}],
                               "temperature": 0, "max_tokens": 5}).encode()
            req = urllib.request.Request(self.url + "/v1/chat/completions",
                                         data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                r.read()
        except Exception as e:
            print(f"[server] amaran warm-up: {e}")

    def ensure(self):
        """Reload current model if idle-unloaded. Returns (reloaded, seconds)."""
        with self._lock:
            if self._alive() and self._healthy():
                self._reset_timer()
                return False, 0.0
            if not self._model:
                raise RuntimeError("no model selected — /model <nama> dahulu")
            return self.load(self._model)

    def stop(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            if self._proc and self._alive():
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None

    def status(self):
        with self._lock:
            return {"model": self._model, "alive": self._alive(),
                    "healthy": self._healthy() if self._alive() else False,
                    "url": self.url, "idle_timeout": self.idle_timeout,
                    "flavor": self.flavor, "ngl": self.ngl}
