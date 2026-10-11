"""Backends: llama-cli subprocess (default, 0 RAM idle) + LM Studio OpenAI API."""
import json
import re
import subprocess
import urllib.request

MAX_TURNS = 6
MAX_USER_CHARS = 1000
MAX_ASSISTANT_CHARS = 500


def transcript_text(messages):
    """Single-string transcript for CLI prompts. Anti-echo policy:
    recent turns only, assistant truncated harder, tool results labelled."""
    turns = []
    for m in (messages or [])[-MAX_TURNS:]:
        role = m.get("role", "user")
        content = (m.get("content", "") or "")
        content = content[:MAX_ASSISTANT_CHARS] if role == "assistant" else content[:MAX_USER_CHARS]
        if role == "tool":
            turns.append(f"[Hasil tool {m.get('name', '')}]: {content}")
        elif role == "assistant":
            turns.append(f"Jawapan sebelum: {content}")
        else:
            turns.append(f"Soalan: {content}")
    return turns


def transcript_messages(messages):
    """Same policy as transcript_text, but as OpenAI message list for APIs."""
    out = []
    for m in (messages or [])[-MAX_TURNS:]:
        role = m.get("role", "user")
        content = (m.get("content", "") or "")
        content = content[:MAX_ASSISTANT_CHARS] if role == "assistant" else content[:MAX_USER_CHARS]
        if role == "tool":
            out.append({"role": "user",
                        "content": f"[Hasil tool {m.get('name', '')}]: {content}"})
        elif role == "assistant":
            out.append({"role": "assistant", "content": f"Jawapan sebelum: {content}"})
        else:
            out.append({"role": role, "content": f"Soalan: {content}"})
    return out


class LlamaCliBackend:
    def __init__(self, cli_path, model_path, ctx=2048, n_predict=500, temp=0.3):
        self.cli = str(cli_path)
        self.model = str(model_path)
        self.ctx = ctx
        self.n_predict = n_predict
        self.temp = temp
        self.kind = "llama-cli"

    def chat_ex(self, system, messages, max_tokens=None):
        """Returns (text, tps, finish) where finish is stop|length|error."""
        n = max_tokens or self.n_predict
        turns = transcript_text(messages)
        full = "\n".join(turns) + "\nJawab soalan terakhir dengan ringkas."
        try:
            p = subprocess.run(
                [self.cli, "-m", self.model, "-c", str(self.ctx),
                 "-n", str(n),
                 "--temp", str(self.temp),
                 "--repeat-penalty", "1.15", "--repeat-last-n", "128",
                 "--log-disable", "-st",
                 "--no-display-prompt",
                 "-r", "Pengguna:", "-r", "Pembantu:",
                 "-r", "<|im_start|>", "-r", "<|im_end|>",
                 "-sys", system, "-p", full],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=300)
        except subprocess.TimeoutExpired:
            return "", None, "length"
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout)[-1000:])
        blob = p.stdout
        m = re.search(r"Generation:\s*([\d.]+)\s*t/s", blob)
        tps = float(m.group(1)) if m else None
        lines = blob.splitlines()
        start = next((i + 1 for i, l in enumerate(lines) if l.startswith("> ")), 0)
        end = next((i for i, l in enumerate(lines) if l.startswith("[ Prompt:")), len(lines))
        text = "\n".join(lines[start:end]).strip() or blob.strip()
        # CLI has no finish_reason: length hit iff output ran near the token budget
        # (~4 chars/token heuristic) without hitting a reverse-prompt stop.
        finish = "length" if len(text) >= n * 4 * 0.9 else "stop"
        return text, tps, finish

    def chat(self, system, messages, max_tokens=None):
        text, tps, _ = self.chat_ex(system, messages, max_tokens)
        return text, tps


class LmStudioBackend:
    def __init__(self, base_url="http://localhost:1234/v1", model="ezcodex-0.5b",
                 temp=0.3, max_tokens=500):
        self.base = base_url.rstrip("/")
        self.model = model
        self.temp = temp
        self.max_tokens = max_tokens
        self.kind = "lmstudio"
    def chat_ex(self, system, messages, max_tokens=None):
        import time
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
            + transcript_messages(messages),
            "temperature": self.temp,
            "max_tokens": max_tokens or self.max_tokens,
        }).encode()
        req = urllib.request.Request(self.base + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode())
        secs = max(0.1, time.time() - t0)
        choice = (data.get("choices") or [{}])[0]
        out = (choice.get("message") or {}).get("content", "")
        tok = (data.get("usage") or {}).get("completion_tokens") or max(1, len(out) // 4)
        return out, tok / secs, choice.get("finish_reason") or "stop"

    def chat(self, system, messages, max_tokens=None):
        out, tps, _ = self.chat_ex(system, messages, max_tokens)
        return out, tps


def pick_backend(name, llama_cli=None, llama_model=None, lm_url="http://localhost:1234/v1",
                 lm_model="ezcodex-0.5b"):
    if name == "llama":
        return LlamaCliBackend(llama_cli, llama_model)
    if name == "lmstudio":
        return LmStudioBackend(lm_url, lm_model)
    # auto: prefer llama-cli (always works offline), fallback set by caller
    return LlamaCliBackend(llama_cli, llama_model)


class ServerBackend:
    """Resident llama-server via ModelManager + OpenAI-compatible API."""
    def __init__(self, manager, model="xcoder", temp=0.3, max_tokens=500):
        self.manager = manager
        self.kind = "server"
        self._api = LmStudioBackend(manager.url + "/v1", model, temp, max_tokens)

    @property
    def model(self):
        return self._api.model

    @model.setter
    def model(self, v):
        self._api.model = v

    def chat(self, system, messages, max_tokens=None):
        self.manager.ensure()
        try:
            return self._api.chat(system, messages, max_tokens)
        finally:
            self.manager._reset_timer()

    def chat_ex(self, system, messages, max_tokens=None):
        self.manager.ensure()
        try:
            return self._api.chat_ex(system, messages, max_tokens)
        finally:
            self.manager._reset_timer()
