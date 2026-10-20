"""Backends: llama-cli subprocess (default, 0 RAM idle) + LM Studio OpenAI API."""
import json
import re
import subprocess
import urllib.request

MAX_TURNS = 4
MAX_USER_CHARS = 1000
MAX_ASSISTANT_CHARS = 300


def transcript_text(messages):
    """ChatML transcript — sama seperti format latihan SFT (bukan label Soalan:/Jawapan:).
    Turns -> blok <|im_start|>role/content/<|im_end|>, diakhiri prefix assistant
    terbuka supaya model menyambung sebagai pembantu. Had potongan dikekalkan
    (assistant dipangkas lebih agresif — bahan bakar role-play loop)."""
    turns = []
    for m in (messages or [])[-MAX_TURNS:]:
        role = m.get("role", "user")
        content = (m.get("content", "") or "")
        content = content[:MAX_ASSISTANT_CHARS] if role == "assistant" else content[:MAX_USER_CHARS]
        if role == "tool":
            content = f"[Hasil tool {m.get('name', '')}]: {content}"
            role = "user"
        elif role != "assistant":
            role = "user"
        turns.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    turns.append("<|im_start|>assistant\n")
    return turns


def is_thinking_model(model_path):
    """Kesan model berasaskan Qwen3.5 (template ada suis thinking).
    Untuk fine-tune biasa, blok `<think></think>` KOSONG melatih jawapan terus."""
    n = str(model_path or "").lower()
    return ("3.5" in n or "0.8b" in n or "1.5b" in n or "think" in n
            or "reason" in n or "qwq" in n)


def wants_thinking(model_path):
    """Model yang DILATIH untuk berfikir (nama mengandungi think/reason):
    prompt dibuka dengan `<think>` supaya model menjana jejak penaakulan."""
    n = str(model_path or "").lower()
    return "think" in n or "reason" in n


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
    def __init__(self, cli_path, model_path, ctx=1024, n_predict=500, temp=0.3,
                 threads=0):
        self.cli = str(cli_path)
        self.model = str(model_path)
        self.ctx = ctx
        self.n_predict = n_predict
        self.temp = temp
        if not threads:
            try:
                from hardware import physical_cores
                threads = physical_cores()
            except Exception:
                import os
                threads = os.cpu_count() or 4
        self.threads = max(1, threads)
        self.kind = "llama-cli"
        self.last_timings = None  # CLI: no prefill/decode split available

    def chat_ex(self, system, messages, max_tokens=None):
        """Returns (text, tps, finish) where finish is stop|length|error."""
        n = max_tokens or self.n_predict
        turns = transcript_text(messages)
        # Sistem di-inline sebagai blok ChatML (bukan -sys): replika tepat format
        # latihan SFT yang disahkan reproduksi verbatim.
        full = f"<|im_start|>system\n{system}<|im_end|>\n" + "\n".join(turns)
        if wants_thinking(self.model):
            # Fine-tune thinking: buka blok supaya model menjana penaakulan
            # sendiri, kemudian `</think>` + jawapan.
            full += "<think>\n"
        elif is_thinking_model(self.model):
            # Suis template natif Qwen3.5: blok <think> kosong = fikir DIMATIKAN.
            # Tanpa ini model membazir ratusan token meracau sebelum menjawab.
            full += "<think>\n\n</think>\n\n"
        try:
            p = subprocess.run(
                [self.cli, "-m", self.model, "-c", str(self.ctx),
                 "-n", str(n), "-t", str(self.threads),
                 "--temp", str(self.temp),
                 "--repeat-penalty", "1.15", "--repeat-last-n", "128",
                 "--log-disable", "-st",
                 "--no-display-prompt",
                 "-r", "Pengguna:", "-r", "Pembantu:",
                 "-r", "<|im_start|>", "-r", "<|im_end|>",
                 "-p", full],
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
        plines = full.splitlines()
        start = next((i + 1 for i, l in enumerate(lines) if l.startswith("> ")), 0)
        if start > 0 and plines and lines[start - 1][2:] == plines[0]:
            # gema prompt: baris pertama berawalan "> ", selebihnya mentah —
            # buang run berterusan yang sepadan tepat dengan prompt. Jika CLI
            # memotong gema dengan "... (truncated)", lompat melepasi baris itu
            # (jika tidak, teks sistem bocor ke jawapan).
            pend, pi, trunc = start, 1, None
            while pend < len(lines):
                if pi < len(plines) and lines[pend] == plines[pi]:
                    pend += 1
                    pi += 1
                    continue
                if "(truncated)" in lines[pend].lower():
                    trunc = pend
                break
            if pi >= len(plines):
                start = pend
            elif trunc is not None:
                start = trunc + 1
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
                 temp=0.3, max_tokens=500, chat_template_kwargs=None):
        self.base = base_url.rstrip("/")
        self.model = model
        self.temp = temp
        self.max_tokens = max_tokens
        self.chat_template_kwargs = chat_template_kwargs
        self.kind = "lmstudio"
        self.last_timings = None

    def chat_ex(self, system, messages, max_tokens=None):
        import time
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
            + transcript_messages(messages),
            "temperature": self.temp,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.chat_template_kwargs:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        body = json.dumps(body).encode()
        req = urllib.request.Request(self.base + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode())
        secs = max(0.1, time.time() - t0)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        # gemma4 (dan model reasoning lain) letak fikiran dalam reasoning_content
        # dengan content kosong — guna sebagai jawapan daripada papar kosong.
        out = msg.get("content", "") or msg.get("reasoning_content", "") or ""
        tok = (data.get("usage") or {}).get("completion_tokens") or max(1, len(out) // 4)
        self.last_timings = data.get("timings")
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
    def __init__(self, manager, model="xcoder", temp=0.3, max_tokens=500,
                 model_path=None):
        self.manager = manager
        self.kind = "server"
        # model_path = fail GGUF sebenar (untuk pilih prompt/kesan thinking);
        # model kekal nama API ("xcoder") yang tidak divalidasi llama-server.
        self.model_path = str(model_path or model)
        # Model Qwen3.5 asas: matikan thinking pada template (jika tidak ia
        # menjana "Thinking Process" panjang sebelum jawapan).
        tkw = ({"enable_thinking": False}
               if is_thinking_model(self.model_path)
               and not wants_thinking(self.model_path) else None)
        self._api = LmStudioBackend(manager.url + "/v1", model, temp, max_tokens,
                                    chat_template_kwargs=tkw)

    @property
    def last_timings(self):
        return self._api.last_timings

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
