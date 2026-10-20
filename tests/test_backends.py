"""backends.py: unified transcript policy. No binaries needed."""
from backends import transcript_messages, transcript_text


def test_transcript_caps_and_labels():
    msgs = [{"role": "user", "content": "u" * 1500},
            {"role": "assistant", "content": "a" * 800},
            {"role": "tool", "name": "run", "content": "r"}]
    tt = transcript_text(msgs)
    tm = transcript_messages(msgs)
    assert len(tm) == 3
    # ChatML: 3 turns + open assistant prefix
    assert len(tt) == 4 and tt[-1] == "<|im_start|>assistant\n"
    # content caps (tags excluded)
    assert tt[0] == "<|im_start|>user\n" + "u" * 1000 + "<|im_end|>"
    assert tt[1] == "<|im_start|>assistant\n" + "a" * 300 + "<|im_end|>"
    assert tt[2] == "<|im_start|>user\n[Hasil tool run]: r<|im_end|>"
    assert tm[2] == {"role": "user", "content": "[Hasil tool run]: r"}
    assert tm[0]["content"].endswith("u" * 1000)


def test_transcript_recent_only():
    msgs = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    assert len(transcript_text(msgs)) == 5  # 4 turns + prefix
    assert transcript_text(msgs)[0].endswith("q16<|im_end|>")


def test_cli_threads_physical():
    from backends import LlamaCliBackend
    be = LlamaCliBackend("nope", "nope.gguf")
    import os
    assert 1 <= be.threads <= (os.cpu_count() or 4)


def test_server_timings_surfaced():
    """chat_ex stores llama-server timings; fmt in ezcodex renders them."""
    import json
    import threading
    import urllib.error
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    payload = {"choices": [{"message": {"content": "ok"},
                            "finish_reason": "stop"}],
               "usage": {"completion_tokens": 10},
               "timings": {"prompt_n": 40, "prompt_ms": 1200.0,
                           "predicted_n": 10, "predicted_ms": 500.0}}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
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
        from backends import LmStudioBackend
        be = LmStudioBackend(f"http://127.0.0.1:{srv.server_port}/v1")
        for _ in range(2):  # loopback Windows boleh putus sekali-sekala
            try:
                out, tps, fin = be.chat_ex("sys", [{"role": "user", "content": "hi"}])
                break
            except (ConnectionError, urllib.error.URLError):
                continue
        assert out == "ok" and fin == "stop" and tps > 0
        assert be.last_timings["predicted_ms"] == 500.0
    finally:
        srv.shutdown()


def test_reasoning_content_fallback():
    """gemma4: content kosong + reasoning_content berisi -> guna reasoning."""
    import json
    import threading
    import urllib.error
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    payload = {"choices": [{"message": {"role": "assistant", "content": "",
                                        "reasoning_content": "fikir: ini imej"},
                            "finish_reason": "stop"}],
               "usage": {"completion_tokens": 5}}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
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
        from backends import LmStudioBackend
        be = LmStudioBackend(f"http://127.0.0.1:{srv.server_port}/v1")
        for _ in range(2):
            try:
                out, _, _ = be.chat_ex("sys", [{"role": "user", "content": "hi"}])
                break
            except (ConnectionError, urllib.error.URLError):
                continue
        assert out == "fikir: ini imej"
    finally:
        srv.shutdown()
