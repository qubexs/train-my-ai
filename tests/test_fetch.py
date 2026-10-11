"""fetch.py: verify + pins. No network."""
from pathlib import Path

from fetch import PINNED, sha256_file, verify


def test_verify_missing(tmp_path):
    ok, reason = verify(tmp_path / "tiada.bin", 10, "abc")
    assert ok is False and "tiada" in reason


def test_verify_size_and_hash(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"GGUF" + b"\x00" * 100)
    assert verify(f, 104, sha256_file(f)) == (True, "ok")
    ok, reason = verify(f, 999, None)
    assert ok is False and "saiz" in reason
    ok, reason = verify(f, 104, "0" * 64)
    assert ok is False and "sha256" in reason


def test_base_model_pinned():
    pin = PINNED["qwen2.5-0.5b-instruct-q4_k_m.gguf"]
    assert pin["size"] == 491400032 and len(pin["sha256"]) == 64
