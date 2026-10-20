"""experts.py: GGUF pins (add/resolve/audit). Tmp dirs only."""
import json
from pathlib import Path

from experts import _gguf_ok, add, audit_models, file_hash, resolve


def _gguf(path, body=b"\x00" * 64):
    Path(path).write_bytes(b"GGUF" + body)
    return str(path)


def test_gguf_magic(tmp_path):
    g = _gguf(tmp_path / "a.gguf")
    assert _gguf_ok(g) is True
    b = tmp_path / "b.gguf"
    b.write_bytes(b"NOPE")
    assert _gguf_ok(b) is False
    assert _gguf_ok(tmp_path / "missing.gguf") is False


def test_add_pins_and_resolve_verifies(tmp_path):
    md = tmp_path / "models"
    g = _gguf(tmp_path / "m.gguf")
    e = add(md, g, name="xcoder-t", domains=["t"])
    assert len(e["sha256"]) == 64 and e["bytes"] > 4
    p, note = resolve(md, "xcoder-t")
    assert p is not None and note == ""
    # tamper -> resolve refuses
    (md / "m.gguf").write_bytes(b"GGUF" + b"\x99" * 64)
    p2, note2 = resolve(md, "xcoder-t")
    assert p2 is None and "semakan" in note2


def test_add_rejects_non_gguf(tmp_path):
    md = tmp_path / "models"
    b = tmp_path / "b.gguf"
    b.write_bytes(b"NOPE")
    try:
        add(md, str(b), name="xcoder-x", domains=["x"])
    except SystemExit as e:
        assert "bukan fail GGUF" in str(e)
    else:
        raise AssertionError("non-GGUF accepted")


def test_audit_hash_ok(tmp_path):
    md = tmp_path / "models"
    _gguf(tmp_path / "m.gguf")
    add(md, str(tmp_path / "m.gguf"), name="xcoder-t", domains=["t"])
    rep = audit_models(md)
    assert rep["models"][0]["hash_ok"] is True
    # unpinned file (dropped straight in) -> None, not False
    _gguf(md / "loose.gguf")
    rep2 = audit_models(md)
    loose = next(i for i in rep2["models"] if i["file"] == "loose.gguf")
    assert loose["hash_ok"] is None
