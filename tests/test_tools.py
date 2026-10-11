"""tools.py: edit/run/rag/jail. Stdlib only, no models. Run from repo root."""
import tools


def test_safe_blocks_escape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    importlib.reload(tools)
    try:
        tools.tool_read("../outside.txt")
    except ValueError as e:
        assert "outside workspace" in str(e)
    else:
        raise AssertionError("escape not blocked")


def test_edit_reports_match_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    importlib.reload(tools)
    (tmp_path / "f.txt").write_text("aaa bbb aaa", encoding="utf-8")
    out = tools.tool_edit("f.txt aaa => zzz")
    assert out.startswith("OK edited")
    assert "2 matches" in out
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "zzz bbb aaa"
    assert tools.tool_edit("f.txt qqq => zzz") == "ERROR: old text not found"
    assert "usage" in tools.tool_edit("f.txt noarrow")


def test_run_python_and_js():
    assert tools.tool_run("py print(40+2)").strip() == "42"
    assert tools.tool_run("print(1+1)").strip() == "2"  # auto: python
    assert tools.tool_run("js console.log(40+2)").strip() == "42"
    assert tools.tool_run("console.log(3*3)").strip() == "9"  # auto: js


def test_rag_bm25_and_cache(monkeypatch):
    from pathlib import Path
    import importlib
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    importlib.reload(tools)
    r1 = tools.tool_rag("docker image container", k=1)
    assert "[docker-basics.md]" in r1
    assert tools.tool_rag("zzz-no-such-topic-zzz") == "(no relevant docs)"
    # cache reuse: same object on repeat call
    assert tools._rag_index() is tools._rag_index()
