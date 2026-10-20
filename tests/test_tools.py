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
    r1 = tools.tool_rag("dockerfile build image", k=1)
    assert "[docker-basics.md]" in r1
    assert tools.tool_rag("zzzqxjvwx") == "(no relevant docs)"
    # cache reuse: same object on repeat call
    assert tools._rag_index() is tools._rag_index()


def test_rag_indexes_learned_subdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    importlib.reload(tools)
    d = tmp_path / "docs" / "learned"
    d.mkdir(parents=True)
    (d / "note1.md").write_text("kubernetes pod ialah unit terkecil",
                                encoding="utf-8")
    out = tools.tool_rag("kubernetes pod unit terkecil", k=1)
    assert "[learned/note1.md]" in out


def test_learn_save_dedup_and_budget(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import store
    monkeypatch.setattr(store, "CHAT_DB", tmp_path / "chat.db")
    monkeypatch.setattr(store, "KNOW_DB", tmp_path / "knowledge.db")
    import importlib
    importlib.reload(tools)
    (tmp_path / "docs").mkdir()
    ctx = {"learn_budget": 2}
    t1 = ("Docker image ialah templat baca-sahaja untuk kontena. " * 6).strip()
    t2 = ("Volume Docker menyimpan data melebihi hayat kontena. " * 6).strip()
    t3 = ("Rangkaian Docker menyambung kontena antara satu sama lain. " * 6).strip()
    p1 = tools.learn_save("apa itu docker image", t1,
                          sources=["https://docs.docker.com/"], ctx=ctx)
    assert p1 is not None and p1.exists()
    assert ctx["learn_budget"] == 1
    assert tools.learn_save("dup", t1, ctx=ctx) is None
    assert ctx["learn_budget"] == 1  # dup tidak makan bajet
    p2 = tools.learn_save("apa itu volume", t2, ctx=ctx)
    assert p2 is not None and ctx["learn_budget"] == 0
    assert tools.learn_save("apa itu network", t3, ctx=ctx) is None
    assert tools.learn_save("pendek", "sikit je", ctx={"learn_budget": 1}) is None
    assert "Sumber: https://docs.docker.com/" in p1.read_text(encoding="utf-8")
    assert store.learned_count() == 2


def test_maybe_learn_skips_errors_and_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    importlib.reload(tools)
    (tmp_path / "docs").mkdir()
    import store
    monkeypatch.setattr(store, "CHAT_DB", tmp_path / "chat.db")
    monkeypatch.setattr(store, "KNOW_DB", tmp_path / "knowledge.db")
    calls = []
    monkeypatch.setattr(store, "log_training",
                        lambda *a, **k: calls.append((a, k)))
    assert tools.maybe_learn({}, "q", "ERROR: x") is None
    assert tools.maybe_learn({}, "q", "(no results)") is None
    long = ("Linux chmod 755 memberi kebenaran baca-laksana kepada semua. " * 6).strip()
    p = tools.maybe_learn({"learn_budget": 3}, "apa itu chmod", long,
                          sources=["https://linux.die.net/"], lang="ms")
    assert p is not None and p.name.endswith(".md")
    assert len(calls) == 1 and calls[0][1].get("kind") == "learn"
