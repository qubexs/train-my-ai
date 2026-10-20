"""websearch.py: parsing + fallback + guard. Tiada rangkaian sebenar - semua canned."""
import websearch

DDG_HTML = """
<html><body>
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fsqlite.org%2Fwhentouse.html&amp;rut=abc">When To Use SQLite</a>
  <a class="result__snippet" href="x">SQLite is often a better choice than <b>big</b> databases.</a>
</div>
<div class="result">
  <a class="result__a" href="https://www.postgresql.org/docs/">PostgreSQL Docs</a>
  <a class="result__snippet" href="y">Official PostgreSQL documentation.</a>
</div>
</body></html>
"""


def test_ddg_parse_and_unwrap():
    r = websearch.parse_ddg(DDG_HTML)
    assert len(r) == 2
    assert r[0]["url"] == "https://sqlite.org/whentouse.html"
    assert r[0]["title"] == "When To Use SQLite"
    assert "big" in r[0]["snippet"]
    assert r[1]["url"] == "https://www.postgresql.org/docs/"


def test_strip_html():
    t = websearch.strip_html(
        "<html><head><title>x</title><style>p{}</style></head>"
        "<body><p>Hello <b>world</b></p><script>alert(1)</script><p>Bye</p></body></html>")
    assert "Hello" in t and "world" in t and "Bye" in t
    assert "alert" not in t and "{}" not in t
    assert "x" not in t.split("\n")[0]  # title dalam head dibuang


def test_fetch_text_handles_error(monkeypatch):
    def boom(url, *a, **k):
        raise OSError("down")
    monkeypatch.setattr(websearch, "_get", boom)
    assert websearch.fetch_text("https://example.com/x").startswith("ERROR:")


def test_search_wikipedia_fallback(monkeypatch):
    def boom(q, k=5):
        raise RuntimeError("blocked")
    monkeypatch.setattr(websearch, "search_ddg", boom)
    monkeypatch.setattr(websearch, "search_wikipedia", lambda q, lang="en", k=2: [
        {"title": "Docker", "url": "https://en.wikipedia.org/wiki/Docker",
         "snippet": "Docker is a platform for containers."}])
    out = websearch.search("apa itu docker", lang="ms")
    assert "HASIL CARIAN" in out
    assert "wikipedia.org" in out
    assert "SUMBER:" in out


def test_search_empty(monkeypatch):
    monkeypatch.setattr(websearch, "search_ddg", lambda q, k=5: [])
    monkeypatch.setattr(websearch, "search_wikipedia", lambda q, lang="en", k=2: [])
    assert websearch.search("zzzqxjvwx", lang="ms") == "(no results)"


def test_search_ddg_success_no_wiki(monkeypatch):
    calls = {"wiki": 0}
    monkeypatch.setattr(websearch, "search_ddg", lambda q, k=5: [
        {"title": "A", "url": "https://a.example/1", "snippet": "aa"},
        {"title": "B", "url": "https://b.example/2", "snippet": "bb"}])

    def wiki(q, lang="en", k=2):
        calls["wiki"] += 1
        return []
    monkeypatch.setattr(websearch, "search_wikipedia", wiki)
    out = websearch.search("x")
    assert calls["wiki"] == 0
    assert "https://a.example/1" in out


def test_host_guard_blocks_private_and_bad_scheme():
    for bad in ("file:///etc/passwd", "ftp://example.com/x",
                "http://127.0.0.1:8080/x", "http://localhost/x",
                "http://10.0.0.5/x", "http://192.168.1.1/",
                "http://169.254.1.1/", "http://[::1]/x", "http://x.local/a"):
        assert not websearch._host_ok(bad), bad
    assert websearch._host_ok("https://example.com/page", resolve=False)


def test_host_guard_resolve_blocks_dns_to_private(monkeypatch):
    def fake_gai(host, port=None, *a, **k):
        return [(2, 1, 6, "", ("192.168.0.10", 0))]
    monkeypatch.setattr(websearch.socket, "getaddrinfo", fake_gai)
    assert not websearch._host_ok("http://sneaky.example/x", resolve=True)

    def public_gai(host, port=None, *a, **k):
        return [(2, 1, 6, "", ("93.184.216.34", 0))]
    monkeypatch.setattr(websearch.socket, "getaddrinfo", public_gai)
    assert websearch._host_ok("http://sneaky.example/x", resolve=True)
