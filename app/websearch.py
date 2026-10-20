"""Web search + page fetch untuk mod online XCoder - stdlib sahaja, tiada pip.

DuckDuckGo HTML (tanpa kunci API) dengan fallback Wikipedia. Semua panggilan
rangkaian berkunci: timeout, saiz, skema http/https, dan host persendirian
(SSRF) ditolak. Kegagalan rangkaian ditukar kepada teks supaya chat tidak
pernah terhenti.
"""
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) XCoder/1.0"
TIMEOUT = 10
MAX_BYTES = 512_000
MAX_TEXT = 6000
MAX_REDIRECTS = 3

_TEXT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml+xml")
_BLOCK_TAGS = ("p", "br", "div", "li", "ul", "ol", "tr", "table", "section",
               "article", "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6")
_SKIP_TAGS = ("script", "style", "noscript", "template", "svg", "head")


def _collapse(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _ip_bad(a):
    return (a.is_private or a.is_loopback or a.is_link_local or a.is_reserved
            or a.is_unspecified or a.is_multicast)


def _host_ok(url, resolve=False):
    """True jika URL selamat: skema http/https + host awam (SSRF ditolak)."""
    try:
        p = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").strip(".")
    if not host:
        return False
    low = host.lower()
    if low == "localhost" or low.endswith((".local", ".internal", ".localhost")):
        return False
    try:
        return not _ip_bad(ipaddress.ip_address(low))
    except ValueError:
        pass
    if not resolve:
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for i in infos:
        try:
            if _ip_bad(ipaddress.ip_address(i[4][0])):
                return False
        except ValueError:
            return False
    return bool(infos)


class _LimitedRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _host_ok(newurl):
            raise urllib.error.HTTPError(newurl, code, "redirect blocked", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = None


def _opener():
    global _OPENER
    if _OPENER is None:
        _OPENER = urllib.request.build_opener(_LimitedRedirect)
    return _OPENER


def _get(url, timeout=TIMEOUT, max_bytes=MAX_BYTES):
    """GET dengan had saiz/masa + guard skema/host. Returns teks atau raise."""
    if not _host_ok(url, resolve=True):
        raise ValueError(f"URL tidak dibenarkan: {url}")
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept-Language": "ms,en;q=0.8",
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.1"})
    with _opener().open(req, timeout=timeout) as r:
        if getattr(r, "status", 200) != 200:
            raise ValueError(f"HTTP {getattr(r, 'status', '?')}")
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype and not (ctype.startswith(_TEXT_TYPES[0])
                          or ctype in _TEXT_TYPES[1:]):
            raise ValueError(f"bukan teks: {ctype}")
        raw = r.read(max_bytes + 1)
    return raw[:max_bytes].decode("utf-8", errors="replace")


class _TextOnly(HTMLParser):
    """Buang tag/script/style, tinggal teks nampak."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def strip_html(blob):
    p = _TextOnly()
    try:
        p.feed(blob or "")
        p.close()
    except Exception:
        pass
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def fetch_text(url, max_chars=MAX_TEXT):
    """Teks bersih dari URL (dipotong). Ralat ditukar kepada teks ERROR."""
    try:
        blob = _get(url)
    except Exception as e:
        return f"ERROR: gagal muat {url}: {e}"
    text = strip_html(blob)
    if not text:
        return "(laman kosong)"
    return text[:max_chars]


_PAGE_SKIP = ("script", "style", "noscript", "template", "svg")


class _PageExtract(HTMLParser):
    """Kumpul title, meta description, pautan <a>, dan teks nampak dari satu laman."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.desc = ""
        self.links = []
        self.parts = []
        self._skip = 0
        self._in_title = 0
        self._a = None
        self._abuf = []

    def handle_starttag(self, tag, attrs):
        if tag in _PAGE_SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        d = dict(attrs)
        if tag == "title":
            self._in_title += 1
        elif tag == "meta":
            key = (d.get("name") or d.get("property") or "").lower()
            if key in ("description", "og:description", "og:title") and not self.desc:
                self.desc = _collapse(d.get("content") or "")
        elif tag == "a":
            href = (d.get("href") or "").strip()
            if href:
                self._a, self._abuf = href, []
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in _BLOCK_TAGS and not self._skip:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _PAGE_SKIP:
            if self._skip:
                self._skip -= 1
            return
        if self._skip:
            return
        if tag == "title" and self._in_title:
            self._in_title -= 1
        elif tag == "a" and self._a is not None:
            text = _collapse("".join(self._abuf))
            if text:
                self.links.append((text, _unwrap(self._a)))
            self._a, self._abuf = None, []
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title += data
        if self._a is not None:
            self._abuf.append(data)
        self.parts.append(data)


def extract_page(url, max_chars=MAX_TEXT, max_links=15):
    """Ekstrak TITLE/DESCRIPTION/LINKS/TEXT bersih dari URL. Ralat -> teks ERROR."""
    try:
        blob = _get(url)
    except Exception as e:
        return f"ERROR: gagal ekstrak {url}: {e}"
    p = _PageExtract()
    try:
        p.feed(blob)
        p.close()
    except Exception:
        pass
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.parts))
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()[:max_chars]
    out = [f"URL: {url}"]
    if _collapse(p.title):
        out.append("TITLE: " + _collapse(p.title))
    if p.desc:
        out.append("DESCRIPTION: " + p.desc)
    seen, links = set(), []
    for t, u in p.links:
        if u in seen or u.startswith(("mailto:", "javascript:", "#")):
            continue
        seen.add(u)
        links.append(f"- {t} -> {u}")
        if len(links) >= max_links:
            break
    if links:
        out.append("LINKS:\n" + "\n".join(links))
    out.append("TEXT:\n" + (text or "(kosong)"))
    return "\n\n".join(out)[:max_chars + 2000]


def _unwrap(href):
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    p = urllib.parse.urlsplit(href)
    if (p.netloc or "").endswith("duckduckgo.com") and p.path.startswith("/l/"):
        u = urllib.parse.parse_qs(p.query).get("uddg", [""])[0]
        if u:
            return urllib.parse.unquote(u)
    return href


class _DDGParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._cur = None
        self._mode = None
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        d = dict(attrs)
        tokens = (d.get("class") or "").split()
        if "result__a" in tokens:
            self._flush()
            self._cur = {"title": "", "url": _unwrap(d.get("href", "")), "snippet": ""}
            self._mode, self._buf = "title", []
        elif "result__snippet" in tokens and self._cur is not None:
            self._mode, self._buf = "snippet", []

    def handle_endtag(self, tag):
        if tag == "a" and self._mode and self._cur is not None:
            text = _collapse("".join(self._buf))
            if self._mode == "title":
                self._cur["title"] = text
            else:
                self._cur["snippet"] = text
            self._mode, self._buf = None, []

    def handle_data(self, data):
        if self._mode:
            self._buf.append(data)

    def _flush(self):
        if self._cur and (self._cur["title"] or self._cur["url"]):
            self.results.append(self._cur)
        self._cur = None

    def close(self):
        self._flush()
        super().close()


def parse_ddg(blob):
    p = _DDGParser()
    try:
        p.feed(blob or "")
        p.close()
    except Exception:
        pass
    return [r for r in p.results if r.get("url")]


def search_ddg(q, k=5):
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(q)
    return parse_ddg(_get(url))[:k]


def search_wikipedia(q, lang="en", k=2):
    """Cari + ringkasan intro Wikipedia (ms/en). Returns [{title,url,snippet}]."""
    api = f"https://{lang}.wikipedia.org/w/api.php"
    p1 = {"action": "query", "list": "search", "srsearch": q,
          "srlimit": str(k), "format": "json", "utf8": "1"}
    data = json.loads(_get(api + "?" + urllib.parse.urlencode(p1)))
    hits = (data.get("query", {}) or {}).get("search", []) or []
    titles = [x.get("title") for x in hits if x.get("title")][:k]
    if not titles:
        return []
    p2 = {"action": "query", "prop": "extracts", "exintro": "1",
          "explaintext": "1", "redirects": "1", "titles": "|".join(titles),
          "format": "json", "utf8": "1"}
    data2 = json.loads(_get(api + "?" + urllib.parse.urlencode(p2)))
    pages = (data2.get("query", {}) or {}).get("pages", {}) or {}
    out = []
    for pg in pages.values():
        t = pg.get("title") or (titles[0] if titles else "")
        if not t:
            continue
        out.append({
            "title": t,
            "url": f"https://{lang}.wikipedia.org/wiki/"
                   + urllib.parse.quote(t.replace(" ", "_")),
            "snippet": _collapse(pg.get("extract", ""))[:1200]})
    return out[:k]


def search(q, lang="ms", k=5):
    """Hasil carian sebagai teks untuk model. DDG dulu, Wikipedia sebagai ganti."""
    q = _collapse(q)
    if not q:
        return "(no results)"
    results = []
    try:
        results = search_ddg(q, k)
    except Exception:
        results = []
    if len(results) < 2:
        seen = {r["url"] for r in results}
        wlan = (lang, "en") if lang == "ms" else ("en",)
        for wl in wlan:
            try:
                for r in search_wikipedia(q, wl, k=2):
                    if r["url"] not in seen:
                        results.append(r)
                        seen.add(r["url"])
            except Exception:
                continue
            if len(results) >= 2:
                break
    if not results:
        return "(no results)"
    if lang == "ms":
        lines = [f"HASIL CARIAN: {q}"]
    else:
        lines = [f"SEARCH RESULTS: {q}"]
    for i, r in enumerate(results, 1):
        sn = _collapse(r.get("snippet", ""))[:400]
        lines.append(f"{i}. {r.get('title') or '(tiada tajuk)'} - {r['url']}")
        if sn:
            lines.append(f"   {sn}")
    src = "SUMBER: " if lang == "ms" else "SOURCES: "
    lines.append(src + ", ".join(r["url"] for r in results[:5]))
    return "\n".join(lines)
