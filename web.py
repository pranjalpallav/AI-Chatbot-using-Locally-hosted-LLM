"""Live web search + crawling.
 
Pipeline: search (DuckDuckGo) -> fetch pages in parallel (robots.txt respected)
-> extract main text -> chunk -> rank chunks against the question with embeddings.
Only pages that were really fetched (HTTP 200) are returned, so every link shown
to the user is one the app actually opened.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib import robotparser
from urllib.parse import urlparse
 
import numpy as np
import requests
 
UA = "Mozilla/5.0 (compatible; DocChatBot/1.0; student-project)"
HEADERS = {"User-Agent": UA, "Accept-Language": "en"}
_robots: dict = {}
 
 
def _cfg(name: str) -> str:
    """Read a setting from env/.env (local) or Streamlit secrets (cloud)."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""
 
 
def tavily_search(query: str, key: str, n: int = 6) -> list[dict]:
    """Search via Tavily (free tier). Returns pages with text already extracted."""
    r = requests.post("https://api.tavily.com/search",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"query": query, "max_results": n, "search_depth": "basic"},
                      timeout=20)
    r.raise_for_status()
    return [{"title": x.get("title", ""), "url": x["url"], "text": x.get("content", "")}
            for x in r.json().get("results", []) if x.get("url") and x.get("content")]
 
 
def search(query: str, n: int = 6) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:                      # older package name
        from duckduckgo_search import DDGS
    results = DDGS().text(query, max_results=n)
    out = []
    for r in results or []:
        url = r.get("href") or r.get("url")
        if url:
            out.append({"title": r.get("title", ""), "url": url, "snippet": r.get("body", "")})
    return out
 
 
def allowed(url: str) -> bool:
    """Check robots.txt (cached per site). If it can't be read, allow."""
    p = urlparse(url)
    origin = f"{p.scheme}://{p.netloc}"
    rp = _robots.get(origin)
    if rp is None:
        rp = robotparser.RobotFileParser()
        try:
            r = requests.get(origin + "/robots.txt", headers=HEADERS, timeout=4)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
            else:
                rp.allow_all = True
        except requests.RequestException:
            rp.allow_all = True
        _robots[origin] = rp
    return rp.can_fetch(UA, url)
 
 
def fetch_page(url: str, timeout: int = 8) -> dict | None:
    """Return {"url": final_url, "text": main_text} or None if unreachable/blocked."""
    import trafilatura
    try:
        if not allowed(url):
            return None
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", "").lower():
            return None
        text = trafilatura.extract(r.text[:1_500_000], include_comments=False, include_tables=False)
        if not text or len(text.split()) < 40:
            return None
        return {"url": r.url, "text": text}
    except Exception:
        return None
 
 
def _split(text: str, size: int = 150, overlap: int = 30, min_words: int = 25) -> list[str]:
    words, out, step = text.split(), [], size - overlap
    for s in range(0, len(words), step):
        piece = " ".join(words[s:s + size])
        if len(piece.split()) >= min_words:
            out.append(piece)
        if s + size >= len(words):
            break
    return out
 
 
def crawl(query: str, embed_fn, n_results: int = 6, k: int = 4, per_url: int = 2):
    """Return (hits, error). hits = top-k chunks: {text, source(title), url, kind, fetched, score}."""
    key = _cfg("TAVILY_API_KEY")
    if key:     # Tavily returns the page text itself, so nothing is crawled (works on cloud)
        try:
            results = tavily_search(query, key, n_results)
        except Exception as e:
            return [], f"search failed: {str(e)[:80]}"
        if not results:
            return [], "no search results"
        pages, min_words = results, 8
    else:       # fallback: DuckDuckGo + crawl the pages ourselves
        try:
            results = search(query, n_results)
        except Exception as e:
            return [], f"search failed: {str(e)[:80]}"
        if not results:
            return [], "no search results"
        with ThreadPoolExecutor(max_workers=6) as ex:
            pages = list(ex.map(lambda r: fetch_page(r["url"]), results))
        min_words = 25
 
    stamp = time.strftime("%d %b %Y, %H:%M")
    chunks = []
    for r, page in zip(results, pages):
        if not page:
            continue
        for piece in _split(page["text"], min_words=min_words)[:15]:
            chunks.append({"text": piece, "source": r["title"] or urlparse(page["url"]).netloc,
                           "url": page["url"], "kind": "web", "fetched": stamp})
    if not chunks:
        return [], "none of the result pages could be opened"
 
    vecs = embed_fn([c["text"] for c in chunks])
    q = embed_fn([query])[0]
    scores = vecs @ q
    hits, per = [], {}
    for i in np.argsort(-scores):
        c = chunks[i]
        if per.get(c["url"], 0) >= per_url:
            continue
        per[c["url"]] = per.get(c["url"], 0) + 1
        hits.append({**c, "score": float(scores[i])})
        if len(hits) >= k:
            break
    return hits, None
