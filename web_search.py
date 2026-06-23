"""
web_search.py
--------------
Pure-Python web search with NO API keys, by scraping search engine HTML
result pages directly. DuckDuckGo's HTML endpoint is primary (no JS,
stable markup, no key); Bing's HTML results are a fallback if DuckDuckGo
fails or returns nothing (rate limit, network hiccup, markup change).

This is intentionally dependency-light: requests + BeautifulSoup only.
"""

from __future__ import annotations

import time
import random
import logging
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("rag_chatbot.web_search")

# A small pool of realistic desktop user-agents. Search engines are more
# likely to serve normal HTML (rather than a CAPTCHA wall) to requests
# that look like an ordinary browser.
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

DEFAULT_TIMEOUT = 10  # seconds per HTTP request


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def _headers() -> dict:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _clean_ddg_redirect(href: str) -> str:
    """DuckDuckGo HTML results wrap links like '/l/?uddg=<urlencoded>'.
    Unwrap to the real target URL when present."""
    if href.startswith("//duckduckgo.com/l/") or href.startswith("/l/"):
        qs = parse_qs(urlparse(href).query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return href


def _search_duckduckgo(query: str, max_results: int) -> list[SearchResult]:
    url = "https://html.duckduckgo.com/html/"
    resp = requests.post(
        url, data={"q": query}, headers=_headers(), timeout=DEFAULT_TIMEOUT
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    results: list[SearchResult] = []
    for result_div in soup.select("div.result"):
        link_tag = result_div.select_one("a.result__a")
        if not link_tag:
            continue
        href = _clean_ddg_redirect(link_tag.get("href", ""))
        title = link_tag.get_text(strip=True)
        snippet_tag = result_div.select_one("a.result__snippet") or result_div.select_one(
            "div.result__snippet"
        )
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

        if href and href.startswith("http"):
            results.append(SearchResult(title=title, url=href, snippet=snippet))
        if len(results) >= max_results:
            break
    return results


def _search_bing(query: str, max_results: int) -> list[SearchResult]:
    url = "https://www.bing.com/search"
    resp = requests.get(
        url, params={"q": query}, headers=_headers(), timeout=DEFAULT_TIMEOUT
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    results: list[SearchResult] = []
    for li in soup.select("li.b_algo"):
        link_tag = li.select_one("h2 a")
        if not link_tag:
            continue
        href = link_tag.get("href", "")
        title = link_tag.get_text(strip=True)
        snippet_tag = li.select_one(".b_caption p")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

        if href and href.startswith("http"):
            results.append(SearchResult(title=title, url=href, snippet=snippet))
        if len(results) >= max_results:
            break
    return results


def search_web(
    query: str,
    max_results: int = 10,
    retries: int = 2,
    delay_range: tuple[float, float] = (0.5, 1.5),
) -> list[SearchResult]:
    """
    Search the web with no API key by scraping DuckDuckGo's HTML results,
    falling back to Bing if that yields nothing usable.

    Args:
        query: search query string.
        max_results: max number of results to return.
        retries: how many times to retry each engine on failure.
        delay_range: random delay (seconds) before each request, to be a
            bit more polite / less bot-like.

    Returns:
        List of SearchResult, possibly empty if both engines fail.
    """
    engines = [("duckduckgo", _search_duckduckgo), ("bing", _search_bing)]

    for engine_name, engine_fn in engines:
        for attempt in range(1, retries + 1):
            try:
                time.sleep(random.uniform(*delay_range))
                results = engine_fn(query, max_results)
                if results:
                    logger.info(
                        "search_web: got %d results from %s for query=%r",
                        len(results),
                        engine_name,
                        query,
                    )
                    return results
                logger.warning(
                    "search_web: %s returned 0 results (attempt %d) for query=%r",
                    engine_name,
                    attempt,
                    query,
                )
            except requests.RequestException as e:
                logger.warning(
                    "search_web: %s failed (attempt %d): %s", engine_name, attempt, e
                )
        # move on to next engine if this one exhausted retries

    logger.error("search_web: all engines failed for query=%r", query)
    return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for r in search_web("what is retrieval augmented generation", max_results=5):
        print(r.title, "->", r.url)
        print(" ", r.snippet[:120])
