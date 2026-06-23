"""
page_fetcher.py
-----------------
Downloads web pages and extracts clean, readable text from the HTML,
stripping nav bars, ads, scripts, footers, etc. Pure Python: requests +
BeautifulSoup, no headless browser.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("rag_chatbot.page_fetcher")

DEFAULT_TIMEOUT = 8
MAX_PAGE_BYTES = 3_000_000  # don't bother downloading/parsing huge pages

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

# Tags whose contents are never useful as "main content" text.
_STRIP_TAGS = [
    "script", "style", "noscript", "nav", "footer", "header", "form",
    "aside", "iframe", "svg", "button", "input", "select", "option",
]


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    error: str | None = None


def _extract_main_text(soup: BeautifulSoup) -> str:
    """Heuristic main-content extraction: prefer <article>/<main>, then
    fall back to the largest text-bearing container, then to <body>."""
    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    candidates = soup.find_all(["article", "main"])
    if not candidates:
        # fall back: pick the div/section with the most paragraph text
        containers = soup.find_all(["div", "section"])
        best, best_len = None, 0
        for c in containers:
            text_len = len(c.get_text(strip=True))
            if text_len > best_len:
                best, best_len = c, text_len
        candidates = [best] if best is not None else []

    if candidates:
        text = "\n".join(c.get_text(separator="\n", strip=True) for c in candidates if c)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines / whitespace left over from stripped tags
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def fetch_page(url: str, timeout: int = DEFAULT_TIMEOUT) -> FetchedPage:
    """Download a URL and extract clean readable text from it.

    Never raises; failures are reported via FetchedPage.success/error so
    callers can keep processing other URLs in a batch without try/except
    boilerplate everywhere.
    """
    try:
        resp = requests.get(
            url, headers=_HEADERS, timeout=timeout, stream=True
        )
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return FetchedPage(
                url=url, title="", text="", success=False,
                error=f"unsupported content-type: {content_type}",
            )

        raw = resp.raw.read(MAX_PAGE_BYTES + 1, decode_content=True)
        if len(raw) > MAX_PAGE_BYTES:
            logger.warning("fetch_page: truncating oversized page %s", url)
        html = raw.decode(resp.encoding or "utf-8", errors="replace")

        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        text = _extract_main_text(soup)
        if not text:
            return FetchedPage(
                url=url, title=title, text="", success=False,
                error="no extractable text",
            )

        return FetchedPage(url=url, title=title, text=text, success=True)

    except requests.RequestException as e:
        logger.warning("fetch_page: failed to fetch %s: %s", url, e)
        return FetchedPage(url=url, title="", text="", success=False, error=str(e))
    except Exception as e:  # noqa: BLE001 - last-resort guard for parser oddities
        logger.warning("fetch_page: unexpected error for %s: %s", url, e)
        return FetchedPage(url=url, title="", text="", success=False, error=str(e))


def fetch_pages(urls: list[str], timeout: int = DEFAULT_TIMEOUT) -> list[FetchedPage]:
    """Fetch multiple URLs sequentially (kept simple/synchronous on purpose
    so behavior is easy to reason about on a local machine; swap in a
    ThreadPoolExecutor here later if you need more speed)."""
    return [fetch_page(u, timeout=timeout) for u in urls]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    page = fetch_page("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    print(page.title)
    print(page.text[:500])
