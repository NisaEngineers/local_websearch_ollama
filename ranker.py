"""
ranker.py
----------
Implements hierarchical relevance ranking over fetched web pages:

  1. DOMAIN TIER (primary sort key): each source domain is bucketed into
     a trust tier (gov/edu/major reference > established news/orgs >
     general web > forums/social/unknown). Higher tier always outranks
     lower tier, regardless of relevance score.

  2. RELEVANCE SCORE (secondary sort key, used to order *within* a tier):
     a lightweight TF-IDF cosine similarity between the query and each
     page's text, plus a small recency-agnostic snippet-overlap boost.
     No external ML deps -- implemented with stdlib `math`/`collections`.

This gives "hierarchical priority": a .gov/.edu/Wikipedia hit with a
mediocre relevance score still beats a highly relevant random blog,
which is usually what people mean by "trust the source first."
"""

from __future__ import annotations

import re
import math
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlparse

from page_fetcher import FetchedPage

# --- Domain trust tiers -----------------------------------------------
# Lower number = higher trust/priority. Extend these lists freely.
TIER_1_SUFFIXES = (".gov", ".edu", ".mil")
TIER_1_DOMAINS = {
    "wikipedia.org", "who.int", "un.org", "nature.com", "nih.gov",
    "arxiv.org", "ieee.org", "acm.org",
}
TIER_2_DOMAINS = {
    # wire services / major international news
    "reuters.com", "apnews.com", "afp.com", "bbc.com", "bbc.co.uk",
    "npr.org", "nytimes.com", "wsj.com", "theguardian.com",
    "economist.com", "bloomberg.com", "aljazeera.com", "cnbc.com",
    "cnn.com", "abcnews.go.com", "cbsnews.com", "nbcnews.com",
    "usatoday.com", "washingtonpost.com", "ft.com", "time.com",
    "axios.com", "politico.com", "thehill.com", "skynews.com",
    "news.sky.com", "dw.com", "france24.com", "euronews.com",
    "csmonitor.com", "newsweek.com",
    # tech / science news and docs
    "techcrunch.com", "arstechnica.com", "wired.com", "theverge.com",
    "stackoverflow.com", "docs.python.org", "developer.mozilla.org",
    "github.com", "scientificamerican.com", "newscientist.com",
}
TIER_4_DOMAINS = {
    "reddit.com", "quora.com", "facebook.com", "twitter.com", "x.com",
    "tiktok.com", "pinterest.com", "instagram.com",
}

TIER_NAMES = {
    1: "authoritative (gov/edu/major reference)",
    2: "established (reputable news/org/tech docs)",
    3: "general web",
    4: "forum/social/unverified",
}


def _registered_domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def domain_tier(url: str) -> int:
    """Return trust tier 1 (highest) to 4 (lowest) for a URL's domain."""
    domain = _registered_domain(url)

    if any(domain.endswith(suffix) for suffix in TIER_1_SUFFIXES):
        return 1
    if domain in TIER_1_DOMAINS or any(domain.endswith("." + d) for d in TIER_1_DOMAINS):
        return 1
    if domain in TIER_2_DOMAINS or any(domain.endswith("." + d) for d in TIER_2_DOMAINS):
        return 2
    if domain in TIER_4_DOMAINS or any(domain.endswith("." + d) for d in TIER_4_DOMAINS):
        return 4
    return 3  # default: general web


_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _tf(tokens: list[str]) -> Counter:
    return Counter(tokens)


def _cosine_tfidf_score(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    """Cosine similarity between query and doc, weighted by IDF computed
    across the candidate document set (so corpus-common words count for
    less). Pure stdlib implementation -- no sklearn/numpy needed."""
    if not query_tokens or not doc_tokens:
        return 0.0

    q_tf = _tf(query_tokens)
    d_tf = _tf(doc_tokens)

    q_vec = {w: q_tf[w] * idf.get(w, 0.0) for w in q_tf}
    d_vec = {w: d_tf[w] * idf.get(w, 0.0) for w in d_tf}

    common = set(q_vec) & set(d_vec)
    dot = sum(q_vec[w] * d_vec[w] for w in common)

    q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
    d_norm = math.sqrt(sum(v * v for v in d_vec.values()))

    if q_norm == 0 or d_norm == 0:
        return 0.0
    return dot / (q_norm * d_norm)


def _compute_idf(all_doc_tokens: list[list[str]]) -> dict[str, float]:
    n_docs = len(all_doc_tokens)
    df = Counter()
    for tokens in all_doc_tokens:
        for w in set(tokens):
            df[w] += 1
    return {w: math.log((n_docs + 1) / (count + 1)) + 1.0 for w, count in df.items()}


@dataclass
class RankedSource:
    page: FetchedPage
    tier: int
    relevance: float

    @property
    def tier_name(self) -> str:
        return TIER_NAMES.get(self.tier, "unknown")


def rank_sources(query: str, pages: list[FetchedPage], top_k: int | None = None) -> list[RankedSource]:
    """
    Rank fetched pages hierarchically:
      sort key = (domain_tier ASC, relevance_score DESC)

    Pages with no extractable text (fetch failures) are excluded.

    Args:
        query: the user's question.
        pages: pages already downloaded via page_fetcher.fetch_pages.
        top_k: if set, only return the top K ranked sources.

    Returns:
        List of RankedSource sorted best-first.
    """
    usable = [p for p in pages if p.success and p.text.strip()]
    if not usable:
        return []

    query_tokens = _tokenize(query)
    doc_token_lists = [_tokenize(p.text) for p in usable]
    idf = _compute_idf(doc_token_lists)

    ranked = []
    for page, doc_tokens in zip(usable, doc_token_lists):
        score = _cosine_tfidf_score(query_tokens, doc_tokens, idf)
        tier = domain_tier(page.url)
        ranked.append(RankedSource(page=page, tier=tier, relevance=score))

    ranked.sort(key=lambda r: (r.tier, -r.relevance))

    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked


if __name__ == "__main__":
    # quick smoke test with fake pages
    fake_pages = [
        FetchedPage(url="https://en.wikipedia.org/wiki/RAG", title="RAG wiki", text="retrieval augmented generation combines retrieval and generation models"),
        FetchedPage(url="https://reddit.com/r/ml/rag", title="reddit rag", text="lol rag is cool i guess retrieval augmented generation"),
        FetchedPage(url="https://random-blog.example.com/post", title="blog", text="retrieval augmented generation retrieval augmented generation retrieval"),
    ]
    for r in rank_sources("retrieval augmented generation", fake_pages):
        print(r.tier, r.tier_name, round(r.relevance, 3), r.page.url)
