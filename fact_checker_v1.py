"""
fact_checker.py
------------------
A lightweight, heuristic post-generation check: small local models like
phi3:mini will sometimes invent specific names, places, or organizations
that are not actually present in the retrieved sources, even when
instructed not to (see README "Known limitations"). This module can't
verify *truth*, only *grounding* -- whether a claim's named entities
literally appear somewhere in the source text the model was given.

It is intentionally simple (regex-based capitalized-phrase extraction,
no NLP dependency) so it stays "pure Python, no extra deps." It will
have false positives/negatives, but it reliably catches the failure
mode seen in testing: a model confidently citing a fabricated person,
city, or organization that doesn't appear anywhere in its sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Words that are capitalized for reasons other than being a proper noun
# (sentence-initial common words, weekday/month names, discourse markers)
# and would otherwise produce noisy false positives.
_COMMON_WORDS = {
    "the", "this", "that", "these", "those", "it", "he", "she", "they",
    "we", "i", "a", "an", "however", "additionally", "despite", "since",
    "source", "sources", "note", "according", "meanwhile", "overall",
    "in", "on", "at", "as", "but", "and", "or", "if", "what", "when",
    "where", "who", "why", "how", "answer", "question",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

# Connector words allowed *inside* a multi-word capitalized phrase
# (e.g. "Prime Minister of Pakistan", "United States of America").
_CONNECTORS = {"of", "the", "and", "for"}

_CAP_PHRASE_RE = re.compile(
    r"\b[A-Z][a-zA-Z.]*(?:\s+(?:[A-Z][a-zA-Z.]*|of|the|and|for))*\b"
)


def extract_entities(text: str) -> set[str]:
    """Extract candidate named-entity-like phrases (people, places, orgs)
    from text via capitalization heuristics. Best-effort, not NLP-grade."""
    candidates: set[str] = set()

    for m in _CAP_PHRASE_RE.finditer(text):
        words = m.group(0).split()
        # trim leading/trailing connector or lowercase words
        while words and (words[0][0].islower()):
            words.pop(0)
        while words and (words[-1][0].islower()):
            words.pop()
        if not words:
            continue

        phrase = " ".join(words)
        lower_first = words[0].lower()

        if len(words) == 1 and (lower_first in _COMMON_WORDS or len(words[0]) < 3):
            continue
        if phrase.lower() in _COMMON_WORDS:
            continue

        candidates.add(phrase)

    return candidates


@dataclass
class GroundingReport:
    checked_entities: set[str]
    unverified_entities: set[str]

    @property
    def has_unverified(self) -> bool:
        return len(self.unverified_entities) > 0


def check_grounding(answer: str, source_text: str) -> GroundingReport:
    """
    Check whether named-entity-like phrases in `answer` actually appear
    in `source_text`. Case-insensitive substring match per entity.

    This is a heuristic safety net, not a fact-checker: it can't tell you
    a claim is *true*, only that its named entities are *not present* in
    what the model was given -- which is a strong signal of fabrication
    when it fires, but won't catch every hallucination (e.g. a wrong
    date built from real entities) and may rarely flag a real entity
    that's mentioned in the sources under different capitalization or
    phrasing.
    """
    entities = extract_entities(answer)
    source_lower = source_text.lower()

    unverified = {e for e in entities if e.lower() not in source_lower}

    return GroundingReport(checked_entities=entities, unverified_entities=unverified)


def annotate_answer(answer: str, report: GroundingReport) -> str:
    """Append a clear warning to the answer if unverified entities were
    found, so the user sees it inline rather than having to dig through
    logs."""
    if not report.has_unverified:
        return answer

    flagged = ", ".join(sorted(report.unverified_entities))
    warning = (
        "\n\n⚠ GROUNDING WARNING: the following names/places/terms in "
        f"the answer above were NOT found in the source text and may be "
        f"hallucinated by the model: {flagged}\n"
        "Treat these specific details with skepticism and verify "
        "independently."
    )
    return answer + warning


if __name__ == "__main__":
    fake_answer = (
        "President Trump announced a deal. Pakistan Prime Minister "
        "Shehbaz Sharif mediated a signing in Geneva, Switzerland."
    )
    fake_sources = (
        "President Trump said the war could end soon. The two sides "
        "are negotiating a ceasefire extension."
    )
    report = check_grounding(fake_answer, fake_sources)
    print("Entities found:", report.checked_entities)
    print("Unverified:", report.unverified_entities)
    print(annotate_answer(fake_answer, report))
