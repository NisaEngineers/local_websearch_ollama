"""
context_builder.py
---------------------
Packs ranked sources into a single context block for the LLM prompt,
respecting a character budget (a cheap stand-in for a token budget --
phi3:mini's context window is modest, so we keep this conservative).

Sources are added in ranked order (tier first, relevance second) so if
the budget runs out, what gets cut is always the lowest-priority
material, never the highest.
"""

from __future__ import annotations

from dataclasses import dataclass

from ranker import RankedSource

# Rough rule of thumb: ~4 chars/token in English. phi3:mini has a 4k
# context window by default; we reserve room for the system prompt,
# the question, and the model's answer, so keep the source context
# itself well under that in characters.
DEFAULT_CHAR_BUDGET = 6000
PER_SOURCE_CHAR_CAP = 1500  # cap any single source so one giant page can't eat the whole budget


@dataclass
class BuiltContext:
    text: str
    used_sources: list[RankedSource]


def build_context(ranked: list[RankedSource], char_budget: int = DEFAULT_CHAR_BUDGET) -> BuiltContext:
    """
    Assemble a numbered, citation-friendly context block from ranked
    sources, stopping once the character budget is exhausted.
    """
    blocks: list[str] = []
    used: list[RankedSource] = []
    remaining = char_budget

    for i, rs in enumerate(ranked, start=1):
        if remaining <= 0:
            break

        snippet = rs.page.text.strip().replace("\n", " ")
        snippet = " ".join(snippet.split())  # collapse whitespace
        snippet = snippet[:PER_SOURCE_CHAR_CAP]

        header = f"[Source {i} | tier={rs.tier} ({rs.tier_name}) | {rs.page.url}]"
        block = f"{header}\n{snippet}"

        if len(block) > remaining:
            block = block[:remaining]
        if not block.strip():
            continue

        blocks.append(block)
        used.append(rs)
        remaining -= len(block) + 2  # account for the join separator

    return BuiltContext(text="\n\n".join(blocks), used_sources=used)


def build_rag_prompt(question: str, context: BuiltContext) -> str:
    """Build the final user-turn prompt that goes to the LLM, instructing
    it to ground its answer in the numbered sources and cite them."""
    if not context.text:
        return (
            f"Question: {question}\n\n"
            "No web sources were available. Answer from your own knowledge "
            "and clearly say that no web search results were found."
        )

    return (
        "Use ONLY the numbered sources below to answer the question. "
        "Cite each fact inline like [Source 1]. Paraphrase in your own "
        "words; do not copy more than a few words verbatim from any "
        "source. Do not add any name, date, place, or claim that isn't "
        "written in one of these sources, even if you think you know it. "
        "If the sources don't fully answer the question, say what's "
        "missing rather than guessing.\n\n"
        f"{context.text}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the sources above, in your own words:"
    )


SYSTEM_PROMPT = (
    "You are a careful research assistant. You answer ONLY using facts "
    "that literally appear in the numbered sources below. The sources are "
    "ordered by trustworthiness (lower tier number = more authoritative); "
    "prefer higher-tier sources when sources disagree, and say so "
    "explicitly if they conflict.\n\n"
    "STRICT RULES:\n"
    "1. Every fact, name, date, number, or quote you state must come from "
    "one of the numbered sources. If you state it, you must cite it as "
    "[Source N] where N is the source it actually came from.\n"
    "2. Never invent a citation. Do not write phrases like 'CBS News "
    "report' or 'Trump's Tweet' or name any person, country, or outlet "
    "as a source unless that exact source appears in the numbered list "
    "below. If something is not in the numbered sources, do not mention "
    "it at all, even if it sounds plausible or you recall it from "
    "training.\n"
    "3. Paraphrase in your own words. Do not copy phrases of more than a "
    "few words directly from a source's text, even with quotation marks. "
    "Summarize what the source says rather than reproducing its wording.\n"
    "4. If the sources don't fully answer the question, explicitly say "
    "what's missing instead of filling the gap with assumed or recalled "
    "information.\n"
    "5. Be concise. Do not pad the answer with speculation."
)
