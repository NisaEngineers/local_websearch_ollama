"""
rag_chatbot.py
----------------
Main entry point. Wires together:

    web_search   -> finds candidate URLs (DuckDuckGo HTML scrape, no API)
    page_fetcher -> downloads & cleans page text
    ranker       -> hierarchical ranking (domain trust tier, then TF-IDF relevance)
    context_builder -> packs ranked sources into a prompt within a budget
    ollama_client -> sends the prompt to a local Ollama model (phi3:mini)

Usage:
    python3 rag_chatbot.py                  # interactive chat loop
    python3 rag_chatbot.py "your question"  # single-shot question

Requirements:
    pip install requests beautifulsoup4 lxml
    Ollama running locally with the model pulled:
        ollama pull phi3:mini
"""

from __future__ import annotations

import sys
import logging
import argparse

from web_search import search_web
from page_fetcher import fetch_pages
from ranker import rank_sources
from context_builder import build_context, build_rag_prompt, SYSTEM_PROMPT
from ollama_client import generate, check_ollama_available, OllamaError, DEFAULT_MODEL
from fact_checker import check_grounding, annotate_answer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rag_chatbot")


def answer_question(
    question: str,
    model: str = DEFAULT_MODEL,
    max_search_results: int = 8,
    top_k_sources: int = 5,
    char_budget: int = 6000,
    verbose: bool = False,
) -> str:
    """Run the full search -> fetch -> rank -> generate pipeline for one question."""

    if verbose:
        print(f"\n[1/4] Searching the web for: {question!r}")
    results = search_web(question, max_results=max_search_results)
    if not results:
        print("  (no search results found; answering from model knowledge only)")
        prompt = build_rag_prompt(question, build_context([]))
        return generate(prompt, model=model, system=SYSTEM_PROMPT)

    if verbose:
        print(f"  found {len(results)} candidate pages, fetching content...")

    pages = fetch_pages([r.url for r in results])

    if verbose:
        ok = sum(p.success for p in pages)
        print(f"[2/4] Fetched {ok}/{len(pages)} pages successfully")

    ranked = rank_sources(question, pages, top_k=top_k_sources)

    if verbose:
        print(f"[3/4] Ranked sources (tier, relevance, url):")
        for r in ranked:
            print(f"   tier={r.tier} rel={r.relevance:.3f}  {r.page.url}")

    context = build_context(ranked, char_budget=char_budget)
    prompt = build_rag_prompt(question, context)

    if verbose:
        print(f"[4/4] Querying {model} via Ollama...\n")

    try:
        answer = generate(prompt, model=model, system=SYSTEM_PROMPT)
    except OllamaError as e:
        return f"[Ollama error] {e}"

    # Heuristic grounding check: flag named entities in the answer that
    # don't actually appear anywhere in the source text the model was
    # given. Small local models can cite a [Source N] confidently while
    # still inventing a name/place not present in that source -- this
    # catches that class of fabrication (see README "Known limitations").
    report = check_grounding(answer, context.text)
    if report.has_unverified:
        if verbose:
            print(f"  [grounding check] unverified entities: {report.unverified_entities}")
        answer = annotate_answer(answer, report)

    if context.used_sources:
        source_list = "\n".join(
            f"  [Source {i}] ({rs.tier_name}) {rs.page.url}"
            for i, rs in enumerate(context.used_sources, start=1)
        )
        answer = f"{answer}\n\nSources used:\n{source_list}"

    return answer


def interactive_loop(model: str, verbose: bool) -> None:
    print(f"RAG chatbot ready (model: {model}). Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        answer = answer_question(question, model=model, verbose=verbose)
        print(f"\nAssistant: {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web-search RAG chatbot using Ollama.")
    parser.add_argument("question", nargs="*", help="Ask a single question and exit.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    parser.add_argument("--verbose", action="store_true", help="Show pipeline steps and ranked sources.")
    args = parser.parse_args()

    if not check_ollama_available():
        print(
            "ERROR: Could not reach Ollama at http://localhost:11434.\n"
            "Make sure Ollama is installed and running (`ollama serve`), "
            f"and the model is pulled (`ollama pull {args.model}`)."
        )
        sys.exit(1)

    if args.question:
        question = " ".join(args.question)
        print(answer_question(question, model=args.model, verbose=args.verbose))
    else:
        interactive_loop(model=args.model, verbose=args.verbose)


if __name__ == "__main__":
    main()
