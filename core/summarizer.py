"""
summarizer.py
-------------
Generates a document summary using a map-reduce approach with Claude:
each chunk group gets a short summary ("map"), then those are combined
into one final summary ("reduce"). Falls back to a dependency-light
extractive summarizer (word-frequency sentence scoring) if no API key
is configured, so the app still works end-to-end out of the box.
"""

import os
import re
from collections import Counter
from typing import List

MODEL = "claude-sonnet-4-6"
GROUP_CHAR_BUDGET = 6000  # how much chunk text to feed per map call

try:
    import anthropic
    _client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
except ImportError:
    anthropic = None
    _client = None


def ai_available() -> bool:
    return _client is not None


def _group_chunks(chunks, budget=GROUP_CHAR_BUDGET):
    groups, current, size = [], [], 0
    for c in chunks:
        if size + len(c.text) > budget and current:
            groups.append(current)
            current, size = [], 0
        current.append(c)
        size += len(c.text)
    if current:
        groups.append(current)
    return groups


def _map_summarize(text: str) -> str:
    resp = _client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": (
                "Summarize the key points of this document excerpt in 3-5 "
                "concise bullet points. Just the bullets, no preamble.\n\n"
                f"---\n{text}\n---"
            ),
        }],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))


def _reduce_summarize(partial_summaries: List[str], style: str) -> str:
    joined = "\n\n".join(f"[Section {i+1}]\n{s}" for i, s in enumerate(partial_summaries))
    style_instructions = {
        "brief": "Write a tight 3-4 sentence executive summary.",
        "detailed": "Write a structured summary with short section headers and bullet points covering all major themes.",
        "bullets": "Write the summary as a single flat list of 6-10 concise bullet points covering the whole document.",
    }
    instruction = style_instructions.get(style, style_instructions["brief"])

    resp = _client.messages.create(
        model=MODEL,
        max_tokens=900,
        messages=[{
            "role": "user",
            "content": (
                f"Below are section-by-section summaries of a single document. "
                f"Combine them into one coherent overall summary of the whole document. "
                f"{instruction}\n\n{joined}"
            ),
        }],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))


def _extractive_fallback(full_text: str, max_sentences: int = 8) -> str:
    """Simple word-frequency sentence scoring — no model, no network,
    works instantly. Not as fluent as an LLM summary, but always available."""
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 25]
    if not sentences:
        return "Document too short to summarize."

    words = re.findall(r"[a-zA-Z]{3,}", full_text.lower())
    stop = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
        "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy",
        "did", "its", "let", "put", "say", "she", "too", "use", "with", "this",
        "that", "from", "have", "will", "your", "they", "their", "been", "than",
    }
    freq = Counter(w for w in words if w not in stop)

    scored = []
    for i, s in enumerate(sentences):
        s_words = re.findall(r"[a-zA-Z]{3,}", s.lower())
        score = sum(freq.get(w, 0) for w in s_words) / (len(s_words) + 1)
        scored.append((score, i, s))

    top = sorted(scored, reverse=True)[:max_sentences]
    top_in_order = [s for _, _, s in sorted(top, key=lambda t: t[1])]
    return "• " + "\n• ".join(top_in_order)


def summarize_document(full_text: str, chunks, style: str = "brief", progress_cb=None) -> dict:
    """Returns {"summary": str, "engine": str}."""
    if not _client:
        return {"summary": _extractive_fallback(full_text), "engine": "extractive-fallback (no API key set)"}

    try:
        groups = _group_chunks(chunks)
        partials = []
        for i, g in enumerate(groups):
            text = " ".join(c.text for c in g)
            partials.append(_map_summarize(text))
            if progress_cb:
                progress_cb(i + 1, len(groups))

        if len(partials) == 1:
            # Single group already fairly summarized; still pass through reduce
            # for consistent style/formatting.
            final = _reduce_summarize(partials, style)
        else:
            final = _reduce_summarize(partials, style)

        return {"summary": final, "engine": MODEL}
    except Exception as e:  # noqa: BLE001
        return {"summary": _extractive_fallback(full_text), "engine": f"extractive-fallback (AI error: {e})"}
