"""
qa.py
-----
Answers a user's question about the uploaded document using
retrieval-augmented generation: pull the most relevant chunks from the
RAG index, then ask Claude to answer strictly from that context (with
source page numbers). Falls back to returning the best-matching raw
chunk if no API key is configured.
"""

import os
from typing import List

MODEL = "claude-sonnet-4-6"

try:
    import anthropic
    _client = anthropic.Anthropic() if os.environ.get("ANTHROPIC_API_KEY") else None
except ImportError:
    anthropic = None
    _client = None


def ai_available() -> bool:
    return _client is not None


def answer_question(question: str, retrieved, chat_history=None) -> dict:
    """retrieved: list of RetrievedChunk from rag_engine.
    Returns {"answer": str, "sources": [...], "engine": str}."""
    if not retrieved:
        return {
            "answer": "I couldn't find anything relevant to that question in the document.",
            "sources": [],
            "engine": "none",
        }

    sources = [
        {"page": r.page, "score": round(r.score, 3), "preview": r.text[:160], "full_text": r.text}
        for r in retrieved
    ]

    if not _client:
        # Fallback: just surface the best-matching chunk directly.
        best = retrieved[0]
        page_note = f" (page {best.page})" if best.page else ""
        return {
            "answer": (
                f"[No API key set — showing the most relevant excerpt instead of a "
                f"generated answer]{page_note}:\n\n{best.text}"
            ),
            "sources": sources,
            "engine": "retrieval-only (no API key set)",
        }

    context = "\n\n".join(
        f"[Source {i+1}, page {r.page or '?'}]\n{r.text}" for i, r in enumerate(retrieved)
    )

    history_block = ""
    if chat_history:
        recent = chat_history[-4:]
        history_block = "\n\nPrevious conversation (for context only):\n" + "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in recent
        )

    prompt = (
        "Answer the user's question using ONLY the source excerpts below. "
        "If the answer isn't contained in the excerpts, say so plainly instead "
        "of guessing. When you use a source, reference it like (Source 1) or "
        "(page 3). Keep the answer concise and directly responsive.\n\n"
        f"Source excerpts:\n{context}"
        f"{history_block}\n\n"
        f"Question: {question}"
    )

    try:
        resp = _client.messages.create(
            model=MODEL,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return {"answer": answer, "sources": sources, "engine": MODEL}
    except Exception as e:  # noqa: BLE001
        best = retrieved[0]
        return {
            "answer": f"AI request failed ({e}). Closest matching excerpt:\n\n{best.text}",
            "sources": sources,
            "engine": "retrieval-only (AI error)",
        }
