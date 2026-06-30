"""
document_loader.py
-------------------
Extracts text from an uploaded PDF or .txt file and splits it into
overlapping chunks suitable for embedding/retrieval (RAG).
"""

import io
import re
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader


@dataclass
class Chunk:
    index: int
    text: str
    page: int | None = None


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from a PDF or plain-text file's bytes."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)
    else:
        return file_bytes.decode("utf-8", errors="ignore")


def extract_text_with_pages(file_bytes: bytes, filename: str):
    """Like extract_text, but returns a list of (page_number, text) so
    chunks can carry page provenance for PDFs. For .txt files, everything
    is treated as a single 'page'."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    else:
        return [(1, file_bytes.decode("utf-8", errors="ignore"))]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-\s+", "", text)  # rejoin hyphenated line-break words
    return text.strip()


def chunk_text(pages, chunk_size: int = 900, overlap: int = 150) -> List[Chunk]:
    """Split (page_num, text) pairs into overlapping word-based chunks,
    tracking which page each chunk mostly came from. chunk_size/overlap
    are in characters."""
    chunks: List[Chunk] = []
    idx = 0

    for page_num, raw in pages:
        text = clean_text(raw)
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # try to end on a sentence boundary for cleaner chunks
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size * 0.5:
                end = boundary + 1

            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(index=idx, text=piece, page=page_num))
                idx += 1

            if end >= len(text):
                break
            start = max(end - overlap, start + 1)

    return chunks


def load_and_chunk(file_bytes: bytes, filename: str, chunk_size: int = 900, overlap: int = 150):
    pages = extract_text_with_pages(file_bytes, filename)
    full_text = clean_text(" ".join(t for _, t in pages))
    chunks = chunk_text(pages, chunk_size=chunk_size, overlap=overlap)
    return full_text, chunks
