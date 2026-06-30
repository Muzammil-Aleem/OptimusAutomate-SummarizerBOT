"""
rag_engine.py
-------------
Builds a searchable index over document chunks and retrieves the most
relevant ones for a given question (the "R" in RAG).

Two backends, auto-selected:
  - SemanticIndex: real sentence embeddings via `sentence-transformers`
    + cosine search, used automatically if that package is installed.
  - TfidfIndex: a dependency-light fallback using scikit-learn's
    TF-IDF vectorizer + cosine similarity. No model download, works
    instantly offline, and is good enough for keyword-rich documents.

Both expose the same interface: build(chunks) / query(text, k).
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False


@dataclass
class RetrievedChunk:
    text: str
    page: int | None
    score: float
    index: int


class TfidfIndex:
    """Lightweight, zero-download retrieval backend."""
    name = "TF-IDF (lightweight, offline)"

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self.matrix = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def query(self, text: str, k: int = 4) -> List[RetrievedChunk]:
        if self.matrix is None or not self.chunks:
            return []
        q_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [
            RetrievedChunk(text=self.chunks[i].text, page=self.chunks[i].page,
                            score=float(sims[i]), index=self.chunks[i].index)
            for i in top_idx if sims[i] > 0
        ]


class SemanticIndex:
    """Real embedding-based retrieval backend (used if sentence-transformers
    is installed). Falls back to brute-force cosine similarity over
    normalized embeddings — fine for document-sized corpora."""
    name = "Semantic embeddings (all-MiniLM-L6-v2)"

    _model = None  # loaded lazily and shared across instances

    def __init__(self):
        self.chunks = []
        self.embeddings = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._model

    def build(self, chunks):
        self.chunks = chunks
        model = self._get_model()
        texts = [c.text for c in chunks]
        self.embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def query(self, text: str, k: int = 4) -> List[RetrievedChunk]:
        if self.embeddings is None or not self.chunks:
            return []
        model = self._get_model()
        q_emb = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.embeddings @ q_emb
        top_idx = np.argsort(sims)[::-1][:k]
        return [
            RetrievedChunk(text=self.chunks[i].text, page=self.chunks[i].page,
                            score=float(sims[i]), index=self.chunks[i].index)
            for i in top_idx if sims[i] > 0
        ]


def build_index(chunks):
    """Pick the best available backend and build an index over the chunks."""
    if _SEMANTIC_AVAILABLE:
        idx = SemanticIndex()
    else:
        idx = TfidfIndex()
    idx.build(chunks)
    return idx


def semantic_available() -> bool:
    return _SEMANTIC_AVAILABLE
