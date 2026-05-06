"""
In-memory vector store with OpenAI embeddings and confidence scoring.

Swap for Pinecone / ChromaDB / pgvector in production.
"""

import json
import numpy as np
from pathlib import Path
from openai import OpenAI

from config.settings import OPENAI_API_KEY, EMBEDDING_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenAI."""
    resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class VectorStore:
    """Simple vector store with cosine-similarity search."""

    def __init__(self):
        self.documents: list[dict] = []
        self.embeddings: list[list[float]] = []

    def load_from_json(self, path: Path) -> None:
        """Ingest documents from a JSON file."""
        with open(path, "r") as f:
            docs = json.load(f)

        texts = [
            f"{d.get('question', d.get('title', ''))}\n{d['answer'] if 'answer' in d else d.get('content', '')}"
            for d in docs
        ]
        self.documents.extend(docs)
        self.embeddings.extend(_embed(texts))

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top-k results with relevance scores."""
        if not self.documents:
            return []

        q_emb = _embed([query])[0]
        scored = [
            (doc, _cosine_sim(q_emb, emb))
            for doc, emb in zip(self.documents, self.embeddings)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in scored[:top_k]:
            results.append({
                "id": doc["id"],
                "category": doc["category"],
                "question": doc.get("question", doc.get("title", "")),
                "answer": doc.get("answer", doc.get("content", "")),
                "relevance_score": round(score, 4),
            })
        return results


# ── Singleton stores (loaded once) ─────────────────────────────────
_it_store: VectorStore | None = None
_hr_store: VectorStore | None = None


def get_it_store() -> VectorStore:
    global _it_store
    if _it_store is None:
        from config.settings import IT_KB_PATH
        _it_store = VectorStore()
        _it_store.load_from_json(IT_KB_PATH)
    return _it_store


def get_hr_store() -> VectorStore:
    global _hr_store
    if _hr_store is None:
        from config.settings import HR_KB_PATH
        _hr_store = VectorStore()
        _hr_store.load_from_json(HR_KB_PATH)
    return _hr_store
