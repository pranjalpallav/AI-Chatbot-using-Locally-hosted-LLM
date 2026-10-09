"""Embedding + FAISS vector store."""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # keep the GPU free for the LLM
    return _model


def embed(texts: list[str]) -> np.ndarray:
    return get_model().encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
    ).astype("float32")


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks: list[dict] = []

    def is_empty(self) -> bool:
        return self.index is None or not self.chunks

    def add(self, chunks: list[dict], vecs: np.ndarray) -> None:
        if not chunks:
            return
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])   # cosine similarity (vectors are normalised)
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.is_empty():
            return []
        scores, ids = self.index.search(embed([query]), min(k, len(self.chunks)))
        return [{**self.chunks[i], "score": float(s)}
                for s, i in zip(scores[0], ids[0]) if i != -1]
