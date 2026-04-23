"""
SENTRY — Embedding Model Wrapper
==================================
Wraps sentence-transformers to convert text into dense vector
representations used for semantic similarity search in ChromaDB.

The model (all-MiniLM-L6-v2) runs entirely locally — no API call,
no cost. It produces 384-dimensional vectors and is fast enough
for real-time query embedding on CPU.

Used by:
    - scripts/ingest_knowledge_base.py  (embed chunks at ingestion time)
    - ai_engine/rag/retriever.py        (embed user query at retrieval time)
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
from config.settings import settings


class Embedder:
    """
    Thin wrapper around SentenceTransformer.

    Provides:
        embed_one(text)   -> List[float]         single text -> vector
        embed_many(texts) -> List[List[float]]   batch texts -> vectors
    """

    def __init__(self) -> None:
        print(f"[Embedder] Loading model: {settings.EMBEDDING_MODEL}")
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print(
            f"[Embedder] Model ready. "
            f"Output dimension: {settings.EMBEDDING_DIMENSION}"
        )

    def embed_one(self, text: str) -> List[float]:
        """
        Embed a single string.

        Args:
            text: The string to embed (query or document chunk).

        Returns:
            A list of floats representing the dense vector.
        """
        if not text or not text.strip():
            raise ValueError("[Embedder] Cannot embed empty or blank text.")

        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of strings efficiently.

        Batching is significantly faster than calling embed_one
        in a loop because the model processes sequences in parallel.

        Args:
            texts: List of strings to embed.

        Returns:
            List of vectors in the same order as input texts.
        """
        if not texts:
            raise ValueError("[Embedder] Cannot embed an empty list.")

        # Filter out any blank entries before sending to the model
        cleaned = [t for t in texts if t and t.strip()]
        if len(cleaned) != len(texts):
            print(
                f"[Embedder] Warning: {len(texts) - len(cleaned)} "
                f"blank entries were skipped."
            )

        vectors = self._model.encode(cleaned, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        """Return the embedding output dimension."""
        return settings.EMBEDDING_DIMENSION