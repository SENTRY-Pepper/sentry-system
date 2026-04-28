"""
SENTRY — Semantic Retriever
=============================
Queries ChromaDB to find the most semantically relevant document
chunks for a given user question.

This is the 'R' in RAG. The retriever:
    1. Embeds the user query using the same model used at ingestion time
    2. Performs cosine similarity search against the vector store
    3. Returns the top-k most relevant chunks with their metadata

The retrieved chunks are then passed to the prompt builder, which
injects them as grounding context before the LLM generates a response.

Used by: ai_engine/rag/pipeline.py
"""

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from typing import List, Dict, Any  # noqa: E402
from pathlib import Path  # noqa: E402

import chromadb  # noqa: E402

from config.settings import settings  # noqa: E402
from ai_engine.embeddings.embedder import Embedder  # noqa: E402

class Retriever:
    """
    Semantic retriever backed by ChromaDB.

    Connects to the persisted vector store built by
    scripts/ingest_knowledge_base.py and performs top-k
    cosine similarity retrieval for any text query.
    """

    def __init__(self) -> None:
        chroma_dir = Path(settings.CHROMA_PERSIST_DIR)

        if not chroma_dir.exists():
            raise FileNotFoundError(
                f"[Retriever] ChromaDB directory not found: {chroma_dir}\n"
                "Run scripts/ingest_knowledge_base.py first."
            )

        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = self._client.get_collection(
            name=settings.CHROMA_COLLECTION_NAME
        )
        self._embedder = Embedder()
        self._top_k = settings.RAG_TOP_K

        stored = self._collection.count()
        print(
            f"[Retriever] Connected to collection "
            f"'{settings.CHROMA_COLLECTION_NAME}' "
            f"({stored} chunks indexed)."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        doc_type_filter: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query:           The user's question or search text.
            top_k:           Number of results to return.
                             Defaults to settings.RAG_TOP_K.
            doc_type_filter: Optional — restrict results to a specific
                             document type: "owasp" or "legal".
                             None means search across all documents.

        Returns:
            List of result dicts, ranked by relevance (most relevant first):
                - "text":        The chunk content.
                - "source":      Originating document filename.
                - "doc_type":    "owasp" or "legal".
                - "chunk_index": Position within the source document.
                - "score":       Cosine similarity score (0–1, higher = more relevant).
        """
        if not query or not query.strip():
            raise ValueError("[Retriever] Query cannot be empty.")

        k = top_k if top_k is not None else self._top_k

        # Embed the query using the same model used at ingestion
        query_vector = self._embedder.embed_one(query)

        # Build optional metadata filter for ChromaDB
        where_filter = None
        if doc_type_filter:
            if doc_type_filter not in ("owasp", "legal"):
                raise ValueError(
                    f"[Retriever] Invalid doc_type_filter: '{doc_type_filter}'. "
                    "Must be 'owasp' or 'legal'."
                )
            where_filter = {"doc_type": {"$eq": doc_type_filter}}

        # Query ChromaDB
        query_kwargs = {
            "query_embeddings": [query_vector],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self._collection.query(**query_kwargs)

        # Unpack and format results
        return self._format_results(results)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return basic stats about the current vector store."""
        return {
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "total_chunks": self._collection.count(),
            "top_k_default": self._top_k,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_results(
        self, raw_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Convert ChromaDB's raw query response into clean result dicts.

        ChromaDB returns distances (lower = more similar for cosine).
        We convert to similarity scores (higher = more similar) so the
        rest of the system works with an intuitive 0–1 scale.
        """
        formatted = []

        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 = identical, 0 = opposite
            similarity_score = round(1 - (distance / 2), 4)

            formatted.append({
                "text": text,
                "source": metadata.get("source", "unknown"),
                "doc_type": metadata.get("doc_type", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
                "score": similarity_score,
            })

        return formatted