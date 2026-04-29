"""
SENTRY — Text Chunker
======================
Splits raw document text into overlapping token-sized chunks
for embedding and storage in ChromaDB.

Why overlapping chunks?
    If a key concept sits at the boundary between two chunks,
    overlap ensures it appears fully in at least one chunk,
    preserving context for the retriever.

Used by: scripts/ingest_knowledge_base.py
"""

import re
import tiktoken
from typing import List, Dict, Any
from config.settings import settings


class Chunker:
    """
    Splits a document string into chunks of `chunk_size` tokens
    with `chunk_overlap` tokens of overlap between consecutive chunks.
    """

    def __init__(self) -> None:
        self.encoding = tiktoken.get_encoding(settings.TIKTOKEN_ENCODING)
        self.chunk_size = settings.RAG_CHUNK_SIZE
        self.chunk_overlap = settings.RAG_CHUNK_OVERLAP

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        source: str,
        doc_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Split `text` into overlapping chunks.

        Args:
            text:     The full cleaned document text.
            source:   Filename or identifier (e.g. "owasp-a01.md").
            doc_type: Category label (e.g. "owasp", "legal").

        Returns:
            List of chunk dicts, each containing:
                - "text":       The chunk string.
                - "source":     The originating document name.
                - "doc_type":   Category of document.
                - "chunk_index": Position of this chunk within the document.
                - "token_count": Number of tokens in this chunk.
        """
        cleaned = self._clean_text(text)
        tokens = self.encoding.encode(cleaned)

        if len(tokens) == 0:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append(
                {
                    "text": chunk_text,
                    "source": source,
                    "doc_type": doc_type,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_tokens),
                }
            )

            chunk_index += 1

            # If we have reached the end, stop
            if end == len(tokens):
                break

            # Slide forward by (chunk_size - overlap)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """
        Normalise raw text before chunking:
          - Collapse multiple blank lines into one.
          - Strip leading/trailing whitespace.
          - Remove null bytes that can appear in PDF extractions.
        """
        # Remove null bytes
        text = text.replace("\x00", "")
        # Collapse 3+ newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple spaces (but preserve newlines)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def token_count(self, text: str) -> int:
        """Return the token count for an arbitrary string."""
        return len(self.encoding.encode(text))
