import re
import tiktoken
from typing import List, Dict, Any
from config.settings import settings


class Chunker:
    def __init__(self) -> None:
        self.encoding = tiktoken.get_encoding(settings.TIKTOKEN_ENCODING)
        self.chunk_size = settings.RAG_CHUNK_SIZE
        self.chunk_overlap = settings.RAG_CHUNK_OVERLAP

    # Public API
    def chunk_text(
        self,
        text: str,
        source: str,
        doc_type: str,
    ) -> List[Dict[str, Any]]:
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

            # Slide forward
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _clean_text(self, text: str) -> str:
        text = text.replace("\x00", "")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def token_count(self, text: str) -> int:
        """Return the token count for an arbitrary string."""
        return len(self.encoding.encode(text))
