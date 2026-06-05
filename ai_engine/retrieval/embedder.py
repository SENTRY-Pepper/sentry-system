from typing import List
from sentence_transformers import SentenceTransformer
from config.settings import settings


class Embedder:
    def __init__(self) -> None:
        print(f"[Embedder] Loading model: {settings.EMBEDDING_MODEL}")
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print(
            f"[Embedder] Model ready. "
            f"Output dimension: {settings.EMBEDDING_DIMENSION}"
        )

    def embed_one(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("[Embedder] Cannot embed empty or blank text.")

        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_many(self, texts: List[str]) -> List[List[float]]:
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
