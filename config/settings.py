"""
SENTRY — Central Configuration
================================
Loads all environment variables and exposes a typed settings object
used across the entire application: AI engine, middleware, evaluation,
and knowledge base ingestion.

Every module imports `settings` from here — nothing reads .env directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root (one level above this file) and load .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    # ------------------------------------------------------------------
    # LLM — OpenAI GPT-4
    # Used by: ai_engine/llm/client.py
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 1024))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.2))
    # Low temperature = more deterministic, factual responses (critical for RAG)

    # ------------------------------------------------------------------
    # Embeddings — sentence-transformers (local, free, no API call)
    # Used by: ai_engine/embeddings/embedder.py
    # ------------------------------------------------------------------
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = 384  # Fixed dimension for all-MiniLM-L6-v2

    # ------------------------------------------------------------------
    # ChromaDB — Vector store for semantic retrieval
    # Used by: ai_engine/rag/retriever.py, scripts/ingest_knowledge_base.py
    # ------------------------------------------------------------------
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR",
        str(PROJECT_ROOT / "knowledge_base" / "vector_store")
    )
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "sentry_knowledge"
    )

    # ------------------------------------------------------------------
    # RAG pipeline parameters
    # Used by: ai_engine/rag/retriever.py, ai_engine/rag/pipeline.py
    # ------------------------------------------------------------------
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", 5))
    # Number of document chunks retrieved per query
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", 512))
    # Max tokens per document chunk during ingestion
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", 64))
    # Token overlap between consecutive chunks to preserve context

    # ------------------------------------------------------------------
    # FastAPI Middleware Server
    # Used by: middleware/main.py
    # ------------------------------------------------------------------
    MIDDLEWARE_HOST: str = os.getenv("MIDDLEWARE_HOST", "0.0.0.0")
    MIDDLEWARE_PORT: int = int(os.getenv("MIDDLEWARE_PORT", 8000))

    # ------------------------------------------------------------------
    # Pepper robot connection (consumed via middleware, set by Timothy)
    # Used by: middleware/routes/pepper_routes.py
    # ------------------------------------------------------------------
    PEPPER_IP: str = os.getenv("PEPPER_IP", "")
    PEPPER_PORT: int = int(os.getenv("PEPPER_PORT", 9559))

    # ------------------------------------------------------------------
    # Knowledge base — file system paths
    # Used by: scripts/ingest_knowledge_base.py
    # ------------------------------------------------------------------
    RAW_OWASP_DIR: Path = PROJECT_ROOT / "knowledge_base" / "raw" / "owasp"
    RAW_LEGAL_DIR: Path = PROJECT_ROOT / "knowledge_base" / "raw" / "legal"
    PROCESSED_DIR: Path = PROJECT_ROOT / "knowledge_base" / "processed"

    # ------------------------------------------------------------------
    # Evaluation
    # Used by: evaluation/metrics/
    # ------------------------------------------------------------------
    EVAL_LOG_DIR: Path = PROJECT_ROOT / "evaluation" / "logs"
    EVAL_REPORT_DIR: Path = PROJECT_ROOT / "evaluation" / "reports"

    # ------------------------------------------------------------------
    # Tokenizer (tiktoken) — for chunk sizing and prompt cost estimation
    # Used by: ai_engine/embeddings/chunker.py
    # ------------------------------------------------------------------
    TIKTOKEN_ENCODING: str = "cl100k_base"
    # cl100k_base is the encoding used by GPT-4

    def validate(self) -> None:
        """
        Called at application startup to catch missing secrets early.
        Raises EnvironmentError with a clear message rather than failing
        silently deep inside an API call.
        """
        if not self.OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file.\n"
                f"Expected location: {PROJECT_ROOT / '.env'}"
            )
        if not self.PEPPER_IP:
            # Non-fatal — warn only, Pepper may not be connected during dev
            print(
                "[SENTRY WARNING] PEPPER_IP is not set. "
                "Middleware will run but cannot reach the robot."
            )

    def __repr__(self) -> str:
        return (
            f"Settings(model={self.LLM_MODEL}, "
            f"embedding={self.EMBEDDING_MODEL}, "
            f"top_k={self.RAG_TOP_K}, "
            f"chunk_size={self.RAG_CHUNK_SIZE})"
        )


# Single shared instance — import this everywhere
settings = Settings()