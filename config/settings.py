import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from pathlib import Path  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Go back a level from config/ to project root, where .env lives
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    # Used by: ai_engine/llm/client.py
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", 1024))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.2))

    # Used by: ai_engine/embeddings/embedder.py
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = 384

    # Used by: ai_engine/rag/retriever.py, scripts/ingest_knowledge_base.py
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "knowledge_base" / "vector_store")
    )
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "sentry_knowledge"
    )

    # RAG pipeline parameters Used by: ai_engine/rag/retriever.py, ai_engine/rag/pipeline.py
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", 5))
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", 512))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", 64))

    # Used by: middleware/main.py
    MIDDLEWARE_HOST: str = os.getenv("MIDDLEWARE_HOST", "0.0.0.0")
    MIDDLEWARE_PORT: int = int(os.getenv("MIDDLEWARE_PORT", 8000))

    # Pepper robot connection (consumed via middleware, set by Timothy)
    # Used by: middleware/routes/pepper_routes.py
    PEPPER_IP: str = os.getenv("PEPPER_IP", "")
    PEPPER_PORT: int = int(os.getenv("PEPPER_PORT", 9559))

    # Used by: scripts/ingest_knowledge_base.py
    RAW_OWASP_DIR: Path = PROJECT_ROOT / "knowledge_base" / "raw" / "owasp"
    RAW_LEGAL_DIR: Path = PROJECT_ROOT / "knowledge_base" / "raw" / "legal"
    PROCESSED_DIR: Path = PROJECT_ROOT / "knowledge_base" / "processed"

    # Used by: evaluation/metrics/
    EVAL_LOG_DIR: Path = PROJECT_ROOT / "evaluation" / "logs"
    EVAL_REPORT_DIR: Path = PROJECT_ROOT / "evaluation" / "reports"

    # Used by: backend/database/connection.py
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DATABASE_URL_SYNC: str = os.getenv("DATABASE_URL_SYNC", "")

    # Network
    LAPTOP_LOCAL_IP: str = os.getenv("LAPTOP_LOCAL_IP", "localhost")

    # Used by: ai_engine/embeddings/chunker.py
    TIKTOKEN_ENCODING: str = "cl100k_base"

    def validate(self) -> None:
        if not self.OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file.\n"
                f"Expected location: {PROJECT_ROOT / '.env'}"
            )
        if not self.DATABASE_URL:
            print(
                "[SENTRY WARNING] DATABASE_URL is not set. "
                "Session logging will not work."
            )
        if not self.PEPPER_IP:
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


# Single shared instance
settings = Settings()
