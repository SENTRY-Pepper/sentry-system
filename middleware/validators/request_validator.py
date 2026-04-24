"""
SENTRY — Request & Response Validators
========================================
Pydantic models that define the exact shape of every request
and response passing through the middleware.

Benefits:
    - FastAPI auto-validates incoming JSON against these models
    - Invalid requests are rejected before reaching the AI engine
    - Auto-generates the interactive API docs at /docs
    - Timothy's Pepper layer has a clear contract to code against
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ------------------------------------------------------------------
# Request models (Pepper → Middleware)
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    """
    Incoming query from the Pepper robot.
    Sent as a POST body to /api/v1/query or /api/v1/query/baseline.
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The user's cybersecurity question captured by Pepper.",
        examples=["What is phishing and how do I avoid it?"],
    )
    doc_type_filter: Optional[str] = Field(
        default=None,
        description=(
            "Optional filter to restrict retrieval to a document type. "
            "Accepted values: 'owasp', 'legal'. Null searches all sources."
        ),
    )
    scenario_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional scenario identifier from Timothy's dialogue system. "
            "Logged for session traceability in the evaluation study."
        ),
        examples=["phishing-01", "usb-drop-03"],
    )

    @field_validator("doc_type_filter")
    @classmethod
    def validate_doc_type(cls, v):
        if v is not None and v not in ("owasp", "legal"):
            raise ValueError(
                "doc_type_filter must be 'owasp', 'legal', or null."
            )
        return v

    @field_validator("query")
    @classmethod
    def validate_query_content(cls, v):
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be blank or whitespace only.")
        return stripped


# ------------------------------------------------------------------
# Response models (Middleware → Pepper)
# ------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """A single retrieved document chunk returned for transparency."""
    source: str
    doc_type: str
    score: float
    chunk_index: int


class QueryResponse(BaseModel):
    """
    Response returned to Pepper after processing a query.
    Contains the generated answer plus full traceability metadata
    for the evaluation study.
    """
    query: str = Field(description="The original question.")
    mode: str = Field(description="'grounded' or 'baseline'.")
    response: str = Field(description="The generated cybersecurity answer.")
    sources: List[str] = Field(
        description="Source documents used to ground the response."
    )
    retrieved_chunks: List[RetrievedChunk] = Field(
        default=[],
        description="Chunks retrieved from ChromaDB (grounded mode only).",
    )
    chunks_used: int = Field(default=0)
    context_tokens: int = Field(default=0)
    prompt_tokens: int
    completion_tokens: int
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    model: str
    scenario_id: Optional[str] = Field(
        default=None,
        description="Echo of the scenario_id from the request, if provided.",
    )


class HealthResponse(BaseModel):
    """Response shape for /health endpoint."""
    status: str
    pipeline_ready: bool
    knowledge_base: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Returned when a request fails validation or processing."""
    error: str
    detail: Optional[str] = None