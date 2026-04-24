"""
SENTRY — Query Route Handlers
================================
Defines the two core API endpoints the Pepper robot calls:

    POST /api/v1/query
        Full RAG pipeline — grounded response.
        This is SENTRY's primary endpoint.

    POST /api/v1/query/baseline
        Baseline LLM only — no retrieval.
        Used during the evaluation study's control condition.

    GET /api/v1/knowledge-base/status
        Returns vector store stats — useful for diagnostics
        and confirming ingestion is current.
"""

import logging
from fastapi import APIRouter, Request, HTTPException

from middleware.validators.request_validator import (
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)

logger = logging.getLogger("sentry.middleware")
router = APIRouter()


def _pipeline_from_request(request: Request):
    """
    Retrieve the shared RAGPipeline instance from app state.
    Raises 503 if the pipeline is not initialised.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="AI pipeline is not ready. Try again in a moment.",
        )
    return pipeline


def _build_response(result: dict, scenario_id: str = None) -> QueryResponse:
    """
    Convert the pipeline result dict into a validated QueryResponse.
    Maps retrieved chunk dicts into RetrievedChunk Pydantic models.
    """
    retrieved = [
        RetrievedChunk(
            source=c["source"],
            doc_type=c["doc_type"],
            score=c["score"],
            chunk_index=c["chunk_index"],
        )
        for c in result.get("retrieved_chunks", [])
    ]

    return QueryResponse(
        query=result["query"],
        mode=result["mode"],
        response=result["response"],
        sources=result["sources"],
        retrieved_chunks=retrieved,
        chunks_used=result.get("chunks_used", 0),
        context_tokens=result.get("context_tokens", 0),
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        retrieval_ms=result["retrieval_ms"],
        generation_ms=result["generation_ms"],
        total_ms=result["total_ms"],
        model=result["model"],
        scenario_id=scenario_id,
    )


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Grounded RAG query",
    description=(
        "Submit a cybersecurity question. Returns a response grounded "
        "in verified OWASP and Kenyan legal documents. "
        "This is SENTRY's primary endpoint — the experimental condition."
    ),
    tags=["Query"],
)
async def grounded_query(body: QueryRequest, request: Request):
    pipeline = _pipeline_from_request(request)

    logger.info(
        f"[/query] scenario={body.scenario_id} "
        f"filter={body.doc_type_filter} "
        f"query='{body.query[:60]}...'"
    )

    try:
        result = pipeline.query_grounded(
            query=body.query,
            doc_type_filter=body.doc_type_filter,
        )
    except Exception as e:
        logger.error(f"[/query] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return _build_response(result, scenario_id=body.scenario_id)


@router.post(
    "/query/baseline",
    response_model=QueryResponse,
    summary="Baseline LLM query (no RAG)",
    description=(
        "Submit a cybersecurity question without retrieval grounding. "
        "Used as the control condition in the evaluation study. "
        "Responses come from GPT-4 parametric memory only."
    ),
    tags=["Query"],
)
async def baseline_query(body: QueryRequest, request: Request):
    pipeline = _pipeline_from_request(request)

    logger.info(
        f"[/query/baseline] scenario={body.scenario_id} "
        f"query='{body.query[:60]}...'"
    )

    try:
        result = pipeline.query_baseline(query=body.query)
    except Exception as e:
        logger.error(f"[/query/baseline] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return _build_response(result, scenario_id=body.scenario_id)


@router.get(
    "/knowledge-base/status",
    summary="Knowledge base status",
    description="Returns vector store statistics. Useful for diagnostics.",
    tags=["Diagnostics"],
)
async def knowledge_base_status(request: Request):
    pipeline = _pipeline_from_request(request)
    return pipeline._retriever.get_collection_stats()