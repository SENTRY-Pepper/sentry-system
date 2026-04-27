"""
SENTRY — Middleware Server Entry Point
=======================================
FastAPI application bridging Pepper/Android app (Timothy)
and the RAG AI engine (Derick).

Routers registered:
    query_routes    — /api/v1/query, /api/v1/query/baseline  (Derick)
    session_routes  — /api/v1/sessions/*                     (Timothy)
    analytics_routes— /api/v1/analytics/*                    (Timothy)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from middleware.routes.query_routes import router as query_router
from middleware.routes.session_routes import router as session_router
from middleware.routes.analytics_routes import router as analytics_router
from backend.database.connection import init_db
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    print("[SENTRY Middleware] Starting up...")
    settings.validate()

    # Initialise database tables
    await init_db()

    # Initialise RAG pipeline
    from ai_engine.rag.pipeline import RAGPipeline
    app.state.pipeline = RAGPipeline()

    print(
        f"[SENTRY Middleware] Server ready on "
        f"{settings.MIDDLEWARE_HOST}:{settings.MIDDLEWARE_PORT}"
    )

    yield

    print("[SENTRY Middleware] Shutting down.")


app = FastAPI(
    title="SENTRY Middleware API",
    description=(
        "REST API bridge between the Pepper robot and Android app "
        "(Timothy's HRI layer) and the RAG AI engine "
        "(Derick's grounding layer)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(query_router, prefix="/api/v1")
app.include_router(session_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "SENTRY Middleware",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health():
    pipeline = getattr(app.state, "pipeline", None)
    stats = {}
    if pipeline:
        stats = pipeline._retriever.get_collection_stats()
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "knowledge_base": stats,
    }