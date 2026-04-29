"""
SENTRY — Database Connection Manager
=======================================
Manages the async PostgreSQL connection using SQLAlchemy 2.0
with the asyncpg driver.

Why async?
    FastAPI is an async framework. Using an async database driver
    means database queries do not block the event loop — the server
    can handle other requests while waiting for a DB response.
    This is important during the evaluation study when Pepper may
    trigger multiple concurrent session logs.

Usage pattern across the codebase:
    from backend.database.connection import get_db

    @router.post("/sessions")
    async def create_session(db: AsyncSession = Depends(get_db)):
        # db is an active session — use it, then it auto-closes
        pass

Used by:
    backend/database/models.py      (table creation)
    middleware/routes/session_routes.py
    middleware/routes/analytics_routes.py
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from config.settings import settings

# ------------------------------------------------------------------
# SQLAlchemy async engine
# ------------------------------------------------------------------

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,  # Set True to log all SQL queries (useful for debug)
    pool_size=5,  # Number of persistent connections in the pool
    max_overflow=10,  # Extra connections allowed beyond pool_size
    pool_pre_ping=True,  # Test connections before using (handles dropped conns)
)

# Session factory — used by get_db() dependency
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    # expire_on_commit=False means we can still access object attributes
    # after a commit without triggering a lazy load error
)


# ------------------------------------------------------------------
# Base class for all ORM models
# ------------------------------------------------------------------


class Base(DeclarativeBase):
    """
    All SQLAlchemy ORM models in SENTRY inherit from this class.
    Provides the metadata registry used by Alembic for migrations.
    """

    pass


# ------------------------------------------------------------------
# FastAPI dependency — injected into route handlers
# ------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields an async database session for use in a single request.
    Automatically closes the session when the request completes,
    whether it succeeded or raised an exception.

    Usage in route handlers:
        from fastapi import Depends
        from backend.database.connection import get_db

        async def my_route(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(MyModel))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ------------------------------------------------------------------
# Table initialisation — called at server startup
# ------------------------------------------------------------------


async def init_db() -> None:
    """
    Creates all tables defined in models.py if they do not exist.
    Called once during FastAPI lifespan startup.

    This is the development approach — in production you would use
    Alembic migrations instead of create_all().
    """
    # Import models here to ensure they are registered with Base.metadata
    import backend.database.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[Database] Tables initialised successfully.")
