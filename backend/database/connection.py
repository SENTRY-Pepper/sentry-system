from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from config.settings import settings

# SQLAlchemy async engine

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# Session factory — used by get_db() dependency
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# Base class for all ORM models


class Base(DeclarativeBase):
    """
    All SQLAlchemy ORM models in SENTRY inherit from this class.
    Provides the metadata registry used by Alembic for migrations.
    """

    pass


# FastAPI dependency — injected into route handlers


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Table initialisation — called at server startup


async def init_db() -> None:
    # Import models here to ensure they are registered with Base.metadata
    import backend.database.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("[Database] Tables initialised successfully.")
