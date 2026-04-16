from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# For development, we use a local SQLite file called sentry_logs.db
# To switch to PostgreSQL later, just change this to: 
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/sentry_db"
SQLALCHEMY_DATABASE_URL = "sqlite:///./sentry_logs.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} # Only needed for SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get the DB session in our FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()