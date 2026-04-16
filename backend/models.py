from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func
from database import Base

class TraineeSession(Base):
    __tablename__ = "trainee_sessions"

    session_id = Column(String, primary_key=True, index=True)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    # We can add end_time and overall_score later when Timothy integrates the frontend logic

class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    log_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, index=True) # Ties this log to the specific trainee session
    user_query = Column(String, nullable=False)
    grounded_response = Column(String, nullable=False)
    source_citation = Column(String, nullable=True)
    similarity_score = Column(Float, nullable=True) # Tracks how well ChromaDB performed
    hallucination_blocked = Column(Boolean, default=False) # True if the safety fallback was triggered
    timestamp = Column(DateTime(timezone=True), server_default=func.now())