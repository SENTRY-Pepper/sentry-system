from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from scripts.sentry_rag import SentryRAG
import models
from database import engine, get_db

# Initialize the RAG engine once when the server starts
rag = SentryRAG()

models.Base.metadata.create_all(bind=engine)
app = FastAPI()

@app.post("/ask")
def ask_sentry(request: models.QueryRequest, db: Session = Depends(get_db)):
    # 1. Use the RAG Engine to get a grounded answer
    answer, sources, score, blocked = rag.query(request.question)

    # 2. Log to Database (for your Phase 2 Analytics)
    new_log = models.InteractionLog(
        session_id=request.session_id,
        user_query=request.question,
        grounded_response=answer,
        source_citation=sources,
        similarity_score=score,
        hallucination_blocked=blocked
    )
    db.add(new_log)
    db.commit()

    return {
        "answer": answer,
        "citation": sources,
        "is_safe": not blocked
    }