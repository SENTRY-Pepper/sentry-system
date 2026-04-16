from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Import our new database files
import models
from database import engine, get_db

# Assuming your RAG logic is wrapped in a class or function here:
# from scripts.sentry_rag import SentryRAG 
# rag_engine = SentryRAG()

# Create the database tables automatically when the app starts
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="SENTRY API Middleware")

# We define what incoming data should look like
class QueryRequest(BaseModel):
    session_id: str
    question: str

@app.post("/ask")
def ask_sentry(request: QueryRequest, db: Session = Depends(get_db)):
    try:
        # 1. Run your RAG logic (Replace with your actual function call)
        # We need the answer, the citation, and the similarity score from ChromaDB
        # Example: answer, citation, score = rag_engine.ask(request.question)
        
        # --- PLACEHOLDER FOR YOUR RAG LOGIC ---
        answer = "Phishing is a fraudulent attempt to obtain sensitive information."
        citation = "[Source: Computer Misuse and Cybercrimes Act, 2018]"
        score = 0.85 
        blocked = False
        if score < 0.7:
            answer = "I do not have verified information on that."
            citation = "None"
            blocked = True
        # --------------------------------------

        # 2. Log the interaction to the Database
        new_log = models.InteractionLog(
            session_id=request.session_id,
            user_query=request.question,
            grounded_response=answer,
            source_citation=citation,
            similarity_score=score,
            hallucination_blocked=blocked
        )
        db.add(new_log)
        db.commit()
        db.refresh(new_log)

        # 3. Return the JSON to Timothy's robot code
        return {
            "status": "success",
            "answer": answer,
            "source": citation,
            "log_id": new_log.log_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))