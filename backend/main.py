from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Siliconing the HuggingFace warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Import your existing Brain logic
from scripts.sentry_rag import SentryBrain

# Initialize FastAPI
app = FastAPI(title="SENTRY AI Backend")

# Initialize the Brain (This happens ONCE when the server starts)
print("<<..>> SENTRY is warming up its memory...")
sentry = SentryBrain()
print("✅ SENTRY is online and ready!")

# Define what a request looks like
class Query(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "online", "message": "SENTRY AI Backend is running"}

@app.post("/ask")
def ask_sentry(query: Query):
    try:
        answer = sentry.ask(query.text)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run this, use the command: uvicorn main:app --reload