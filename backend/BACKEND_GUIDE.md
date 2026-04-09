# SENTRY Backend: AI & RAG Engine

## Overview
This backend handles the "Intelligence" of SENTRY. It uses a Retrieval-Augmented Generation (RAG) pipeline to ensure Pepper's advice is grounded in Kenyan Law and OWASP standards.

## Tech Stack
- **Database:** ChromaDB (Vector Store)
- **Embeddings:** all-MiniLM-L6-v2 (Sentence-Transformers)
- **LLM:** OpenAI GPT-4o-mini
- **API Framework:** FastAPI (Pending)

## Integration for HRI (Timothy)
To get a response from SENTRY, the Robot scripts must send an HTTP POST request to the `/ask` endpoint:
- **Input:** JSON object `{"question": "string"}`
- **Output:** JSON object `{"answer": "string", "source": "string"}`

## Current Status
- [x] OWASP Ingestion
- [x] Kenyan Law Ingestion (CMCA 2018 & DPA 2019)
- [x] RAG Retrieval Logic
- [ ] Live API Endpoint (FastAPI)