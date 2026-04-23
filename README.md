# SENTRY — AI-Powered Cybersecurity Training System

**Jomo Kenyatta University of Agriculture and Technology**
BCT 2406 Final Year Project | 2025–2026

---

## Team

| Member | Student ID | Role |
|---|---|---|
| Derick Richard Tsumah | SCT212-0192/2022 | Generative AI Grounding & RAG Integration |
| Timothy Wachira | SCT212-0178/2021 | Human-Robot Interaction, Scenarios & Analytics |

**Supervisors:** Dr. Eunice Njeri · Prof. Waweru Mwangi

---

## Project Overview

SENTRY integrates a **Pepper humanoid robot** with a **Retrieval-Augmented Generation (RAG)**
pipeline to deliver grounded, reliable cybersecurity awareness training for Small and Medium
Enterprises (SMEs).

The system addresses a core reliability problem in conversational AI: Large Language Models
hallucinate. In a cybersecurity training context, a hallucinated explanation of a vulnerability
or an incorrect mitigation strategy can actively cause harm. SENTRY's RAG architecture
constrains every generated response to verified, curated domain knowledge — reducing
hallucinations and increasing user trust.

---

## Architecture Overview
┌─────────────────────────────────────────────────────────────┐
│                    SENTRY SYSTEM                            │
│                                                             │
│  ┌──────────────┐     ┌──────────────┐    ┌─────────────┐  │
│  │    Pepper    │────▶│  Middleware  │───▶│  AI Engine  │  │
│  │    Robot     │◀────│  (FastAPI)   │◀───│  RAG + LLM  │  │
│  └──────────────┘     └──────┬───────┘    └──────┬──────┘  │
│   Timothy (HRI)              │                   │          │
│                        ┌─────▼──────┐   ┌────────▼──────┐  │
│                        │  Backend   │   │ Knowledge Base│  │
│                        │  (DB +     │   │ ChromaDB +    │  │
│                        │ Analytics) │   │ OWASP + Legal │  │
│                        └────────────┘   └───────────────┘  │
│                         Timothy                Derick        │
└─────────────────────────────────────────────────────────────┘

---

## Repository Structure
sentry-system/
│
├── ai_engine/                  # RAG pipeline, LLM integration, embeddings
│   ├── rag/                    # Retriever, pipeline orchestration, prompt builder
│   ├── llm/                    # GPT-4 client wrapper
│   └── embeddings/             # Chunker, embedding model wrapper
│
├── knowledge_base/             # All cybersecurity source documents
│   ├── raw/
│   │   ├── owasp/              # OWASP Top 10 markdown files
│   │   └── legal/              # Computer Misuse Act PDF, Data Protection Act PDF
│   ├── processed/              # Cleaned, chunked text output (JSON)
│   └── vector_store/           # ChromaDB persisted index (gitignored)
│
├── middleware/                 # FastAPI server — bridge between Pepper and AI
│   ├── routes/                 # API route handlers
│   └── validators/             # Request validation and safety checks
│
├── pepper_interface/           # NAOqi dialogue and scenario logic (Timothy)
│   ├── scenarios/              # Scenario definitions per attack type
│   └── dialogue/               # Dialogue state machines
│
├── backend/                    # Session logging, analytics, dashboard (Timothy)
│   ├── database/               # DB models
│   ├── analytics/              # Metrics computation
│   └── dashboard/              # Reporting interface
│
├── evaluation/                 # Experiment tracking (Derick)
│   ├── metrics/                # Hallucination rate, grounding accuracy scripts
│   ├── logs/                   # Interaction logs from user study (gitignored)
│   └── reports/                # Generated evaluation reports
│
├── tests/
│   ├── unit/                   # Unit tests per module
│   └── integration/            # End-to-end middleware + AI tests
│
├── docs/
│   ├── architecture/           # Architecture diagrams and design docs
│   └── api_specs/              # FastAPI endpoint documentation
│
├── config/
│   └── settings.py             # Central typed configuration (reads .env)
│
├── scripts/
│   └── ingest_knowledge_base.py  # One-time ingestion: raw docs → ChromaDB
│
├── .env                        # Your secrets — NEVER committed
├── .env.example                # Template committed to repo
├── .gitignore
├── requirements.txt
└── README.md

---

## Technology Stack

### Derick's Layer (AI / RAG)

| Tool | Version | Purpose |
|---|---|---|
| `openai` | 1.30.1 | GPT-4 API client for response generation |
| `chromadb` | 0.5.0 | Vector database for semantic document retrieval |
| `sentence-transformers` | 2.7.0 | Local embedding model (`all-MiniLM-L6-v2`) |
| `pdfplumber` | 0.11.0 | PDF text extraction for legal documents |
| `tiktoken` | 0.7.0 | Token counting for chunk sizing and cost control |
| `fastapi` | 0.111.0 | Middleware REST API server |
| `uvicorn` | 0.30.0 | ASGI server for FastAPI |
| `pydantic` | 2.7.3 | Request/response data validation |
| `python-dotenv` | 1.0.1 | Environment variable loading |
| `numpy` | 1.26.4 | Vector math support |
| `pandas` | 2.2.2 | Evaluation data handling and reporting |
| `pytest` | 8.2.2 | Testing framework |
| `httpx` | 0.27.0 | Async HTTP client for integration tests |

### Timothy's Layer (HRI / Backend)

| Tool | Purpose |
|---|---|
| NAOqi SDK (Python 2.7) | Pepper robot control |
| Flask | Backend REST API |
| PostgreSQL | Session and analytics database |
| Chart.js | Dashboard visualisations |

---

## Setup Instructions

### Prerequisites
- Python 3.11.x (strictly — 3.12+ breaks several dependencies)
- Git
- OpenAI API key with billing enabled

### 1. Clone the repository

```bash
git clone https://github.com/SENTRY-Pepper/sentry-system.git
cd sentry-system
```

### 2. Create and activate virtual environment

```bash
# Windows
"C:\Program Files\Python311\python.exe" -m venv venv
venv\Scripts\activate

# Confirm version
python --version   # Must print Python 3.11.x
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment

```bash
copy .env.example .env
# Open .env and fill in OPENAI_API_KEY
```

### 5. Ingest the knowledge base

This only needs to run once (or when source documents change):

```bash
python scripts/ingest_knowledge_base.py
```

### 6. Start the middleware server

```bash
uvicorn middleware.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Verify the API is running

Open your browser at `http://localhost:8000/docs` — FastAPI's auto-generated
interactive documentation.

---

## Knowledge Base Sources

| Document | Source | Coverage |
|---|---|---|
| OWASP Top 10 | OWASP GitHub (official .md files) | Web application vulnerabilities |
| Computer Misuse and Cybercrimes Act 2018 | Kenya Law | Kenyan cybercrime legislation |
| Data Protection Act 2019 | Kenya Law | Data privacy obligations |

---

## Evaluation Design (Phase 5)

The study uses a **between-subjects randomised controlled design**:

- **Control condition:** LLM without retrieval grounding (baseline GPT-4)
- **Experimental condition:** RAG-enhanced system (SENTRY full pipeline)
- **Participants:** ~40 individuals (20 per group)

**Metrics measured:**

| Metric | Type | Tool |
|---|---|---|
| Hallucination rate | Quantitative | Expert annotation + `evaluation/metrics/` |
| Grounding accuracy | Quantitative | Claim traceability to retrieved chunks |
| Perceived trust | Subjective | Post-session questionnaire |
| Response latency | Quantitative | Middleware request timing |
| Engagement metrics | Quantitative | Session logs |

---

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# With output
pytest -v
```

---

## Git Workflow
main        — stable, supervisor-presentable at all times
dev         — active development branch
feature/*   — individual feature branches (merged into dev via PR)

```bash
# Start a new feature
git checkout dev
git checkout -b feature/rag-retriever

# After work is done
git add .
git commit -m "feat(rag): implement top-k cosine similarity retriever"
git push origin feature/rag-retriever
# Open a Pull Request into dev on GitHub
```

---

## API Endpoints (Middleware)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health check |
| `POST` | `/query` | Submit a user query — returns grounded response |
| `POST` | `/query/baseline` | Submit query to baseline LLM (no RAG) — for evaluation |
| `GET` | `/knowledge-base/status` | Check vector store document count |

Full interactive docs available at `/docs` when server is running.

---

## License

Academic project — Jomo Kenyatta University of Agriculture and Technology, 2026.