# SENTRY — AI-Powered Cybersecurity Training System

**Jomo Kenyatta University of Agriculture and Technology**
BCT 2406 Final Year Project | 2025–2026

![CI](https://github.com/SENTRY-Pepper/sentry-system/actions/workflows/ci.yml/badge.svg)

---

## Team

| Member | Student ID | Role |
|---|---|---|
| Derick Richard Tsumah | SCT212-0192/2022 | Generative AI Grounding & RAG Integration |
| Timothy Wachira | SCT212-0178/2021 | Human-Robot Interaction, Scenarios & Analytics |

**Supervisors:** Dr. Eunice Njeri · Dr. Richard Rimiru

---

## Project Overview

SENTRY integrates a **Pepper humanoid robot** with a **Retrieval-Augmented Generation (RAG)**
pipeline to deliver grounded, reliable cybersecurity awareness training for Small and Medium
Enterprises (SMEs) in Kenya.

The core research question: can RAG-based response grounding measurably reduce hallucinations
and increase user trust in an embodied AI tutoring system?

The system addresses a fundamental reliability problem — Large Language Models hallucinate.
In a cybersecurity training context, a hallucinated explanation of a vulnerability or an
incorrect mitigation strategy can cause direct harm. SENTRY's RAG architecture constrains
every generated response to verified, curated domain knowledge, reducing hallucinations and
increasing response traceability.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Employee + Android Tablet App                  │
│         Kotlin · Retrofit · Jetpack Compose · ADB          │
└────────────────────┬──────────────────────┬────────────────┘
                     │ HTTP                 │ HTTP
          ┌──────────▼──────────┐           │
          │   Pepper Robot      │           │
          │   NAOqi Python 2.7  │           │
          │   Speech · Gestures │           │
          └──────────┬──────────┘           │
                     │ HTTP                 │
┌────────────────────▼──────────────────────▼────────────────┐
│              FastAPI Middleware  (port 8000)                │
│   /query  /query/baseline  /sessions/*  /analytics/*       │
└──────────────┬─────────────────────────┬───────────────────┘
               │                         │
   ┌───────────▼──────────┐  ┌───────────▼──────────┐
   │   RAG + AI Engine    │  │  Backend + Database   │
   │  Retriever · GPT-4   │  │  PostgreSQL 17        │
   │  HallucinationScorer │  │  SQLAlchemy async     │
   └───────────┬──────────┘  └──────────────────────┘
               │
   ┌───────────▼──────────┐
   │  ChromaDB            │
   │  175 chunks indexed  │
   │  OWASP + Legal docs  │
   └──────────────────────┘
```

---

## Repository Structure

```
sentry-system/
│
├── .github/workflows/          # CI/CD pipelines
│   ├── ci.yml                  # Unit tests + lint on push/PR
│   ├── integration.yml         # Integration tests on PR to main
│   └── dependency-review.yml   # Supply chain vulnerability check
│
├── ai_engine/                  # RAG pipeline + LLM (Derick)
│   ├── rag/
│   │   ├── pipeline.py         # End-to-end orchestrator
│   │   ├── retriever.py        # ChromaDB semantic search
│   │   └── prompt_builder.py   # Grounded prompt assembly
│   ├── llm/
│   │   └── client.py           # GPT-4 baseline + grounded modes
│   └── embeddings/
│       ├── chunker.py          # Token-aware text splitter
│       └── embedder.py         # sentence-transformers wrapper
│
├── knowledge_base/             # Cybersecurity source documents (Derick)
│   ├── raw/
│   │   ├── owasp/              # OWASP Top 10 2025 markdown files (16)
│   │   └── legal/              # Computer Misuse Act + Data Protection Act PDFs
│   ├── processed/              # Ingestion report JSON
│   └── vector_store/           # ChromaDB index (gitignored — regenerate via script)
│
├── middleware/                 # FastAPI server — shared layer
│   ├── main.py                 # App entry point + lifespan startup
│   ├── routes/
│   │   ├── query_routes.py     # /query + /query/baseline (Derick)
│   │   ├── session_routes.py   # /sessions/* (Timothy)
│   │   └── analytics_routes.py # /analytics/* (Timothy)
│   └── validators/
│       ├── request_validator.py
│       └── session_validator.py
│
├── backend/                    # Database layer (Timothy)
│   └── database/
│       ├── connection.py       # Async PostgreSQL connection
│       └── models.py           # ORM table definitions
│
├── pepper_interface/           # NAOqi HRI layer (Timothy)
│   ├── scenarios/              # Phishing, USB, password, network, social engineering
│   ├── dialogue/               # State machines + dialogue manager
│   ├── middleware_client.py    # HTTP client calling FastAPI
│   └── TIMOTHY_GUIDE.md        # Full integration guide for Timothy
│
├── mobile_app/                 # Android app for Pepper's tablet (Timothy)
│   └── README_MOBILE.md        # Android Studio setup + ADB deployment guide
│
├── evaluation/                 # Research evaluation module (Derick)
│   ├── metrics/
│   │   ├── hallucination_scorer.py  # N-gram overlap grounding metric
│   │   └── grounding_scorer.py      # Session logger + DataFrame export
│   ├── run_evaluation.py        # Per-participant session runner
│   ├── analyse_results.py       # Aggregate statistical analysis (t-test, Cohen's d)
│   ├── logs/                    # Session JSON logs (gitignored)
│   └── reports/                 # CSV + JSON evaluation reports (gitignored)
│
├── tests/
│   ├── unit/                   # Per-module unit tests
│   └── integration/            # End-to-end endpoint tests
│
├── docs/
│   ├── architecture/           # System architecture diagram (SVG)
│   └── api_specs/              # OpenAPI JSON spec (export via curl)
│
├── scripts/
│   └── ingest_knowledge_base.py  # Populates ChromaDB from raw documents
│
├── config/
│   └── settings.py             # Central typed configuration
│
├── conftest.py                 # Pytest path configuration
├── .env.example                # Environment variable template
├── requirements.txt            # Pinned Python dependencies
└── README.md                   # This file
```

---

## Technology Stack

### Derick's Layer — RAG / AI Engine

| Package | Version | Purpose |
|---|---|---|
| `openai` | 1.30.1 | GPT-4 API — response generation |
| `chromadb` | 0.5.0 | Vector database — semantic retrieval |
| `sentence-transformers` | 2.7.0 | Local embedding model (`all-MiniLM-L6-v2`) |
| `pdfplumber` | 0.11.0 | PDF text extraction for legal documents |
| `tiktoken` | 0.7.0 | Token counting for chunk sizing |
| `fastapi` | 0.111.0 | Middleware REST API server |
| `uvicorn` | 0.30.0 | ASGI server |
| `pydantic` | 2.7.3 | Request/response validation |
| `python-dotenv` | 1.0.1 | Environment variable loading |
| `numpy` | 1.26.4 | Vector math |
| `pandas` | 2.2.2 | Evaluation data handling |
| `scipy` | latest | t-test and statistical analysis |
| `pytest` | 8.2.2 | Testing framework |
| `httpx` | 0.27.0 | Async HTTP client for tests |

### Timothy's Layer — HRI / Backend

| Tool | Purpose |
|---|---|
| NAOqi SDK (Python 2.7) | Pepper robot control |
| Android Studio + Kotlin | Tablet app development |
| Retrofit | Android HTTP client |
| PostgreSQL 17 | Session and analytics database |
| SQLAlchemy 2.0 (async) | ORM |

---

## Setup

### Prerequisites

- Python 3.11.x (strictly — 3.12+ breaks several dependencies)
- PostgreSQL 17
- Git
- OpenAI API key with billing enabled

### 1. Clone

```bash
git clone https://github.com/SENTRY-Pepper/sentry-system.git
cd sentry-system
```

### 2. Virtual environment

```bash
"C:\Program Files\Python311\python.exe" -m venv venv
venv\Scripts\activate
python --version   # Must print Python 3.11.x
```

### 3. Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Environment

```bash
copy .env.example .env
# Fill in OPENAI_API_KEY, DATABASE_URL, LAPTOP_LOCAL_IP
```

### 5. Database

```bash
psql -U postgres -c "CREATE DATABASE sentry_db;"
```

### 6. Knowledge base ingestion

Run once — or whenever source documents change:

```bash
python scripts/ingest_knowledge_base.py
```

Expected output: 175 chunks across 18 documents stored in ChromaDB.

### 7. Start the middleware server

```bash
uvicorn middleware.main:app --reload --host 0.0.0.0 --port 8000
```

### 8. Verify

Open `http://localhost:8000/docs` — FastAPI interactive documentation.

```bash
curl http://localhost:8000/health
# {"status":"healthy","pipeline_ready":true,"knowledge_base":{...}}
```

---

## Knowledge Base

| Document | Source | Type |
|---|---|---|
| OWASP Top 10 2025 (16 files) | OWASP GitHub | Cybersecurity |
| Computer Misuse and Cybercrimes Act 2018 | Kenya Law | Legal |
| Data Protection Act 2019 | Kenya Law | Legal |

Total: 175 chunks · 512 tokens each · 64-token overlap · `all-MiniLM-L6-v2` embeddings

---

## API Endpoints

| Method | Endpoint | Description | Owner |
|---|---|---|---|
| `GET` | `/health` | Server health + pipeline status | Shared |
| `GET` | `/` | Service identification | Shared |
| `POST` | `/api/v1/query` | Grounded RAG response | Derick |
| `POST` | `/api/v1/query/baseline` | Baseline LLM (no RAG) | Derick |
| `GET` | `/api/v1/knowledge-base/status` | Vector store stats | Derick |
| `POST` | `/api/v1/sessions/start` | Begin training session | Timothy |
| `POST` | `/api/v1/sessions/end` | Close session + compute gain | Timothy |
| `POST` | `/api/v1/sessions/interaction` | Log scenario interaction | Timothy |
| `POST` | `/api/v1/sessions/eval-log` | Log evaluation record | Timothy |
| `GET` | `/api/v1/sessions/{id}` | Get session summary | Timothy |
| `GET` | `/api/v1/analytics/study` | Full study aggregate | Timothy |
| `GET` | `/api/v1/analytics/sessions` | List all sessions | Timothy |
| `GET` | `/api/v1/analytics/organisation/{id}` | Org-level analytics | Timothy |

Full interactive docs: `http://localhost:8000/docs`
OpenAPI spec: `docs/api_specs/sentry_api_spec.json`

---

## Evaluation Study

**Design:** Between-subjects randomised controlled trial

| Condition | Description | n |
|---|---|---|
| Experimental (grounded) | Full RAG pipeline | 20 |
| Control (baseline) | LLM only, no retrieval | 20 |

**Metrics:**

| Metric | Type | Method |
|---|---|---|
| Grounding accuracy | Quantitative | N-gram overlap (automated) + expert annotation |
| Hallucination rate | Quantitative | 1 − grounding accuracy |
| Response latency | Quantitative | Middleware timing |
| Knowledge gain | Quantitative | Post-score − pre-score |
| Perceived trust | Subjective | Post-session questionnaire |

**Running the study:**

```bash
# Per participant
python evaluation/run_evaluation.py \
    --participant P001 \
    --condition grounded \
    --organisation JKUAT_PILOT \
    --pre-score 45.0 \
    --post-score 72.0

# After all sessions complete
python evaluation/analyse_results.py
```

**Pilot results (n=1, grounded condition):**

| Query | Grounding accuracy (RAG) | Hallucination rate (baseline) |
|---|---|---|
| Phishing | 0.125 | 1.0 |
| Legal penalties | 0.625 | 1.0 |
| SQL injection | 0.615 | 1.0 |
| USB drive | 0.0* | 1.0 |
| Social engineering | 0.0* | 1.0 |
| **Mean** | **0.273** | **1.0** |

*Automated scorer underestimates grounding when LLM paraphrases retrieved content.
Retrieval confirmed working (top score 0.6939). Expert annotation is the primary metric.

---

## Running Tests

```bash
# All unit tests
pytest tests/unit/ -v

# Integration tests (requires server running)
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_pipeline.py -v
```

---

## Git Workflow

```
main    — stable, supervisor-presentable
dev     — active development
feature/* — individual features, merged into dev via PR
```

```bash
git checkout dev
git checkout -b feature/your-feature-name
# ... work ...
git add .
git commit -m "feat(module): description"
git push origin feature/your-feature-name
# Open PR into dev on GitHub
```

---

## Network Configuration (for Pepper + Android app)

Both Pepper's tablet and the laptop must be on the same WiFi network.

```bash
# Find your laptop's local IP
ipconfig   # Look for: Wireless LAN adapter Wi-Fi → IPv4 Address

# Allow port 8000 through Windows Firewall (run once as Administrator)
New-NetFirewallRule -DisplayName "SENTRY Middleware" `
    -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

Update `LAPTOP_LOCAL_IP` in `.env` and `BASE_URL` in the Android app's `ApiClient.kt`.

---

## Supervisors

- Dr. Eunice Njeri
- Dr. Richard Rimiru

Jomo Kenyatta University of Agriculture and Technology
BCT 2406 | 2025–2026