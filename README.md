# SENTRY

SENTRY is an AI-powered cybersecurity training research prototype that combines
a FastAPI middleware, a custom Retrieval-Augmented Generation (RAG) pipeline,
PostgreSQL analytics storage, an Android trainee/admin app, and a Pepper robot
interface.

The project investigates whether RAG-grounded responses can reduce
hallucination and improve factual reliability in an embodied cybersecurity
training system.

## Project Context

- Institution: Jomo Kenyatta University of Agriculture and Technology
- Course: BCT 2406 Final Year Project
- Academic year: 2025-2026
- Repository focus: research prototype, onboarding-grade implementation, and
  incremental path toward a production cybersecurity learning platform

## Team

| Member | Student ID | Focus |
|---|---|---|
| Derick Richard Tsumah | SCT212-0192/2022 | RAG pipeline, AI grounding, evaluation framework |
| Timothy Wachira | SCT212-0178/2021 | HRI layer, scenarios, analytics, Android app |

Supervisors: Dr. Eunice Njeri, Dr. Richard Rimiru

## Current Status

SENTRY currently supports:

- FastAPI middleware on port 8000.
- Custom RAG pipeline over OWASP and Kenyan legal documents.
- PostgreSQL-backed training sessions, interactions, assessments, and
  evaluation logs.
- Android Jetpack Compose app with trainee and admin flows.
- Admin and Manager role split: Admin for research analytics, Manager for
  organisation training operations.
- Backend Organisation/User model with trainee account management and
  organisation analytics.
- Android OWASP Top 10 curriculum with fixed A-D assessments, local saved
  feedback, and trainee progress tracking.
- Backend-backed Android authentication with role and organisation persistence.
- Organisation analytics keyed by canonical organisation ID strings.
- Pepper robot interface with scenario orchestration and simulation support.
- Evaluation utilities for grounded-vs-baseline research comparison.

SENTRY is not yet a production multi-tenant platform. Prototype user and
organisation tables now exist, Android login uses the backend login endpoint,
and Admin/Manager analytics routes enforce roles. Signed/expiring tokens,
strong password hashing, and detailed cross-device per-question learning
history remain future work.

## Architecture

```text
Android App
  Jetpack Compose, Hilt, Ktor CIO, DataStore
  trainee/admin screens, chat, OWASP curriculum, session flow, analytics

FastAPI Middleware
  /api/v1/query
  /api/v1/query/baseline
  /api/v1/sessions/*
  /api/v1/analytics/*
  /api/v1/manager/*
  /api/v1/users/login

RAG Engine
  ingestion -> retrieval -> prompt building -> OpenAI generation
  ChromaDB vector store over OWASP and Kenyan legal sources

PostgreSQL
  organisations
  users
  training_sessions
  scenario_interactions
  assessment_results
  evaluation_logs

Pepper Interface
  NAOqi-compatible client, dialogue state machine, scenario classes
```

## Repository Layout

```text
config/                  Runtime settings via pydantic-settings
ai_engine/               RAG, generation, retrieval, ingestion modules
backend/database/        SQLAlchemy async engine and ORM models
middleware/              FastAPI app, routes, Pydantic validators
evaluation/              Research metrics, session runner, statistical analysis
knowledge_base/          Raw OWASP/legal sources and ingestion report
pepper_interface/        Pepper client, middleware client, dialogue, scenarios
scripts/                 CLI wrappers, mainly knowledge-base ingestion
tests/                   Python unit and integration tests
mobile_app/              Android app, Gradle project, Compose UI
docs/                    OpenAPI export and architecture asset
SENTRY_skills/           Project onboarding and workflow documentation
```

Generated or local-only folders are intentionally ignored and may be absent:

- `knowledge_base/vector_store/`
- `evaluation/logs/`
- `evaluation/reports/`
- Python caches
- Gradle and Android build output
- Android Studio local state

## Backend And RAG

The Python backend is centered on `middleware/main.py`. During startup it:

1. Validates settings.
2. Initializes database tables.
3. Creates a shared `RAGPipeline`.
4. Registers query, session, and analytics routers.

Canonical RAG modules:

- `ai_engine.ingestion`: document loading and chunking.
- `ai_engine.retrieval`: embedding and ChromaDB retrieval.
- `ai_engine.generation`: prompt construction and OpenAI calls.
- `ai_engine.rag.pipeline`: grounded and baseline orchestration.

Compatibility shims remain under `ai_engine.embeddings`, `ai_engine.llm`, and
selected `ai_engine.rag` modules for older imports.

## Android App

The Android app lives in `mobile_app/` and currently uses:

- Kotlin
- Jetpack Compose
- Hilt
- Ktor CIO
- Kotlin serialization
- DataStore preferences
- Timber
- Min SDK 23 for Pepper tablet compatibility

Current source root:

```text
mobile_app/app/src/main/java/com/sentry/app/
```

Important packages:

- `core/navigation`: routes, `UserRole`, nav graph, single-top helper.
- `core/network`: Ktor client and network result handling.
- `core/organisation`: canonical organisation ID normalization.
- `data/local`: DataStore-backed session persistence.
- `data/remote/api`: Ktor endpoint extension functions.
- `data/repository`: auth, session, query, and analytics repositories.
- `features/trainee/curriculum`: OWASP Top 10 modules, saved answer feedback,
  and local progress persistence.
- `features`: splash, auth, trainee, admin, chat, settings screens.

The current app does not use Retrofit or OkHttp interceptors.

### OWASP Training Flow

The trainee session flow is now based on the OWASP Top 10:

- Ten modules map to A01 through A10.
- Every module includes educational content, a workplace scenario, four A-D
  answers, one objectively correct answer, and practical feedback.
- Feedback is saved locally in the app for both correct and incorrect answers.
  The app does not call the RAG API for these fixed assessment explanations.
- The session still logs interactions and completes the session through the
  backend so organisation analytics use real completion and score data.
- Open-ended grounded RAG remains available through the chat screen.

## Authentication, Roles, And Organisation Flow

Authentication is backend-backed but still prototype-grade:

- `AuthRepository.login()` validates basic form input.
- Android calls `POST /api/v1/users/login`.
- The returned bearer token is stored in DataStore.
- Roles are `trainee`, `manager`, and `admin`.
- Manager/Admin login requires an organisation value.
- Admin/Manager analytics routes validate bearer tokens and roles.

The backend now stores organisations and users:

- Admin: research-wide grounded-vs-baseline analytics.
- Manager: organisation trainee management and performance analytics.
- Trainee: learning sessions and personal progress.

Android normalizes organisation names before analytics calls:

```text
Sentry Study -> SENTRY_STUDY
Heritage Insurance -> HERITAGE_INSURANCE
jkuat-pilot -> JKUAT_PILOT
```

The admin analytics endpoint expects the canonical ID:

```text
GET /api/v1/analytics/organisation/{organisation_id}
```

## API Endpoints

```text
GET  /health
GET  /api/v1/knowledge-base/status
POST /api/v1/query
POST /api/v1/query/baseline
POST /api/v1/sessions/start
POST /api/v1/sessions/end
GET  /api/v1/sessions/{session_id}
POST /api/v1/sessions/interaction
POST /api/v1/sessions/eval-log
GET  /api/v1/analytics/study
GET  /api/v1/analytics/sessions
GET  /api/v1/analytics/organisation/{organisation_id}
POST /api/v1/organisations
POST /api/v1/users/login
GET  /api/v1/manager/trainees
POST /api/v1/manager/trainees
PATCH /api/v1/manager/trainees/{user_id}/deactivate
GET  /api/v1/manager/analytics/overview
GET  /api/v1/manager/analytics/weaknesses
```

Interactive docs are available at:

```text
http://localhost:8000/docs
```

## Setup

### Prerequisites

- Python 3.11
- PostgreSQL
- Git
- OpenAI API key for live RAG generation
- Android Studio / Android SDK for mobile development
- Java/Gradle environment suitable for the Android project

### Python Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Then fill in database and OpenAI settings.

### Database

Create the configured PostgreSQL database before starting middleware. The app
currently creates tables from SQLAlchemy metadata at startup. There is no active
Alembic migration workflow.

### Knowledge Base

The ChromaDB vector store is generated and gitignored. Regenerate it before
running middleware or live RAG tests:

```powershell
.\venv\Scripts\python.exe scripts\ingest_knowledge_base.py
```

### Start Middleware

```powershell
uvicorn middleware.main:app --reload --host 0.0.0.0 --port 8000
```

Verify:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## Android Development

From `mobile_app/`:

```powershell
.\gradlew.bat :app:testDebugUnitTest
.\gradlew.bat :app:assembleDebug
```

For emulator-to-localhost middleware calls, use:

```text
http://10.0.2.2:8000
```

For a real device or Pepper tablet, use the laptop's LAN IP and ensure port
8000 is reachable on the same network.

Android verification may need network access after Gradle caches are cleaned.

## Pepper Robot Integration

Pepper integration is available through `pepper_interface/`. The robot client
is kept Python 2.7 compatible for NAOqi and supports:

- Speech output, gestures, and tablet display.
- Spoken A-D scenario answers.
- Spoken `question` and `repeat` commands.
- Grounded or baseline answers for supported trainee questions.
- Session and interaction logging through the same FastAPI API used by the
  Android app.

Pepper and the laptop must be on the same WiFi. Start middleware on all
interfaces:

```powershell
uvicorn middleware.main:app --reload --host 0.0.0.0 --port 8000
```

Then run Pepper with the laptop LAN IP:

```bash
python pepper_interface/pepper_client.py \
  --ip PEPPER_IP_ADDRESS \
  --port 9559 \
  --middleware http://LAPTOP_IP_ADDRESS:8000 \
  --participant PEPPER_P001 \
  --condition grounded \
  --organisation SENTRY_STUDY \
  --response-mode voice
```

Laptop smoke test:

```powershell
.\venv\Scripts\python.exe pepper_interface\pepper_client.py `
  --simulation `
  --middleware http://localhost:8000 `
  --participant PEPPER_SIM `
  --condition grounded `
  --organisation SENTRY_STUDY `
  --response-mode auto
```

Pepper's built-in NAOqi speech recognition is vocabulary-based. The current
implementation prioritizes reliable low-latency HRI for answer choices and
supported short questions. See `pepper_interface/GUIDE.md` for full setup,
login details, latency controls, and physical test checklist.

## Testing

Recommended local Python verification:

```powershell
.\venv\Scripts\python.exe -m pytest tests\unit -m "not live" -q
```

Broader non-live verification:

```powershell
.\venv\Scripts\python.exe -m pytest tests -m "not live" -q
```

Known test constraints:

- Some integration tests expect middleware running at `localhost:8000`.
- Tests marked `@pytest.mark.live` may call OpenAI and require the generated
  ChromaDB vector store.
- Android Gradle verification may require dependency downloads.

## Evaluation Study

The research compares two conditions:

- Grounded: response generated with retrieved OWASP/legal context.
- Baseline: response generated without retrieval context.

Tracked metrics include:

- Grounding accuracy.
- Hallucination rate.
- Knowledge gain.
- Response latency.
- Token usage.
- Session and assessment outcomes.

Evaluation tooling lives under `evaluation/`.

## Current Roadmap

### Phase 5: Production OWASP Training System

- Implemented in the Android trainee flow with local OWASP modules and saved
  answer feedback.
- Remaining follow-up: verify on API 23/Pepper tablet and decide whether
  detailed per-question history should move from local storage to backend
  persistence.

### Phase 6: Organisation Analytics System

- Implemented additive Organisation/User models.
- Implemented Admin research analytics vs Manager organisation analytics split.
- Implemented Manager trainee create/deactivate flow.
- Implemented organisation overview, trainee performance, department
  performance, and OWASP weakness analytics from real session activity.
- Remaining: signed/expiring tokens, stronger password hashing, wider endpoint
  authorization, and richer historical user trend persistence.

### Phase 7: Final Review

- Review architecture, mobile, backend, RAG, security, scalability, and
  technical debt.

## Development Guardrails

- Do not modify database schemas without explicit approval.
- Do not remove tests without explicit approval.
- Keep `pepper_interface/pepper_client.py` Python 2.7 compatible.
- Treat `backend/database/models.py` as schema-sensitive.
- Treat source code as truth when documentation and implementation disagree.

## Documentation

The synchronized onboarding documentation lives in:

```text
SENTRY_skills/sentry-skill/
```

Start with `SENTRY_skills/sentry-skill/SKILL.md`, then use the files in
`SENTRY_skills/sentry-skill/references/` for architecture, backend, Android,
workflows, roadmap, and troubleshooting details.
