# SENTRY: Integrating a Pepper Robot with Artificial Intelligence for Automated Cybersecurity Training

**Student:** Derick Richard Tsumah  
**Registration Number:** SCT212-0192/2022  
**Programme:** BSc. Computer Technology  
**Course:** BCT 2406 Project Definition and Implementation  
**Institution:** Jomo Kenyatta University of Agriculture and Technology  
**Supervisors:** Dr. Eunice Njeri and Prof. Waweru Mwangi  
**Submission Date:** June 2026  

---

# Abstract

SENTRY is an artificial intelligence powered cybersecurity training prototype that investigates how Retrieval-Augmented Generation (RAG) can improve the reliability of conversational responses delivered through an embodied training interface. The project responds to two related problems. First, organisations increasingly need practical cybersecurity awareness training because non-technical employees remain exposed to phishing, insecure credential handling, unsafe file sharing, social engineering and other common security risks. Secondly, large language models can generate fluent but unsupported responses, a behaviour commonly described as hallucination, which is especially risky when the model is used in an educational robot that users may perceive as credible.

The implemented system combines a Python FastAPI middleware, a custom RAG pipeline, a PostgreSQL persistence layer, an Android Jetpack Compose application, and a Pepper robot interface. The RAG pipeline indexes curated cybersecurity and legal knowledge sources, including OWASP application security material and Kenyan cyber/data protection legislation. User questions are embedded, matched against a ChromaDB vector store, and passed to a language model with retrieved context and grounding instructions. A baseline endpoint without retrieval is implemented for comparison. The Android application supports trainee, manager and administrator roles, fixed OWASP Top 10 assessment modules, local progress storage, open-ended grounded chat, and organisation-level analytics. The Pepper layer provides NAOqi-compatible speech, gesture, tablet display, scenario flow and simulation support.

The project follows a Design Science Research methodology: the artefact was designed, implemented, tested and evaluated against technical and educational objectives. Automated unit tests cover prompt construction, RAG orchestration, hallucination scoring, database models and Pepper-compatible interfaces. Integration tests exercise session lifecycle endpoints and analytics flows when the middleware is running. The evaluation framework records grounding accuracy, hallucination rate, response latency, token usage and knowledge gain so that grounded and baseline conditions can be compared statistically using Welch's t-test and effect size analysis after participant data collection.

The main contribution of SENTRY is a working architecture for grounded generative AI in a cybersecurity tutoring context, with traceable responses, structured assessment data and multi-interface delivery through mobile and robot channels. The prototype is not yet production-ready: authentication uses prototype bearer tokens, database schema evolution still relies on startup table creation rather than migrations, offline Android results are not synchronised later, and physical Pepper/API 23 validation remains a future activity. Nevertheless, the implementation demonstrates a practical path toward trustworthy embodied cybersecurity training.

---

# Acknowledgements

I thank my supervisors, Dr. Eunice Njeri and Prof. Waweru Mwangi, for their academic guidance during the definition and implementation of this project. I also acknowledge the support of my project colleague Timothy Wachira, whose work on the human-robot interaction and mobile interface complemented the grounding and evaluation architecture described in this report. I am grateful to the Jomo Kenyatta University of Agriculture and Technology community for providing the learning environment in which this capstone project was developed.

---

# Dedication

This project is dedicated to learners and organisations seeking safer ways to understand cybersecurity risks through practical, accessible and trustworthy technology.

---

# Table of Contents

1. Chapter 1: Introduction  
2. Chapter 2: State of the Art and Literature Review  
3. Chapter 3: Approach and Methodology  
4. Chapter 4: System Analysis and Design  
5. Chapter 5: System Implementation  
6. Chapter 6: Testing, Results and Evaluation  
7. Chapter 7: Conclusions and Recommendations  
8. References  
9. Appendices  

---

# List of Figures

**Figure 1.1:** Conceptual overview of SENTRY as a grounded cybersecurity training system.  
**Figure 4.1:** System context diagram showing trainee, Pepper robot, Android app, middleware, RAG engine and database.  
**Figure 4.2:** Component architecture of the SENTRY platform.  
**Figure 4.3:** RAG pipeline sequence from user question to grounded response.  
**Figure 4.4:** Entity relationship diagram for SENTRY database models.  
**Figure 5.1:** Android trainee home screen.  
**Figure 5.2:** OWASP module assessment screen.  
**Figure 5.3:** Grounded chat response with sources.  
**Figure 5.4:** Admin/manager analytics dashboard.  
**Figure 5.5:** Pepper tablet scenario and feedback screens.  

> Insert the exported architecture image from `docs/architecture/sentry_architecture.svg` as Figure 4.2 after conversion to a Word-compatible image format.

---

# List of Tables

**Table 3.1:** Technologies used in SENTRY.  
**Table 4.1:** Functional requirements.  
**Table 4.2:** Non-functional requirements.  
**Table 4.3:** Database entities.  
**Table 5.1:** Implemented features mapped to repository modules.  
**Table 6.1:** Automated test coverage summary.  
**Table 6.2:** Evaluation metrics and interpretation.  
**Table 7.1:** Limitations and recommended future work.  

---

# Abbreviations

**AI:** Artificial Intelligence  
**API:** Application Programming Interface  
**ASR:** Automatic Speech Recognition  
**DSR:** Design Science Research  
**HRI:** Human-Robot Interaction  
**LLM:** Large Language Model  
**NAOqi:** SoftBank Robotics Pepper robot software framework  
**OWASP:** Open Worldwide Application Security Project  
**RAG:** Retrieval-Augmented Generation  
**RBAC:** Role-Based Access Control  
**SME:** Small and Medium Enterprise  
**TTS:** Text-to-Speech  

---

# Chapter 1: Introduction

## 1.1 Motivation and Background

Cybersecurity awareness is now a practical organisational requirement rather than a purely technical concern. Employees interact with emails, web applications, shared documents, passwords, administrative workflows and external suppliers every day. Mistakes such as approving unexpected multi-factor authentication prompts, reusing passwords, enabling unsafe macros or ignoring suspicious login activity can expose organisations to significant security risks. Training therefore needs to be understandable, memorable and connected to realistic workplace decisions.

SENTRY, which stands for **Integrating a Pepper Robot with Artificial Intelligence for Automated Cybersecurity Training**, explores how an embodied conversational system can support this need. The project combines social robotics, mobile learning, cybersecurity education and retrieval-grounded generative AI. The Pepper robot and Android interface are used to present practical training scenarios, while the AI subsystem provides open-ended explanations grounded in curated cybersecurity knowledge.

The project was motivated by a key weakness in modern conversational AI. Large language models can produce fluent explanations across many domains, but research has shown that they may also generate unsupported or false content (Ji et al., 2023). In a cybersecurity tutoring system, this weakness is not merely inconvenient. Incorrect advice can damage user trust, mislead learners and normalise unsafe practice. Trust research in human-robot interaction has also shown that errors reduce user compliance and perceived reliability (Hancock et al., 2011; Salem et al., 2015; de Visser et al., 2018). Therefore, the SENTRY project treats factual grounding as a core system requirement.

## 1.2 Statement of the Problem

Traditional cybersecurity awareness training is often delivered through static slides, videos or quizzes. These approaches can communicate policy but may not adequately simulate real workplace choices. At the same time, a fully open-ended AI tutor can provide adaptive explanations but may hallucinate when it lacks verified context. This creates the following problem:

**How can an AI-powered cybersecurity training system provide interactive, engaging and traceable guidance while reducing unsupported responses from a large language model?**

This problem has technical, educational and ethical dimensions. Technically, the system needs retrieval, prompt control, structured APIs, persistent evaluation logs and reliable user interfaces. Educationally, the system needs realistic scenario-based training and measurable assessment outcomes. Ethically, the system must avoid overclaiming, protect participant data and remain transparent about its limitations.

## 1.3 Proposed Solution

SENTRY proposes and implements a modular grounded tutoring architecture. Instead of allowing the language model to answer solely from its internal parameters, the system retrieves relevant material from curated cybersecurity and legal documents before generating a response. The retrieved context is inserted into a prompt that instructs the model to answer only from verified information. The system also exposes a baseline endpoint without retrieval so that grounded and ungrounded responses can be compared.

The implemented prototype includes:

- A FastAPI middleware that exposes query, session, analytics, user and organisation endpoints.
- A RAG engine using sentence-transformer embeddings, ChromaDB retrieval, context-window management and OpenAI generation.
- A PostgreSQL schema for organisations, users, training sessions, scenario interactions, assessment results and evaluation logs.
- An Android Jetpack Compose app with trainee, manager and administrator flows.
- A Pepper robot interface with speech, gestures, tablet display, scenario control and simulation mode.
- Evaluation tools for grounding accuracy, hallucination rate, latency, token usage and statistical analysis.

> **Figure 1.1 Placeholder:** Insert a high-level concept diagram showing the trainee interacting with Android/Pepper, the middleware routing requests, the RAG pipeline retrieving from OWASP/legal sources, and PostgreSQL storing analytics.

## 1.4 Objectives

### 1.4.1 Main Objective

To design, implement and evaluate a retrieval-augmented AI architecture for an interactive cybersecurity training system that improves response grounding and supports scenario-based learning through Android and Pepper robot interfaces.

### 1.4.2 Specific Objectives

1. To investigate existing work on social robots in education, trust in human-robot interaction, hallucination in language models and retrieval-augmented generation.
2. To design a modular architecture that separates user interaction, middleware, retrieval, generation, persistence and analytics.
3. To develop a curated cybersecurity knowledge base using OWASP application security content and Kenyan legal documents.
4. To implement grounded and baseline AI response modes for comparative evaluation.
5. To implement trainee, manager and administrator workflows for cybersecurity training and analytics.
6. To implement a Pepper-compatible interface for scenario delivery, speech interaction and tablet feedback.
7. To evaluate the prototype using automated tests, interaction logs, grounding metrics, hallucination metrics, latency metrics and planned participant-study analysis.

## 1.5 Scope

The project scope is a research prototype for cybersecurity training, not a production multi-tenant enterprise platform. The implemented system supports local middleware deployment, Android client interaction, Pepper simulation/NAOqi integration, PostgreSQL-backed logging and RAG-based open-ended responses. Production features such as signed expiring tokens, full audit logging, scalable deployment, offline result synchronisation and manager-authored curriculum governance are identified as future work.

---

# Chapter 2: State of the Art and Literature Review

## 2.1 Social Robots in Education

Social robots have been studied as educational companions because embodiment can support attention, engagement and social presence. Tanaka, Cicourel and Movellan (2007) demonstrated early socialisation between toddlers and robots in an education setting, showing that physical robots can sustain interaction in ways that differ from ordinary screen interfaces. Belpaeme et al. (2018) reviewed educational social robots and found that embodiment, adaptive behaviour and structured interaction design are important for effective robot-supported learning.

However, the literature also shows that social behaviour alone is insufficient. Kennedy, Baxter and Belpaeme (2015) showed that a robot tutor's social behaviour can negatively affect learning when it is not properly calibrated. Ahmad, Mubin and Orlando (2016) further highlighted that teachers expect educational robots to support clear roles rather than replace human instruction. These findings shaped SENTRY's design: Pepper is used as a guided training interface, while the learning content, assessment logic and data evaluation remain structured and auditable.

## 2.2 Trust and Reliability in Human-Robot Interaction

Trust is central to human-robot interaction. Hancock et al. (2011) found that robot performance, human factors and environmental factors all influence trust. de Visser, Pak and Shaw (2018) argued that as systems move from automation to autonomy, trust repair becomes increasingly important after failure. For AI tutoring, a fluent but incorrect explanation can be especially harmful because users may assume the robot is knowledgeable.

SENTRY addresses this by designing for epistemic accountability. Responses generated through the grounded endpoint carry source metadata, retrieved chunks and latency/token traces. The Android chat interface displays response sources, while Pepper tablet feedback can show source tags. This does not eliminate all risk, but it makes system behaviour more transparent and measurable.

## 2.3 Large Language Models and Hallucination

Large language models have shown strong few-shot learning and natural language generation performance (Brown et al., 2020). Their usefulness in tutoring arises from the ability to explain concepts conversationally and adapt wording to the user. Nevertheless, hallucination remains a major weakness. Ji et al. (2023) describe hallucination as generated content that is unsupported by the source or inconsistent with factual knowledge. In cybersecurity education, hallucination can lead to unsafe recommendations or false legal/security claims.

The SENTRY project therefore does not treat the LLM as the sole source of truth. The model is used as a natural language generator conditioned by retrieved context. The implementation further supports a baseline mode so that the effect of retrieval can be measured rather than assumed.

## 2.4 Retrieval-Augmented Generation

Retrieval-Augmented Generation combines information retrieval with language generation. Lewis et al. (2020) introduced RAG for knowledge-intensive NLP tasks, showing how retrieved documents can support generation. Karpukhin et al. (2020) demonstrated dense passage retrieval using learned embeddings for semantic search. Borgeaud et al. (2022) later showed that retrieval-enhanced language models can improve performance by accessing external text rather than relying only on parameters.

SENTRY adapts these ideas to an embodied training setting. The system chunks OWASP and legal documents, embeds them using `all-MiniLM-L6-v2`, stores them in ChromaDB and retrieves top relevant chunks for a user query. The prompt builder enforces a context token budget and includes source identifiers, while the generation client applies grounding instructions. This adapts text-based RAG principles into a robot/mobile tutoring architecture.

## 2.5 Cybersecurity Education and Knowledge Sources

The cybersecurity domain requires accurate and current content. SENTRY uses OWASP application security material because OWASP provides widely used categories for explaining common software and organisational security risks. The prototype also includes Kenyan legal documents, including the Computer Misuse and Cybercrimes Act and the Data Protection Act, because the project is situated in Kenya and legal/security training should be locally relevant.

The Android curriculum maps to OWASP Top 10 style categories, including broken access control, security misconfiguration, software supply chain failures, cryptographic failures, injection, insecure design, authentication failures, integrity failures, logging and alerting failures, and mishandling exceptional conditions. This gives the training flow practical structure while preserving RAG for open-ended questions.

## 2.6 Research Gap

Existing work separately demonstrates the promise of social robots in education, the importance of trust in HRI and the value of retrieval augmentation for language models. The gap addressed by SENTRY lies at their intersection: there is limited practical evidence on how a retrieval-grounded LLM pipeline can be integrated into a robot-supported cybersecurity training system with measurable grounding, hallucination and learning analytics.

SENTRY contributes by implementing this integration in a working prototype. The contribution is not only the use of a robot, nor only the use of RAG, but the combination of scenario control, grounded open-ended explanation, session logging and evaluation tooling in one system.

---

# Chapter 3: Approach and Methodology

## 3.1 Methodological Approach

The project follows **Design Science Research (DSR)** because it creates and evaluates a computing artefact intended to solve a real problem. The artefact is the SENTRY grounded cybersecurity training platform. The methodology combines literature review, requirements engineering, system design, iterative implementation, automated testing and evaluation planning.

The development process followed these phases:

1. **Problem scoping:** Identification of hallucination risk, cybersecurity awareness needs and Pepper robot constraints.
2. **Literature review:** Review of educational social robots, trust in HRI, LLM hallucination, RAG and cybersecurity education.
3. **Architecture design:** Separation of Android/Pepper interfaces, FastAPI middleware, RAG engine, database and evaluation modules.
4. **Knowledge-base construction:** Collection, preprocessing, chunking and indexing of OWASP and Kenyan legal sources.
5. **Implementation:** Development of Python backend, RAG pipeline, Android app, Pepper interface and evaluation tools.
6. **Testing and validation:** Unit tests, integration tests, Android unit tests, simulation runs and evaluation framework checks.
7. **Documentation and reflection:** Final report, API documentation, architecture notes and future work analysis.

## 3.2 Technologies

| Layer | Technologies | Purpose |
|---|---|---|
| Backend middleware | Python, FastAPI, Pydantic, Uvicorn | REST API and validation |
| Database | PostgreSQL, SQLAlchemy async ORM | Sessions, users, analytics and evaluation logs |
| RAG engine | ChromaDB, sentence-transformers, tiktoken, OpenAI API | Retrieval, prompt construction and generation |
| Knowledge ingestion | Markdown/PDF readers, chunker, embedding pipeline | Prepare source documents for retrieval |
| Mobile app | Kotlin, Jetpack Compose, Hilt, Ktor CIO, DataStore | Android trainee/admin/manager interface |
| Pepper interface | Python 2.7 compatible NAOqi client, tablet HTML, speech recognition | Robot interaction and simulation |
| Evaluation | pytest, pandas, NumPy, SciPy | Automated tests and statistical analysis |
| Documentation | Markdown, OpenAPI, SENTRY_skills notes | Technical and academic reporting |

## 3.3 Data

### 3.3.1 Secondary Knowledge Sources

The knowledge base contains two main categories:

- **OWASP Markdown documents:** Application security categories and guidance.
- **Kenyan legal PDFs:** Computer Misuse and Cybercrimes Act and Data Protection Act.

The ingestion report shows 18 documents processed into 175 chunks using a 512-token chunk size and 64-token overlap. The embedding model is `all-MiniLM-L6-v2`, and the ChromaDB collection is named `sentry_knowledge`.

### 3.3.2 Primary Interaction Data

The system is designed to collect anonymised training data:

- participant identifier,
- organisation identifier,
- training condition (`grounded` or `baseline`),
- scenario interaction records,
- selected answers,
- response timing,
- pre/post assessment scores,
- knowledge gain,
- retrieved sources,
- grounding accuracy and hallucination metrics.

Raw audio is not stored by the implemented backend. Android speech recognition and Pepper speech recognition are used for interaction, but the persisted records are text and metadata.

## 3.4 Evaluation, Testing and Validation

The evaluation design compares grounded and baseline conditions. The implemented tools support:

- grounding accuracy,
- hallucination rate,
- grounding improvement,
- response latency,
- token usage,
- session completion,
- assessment score and knowledge gain,
- organisation-level analytics,
- statistical testing using Welch's t-test and Cohen's d.

Formal participant testing is supported by the evaluation scripts but should be completed before final empirical claims are made. Until participant data is collected, the report treats automated and simulation testing as implementation validation rather than proof of learning effectiveness.

## 3.5 Responsible Computing and Ethical Considerations

### 3.5.1 Privacy and Data Protection

The system stores anonymised participant identifiers and organisation identifiers rather than personal names for training sessions. Manager views are designed around anonymised trainee IDs, departments and performance indicators. The design aligns with data minimisation principles in the Kenya Data Protection Act by avoiding raw audio retention and collecting only data relevant to learning and evaluation.

### 3.5.2 Safety and Overreliance

SENTRY is an educational tool, not a replacement for security professionals or legal counsel. Grounded prompts instruct the model not to invent information and to say when context is insufficient. Legal references are framed as educational guidance rather than definitive legal advice.

### 3.5.3 Bias and Accessibility

Speech recognition can behave differently across accents, environments and devices. The Android interface therefore retains touch and typed alternatives. Pepper recognition uses a compact vocabulary for A-D responses to reduce recognition errors. Future work should include testing with diverse users and target hardware.

### 3.5.4 Security

The project includes role separation and route guards for admin/manager analytics. However, authentication remains prototype-grade. Signed expiring tokens, stronger password hashing, rate limiting and wider endpoint scoping are required before deployment in a real organisation.

---

# Chapter 4: System Analysis and Design

## 4.1 System Analysis

### 4.1.1 Stakeholders

- **Trainees:** Employees or learners completing cybersecurity training.
- **Managers:** Organisation representatives who monitor anonymised performance and weakness areas.
- **Administrators:** Research or system administrators who review study analytics and manage trainee accounts.
- **Researchers:** Project evaluators comparing grounded and baseline responses.
- **Security educators:** Future users who may adapt training content.

### 4.1.2 Functional Requirements

| ID | Requirement | Implementation Evidence |
|---|---|---|
| FR1 | Start and end training sessions | `/api/v1/sessions/start`, `/api/v1/sessions/end` |
| FR2 | Log scenario interactions | `/api/v1/sessions/interaction` |
| FR3 | Answer open-ended cybersecurity questions using grounded RAG | `/api/v1/query` and Android chat |
| FR4 | Provide baseline LLM-only responses for comparison | `/api/v1/query/baseline` |
| FR5 | Store training, assessment and evaluation data | SQLAlchemy models in `backend/database/models.py` |
| FR6 | Support trainee OWASP module training | `OwaspCurriculum.kt` and `SessionViewModel.kt` |
| FR7 | Support role-based admin and manager analytics | `analytics_routes.py`, `user_routes.py` |
| FR8 | Support Pepper speech/tablet scenario delivery | `pepper_client.py`, `dialogue_manager.py` |
| FR9 | Provide evaluation metrics and reports | `evaluation/metrics`, `run_evaluation.py`, `analyse_results.py` |

### 4.1.3 Non-Functional Requirements

| Requirement | Design Response |
|---|---|
| Reliability | Deterministic Android assessment content; backend session validation; fallback messages when server is unreachable |
| Traceability | Query responses include sources, retrieved chunks, latency and token metadata |
| Maintainability | Layered repository structure and compatibility shims for older imports |
| Security | Role guards for admin/manager analytics; anonymised participant records; future hardening identified |
| Usability | Android touch UI, typed chat, speech input and Pepper tablet display |
| Performance | Top-k retrieval, token budget control and stored embeddings in ChromaDB |
| Portability | Local FastAPI server, Android emulator URL support and Pepper simulation mode |

### 4.1.4 Feasibility

The project is technically feasible because the implementation uses widely available frameworks and separates high-risk dependencies. Pepper runs a Python 2.7-compatible client, while AI processing runs on a modern Python backend. Android uses standard Jetpack Compose and Ktor libraries. The main feasibility constraint is that live RAG requires an OpenAI API key and a generated vector store.

## 4.2 System Design

### 4.2.1 Architectural Overview

SENTRY uses a layered architecture:

1. **Client layer:** Android app and Pepper robot interface.
2. **Middleware layer:** FastAPI routes for query, session, analytics and user flows.
3. **AI layer:** RAG retrieval, prompt construction and LLM generation.
4. **Persistence layer:** PostgreSQL for structured data and ChromaDB for vector retrieval.
5. **Evaluation layer:** scoring, logging and statistical analysis scripts.

> **Figure 4.1 Placeholder:** Insert system context diagram.

> **Figure 4.2 Placeholder:** Insert component architecture diagram from `docs/architecture/sentry_architecture.svg`.

### 4.2.2 RAG Sequence

The grounded response sequence is:

1. A trainee asks a cybersecurity question through Android chat or Pepper.
2. The middleware validates the request.
3. The RAG pipeline embeds the query using the same embedding model used during ingestion.
4. ChromaDB retrieves top-k relevant chunks from OWASP/legal sources.
5. Retrieved chunks are filtered by relevance and fitted into a context token budget.
6. The prompt builder constructs a verified-context prompt.
7. The OpenAI client generates a response under grounding instructions.
8. The API returns the response, sources, chunks, token usage and latency.
9. Evaluation logs can store grounding and hallucination metrics.

> **Figure 4.3 Placeholder:** Insert sequence diagram for grounded query flow.

### 4.2.3 Database Design

| Entity | Purpose |
|---|---|
| `organisations` | Stores organisation name, canonical ID and active flag |
| `users` | Stores participant ID, role, PIN hash, organisation, department and status |
| `training_sessions` | Stores participant, condition, organisation, timing and scores |
| `scenario_interactions` | Stores decisions, responses, timing, correction loops and sources |
| `assessment_results` | Stores pre/post scores and improvement |
| `evaluation_logs` | Stores query, response, grounding metrics, latency, token usage and sources |

> **Figure 4.4 Placeholder:** Insert ERD showing organisations, users, training sessions, scenario interactions, assessment results and evaluation logs.

### 4.2.4 API Design

The implemented API includes:

- `GET /health`
- `GET /api/v1/knowledge-base/status`
- `POST /api/v1/query`
- `POST /api/v1/query/baseline`
- `POST /api/v1/sessions/start`
- `POST /api/v1/sessions/end`
- `GET /api/v1/sessions/{session_id}`
- `POST /api/v1/sessions/interaction`
- `POST /api/v1/sessions/eval-log`
- `GET /api/v1/analytics/study`
- `GET /api/v1/analytics/sessions`
- `GET /api/v1/analytics/organisation/{organisation_id}`
- `POST /api/v1/organisations`
- `POST /api/v1/users/login`
- `GET /api/v1/manager/trainees`
- `POST /api/v1/manager/trainees`
- `PATCH /api/v1/manager/trainees/{user_id}/deactivate`
- `GET /api/v1/manager/analytics/overview`
- `GET /api/v1/manager/analytics/weaknesses`

FastAPI interactive documentation is available at `/docs` when the middleware is running.

### 4.2.5 User Interface Design

The Android interface separates users by role:

- **Trainee:** module cards, OWASP assessments, results, chat and settings.
- **Manager:** organisation performance, trainee list and weakness analytics.
- **Admin:** research-wide analytics and trainee account management.

The Pepper interface uses speech, gestures and tablet HTML pages for welcome, scenario prompt, feedback and results. The robot flow uses a state machine and controlled vocabularies to keep interaction reliable.

---

# Chapter 5: System Implementation

## 5.1 Backend Middleware

The backend middleware is implemented in `middleware/main.py`. On startup, it validates settings, initialises database tables, constructs a shared `RAGPipeline` and registers the query, session, analytics and user routers. CORS is enabled for development clients.

The query router exposes grounded and baseline endpoints. The session router creates sessions, records interactions, closes sessions and logs evaluation records. Analytics routes aggregate study-level and organisation-level metrics. User routes implement prototype login, organisation creation, trainee management and manager analytics.

## 5.2 RAG Pipeline Implementation

The RAG pipeline is implemented in `ai_engine/rag/pipeline.py` and uses:

- `Retriever` for ChromaDB semantic search,
- `PromptBuilder` for context formatting and token budgeting,
- `LLMClient` for baseline and grounded generation.

The retriever converts ChromaDB cosine distance into a similarity score and filters results below the configured relevance threshold. The prompt builder limits context by `RAG_CONTEXT_TOKEN_BUDGET` and `RAG_MAX_CONTEXT_CHUNKS`. The grounded system prompt instructs the model to use only verified context and to acknowledge insufficient context instead of guessing.

## 5.3 Knowledge Base Ingestion

The ingestion pipeline processes Markdown and PDF files from `knowledge_base/raw`. It cleans text, chunks it using token windows, embeds chunks, stores them in ChromaDB and writes an ingestion report. The current ingestion report records 18 documents and 175 chunks.

## 5.4 Database and Analytics Implementation

The SQLAlchemy models implement the research and training data structure. `TrainingSession` is the central entity, linked to optional users, interactions, assessment results and evaluation logs. Manager analytics aggregate sessions and risky answers by organisation, department, trainee and weakness category. Admin analytics focus on grounded-vs-baseline study metrics.

The current schema is created at startup using SQLAlchemy metadata and additive schema updates. Although `alembic` is listed in dependencies, a migration workflow has not yet been activated.

## 5.5 Android Application Implementation

The Android application is implemented in Kotlin using Jetpack Compose. It includes:

- Hilt dependency injection,
- Ktor CIO networking,
- DataStore token/progress storage,
- role-based navigation,
- OWASP curriculum modules,
- speech-assisted training interaction,
- local offline session fallback,
- open-ended grounded chat,
- admin and manager analytics screens.

The fixed OWASP curriculum includes ten modules and twenty questions. Each module has two practical workplace scenarios, four A-D choices, one correct answer and stored feedback for all answers. Fixed assessment feedback is local rather than generated through RAG, which ensures consistent grading. Open-ended questions remain available through the chat screen and grounded query endpoint.

> **Figure 5.1 Placeholder:** Insert Android trainee home screenshot.  
> **Figure 5.2 Placeholder:** Insert OWASP assessment screenshot.  
> **Figure 5.3 Placeholder:** Insert grounded chat screenshot with sources.

## 5.6 Pepper Robot Interface Implementation

The Pepper interface is implemented in Python with Python 2.7 compatibility for NAOqi. It supports:

- speech output through Pepper TTS,
- gesture animations,
- tablet scenario and feedback pages,
- A-D answer recognition using controlled vocabulary,
- `question` and `repeat` commands,
- middleware communication through HTTP,
- simulation mode for development without physical hardware.

The Pepper client remains intentionally conservative because NAOqi speech recognition is vocabulary-based. This improves reliability for multiple-choice scenarios while keeping free-form questions available through supported short prompts.

> **Figure 5.5 Placeholder:** Insert Pepper tablet screenshots or photographs from a simulation/physical run.

## 5.7 Evaluation Tooling

The evaluation package supports the planned grounded-vs-baseline study. `HallucinationScorer` computes automated grounding accuracy and hallucination rate using n-gram overlap against retrieved context. `GroundingScorer` stores per-query and aggregate reports. `run_evaluation.py` runs standardised evaluation sessions, and `analyse_results.py` combines participant CSV files and performs statistical analysis.

The automated scorer is a proxy metric and should be complemented by human expert annotation for final claims. This limitation is explicitly recognised in the code comments and in this report.

## 5.8 Implemented Feature Summary

| Feature | Status | Notes |
|---|---|---|
| Grounded RAG query endpoint | Implemented | Uses OWASP/legal vector store |
| Baseline query endpoint | Implemented | Used for control condition |
| Android trainee OWASP training | Implemented | Local deterministic assessment |
| Android grounded chat | Implemented | Calls `/api/v1/query` |
| PostgreSQL session logging | Implemented | Sessions, interactions, assessments, evaluation logs |
| Admin and manager analytics | Implemented | Role-guarded prototype routes |
| Pepper scenario interaction | Implemented | Simulation and NAOqi-compatible runtime |
| Evaluation scripts | Implemented | Requires participant runs for final empirical results |
| Production authentication | Partially implemented | Prototype tokens and SHA-256 PIN hash |
| Offline result sync | Not implemented | Local-only offline sessions |
| Manager-authored curriculum | Not implemented | Future governance requirement |

---

# Chapter 6: Testing, Results and Evaluation

## 6.1 Testing Strategy

Testing was performed at multiple levels:

| Test Type | Evidence | Purpose |
|---|---|---|
| Unit tests | `tests/unit` | Validate prompt building, RAG orchestration, scorer logic, database models and Pepper-compatible interfaces |
| Integration tests | `tests/integration` | Validate session endpoints and full pipeline flows when middleware is running |
| Android unit tests | `mobile_app/app/src/test` | Validate curriculum and organisation ID logic |
| Simulation tests | Pepper `--simulation` mode | Validate robot flow without physical hardware |
| Manual run checks | README commands and API docs | Validate middleware startup and Android build |

## 6.2 Automated Verification

The recommended Python command for non-live unit testing is:

```powershell
.\venv\Scripts\python.exe -m pytest tests\unit -m "not live" -q
```

Live tests are marked separately because they may require OpenAI access and a generated ChromaDB vector store. Integration tests require the FastAPI middleware to be running at `localhost:8000`.

Android verification is performed with:

```powershell
cd mobile_app
.\gradlew.bat :app:testDebugUnitTest
.\gradlew.bat :app:assembleDebug
```

## 6.3 Model and RAG Validation

The RAG pipeline is validated through:

- retrieval result inspection,
- source metadata returned in API responses,
- token and latency tracking,
- baseline versus grounded comparison,
- hallucination scorer output,
- evaluation logs persisted to PostgreSQL.

The implemented automated hallucination metric defines grounding accuracy as the proportion of generated sentences with meaningful n-gram overlap against retrieved context. This is useful for consistency but may penalise valid paraphrases. The final evaluation should therefore combine automated scoring with expert review.

## 6.4 Evaluation Metrics

| Metric | Meaning | Desired Direction |
|---|---|---|
| Grounding accuracy | Proportion of response sentences traceable to retrieved context | Higher |
| Hallucination rate | `1 - grounding_accuracy` | Lower |
| Grounding improvement | Grounded score minus baseline score | Higher |
| Retrieval latency | Time spent retrieving context | Lower, within usability limits |
| Generation latency | Time spent generating response | Lower, within usability limits |
| Knowledge gain | Post-assessment score minus pre-assessment score | Higher |
| Completion rate | Completed sessions over started sessions | Higher |
| Risk rate | Risky answers over total answers for a category | Lower |

## 6.5 User Feedback

Formal participant feedback is planned but not yet present in the repository as completed survey data. The system is prepared to support a controlled study with approximately 40 participants split between grounded and baseline conditions, as described in the original proposal. The final empirical report should include:

- participant demographics summary without personally identifiable information,
- pre/post assessment score table,
- trust and reliability questionnaire results,
- qualitative feedback themes,
- modifications made after feedback,
- comparison between grounded and baseline groups.

> **Table 6.3 Placeholder:** Insert participant study descriptive statistics after data collection.  
> **Table 6.4 Placeholder:** Insert trust/reliability survey results after data collection.

## 6.6 Comparison with Existing Systems

Compared with static cybersecurity awareness training, SENTRY provides interactive scenarios, assessment logging and open-ended question answering. Compared with ordinary chatbot training, SENTRY adds retrieval grounding and source traceability. Compared with scripted robot tutoring systems, SENTRY keeps structured scenario control while allowing grounded open-ended explanations. The main disadvantage is increased system complexity: RAG requires vector-store preparation, OpenAI access, middleware availability and additional evaluation logic.

## 6.7 Results Summary

At implementation level, the project achieved the following:

- a working FastAPI middleware,
- a working RAG architecture over curated cybersecurity/legal sources,
- a structured Android OWASP training flow,
- role-based admin and manager analytics,
- Pepper-compatible interaction and simulation,
- persistent session and evaluation logging,
- automated test coverage for major backend and mobile logic,
- an evaluation framework ready for participant data.

Final claims about learning improvement and user trust should be made only after completing the planned participant study and statistical analysis.

---

# Chapter 7: Conclusions and Recommendations

## 7.1 Summary of Achievements

SENTRY successfully implements a grounded AI cybersecurity training prototype across backend, AI, mobile, robot and evaluation layers. The system demonstrates that RAG can be integrated into an embodied tutoring architecture while preserving traceability through retrieved sources, evaluation logs and session analytics. The Android application adds practical trainee workflows, while the Pepper interface demonstrates how the same middleware can support robot-mediated training.

The project also contributes a clear evaluation design. Grounded and baseline response modes are implemented, and the evaluation scripts can calculate grounding accuracy, hallucination rate, latency cost and statistical significance once participant data is collected.

## 7.2 Contributions

The main contributions are:

1. A modular architecture for RAG-grounded cybersecurity training.
2. A curated OWASP/legal knowledge ingestion pipeline.
3. A comparative grounded-versus-baseline evaluation framework.
4. A role-based Android training and analytics interface.
5. A Pepper-compatible scenario interaction layer with simulation support.
6. A documented prototype that can be extended toward organisational pilot deployment.

## 7.3 Limitations

| Limitation | Impact |
|---|---|
| Prototype authentication | Tokens are not signed or expired; PIN hashing is not production-grade |
| No active Alembic migration workflow | Schema drift risk as models evolve |
| Offline Android results are local-only | Offline sessions do not appear in manager/admin analytics |
| Formal user study data not yet included | Learning/trust claims remain evaluation-ready rather than proven |
| Physical Pepper/API 23 validation pending | Real hardware behaviour may differ from simulation |
| Automated hallucination scoring is approximate | Paraphrased grounded answers may be under-scored |
| Manager curriculum authoring absent | Organisation-specific training content requires future workflow |

## 7.4 Recommendations and Future Work

Future work should prioritise:

1. Replace prototype tokens with signed, expiring tokens and add rate limiting.
2. Replace SHA-256 PIN hashing with Argon2id or bcrypt using salts.
3. Introduce Alembic migrations for database schema management.
4. Complete physical Pepper and API 23 Android hardware validation.
5. Conduct the planned participant study and report statistical results.
6. Add server-side per-question progress if cross-device history is required.
7. Implement offline session synchronisation with duplicate-submission protection.
8. Add manager-controlled document upload and review governance before organisation content affects training.
9. Add exportable admin research reports and analytics filters.
10. Add CI/CD workflows for Python and Android verification.

## 7.5 Lessons Learned and Reflection

The project demonstrated that trustworthy AI systems require more than a capable language model. Practical grounding requires knowledge curation, retrieval design, prompt constraints, logging, evaluation metrics and honest treatment of limitations. The work also showed the value of modular architecture: Pepper's Python 2.7 constraints did not prevent use of a modern Python AI backend because middleware separated robot interaction from AI computation.

From a software engineering perspective, the project strengthened skills in FastAPI, asynchronous SQLAlchemy, Android Compose, Ktor networking, vector retrieval, prompt engineering, test design and technical documentation. From a research perspective, it clarified that evaluation must measure not only whether a system works, but whether it improves reliability, trust and learning in a way that can be defended with evidence.

---

# References

Ahmad, M. I., Mubin, O., & Orlando, J. (2016). Understanding behaviours and roles for social and adaptive robots in education: Teacher's perspective. *Proceedings of the 4th International Conference on Human Agent Interaction*, 297-304. https://doi.org/10.1145/2974804.2974829

Belpaeme, T., Kennedy, J., Ramachandran, A., Scassellati, B., & Tanaka, F. (2018). Social robots for education: A review. *Science Robotics, 3*(21), 1-9. https://doi.org/10.1126/scirobotics.aat5954

Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., van den Driessche, G., Lespiau, J.-B., Damoc, B., Clark, A., De, D., Casas, L., Guy, A., Menick, J., Ring, R., Hennigan, T., Huang, S., Maggiore, L., Jones, C., et al. (2022). Improving language models by retrieving from trillions of tokens. *Proceedings of the 39th International Conference on Machine Learning*.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems, 33*.

de Visser, E. J., Pak, R., & Shaw, T. H. (2018). From automation to autonomy: The importance of trust repair in human-machine interaction. *Ergonomics, 61*(10), 1409-1427. https://doi.org/10.1080/00140139.2018.1457725

Hancock, P. A., Billings, D. R., Schaefer, K. E., Chen, J. Y. C., de Visser, E. J., & Parasuraman, R. (2011). A meta-analysis of factors affecting trust in human-robot interaction. *Human Factors, 53*(5), 517-527. https://doi.org/10.1177/0018720811417254

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., & Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys, 55*(12). https://doi.org/10.1145/3571730

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense passage retrieval for open-domain question answering. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*.

Kennedy, J., Baxter, P., & Belpaeme, T. (2015). The robot who tried too hard: Social behaviour of a robot tutor can negatively affect child learning. *Proceedings of the Tenth Annual ACM/IEEE International Conference on Human-Robot Interaction*, 67-74. https://doi.org/10.1145/2696454.2696457

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459-9474.

Open Worldwide Application Security Project. (2025). *OWASP Top 10 application security documentation*. Local project knowledge base files under `knowledge_base/raw/owasp`.

Republic of Kenya. (2018). *Computer Misuse and Cybercrimes Act*. Local project knowledge base file under `knowledge_base/raw/legal`.

Republic of Kenya. (2019). *Data Protection Act*. Local project knowledge base file under `knowledge_base/raw/legal`.

Tanaka, F., Cicourel, A., & Movellan, J. R. (2007). Socialization between toddlers and robots at an early childhood education center. *Proceedings of the National Academy of Sciences, 104*(46), 17954-17958. https://doi.org/10.1073/pnas.0707769104

---

# Appendices

## Appendix A: Project Timeline

| Phase | Activities | Deliverables |
|---|---|---|
| Requirements | Problem scoping, literature review, stakeholder analysis | Requirements and proposal |
| Architecture | Middleware, RAG, database, Android and Pepper design | Architecture diagrams and API design |
| RAG development | Ingestion, embeddings, retrieval and prompt building | Working grounded query pipeline |
| Application development | Android screens, role flows, session handling | Trainee/admin/manager app |
| Pepper integration | NAOqi client, state machine, tablet views | Robot/simulation interaction flow |
| Evaluation | Tests, scoring scripts, analytics endpoints | Evaluation framework |
| Documentation | Final report, README, SENTRY_skills notes | Submission-ready documentation |

## Appendix B: Budget

| Item | Description | Estimated Cost (KES) |
|---|---|---:|
| OpenAI API credits | GPT generation and possible audio transcription | 15,000 |
| Vector database/runtime resources | Local ChromaDB now; hosted option for pilot | 5,000 |
| Android/Pepper testing resources | Device/network testing support | 5,000 |
| Documentation and printing | Final report production | 3,000 |
| **Total** |  | **28,000** |

## Appendix C: Key Repository Paths

| Path | Purpose |
|---|---|
| `ai_engine/` | RAG, retrieval, generation and ingestion modules |
| `middleware/` | FastAPI application and routes |
| `backend/database/` | SQLAlchemy models and database connection |
| `mobile_app/` | Android Jetpack Compose application |
| `pepper_interface/` | Pepper robot client, dialogue and scenarios |
| `evaluation/` | Scoring and statistical analysis tools |
| `knowledge_base/` | Raw and processed knowledge-base sources |
| `docs/api_specs/` | OpenAPI snapshot |
| `SENTRY_skills/` | Project architecture and workflow notes |

## Appendix D: Test and Run Commands

```powershell
# Python environment
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Regenerate vector store
.\venv\Scripts\python.exe scripts\ingest_knowledge_base.py

# Start middleware
uvicorn middleware.main:app --reload --host 0.0.0.0 --port 8000

# Run non-live Python unit tests
.\venv\Scripts\python.exe -m pytest tests\unit -m "not live" -q

# Android tests and debug build
cd mobile_app
.\gradlew.bat :app:testDebugUnitTest
.\gradlew.bat :app:assembleDebug

# Pepper simulation
.\venv\Scripts\python.exe pepper_interface\pepper_client.py --simulation --middleware http://localhost:8000 --participant PEPPER_SIM --condition grounded --organisation SENTRY_STUDY --response-mode auto
```

## Appendix E: Screenshot Checklist

- Android splash screen.
- Android authentication screen.
- Trainee home with OWASP module cards.
- OWASP question screen with A-D options.
- Feedback screen after answer selection.
- Results screen.
- Grounded chat response with source list.
- Admin research analytics dashboard.
- Manager organisation analytics dashboard.
- Pepper welcome tablet screen.
- Pepper scenario tablet screen.
- Pepper feedback tablet screen.

## Appendix F: User Study Instruments

> Insert final consent form, pre-assessment questionnaire, post-assessment questionnaire, trust/reliability scale and interview guide after supervisor approval.

## Appendix G: API Artefact

> Insert or attach the generated OpenAPI specification from the running FastAPI app. If the committed `docs/api_specs/sentry_api_spec.json` is used, regenerate it first to ensure it includes the latest user and manager routes.

