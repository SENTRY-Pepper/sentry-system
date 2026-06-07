## Current Integration Status

| Component | Status |
|---|---|
| FastAPI middleware server | Complete — running on port 8000 |
| `/api/v1/query` endpoint | Complete — grounded RAG responses |
| `/api/v1/query/baseline` endpoint | Complete — control condition |
| `/api/v1/sessions/*` endpoints | Complete — full session lifecycle |
| `/api/v1/analytics/*` endpoints | Complete — study + org analytics |
| PostgreSQL database | Complete — 4 tables initialised |
| API documentation | `http://localhost:8000/docs` |
| OpenAPI spec | `docs/api_specs/sentry_api_spec.json` |
| Android app | Your workstream — see `mobile_app/README_MOBILE.md` |
| NAOqi Pepper layer | Your workstream — this guide |

sentry-system/
│
├── ai_engine/                      # DERICK — RAG + LLM (unchanged)
│   ├── rag/
│   ├── llm/
│   └── embeddings/
│
├── knowledge_base/                 # DERICK — Vector store (unchanged)
│   ├── raw/
│   ├── processed/
│   └── vector_store/
│
├── middleware/                     # SHARED — FastAPI server (expanded)
│   ├── routes/
│   │   ├── query_routes.py         # DERICK — AI query endpoints (done)
│   │   ├── session_routes.py       # TIMOTHY — Session CRUD endpoints
│   │   └── analytics_routes.py     # TIMOTHY — Dashboard data endpoints
│   ├── validators/
│   │   ├── request_validator.py    # DERICK (done)
│   │   └── session_validator.py    # TIMOTHY — Session Pydantic models
│   └── main.py                     # SHARED — registers all routers
│
├── backend/                        # TIMOTHY (Derick sets up foundation)
│   ├── database/
│   │   ├── connection.py           # PostgreSQL async connection
│   │   ├── models.py               # SQLAlchemy table definitions
│   │   └── migrations/             # Alembic schema versioning
│   ├── analytics/
│   │   ├── session_analytics.py    # Aggregates metrics per session
│   │   └── report_generator.py     # Organisation-level reports
│   └── dashboard/
│       └── exports/                # CSV/JSON report exports
│
├── mobile_app/                     # TIMOTHY — Android Studio project
│   ├── app/
│   │   ├── src/
│   │   │   └── main/
│   │   │       ├── java/com/sentry/app/
│   │   │       │   ├── MainActivity.kt
│   │   │       │   ├── network/
│   │   │       │   │   └── ApiClient.kt     # Retrofit calls to middleware
│   │   │       │   ├── ui/
│   │   │       │   │   ├── WelcomeScreen.kt
│   │   │       │   │   ├── ScenarioScreen.kt
│   │   │       │   │   ├── FeedbackScreen.kt
│   │   │       │   │   └── ResultsScreen.kt
│   │   │       │   ├── models/
│   │   │       │   │   ├── QueryRequest.kt
│   │   │       │   │   └── QueryResponse.kt
│   │   │       │   └── viewmodels/
│   │   │       │       └── SessionViewModel.kt
│   │   │       └── res/
│   │   │           ├── layout/
│   │   │           └── values/
│   │   └── build.gradle
│   ├── figma/
│   │   └── SENTRY_UI_Design.fig    # Figma export/link file
│   ├── build.gradle
│   └── README_MOBILE.md            # Setup + ADB deployment guide
│
├── pepper_interface/               # TIMOTHY — NAOqi layer
│   ├── scenarios/
│   │   ├── phishing_scenario.py
│   │   ├── usb_drop_scenario.py
│   │   ├── password_hygiene_scenario.py
│   │   ├── network_hygiene_scenario.py
│   │   └── social_engineering_scenario.py
│   ├── dialogue/
│   │   ├── state_machine.py
│   │   ├── dialogue_manager.py
│   │   └── response_parser.py
│   ├── pepper_client.py
│   ├── middleware_client.py        # HTTP calls to FastAPI
│   └── TIMOTHY_GUIDE.md
│
├── evaluation/                     # DERICK (done)
│   ├── metrics/
│   ├── logs/
│   └── reports/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── docs/
│   ├── architecture/
│   └── api_specs/
│
├── scripts/
│   └── ingest_knowledge_base.py
│
├── config/
│   └── settings.py
│
├── .env
├── .env.example
├── requirements.txt
└── README.md

How Everything Connects
This is the critical picture to understand:
┌─────────────────────────────────────────────────────────────────┐
│                     LOCAL WIFI NETWORK                          │
│                                                                 │
│  ┌──────────────────┐         ┌──────────────────────────────┐  │
│  │  Pepper Robot    │         │   Laptop (Dev Machine)       │  │
│  │                  │         │                              │  │
│  │  NAOqi Python2.7 │◀───────▶│  FastAPI Middleware          │  │
│  │  pepper_client.py│  HTTP   │  uvicorn port 8000           │  │
│  │                  │         │                              │  │
│  │  ┌─────────────┐ │         │  /api/v1/query    (Derick)   │  │
│  │  │Android Tab  │ │         │  /api/v1/sessions (Timothy)  │  │
│  │  │SENTRY App   │◀──────────│  /api/v1/analytics(Timothy)  │  │
│  │  │(Kotlin APK) │ │  HTTP   │                              │  │
│  │  └─────────────┘ │         │  RAG Pipeline                │  │
│  └──────────────────┘         │  ChromaDB (175 chunks)       │  │
│                               │  GPT-4 API                   │  │
│                               │                              │  │
│                               │  PostgreSQL                  │  │
│                               │  (session logs, analytics)   │  │
│                               └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
Key point: Both Pepper's NAOqi layer and the Android tablet app communicate with the same FastAPI server over WiFi. The laptop runs the server, Pepper and the tablet are clients on the same network. This is exactly how predecessor projects work.

What Each Person Builds
Derick (you) — already done + database foundation
Your AI engine is complete. You additionally set up:

backend/database/connection.py — PostgreSQL connection
backend/database/models.py — Session and analytics tables
Register Timothy's future routes in middleware/main.py

Timothy — three workstreams
Workstream 1: Android App (mobile_app/)

Design screens in Figma collaboratively
Build in Android Studio using Kotlin + Jetpack Compose
Uses Retrofit to call FastAPI endpoints over WiFi
Sideloaded onto Pepper's tablet via ADB
Displays scenario prompts, receives AI feedback, shows results

Workstream 2: Pepper NAOqi layer (pepper_interface/)

Python 2.7 scripts running on Pepper
Controls speech, gestures, tablet display
Calls middleware via HTTP for AI responses

Workstream 3: Backend routes (middleware/routes/)

session_routes.py — creates and logs sessions to PostgreSQL
analytics_routes.py — serves aggregated data

ADB Sideloading — How It Works
This is how the APK gets onto Pepper's tablet:
bash# 1. Enable developer mode on Pepper's tablet
#    Settings → About → tap Build Number 7 times

# 2. Connect laptop to Pepper's tablet via USB
adb devices
# Should show the tablet's serial number

# 3. Install the APK
adb install app/build/outputs/apk/debug/app-debug.apk

# 4. Launch it
adb shell am start -n com.sentry.app/.MainActivity

# 5. For updates, uninstall first
adb uninstall com.sentry.app
adb install app/build/outputs/apk/debug/app-debug.apk
The app then connects to FastAPI via the laptop's local IP address (e.g. http://192.168.1.100:8000). Both must be on the same WiFi network.

Network Configuration for the App
The Android app cannot use localhost — it must use your laptop's actual local IP. Add this to your .env.example:
env# Network — for Android app connection
LAPTOP_LOCAL_IP=192.168.x.x    # Your laptop's IP on the shared WiFi
The Kotlin ApiClient.kt will use this IP as its base URL:
kotlin// ApiClient.kt
object ApiClient {
    // Change this to your laptop's local IP when running
    private const val BASE_URL = "http://192.168.x.x:8000/"

    val retrofit: Retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
}
Also add this to middleware/main.py — FastAPI needs to accept connections from non-localhost:
python# Already set in your current main.py — confirm this line exists:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Android app on same network
    ...
)
Your server start command also needs to bind to all interfaces — which it already does:
bashuvicorn middleware.main:app --reload --host 0.0.0.0 --port 8000
0.0.0.0 means it accepts connections from any device on the network, not just localhost.

Android App Screens (from Figma → Kotlin)
These are the five screens Timothy builds, matching your proposal's scenario modules:
WelcomeScreen      — Employee login/session start, health check ping
ScenarioScreen     — Displays the attack scenario, captures response
LoadingScreen      — Shows while waiting for AI response (covers latency)
FeedbackScreen     — Displays Derick's grounded AI response + sources
ResultsScreen      — Post-session score, risk profile, recommendations
Each screen calls the middleware via Retrofit. The FeedbackScreen specifically displays:

response field → the spoken/displayed AI feedback
sources field → "Based on: OWASP Top 10, Computer Misuse Act" attribution
retrieved_chunks → optional expandable detail for transparency

Immediate Action Plan
Right now — you (Derick):

Create the mobile_app/ folder and mobile_app/README_MOBILE.md with the ADB deployment guide
Set up backend/database/models.py and connection.py — I will write these for you now
Add asyncpg and sqlalchemy to requirements
Update middleware/main.py to register Timothy's future routes as placeholders

Timothy (in parallel):

Install Android Studio
Create new project: File → New → New Project → Empty Activity with Kotlin
Set package name: com.sentry.app
Add Retrofit dependency to build.gradle
Design Figma screens — you both review together
Build ApiClient.kt pointing to your middleware IP

Together:

Agree on the exact JSON contract for session logging (what the app POSTs when a session starts/ends)
I will define that as a Pydantic model once you agree