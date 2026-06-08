# SENTRY Pepper Robot Integration Guide

## Integration Status

SENTRY can be integrated with a Pepper robot for a guided cybersecurity
training session.

Implemented robot capabilities:

- Pepper speech output through `ALTextToSpeech`.
- Pepper gestures through `ALAnimationPlayer`.
- Pepper tablet scenario and feedback display through `ALTabletService`.
- Spoken A-D answer selection through `ALSpeechRecognition`.
- Spoken commands for `question` and `repeat`.
- Grounded or baseline answers for trainee questions through the existing
  FastAPI query endpoints.
- Session start, interaction logging, and session completion through the
  existing session API.
- Simulation mode for laptop-side testing without physical Pepper hardware.

Important limitation: Pepper's built-in NAOqi speech recognition is
vocabulary-based. The current implementation is intentionally optimized for
reliable low-latency recognition of answer options and a small question
vocabulary. Fully open dictation should be added later through the Android
tablet microphone or an external speech-to-text service if the study requires
unrestricted natural-language questions.

## Runtime Architecture

```text
Pepper Robot
  pepper_interface/pepper_client.py
  NAOqi Python 2.7
  TTS, gestures, ASR, tablet display

Laptop / Server
  FastAPI middleware on port 8000
  RAG pipeline
  PostgreSQL
  ChromaDB vector store

Android App / Pepper Tablet
  Optional visual learning interface
  Backend-backed login
  Trainee, Manager, Admin flows
```

Pepper and the Android app are clients. The laptop runs the middleware and must
be reachable over the same WiFi network.

ADB is not used to deploy the Pepper robot Python client. ADB is only relevant
if you choose to install the optional Android app onto Pepper's Android tablet.
Robot speech, gestures, listening, and tablet HTML display are controlled
through NAOqi services.

## Backend Preparation

From the repository root:

```powershell
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe scripts\ingest_knowledge_base.py
uvicorn middleware.main:app --reload --host 0.0.0.0 --port 8000
```

If PowerShell activation fails because of a stray parenthesis, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Verify locally:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/api/v1/knowledge-base/status
```

Find the laptop IP address:

```powershell
ipconfig
```

Use the IPv4 address for the active WiFi adapter, for example
`192.168.1.100`.

Verify from another device on the same WiFi:

```text
http://192.168.1.100:8000/health
```

## Pepper Connection

Pepper must be on the same WiFi network as the laptop.

Run on Pepper or from a machine with the NAOqi Python SDK:

```bash
python pepper_interface/pepper_client.py \
  --ip PEPPER_IP_ADDRESS \
  --port 9559 \
  --middleware http://LAPTOP_IP_ADDRESS:8000 \
  --participant PEPPER_P001 \
  --condition grounded \
  --organisation SENTRY_STUDY \
  --response-mode voice \
  --response-timeout 12
```

For a non-interactive laptop smoke run:

```powershell
.\venv\Scripts\python.exe pepper_interface\pepper_client.py `
  --simulation `
  --middleware http://localhost:8000 `
  --participant PEPPER_SIM `
  --condition grounded `
  --organisation SENTRY_STUDY `
  --response-mode auto
```

`--response-mode auto` selects correct answers automatically and is intended
for smoke verification only. Use `--response-mode voice` for real HRI testing.

## Spoken Interaction Contract

For each scenario, Pepper asks the trainee to say:

- `option A`
- `option B`
- `option C`
- `option D`
- `question`
- `repeat`

When the trainee says `question`, Pepper prompts for one of the supported
question phrases:

- `what is phishing`
- `why is this risky`
- `what should I do`
- `what is the law`
- `how can I stay safe`
- `repeat`

The answer is routed through the current session condition:

- `grounded` uses `POST /api/v1/query`.
- `baseline` uses `POST /api/v1/query/baseline`.

## Login Details

The Pepper Python session does not require Android login. It identifies the
research participant through CLI arguments:

- Participant: `--participant PEPPER_P001`
- Organisation: `--organisation SENTRY_STUDY`
- Condition: `--condition grounded` or `baseline`

For the Android app or Pepper tablet app, the backend login endpoint creates
prototype users on first login. Use stable values during testing:

```text
Trainee
Participant ID: PEPPER_P001
PIN: 1234
Role: Trainee
Organisation: SENTRY_STUDY

Manager
Participant ID: MANAGER_001
PIN: 1234
Role: Manager
Organisation: SENTRY_STUDY

Admin
Participant ID: ADMIN_001
PIN: 1234
Role: Admin
Organisation: SENTRY_STUDY
```

Use the same PIN after first login. This authentication is prototype-grade and
is not yet suitable for production deployment.

## Latency Methodology

For smoother human-robot interaction:

- Run middleware on the same LAN as Pepper.
- Bind FastAPI to `0.0.0.0`.
- Generate `knowledge_base/vector_store/` before the session.
- Keep the laptop plugged in and avoid VPN routing during tests.
- Use `--response-timeout 12` to `15` seconds in noisy rooms.
- Keep Pepper question vocabulary compact for faster ASR.
- Pepper immediately says "I am thinking about your response" before calling
  the RAG endpoint.
- Spoken AI feedback is shortened before TTS so the robot does not read long
  paragraphs.
- Interaction logs store `ai_latency_ms` for later analysis.

Expected latency profile on a stable LAN:

```text
ASR recognition:       usually under 1-2 seconds after speech
Local scenario logic:  negligible
RAG retrieval:         usually sub-second once vector store is warm
LLM generation:        dominant variable, often several seconds
Pepper TTS:            starts immediately after response text is available
```

If live RAG latency is too high for a study session, run the fixed OWASP
assessment flow on Android for deterministic feedback and reserve Pepper RAG
questions for short prompted explanations.

## Pre-Physical Test Checklist

1. PostgreSQL is running.
2. `.env` contains valid database and OpenAI settings.
3. `scripts/ingest_knowledge_base.py` has generated the vector store.
4. Middleware starts without the ChromaDB missing-directory error.
5. `http://LAPTOP_IP:8000/health` works from another device.
6. Pepper and laptop are on the same WiFi.
7. Pepper IP is reachable.
8. Simulation smoke run succeeds with `--response-mode auto`.
9. Physical run uses `--response-mode voice`.
10. Room noise is low enough for Pepper ASR.

## Verification Commands

```powershell
.\venv\Scripts\python.exe -m compileall pepper_interface
.\venv\Scripts\python.exe -m pytest tests\unit\test_pepper_interface.py -q
.\venv\Scripts\python.exe -m pytest tests\unit -m "not live" -q
```

Physical Pepper TTS, gesture, tablet, and microphone behavior still require a
robot because NAOqi services are hardware/runtime services.
