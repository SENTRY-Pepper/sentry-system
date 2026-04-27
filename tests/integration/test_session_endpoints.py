"""
SENTRY — Session & Analytics Endpoints Integration Test
Run: python tests/integration/test_session_endpoints.py
"""

import sys
import asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import httpx

BASE_URL = "http://localhost:8000"


async def test_full_session_flow():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:

        print("=== Test 1: Health Check ===")
        r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["pipeline_ready"] is True
        print(f"  Status: {data['status']}")
        print(f"  Pipeline ready: {data['pipeline_ready']}")
        print(">>> PASSED")

        print("\n=== Test 2: Start Session ===")
        r = await client.post("/api/v1/sessions/start", json={
            "participant_id": "TEST_P001",
            "condition": "grounded",
            "organisation_id": "TEST_ORG",
        })
        assert r.status_code == 200
        session = r.json()
        session_id = session["session_id"]
        print(f"  Session ID:     {session_id}")
        print(f"  Participant:    {session['participant_id']}")
        print(f"  Condition:      {session['condition']}")
        print(">>> PASSED")

        print("\n=== Test 3: Log Scenario Interaction ===")
        r = await client.post("/api/v1/sessions/interaction", json={
            "session_id": session_id,
            "scenario_id": "phishing-01",
            "scenario_type": "phishing",
            "decision": "correct",
            "employee_response": "I would report it to IT immediately",
            "response_time_ms": 4200,
            "correction_loops": 0,
            "ai_latency_ms": 6800.0,
            "ai_sources": "Computer-Misuse-and-Cybercrimes-Act.pdf",
        })
        assert r.status_code == 200
        interaction = r.json()
        print(f"  Interaction ID: {interaction['interaction_id']}")
        print(f"  Decision:       {interaction['decision']}")
        print(f"  Logged:         {interaction['logged']}")
        print(">>> PASSED")

        print("\n=== Test 4: Log Evaluation Record ===")
        r = await client.post("/api/v1/sessions/eval-log", json={
            "session_id": session_id,
            "scenario_id": "phishing-01",
            "query": "What is phishing?",
            "mode": "grounded",
            "response": "Phishing is a cybercrime under the Computer Misuse Act...",
            "grounding_accuracy": 0.75,
            "hallucination_rate": 0.25,
            "grounding_improvement": 0.75,
            "retrieval_ms": 350.0,
            "generation_ms": 6200.0,
            "total_ms": 6550.0,
            "prompt_tokens": 400,
            "completion_tokens": 180,
            "sources": "Computer-Misuse-and-Cybercrimes-Act.pdf",
        })
        assert r.status_code == 200
        print(f"  Logged: {r.json()['logged']}")
        print(">>> PASSED")

        print("\n=== Test 5: End Session ===")
        r = await client.post("/api/v1/sessions/end", json={
            "session_id": session_id,
            "pre_assessment_score": 45.0,
            "post_assessment_score": 72.0,
            "duration_seconds": 920,
        })
        assert r.status_code == 200
        ended = r.json()
        print(f"  Knowledge gain:      {ended['knowledge_gain']}")
        print(f"  Relative improvement:{ended['relative_improvement_pct']}%")
        print(f"  Is complete:         {ended['is_complete']}")
        print(">>> PASSED")

        print("\n=== Test 6: Get Session Summary ===")
        r = await client.get(f"/api/v1/sessions/{session_id}")
        assert r.status_code == 200
        summary = r.json()
        print(f"  Session ID:     {summary['session_id']}")
        print(f"  Is complete:    {summary['is_complete']}")
        print(f"  Knowledge gain: {summary['knowledge_gain']}")
        print(">>> PASSED")

        print("\n=== Test 7: Study Analytics ===")
        r = await client.get("/api/v1/analytics/study")
        assert r.status_code == 200
        analytics = r.json()
        print(f"  Total sessions:              {analytics['total_sessions']}")
        print(f"  Grounded sessions:           {analytics['grounded_sessions']}")
        print(f"  Mean grounding accuracy:     {analytics['mean_grounding_accuracy_grounded']}")
        print(f"  Mean hallucination (grounded):{analytics['mean_hallucination_rate_grounded']}")
        print(f"  Mean improvement:            {analytics['mean_grounding_improvement']}")
        print(">>> PASSED")

        print("\n=== Test 8: Organisation Analytics ===")
        r = await client.get("/api/v1/analytics/organisation/TEST_ORG")
        assert r.status_code == 200
        org = r.json()
        print(f"  Organisation:       {org['organisation_id']}")
        print(f"  Total sessions:     {org['total_sessions']}")
        print(f"  Completed:          {org['completed_sessions']}")
        print(f"  Mean knowledge gain:{org['mean_knowledge_gain']}")
        print(">>> PASSED")

        print("\n" + "=" * 60)
        print("All session & analytics endpoint tests PASSED")
        print("=" * 60)


if __name__ == "__main__":
    print("Make sure the middleware server is running:")
    print("uvicorn middleware.main:app --reload --host 0.0.0.0 --port 8000")
    print()
    asyncio.run(test_full_session_flow())