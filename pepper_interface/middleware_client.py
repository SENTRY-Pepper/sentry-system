"""
SENTRY — Middleware HTTP Client
=================================
Python 3.x HTTP client that calls Derick's FastAPI middleware.
Used by the dialogue manager to get AI responses and log sessions.

For the NAOqi Python 2.7 layer on Pepper itself, see pepper_client.py
which uses urllib2 compatible calls for the same endpoints.
"""

import json
import urllib.request
import urllib.error
from typing import Optional


class MiddlewareClient:
    """
    Thin HTTP wrapper around SENTRY's FastAPI middleware.
    All endpoints documented at http://localhost:8000/docs
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = 30  # seconds — GPT-4 can take up to 20s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise RuntimeError(f"HTTP {e.code} from {path}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach middleware at {self._base_url}. "
                f"Is the server running? Error: {e.reason}"
            )

    def _get(self, path: str) -> dict:
        url = f"{self._base_url}{path}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach {url}: {e.reason}")

    # ------------------------------------------------------------------
    # Health and status
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        """Ping the middleware to confirm it is alive and pipeline is ready."""
        return self._get("/health")

    def is_ready(self) -> bool:
        """Returns True if the AI pipeline is initialised and ready."""
        try:
            data = self.health_check()
            return data.get("pipeline_ready", False)
        except Exception:
            return False

    def knowledge_base_status(self) -> dict:
        """Returns vector store stats."""
        return self._get("/api/v1/knowledge-base/status")

    # ------------------------------------------------------------------
    # AI query endpoints
    # ------------------------------------------------------------------

    def grounded_query(self, query: str, scenario_id: str = None) -> dict:
        """
        POST /api/v1/query — grounded RAG response.
        This is the experimental condition.
        """
        return self._post(
            "/api/v1/query",
            {
                "query": query,
                "scenario_id": scenario_id,
            },
        )

    def baseline_query(self, query: str, scenario_id: str = None) -> dict:
        """
        POST /api/v1/query/baseline — ungrounded LLM response.
        This is the control condition for the evaluation study.
        """
        return self._post(
            "/api/v1/query/baseline",
            {
                "query": query,
                "scenario_id": scenario_id,
            },
        )

    # ------------------------------------------------------------------
    # Session management endpoints
    # ------------------------------------------------------------------

    def start_session(
        self,
        participant_id: str,
        condition: str,
        organisation_id: str = "SENTRY_STUDY",
    ) -> dict:
        """POST /api/v1/sessions/start"""
        return self._post(
            "/api/v1/sessions/start",
            {
                "participant_id": participant_id,
                "condition": condition,
                "organisation_id": organisation_id,
            },
        )

    def end_session(
        self,
        session_id: str,
        pre_score: float,
        post_score: float,
        duration_seconds: int,
    ) -> dict:
        """POST /api/v1/sessions/end"""
        return self._post(
            "/api/v1/sessions/end",
            {
                "session_id": session_id,
                "pre_assessment_score": pre_score,
                "post_assessment_score": post_score,
                "duration_seconds": duration_seconds,
            },
        )

    def log_interaction(
        self,
        session_id: str,
        scenario_id: str,
        scenario_type: str,
        decision: str,
        employee_response: str,
        response_time_ms: int,
        correction_loops: int,
        ai_latency_ms: float,
        ai_sources: str,
    ) -> dict:
        """POST /api/v1/sessions/interaction"""
        return self._post(
            "/api/v1/sessions/interaction",
            {
                "session_id": session_id,
                "scenario_id": scenario_id,
                "scenario_type": scenario_type,
                "decision": decision,
                "employee_response": employee_response,
                "response_time_ms": response_time_ms,
                "correction_loops": correction_loops,
                "ai_latency_ms": ai_latency_ms,
                "ai_sources": ai_sources,
            },
        )

    def log_eval_record(
        self,
        session_id: str,
        scenario_id: str,
        query: str,
        mode: str,
        response: str,
        grounding_accuracy: Optional[float],
        hallucination_rate: Optional[float],
        grounding_improvement: Optional[float],
        retrieval_ms: float,
        generation_ms: float,
        total_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        sources: str,
    ) -> dict:
        """POST /api/v1/sessions/eval-log"""
        return self._post(
            "/api/v1/sessions/eval-log",
            {
                "session_id": session_id,
                "scenario_id": scenario_id,
                "query": query,
                "mode": mode,
                "response": response,
                "grounding_accuracy": grounding_accuracy,
                "hallucination_rate": hallucination_rate,
                "grounding_improvement": grounding_improvement,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "sources": sources,
            },
        )

    def get_session(self, session_id: str) -> dict:
        """GET /api/v1/sessions/{session_id}"""
        return self._get(f"/api/v1/sessions/{session_id}")
