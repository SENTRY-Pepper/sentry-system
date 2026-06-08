# -*- coding: utf-8 -*-
"""
SENTRY - Middleware HTTP Client
===============================
Python 2.7/3.x-compatible HTTP wrapper for the FastAPI middleware.
"""

import json

try:
    import urllib.request as urllib_request
    import urllib.error as urllib_error
except ImportError:  # Python 2.7
    import urllib2 as urllib_request
    import urllib2 as urllib_error


class MiddlewareClient(object):
    """
    Thin HTTP wrapper around SENTRY's FastAPI middleware.
    """

    def __init__(self, base_url="http://localhost:8000", timeout=30):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _post(self, path, payload):
        url = "%s%s" % (self._base_url, path)
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib_request.urlopen(req, timeout=self._timeout)
            try:
                return json.loads(resp.read().decode("utf-8"))
            finally:
                resp.close()
        except urllib_error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise RuntimeError("HTTP %s from %s: %s" % (e.code, path, body))
        except urllib_error.URLError as e:
            raise RuntimeError(
                "Cannot reach middleware at %s. Is the server running? Error: %s"
                % (self._base_url, getattr(e, "reason", e))
            )

    def _get(self, path):
        url = "%s%s" % (self._base_url, path)
        req = urllib_request.Request(url)
        try:
            resp = urllib_request.urlopen(req, timeout=10)
            try:
                return json.loads(resp.read().decode("utf-8"))
            finally:
                resp.close()
        except urllib_error.URLError as e:
            raise RuntimeError("Cannot reach %s: %s" % (url, getattr(e, "reason", e)))

    def health_check(self):
        return self._get("/health")

    def is_ready(self):
        try:
            data = self.health_check()
            return data.get("pipeline_ready", False)
        except Exception:
            return False

    def knowledge_base_status(self):
        return self._get("/api/v1/knowledge-base/status")

    def grounded_query(self, query, scenario_id=None):
        return self._post(
            "/api/v1/query",
            {
                "query": query,
                "scenario_id": scenario_id,
            },
        )

    def baseline_query(self, query, scenario_id=None):
        return self._post(
            "/api/v1/query/baseline",
            {
                "query": query,
                "scenario_id": scenario_id,
            },
        )

    def start_session(
        self,
        participant_id,
        condition,
        organisation_id="SENTRY_STUDY",
    ):
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
        session_id,
        pre_score,
        post_score,
        duration_seconds,
    ):
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
        session_id,
        scenario_id,
        scenario_type,
        decision,
        employee_response,
        response_time_ms,
        correction_loops,
        ai_latency_ms,
        ai_sources,
    ):
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
        session_id,
        scenario_id,
        query,
        mode,
        response,
        grounding_accuracy,
        hallucination_rate,
        grounding_improvement,
        retrieval_ms,
        generation_ms,
        total_ms,
        prompt_tokens,
        completion_tokens,
        sources,
    ):
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

    def get_session(self, session_id):
        return self._get("/api/v1/sessions/%s" % session_id)
