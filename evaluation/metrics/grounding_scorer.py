"""
SENTRY — Grounding Scorer & Session Logger
============================================
Wraps the HallucinationScorer and adds:

    - Session-level result accumulation across multiple queries
    - JSON log writing to evaluation/logs/ for each session
    - Summary report generation with aggregate statistics
    - pandas DataFrame export for statistical analysis (t-test etc.)

This is what runs during the evaluation study (Phase 5).
Each participant's session produces one log file.
Aggregate analysis across all sessions produces the final report.

Used by: evaluation/run_evaluation.py (Phase 5)
         tests/unit/test_grounding_scorer.py
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from evaluation.metrics.hallucination_scorer import HallucinationScorer
from config.settings import settings


class GroundingScorer:
    """
    Session-level evaluation manager.

    Usage:
        scorer = GroundingScorer(participant_id="P001", condition="grounded")
        scorer.record(grounded_result, baseline_result)
        scorer.record(grounded_result_2, baseline_result_2)
        report = scorer.generate_report()
        scorer.save_session_log()
    """

    def __init__(
        self,
        participant_id: str,
        condition: str,
        scenario_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            participant_id: Anonymised participant identifier (e.g. "P001").
            condition:      "grounded" or "baseline" — which condition
                            this participant is in for the study.
            scenario_ids:   Optional list of scenario IDs for this session.
        """
        if condition not in ("grounded", "baseline"):
            raise ValueError(
                "condition must be 'grounded' or 'baseline'."
            )

        self.participant_id = participant_id
        self.condition = condition
        self.scenario_ids = scenario_ids or []
        self.session_id = str(uuid.uuid4())[:8]
        self.started_at = datetime.utcnow().isoformat()

        self._scorer = HallucinationScorer()
        self._records: List[Dict[str, Any]] = []

        # Ensure log directory exists
        Path(settings.EVAL_LOG_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.EVAL_REPORT_DIR).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        grounded_result: Dict[str, Any],
        baseline_result: Dict[str, Any],
        scenario_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Score one query pair and store the result.

        Args:
            grounded_result: Output from pipeline.query_grounded()
            baseline_result: Output from pipeline.query_baseline()
            scenario_id:     Optional scenario label for this query.

        Returns:
            The comparison result dict for this query pair.
        """
        comparison = self._scorer.score_pair(
            grounded_result=grounded_result,
            baseline_result=baseline_result,
        )
        comparison["scenario_id"] = scenario_id
        comparison["timestamp"] = datetime.utcnow().isoformat()

        self._records.append(comparison)
        return comparison

    def generate_report(self) -> Dict[str, Any]:
        """
        Aggregate all recorded query pairs into a session summary.

        Returns:
            Report dict with mean scores across all queries,
            suitable for inclusion in the evaluation study results.
        """
        if not self._records:
            return {"error": "No records to report."}

        grounding_accuracies = [
            r["grounded"]["grounding_accuracy"] for r in self._records
        ]
        hallucination_rates_grounded = [
            r["grounded"]["hallucination_rate"] for r in self._records
        ]
        hallucination_rates_baseline = [
            r["baseline"]["hallucination_rate"] for r in self._records
        ]
        improvements = [
            r["grounding_improvement"] for r in self._records
        ]
        latency_costs = [
            r["latency_cost_ms"] for r in self._records
        ]

        def mean(lst):
            return round(sum(lst) / len(lst), 4) if lst else 0.0

        report = {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "condition": self.condition,
            "started_at": self.started_at,
            "completed_at": datetime.utcnow().isoformat(),
            "total_queries": len(self._records),
            "aggregate": {
                "mean_grounding_accuracy": mean(grounding_accuracies),
                "mean_hallucination_rate_grounded": mean(
                    hallucination_rates_grounded
                ),
                "mean_hallucination_rate_baseline": mean(
                    hallucination_rates_baseline
                ),
                "mean_grounding_improvement": mean(improvements),
                "mean_latency_cost_ms": mean(latency_costs),
            },
            "per_query": self._records,
        }

        return report

    def save_session_log(self) -> Path:
        """
        Write the full session log to evaluation/logs/.
        Filename includes participant ID and session ID for traceability.
        """
        report = self.generate_report()
        filename = (
            f"session_{self.participant_id}_{self.session_id}.json"
        )
        log_path = Path(settings.EVAL_LOG_DIR) / filename

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"[GroundingScorer] Session log saved: {log_path}")
        return log_path

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export per-query results as a pandas DataFrame.
        Used for statistical analysis (t-test, effect size calculation).

        Columns:
            participant_id, condition, scenario_id,
            grounding_accuracy, hallucination_rate_grounded,
            hallucination_rate_baseline, grounding_improvement,
            latency_cost_ms
        """
        rows = []
        for r in self._records:
            rows.append({
                "participant_id": self.participant_id,
                "condition": self.condition,
                "scenario_id": r.get("scenario_id", ""),
                "query": r["query"],
                "grounding_accuracy": r["grounded"]["grounding_accuracy"],
                "hallucination_rate_grounded": r["grounded"]["hallucination_rate"],
                "hallucination_rate_baseline": r["baseline"]["hallucination_rate"],
                "grounding_improvement": r["grounding_improvement"],
                "hallucination_reduction": r["hallucination_reduction"],
                "latency_cost_ms": r["latency_cost_ms"],
            })

        return pd.DataFrame(rows)