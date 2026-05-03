"""
SENTRY — Aggregate Results Analyser
======================================
Runs after all participant evaluation sessions are complete.
Loads all session CSVs from evaluation/reports/, combines them
into one dataset, and performs the statistical analysis required
by the evaluation study (Section 3.4 of the proposal):

    - Independent samples t-test (grounding accuracy: grounded vs baseline)
    - Independent samples t-test (hallucination rate: grounded vs baseline)
    - Cohen's d effect size calculation
    - Knowledge gain comparison across conditions
    - Full statistical report saved to evaluation/reports/

Usage:
    python evaluation/analyse_results.py

Run this once after all ~40 participant sessions are complete.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import settings


# ------------------------------------------------------------------
# Statistical helpers
# ------------------------------------------------------------------


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size between two groups.
    d = (mean1 - mean2) / pooled_std

    Interpretation:
        0.2 = small effect
        0.5 = medium effect
        0.8 = large effect
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return round(float((np.mean(group1) - np.mean(group2)) / pooled_std), 4)


def interpret_effect(d: float) -> str:
    """Return a human-readable effect size label."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def run_ttest(
    grounded: np.ndarray,
    baseline: np.ndarray,
    metric_name: str,
) -> dict:
    """
    Run an independent samples t-test and return full results.
    Uses Welch's t-test (equal_var=False) which does not assume
    equal variances — appropriate for small samples.
    """
    if len(grounded) < 2 or len(baseline) < 2:
        return {
            "metric": metric_name,
            "grounded_mean": round(float(np.mean(grounded)), 4)
            if len(grounded) > 0
            else None,
            "grounded_std": round(float(np.std(grounded, ddof=1)), 4)
            if len(grounded) > 1
            else None,
            "baseline_mean": round(float(np.mean(baseline)), 4)
            if len(baseline) > 0
            else None,
            "baseline_std": round(float(np.std(baseline, ddof=1)), 4)
            if len(baseline) > 1
            else None,
            "t_statistic": None,
            "p_value": None,
            "significant": False,
            "cohens_d": None,
            "effect_size": "insufficient data",
            "n_grounded": int(len(grounded)),
            "n_baseline": int(len(baseline)),
            "note": "Insufficient data — both conditions need at least 2 observations for t-test.",
        }

    t_stat, p_value = stats.ttest_ind(grounded, baseline, equal_var=False)
    d = cohens_d(grounded, baseline)

    return {
        "metric": metric_name,
        "grounded_mean": round(float(np.mean(grounded)), 4),
        "grounded_std": round(float(np.std(grounded, ddof=1)), 4),
        "baseline_mean": round(float(np.mean(baseline)), 4),
        "baseline_std": round(float(np.std(baseline, ddof=1)), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),  # Cast to Python bool — not numpy bool_
        "cohens_d": d,
        "effect_size": interpret_effect(d),
        "n_grounded": int(len(grounded)),
        "n_baseline": int(len(baseline)),
    }


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def load_all_sessions() -> pd.DataFrame:
    """
    Load all session CSV files from evaluation/reports/ and
    combine into one DataFrame.
    """
    report_dir = Path(settings.EVAL_REPORT_DIR)
    csv_files = sorted(report_dir.glob("session_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No session CSV files found in {report_dir}\n"
            "Run evaluation/run_evaluation.py for each participant first."
        )

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        frames.append(df)
        print(f"  Loaded: {csv_path.name} ({len(df)} rows)")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows loaded:  {len(combined)}")
    print(f"  Participants:       {combined['participant_id'].nunique()}")
    print(f"  Conditions found:   {combined['condition'].unique().tolist()}")
    return combined


# ------------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------------


def analyse(df: pd.DataFrame) -> dict:
    """
    Run the full statistical analysis on the combined dataset.
    Returns a report dict suitable for JSON serialisation.
    """
    conditions = df["condition"].unique().tolist()
    grounded_df = df[df["condition"] == "grounded"]
    baseline_df = df[df["condition"] == "baseline"]

    print(f"\n  Grounded condition rows: {len(grounded_df)}")
    print(f"  Baseline condition rows: {len(baseline_df)}")

    if len(baseline_df) == 0:
        print(
            "\n  [Warning] No baseline condition data found.\n"
            "  Descriptive statistics will be computed for grounded only.\n"
            "  T-tests require data from both conditions.\n"
            "  Run baseline participant sessions to enable full analysis."
        )

    # Extract metric arrays safely
    grounded_accuracy = (
        grounded_df["grounding_accuracy"].values
        if len(grounded_df) > 0
        else np.array([])
    )
    baseline_accuracy = (
        baseline_df["grounding_accuracy"].values
        if len(baseline_df) > 0
        else np.array([])
    )

    grounded_hallucination = (
        grounded_df["hallucination_rate_grounded"].values
        if len(grounded_df) > 0
        else np.array([])
    )
    baseline_hallucination = (
        baseline_df["hallucination_rate_baseline"].values
        if len(baseline_df) > 0
        else np.array([])
    )

    grounded_improvement = (
        grounded_df["grounding_improvement"].values
        if len(grounded_df) > 0
        else np.array([])
    )

    # Descriptive statistics
    def safe_mean(arr):
        return round(float(np.mean(arr)), 4) if len(arr) > 0 else None

    def safe_std(arr):
        return round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else None

    descriptive = {
        "grounded": {
            "n": int(len(grounded_df)),
            "mean_grounding_accuracy": safe_mean(grounded_accuracy),
            "std_grounding_accuracy": safe_std(grounded_accuracy),
            "mean_hallucination_rate": safe_mean(grounded_hallucination),
            "mean_improvement": safe_mean(grounded_improvement),
        },
        "baseline": {
            "n": int(len(baseline_df)),
            "mean_grounding_accuracy": safe_mean(baseline_accuracy),
            "std_grounding_accuracy": safe_std(baseline_accuracy),
            "mean_hallucination_rate": safe_mean(baseline_hallucination),
        },
    }

    # Hypothesis tests
    accuracy_ttest = run_ttest(
        grounded_accuracy,
        baseline_accuracy,
        "grounding_accuracy",
    )
    hallucination_ttest = run_ttest(
        grounded_hallucination,
        baseline_hallucination,
        "hallucination_rate",
    )

    return {
        "conditions_present": conditions,
        "descriptive_statistics": descriptive,
        "hypothesis_tests": {
            "grounding_accuracy": accuracy_ttest,
            "hallucination_rate": hallucination_ttest,
        },
        "total_queries_analysed": int(len(df)),
        "total_participants": int(df["participant_id"].nunique()),
    }


def print_report(report: dict) -> None:
    """Print the statistical report in a readable format."""
    print("\n" + "=" * 60)
    print("SENTRY — STATISTICAL ANALYSIS REPORT")
    print("=" * 60)

    desc = report["descriptive_statistics"]
    print(f"\n  Total queries analysed: {report['total_queries_analysed']}")
    print(f"  Total participants:     {report['total_participants']}")
    print(f"  Conditions present:     {report['conditions_present']}")

    print("\n  --- Descriptive Statistics ---")
    g = desc["grounded"]
    b = desc["baseline"]
    print(f"  Grounded (n={g['n']}):")
    print(f"    Mean grounding accuracy:  {g['mean_grounding_accuracy']}")
    print(f"    Std grounding accuracy:   {g['std_grounding_accuracy']}")
    print(f"    Mean hallucination rate:  {g['mean_hallucination_rate']}")
    print(f"    Mean improvement:         {g['mean_improvement']}")
    print(f"  Baseline (n={b['n']}):")
    print(f"    Mean grounding accuracy:  {b['mean_grounding_accuracy']}")
    print(f"    Mean hallucination rate:  {b['mean_hallucination_rate']}")

    tests = report["hypothesis_tests"]
    print("\n  --- Hypothesis Tests (Welch's t-test, α=0.05) ---")

    for key, result in tests.items():
        print(f"\n  {result['metric'].replace('_', ' ').title()}:")
        if result.get("note"):
            print(f"    Note: {result['note']}")
            print(
                f"    Grounded: M={result['grounded_mean']} (n={result['n_grounded']})"
            )
            print(f"    Baseline: n={result['n_baseline']}")
        else:
            print(
                f"    Grounded: M={result['grounded_mean']} SD={result['grounded_std']}"
            )
            print(
                f"    Baseline: M={result['baseline_mean']} SD={result['baseline_std']}"
            )
            print(f"    t = {result['t_statistic']}")
            print(f"    p = {result['p_value']}")
            print(f"    Significant (p<0.05): {result['significant']}")
            print(
                f"    Cohen's d = {result['cohens_d']} ({result['effect_size']} effect)"
            )


def main() -> None:
    print("=" * 60)
    print("SENTRY — Aggregate Results Analyser")
    print("=" * 60)
    print("\n[Load] Reading session files...")

    df = load_all_sessions()
    report = analyse(df)
    print_report(report)

    # Save report — all values are now JSON-safe Python native types
    report_path = Path(settings.EVAL_REPORT_DIR) / "final_statistical_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved:  {report_path}")

    # Save combined dataset
    csv_path = Path(settings.EVAL_REPORT_DIR) / "combined_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Dataset saved: {csv_path}")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
