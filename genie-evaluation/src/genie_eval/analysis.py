"""Cross-run analysis utilities for the Genie evaluation harness."""

from __future__ import annotations

import pandas as pd

_CHANGE_LABELS: dict[tuple[bool, bool], str] = {
    (True, True): "stable_pass",
    (True, False): "regression",
    (False, True): "improvement",
    (False, False): "stable_fail",
}


def compare_runs(run_id_a: str, run_id_b: str) -> pd.DataFrame:
    """Compare per-question results between two MLflow evaluation runs.

    Both runs must have been produced by EvaluationRunner, which logs
    per-question data to an 'eval_results.json' MLflow artifact.

    Args:
        run_id_a: MLflow run ID for the baseline (e.g. before an ontology change).
        run_id_b: MLflow run ID for the comparison (e.g. after the change).

    Returns:
        DataFrame with one row per question found in both runs, columns:
            question, category, difficulty,
            judge_correct_a, judge_correct_b, change
        where `change` is one of:
            'improvement'  — wrong in A, correct in B
            'regression'   — correct in A, wrong in B
            'stable_pass'  — correct in both
            'stable_fail'  — wrong in both
            'not_judged'   — at least one run has no verdict (None/NaN)
    """
    import mlflow

    df_a = mlflow.load_table("eval_results.json", run_ids=[run_id_a])
    df_b = mlflow.load_table("eval_results.json", run_ids=[run_id_b])

    cols = ["question", "category", "difficulty", "judge_correct"]
    merged = df_a[cols].merge(df_b[cols], on="question", suffixes=("_a", "_b"))

    # Keep category/difficulty from run_a; drop run_b duplicates
    merged = merged.rename(columns={"category_a": "category", "difficulty_a": "difficulty"})
    merged = merged.drop(columns=["category_b", "difficulty_b"], errors="ignore")

    merged["change"] = merged.apply(
        lambda r: _classify(r["judge_correct_a"], r["judge_correct_b"]),
        axis=1,
    )

    return merged[["question", "category", "difficulty", "judge_correct_a", "judge_correct_b", "change"]]


def _classify(a: object, b: object) -> str:
    try:
        if pd.isna(a) or pd.isna(b):
            return "not_judged"
    except (TypeError, ValueError):
        pass
    return _CHANGE_LABELS.get((bool(a), bool(b)), "not_judged")
