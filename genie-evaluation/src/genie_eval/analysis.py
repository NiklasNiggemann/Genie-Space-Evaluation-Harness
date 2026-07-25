"""Cross-run analysis utilities and result-set scoring for the Genie evaluation harness."""

from __future__ import annotations

import json

import pandas as pd

_CHANGE_LABELS: dict[tuple[bool, bool], str] = {
    (True, True): "stable_pass",
    (True, False): "regression",
    (False, True): "improvement",
    (False, False): "stable_fail",
}


# ---- Result-set scoring -------------------------------------------------- #

def execute_sql(
    sql: str,
    *,
    client: object,
    warehouse_id: str,
    timeout: str = "30s",
) -> list[dict]:
    """Execute SQL on a Databricks SQL warehouse and return rows as dicts.

    Args:
        sql: The SQL statement to execute.
        client: A WorkspaceClient instance.
        warehouse_id: Databricks SQL warehouse ID.
        timeout: Server-side wait timeout (e.g. '30s', '2m').

    Returns:
        List of rows, each as a {column: value} dict.

    Raises:
        RuntimeError: If the statement fails or times out.
    """
    from databricks.sdk.service.sql import StatementState

    stmt = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout=timeout,
    )

    if stmt.status.state != StatementState.SUCCEEDED:
        error_msg = getattr(stmt.status.error, "message", str(stmt.status.state))
        raise RuntimeError(f"SQL execution failed ({stmt.status.state}): {error_msg}")

    if not stmt.result or not stmt.result.data_array:
        return []

    columns = [col.name for col in stmt.manifest.schema.columns]
    return [dict(zip(columns, row)) for row in stmt.result.data_array]


def compare_result_sets(rows_a: list[dict], rows_b: list[dict]) -> bool:
    """Compare two result sets as unordered bags of rows.

    Row ordering is ignored (acceptable unless LIMIT semantics differ).
    Column names are ignored — only values are compared — to tolerate alias
    differences between generated and expected SQL.

    Args:
        rows_a: Result rows from generated SQL (Genie's response).
        rows_b: Result rows from expected SQL (direct execution).

    Returns:
        True if both sets contain the same multiset of value tuples.
    """
    if len(rows_a) != len(rows_b):
        return False

    def row_signature(row: dict) -> str:
        return json.dumps(sorted(str(v) for v in row.values()))

    return sorted(row_signature(r) for r in rows_a) == sorted(row_signature(r) for r in rows_b)


def compare_spaces(results: dict[str, "EvalSuiteResults"]) -> pd.DataFrame:
    """Summarise accuracy across multiple Genie Spaces from a MultiSpaceRunner run.

    Args:
        results: Dict mapping space name → EvalSuiteResults, as returned by
                 MultiSpaceRunner.run().

    Returns:
        DataFrame with one row per space, sorted by accuracy descending:
            name, space_id, accuracy, completion_rate, total
    """
    return (
        pd.DataFrame([
            {
                "name": name,
                "space_id": suite.space_id,
                "accuracy": suite.accuracy,
                "completion_rate": suite.completion_rate,
                "total": suite.total,
            }
            for name, suite in results.items()
        ])
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )


# ---- Run-over-run diff --------------------------------------------------- #

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
