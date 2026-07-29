"""LLM judge for semantic SQL correctness.

Uses MLflow's `make_judge` to create a boolean scorer that evaluates whether
generated SQL is semantically equivalent to expected SQL — i.e., would produce
the same result set regardless of syntactic differences.
"""

from __future__ import annotations

# ---- Judge Prompt ----
# Versioned here so changes are reviewable and testable.

SQL_JUDGE_INSTRUCTIONS = """
You are an expert SQL evaluator for Databricks SQL. Determine whether the
generated SQL in {{ outputs }} is semantically equivalent to the expected SQL
in {{ expectations }} — meaning they would produce the same result set.

Two SQL statements are semantically equivalent if they:
1. Query the same tables (possibly with different aliases)
2. Apply the same filters (possibly in different order or syntax)
3. Produce the same columns with the same aggregation logic
4. Return the same rows (ordering differences are acceptable unless LIMIT is used)

Acceptable differences (still rate True):
- Different column aliases, join syntax, whitespace, equivalent date functions

Must match (rate False if different):
- Different tables, different aggregation functions (SUM vs COUNT),
  missing/extra filter conditions, different GROUP BY columns, different LIMIT values

The original question asked was: {{ inputs }}

Rate True if the generated SQL would produce equivalent results to the expected SQL.
Rate False if they would produce meaningfully different results.
"""

# Default model endpoint for the judge LLM
DEFAULT_JUDGE_MODEL = "databricks:/databricks-claude-sonnet-4-6"


def create_sql_judge(
    model: str = DEFAULT_JUDGE_MODEL,
    instructions: str = SQL_JUDGE_INSTRUCTIONS,
):
    """Create an MLflow judge scorer for semantic SQL correctness.

    Args:
        model: Model serving endpoint for the judge LLM.
        instructions: Jinja2-templated instructions for the judge.

    Returns:
        An MLflow scorer object usable with `mlflow.genai.evaluate()`.
    """
    from mlflow.genai import make_judge  # lazy — only available on Databricks

    return make_judge(
        name="sql_semantic_correctness",
        instructions=instructions,
        model=model,
        feedback_value_type=bool,
    )
