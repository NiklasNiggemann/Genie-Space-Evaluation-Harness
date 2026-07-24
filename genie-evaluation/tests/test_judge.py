"""Tests for genie_eval.judge — judge prompt and creation.

These are regression tests to ensure the judge prompt doesn't
accidentally change in ways that break scoring consistency.
"""

from genie_eval.judge import (
    DEFAULT_JUDGE_MODEL,
    SQL_JUDGE_INSTRUCTIONS,
    create_sql_judge,
)


class TestJudgePrompt:
    """Regression tests for the SQL judge prompt."""

    def test_prompt_contains_semantic_equivalence_criteria(self):
        assert "semantically equivalent" in SQL_JUDGE_INSTRUCTIONS

    def test_prompt_references_template_variables(self):
        """The prompt must contain the MLflow template vars."""
        assert "{{ outputs }}" in SQL_JUDGE_INSTRUCTIONS
        assert "{{ expectations }}" in SQL_JUDGE_INSTRUCTIONS
        assert "{{ inputs }}" in SQL_JUDGE_INSTRUCTIONS

    def test_prompt_lists_acceptable_differences(self):
        assert "column aliases" in SQL_JUDGE_INSTRUCTIONS

    def test_prompt_lists_must_match_criteria(self):
        assert "Different tables" in SQL_JUDGE_INSTRUCTIONS
        assert "GROUP BY" in SQL_JUDGE_INSTRUCTIONS

    def test_default_model_is_set(self):
        assert "databricks" in DEFAULT_JUDGE_MODEL


class TestCreateSqlJudge:
    """Tests for judge creation (no LLM call)."""

    def test_creates_scorer_with_default_args(self):
        judge = create_sql_judge()
        # The returned object should be a callable scorer
        assert judge is not None
        assert hasattr(judge, "name") or callable(judge)

    def test_creates_scorer_with_custom_model(self):
        judge = create_sql_judge(model="databricks:/databricks-meta-llama-3-3-70b-instruct")
        assert judge is not None
