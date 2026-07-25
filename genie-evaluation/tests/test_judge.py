"""Tests for genie_eval.judge — judge prompt and creation.

These are regression tests to ensure the judge prompt doesn't
accidentally change in ways that break scoring consistency.
"""

import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from genie_eval.judge import (
    DEFAULT_JUDGE_MODEL,
    SQL_JUDGE_INSTRUCTIONS,
    create_sql_judge,
)


@contextmanager
def _fake_mlflow_genai():
    """Inject a fake mlflow.genai module so create_sql_judge can be called locally."""
    fake_genai = types.ModuleType("mlflow.genai")
    fake_genai.make_judge = MagicMock(return_value=MagicMock(name="scorer"))
    with patch.dict(sys.modules, {"mlflow.genai": fake_genai}):
        yield fake_genai


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
        with _fake_mlflow_genai() as genai:
            judge = create_sql_judge()
        assert judge is not None
        genai.make_judge.assert_called_once_with(
            name="sql_semantic_correctness",
            instructions=SQL_JUDGE_INSTRUCTIONS,
            model=DEFAULT_JUDGE_MODEL,
            feedback_value_type=bool,
        )

    def test_creates_scorer_with_custom_model(self):
        custom_model = "databricks:/databricks-meta-llama-3-3-70b-instruct"
        with _fake_mlflow_genai() as genai:
            create_sql_judge(model=custom_model)
        genai.make_judge.assert_called_once_with(
            name="sql_semantic_correctness",
            instructions=SQL_JUDGE_INSTRUCTIONS,
            model=custom_model,
            feedback_value_type=bool,
        )
