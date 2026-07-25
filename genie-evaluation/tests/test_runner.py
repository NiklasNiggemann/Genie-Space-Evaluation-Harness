"""Tests for genie_eval.runner and genie_eval.models aggregate behaviour."""

import logging

import pandas as pd
import pytest

from genie_eval.models import EvalResult, EvalSuiteResults


def _result(judge_correct=None, status="COMPLETED", generated_sql="SELECT 1", expected_sql="SELECT 1"):
    return EvalResult(
        question="q",
        category="c",
        difficulty="d",
        status=status,
        generated_sql=generated_sql,
        expected_sql=expected_sql,
        judge_correct=judge_correct,
    )


class TestEvalSuiteResultsAccuracy:
    def test_empty_results_returns_zero(self):
        assert EvalSuiteResults().accuracy == 0.0

    def test_all_correct(self):
        suite = EvalSuiteResults(results=[_result(True), _result(True)])
        assert suite.accuracy == 1.0

    def test_all_incorrect(self):
        suite = EvalSuiteResults(results=[_result(False), _result(False)])
        assert suite.accuracy == 0.0

    def test_unjudged_none_counts_as_incorrect(self):
        # A text-only response has judge_correct=None — treated as wrong
        suite = EvalSuiteResults(results=[_result(True), _result(None)])
        assert suite.accuracy == pytest.approx(0.5)

    def test_mixed_true_false_none(self):
        suite = EvalSuiteResults(results=[_result(True), _result(False), _result(None)])
        assert suite.accuracy == pytest.approx(1 / 3)

    def test_completion_rate_counts_only_completed(self):
        suite = EvalSuiteResults(results=[
            _result(status="COMPLETED"),
            _result(status="FAILED"),
            _result(status="ERROR"),
        ])
        assert suite.completion_rate == pytest.approx(1 / 3)


class TestVerdictMerge:
    """Verify that the verdict merge in _run_judge handles column naming issues gracefully."""

    def _make_runner(self, space_id="space-1"):
        from unittest.mock import MagicMock, patch
        from genie_eval.runner import EvaluationRunner

        with patch("genie_eval.runner.WorkspaceClient"):
            runner = EvaluationRunner(space_id=space_id, verbose=False)
        return runner

    def test_missing_verdict_column_logs_warning(self, caplog):
        import sys
        import types

        import mlflow

        from unittest.mock import MagicMock, patch
        from genie_eval.runner import EvaluationRunner

        with patch("genie_eval.runner.WorkspaceClient"):
            runner = EvaluationRunner(space_id="space-1", verbose=False)

        results = [_result()]

        fake_judge_results = MagicMock()
        fake_judge_results.tables = {
            "eval_results": pd.DataFrame({"some_other_column": [True]})
        }
        fake_judge_results.metrics = {}

        fake_genai = types.ModuleType("mlflow.genai")
        fake_genai.evaluate = MagicMock(return_value=fake_judge_results)

        with patch("genie_eval.runner.create_sql_judge"), \
             patch.dict(sys.modules, {"mlflow.genai": fake_genai}), \
             patch.object(mlflow, "genai", fake_genai, create=True), \
             patch("mlflow.start_run"), \
             patch("mlflow.log_param"), \
             patch("mlflow.log_metric"), \
             caplog.at_level(logging.WARNING, logger="genie_eval.runner"):
            suite = runner._run_judge(results)

        assert any("missing expected column" in r.message for r in caplog.records)
        assert results[0].judge_correct is None

    def test_space_id_set_even_when_no_judgeable_cases(self):
        from unittest.mock import patch
        from genie_eval.runner import EvaluationRunner

        with patch("genie_eval.runner.WorkspaceClient"):
            runner = EvaluationRunner(space_id="my-space", verbose=False)

        # No generated SQL → nothing to judge
        results = [_result(generated_sql="", expected_sql="SELECT 1")]
        suite = runner._run_judge(results)

        assert suite.space_id == "my-space"
