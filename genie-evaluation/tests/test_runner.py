"""Tests for genie_eval.runner and genie_eval.models aggregate behaviour."""

import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from genie_eval.models import EvalResult, EvalSuiteResults


def _result(judge_correct=None, status="COMPLETED", generated_sql="SELECT 1", expected_sql="SELECT 1", category="c", difficulty="d"):
    return EvalResult(
        question="q",
        category=category,
        difficulty=difficulty,
        status=status,
        generated_sql=generated_sql,
        expected_sql=expected_sql,
        judge_correct=judge_correct,
    )


def _make_runner(space_id="space-1", max_workers=1):
    from genie_eval.runner import EvaluationRunner
    with patch("genie_eval.runner.WorkspaceClient"):
        return EvaluationRunner(space_id=space_id, verbose=False, max_workers=max_workers)


# ---- EvalSuiteResults.accuracy ------------------------------------------- #

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


# ---- Filtering ------------------------------------------------------------ #

class TestFiltering:
    def _tc(self, question, category, difficulty="medium"):
        from genie_eval.models import TestCase
        return TestCase(question=question, expected_sql="SELECT 1", category=category, difficulty=difficulty)

    def _run_filtered(self, runner, test_cases, **kwargs):
        """Call runner.run with ask_genie mocked to return immediately."""
        fake_result = {
            "status": "COMPLETED",
            "message": {"attachments": []},
            "conversation_id": "c",
            "message_id": "m",
            "elapsed_seconds": 0.1,
        }
        with patch("genie_eval.runner.ask_genie", return_value=fake_result), \
             patch.object(runner, "_run_judge", side_effect=lambda r: EvalSuiteResults(results=r, space_id=runner.space_id)):
            return runner.run(test_cases, **kwargs)

    def test_no_filter_runs_all(self):
        runner = _make_runner()
        cases = [self._tc("q1", "aggregation"), self._tc("q2", "join")]
        results = self._run_filtered(runner, cases)
        assert results.total == 2

    def test_filter_by_category(self):
        runner = _make_runner()
        cases = [self._tc("q1", "aggregation"), self._tc("q2", "join"), self._tc("q3", "filter")]
        results = self._run_filtered(runner, cases, categories=["join"])
        assert results.total == 1
        assert results.results[0].category == "join"

    def test_filter_by_difficulty(self):
        runner = _make_runner()
        cases = [self._tc("q1", "agg", "easy"), self._tc("q2", "agg", "hard")]
        results = self._run_filtered(runner, cases, difficulties=["easy"])
        assert results.total == 1
        assert results.results[0].difficulty == "easy"

    def test_filter_by_category_and_difficulty(self):
        runner = _make_runner()
        cases = [
            self._tc("q1", "join", "easy"),
            self._tc("q2", "join", "hard"),
            self._tc("q3", "filter", "easy"),
        ]
        results = self._run_filtered(runner, cases, categories=["join"], difficulties=["easy"])
        assert results.total == 1
        assert results.results[0].question == "q1"

    def test_filter_matching_nothing_returns_empty(self):
        runner = _make_runner()
        cases = [self._tc("q1", "aggregation")]
        results = self._run_filtered(runner, cases, categories=["nonexistent"])
        assert results.total == 0


# ---- Parallel evaluation -------------------------------------------------- #

class TestParallelEval:
    def test_parallel_preserves_result_order(self):
        from genie_eval.models import TestCase

        questions = [f"question {i}" for i in range(6)]
        test_cases = [
            TestCase(question=q, expected_sql="SELECT 1", category="c", difficulty="d")
            for q in questions
        ]

        call_order = []

        def fake_ask_genie(space_id, question, *, client, **kwargs):
            call_order.append(question)
            return {
                "status": "COMPLETED",
                "message": {"attachments": [{"query": {"query": f"SELECT '{question}'"}}]},
                "conversation_id": "c",
                "message_id": "m",
                "elapsed_seconds": 0.01,
            }

        runner = _make_runner(max_workers=3)
        with patch("genie_eval.runner.ask_genie", side_effect=fake_ask_genie), \
             patch.object(runner, "_run_judge", side_effect=lambda r: EvalSuiteResults(results=r, space_id=runner.space_id)):
            results = runner.run(test_cases)

        assert results.total == len(questions)
        result_questions = [r.question for r in results.results]
        assert result_questions == questions  # order preserved regardless of completion order


# ---- Verdict merge -------------------------------------------------------- #

class TestVerdictMerge:
    def test_missing_verdict_column_logs_warning(self, caplog):
        runner = _make_runner()
        results = [_result()]

        fake_judge_results = MagicMock()
        fake_judge_results.tables = {
            "eval_results": pd.DataFrame({"some_other_column": [True]})
        }
        fake_judge_results.metrics = {}

        import mlflow
        fake_genai = types.ModuleType("mlflow.genai")
        fake_genai.evaluate = MagicMock(return_value=fake_judge_results)

        with patch("genie_eval.runner.create_sql_judge"), \
             patch.dict(sys.modules, {"mlflow.genai": fake_genai}), \
             patch.object(mlflow, "genai", fake_genai, create=True), \
             patch("mlflow.start_run"), \
             patch("mlflow.log_param"), \
             patch("mlflow.log_metric"), \
             patch("mlflow.log_table"), \
             caplog.at_level(logging.WARNING, logger="genie_eval.runner"):
            runner._run_judge(results)

        assert any("missing expected column" in r.message for r in caplog.records)
        assert results[0].judge_correct is None

    def test_space_id_set_even_when_no_judgeable_cases(self):
        runner = _make_runner(space_id="my-space")
        results = [_result(generated_sql="", expected_sql="SELECT 1")]
        suite = runner._run_judge(results)
        assert suite.space_id == "my-space"


# ---- compare_runs --------------------------------------------------------- #

class TestCompareRuns:
    def _make_df(self, rows):
        return pd.DataFrame(rows, columns=["question", "category", "difficulty", "judge_correct"])

    def _mock_load_table(self, df_a, df_b):
        call_count = [0]
        dfs = [df_a, df_b]

        def fake_load(artifact_file, run_ids):
            idx = call_count[0]
            call_count[0] += 1
            return dfs[idx]

        return fake_load

    def test_improvement(self):
        from genie_eval.analysis import compare_runs

        df_a = self._make_df([("What is revenue?", "agg", "easy", False)])
        df_b = self._make_df([("What is revenue?", "agg", "easy", True)])

        with patch("mlflow.load_table", side_effect=self._mock_load_table(df_a, df_b)):
            result = compare_runs("run-a", "run-b")

        assert result.iloc[0]["change"] == "improvement"

    def test_regression(self):
        from genie_eval.analysis import compare_runs

        df_a = self._make_df([("Count by product?", "agg", "easy", True)])
        df_b = self._make_df([("Count by product?", "agg", "easy", False)])

        with patch("mlflow.load_table", side_effect=self._mock_load_table(df_a, df_b)):
            result = compare_runs("run-a", "run-b")

        assert result.iloc[0]["change"] == "regression"

    def test_stable_pass(self):
        from genie_eval.analysis import compare_runs

        df_a = self._make_df([("Revenue?", "agg", "easy", True)])
        df_b = self._make_df([("Revenue?", "agg", "easy", True)])

        with patch("mlflow.load_table", side_effect=self._mock_load_table(df_a, df_b)):
            result = compare_runs("run-a", "run-b")

        assert result.iloc[0]["change"] == "stable_pass"

    def test_stable_fail(self):
        from genie_eval.analysis import compare_runs

        df_a = self._make_df([("Revenue?", "agg", "easy", False)])
        df_b = self._make_df([("Revenue?", "agg", "easy", False)])

        with patch("mlflow.load_table", side_effect=self._mock_load_table(df_a, df_b)):
            result = compare_runs("run-a", "run-b")

        assert result.iloc[0]["change"] == "stable_fail"

    def test_not_judged_when_none(self):
        from genie_eval.analysis import compare_runs

        df_a = self._make_df([("Revenue?", "agg", "easy", None)])
        df_b = self._make_df([("Revenue?", "agg", "easy", True)])

        with patch("mlflow.load_table", side_effect=self._mock_load_table(df_a, df_b)):
            result = compare_runs("run-a", "run-b")

        assert result.iloc[0]["change"] == "not_judged"

    def test_output_columns(self):
        from genie_eval.analysis import compare_runs

        df_a = self._make_df([("q?", "join", "hard", True)])
        df_b = self._make_df([("q?", "join", "hard", True)])

        with patch("mlflow.load_table", side_effect=self._mock_load_table(df_a, df_b)):
            result = compare_runs("run-a", "run-b")

        assert list(result.columns) == ["question", "category", "difficulty", "judge_correct_a", "judge_correct_b", "change"]
