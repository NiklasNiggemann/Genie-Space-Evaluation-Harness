"""Evaluation runner — orchestrates the full eval loop.

Coordinates: loading test cases, calling the Genie API, running the LLM judge,
and assembling EvalSuiteResults for downstream reporting/storage.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient

from .api import ask_genie, extract_result, extract_sql, extract_text_response
from .judge import create_sql_judge
from .models import EvalResult, EvalSuiteResults, TestCase


def load_test_suite(path: str | Path) -> list[TestCase]:
    """Load test cases from a YAML file.

    Expected format:
        - question: "..."
          expected_sql: "..."
          category: "..."
          difficulty: "..."
          expected_result_contains: null  # optional

    Args:
        path: Path to the YAML file (relative or absolute).

    Returns:
        List of TestCase objects.
    """
    import yaml  # lazy import — only needed when loading from file

    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    return [
        TestCase(
            question=item["question"],
            expected_sql=item["expected_sql"],
            category=item.get("category", "general"),
            difficulty=item.get("difficulty", "medium"),
            expected_result_contains=item.get("expected_result_contains"),
        )
        for item in raw
    ]


class EvaluationRunner:
    """Run a complete evaluation suite against a Genie Agent.

    Usage:
        runner = EvaluationRunner(space_id="...")
        results = runner.run(test_cases)
        results.to_delta("catalog.schema.genie_eval_results")
    """

    def __init__(
        self,
        space_id: str,
        *,
        client: WorkspaceClient | None = None,
        experiment_name: str | None = None,
        judge_model: str | None = None,
        accuracy_threshold: float = 0.75,
        verbose: bool = True,
    ):
        self.space_id = space_id
        self.client = client or WorkspaceClient()
        self.experiment_name = experiment_name
        self.judge_model = judge_model
        self.accuracy_threshold = accuracy_threshold
        self.verbose = verbose

    def run(self, test_cases: list[TestCase]) -> EvalSuiteResults:
        """Execute the full evaluation loop.

        1. Ask each question to the Genie Agent
        2. Score with the LLM judge
        3. Log to MLflow
        4. Return aggregated results
        """
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f" GENIE AGENT EVALUATION RUN")
            print(f" Space: {self.space_id}")
            print(f" Test cases: {len(test_cases)}")
            print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}\n")

        # ---- Phase 1: Ask Genie ----
        eval_results: list[EvalResult] = []

        for i, tc in enumerate(test_cases, 1):
            if self.verbose:
                print(f"[{i}/{len(test_cases)}] \u25b6 {tc.question}")

            try:
                result = ask_genie(
                    self.space_id, tc.question, client=self.client
                )
                generated_sql = extract_sql(result)
                query_results = extract_result(result)
                text_response = extract_text_response(result)
                status = result["status"]
                elapsed = result["elapsed_seconds"]

                # Check expected result substring
                result_match = None
                if tc.expected_result_contains:
                    result_str = json.dumps(query_results)
                    result_match = tc.expected_result_contains in result_str

                record = EvalResult(
                    question=tc.question,
                    category=tc.category,
                    difficulty=tc.difficulty,
                    status=status,
                    generated_sql=generated_sql.strip() if generated_sql else "",
                    expected_sql=tc.expected_sql.strip(),
                    text_response=text_response[:200] if text_response else "",
                    result_preview=json.dumps(query_results[:3]) if query_results else "[]",
                    result_contains_expected=result_match,
                    execution_time_seconds=elapsed,
                    conversation_id=result["conversation_id"],
                    message_id=result["message_id"],
                )

                if self.verbose:
                    if status == "COMPLETED" and generated_sql:
                        print(f"       \u2713 Completed in {elapsed}s | SQL generated")
                    elif status == "COMPLETED":
                        print(f"       \u2713 Completed in {elapsed}s | Text response")
                    else:
                        print(f"       \u2717 {status} after {elapsed}s")

            except Exception as e:
                record = EvalResult(
                    question=tc.question,
                    category=tc.category,
                    difficulty=tc.difficulty,
                    status="ERROR",
                    text_response=str(e)[:200],
                )
                if self.verbose:
                    print(f"       \u2717 ERROR: {str(e)[:80]}")

            eval_results.append(record)

        # ---- Phase 2: LLM Judge Scoring ----
        suite_results = self._run_judge(eval_results)
        suite_results.space_id = self.space_id

        if self.verbose:
            print(f"\n{'='*70}")
            print(f" EVALUATION COMPLETE \u2014 Accuracy: {suite_results.accuracy:.0%}")
            print(f"{'='*70}")

        return suite_results

    def _run_judge(self, eval_results: list[EvalResult]) -> EvalSuiteResults:
        """Run the LLM judge on results that have both generated + expected SQL."""
        # Filter to cases with both generated and expected SQL
        judgeable = [
            r for r in eval_results if r.generated_sql and r.expected_sql
        ]

        if not judgeable:
            # No SQL pairs to judge — mark all as None
            return EvalSuiteResults(results=eval_results)

        judge = create_sql_judge(
            model=self.judge_model or "databricks:/databricks-claude-sonnet-4"
        )

        eval_data = pd.DataFrame(
            {
                "inputs": [{"question": r.question} for r in judgeable],
                "outputs": [r.generated_sql for r in judgeable],
                "expectations": [
                    {"expected_sql": r.expected_sql} for r in judgeable
                ],
            }
        )

        run_name = f"genie-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("space_id", self.space_id)
            mlflow.log_param("num_test_cases", len(eval_results))
            mlflow.log_param("num_evaluated", len(judgeable))

            judge_results = mlflow.genai.evaluate(
                data=eval_data,
                scorers=[judge],
            )

            if hasattr(judge_results, "metrics") and judge_results.metrics:
                for name, value in judge_results.metrics.items():
                    mlflow.log_metric(name, value)

        # Merge verdicts back into eval_results
        if hasattr(judge_results, "tables") and "eval_results" in judge_results.tables:
            judge_df = judge_results.tables["eval_results"]
            verdicts = judge_df["sql_semantic_correctness/value"].tolist()

            judgeable_idx = 0
            for r in eval_results:
                if r.generated_sql and r.expected_sql:
                    r.judge_correct = bool(verdicts[judgeable_idx])
                    judgeable_idx += 1

        return EvalSuiteResults(
            results=eval_results,
            space_id=self.space_id,
            run_id=run.info.run_id,
        )
