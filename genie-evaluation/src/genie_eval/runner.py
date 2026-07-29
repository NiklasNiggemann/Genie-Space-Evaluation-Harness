"""Evaluation runner — orchestrates the full eval loop.

Coordinates: loading test cases, calling the Genie API, running the LLM judge,
and assembling EvalSuiteResults for downstream reporting/storage.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient

from .analysis import compare_result_sets, execute_sql
from .api import ask_genie, extract_result, extract_sql, extract_text_response
from .judge import DEFAULT_JUDGE_MODEL, create_sql_judge
from .models import EvalResult, EvalSuiteResults, TestCase

logger = logging.getLogger(__name__)


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
        runner = EvaluationRunner(space_id="...", max_workers=8)
        results = runner.run(test_cases, categories=["join"])
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
        max_workers: int = 1,
        warehouse_id: str | None = None,
        verbose: bool = True,
    ):
        self.space_id = space_id
        self.client = client or WorkspaceClient()
        self.experiment_name = experiment_name
        self.judge_model = judge_model
        self.accuracy_threshold = accuracy_threshold
        self.max_workers = max_workers
        self.warehouse_id = warehouse_id
        self.verbose = verbose

    def run(
        self,
        test_cases: list[TestCase],
        *,
        categories: list[str] | None = None,
        difficulties: list[str] | None = None,
    ) -> EvalSuiteResults:
        """Execute the full evaluation loop.

        1. Optionally filter test cases by category and/or difficulty
        2. Ask each question to the Genie Agent (in parallel if max_workers > 1)
        3. Score with the LLM judge
        4. Log to MLflow
        5. Return aggregated results

        Args:
            test_cases: Full list of test cases to evaluate.
            categories: If set, only run cases whose category is in this list.
            difficulties: If set, only run cases whose difficulty is in this list.
        """
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        filtered = [
            tc for tc in test_cases
            if (categories is None or tc.category in categories)
            and (difficulties is None or tc.difficulty in difficulties)
        ]

        if self.verbose:
            skipped = len(test_cases) - len(filtered)
            print(f"\n{'='*70}")
            print(f" GENIE AGENT EVALUATION RUN")
            print(f" Space: {self.space_id}")
            print(f" Test cases: {len(filtered)}" + (f"  ({skipped} filtered out)" if skipped else ""))
            if self.max_workers > 1:
                print(f" Workers: {self.max_workers} (parallel)")
            print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}\n")

        # ---- Phase 1: Ask Genie ----
        n = len(filtered)

        if self.max_workers == 1:
            eval_results = [
                self._evaluate_single(i, n, tc)
                for i, tc in enumerate(filtered, 1)
            ]
        else:
            eval_results: list[EvalResult | None] = [None] * n
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._evaluate_single, i, n, tc): i - 1
                    for i, tc in enumerate(filtered, 1)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    eval_results[idx] = future.result()

        # ---- Phase 2: LLM Judge Scoring ----
        suite_results = self._run_judge(eval_results)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f" EVALUATION COMPLETE — Accuracy: {suite_results.accuracy:.0%}")
            print(f"{'='*70}")

        return suite_results

    def _evaluate_single(self, i: int, n: int, tc: TestCase) -> EvalResult:
        """Evaluate one test case. Safe to call from multiple threads."""
        if self.verbose:
            print(f"[{i}/{n}] ▶ {tc.question}")

        try:
            result = ask_genie(self.space_id, tc.question, client=self.client)
            generated_sql = extract_sql(result)
            query_results = extract_result(result)
            text_response = extract_text_response(result)
            status = result["status"]
            elapsed = result["elapsed_seconds"]

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

            # Optionally compare actual result sets (requires warehouse_id)
            if self.warehouse_id and record.generated_sql and record.expected_sql:
                try:
                    expected_rows = execute_sql(
                        record.expected_sql,
                        client=self.client,
                        warehouse_id=self.warehouse_id,
                    )
                    record.result_set_correct = compare_result_sets(query_results, expected_rows)
                except Exception as e:
                    logger.debug("Result-set comparison failed for %r: %s", tc.question, e)

            if self.verbose:
                if status == "COMPLETED" and generated_sql:
                    print(f"       ✓ Completed in {elapsed}s | SQL generated")
                elif status == "COMPLETED":
                    print(f"       ✓ Completed in {elapsed}s | Text response")
                else:
                    print(f"       ✗ {status} after {elapsed}s")

        except Exception as e:
            logger.debug("Exception evaluating test case %r", tc.question, exc_info=True)
            record = EvalResult(
                question=tc.question,
                category=tc.category,
                difficulty=tc.difficulty,
                status="ERROR",
                text_response=str(e)[:200],
            )
            if self.verbose:
                print(f"       ✗ ERROR: {str(e)[:80]}")

        return record

    def _run_judge(self, eval_results: list[EvalResult]) -> EvalSuiteResults:
        """Run the LLM judge on results that have both generated + expected SQL."""
        judgeable = [r for r in eval_results if r.generated_sql and r.expected_sql]

        if not judgeable:
            return EvalSuiteResults(results=eval_results, space_id=self.space_id)

        judge = create_sql_judge(model=self.judge_model or DEFAULT_JUDGE_MODEL)

        eval_data = pd.DataFrame(
            {
                "inputs": [{"question": r.question} for r in judgeable],
                "outputs": [{"generated_sql": r.generated_sql} for r in judgeable],
                "expectations": [{"expected_sql": r.expected_sql} for r in judgeable],
            }
        )

        run_name = f"genie-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("space_id", self.space_id)
            mlflow.log_param("num_test_cases", len(eval_results))
            mlflow.log_param("num_evaluated", len(judgeable))

            judge_results = mlflow.genai.evaluate(  # type: ignore[attr-defined]  # Databricks-only
                data=eval_data,
                scorers=[judge],
            )

            if hasattr(judge_results, "metrics") and judge_results.metrics:
                for name, value in judge_results.metrics.items():
                    mlflow.log_metric(name, value)

            # Merge verdicts back into eval_results
            if hasattr(judge_results, "tables") and "eval_results" in judge_results.tables:
                judge_df = judge_results.tables["eval_results"]
                col = "sql_semantic_correctness/value"
                if col not in judge_df.columns:
                    logger.warning(
                        "Judge results missing expected column %r; verdicts not applied. "
                        "Available columns: %s",
                        col,
                        list(judge_df.columns),
                    )
                else:
                    verdicts = judge_df[col].tolist()
                    judgeable_idx = 0
                    for r in eval_results:
                        if r.generated_sql and r.expected_sql:
                            r.judge_correct = bool(verdicts[judgeable_idx])
                            judgeable_idx += 1

            # Log per-question results so compare_runs can diff across runs
            per_q_df = pd.DataFrame([
                {
                    "question": r.question,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "status": r.status,
                    "judge_correct": r.judge_correct,
                    "execution_time_seconds": r.execution_time_seconds,
                }
                for r in eval_results
            ])
            mlflow.log_table(data=per_q_df, artifact_file="eval_results.json")

        return EvalSuiteResults(
            results=eval_results,
            space_id=self.space_id,
            run_id=run.info.run_id,
        )


class MultiSpaceRunner:
    """Run the same evaluation suite against multiple Genie Spaces in parallel.

    Useful for comparing dev vs prod, or A/B testing ontology changes before
    promoting them. All spaces run concurrently; questions within each space
    can also be parallelised via max_workers_per_space.

    Usage:
        runner = MultiSpaceRunner(
            space_ids={"dev": "space-dev-id", "prod": "space-prod-id"},
            max_workers_per_space=4,
        )
        results = runner.run(test_cases)
        # results = {"dev": EvalSuiteResults, "prod": EvalSuiteResults}

        from genie_eval import compare_spaces
        df = compare_spaces(results)
        assert results["dev"].accuracy >= results["prod"].accuracy, "Dev regressed vs prod"
    """

    def __init__(
        self,
        space_ids: dict[str, str],
        *,
        client: WorkspaceClient | None = None,
        experiment_name: str | None = None,
        judge_model: str | None = None,
        accuracy_threshold: float = 0.75,
        max_workers_per_space: int = 1,
        warehouse_id: str | None = None,
        verbose: bool = True,
    ):
        self.space_ids = space_ids
        self._runner_kwargs: dict = {
            "client": client,
            "experiment_name": experiment_name,
            "judge_model": judge_model,
            "accuracy_threshold": accuracy_threshold,
            "max_workers": max_workers_per_space,
            "warehouse_id": warehouse_id,
            "verbose": verbose,
        }

    def run(
        self,
        test_cases: list[TestCase],
        *,
        categories: list[str] | None = None,
        difficulties: list[str] | None = None,
    ) -> dict[str, EvalSuiteResults]:
        """Run the same test suite against all spaces in parallel.

        Args:
            test_cases: Test cases to run against every space.
            categories: Optional category filter applied to all spaces.
            difficulties: Optional difficulty filter applied to all spaces.

        Returns:
            Dict mapping space name → EvalSuiteResults.
        """
        def _run_one(item: tuple[str, str]) -> tuple[str, EvalSuiteResults]:
            name, space_id = item
            runner = EvaluationRunner(space_id=space_id, **self._runner_kwargs)
            return name, runner.run(test_cases, categories=categories, difficulties=difficulties)

        with ThreadPoolExecutor(max_workers=len(self.space_ids)) as executor:
            pairs = list(executor.map(_run_one, self.space_ids.items()))

        return dict(pairs)
