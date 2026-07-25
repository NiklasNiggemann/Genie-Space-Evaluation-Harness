"""Data models for the Genie Evaluation Harness.

Defines typed dataclasses for test cases, evaluation results, and suite-level
metrics. These provide a shared schema across the runner, judge, and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestCase:
    """A single evaluation test case.

    Attributes:
        question: The natural-language question to send to Genie.
        expected_sql: The ground-truth SQL that answers the question.
        category: Grouping label (e.g. 'aggregation', 'join', 'filter').
        difficulty: Complexity level ('easy', 'medium', 'hard').
        expected_result_contains: Optional substring expected in the result.
    """

    question: str
    expected_sql: str
    category: str = "general"
    difficulty: str = "medium"
    expected_result_contains: Optional[str] = None


@dataclass
class EvalResult:
    """The result of evaluating a single test case.

    Populated by the evaluation runner after asking Genie and (optionally)
    running the LLM judge and result-set comparison.
    """

    question: str
    category: str
    difficulty: str
    status: str  # COMPLETED, FAILED, ERROR, TIMEOUT
    generated_sql: str = ""
    expected_sql: str = ""
    text_response: str = ""
    result_preview: str = "[]"
    result_contains_expected: Optional[bool] = None
    result_set_correct: Optional[bool] = None  # set when warehouse_id is provided
    execution_time_seconds: float = 0.0
    conversation_id: str = ""
    message_id: str = ""
    judge_correct: Optional[bool] = None


@dataclass
class EvalSuiteResults:
    """Aggregate results for a complete evaluation run.

    Provides convenience properties for accuracy, completion rate,
    and Delta table export, plus breakdown methods for failure analysis.
    """

    results: list[EvalResult] = field(default_factory=list)
    space_id: str = ""
    run_id: str = ""

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def completed(self) -> int:
        return sum(1 for r in self.results if r.status == "COMPLETED")

    @property
    def accuracy(self) -> float:
        """Fraction of test cases judged semantically correct.

        Cases where judge_correct is None (no SQL generated, or unjudged) count
        as incorrect — a non-answer is treated the same as a wrong answer.
        """
        if not self.results:
            return 0.0
        correct = sum(1 for r in self.results if r.judge_correct is True)
        return correct / len(self.results)

    @property
    def completion_rate(self) -> float:
        """Fraction of test cases that received a response (any status)."""
        if not self.results:
            return 0.0
        return self.completed / len(self.results)

    def report(self) -> "pd.DataFrame":
        """Return a per-question breakdown as a DataFrame.

        Columns: question, category, difficulty, status, judge_correct,
                 result_set_correct, execution_time_s, generated_sql, expected_sql
        """
        import pandas as pd

        return pd.DataFrame([
            {
                "question": r.question,
                "category": r.category,
                "difficulty": r.difficulty,
                "status": r.status,
                "judge_correct": r.judge_correct,
                "result_set_correct": r.result_set_correct,
                "execution_time_s": r.execution_time_seconds,
                "generated_sql": r.generated_sql,
                "expected_sql": r.expected_sql,
            }
            for r in self.results
        ])

    def summary_by_category(self) -> "pd.DataFrame":
        """Accuracy and completion rate grouped by category."""
        return self._grouped_summary("category")

    def summary_by_difficulty(self) -> "pd.DataFrame":
        """Accuracy and completion rate grouped by difficulty."""
        return self._grouped_summary("difficulty")

    def _grouped_summary(self, group_col: str) -> "pd.DataFrame":
        import pandas as pd

        df = self.report()
        return (
            df.groupby(group_col, sort=False)
            .agg(
                total=("question", "count"),
                completed=("status", lambda s: (s == "COMPLETED").sum()),
                correct=("judge_correct", lambda x: x.eq(True).sum()),
            )
            .assign(
                accuracy=lambda d: d["correct"] / d["total"],
                completion_rate=lambda d: d["completed"] / d["total"],
            )
            .reset_index()
        )

    def to_delta(self, table_name: str) -> None:
        """Write results to a Delta table for historical tracking.

        Args:
            table_name: Fully qualified table name (catalog.schema.table).
        """
        import pandas as pd
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("No active SparkSession — run inside Databricks.")

        records = [
            {
                "question": r.question,
                "category": r.category,
                "difficulty": r.difficulty,
                "status": r.status,
                "generated_sql": r.generated_sql,
                "expected_sql": r.expected_sql,
                "judge_correct": r.judge_correct,
                "result_set_correct": r.result_set_correct,
                "execution_time_seconds": r.execution_time_seconds,
                "space_id": self.space_id,
                "run_id": self.run_id,
            }
            for r in self.results
        ]

        df = spark.createDataFrame(pd.DataFrame(records))
        df.write.mode("append").saveAsTable(table_name)
