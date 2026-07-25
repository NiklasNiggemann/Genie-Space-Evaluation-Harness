"""genie_eval — Evaluation harness for Databricks Genie Agents.

Public API:
    EvaluationRunner      — Run a full eval suite and get scored results.
    MultiSpaceRunner      — Run the same suite against multiple Spaces in parallel.
    load_test_suite       — Load TestCase objects from a YAML file.
    ask_genie             — Submit a single question to the Conversation API.
    extract_sql           — Pull generated SQL from a Genie response.
    compare_runs          — Diff two MLflow runs, surfacing regressions/improvements.
    compare_spaces        — Summarise accuracy across multiple Spaces.
    compare_result_sets   — Compare two result-set row lists for equivalence.
    execute_sql           — Execute SQL on a Databricks SQL warehouse.
    TestCase              — Data model for a single test case.
    EvalResult            — Data model for a single evaluation result.
    EvalSuiteResults      — Aggregate results with .accuracy, .report(), .to_delta().
"""

from .analysis import compare_result_sets, compare_runs, compare_spaces, execute_sql
from .api import ask_genie, extract_result, extract_sql, extract_text_response
from .judge import create_sql_judge
from .models import EvalResult, EvalSuiteResults, TestCase
from .runner import EvaluationRunner, MultiSpaceRunner, load_test_suite

__all__ = [
    "ask_genie",
    "compare_result_sets",
    "compare_runs",
    "compare_spaces",
    "create_sql_judge",
    "EvalResult",
    "EvalSuiteResults",
    "EvaluationRunner",
    "execute_sql",
    "extract_result",
    "extract_sql",
    "extract_text_response",
    "load_test_suite",
    "MultiSpaceRunner",
    "TestCase",
]
