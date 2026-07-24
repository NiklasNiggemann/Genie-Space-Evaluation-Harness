"""genie_eval — Evaluation harness for Databricks Genie Agents.

Public API:
    EvaluationRunner  — Run a full eval suite and get scored results.
    load_test_suite   — Load TestCase objects from a YAML file.
    ask_genie         — Submit a single question to the Conversation API.
    extract_sql       — Pull generated SQL from a Genie response.
    TestCase          — Data model for a single test case.
    EvalResult        — Data model for a single evaluation result.
    EvalSuiteResults  — Aggregate results with .accuracy and .to_delta().
"""

from .api import ask_genie, extract_result, extract_sql, extract_text_response
from .judge import create_sql_judge
from .models import EvalResult, EvalSuiteResults, TestCase
from .runner import EvaluationRunner, load_test_suite

__all__ = [
    "ask_genie",
    "create_sql_judge",
    "EvalResult",
    "EvalSuiteResults",
    "EvaluationRunner",
    "extract_result",
    "extract_sql",
    "extract_text_response",
    "load_test_suite",
    "TestCase",
]
