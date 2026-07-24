"""Unit tests for genie_eval.api — extraction functions.

These tests use mocked API responses (no network calls) to verify
that SQL, results, and text are correctly extracted from the Genie
response structure.
"""

from genie_eval.api import extract_result, extract_sql, extract_text_response


class TestExtractSql:
    """Tests for extract_sql()."""

    def test_extracts_sql_from_query_attachment(self, completed_response_with_sql):
        sql = extract_sql(completed_response_with_sql)
        assert "SUM(totalPrice)" in sql
        assert "sales_transactions" in sql

    def test_returns_empty_string_when_no_sql(self, completed_response_text_only):
        sql = extract_sql(completed_response_text_only)
        assert sql == ""

    def test_returns_empty_string_on_failed_response(self, failed_response):
        sql = extract_sql(failed_response)
        assert sql == ""

    def test_handles_nested_message_key(self):
        """extract_sql should work whether 'message' key is present or not."""
        raw = {
            "status": "COMPLETED",
            "attachments": [{"query": {"query": "SELECT 1"}}],
        }
        assert extract_sql(raw) == "SELECT 1"

    def test_fallback_to_sql_key(self):
        """Some API versions may use 'sql' instead of 'query' inside query attachment."""
        raw = {
            "message": {
                "attachments": [{"query": {"sql": "SELECT 2"}}]
            }
        }
        assert extract_sql(raw) == "SELECT 2"


class TestExtractResult:
    """Tests for extract_result()."""

    def test_extracts_from_query_result_data(self):
        msg = {
            "message": {
                "attachments": [
                    {
                        "query": {
                            "query": "SELECT 1",
                            "query_result": {
                                "data": [{"col1": "val1"}, {"col1": "val2"}]
                            },
                        }
                    }
                ]
            }
        }
        result = extract_result(msg)
        assert len(result) == 2
        assert result[0] == {"col1": "val1"}

    def test_returns_empty_list_when_no_results(self, completed_response_text_only):
        result = extract_result(completed_response_text_only)
        assert result == []


class TestExtractTextResponse:
    """Tests for extract_text_response()."""

    def test_extracts_text_content(self, completed_response_text_only):
        text = extract_text_response(completed_response_text_only)
        assert "cannot determine" in text

    def test_returns_empty_when_no_text_attachment(self, completed_response_with_sql):
        text = extract_text_response(completed_response_with_sql)
        assert text == ""
