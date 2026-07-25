"""Unit tests for genie_eval.api — extraction functions.

These tests use mocked API responses (no network calls) to verify
that SQL, results, and text are correctly extracted from the Genie
response structure.
"""

from unittest.mock import MagicMock

from genie_eval.api import ask_genie, extract_result, extract_sql, extract_text_response


class TestAskGenie:
    """Tests for ask_genie() — uses a mocked WorkspaceClient (no network)."""

    def _mock_client(self, *poll_responses):
        """Build a mock client where the first do() call is start-conversation
        and subsequent calls are message poll responses."""
        client = MagicMock()
        client.api_client.do.side_effect = [
            {"conversation_id": "conv-1", "message_id": "msg-1"},
            *poll_responses,
        ]
        return client

    def test_returns_completed_on_first_poll(self):
        client = self._mock_client({"status": "COMPLETED", "attachments": []})
        result = ask_genie("space-1", "What is revenue?", client=client, poll_interval=0)
        assert result["status"] == "COMPLETED"
        assert result["conversation_id"] == "conv-1"
        assert result["message_id"] == "msg-1"

    def test_polls_until_terminal(self):
        client = self._mock_client(
            {"status": "EXECUTING_QUERY"},
            {"status": "EXECUTING_QUERY"},
            {"status": "COMPLETED", "attachments": []},
        )
        result = ask_genie("space-1", "test?", client=client, poll_interval=0)
        assert result["status"] == "COMPLETED"
        assert client.api_client.do.call_count == 4  # 1 start + 3 polls

    def test_returns_timeout_when_max_polls_exceeded(self):
        client = self._mock_client(*[{"status": "EXECUTING_QUERY"}] * 3)
        result = ask_genie("space-1", "test?", client=client, poll_interval=0, max_polls=3)
        assert result["status"] == "TIMEOUT"

    def test_returns_failed_status_as_terminal(self):
        client = self._mock_client({"status": "FAILED"})
        result = ask_genie("space-1", "bad query", client=client, poll_interval=0)
        assert result["status"] == "FAILED"

    def test_elapsed_seconds_is_float(self):
        client = self._mock_client({"status": "COMPLETED", "attachments": []})
        result = ask_genie("space-1", "test?", client=client, poll_interval=0)
        assert isinstance(result["elapsed_seconds"], float)


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
