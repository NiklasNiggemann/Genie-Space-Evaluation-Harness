"""Genie Conversation API helpers.

Provides functions to submit questions to a Databricks Genie Agent
(formerly Genie Space) and extract structured outputs from the response.

The API still uses /genie/spaces/ in the path — the rename is UI-only.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from databricks.sdk import WorkspaceClient

# Default polling parameters
DEFAULT_POLL_INTERVAL = 3   # seconds between status checks
DEFAULT_MAX_POLLS = 60      # max polls before timeout (= 3 min)

# Retry parameters for transient network/server errors
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0   # base delay in seconds; doubles on each attempt

# Terminal statuses that indicate Genie is done processing
_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"}


def _with_retry(fn: Callable, *, max_retries: int, base_delay: float) -> Any:
    """Call fn(), retrying on exception with exponential backoff.

    All exception types are retried — permanent errors (bad space_id, wrong auth)
    will simply exhaust retries and re-raise. The cost of 2 extra attempts is
    acceptable; missing a transient failure is not.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


def ask_genie(
    space_id: str,
    question: str,
    *,
    client: WorkspaceClient | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    max_polls: int = DEFAULT_MAX_POLLS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> dict[str, Any]:
    """Submit a natural-language question to a Genie Agent and wait for the response.

    Uses the Genie Conversation API:
      1. POST /start-conversation to submit the question
      2. Poll GET /messages/{id} until status leaves IN_PROGRESS/EXECUTING_QUERY

    Both API calls are retried on transient failures (network errors, 5xx) with
    exponential backoff before raising.

    Args:
        space_id: The Genie Agent ID (from the URL or API).
        question: Natural-language question to ask.
        client: Optional WorkspaceClient instance. Created automatically if None.
        poll_interval: Seconds between status checks.
        max_polls: Maximum number of polls before timeout.
        max_retries: Retry attempts for transient API failures.
        retry_delay: Base delay in seconds for the first retry (doubles each time).

    Returns:
        dict with keys: status, message, conversation_id, message_id, elapsed_seconds
    """
    if client is None:
        client = WorkspaceClient()

    start_time = time.time()

    # Step 1: Start a new conversation
    response = _with_retry(
        lambda: client.api_client.do(
            method="POST",
            path=f"/api/2.0/genie/spaces/{space_id}/start-conversation",
            body={"content": question},
        ),
        max_retries=max_retries,
        base_delay=retry_delay,
    )

    conversation_id = response["conversation_id"]
    message_id = response["message_id"]

    # Step 2: Poll until the message reaches a terminal status
    msg_response: dict[str, Any] = {}
    for _ in range(max_polls):
        msg_response = _with_retry(
            lambda: client.api_client.do(
                method="GET",
                path=(
                    f"/api/2.0/genie/spaces/{space_id}"
                    f"/conversations/{conversation_id}/messages/{message_id}"
                ),
            ),
            max_retries=max_retries,
            base_delay=retry_delay,
        )

        status = msg_response.get("status", "UNKNOWN")
        if status in _TERMINAL_STATUSES:
            elapsed = time.time() - start_time
            return {
                "status": status,
                "message": msg_response,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "elapsed_seconds": round(elapsed, 2),
            }

        time.sleep(poll_interval)

    # Timeout reached
    elapsed = time.time() - start_time
    return {
        "status": "TIMEOUT",
        "message": msg_response,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "elapsed_seconds": round(elapsed, 2),
    }


def extract_sql(message: dict[str, Any]) -> str:
    """Extract the generated SQL from a Genie message response.

    Genie returns SQL in attachments[].query.query.

    Returns:
        The generated SQL string, or empty string if not found.
    """
    msg_data = message.get("message", message)
    attachments = msg_data.get("attachments", []) or []

    for attachment in attachments:
        if "query" in attachment:
            query_info = attachment["query"]
            if "query" in query_info:
                return query_info["query"]
            if "sql" in query_info:
                return query_info["sql"]

    return ""


def extract_result(message: dict[str, Any]) -> list:
    """Extract query result rows from a Genie message response.

    Returns:
        List of result rows (each row is a dict), or empty list if no results.
    """
    msg_data = message.get("message", message)
    attachments = msg_data.get("attachments", []) or []

    for attachment in attachments:
        if "query" in attachment:
            query_info = attachment["query"]
            if "query_result" in query_info:
                result = query_info["query_result"]
                if "data" in result:
                    return result["data"]
                return result
            if "result" in query_info:
                return query_info["result"]

    return []


def extract_text_response(message: dict[str, Any]) -> str:
    """Extract Genie's narrative text response (not the user's input question).

    Genie's text reply is in attachments[].text.content, while the top-level
    'content' field is the user's original question.
    """
    msg_data = message.get("message", message)
    attachments = msg_data.get("attachments", []) or []

    for attachment in attachments:
        if "text" in attachment:
            text_info = attachment["text"]
            if "content" in text_info:
                return text_info["content"]

    return ""
