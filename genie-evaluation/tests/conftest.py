"""Shared pytest fixtures for genie_eval tests."""

import pytest


# ---- Sample API Responses (mocked) ----

@pytest.fixture
def completed_response_with_sql():
    """A realistic Genie API response with SQL in the attachment."""
    return {
        "status": "COMPLETED",
        "message": {
            "status": "COMPLETED",
            "attachments": [
                {
                    "query": {
                        "query": "SELECT SUM(totalPrice) AS total_revenue FROM samples.bakehouse.sales_transactions",
                        "description": "Total revenue across all transactions",
                    }
                }
            ],
        },
        "conversation_id": "conv-123",
        "message_id": "msg-456",
        "elapsed_seconds": 12.5,
    }


@pytest.fixture
def completed_response_text_only():
    """A Genie response with a text reply (no SQL generated)."""
    return {
        "status": "COMPLETED",
        "message": {
            "status": "COMPLETED",
            "attachments": [
                {
                    "text": {
                        "content": "I cannot determine the answer from the available data."
                    }
                }
            ],
        },
        "conversation_id": "conv-789",
        "message_id": "msg-012",
        "elapsed_seconds": 5.3,
    }


@pytest.fixture
def failed_response():
    """A Genie response indicating failure."""
    return {
        "status": "FAILED",
        "message": {"status": "FAILED", "attachments": []},
        "conversation_id": "conv-fail",
        "message_id": "msg-fail",
        "elapsed_seconds": 2.1,
    }


@pytest.fixture
def sample_test_cases():
    """A minimal set of test cases for unit testing."""
    from genie_eval.models import TestCase

    return [
        TestCase(
            question="What is the total revenue?",
            expected_sql="SELECT SUM(totalPrice) AS total_revenue FROM samples.bakehouse.sales_transactions",
            category="aggregation",
            difficulty="easy",
        ),
        TestCase(
            question="How many transactions were paid with Visa?",
            expected_sql="SELECT COUNT(*) FROM samples.bakehouse.sales_transactions WHERE paymentMethod = 'visa'",
            category="filter",
            difficulty="easy",
            expected_result_contains="1083",
        ),
    ]
