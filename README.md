# Genie Space Evaluation Harness

A programmatic evaluation framework for testing the reliability of **Databricks Genie Spaces** — the natural-language "Ask Your Data" interface.

---

## Motivation

When deploying an AI-powered data assistant, the central engineering challenge is **guaranteeing reliability**. A chatbot that occasionally produces wrong answers erodes trust faster than having no chatbot at all. The specific danger is that wrong answers arrive with the same formatting and apparent authority as correct ones — there is no signal, no confidence indicator, no way for the user to assess plausibility. The evaluation harness exists to make that silent failure detectable before it reaches the user.

This harness implements the *evaluation* leg of the [Data Flywheel](https://www.sh-reya.com/blog/ai-engineering-flywheel/):

> Evaluate → Identify failure patterns → Improve instructions → Re-evaluate → Ship with confidence

---

## What It Does

| Capability | Description |
|---|---|
| **API-driven question execution** | Programmatically submits natural-language questions to a Genie Space via the Conversation API |
| **Result comparison against ground truth** | Compares generated SQL and query results to expected answers |
| **MLflow-based scoring** | Uses an LLM judge to assess *semantic* SQL correctness (not just exact-match) |
| **Data Flywheel iteration tracking** | Measures improvement over time as instructions and examples are refined |

---

## Architecture Overview

```
┌──────────────────┐        ┌─────────────────┐        ┌──────────────────┐
│  Test Suite      │───────▶│  Genie Space    │───────▶│  Result          │
│  (test_cases[])  │  ask   │  (Conversation  │  poll  │  Collection      │
│                  │        │   API)          │        │  (eval_records)  │
└──────────────────┘        └─────────────────┘        └────────┬─────────┘
                                                                │
                                                                ▼
                                                       ┌──────────────────┐
                                                       │  LLM Judge       │
                                                       │  (MLflow +       │
                                                       │   Claude Sonnet) │
                                                       └────────┬─────────┘
                                                                │
                                                                ▼
                                                       ┌──────────────────┐
                                                       │  MLflow          │
                                                       │  Experiment      │
                                                       │  (tracking)      │
                                                       └──────────────────┘
```

---

## Notebook Structure

| # | Cell | Purpose |
|---|------|----------|
| 1 | **Install Dependencies** | Installs `mlflow` and restarts the Python kernel |
| 2 | **Setup & Imports** | Imports SDK, MLflow, pandas; initializes `WorkspaceClient` |
| 3 | **Configuration** | Sets the Genie Space ID, polling parameters, and MLflow experiment path |
| 4 | **Genie API Helpers** | Core functions: `ask_genie()`, `extract_sql()`, `extract_result()`, `extract_text_response()` |
| 5 | **Test Suite Definition** | Benchmark test cases with expected SQL and result assertions |
| 6 | **Run Evaluation** | Main loop — submits each question, collects responses, builds `eval_results` DataFrame |
| 7 | **MLflow Scoring** | Custom LLM judge evaluates semantic SQL equivalence; logs results to MLflow |
| 8 | **Results Summary** | Aggregates metrics by category/difficulty, prints recommendations |
| 9 | **Data Flywheel — Next Steps** | Documents the iterative improvement process |

---

## How the Evaluation Works

### 1. Question Submission

Each test case is submitted via the REST API:

```
POST /api/2.0/genie/spaces/{space_id}/start-conversation
```

The harness polls the message endpoint until a terminal status is reached (`COMPLETED`, `FAILED`, `CANCELLED`, or timeout after 3 minutes).

### 2. Scoring

Two scoring mechanisms:

1. **Exact result matching** — checks if `expected_result_contains` value appears in the output
2. **Semantic SQL judge** — an LLM (Claude Sonnet via Databricks Foundation Models) evaluates whether the generated SQL is *semantically equivalent* to the expected SQL

### 3. MLflow Tracking

All evaluation runs are logged to an MLflow experiment, enabling time-series comparison of accuracy across iterations.

---

## Test Case Format

```python
{
    "question": "How many transactions were paid with Visa?",
    "expected_sql": """
        SELECT COUNT(*) AS visa_transactions
        FROM samples.bakehouse.sales_transactions
        WHERE paymentMethod = 'visa'
    """,
    "expected_result_contains": "1083",
    "category": "filter",
    "difficulty": "easy"
}
```

Categories: `aggregation`, `filter`, `join`, `time_filter`, `ambiguous`

---

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `SPACE_ID` | Your Genie Space ID (from the URL) | Bakehouse Analytics demo |
| `POLL_INTERVAL_SECONDS` | Seconds between status polls | `3` |
| `MAX_POLL_ATTEMPTS` | Max polls before timeout | `60` (= 3 min) |
| `EXPERIMENT_NAME` | MLflow experiment path | `/Users/<you>/genie-eval-experiment` |

---

## Prerequisites

- **Databricks workspace** with access to Genie Spaces
- **Compute**: Any cluster or serverless compute (no GPU required)
- **Foundation Model**: Access to `databricks-claude-sonnet-4` (for the LLM judge)

---

## The Data Flywheel — Continuous Improvement

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  1. EVALUATE    │ ──▶ │  2. IDENTIFY     │ ──▶ │  3. IMPROVE     │
│  Run harness    │     │  Find patterns   │     │  Add instructions│
│  Score results  │     │  in failures     │     │  & example SQL   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        ▲                                               │
        │     ┌──────────────────┐                      │
        └──── │  4. RE-EVALUATE  │ ◀────────────────────┘
              │  Measure delta   │
              │  Track in MLflow │
              └──────────────────┘
```

### Improvement levers in the Genie Space:

| Lever | When to use |
|-------|-------------|
| **Text instructions** | Genie misinterprets business jargon or applies wrong logic |
| **Example SQL** | Genie picks wrong tables or uses wrong join patterns |
| **Column configurations** | Genie can't filter correctly on categorical columns |
| **Join specifications** | Genie doesn't know how tables relate |

---

## Key Insight

> A Genie Space without evaluation is a **hope-driven deployment**.
> A Genie Space with a harness is an **engineering-driven deployment**.

The Data Flywheel guarantees that every failure becomes a learning opportunity, accuracy is measured (not assumed), and stakeholders receive quantitative confidence metrics rather than just demos.

---

## References

- [Databricks Genie Spaces Documentation](https://docs.databricks.com/en/genie/index.html)
- [Genie Conversation API](https://docs.databricks.com/api/workspace/genie)
- [MLflow GenAI Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [The AI Engineering Flywheel](https://www.sh-reya.com/blog/ai-engineering-flywheel/) — Shreya Shankar
