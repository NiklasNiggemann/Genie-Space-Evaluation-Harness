# Genie Agent Evaluation Harness

A programmatic evaluation framework for testing the reliability of **Databricks Genie Agents** (formerly Genie Spaces) — the natural-language "Ask Your Data" interface, accessible standalone or via **Genie One**.

---

## Motivation

When deploying an AI-powered data assistant, the central engineering challenge is **guaranteeing reliability**. A chatbot that occasionally produces wrong answers erodes trust faster than having no chatbot at all.

This harness implements the *evaluation* leg of the [Data Flywheel](https://www.sh-reya.com/blog/ai-engineering-flywheel/):

> Evaluate → Identify failure patterns → Improve instructions → Re-evaluate → Ship with confidence

---

## What It Does

| Capability | Description |
|---|---|
| **API-driven question execution** | Programmatically submits natural-language questions to a Genie Agent via the Conversation API |
| **Result comparison against ground truth** | Compares generated SQL and query results to expected answers |
| **MLflow-based scoring** | Uses an LLM judge to assess *semantic* SQL correctness (not just exact-match) |
| **Data Flywheel iteration tracking** | Measures improvement over time as instructions and examples are refined |

---

## Architecture Overview

```
┌──────────────────┐        ┌─────────────────┐        ┌──────────────────┐
│  Test Suite      │───────▶│  Genie Agent    │───────▶│  Result          │
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

The notebook is organized into sequential cells:

| # | Cell | Purpose |
|---|------|----------|
| 1 | **Install Dependencies** | Installs `mlflow` and restarts the Python kernel |
| 2 | **Setup & Imports** | Imports SDK, MLflow, pandas; initializes `WorkspaceClient` |
| 3 | **Configuration** | Sets the Genie Agent ID, polling parameters, and MLflow experiment path |
| 4 | **Genie API Helpers** | Core functions: `ask_genie()`, `extract_sql()`, `extract_result()`, `extract_text_response()` |
| 5 | **Test Suite Definition** | Benchmark test cases with expected SQL and result assertions |
| 6 | **Run Evaluation** | Main loop — submits each question, collects responses, builds `eval_results` DataFrame |
| 7 | **MLflow Scoring** | Custom LLM judge evaluates semantic SQL equivalence; logs results to MLflow |
| 8 | **Results Summary** | Aggregates metrics by category/difficulty, prints recommendations |
| 9 | **Data Flywheel — Next Steps** | Documents the iterative improvement process |

---

## How the Evaluation Works

### 1. Question Submission

Each test case is submitted to the Genie Agent via the REST API (the path still uses `/genie/spaces/` — the rename is UI-only):

```
POST /api/2.0/genie/spaces/{space_id}/start-conversation
```

The harness then polls the message endpoint until a terminal status is reached (`COMPLETED`, `FAILED`, `CANCELLED`, or `TIMEOUT` after 3 minutes).

### 2. Response Extraction

From the Genie response, we extract:
- **Generated SQL** — the query Genie produced
- **Query results** — the data rows returned
- **Text response** — any narrative explanation

### 3. Scoring

Two scoring mechanisms:

1. **Exact result matching** — checks if `expected_result_contains` value appears in the output (e.g., `"1083"` for a count query)
2. **Semantic SQL judge** — an LLM (Claude Sonnet via Databricks Foundation Models) evaluates whether the generated SQL is *semantically equivalent* to the expected SQL, even if syntactically different

The judge considers:
- Same tables queried
- Same filter logic applied
- Same aggregation semantics
- Same result set produced

### 4. MLflow Tracking

All evaluation runs are logged to an MLflow experiment, enabling:
- Time-series comparison of accuracy across iterations
- Per-question drill-down into failures
- Quantitative confidence metrics for stakeholders

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
    "expected_result_contains": "1083",  # Optional: substring match on results
    "category": "filter",                # For grouping in analysis
    "difficulty": "easy"                  # easy | medium | hard
}
```

Categories covered in the default suite:
- `aggregation` — SUM, COUNT, GROUP BY
- `filter` — WHERE clause accuracy
- `join` — Multi-table relationships
- `time_filter` — Date/time interpretation
- `ambiguous` — Robustness to vague phrasing

---

## Configuration

Before running, set these values in the **Configuration** cell:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `SPACE_ID` | Your Genie Agent ID (from the URL; API still calls it "space") | `"01f16364..."` (Bakehouse Analytics demo) |
| `POLL_INTERVAL_SECONDS` | Seconds between status polls | `3` |
| `MAX_POLL_ATTEMPTS` | Max polls before timeout | `60` (= 3 min) |
| `EXPERIMENT_NAME` | MLflow experiment path | `/Users/<you>/genie-eval-experiment` |

---

## Prerequisites

- **Databricks workspace** with access to Genie Agents
- **Compute**: Any cluster or serverless compute (no GPU required)
- **Permissions**: Access to the target Genie Agent and the underlying tables
- **Foundation Model**: Access to `databricks-claude-sonnet-4` (for the LLM judge)

---

## Running the Harness

1. Open the notebook in your Databricks workspace
2. Update `SPACE_ID` to point to your Genie Agent
3. Customize `test_cases` for your data and expected answers
4. Run all cells sequentially
5. Review the summary output and MLflow experiment

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

### Improvement levers in the Genie Ontology:

| Lever | When to use |
|-------|-------------|
| **Text instructions** | Genie misinterprets business jargon or applies wrong logic |
| **Example SQL** | Genie picks wrong tables or uses wrong join patterns |
| **Column configurations** | Genie can't filter correctly on categorical columns |
| **Join specifications** | Genie doesn't know how tables relate |

---

## Output Metrics

The harness produces:

- **Completion rate** — % of questions that received a valid response
- **SQL generation rate** — % of questions that produced SQL (vs. text-only answers)
- **Semantic correctness** — % of generated SQL judged equivalent to expected SQL
- **Per-category breakdown** — Identifies which question types need attention
- **Per-difficulty breakdown** — Validates that complexity correlates with failure
- **Average response time** — Monitors latency

---

## Extending the Harness

**Add more test cases**: Expand `test_cases` with questions specific to your domain. Aim for 20–50 cases covering all common user questions.

**Custom judges**: Modify `SQL_JUDGE_INSTRUCTIONS` or add additional judges (e.g., for result correctness, not just SQL equivalence).

**Automation**: Schedule the notebook as a Databricks Job to run nightly or after Genie Ontology changes.

**Alerting**: Add a cell that fails the notebook (via `assert`) if accuracy drops below a threshold — useful for CI/CD integration.

---

## Key Insight

> A Genie Agent without evaluation is a **hope-driven deployment**.
> A Genie Agent with a harness is an **engineering-driven deployment**.

The Data Flywheel guarantees that every failure becomes a learning opportunity, accuracy is measured (not assumed), regressions are caught immediately, and stakeholders receive quantitative confidence metrics rather than just demos.

---

## References

- [Databricks Genie Agents Documentation](https://docs.databricks.com/en/genie/index.html)
- [Genie Conversation API](https://docs.databricks.com/api/workspace/genie)
- [MLflow GenAI Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [The AI Engineering Flywheel](https://www.sh-reya.com/blog/ai-engineering-flywheel/) — Shreya Shankar
