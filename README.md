# Genie Agent Evaluation Harness

A production-grade evaluation framework for testing the reliability of **Databricks Genie Agents** — the natural-language "Ask Your Data" interface.

---

## Quick Start

```python
from genie_eval import EvaluationRunner, load_test_suite

suite = load_test_suite("genie-evaluation/test_suites/bakehouse_suite.yaml")
runner = EvaluationRunner(space_id="your-genie-agent-id", max_workers=8)
results = runner.run(suite)

print(f"Accuracy: {results.accuracy:.0%}")
assert results.accuracy >= 0.75, "Quality gate failed"
```

## Installation

```bash
# Development (local unit tests, no Databricks needed)
cd genie-evaluation
pip install -e ".[dev]"

# From Git (e.g. in a Databricks notebook)
pip install git+https://github.com/NiklasNiggemann/Genie-Space-Evaluation-Harness.git#subdirectory=genie-evaluation
```

### Library-first architecture

All business logic lives in the `genie_eval` Python library (`src/genie_eval/`). The Databricks notebook (`notebooks/run_evaluation.py`) is intentionally thin — it only sets parameters and calls the library. This separation means:

- **Unit tests run locally** via `pytest` without touching Databricks
- **The library is versioned** as a wheel; each deployment is a known, immutable artifact
- **The notebook is readable** as a config file, not an implementation

To deploy a new version to Databricks, build the wheel and upload it to the workspace:

```bash
pip wheel genie-evaluation/ -w /tmp/dist/ --no-deps
databricks workspace mkdirs /Workspace/Users/<you>/genie-eval
databricks workspace import /Workspace/Users/<you>/genie-eval/genie_eval-0.1.0-py3-none-any.whl \
  --file /tmp/dist/genie_eval-0.1.0-py3-none-any.whl --format RAW --overwrite
```

The notebook's first cell installs the wheel:

```python
%pip install /Workspace/Users/<you>/genie-eval/genie_eval-0.1.0-py3-none-any.whl -q
```

## Project Structure

```
.
├── genie-evaluation/
│   ├── src/genie_eval/
│   │   ├── __init__.py       # Public API exports
│   │   ├── api.py            # Genie Conversation API helpers
│   │   ├── analysis.py       # compare_runs, compare_spaces, result-set scoring
│   │   ├── judge.py          # LLM judge prompt & scorer creation
│   │   ├── models.py         # TestCase, EvalResult, EvalSuiteResults
│   │   └── runner.py         # EvaluationRunner, MultiSpaceRunner, load_test_suite
│   ├── tests/                # pytest suite (82 tests)
│   ├── test_suites/
│   │   ├── bakehouse_suite.yaml   # Full evaluation suite
│   │   └── golden_suite.yaml      # Locked regression baseline
│   ├── scripts/ci_eval.py    # GitHub Actions evaluation script
│   ├── notebooks/run_evaluation.py  # Thin orchestration notebook (scheduled)
│   ├── databricks.yml        # Declarative Automation Bundle config
│   └── pyproject.toml        # Package metadata & dependencies
├── genie-ontology/           # Ontology-as-code (instructions, SQL examples, join specs)
└── .github/workflows/genie_eval.yml  # CI trigger on ontology/test-suite PRs
```

---

## Core Concepts

### EvaluationRunner

Runs a test suite against a single Genie Agent. Handles parallel API calls, LLM judging, and MLflow tracking in one call.

```python
runner = EvaluationRunner(
    space_id="your-agent-id",
    experiment_name="/Shared/genie-eval",   # MLflow experiment
    max_workers=8,                           # parallel Genie API calls
    warehouse_id="abc123",                   # enables result-set scoring
    accuracy_threshold=0.75,
    verbose=True,
)

results = runner.run(suite)

# Filter without editing the YAML
results = runner.run(suite, categories=["join", "time_filter"])
results = runner.run(suite, difficulties=["hard"])
```

| Parameter | Default | Description |
|---|---|---|
| `space_id` | required | Genie Agent ID (from the URL) |
| `experiment_name` | `None` | MLflow experiment path |
| `judge_model` | `databricks:/databricks-claude-sonnet-4-6` | LLM endpoint for the judge |
| `accuracy_threshold` | `0.75` | Minimum accuracy before the job fails |
| `max_workers` | `1` | Parallel Genie API workers |
| `warehouse_id` | `None` | SQL warehouse for result-set scoring |
| `verbose` | `True` | Print progress during evaluation |

---

### MultiSpaceRunner

Run the same suite against multiple Spaces simultaneously — useful for dev/prod comparison or A/B testing an ontology change.

```python
from genie_eval import MultiSpaceRunner, compare_spaces

runner = MultiSpaceRunner(
    space_ids={"dev": "space-dev-id", "prod": "space-prod-id"},
    max_workers_per_space=4,
)
results = runner.run(suite)

# Gate: dev must not regress vs prod
df = compare_spaces(results)
assert results["dev"].accuracy >= results["prod"].accuracy
```

`compare_spaces(results)` returns a DataFrame with columns `name, space_id, accuracy, completion_rate, total`, sorted by accuracy descending.

`MultiSpaceRunner` accepts the same kwargs as `EvaluationRunner`, replacing `space_id` with `space_ids: dict[str, str]` and `max_workers` with `max_workers_per_space`.

---

### Failure Reporting

```python
results.report()                 # per-question DataFrame
results.summary_by_category()   # accuracy + completion rate per category
results.summary_by_difficulty()  # same, per difficulty
```

`report()` columns: `question, category, difficulty, status, judge_correct, result_set_correct, execution_time_s, generated_sql, expected_sql`

---

### Run-over-run Diff

Compare two MLflow runs to surface regressions and improvements:

```python
from genie_eval import compare_runs

df = compare_runs(run_id_before, run_id_after)
print(df[df.change == "regression"])    # what broke
print(df[df.change == "improvement"])   # what got better
# change values: improvement | regression | stable_pass | stable_fail | not_judged
```

---

### Result-set Scoring

When `warehouse_id` is provided, the runner executes the expected SQL directly against the warehouse and compares actual row values against what Genie returned. This catches cases where the LLM judge says "equivalent" but the data disagrees.

```python
runner = EvaluationRunner(space_id="...", warehouse_id="abc123")
results = runner.run(suite)
# EvalResult.result_set_correct is True/False/None for each case
```

Column names are ignored during comparison — only values are compared — so alias differences between generated and expected SQL don't cause false negatives.

---

### Golden Suite

`test_suites/golden_suite.yaml` is a small locked set of unambiguous questions that must **always** pass. The CI pipeline enforces 100% accuracy on it before evaluating the full suite.

Modify it only via a deliberate, reviewed PR — never during an active ontology iteration.

---

## Writing Test Cases

```yaml
- question: "What is the total revenue?"
  expected_sql: |
    SELECT SUM(totalPrice) AS total_revenue
    FROM samples.bakehouse.sales_transactions
  category: aggregation
  difficulty: easy
  expected_result_contains: null   # optional substring check on the result
```

`category` and `difficulty` are free-form strings. Common categories: `aggregation`, `filter`, `join`, `time_filter`, `ambiguous`. Common difficulties: `easy`, `medium`, `hard`.

---

## Running Tests

```bash
cd genie-evaluation
pip install -e ".[dev]"
pytest
```

---

## The Data Flywheel

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  1. EVALUATE    │ ──▶ │  2. IDENTIFY     │ ──▶ │  3. IMPROVE     │
│  Run harness    │     │  compare_runs()  │     │  Add instructions│
│  Score results  │     │  regression diff │     │  & example SQL   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        ▲                                               │
        │     ┌──────────────────┐                      │
        └──── │  4. RE-EVALUATE  │ ◀────────────────────┘
              │  Measure delta   │
              │  Track in MLflow │
              └──────────────────┘
```

| Lever | When to use |
|-------|-------------|
| **Text instructions** | Genie misinterprets business jargon or applies wrong logic |
| **Example SQL** | Genie picks wrong tables or join patterns |
| **Column configurations** | Genie can't filter correctly on categorical columns |
| **Join specifications** | Genie doesn't know how tables relate |
| **Prompt matching** | Genie can't resolve user terms to column values (e.g. "Florida" → `FL`) |
| **Trusted assets** | Business-critical metrics that must never be approximated |

---

## Deployment (DAB)

The scheduled nightly job is managed via a Databricks Asset Bundle (`genie-evaluation/databricks.yml`). Deploy it with the CLI:

```bash
# Deploy job config to dev
databricks bundle deploy --target dev

# Trigger a one-off run immediately
databricks bundle run genie_eval_nightly --target dev

# Deploy to prod
databricks bundle deploy --target prod
```

> **Note:** Deploy the wheel to the workspace first (see Installation above) before running the job, so the notebook's `%pip install` can resolve it.

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/genie_eval.yml`) triggers automatically on PRs that change `genie-ontology/**` or `genie-evaluation/test_suites/**`. It:

1. Runs the **golden suite** at 100% threshold
2. Runs the **full suite** at `ACCURACY_THRESHOLD` (default 75%)
3. Posts a per-category accuracy breakdown as a PR comment
4. Fails the check if either gate fails

Required secrets:

| Secret | Description |
|--------|-------------|
| `DATABRICKS_HOST` | Workspace URL |
| `DATABRICKS_TOKEN` | PAT or service principal token |
| `GENIE_SPACE_ID` | Genie Agent ID to evaluate |

Optional env vars: `ACCURACY_THRESHOLD`, `GOLDEN_THRESHOLD`, `WAREHOUSE_ID`, `MAX_WORKERS`.

---

## Architecture

```
┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Test Suite      │────▶│  Genie Agent    │────▶│  EvalResult      │
│  (YAML)          │ ask │  (Conversation  │ poll│  Collection      │
└──────────────────┘     │   API)          │     └────────┬─────────┘
                         └─────────────────┘              │
                                               ┌──────────┴───────────┐
                                               ▼                      ▼
                                      ┌──────────────┐     ┌──────────────────┐
                                      │  LLM Judge   │     │  Result-set      │
                                      │  (MLflow +   │     │  Comparison      │
                                      │   Claude)    │     │  (SQL warehouse) │
                                      └──────┬───────┘     └──────────────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │  MLflow          │
                                    │  Experiment      │
                                    │  + compare_runs  │
                                    └──────────────────┘
```

---

## References

- [Databricks Genie Agents Documentation](https://docs.databricks.com/en/genie/index.html)
- [Genie Conversation API](https://docs.databricks.com/api/workspace/genie)
- [MLflow GenAI Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
