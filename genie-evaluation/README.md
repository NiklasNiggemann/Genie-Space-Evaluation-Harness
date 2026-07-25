# genie-evaluation

A production-grade evaluation harness for Databricks Genie Agents.

## What it does

1. **Ask** — Submits natural-language questions to a Genie Agent via the Conversation API
2. **Compare** — Extracts generated SQL and compares against ground-truth expected queries
3. **Score** — Uses an LLM judge (via MLflow) for semantic equivalence scoring
4. **Track** — Logs results to MLflow experiments and (optionally) Delta tables
5. **Gate** — Fails the job if accuracy drops below a configurable threshold

## Quick Start

```python
from genie_eval import EvaluationRunner, load_test_suite

suite = load_test_suite("test_suites/bakehouse_suite.yaml")
runner = EvaluationRunner(space_id="your-genie-agent-id")
results = runner.run(suite)

print(f"Accuracy: {results.accuracy:.0%}")
assert results.accuracy >= 0.75
```

## Installation

```bash
# From workspace path (on Databricks)
%pip install /Workspace/path/to/genie-evaluation

# From Git (with pip)
pip install git+https://github.com/your-org/genie-evaluation.git

# Development install
pip install -e ".[dev]"
```

## Project Structure

```
genie-evaluation/
├── src/genie_eval/           # The Python library
│   ├── __init__.py           # Public API exports
│   ├── api.py                # Genie Conversation API helpers
│   ├── judge.py              # LLM judge prompt & scorer creation
│   ├── models.py             # TestCase, EvalResult, EvalSuiteResults
│   └── runner.py             # Evaluation loop orchestration
├── tests/                    # Unit tests (pytest)
│   ├── conftest.py           # Shared fixtures
│   ├── test_api.py           # Tests for extraction functions and ask_genie
│   ├── test_judge.py         # Judge prompt regression tests
│   └── test_runner.py        # EvalSuiteResults and verdict merge tests
├── notebooks/
│   └── run_evaluation.py     # Thin orchestration notebook (scheduled)
├── test_suites/
│   └── bakehouse_suite.yaml  # Test cases as data
├── databricks.yml            # Declarative Automation Bundle config
├── pyproject.toml            # Package metadata & dependencies
└── README.md
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Deployment (DAB)

```bash
# Deploy to dev
databricks bundle deploy --target dev

# Run the job immediately
databricks bundle run genie_eval_nightly --target dev

# Deploy to prod
databricks bundle deploy --target prod
```

## Configuration

The runner accepts these parameters:

| Parameter | Default | Description |
|---|---|---|
| `space_id` | (required) | Genie Agent ID from the URL |
| `experiment_name` | None | MLflow experiment path for logging |
| `judge_model` | `databricks:/databricks-claude-sonnet-4` | LLM endpoint for the judge |
| `accuracy_threshold` | 0.75 | Minimum accuracy before the job fails |
| `verbose` | True | Print progress during evaluation |

## Writing Test Cases

Add test cases to a YAML file:

```yaml
- question: "What is the total revenue?"
  expected_sql: |
    SELECT SUM(totalPrice) AS total_revenue
    FROM samples.bakehouse.sales_transactions
  category: aggregation
  difficulty: easy
  expected_result_contains: null  # optional substring check
```

Categories and difficulties are free-form strings — use whatever makes sense for your domain.

## The Data Flywheel

This harness implements the continuous improvement loop:

```
Evaluate → Identify failures → Improve ontology → Re-evaluate → Ship with confidence
```

Each run is logged to MLflow, creating a time series of accuracy scores
that shows whether ontology changes help or hurt.

## Related

- [Blog Series: Evaluating Genie Agents](https://www.databricks.com/blog/reliable-by-design-evaluation-harness-databricks-genie)
- [Databricks Genie Documentation](https://docs.databricks.com/en/genie/index.html)
- [MLflow GenAI Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
