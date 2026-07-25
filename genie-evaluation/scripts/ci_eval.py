#!/usr/bin/env python3
"""CI evaluation script — triggered by GitHub Actions on ontology/test-suite changes.

Required environment variables:
    DATABRICKS_HOST       Databricks workspace URL
    DATABRICKS_TOKEN      Personal access token or service principal token
    GENIE_SPACE_ID        Genie Agent ID to evaluate

Optional environment variables:
    MLFLOW_EXPERIMENT_NAME  MLflow experiment path (default: /Shared/genie-eval-ci)
    ACCURACY_THRESHOLD      Minimum acceptable accuracy 0-1 (default: 0.75)
    WAREHOUSE_ID            SQL warehouse ID for result-set scoring (optional)
    MAX_WORKERS             Parallel API workers (default: 4)
    TEST_SUITE_PATH         Path to YAML test suite (default: test_suites/bakehouse_suite.yaml)

Writes eval_summary.md for the GitHub Actions PR comment step.
Exits with code 1 if accuracy is below threshold.
"""

import os
import sys
from pathlib import Path

from genie_eval import EvaluationRunner, load_test_suite

SPACE_ID = os.environ["GENIE_SPACE_ID"]
THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.75"))
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/genie-eval-ci")
WAREHOUSE_ID = os.environ.get("WAREHOUSE_ID")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))

suite_path = os.environ.get(
    "TEST_SUITE_PATH",
    str(Path(__file__).parent.parent / "test_suites/bakehouse_suite.yaml"),
)

suite = load_test_suite(suite_path)
runner = EvaluationRunner(
    space_id=SPACE_ID,
    experiment_name=EXPERIMENT,
    accuracy_threshold=THRESHOLD,
    max_workers=MAX_WORKERS,
    warehouse_id=WAREHOUSE_ID,
)
results = runner.run(suite)

passed = results.accuracy >= THRESHOLD
gate_line = "✅ Passed quality gate" if passed else f"❌ Failed quality gate (threshold: {THRESHOLD:.0%})"

# Per-category breakdown table
cat_df = results.summary_by_category()
cat_rows = "\n".join(
    f"| {row.category} | {row.total} | {row.accuracy:.0%} |"
    for _, row in cat_df.iterrows()
)

summary = f"""## Genie Agent Evaluation

| Metric | Value |
|--------|-------|
| Accuracy | **{results.accuracy:.0%}** |
| Completion rate | {results.completion_rate:.0%} |
| Total questions | {results.total} |
| MLflow run | `{results.run_id}` |

{gate_line}

### By category

| Category | Questions | Accuracy |
|----------|-----------|----------|
{cat_rows}
"""

Path("eval_summary.md").write_text(summary)
print(summary)

sys.exit(0 if passed else 1)
