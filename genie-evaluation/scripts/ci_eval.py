#!/usr/bin/env python3
"""CI evaluation script — triggered by GitHub Actions on ontology/test-suite changes.

Runs two evaluations in sequence:
  1. Golden suite   — a small locked set; must pass at 100% (configurable).
  2. Full suite     — the complete test suite; must meet ACCURACY_THRESHOLD.

Required environment variables:
    DATABRICKS_HOST       Databricks workspace URL
    DATABRICKS_TOKEN      Personal access token or service principal token
    GENIE_SPACE_ID        Genie Agent ID to evaluate

Optional environment variables:
    MLFLOW_EXPERIMENT_NAME  MLflow experiment path (default: /Shared/genie-eval-ci)
    ACCURACY_THRESHOLD      Minimum full-suite accuracy 0–1 (default: 0.75)
    GOLDEN_THRESHOLD        Minimum golden-suite accuracy 0–1 (default: 1.0)
    WAREHOUSE_ID            SQL warehouse ID for result-set scoring (optional)
    MAX_WORKERS             Parallel API workers (default: 4)
    TEST_SUITE_PATH         Path to full YAML suite (default: test_suites/bakehouse_suite.yaml)
    GOLDEN_SUITE_PATH       Path to golden YAML suite (default: test_suites/golden_suite.yaml)

Writes eval_summary.md for the GitHub Actions PR comment step.
Exits with code 1 if either gate fails.
"""

import os
import sys
from pathlib import Path

from genie_eval import EvaluationRunner, load_test_suite

SCRIPTS_DIR = Path(__file__).parent
SUITES_DIR = SCRIPTS_DIR.parent / "test_suites"

SPACE_ID = os.environ["GENIE_SPACE_ID"]
THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.75"))
GOLDEN_THRESHOLD = float(os.environ.get("GOLDEN_THRESHOLD", "1.0"))
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "/Shared/genie-eval-ci")
WAREHOUSE_ID = os.environ.get("WAREHOUSE_ID")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))

golden_path = os.environ.get("GOLDEN_SUITE_PATH", str(SUITES_DIR / "golden_suite.yaml"))
suite_path = os.environ.get("TEST_SUITE_PATH", str(SUITES_DIR / "bakehouse_suite.yaml"))

runner = EvaluationRunner(
    space_id=SPACE_ID,
    experiment_name=EXPERIMENT,
    max_workers=MAX_WORKERS,
    warehouse_id=WAREHOUSE_ID,
)

# ---- 1. Golden suite (strict gate) ----------------------------------------

print("\n=== GOLDEN SUITE ===")
golden_suite = load_test_suite(golden_path)
golden_results = runner.run(golden_suite)
golden_passed = golden_results.accuracy >= GOLDEN_THRESHOLD
golden_icon = "✅" if golden_passed else "❌"

# ---- 2. Full suite (quality gate) -----------------------------------------

print("\n=== FULL SUITE ===")
full_suite = load_test_suite(suite_path)
full_results = runner.run(full_suite)
full_passed = full_results.accuracy >= THRESHOLD
full_icon = "✅" if full_passed else "❌"

# ---- PR comment -----------------------------------------------------------

cat_rows = "\n".join(
    f"| {row.category} | {row.total} | {row.accuracy:.0%} |"
    for _, row in full_results.summary_by_category().iterrows()
)

summary = f"""## Genie Agent Evaluation

### Golden suite {golden_icon}

| Metric | Value |
|--------|-------|
| Accuracy | **{golden_results.accuracy:.0%}** (threshold: {GOLDEN_THRESHOLD:.0%}) |
| Questions | {golden_results.total} |
| MLflow run | `{golden_results.run_id}` |

{"✅ All golden cases passed" if golden_passed else "❌ Golden regression — one or more locked questions failed"}

### Full suite {full_icon}

| Metric | Value |
|--------|-------|
| Accuracy | **{full_results.accuracy:.0%}** (threshold: {THRESHOLD:.0%}) |
| Completion rate | {full_results.completion_rate:.0%} |
| Total questions | {full_results.total} |
| MLflow run | `{full_results.run_id}` |

{"✅ Passed quality gate" if full_passed else f"❌ Failed quality gate (threshold: {THRESHOLD:.0%})"}

#### By category

| Category | Questions | Accuracy |
|----------|-----------|----------|
{cat_rows}
"""

Path("eval_summary.md").write_text(summary)
print(summary)

sys.exit(0 if (golden_passed and full_passed) else 1)
