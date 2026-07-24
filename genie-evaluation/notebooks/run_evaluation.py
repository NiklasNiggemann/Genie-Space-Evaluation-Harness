# Databricks notebook source
# MAGIC %md
# MAGIC # Genie Agent Evaluation — Production Runner
# MAGIC
# MAGIC Thin orchestration notebook scheduled via Lakeflow Jobs.
# MAGIC All logic lives in the `genie_eval` library — this notebook
# MAGIC only configures, runs, and asserts.

# COMMAND ----------

# MAGIC %pip install /Workspace/Users/niklas.niggemann@codecentric.de/Genie Evaluation Harness/genie-evaluation -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from genie_eval import EvaluationRunner, load_test_suite

# Configuration — override via job parameters (dbutils.widgets)
try:
    SPACE_ID = dbutils.widgets.get("space_id")
    ACCURACY_THRESHOLD = float(dbutils.widgets.get("accuracy_threshold"))
    TEST_SUITE_PATH = dbutils.widgets.get("test_suite")
except Exception:
    # Defaults for interactive use
    SPACE_ID = "01f16364ce181c628265e3815d9214cc"
    ACCURACY_THRESHOLD = 0.75
    TEST_SUITE_PATH = "test_suites/bakehouse_suite.yaml"

print(f"Space ID: {SPACE_ID}")
print(f"Accuracy threshold: {ACCURACY_THRESHOLD:.0%}")
print(f"Test suite: {TEST_SUITE_PATH}")

# COMMAND ----------

# Load test cases from versioned YAML
suite = load_test_suite(TEST_SUITE_PATH)
print(f"Loaded {len(suite)} test cases")

# COMMAND ----------

# Run evaluation (includes Genie API calls + LLM judge + MLflow logging)
runner = EvaluationRunner(
    space_id=SPACE_ID,
    experiment_name="/Users/niklas.niggemann@codecentric.de/genie-eval-experiment",
)
results = runner.run(suite)

# COMMAND ----------

# Write to Delta for historical tracking
# results.to_delta("catalog.schema.genie_eval_results")  # uncomment when target table exists

print(f"\nAccuracy: {results.accuracy:.0%} ({results.completed}/{results.total} completed)")

# COMMAND ----------

# Quality gate — fail the job if accuracy regressed
assert results.accuracy >= ACCURACY_THRESHOLD, (
    f"\u274c Accuracy {results.accuracy:.0%} is below threshold {ACCURACY_THRESHOLD:.0%}. "
    f"Review failing questions and update the Genie Ontology."
)

print(f"\u2705 Accuracy {results.accuracy:.0%} meets threshold {ACCURACY_THRESHOLD:.0%}")

