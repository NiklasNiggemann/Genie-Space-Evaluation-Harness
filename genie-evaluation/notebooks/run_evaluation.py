# Databricks notebook source
# MAGIC %md
# MAGIC # Genie Agent Evaluation — Production Runner
# MAGIC
# MAGIC Thin orchestration notebook scheduled via Lakeflow Jobs.
# MAGIC All logic lives in the `genie_eval` library — this notebook
# MAGIC only configures, runs, and asserts.

# COMMAND ----------

# MAGIC %pip install "mlflow>=3.0" -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import subprocess
from databricks.sdk.runtime import dbutils

# install_mode widget: "wheel" = pinned production build, "git" = latest main (dev iteration)
dbutils.widgets.dropdown("install_mode", "wheel", ["wheel", "git"])
INSTALL_MODE = dbutils.widgets.get("install_mode")

if INSTALL_MODE == "git":
    subprocess.run([
        "pip", "install", "-q", "--upgrade",
        "git+https://github.com/NiklasNiggemann/Genie-Space-Evaluation-Harness.git#subdirectory=genie-evaluation",
    ], check=True)
    print("✅ Installed genie_eval from git (latest main)")
else:
    subprocess.run([
        "pip", "install", "-q",
        "/Workspace/Users/niklas.niggemann@codecentric.de/genie-eval/genie_eval-0.1.0-py3-none-any.whl",
    ], check=True)
    print("✅ Installed genie_eval from wheel (pinned)")

# COMMAND ----------

from genie_eval import EvaluationRunner, load_test_suite

# Job parameters — set defaults here, override via Lakeflow job parameters
dbutils.widgets.text("space_id",           "01f16364ce181c628265e3815d9214cc")
dbutils.widgets.text("accuracy_threshold", "0.75")
dbutils.widgets.text("test_suite",         "test_suites/bakehouse_suite.yaml")
dbutils.widgets.text("golden_suite",       "test_suites/golden_suite.yaml")
dbutils.widgets.text("max_workers",        "4")
dbutils.widgets.text("warehouse_id",       "")

SPACE_ID           = dbutils.widgets.get("space_id")
ACCURACY_THRESHOLD = float(dbutils.widgets.get("accuracy_threshold"))
TEST_SUITE_PATH    = dbutils.widgets.get("test_suite")
GOLDEN_SUITE_PATH  = dbutils.widgets.get("golden_suite")
MAX_WORKERS        = int(dbutils.widgets.get("max_workers"))
WAREHOUSE_ID       = dbutils.widgets.get("warehouse_id") or None

print(f"Space ID:          {SPACE_ID}")
print(f"Accuracy threshold:{ACCURACY_THRESHOLD:.0%}")
print(f"Test suite:        {TEST_SUITE_PATH}")
print(f"Max workers:       {MAX_WORKERS}")

# COMMAND ----------

runner = EvaluationRunner(
    space_id=SPACE_ID,
    experiment_name="/Shared/genie-eval",
    max_workers=MAX_WORKERS,
    warehouse_id=WAREHOUSE_ID,
)

# COMMAND ----------
# MAGIC %md ### 1 — Golden suite (strict gate)

golden_suite = load_test_suite(GOLDEN_SUITE_PATH)
golden_results = runner.run(golden_suite)

assert golden_results.accuracy >= 1.0, (
    f"❌ Golden regression: {golden_results.accuracy:.0%} — one or more locked "
    "questions failed. Fix the ontology or the golden suite before proceeding."
)
print(f"✅ Golden suite: {golden_results.accuracy:.0%} ({golden_results.total} questions)")

# COMMAND ----------
# MAGIC %md ### 2 — Full suite (quality gate)

full_suite = load_test_suite(TEST_SUITE_PATH)
results = runner.run(full_suite)

# COMMAND ----------

display(results.summary_by_category())

# COMMAND ----------

# Uncomment to persist results to Delta for historical trend analysis
# results.to_delta("catalog.schema.genie_eval_results")

# COMMAND ----------

assert results.accuracy >= ACCURACY_THRESHOLD, (
    f"❌ Accuracy {results.accuracy:.0%} is below threshold {ACCURACY_THRESHOLD:.0%}. "
    "Review the failures above and update the Genie Ontology."
)
print(f"✅ Accuracy {results.accuracy:.0%} meets threshold {ACCURACY_THRESHOLD:.0%}")
