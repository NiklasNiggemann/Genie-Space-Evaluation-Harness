# Manual Testing Guide

Step-by-step checklist for verifying the harness end-to-end against a real Genie Space.

---

## Prerequisites

- Databricks credentials configured (`DATABRICKS_HOST`, `DATABRICKS_TOKEN` or `databricks auth login`)
- A running Genie Space (grab the `space_id` from the URL)
- Package installed: `pip install -e ".[dev]"` from `genie-evaluation/`

---

## 1. Smoke test — single question

```python
from genie_eval import EvaluationRunner, load_test_suite

runner = EvaluationRunner(space_id="<your-space-id>", verbose=True)
suite = load_test_suite("test_suites/bakehouse_suite.yaml")

results = runner.run(suite[:1])
print(results.accuracy)
```

**Pass:** no exceptions, `accuracy` is `0.0` or `1.0`, judge verdict printed.

---

## 2. Golden suite — must be 100%

```python
golden = load_test_suite("test_suites/golden_suite.yaml")
results = runner.run(golden)
assert results.accuracy == 1.0, f"Golden suite failed: {results.accuracy:.0%}"
```

**Pass:** assertion holds. This is the sharpest sanity check — if this fails, stop here.

---

## 3. Failure reporting — verify DataFrame shape

```python
results = runner.run(suite)

report = results.report()
assert set(report.columns) >= {
    "question", "category", "difficulty", "status",
    "judge_correct", "result_set_correct",
    "execution_time_s", "generated_sql", "expected_sql",
}

print(results.summary_by_category())
print(results.summary_by_difficulty())
```

**Pass:** all expected columns present, summaries print without error.

---

## 4. Result-set scoring — needs a SQL warehouse

```python
runner_rs = EvaluationRunner(
    space_id="<your-space-id>",
    warehouse_id="<your-warehouse-id>",
    verbose=True,
)
results = runner_rs.run(suite[:3])

rs_col = results.report()["result_set_correct"]
assert rs_col.notna().any(), "result_set_correct is all None — warehouse scoring not running"
print(rs_col)
```

**Pass:** at least one row has a non-null `result_set_correct` value.

---

## 5. Category / difficulty filtering

```python
results_join = runner.run(suite, categories=["join"])
results_hard = runner.run(suite, difficulties=["hard"])

assert all(results_join.report()["category"] == "join")
assert all(results_hard.report()["difficulty"] == "hard")
```

**Pass:** filtered results contain only the expected rows.

---

## 6. MultiSpaceRunner + compare_spaces

_Requires two Genie Spaces (e.g. dev and prod)._

```python
from genie_eval import MultiSpaceRunner, compare_spaces

runner_multi = MultiSpaceRunner(
    space_ids={"dev": "<dev-space-id>", "prod": "<prod-space-id>"},
    max_workers_per_space=4,
)
results = runner_multi.run(golden)

df = compare_spaces(results)
print(df)   # columns: name, space_id, accuracy, completion_rate, total
assert results["dev"].accuracy >= results["prod"].accuracy, "dev regressed vs prod"
```

**Pass:** DataFrame has both rows, assertion holds.

---

## 7. MLflow tracking (optional)

```python
runner_mlflow = EvaluationRunner(
    space_id="<your-space-id>",
    experiment_name="/Shared/genie-eval-manual-test",
)
results = runner_mlflow.run(golden)
```

**Pass:** run appears in the MLflow UI under `/Shared/genie-eval-manual-test` with accuracy logged as a metric.
