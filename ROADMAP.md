# Genie Agent Evaluation Harness — Roadmap

## Short-term (High value, low effort) ✅

### 1. Parallel evaluation ✅
Run Genie API calls concurrently via `ThreadPoolExecutor`. The API is I/O-bound;
parallelism cuts a 40-question run from ~10 min to ~2 min with no accuracy impact.

```python
runner = EvaluationRunner(space_id="...", max_workers=8)
```

### 2. Category / difficulty filtering ✅
Filter which test cases to run without editing the YAML file. Useful for iterating
on a specific failure area (e.g. joins) without re-running the whole suite.

```python
results = runner.run(suite, categories=["join", "time_filter"])
results = runner.run(suite, difficulties=["hard"])
```

### 3. Run-over-run diff ✅
Compare two MLflow runs and surface per-question regressions and improvements.
Turns the MLflow history into an actionable diff rather than raw numbers.

```python
from genie_eval import compare_runs

df = compare_runs(run_id_before, run_id_after)
print(df[df.change == "regression"])
```

---

## Medium-term (Higher effort, high impact) ✅

### 4. Result-set scoring ✅
Execute the expected SQL directly against a Databricks SQL warehouse and compare
actual output rows against what Genie returned. Catches cases where the LLM judge
says "equivalent" but the data disagrees due to Genie catalog quirks or missing
permissions. Opt-in via `warehouse_id`.

```python
runner = EvaluationRunner(space_id="...", warehouse_id="abc123")
# result.result_set_correct is populated for each test case
```

### 5. CI/CD integration ✅
GitHub Actions workflow triggered on `genie-ontology/**` and `test_suites/**` PR
changes. Runs the evaluation suite and posts accuracy delta as a PR comment.
Fails the check if accuracy drops below threshold.

See `.github/workflows/genie_eval.yml` and `genie-evaluation/scripts/ci_eval.py`.

### 6. Richer failure reporting ✅
`EvalSuiteResults` now exposes structured breakdown methods for post-run analysis:

```python
results.report()                # per-question DataFrame
results.summary_by_category()  # accuracy + completion rate per category
results.summary_by_difficulty() # accuracy + completion rate per difficulty
```

---

## Long-term

### 7. Multi-Space comparison
Run the same test suite against a dev Space and a prod Space in parallel and diff
the results. Gate promotion on the dev Space matching or exceeding prod accuracy.

### 8. Golden set management
Version-control a locked "golden" test suite separately from the experimental one.
The golden set is never modified mid-iteration — it provides a stable regression
baseline even as new questions are added.
