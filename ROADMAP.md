# Genie Agent Evaluation Harness — Roadmap

## Short-term (High value, low effort)

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

## Medium-term (Higher effort, high impact)

### 4. Result-set scoring
Execute both the generated SQL and expected SQL against Databricks and compare
actual output rows. Catches cases where the LLM judge says "equivalent" but the
data disagrees due to Genie catalog quirks or missing permissions.

### 5. CI/CD integration
Wire the DAB job into a PR check: when `genie-ontology/` files change, automatically
trigger an eval run and post the accuracy delta as a PR comment. Fail the check if
accuracy regresses below threshold.

### 6. Richer failure reporting
After each run, auto-generate a breakdown table (by category, difficulty, and
question) with the judge's full rationale, not just True/False. Surface this in the
MLflow UI or as a notebook output cell.

---

## Long-term

### 7. Multi-Space comparison
Run the same test suite against a dev Space and a prod Space in parallel and diff
the results. Gate promotion on the dev Space matching or exceeding prod accuracy.

### 8. Golden set management
Version-control a locked "golden" test suite separately from the experimental one.
The golden set is never modified mid-iteration — it provides a stable regression
baseline even as new questions are added.
