# Genie Agent Evaluation Harness — Roadmap

All items completed. ✅

## Short-term ✅

### 1. Parallel evaluation ✅
`EvaluationRunner(max_workers=8)` — concurrent Genie API calls via `ThreadPoolExecutor`.

### 2. Category / difficulty filtering ✅
`runner.run(suite, categories=["join"], difficulties=["hard"])` — filter without editing YAML.

### 3. Run-over-run diff ✅
`compare_runs(run_id_a, run_id_b)` — per-question change classification across MLflow runs.

---

## Medium-term ✅

### 4. Result-set scoring ✅
`EvaluationRunner(warehouse_id="...")` — executes expected SQL and compares value-only row
bags against Genie's returned rows. Populates `EvalResult.result_set_correct`.

### 5. CI/CD integration ✅
`.github/workflows/genie_eval.yml` triggers on `genie-ontology/**` and `test_suites/**` PR
changes. `scripts/ci_eval.py` runs the golden suite (strict gate) then the full suite, and
posts an accuracy breakdown as a PR comment.

### 6. Richer failure reporting ✅
`results.report()`, `results.summary_by_category()`, `results.summary_by_difficulty()` —
per-question and grouped breakdown DataFrames, used by the CI script automatically.

---

## Long-term ✅

### 7. Multi-Space comparison ✅
`MultiSpaceRunner(space_ids={"dev": "...", "prod": "..."})` runs the same suite against
multiple Spaces in parallel and returns `dict[name, EvalSuiteResults]`.
`compare_spaces(results)` produces a summary DataFrame sorted by accuracy for easy promotion
gating: `assert results["dev"].accuracy >= results["prod"].accuracy`.

### 8. Golden set management ✅
`test_suites/golden_suite.yaml` — a small locked subset of high-confidence questions.
The CI pipeline runs it with `GOLDEN_THRESHOLD=1.0` (configurable) before the full suite.
Modify only via deliberate, reviewed PRs — never during an active ontology iteration.
