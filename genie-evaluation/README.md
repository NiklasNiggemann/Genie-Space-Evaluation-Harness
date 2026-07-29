# Genie Agent Evaluation Harness

Automated evaluation framework for Databricks Genie Spaces. Tests whether a Genie Agent correctly answers business questions by comparing generated SQL against known-correct expected SQL.

## Quick Start

1. Open `notebooks/run_evaluation` in Databricks
2. Attach to a serverless compute cluster
3. Run All — the notebook installs the library, runs both suites, and asserts accuracy thresholds

## Project Structure

```
genie-evaluation/
├── notebooks/
│   └── run_evaluation          # Orchestration notebook (scheduled via Lakeflow Jobs)
├── src/
│   └── genie_eval/             # Core library (EvaluationRunner, scoring, etc.)
├── test_suites/
│   ├── bakehouse_suite.yaml    # Full evaluation suite
│   └── golden_suite.yaml       # Locked regression suite (must pass at 100%)
├── tests/                      # Unit tests
├── scripts/                    # Utility scripts
├── databricks.yml              # DABs bundle configuration
└── pyproject.toml              # Package metadata
```

## Installing the Genie Code Custom Skill

This repo ships a **custom skill** for Databricks Genie Code (the AI assistant) that enables interactive test-case generation directly from chat.

### What it does

The `generate-test-case` skill guides you through adding new evaluation test cases via a conversational workflow — drafting the expected SQL, choosing category/difficulty, and deciding whether the case belongs in the golden suite.

### Setup

The skill lives at the **repo root** (one level above `genie-evaluation/`):

```
Genie-Space-Evaluation-Harness/
├── .assistant/
│   └── skills/
│       └── generate-test-case/
│           └── SKILL.md        ← the skill definition
├── genie-evaluation/           ← this project
└── ...
```

**Option A — Automatic (recommended):**

Genie Code discovers `.assistant/skills/` directories relative to the workspace folder you're working in. As long as the `.assistant/` folder exists at the repo root (`Genie-Space-Evaluation-Harness/`), the skill is automatically available when you open any notebook in this repo.

**Option B — Persist to long-term memory:**

For reliable cross-session access (even when working from sub-folders), run this once in any Genie Code chat:

> "Read the skill at `.assistant/skills/generate-test-case/SKILL.md` and save it to your long-term memory."

Genie Code will store the schema, test format, and workflow in `~/.assistant_instructions.md` so it's available in every future session.

### Usage

Once installed, simply ask Genie Code:

> "Add a new test case for: What is the average order value per country?"

It will walk you through:
1. Drafting the expected SQL (using the `samples.bakehouse` schema)
2. Choosing a category (`aggregation`, `filter`, `join`, `time_filter`, `ambiguous`)
3. Setting difficulty (`easy`, `medium`, `hard`)
4. Deciding on `expected_result_contains` (for stable scalar answers)
5. Whether the test qualifies for the golden suite
6. Outputting the final YAML entry for the correct file(s)

## Configuration

The `run_evaluation` notebook accepts these parameters (set via widgets or Lakeflow Job overrides):

| Parameter | Default | Description |
|---|---|---|
| `install_mode` | `local` | How to install `genie_eval`: `local`, `git`, or `wheel` |
| `space_id` | `01f16364...` | Genie Space ID to evaluate |
| `accuracy_threshold` | `0.75` | Minimum accuracy to pass the quality gate |
| `test_suite` | `test_suites/bakehouse_suite.yaml` | Path to the full test suite |
| `golden_suite` | `test_suites/golden_suite.yaml` | Path to the golden (regression) suite |
| `max_workers` | `4` | Concurrent Genie API calls |
| `warehouse_id` | (empty) | Optional SQL warehouse for result comparison |
