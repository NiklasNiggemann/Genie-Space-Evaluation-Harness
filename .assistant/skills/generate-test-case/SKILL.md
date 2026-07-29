---
name: generate-test-case
description: Interactively generate a new test case for the Genie Agent evaluation suite. Guides you through the question, expected SQL, category, difficulty, and whether it belongs in the golden suite.
---

## Purpose

This skill helps you add new entries to the evaluation test suites in `genie-evaluation/test_suites/`. Test cases measure how well the Genie Agent answers business questions by comparing its generated SQL against a known-correct expected SQL.

## Test suite format

```yaml
- question: "What is the total revenue across all transactions?"
  expected_sql: |
    SELECT SUM(totalPrice) AS total_revenue
    FROM samples.bakehouse.sales_transactions
  category: aggregation
  difficulty: easy
  expected_result_contains: null   # optional: stable substring expected in the result
```

**Files:**
- `bakehouse_suite.yaml` — full evaluation suite, all test cases go here
- `golden_suite.yaml` — small locked subset; only unambiguous, stable questions; must always pass at 100%

## Database schema

The Genie Agent queries the `samples.bakehouse` catalog:

| Table | Key columns |
|---|---|
| `sales_transactions` | `dateTime`, `totalPrice`, `quantity`, `product`, `paymentMethod`, `customerID`, `franchiseID` |
| `sales_customers` | `customerID`, `first_name`, `last_name`, `country` |
| `sales_franchises` | `franchiseID`, `name`, `country`, `supplierID` |
| `sales_suppliers` | `supplierID`, `name`, `ingredient` |

## Steps

Work through these steps interactively:

### 1. Get the question
Ask the user what business question they want to test. It should be phrased naturally, the way a business user would type it into Genie.

### 2. Draft the expected SQL
Write the expected SQL based on the schema above. Show it to the user and confirm it's correct. Iterate until they approve.

SQL rules:
- Use explicit date ranges (`dateTime >= '2024-05-01' AND dateTime < '2024-06-01'`), not `MONTH()` / `YEAR()` — Databricks SQL handles range comparisons more reliably
- Always alias columns in JOIN queries to avoid ambiguity
- Never invent columns or tables not listed in the schema above

### 3. Suggest category and difficulty

**Category:**
- `aggregation` — SUM, COUNT, AVG, GROUP BY on a single table
- `filter` — WHERE clause filtering, single table
- `join` — requires joining two or more tables
- `time_filter` — date or time range filtering
- `ambiguous` — question has more than one valid interpretation; note the assumption made

**Difficulty:**
- `easy` — single table, one operation
- `medium` — one join, or a non-trivial filter/aggregation
- `hard` — multiple joins, subqueries, or complex business logic

Suggest based on the SQL, but let the user override.

### 4. Check for expected_result_contains
If the question has a known, stable scalar answer (e.g. a count that won't change as the dataset grows), suggest adding it as `expected_result_contains`. Otherwise set it to `null` or omit it.

### 5. Decide: golden suite candidate?
A question belongs in `golden_suite.yaml` if it is:
- **Unambiguous** — only one correct interpretation
- **Stable** — the result won't change as data grows or changes
- **Easy or medium** difficulty

If yes → add to both `golden_suite.yaml` and `bakehouse_suite.yaml`.
If no → `bakehouse_suite.yaml` only.

Remind the user: golden suite changes require a deliberate, reviewed PR — never edit it during an active ontology iteration.

### 6. Output the final YAML
Produce the complete YAML entry, formatted and ready to paste. Tell the user exactly which file(s) to add it to and where (append to the end of the file).
