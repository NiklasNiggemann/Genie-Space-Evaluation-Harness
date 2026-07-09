# Genie Space Evaluation Harness

A programmatic evaluation framework for testing the reliability of **Databricks Genie Spaces** — the natural-language "Ask Your Data" interface.

---

## Motivation

When deploying an AI-powered data assistant, the central engineering challenge is **guaranteeing reliability**. A chatbot that occasionally produces wrong answers erodes trust faster than having no chatbot at all. The specific danger is that wrong answers arrive with the same formatting and apparent authority as correct ones — there is no signal, no confidence indicator, no way for the user to assess plausibility. The evaluation harness exists to make that silent failure detectable before it reaches the user.

This harness implements the *evaluation* leg of the [Data Flywheel](https://www.sh-reya.com/blog/ai-engineering-flywheel/):

> Evaluate → Identify failure patterns → Improve instructions → Re-evaluate → Ship with confidence

---

## What It Does

| Capability | Description |
|---|---|
| **API-driven question execution** | Submits questions to a Genie Space via the Conversation API |
| **Ground truth comparison** | Compares generated SQL and results to expected answers |
| **MLflow-based scoring** | LLM judge assesses *semantic* SQL correctness (not just exact-match) |
| **Data Flywheel iteration tracking** | Measures improvement over time as instructions are refined |

---

## Quick Start

1. Set `SPACE_ID` in the **Configuration** cell
2. Replace `test_cases` with questions for your data model
3. Set `EXPERIMENT_NAME` to your MLflow experiment path
4. Run all cells sequentially

The default suite uses `samples.bakehouse` — available in every Databricks workspace, no setup required.

---

## Key Insight

> A Genie Space without evaluation is a **hope-driven deployment**.
> A Genie Space with a harness is an **engineering-driven deployment**.

---

## References

- [Databricks Genie Spaces Documentation](https://docs.databricks.com/en/genie/index.html)
- [Genie Conversation API](https://docs.databricks.com/api/workspace/genie)
- [MLflow GenAI Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [The AI Engineering Flywheel](https://www.sh-reya.com/blog/ai-engineering-flywheel/) — Shreya Shankar
