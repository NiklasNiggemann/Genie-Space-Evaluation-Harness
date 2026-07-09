# Databricks notebook source
# DBTITLE 1,Genie Space Evaluation Harness
# MAGIC %md
# MAGIC # Genie Space Evaluation Harness
# MAGIC
# MAGIC A programmatic framework for **testing and improving** Databricks Genie Spaces — the natural-language "Ask Your Data" interface.
# MAGIC
# MAGIC ## What this notebook covers
# MAGIC
# MAGIC | Step | What happens |
# MAGIC |---|---|
# MAGIC | 1. **Ask** | Submit questions to a Genie Space via the Conversation API |
# MAGIC | 2. **Compare** | Match generated SQL against expected ground-truth queries |
# MAGIC | 3. **Score** | Use an LLM judge (MLflow) for semantic equivalence — not just exact-match |
# MAGIC | 4. **Iterate** | Track accuracy over time, identify failure patterns, improve instructions |
# MAGIC
# MAGIC ## Core Idea
# MAGIC
# MAGIC > A Genie Space without evaluation is a **hope-driven deployment**.  
# MAGIC > A Genie Space *with* a harness is an **engineering-driven deployment**.
# MAGIC
# MAGIC This implements the [Data Flywheel](https://www.sh-reya.com/blog/ai-engineering-flywheel/):  
# MAGIC **Evaluate → Identify failures → Improve → Re-evaluate → Ship with confidence**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Getting Started
# MAGIC
# MAGIC This notebook uses `samples.bakehouse` — a Databricks sample dataset available in every workspace. No additional data setup is required to run the default test suite.
# MAGIC
# MAGIC To adapt it to your own Genie Space:
# MAGIC 1. Set `SPACE_ID` in the **Configuration** cell (from the URL: `.../genie/rooms/<SPACE_ID>`)
# MAGIC 2. Replace `test_cases` in the **Test Suite** cell with questions and expected SQL for your data model
# MAGIC 3. Run all cells sequentially
# MAGIC
# MAGIC The `genie-space-config/` folder shows the configuration-as-code pattern in action: versioned instructions, example SQL, and join specifications that can be reviewed and diffed alongside the test suite.

# COMMAND ----------

# DBTITLE 1,Prerequisites
# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC
# MAGIC **Who is this for?**  
# MAGIC Data engineers, analysts, and platform teams who deploy or manage Genie Spaces and want to ensure answer quality before going live.
# MAGIC
# MAGIC **What you need:**
# MAGIC
# MAGIC | Requirement | Details |
# MAGIC |---|---|
# MAGIC | Databricks workspace | With access to a SQL warehouse |
# MAGIC | A Genie Space | Any Space — we use the `samples.bakehouse` dataset as an example |
# MAGIC | Python basics | Comfortable reading Python; no ML expertise required |
# MAGIC | MLflow (optional) | Pre-installed on Databricks ML Runtime; used for LLM-as-judge scoring |

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install mlflow -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup & Imports
# Setup & Imports
# ================
from databricks.sdk import WorkspaceClient
try:    import mlflow
except ModuleNotFoundError:
    print('mlflow module is not installed. Please install it to use related functionalities.')

try:    import mlflow.genai
except ModuleNotFoundError:
    print('mlflow.genai module is not installed. Please install mlflow with genai support.')

from mlflow.genai import make_judge
import pandas as pd
import time
import json
from datetime import datetime

# Initialize the Databricks Workspace Client
# (Automatically authenticated when running on Databricks compute)
w = WorkspaceClient()

print(f"✓ Workspace client initialized")
print(f"  Host: {w.config.host}")
print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# COMMAND ----------

# DBTITLE 1,Configuration
# Configuration
# =============
# Set your Genie Space ID and evaluation parameters here.

# ─── Genie Space ID ───────────────────────────────────────────────────────────
# Find this in your Genie Space URL:
#   https://<workspace>.databricks.com/genie/rooms/<SPACE_ID>
SPACE_ID = "01f16364ce181c628265e3815d9214cc"  # Bakehouse Analytics — replace with your Space ID

# ─── Polling Configuration ────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS = 3
MAX_POLL_ATTEMPTS = 60  # 3 min total

# ─── MLflow Experiment ────────────────────────────────────────────────────────
EXPERIMENT_NAME = "/Users/<your-username>/genie-eval-experiment"
mlflow.set_experiment(EXPERIMENT_NAME)

assert SPACE_ID, "⚠️  Please set SPACE_ID above before running the evaluation."
print(f"✓ Configuration loaded")
print(f"  Space ID: {SPACE_ID}")
print(f"  Timeout:  {POLL_INTERVAL_SECONDS * MAX_POLL_ATTEMPTS}s")
print(f"  Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# DBTITLE 1,Genie API Helpers
# Genie API Helpers
# =================
# Functions to interact with Genie Spaces programmatically via the REST API.

def ask_genie(space_id: str, question: str) -> dict:
    """
    Submit a natural-language question to a Genie Space and wait for the response.
    Uses the Genie Conversation API:
      1. POST /start-conversation to submit the question
      2. Poll GET /messages/{id} until status reaches a terminal state
    """
    start_time = time.time()
    response = w.api_client.do(
        method="POST",
        path=f"/api/2.0/genie/spaces/{space_id}/start-conversation",
        body={"content": question}
    )
    conversation_id = response["conversation_id"]
    message_id = response["message_id"]
    terminal_statuses = {"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"}
    for attempt in range(MAX_POLL_ATTEMPTS):
        msg_response = w.api_client.do(
            method="GET",
            path=f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}"
        )
        status = msg_response.get("status", "UNKNOWN")
        if status in terminal_statuses:
            elapsed = time.time() - start_time
            return {"status": status, "message": msg_response,
                    "conversation_id": conversation_id, "message_id": message_id,
                    "elapsed_seconds": round(elapsed, 2)}
        time.sleep(POLL_INTERVAL_SECONDS)
    elapsed = time.time() - start_time
    return {"status": "TIMEOUT", "message": msg_response,
            "conversation_id": conversation_id, "message_id": message_id,
            "elapsed_seconds": round(elapsed, 2)}


def extract_sql(message: dict) -> str:
    """Extract the generated SQL from a Genie message response."""
    msg_data = message.get("message", message)
    attachments = msg_data.get("attachments", []) or []
    for attachment in attachments:
        if "query" in attachment:
            query_info = attachment["query"]
            if "query" in query_info:
                return query_info["query"]
            if "sql" in query_info:
                return query_info["sql"]
    return ""


def extract_result(message: dict) -> list:
    """Extract query result rows from a Genie message response."""
    msg_data = message.get("message", message)
    attachments = msg_data.get("attachments", []) or []
    for attachment in attachments:
        if "query" in attachment:
            query_info = attachment["query"]
            if "query_result" in query_info:
                result = query_info["query_result"]
                return result.get("data", result) if isinstance(result, dict) else result
            if "result" in query_info:
                return query_info["result"]
    if "query_result" in msg_data:
        qr = msg_data["query_result"]
        return qr.get("data", qr) if isinstance(qr, dict) else (qr if isinstance(qr, list) else [])
    return []


def extract_text_response(message: dict) -> str:
    """Extract Genie's narrative text response."""
    msg_data = message.get("message", message)
    attachments = msg_data.get("attachments", []) or []
    for attachment in attachments:
        if "text" in attachment:
            return attachment["text"].get("content", "")
    return ""


print("✓ Genie API helper functions defined")
print("  • ask_genie(space_id, question) → polls until response ready")
print("  • extract_sql(message) → generated SQL string")
print("  • extract_result(message) → query result rows")
print("  • extract_text_response(message) → narrative text")

# COMMAND ----------

# DBTITLE 1,Live Demo — Single Question
# Quick Start — Single Question
# ==============================
# Run this cell to try a single question before running the full evaluation suite.

DEMO_QUESTION = "What are the top 5 best-selling products?"

print(f"Asking Genie: '{DEMO_QUESTION}'")
print("Waiting for response...\n")

demo_result = ask_genie(SPACE_ID, DEMO_QUESTION)

print("="*60)
print(" RESPONSE ANATOMY")
print("="*60)
print(f"\n  Status:        {demo_result['status']}")
print(f"  Response time: {demo_result['elapsed_seconds']}s")
print(f"  Conv. ID:      {demo_result['conversation_id'][:12]}...")

generated_sql = extract_sql(demo_result)
text_response = extract_text_response(demo_result)
result_rows = extract_result(demo_result)

if generated_sql:
    print(f"\n  ┌─ Generated SQL ({'\u2713' if generated_sql else '\u2717'}) ──────────────────────")
    for line in generated_sql.strip().split('\n'):
        print(f"  │  {line}")
    print(f"  └{'\u2500'*50}")

if text_response:
    print(f"\n  Text response: {text_response[:150]}")

if result_rows:
    print(f"\n  Result preview ({len(result_rows)} rows):")
    for row in result_rows[:5]:
        print(f"    {row}")

print("\n" + "─"*60)

# COMMAND ----------

# DBTITLE 1,Test Suite Definition
# Test Suite Definition
# =====================
# Customize these test cases for your Genie Space and data model.

test_cases = [
    {
        "question": "What is the total revenue across all transactions?",
        "expected_sql": """
            SELECT SUM(totalPrice) AS total_revenue
            FROM samples.bakehouse.sales_transactions
        """,
        "expected_result_contains": None,
        "category": "aggregation", "difficulty": "easy"
    },
    {
        "question": "How many transactions were paid with Visa?",
        "expected_sql": """
            SELECT COUNT(*) AS visa_transactions
            FROM samples.bakehouse.sales_transactions
            WHERE paymentMethod = 'visa'
        """,
        "expected_result_contains": "1083",
        "category": "filter", "difficulty": "easy"
    },
    {
        "question": "What is the total revenue per product?",
        "expected_sql": """
            SELECT product, SUM(totalPrice) AS revenue
            FROM samples.bakehouse.sales_transactions
            GROUP BY product ORDER BY revenue DESC
        """,
        "expected_result_contains": None,
        "category": "aggregation", "difficulty": "easy"
    },
    {
        "question": "Show me total sales per franchise in Japan",
        "expected_sql": """
            SELECT f.name, SUM(t.totalPrice) AS total_sales
            FROM samples.bakehouse.sales_transactions t
            JOIN samples.bakehouse.sales_franchises f ON t.franchiseID = f.franchiseID
            WHERE f.country = 'Japan'
            GROUP BY f.name ORDER BY total_sales DESC
        """,
        "expected_result_contains": None,
        "category": "join", "difficulty": "medium"
    },
    {
        "question": "Which customers from Australia have spent the most in total?",
        "expected_sql": """
            SELECT c.first_name, c.last_name, SUM(t.totalPrice) AS total_spent
            FROM samples.bakehouse.sales_transactions t
            JOIN samples.bakehouse.sales_customers c ON t.customerID = c.customerID
            WHERE c.country = 'Australia'
            GROUP BY c.first_name, c.last_name ORDER BY total_spent DESC
        """,
        "expected_result_contains": None,
        "category": "join", "difficulty": "medium"
    },
    {
        "question": "What were the total sales in May 2024?",
        "expected_sql": """
            SELECT SUM(totalPrice) AS may_sales
            FROM samples.bakehouse.sales_transactions
            WHERE dateTime >= '2024-05-01' AND dateTime < '2024-06-01'
        """,
        "expected_result_contains": None,
        "category": "time_filter", "difficulty": "medium"
    },
    {
        "question": "What's the most popular product?",
        "expected_sql": """
            SELECT product, COUNT(*) AS transaction_count
            FROM samples.bakehouse.sales_transactions
            GROUP BY product ORDER BY transaction_count DESC LIMIT 1
        """,
        "expected_result_contains": "Golden Gate Ginger",
        "category": "ambiguous", "difficulty": "medium"
    },
    {
        "question": "Which supplier provides ingredients to the franchise with the highest revenue?",
        "expected_sql": """
            SELECT s.name AS supplier_name, s.ingredient,
                   f.name AS franchise_name, SUM(t.totalPrice) AS franchise_revenue
            FROM samples.bakehouse.sales_transactions t
            JOIN samples.bakehouse.sales_franchises f ON t.franchiseID = f.franchiseID
            JOIN samples.bakehouse.sales_suppliers s ON f.supplierID = s.supplierID
            GROUP BY s.name, s.ingredient, f.name
            ORDER BY franchise_revenue DESC LIMIT 1
        """,
        "expected_result_contains": None,
        "category": "join", "difficulty": "hard"
    },
]

print(f"✓ Test suite loaded: {len(test_cases)} test cases")
for i, tc in enumerate(test_cases, 1):
    print(f"  [{i}] ({tc['category']}/{tc['difficulty']}) {tc['question'][:60]}...")

# COMMAND ----------

# DBTITLE 1,Run Evaluation
# Run Evaluation
# ==============
print("\n" + "="*70)
print(" GENIE SPACE EVALUATION RUN")
print(f" Space: {SPACE_ID}")
print(f" Test cases: {len(test_cases)}")
print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70 + "\n")

eval_records = []

for i, test_case in enumerate(test_cases, 1):
    question = test_case["question"]
    print(f"[{i}/{len(test_cases)}] ▶ {question}")
    try:
        result = ask_genie(SPACE_ID, question)
        generated_sql = extract_sql(result)
        query_results = extract_result(result)
        text_response = extract_text_response(result)
        status = result["status"]
        elapsed = result["elapsed_seconds"]
        result_match = None
        if test_case.get("expected_result_contains"):
            result_str = json.dumps(query_results)
            result_match = test_case["expected_result_contains"] in result_str
        record = {
            "question": question,
            "category": test_case["category"],
            "difficulty": test_case["difficulty"],
            "status": status,
            "generated_sql": generated_sql.strip() if generated_sql else "",
            "expected_sql": test_case.get("expected_sql", "").strip(),
            "text_response": text_response[:200] if text_response else "",
            "result_preview": json.dumps(query_results[:3]) if query_results else "[]",
            "result_contains_expected": result_match,
            "execution_time_seconds": elapsed,
            "conversation_id": result["conversation_id"],
            "message_id": result["message_id"],
        }
        if status == "COMPLETED" and generated_sql:
            print(f"       ✓ Completed in {elapsed}s | SQL generated ({len(generated_sql)} chars)")
        elif status == "COMPLETED":
            print(f"       ✓ Completed in {elapsed}s | Text response (no SQL)")
        else:
            print(f"       ✗ {status} after {elapsed}s")
    except Exception as e:
        record = {
            "question": question, "category": test_case["category"],
            "difficulty": test_case["difficulty"], "status": "ERROR",
            "generated_sql": "", "expected_sql": test_case.get("expected_sql", "").strip(),
            "text_response": str(e)[:200], "result_preview": "[]",
            "result_contains_expected": None, "execution_time_seconds": 0,
            "conversation_id": "", "message_id": "",
        }
        print(f"       ✗ ERROR: {str(e)[:80]}")
    eval_records.append(record)
    print()

eval_results = pd.DataFrame(eval_records)
print("\n" + "="*70)
print(f" EVALUATION COMPLETE — {len(eval_results)} questions processed")
print(f" Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
display(eval_results[["question", "category", "status", "execution_time_seconds"]].head(20))

# COMMAND ----------

# DBTITLE 1,MLflow Scoring with Custom Judge
# MLflow Scoring with Custom Judge
# =================================
# Use an LLM judge to evaluate semantic SQL correctness.

SQL_JUDGE_INSTRUCTIONS = """
You are an expert SQL evaluator for Databricks SQL. Determine whether the
generated SQL in {{ outputs }} is semantically equivalent to the expected SQL
in {{ expectations }} — meaning they would produce the same result set.

Acceptable differences (still rate True):
- Different column aliases, join syntax, whitespace, equivalent date functions

Must match (rate False if different):
- Different tables, different aggregation functions (SUM vs COUNT),
  missing/extra filter conditions, different GROUP BY columns, different LIMIT values

The original question asked was: {{ inputs }}

Rate True if the generated SQL would produce equivalent results to the expected SQL.
Rate False if they would produce meaningfully different results.
"""

sql_correctness_judge = make_judge(
    name="sql_semantic_correctness",
    instructions=SQL_JUDGE_INSTRUCTIONS,
    model="databricks:/databricks-claude-sonnet-4",
    feedback_value_type=bool,
)

eval_with_sql = eval_results[
    (eval_results["generated_sql"] != "") &
    (eval_results["expected_sql"] != "")
].copy()

if len(eval_with_sql) == 0:
    print("⚠️  No test cases have both generated SQL and expected SQL.")
else:
    print(f"✓ Evaluating {len(eval_with_sql)} test cases with SQL judge...")
    eval_data = pd.DataFrame({
        "inputs": [{"question": q} for q in eval_with_sql["question"].tolist()],
        "outputs": eval_with_sql["generated_sql"].tolist(),
        "expectations": [{"expected_sql": sql} for sql in eval_with_sql["expected_sql"].tolist()],
    })
    with mlflow.start_run(run_name=f"genie-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.log_param("space_id", SPACE_ID)
        mlflow.log_param("num_test_cases", len(test_cases))
        mlflow.log_param("num_evaluated", len(eval_with_sql))
        mlflow.log_param("eval_timestamp", datetime.now().isoformat())
        judge_results = mlflow.genai.evaluate(
            data=eval_data,
            scorers=[sql_correctness_judge],
        )
        if hasattr(judge_results, 'metrics') and judge_results.metrics:
            for metric_name, metric_value in judge_results.metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        print(f"\n✓ Evaluation logged to MLflow (Run ID: {run.info.run_id})")
        print(f"  Experiment: {EXPERIMENT_NAME}")
    print("\n" + "-"*70)
    print(" LLM JUDGE RESULTS: SQL Semantic Correctness")
    print("-"*70)
    if hasattr(judge_results, 'tables') and "eval_results" in judge_results.tables:
        judge_df = judge_results.tables["eval_results"]
        display(judge_df)
    elif hasattr(judge_results, 'metrics'):
        print(f"\nMetrics: {judge_results.metrics}")

# COMMAND ----------

# DBTITLE 1,Results Summary & Recommendations
# Results Summary & Recommendations
# ==================================
if 'judge_df' in dir() and judge_df is not None and "sql_semantic_correctness/value" in judge_df.columns:
    judge_verdicts = judge_df["sql_semantic_correctness/value"].tolist()
    eval_results["judge_correct"] = False
    sql_mask = (eval_results["generated_sql"] != "") & (eval_results["expected_sql"] != "")
    eval_results.loc[sql_mask, "judge_correct"] = judge_verdicts
else:
    eval_results["judge_correct"] = eval_results["status"] == "COMPLETED"
eval_results["judge_correct"] = eval_results["judge_correct"].astype(bool)

print("\n" + "="*70)
print(" EVALUATION RESULTS SUMMARY")
print("="*70)
total = len(eval_results)
completed = len(eval_results[eval_results["status"] == "COMPLETED"])
with_sql = len(eval_results[eval_results["generated_sql"] != ""])
failed_status = len(eval_results[eval_results["status"].isin(["FAILED", "ERROR", "TIMEOUT"])])
judge_correct = eval_results["judge_correct"].sum()
judge_incorrect = len(eval_results[eval_results["judge_correct"] == False])
completion_rate = (completed / total * 100) if total > 0 else 0
accuracy_rate = (judge_correct / total * 100) if total > 0 else 0
avg_response_time = eval_results["execution_time_seconds"].mean()
print(f"\n  ┌────────────────────────────────────────────────────┐")
print(f"  │  Total test cases:         {total:>5}              │")
print(f"  │  Genie responded:          {completed:>5} ({completion_rate:.0f}%)       │")
print(f"  │  SQL generated:            {with_sql:>5}              │")
print(f"  │  Semantically correct:     {int(judge_correct):>5} ({accuracy_rate:.0f}%) ✓    │")
print(f"  │  Semantically incorrect:   {judge_incorrect:>5}        ✗    │")
print(f"  │  Failed/Error/Timeout:     {failed_status:>5}              │")
print(f"  │  Avg response time:        {avg_response_time:.1f}s             │")
print(f"  └────────────────────────────────────────────────────┘")
print("\n\n  PER-CATEGORY ACCURACY")
print("  " + "-"*50)
category_stats = eval_results.groupby("category").agg(
    total_questions=("question", "count"),
    correct=("judge_correct", "sum"),
    avg_time=("execution_time_seconds", "mean")
).reset_index()
category_stats["accuracy"] = (category_stats["correct"].astype(float) / category_stats["total_questions"] * 100).round(1)
for _, row in category_stats.iterrows():
    status_icon = "✓" if row["accuracy"] >= 80 else "⚠️" if row["accuracy"] >= 50 else "✗"
    print(f"  {status_icon} {row['category']:<15} | {row['accuracy']:>5.1f}% correct | "
          f"{int(row['correct'])}/{row['total_questions']} passed | avg {row['avg_time']:.1f}s")
weak_categories = category_stats[category_stats["accuracy"] < 80]
if len(weak_categories) > 0:
    print("\n  ⚠️  Categories below 80% accuracy need attention:")
    for _, row in weak_categories.iterrows():
        print(f"     • '{row['category']}' ({row['accuracy']}%) — review failures below")
else:
    print("\n  ✓ All categories are at or above 80% accuracy.")
incorrect_questions = eval_results[eval_results["judge_correct"] == False]
if len(incorrect_questions) > 0:
    print(f"\n  ✗ {len(incorrect_questions)} question(s) produced semantically incorrect SQL:")
    for _, row in incorrect_questions.iterrows():
        print(f"     • [{row['category']}/{row['difficulty']}] \"{row['question'][:60]}\"")
display(eval_results)

# COMMAND ----------

# DBTITLE 1,Results Visualization
# Results Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Genie Space Evaluation Results", fontsize=14, fontweight="bold", y=1.02)

# Chart 1: Correct/Incorrect by Category
ax1 = axes[0]
cats = eval_results.groupby("category").agg(
    correct=("judge_correct", lambda x: x.sum()),
    incorrect=("judge_correct", lambda x: (x == False).sum())
)
x_pos = np.arange(len(cats.index))
ax1.bar(x_pos, cats["correct"], 0.6, color="#2ecc71", label="Correct")
ax1.bar(x_pos, cats["incorrect"], 0.6, bottom=cats["correct"], color="#e74c3c", label="Incorrect")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(cats.index, rotation=30, ha="right", fontsize=9)
ax1.set_ylabel("Number of Questions")
ax1.set_title("Accuracy by Category (LLM Judge)")
ax1.legend(loc="upper right", fontsize=8)

# Chart 2: Overall Accuracy Gauge
ax2 = axes[1]
total = len(eval_results)
correct_count = int(eval_results["judge_correct"].sum())
rate = correct_count / total * 100 if total > 0 else 0
ax2.barh([0], [100], height=0.5, color="#ecf0f1", edgecolor="#bdc3c7")
bar_color = "#2ecc71" if rate >= 80 else "#f39c12" if rate >= 50 else "#e74c3c"
ax2.barh([0], [rate], height=0.5, color=bar_color)
ax2.set_xlim(0, 110)
ax2.set_yticks([])
ax2.set_xlabel("Semantic Accuracy (%)")
ax2.set_title(f"Overall Accuracy: {rate:.0f}% ({correct_count}/{total})")
ax2.axvline(x=80, color="#2c3e50", linestyle="--", alpha=0.5, linewidth=1)
ax2.text(81, 0.3, "80% target", fontsize=8, alpha=0.7)

# Chart 3: Response Time by Difficulty
ax3 = axes[2]
diff_order = ["easy", "medium", "hard"]
time_by_diff = eval_results.groupby("difficulty")["execution_time_seconds"].agg(["mean", "std"]).reindex(diff_order).fillna(0)
ax3.bar(range(len(diff_order)), time_by_diff["mean"], color=["#3498db", "#f39c12", "#e74c3c"],
        width=0.6, yerr=time_by_diff["std"], capsize=4, alpha=0.85)
ax3.set_xticks(range(len(diff_order)))
ax3.set_xticklabels(diff_order)
ax3.set_ylabel("Response Time (seconds)")
ax3.set_title("Avg Response Time by Difficulty")

plt.tight_layout()
plt.show()
print("\nTip: Compare before/after runs when you add instructions or example SQL.")

# COMMAND ----------

# DBTITLE 1,Improvement Demo — Re-test a Single Question
# Single Question Re-test
# =======================
# After identifying a failing question, use this cell to re-test without running the full suite.
# Workflow:
#   1. Update your Genie Space (add an instruction, example SQL, or join spec)
#   2. Set RETEST_INDEX to the failing test case (0-based index)
#   3. Run this cell to confirm the fix

RETEST_INDEX = 7  # 0-based: question #8 (complex multi-hop JOIN)
retest_case = test_cases[RETEST_INDEX]
retest_question = retest_case["question"]

print("="*60)
print(" IMPROVEMENT CYCLE — Single Question Re-test")
print("="*60)
print(f"\n  Question: '{retest_question}'")
print(f"  Category: {retest_case['category']} | Difficulty: {retest_case['difficulty']}")
print(f"\n  Sending to Genie...")

retest_result = ask_genie(SPACE_ID, retest_question)
retest_sql = extract_sql(retest_result)
print(f"\n  Status: {retest_result['status']} ({retest_result['elapsed_seconds']}s)")

border = "─" * 50
print("\n  ┌─ Generated SQL ────────────────────────────────────")
for line in (retest_sql.strip().split('\n') if retest_sql else ["(no SQL generated)"]):
    print(f"  │  {line}")
print(f"  └{border}")
print("\n  ┌─ Expected SQL ─────────────────────────────────────")
for line in retest_case['expected_sql'].strip().split('\n'):
    print(f"  │  {line.strip()}")
print(f"  └{border}")

if retest_sql:
    print("\n  Running LLM judge on this pair...")
    single_eval_data = pd.DataFrame({
        "inputs": [{"question": retest_question}],
        "outputs": [retest_sql],
        "expectations": [{"expected_sql": retest_case['expected_sql'].strip()}],
    })
    single_result = mlflow.genai.evaluate(data=single_eval_data, scorers=[sql_correctness_judge])
    single_df = single_result.tables["eval_results"]
    judge_value = single_df["sql_semantic_correctness/value"].iloc[0]
    print(f"  Judge verdict: {'✓ PASS' if judge_value else '✗ FAIL'}")
else:
    print("\n  ⚠️  No SQL generated — cannot run judge.")

print("\n" + "─"*60)
print(" 💡 If FAILED: update your Genie Space, then re-run this cell.")
print("    That's the Data Flywheel in action!")

# COMMAND ----------

# DBTITLE 1,Data Flywheel — Next Steps
# MAGIC %md
# MAGIC # Data Flywheel — Next Steps
# MAGIC
# MAGIC The evaluation above is not a one-time activity. It's the **engine** of a continuous improvement loop — the [Data Flywheel](https://www.sh-reya.com/blog/ai-engineering-flywheel/) applied to "Ask Your Data".
# MAGIC
# MAGIC ## The Iteration Loop
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
# MAGIC │  1. EVALUATE     │ →→→ │  2. IDENTIFY      │ →→→ │  3. IMPROVE       │
# MAGIC │  Run harness    │     │  Find patterns   │     │  Add instructions│
# MAGIC │  Score results  │     │  in failures     │     │  & example SQL   │
# MAGIC └─────────────────┘     └──────────────────┘     └─────────────────┘
# MAGIC         ↑                                               │
# MAGIC         │     ┌──────────────────┐                     │
# MAGIC         └─── │  4. RE-EVALUATE  │ ←←←←←←←←←←←←←←←←←←←←┘
# MAGIC               │  Measure delta   │
# MAGIC               │  Track in MLflow │
# MAGIC               └──────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ## Step-by-Step Process
# MAGIC
# MAGIC ### 1. Review Failed Questions
# MAGIC Examine the `eval_results` DataFrame above. For each failed or incorrect answer:
# MAGIC - Was the question ambiguous? → Add text instructions to the Genie Space.
# MAGIC - Did Genie pick the wrong table? → Add join specifications or table descriptions.
# MAGIC - Did Genie misinterpret business jargon? → Add a text instruction defining that term.
# MAGIC
# MAGIC ### 2. Identify Patterns
# MAGIC Group failures by `category`. Common patterns:
# MAGIC - **Aggregation failures**: Add “When asked about revenue, use SUM(amount)” as an instruction.
# MAGIC - **Filter failures**: Enable `get_example_values` on ambiguous columns.
# MAGIC - **Join failures**: Add explicit join specifications.
# MAGIC - **Time-based failures**: Add instructions like “‘last month’ means the previous calendar month”.
# MAGIC
# MAGIC ### 3. Improve the Genie Space
# MAGIC
# MAGIC | Lever | When to use |
# MAGIC |---|---|
# MAGIC | **Text instructions** | Business jargon, time conventions, table ownership |
# MAGIC | **Example SQL** | Wrong join pattern or aggregation structure |
# MAGIC | **Column configurations** | Filter values not recognized |
# MAGIC | **Join specifications** | Multi-table join failing or missing |
# MAGIC
# MAGIC **Always prefer the most targeted lever.** Column config and join specs first; broad instructions last.
# MAGIC
# MAGIC ### 4. Re-evaluate & Track
# MAGIC Re-run this notebook after each improvement. Each run is logged to MLflow:
# MAGIC
# MAGIC ```python
# MAGIC runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
# MAGIC display(runs[["run_id", "start_time", "params.num_test_cases",
# MAGIC               "metrics.sql_semantic_correctness/rating/average"]])
# MAGIC ```
# MAGIC
# MAGIC ## Further Reading
# MAGIC
# MAGIC - [Part 01: The Semantic Layer: Shared Business Meaning as Infrastructure](https://www.databricks.com/blog/semantic-layer-architecture-components-design-patterns-and-ai-integration)
# MAGIC - [Part 02: Managing the Hidden Technical Debt of Generative AI](https://www.databricks.com/blog/hidden-technical-debt-genai-systems)
# MAGIC - [Part 03: Reliable by Design: Building an Evaluation Harness for Databricks Genie](https://www.databricks.com/blog/reliable-by-design-evaluation-harness-databricks-genie)
# MAGIC - [Part 04: Closing the Loop: A Prompt Iteration Workflow for Databricks Genie](#)

# COMMAND ----------


