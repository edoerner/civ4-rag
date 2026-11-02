# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Generation evaluation — methodology
# MAGIC %md
# MAGIC # Sprint 5 — Generation evaluation
# MAGIC
# MAGIC Evaluates the **full RAG pipeline** (Sprint 3 retrieval + Sprint 4 generation) on the 24 curated questions, using **five LLM judges**.
# MAGIC
# MAGIC ## Method
# MAGIC
# MAGIC For each question, the pipeline is run with a configurable `(top_k, system_prompt, temperature)` — otherwise identical to the production `rag_query()` path (same `retrieve`, `build_context`, prompt structure, model endpoint, and `build_sources`). The generated answer is then scored by five MLflow genai judges (Databricks-hosted judge backend), each returning **yes → 1 / no → 0**, averaged over questions:
# MAGIC
# MAGIC | Judge | Type | What it checks |
# MAGIC | --- | --- | --- |
# MAGIC | `faithfulness` | built-in `is_grounded` | Is the answer supported by the retrieved context? (anti-hallucination) |
# MAGIC | `context_relevance` | built-in `is_context_relevant` | Is the retrieved context relevant to the question? (retrieval quality, generation-side view) |
# MAGIC | `answer_relevance` | custom | Does the answer address the question? |
# MAGIC | `citation_correctness` | custom | Do the cited `[Source N]` labels exist in the retrieved list AND support the claims attributed to them? |
# MAGIC | `answer_correctness` | built-in `is_correct` | Does the answer match the curated `expected_answer` ground truth? |
# MAGIC
# MAGIC This notebook runs **Configuration A** — the production baseline (`top_k=5`, prompt v1, temperature 0.1). Configurations B and C are compared in notebook `04_mlflow_experiments`.
# MAGIC
# MAGIC ## Limitations (documented)
# MAGIC
# MAGIC * **LLM judges are nondeterministic** — repeated runs of the same configuration can produce slightly different scores. Retrieval metrics are deterministic.
# MAGIC * Judges are yes/no, so partial quality differences within an answer are not captured.
# MAGIC * `answer_correctness` compares against short factual ground-truth summaries, so it measures factual coverage, not style.
# MAGIC
# MAGIC **Outputs:** `workspace.gcivilization.civ4_generation_eval` + `civ4_evaluation_summary` (idempotent MERGE per config) + one MLflow run (`stage = full_pipeline`).

# COMMAND ----------

# DBTITLE 1,Setup — module, questions, eval tables
import sys, os, importlib

for _eval_dir in (
    "/Workspace/Repos/endoerner@gmail.com/civ4-rag/05_evaluation",
    "/Repos/endoerner@gmail.com/civ4-rag/05_evaluation",
    "/Workspace/Users/endoerner@gmail.com/civ4-rag/05_evaluation",
):
    if os.path.isdir(_eval_dir):
        sys.path.insert(0, _eval_dir)
        break
importlib.invalidate_caches()
import rag_eval_utils as evu

# Imported modules don't inherit notebook builtins: inject the notebook's Spark
# session into the module so its functions can use `spark` at call time.
evu.spark = spark

questions = evu.load_eval_questions()
print(f"Loaded {len(questions)} questions | version {questions['eval_dataset_version'].iloc[0]}")

evu.bootstrap_eval_tables()

config = evu.CONFIGS["A_baseline_top5_prompt_v1"]
print(f"Config A: top_k={config['top_k']}, prompt={config['prompt_version']}, temperature={config['temperature']}")
print(f"Generation model: {__import__('rag_utils').LLM_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,Run Configuration A (baseline) and persist
# Run Configuration A (production baseline) end-to-end:
# 24 questions x (retrieve -> generate -> 5 judges), MLflow run, Delta MERGE.
#
# ~2-4 minutes: each question costs 1 retrieval + 1 generation + 5 judge calls
# (judges run Databricks-hosted LLMs). Cells below are idempotent — rerunning
# MERGEs new results over the same (config, question) rows.

run_id, metrics = evu.run_and_persist("A_baseline_top5_prompt_v1", questions)
print(f"Config A done | MLflow run {run_id}")

for k in ["recall_at_1", "recall_at_5", "recall_at_10", "precision_at_5", "mrr",
          "faithfulness", "context_relevance", "answer_relevance",
          "citation_correctness", "answer_correctness", "judge_error_rate"]:
    # depth-10 metrics only exist when the config retrieves at top_k >= 10
    if k in metrics:
        print(f"{k:22s} {metrics[k]:.3f}")

# COMMAND ----------

# DBTITLE 1,Inspect Config A per-question results
# Inspect per-question results for Config A: judge scores, judge errors, retrieval.

import pandas as pd

rows = spark.sql("""
    SELECT question_id, category, difficulty,
           judge_faithfulness, judge_context_relevance, judge_answer_relevance,
           judge_citation_correctness, judge_answer_correctness,
           recall_at_5, mrr
    FROM workspace.gcivilization.civ4_generation_eval
    WHERE eval_run_config = 'A_baseline_top5_prompt_v1'
    ORDER BY question_id
""").toPandas()

assert len(rows) == 24, f"expected 24 rows, got {len(rows)}"
judge_cols = [c for c in rows.columns if c.startswith("judge_") and not c.endswith("_error")]
nulls = {c: int(rows[c].isna().sum()) for c in judge_cols if rows[c].isna().any()}
assert not nulls, f"null judge scores: {nulls}"
print(f"OK: 24/24 rows, all judge scores non-null")
display(rows)

# COMMAND ----------

# DBTITLE 1,Inspect lowest-scoring answers (judge sanity check)
# Inspect the weakest dimension for Config A: the answers behind the lowest-scoring
# judge, with the question and answer side by side (spot-check for judge sanity).

low = spark.sql("""
    SELECT question_id, question, answer, judge_citation_correctness, judge_answer_correctness
    FROM workspace.gcivilization.civ4_generation_eval
    WHERE eval_run_config = 'A_baseline_top5_prompt_v1'
      AND (judge_citation_correctness = 0 OR judge_answer_correctness = 0)
    ORDER BY question_id
""").toPandas()
print(f"{len(low)} questions with citation or correctness failures")
for _, r in low.iterrows():
    print(f"\n=== {r.question_id} (citation={r.judge_citation_correctness}, correctness={r.judge_answer_correctness})")
    print("Q:", r.question)
    print("A:", r.answer[:400])