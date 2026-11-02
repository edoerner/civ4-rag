# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,A/B/C experiment design
# MAGIC %md
# MAGIC # Sprint 5 — Configuration comparison experiments
# MAGIC
# MAGIC Runs a controlled **A/B/C comparison of RAG configurations** on the same 24-question evaluation set, logging every run to MLflow and persisting results to Delta.
# MAGIC
# MAGIC ## Experiment design
# MAGIC
# MAGIC | Config | top_k | prompt | temperature | What it tests |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | A `A_baseline_top5_prompt_v1` | 5 | v1 (Sprint 4 production prompt) | 0.1 | Production baseline (run in notebook 03) |
# MAGIC | B `B_top10_prompt_v1` | 10 | v1 | 0.1 | More retrieved context |
# MAGIC | C `C_top5_prompt_v2` | 5 | v2 (concise bullet-point answers) | 0.1 | Prompt phrasing |
# MAGIC
# MAGIC Only **one parameter changes per experiment** relative to the baseline, so metric differences can be attributed to that parameter. All three configurations share:
# MAGIC
# MAGIC * the same evaluation set (`civ4_eval_questions`),
# MAGIC * the same retrieval method (Vector Search ANN over `civ4_chunks_embeddings_index`),
# MAGIC * the same generation model (`databricks-qwen3-next-80b-a3b-instruct`) and temperature 0.1,
# MAGIC * the same five LLM judges.
# MAGIC
# MAGIC ## Known limitations (documented)
# MAGIC
# MAGIC * **Judge nondeterminism**: LLM judges can score slightly differently across runs. Differences between configs of less than ~0.1 should not be treated as conclusive.
# MAGIC * **Sample size**: 24 questions — the comparison is directional, not statistically significant.
# MAGIC * Document-level retrieval ground truth makes `precision@K` a lower bound.
# MAGIC
# MAGIC **Outputs:** rows in `civ4_generation_eval` / `civ4_evaluation_summary` (MERGE per config — idempotent) + one MLflow run per config.

# COMMAND ----------

# DBTITLE 1,Setup — module, questions, verify Config A exists
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
evu.bootstrap_eval_tables()

# Verify Config A is already persisted (notebook 03 must run first)
a_count = spark.sql("""
    SELECT count(*) AS n FROM workspace.gcivilization.civ4_generation_eval
    WHERE eval_run_config = 'A_baseline_top5_prompt_v1'
""").collect()[0].n
assert a_count == 24, f"Config A has {a_count}/24 rows — run notebook 03 first"
print(f"Config A present (24/24 rows). Configs B and C will be run against the same {len(questions)} questions.")

# COMMAND ----------

# DBTITLE 1,Experiment B — top_k=10
# Experiment B — top_k=10 vs baseline top_k=5 (prompt v1, temperature 0.1).
# Tests whether more retrieved context improves judge scores.

run_id_b, metrics_b = evu.run_and_persist("B_top10_prompt_v1", questions)
print(f"Experiment B done | MLflow run {run_id_b}")
for k in ["recall_at_5", "recall_at_10", "precision_at_5", "precision_at_10", "mrr",
          "faithfulness", "context_relevance", "answer_relevance",
          "citation_correctness", "answer_correctness"]:
    print(f"{k:22s} {metrics_b[k]:.3f}")

# COMMAND ----------

# DBTITLE 1,Experiment C — prompt v2
# Experiment C — prompt v2 (concise bullet-point answers) vs baseline v1
# (top_k=5, temperature 0.1). Tests whether prompt phrasing changes quality.

run_id_c, metrics_c = evu.run_and_persist("C_top5_prompt_v2", questions)
print(f"Experiment C done | MLflow run {run_id_c}")
for k in ["recall_at_1", "recall_at_5", "mrr",
          "faithfulness", "context_relevance", "answer_relevance",
          "citation_correctness", "answer_correctness"]:
    print(f"{k:22s} {metrics_c[k]:.3f}")

# COMMAND ----------

# DBTITLE 1,Compare A / B / C
# Side-by-side comparison of all three configurations (from the Delta summary table).

display(evu.compare_configs())

# COMMAND ----------

# DBTITLE 1,MLflow run evidence
# MLflow evidence: all full_pipeline runs in the experiment, with key params + metrics.

import mlflow
runs = evu.list_mlflow_runs(stage="full_pipeline")
assert len(runs) >= 3, f"expected >= 3 full_pipeline runs, found {len(runs)}"
display(runs)

# All runs must point at the same eval dataset version (controlled comparison)
from mlflow.tracking import MlflowClient
exp = mlflow.get_experiment_by_name(evu.EXPERIMENT_NAME)
versions = set()
for _, r in runs.iterrows():
    versions.add(MlflowClient().get_run(r.run_id).data.params.get("eval_dataset_version"))
assert len(versions) == 1, f"runs used different dataset versions: {versions}"
print(f"OK: {len(runs)} runs, all on dataset version {versions.pop()}")