# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Retrieval evaluation — methodology
# MAGIC %md
# MAGIC # Sprint 5 — Retrieval evaluation
# MAGIC
# MAGIC Evaluates the Sprint 3 retrieval layer **independently from generation**: can the retriever find the evidence needed to answer each question?
# MAGIC
# MAGIC ## Method
# MAGIC
# MAGIC Each of the 24 evaluation questions is issued against the Vector Search index (`workspace.gcivilization.civ4_chunks_embeddings_index`, self-managed `databricks-gte-large-en` embeddings) at a **fixed retrieval depth of 10**. Metrics are then computed at K = 1, 3, 5, 10 by slicing the ranked result list.
# MAGIC
# MAGIC ## Metric definitions
# MAGIC
# MAGIC * **Relevance is document-level**: a retrieved chunk is *relevant* if its `document_id` is one of the question's `expected_document_ids`. Exact chunk-level ground truth is not specified (chunks within a document overlap heavily).
# MAGIC * **Recall@K** — 1 if at least one expected document appears in the top K retrieved results, else 0 (hit rate, averaged over questions).
# MAGIC * **Precision@K** — (# retrieved chunks in the top K that come from expected documents) / K. Because ground truth is document-level, this is a lower bound on true precision.
# MAGIC * **MRR** — mean reciprocal rank of the first relevant result (1/rank; 0 if no relevant result retrieved).
# MAGIC * **Missing ground truth** — none: every question has validated `expected_document_ids` (notebook 01). An empty retrieval is scored 0 for all metrics (none occurred in the recorded run).
# MAGIC
# MAGIC **Outputs:** `workspace.gcivilization.civ4_retrieval_eval` (overwrite, deterministic) + one MLflow run (`stage = retrieval`).

# COMMAND ----------

# DBTITLE 1,Setup — import evaluation module, load questions
import sys, os, importlib

# Import the shared evaluation module (repo or workspace staging location)
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
import rag_utils

# Imported modules don't inherit notebook builtins: inject the notebook's Spark
# session into the module so its functions can use `spark` at call time.
evu.spark = spark

questions = evu.load_eval_questions()
EVAL_DATASET_VERSION = questions["eval_dataset_version"].iloc[0]
print(f"Loaded {len(questions)} questions | version {EVAL_DATASET_VERSION}")
print(f"Index: {rag_utils.INDEX_NAME} | embedding model: {rag_utils.EMBEDDING_MODEL}")

# COMMAND ----------

# DBTITLE 1,Run retrieval at depth 10 and compute metrics
# Retrieve at fixed depth 10 for all questions (embedding calls only — no LLM),
# then compute document-level retrieval metrics per question.

import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

RETRIEVAL_DEPTH = 10

def retrieve_one(q):
    return q["question_id"], q, rag_utils.retrieve(q["question"], top_k=RETRIEVAL_DEPTH)

t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    retrieval_results = {qid: (q, chunks)
                         for qid, q, chunks in ex.map(retrieve_one, questions.to_dict("records"))}
print(f"Retrieved for {len(retrieval_results)} questions in {time.time()-t0:.1f}s")

per_question_rows = []
for qid in sorted(retrieval_results):
    q, chunks = retrieval_results[qid]
    m = evu.retrieval_metrics_for_question(q["expected_document_ids"], chunks)
    per_question_rows.append({
        "question_id": qid, "question": q["question"],
        "category": q["category"], "difficulty": q["difficulty"],
        "expected_document_ids": q["expected_document_ids"],
        "retrieved_document_ids": [c.get("document_id") for c in chunks],
        "retrieved_chunk_ids": [c.get("chunk_id") for c in chunks],
        "retrieved_scores": [float(c.get("score", 0.0)) for c in chunks],
        "recall_at_1": m["recall_at_1"], "recall_at_3": m["recall_at_3"],
        "recall_at_5": m["recall_at_5"], "recall_at_10": m["recall_at_10"],
        "precision_at_5": m["precision_at_5"], "precision_at_10": m["precision_at_10"],
        "mrr": m["mrr"], "first_relevant_rank": m["first_relevant_rank"],
        "expected_docs_found": m["expected_docs_found"], "n_retrieved": m["n_retrieved"],
    })

metrics_summary = {m: sum(r[m] for r in per_question_rows) / len(per_question_rows)
                   for m in ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
                             "precision_at_5", "precision_at_10", "mrr"]}
for k, v in metrics_summary.items():
    print(f"{k:18s} {v:.3f}")

# Sanity checks: no empty retrievals, result schema consistent
empty = [r["question_id"] for r in per_question_rows if r["n_retrieved"] == 0]
assert not empty, f"empty retrieval for {empty}"
assert all(len(r["retrieved_document_ids"]) == r["n_retrieved"] for r in per_question_rows)
print("Retrieval schema + non-empty checks passed")

# COMMAND ----------

# DBTITLE 1,Persist results + MLflow retrieval run
# Persist per-question results (deterministic overwrite) and log one MLflow
# retrieval run with parameters + metrics + artifacts.

import mlflow, tempfile, os
from datetime import datetime, timezone

config_name = "retrieval_depth10"
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

pdf = pd.DataFrame(per_question_rows)
pdf["eval_run_config"] = config_name
pdf["eval_dataset_version"] = EVAL_DATASET_VERSION
pdf["eval_timestamp"] = now
spark.createDataFrame(pdf).write.mode("overwrite").saveAsTable(evu.RETRIEVAL_EVAL_TABLE)
print(f"Saved {evu.RETRIEVAL_EVAL_TABLE} ({len(pdf)} rows)")

mlflow.set_experiment(evu.EXPERIMENT_NAME)
with mlflow.start_run(run_name="retrieval_eval_depth10") as run:
    mlflow.set_tags({"stage": "retrieval", "config_name": config_name})
    mlflow.log_params({
        "retrieval_depth": RETRIEVAL_DEPTH,
        "embedding_model": rag_utils.EMBEDDING_MODEL,
        "retrieval_method": "vector_search_ann",
        "index_name": rag_utils.INDEX_NAME,
        "eval_dataset_version": EVAL_DATASET_VERSION,
        "n_questions": len(per_question_rows),
        "relevance_level": "document",
    })
    mlflow.log_metrics(metrics_summary)
    tmp = tempfile.mkdtemp()
    pdf.drop(columns=["question"]).to_json(
        os.path.join(tmp, "retrieval_eval_per_question.json"), orient="records", indent=2)
    questions.to_csv(os.path.join(tmp, "eval_questions.csv"), index=False)
    mlflow.log_artifacts(tmp)
    print("MLflow retrieval run:", run.info.run_id)

# Per-category retrieval quality
display(pdf.groupby("category")[["recall_at_5", "recall_at_10", "mrr"]]
        .mean().round(3).reset_index())

# COMMAND ----------

