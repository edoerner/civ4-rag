"""
rag_eval_utils.py — Shared evaluation framework for the Civ4 RAG pipeline (Sprint 5).

Imported by the 05_evaluation notebooks so evaluation logic lives in ONE place,
mirroring the rag_utils.py pattern used in Sprint 4.

The evaluation layer REUSES the Sprint 3/4 implementation (rag_utils) and only
parameterizes top_k, temperature, and the prompt version. It does NOT duplicate
the RAG pipeline: retrieval, context building, generation and source building
all call Sprint 4 functions directly.

Metric definitions (see README.md in this folder for the full methodology):

  Retrieval — relevance is DOCUMENT-LEVEL:
    relevant result : a retrieved chunk whose document_id is in the question's
                      expected_document_ids
    Recall@K       : 1 if at least one expected document appears in the top K
                     retrieved results, else 0 (averaged over questions)
    Precision@K    : (# top-K retrieved chunks from expected documents) / K
    MRR            : 1 / rank of the first relevant result (0 if none)
    Missing GT     : none — every question has validated expected_document_ids
                     (notebook 01); empty retrievals are scored 0 everywhere.

  Generation — MLflow genai judges (Databricks-hosted LLM judges),
  yes -> 1 / no -> 0, averaged over questions:
    faithfulness         : is_grounded — is the answer supported by the
                           retrieved context?
    context_relevance    : is_context_relevant — is the retrieved context
                           relevant to the question?
    answer_relevance     : custom judge — does the answer address the question?
    citation_correctness : custom judge — do the cited [Source N] labels exist
                           in the retrieved list AND support the claims
                           attributed to them?
    answer_correctness   : is_correct — does the answer match the curated
                           expected_answer ground truth?

LIMITATION: LLM judges are nondeterministic — repeated runs of the SAME
configuration can produce slightly different scores. Retrieval metrics are
deterministic (embedding + ANN search).
"""

import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import mlflow
import pandas as pd
from mlflow.genai import judges

# ============================================================
# Import the Sprint 4 RAG module (both serverless /Workspace mount
# and classic cluster /Repos path are supported)
# ============================================================
for _rag_dir in (
    "/Workspace/Repos/endoerner@gmail.com/civ4-rag/04_generation",
    "/Repos/endoerner@gmail.com/civ4-rag/04_generation",
):
    if os.path.isdir(_rag_dir):
        sys.path.insert(0, _rag_dir)
        break
import rag_utils  # noqa: E402  (Sprint 4 shared module)

# ============================================================
# Configuration
# ============================================================
CATALOG = "workspace"
SCHEMA = "gcivilization"
EVAL_QUESTIONS_TABLE = f"{CATALOG}.{SCHEMA}.civ4_eval_questions"
RETRIEVAL_EVAL_TABLE = f"{CATALOG}.{SCHEMA}.civ4_retrieval_eval"
GENERATION_EVAL_TABLE = f"{CATALOG}.{SCHEMA}.civ4_generation_eval"
SUMMARY_TABLE = f"{CATALOG}.{SCHEMA}.civ4_evaluation_summary"

EXPERIMENT_NAME = "/Users/endoerner@gmail.com/civ4-rag-evaluation"
RETRIEVAL_DEPTH = 10  # fixed retrieval depth used by notebook 02
K_VALUES = (1, 3, 5, 10)

# ---- Prompt versions ----------------------------------------------
# v1 = the Sprint 4 baseline system prompt, unchanged.
PROMPT_V1 = rag_utils.SYSTEM_PROMPT

# v2 = evaluation variant: concise bullet-point answers.
PROMPT_V2 = """\
You are a Civilization IV strategy advisor. Answer the user's question about \
Civilization IV strategy using ONLY the provided context from retrieved \
strategy-guide documents.

Instructions:
1. Answer with 3-6 short bullet points. Put each point on its own line.
2. Use only information found in the CONTEXT section. Do not invent facts.
3. If the context is insufficient to answer, say so explicitly instead of \
guessing.
4. Cite the sources you use inline with their [Source N] label.
5. Do not cite sources that do not appear in the CONTEXT section.
"""

# ---- RAG configurations compared in Sprint 5 ----------------------
CONFIGS = {
    "A_baseline_top5_prompt_v1": dict(
        config_name="A_baseline_top5_prompt_v1", top_k=5,
        prompt_version="v1", system_prompt=PROMPT_V1, temperature=0.1),
    "B_top10_prompt_v1": dict(
        config_name="B_top10_prompt_v1", top_k=10,
        prompt_version="v1", system_prompt=PROMPT_V1, temperature=0.1),
    "C_top5_prompt_v2": dict(
        config_name="C_top5_prompt_v2", top_k=5,
        prompt_version="v2", system_prompt=PROMPT_V2, temperature=0.1),
}

# ============================================================
# LLM judges (MLflow genai; Databricks-hosted judge backend)
# ============================================================
ANSWER_RELEVANCE_JUDGE = judges.make_judge(
    name="answer_relevance",
    instructions=(
        "You are evaluating whether a generated answer addresses the user's question.\n\n"
        "INPUTS (a dict containing 'question'):\n{{ inputs }}\n\n"
        "OUTPUTS (the generated answer to evaluate):\n{{ outputs }}\n\n"
        "Return 'yes' if the answer directly addresses the question in INPUTS and is "
        "on-topic. A partial but on-topic answer counts as 'yes'. Return 'no' if the "
        "answer is off-topic, evasive, or does not respond to the question."
    ),
)

CITATION_CORRECTNESS_JUDGE = judges.make_judge(
    name="citation_correctness",
    instructions=(
        "You are evaluating a RAG answer's source citations.\n\n"
        "INPUTS (a dict containing 'question' and 'sources': the retrieved sources "
        "with their [Source N] labels, titles, and full content — the ONLY "
        "sources that may be cited):\n{{ inputs }}\n\n"
        "OUTPUTS (the generated answer to evaluate):\n{{ outputs }}\n\n"
        "Return 'yes' if every [Source N] label used in the answer refers to a source "
        "in the retrieved-sources list AND the cited source excerpts plausibly support "
        "the specific claims attributed to them. Return 'no' if the answer cites labels "
        "not present in the list, misattributes claims to the wrong source, or contains "
        "no [Source N] citation at all."
    ),
)


def judge_to_score(fb):
    """Map an MLflow judge Feedback to (1|0|None, error-string)."""
    if fb is None:
        return None, "no feedback"
    if getattr(fb, "error", None):
        return None, str(fb.error)
    v = getattr(fb.value, "value", fb.value)
    s = str(v).strip().lower()
    if s in ("yes", "true", "1"):
        return 1, None
    if s in ("no", "false", "0"):
        return 0, None
    return None, f"unexpected value {v!r}"


# ============================================================
# Configurable RAG invocation — thin adapters over rag_utils
# ============================================================
def build_user_prompt(query, retrieved_chunks):
    """Same context block Sprint 4 builds (prompt version only changes the SYSTEM message)."""
    return f"CONTEXT:\n{rag_utils.build_context(retrieved_chunks)}\n\nQUESTION:\n{query}\n"


def generate_answer_with(system_prompt, user_prompt, temperature):
    """Sprint 4 generate_answer with a parameterized temperature."""
    response = rag_utils.deploy_client.predict(
        endpoint=rag_utils.LLM_ENDPOINT,
        inputs={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": rag_utils.LLM_MAX_TOKENS,
            "temperature": temperature,
        },
    )
    return response["choices"][0]["message"]["content"]


# ============================================================
# Retrieval metrics (document-level relevance)
# ============================================================
def retrieval_metrics_for_question(expected_doc_ids, retrieved_chunks,
                                    k_values=K_VALUES):
    """Compute Recall@K / Precision@K / MRR for one question.

    Relevance is DOCUMENT-LEVEL: a retrieved chunk counts as relevant when its
    document_id is one of the question's expected_document_ids.
    """
    expected = set(expected_doc_ids)
    retrieved_docs = [c.get("document_id") for c in retrieved_chunks]
    row = {}
    for k in k_values:
        top_k_docs = retrieved_docs[:k]
        row[f"recall_at_{k}"] = int(any(d in expected for d in top_k_docs)) if top_k_docs else 0
        row[f"precision_at_{k}"] = sum(1 for d in top_k_docs if d in expected) / k
    first_relevant_rank = next(
        (i + 1 for i, d in enumerate(retrieved_docs) if d in expected), None)
    row["mrr"] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    row["first_relevant_rank"] = first_relevant_rank
    row["expected_docs_found"] = sorted(expected & set(retrieved_docs))
    row["n_retrieved"] = len(retrieved_docs)
    return row


def add_uniform_retrieval_metrics(rows, top_k):
    """Attach a uniform set of retrieval-metric columns to per-question rows.

    Columns for K > top_k are set to None (not applicable), so all
    configurations share one table schema.
    """
    for r in rows:
        expected = set(r["expected_document_ids"])
        docs = r["retrieved_document_ids"]
        for kk in K_VALUES:
            r[f"recall_at_{kk}"] = (
                int(any(d in expected for d in docs[:kk])) if kk <= top_k else None)
        for kk in (5, 10):
            r[f"precision_at_{kk}"] = (
                sum(1 for d in docs[:kk] if d in expected) / kk
                if kk <= top_k else None)
        rank = next((i + 1 for i, d in enumerate(docs) if d in expected), None)
        r["mrr"] = 1.0 / rank if rank else 0.0
    return rows


def uniform_metrics(metrics, top_k):
    """Normalize an aggregate-metrics dict to the uniform metric column set."""
    out = dict(metrics)
    for kk in K_VALUES:
        out.setdefault(f"recall_at_{kk}", None)
    for kk in (5, 10):
        out.setdefault(f"precision_at_{kk}", None)
    return out


# ============================================================
# Full-pipeline evaluation (retrieval + generation + judges)
# ============================================================
JUDGE_NAMES = ("faithfulness", "context_relevance", "answer_relevance",
               "citation_correctness", "answer_correctness")


def evaluate_one_question(q, config):
    """Run the full pipeline for ONE question and judge the result.

    Stages (reusing Sprint 4 building blocks):
      1. retrieve(query, top_k)          — same retrieval call as production
      2. build context / user prompt     — Sprint 4 build_context
      3. generate with (prompt, temperature) from the config
      4. build_sources                   — Sprint 4 source references
      5. five LLM judges (see module docstring)
    """
    chunks = rag_utils.retrieve(q["question"], top_k=config["top_k"])
    assert chunks, f"empty retrieval for {q['question_id']}"
    assert all(c.get("document_id") for c in chunks), "retrieved chunk missing document_id"

    user_prompt = build_user_prompt(q["question"], chunks)
    answer = generate_answer_with(config["system_prompt"], user_prompt, config["temperature"])
    assert answer and answer.strip(), f"empty answer for {q['question_id']}"

    sources = rag_utils.build_sources(chunks)
    assert sources, f"no sources for {q['question_id']}"

    context_texts = [c["chunk_text"] for c in chunks]
    # Source list for the citation judge mirrors the EXACT chunk-level
    # [Source N] labels the generator saw (full text, so claim support can
    # actually be verified).
    sources_str = "\n\n".join(
        f"[Source {i+1}] {c['title']} ({c['chunk_id']}) — full content: {c['chunk_text']}"
        for i, c in enumerate(chunks))

    feedbacks = {
        "faithfulness": judges.is_grounded(
            request=q["question"], response=answer, context=context_texts),
        "context_relevance": judges.is_context_relevant(
            request=q["question"], context=context_texts),
        "answer_relevance": ANSWER_RELEVANCE_JUDGE(
            inputs={"question": q["question"]}, outputs=answer),
        "citation_correctness": CITATION_CORRECTNESS_JUDGE(
            inputs={"question": q["question"], "sources": sources_str}, outputs=answer),
        "answer_correctness": judges.is_correct(
            request=q["question"], response=answer, expected_response=q["expected_answer"]),
    }
    scores = {}
    errors = {}
    for name, fb in feedbacks.items():
        scores[name], errors[name] = judge_to_score(fb)

    return {
        "question_id": q["question_id"], "question": q["question"],
        "category": q["category"], "difficulty": q["difficulty"],
        "answer": answer,
        "retrieved_document_ids": [c.get("document_id") for c in chunks],
        "retrieved_chunk_ids": [c.get("chunk_id") for c in chunks],
        "retrieved_scores": [float(c.get("score", 0.0)) for c in chunks],
        "expected_document_ids": q["expected_document_ids"],
        "n_sources": len(sources),
        **{f"judge_{k}": v for k, v in scores.items()},
        **{f"judge_{k}_error": errors[k] for k in errors},
    }


def run_full_evaluation(config_key, questions_pdf, max_workers=3):
    """Evaluate one RAG configuration over the full question set.

    Returns (config, rows, metrics, elapsed_seconds).
    """
    config = CONFIGS[config_key]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(lambda q: evaluate_one_question(q, config),
                           questions_pdf.to_dict("records")))
    elapsed = time.time() - t0

    n = len(rows)
    metrics = {}
    add_uniform_retrieval_metrics(rows, config["top_k"])
    for kk in K_VALUES:
        if kk <= config["top_k"]:
            metrics[f"recall_at_{kk}"] = sum(r[f"recall_at_{kk}"] for r in rows) / n
    for kk in (5, 10):
        if kk <= config["top_k"]:
            metrics[f"precision_at_{kk}"] = sum(r[f"precision_at_{kk}"] for r in rows) / n
    metrics["mrr"] = sum(r["mrr"] for r in rows) / n
    for j in JUDGE_NAMES:
        valid = [r[f"judge_{j}"] for r in rows if r[f"judge_{j}"] is not None]
        metrics[j] = sum(valid) / len(valid) if valid else 0.0
        metrics[f"{j}_answered"] = len(valid) / n
    metrics["judge_error_rate"] = sum(
        1 for r in rows for j in JUDGE_NAMES if r[f"judge_{j}"] is None) / (n * len(JUDGE_NAMES))
    return config, rows, metrics, elapsed


# ============================================================
# Delta persistence (idempotent MERGE per config)
# ============================================================
def _table_has_columns(table, cols):
    existing = {c.name for c in spark.catalog.listColumns(table)}
    return all(c in existing for c in cols)


def bootstrap_eval_tables():
    """Create the Sprint 5 result tables if they do not exist, and evolve
    their schema to the uniform metric column set. Never touches the
    Sprint 2-4 tables."""
    if not spark.catalog.tableExists(GENERATION_EVAL_TABLE):
        spark.sql(f"""
            CREATE TABLE {GENERATION_EVAL_TABLE} (
                question_id STRING, question STRING, category STRING, difficulty STRING,
                answer STRING,
                retrieved_document_ids ARRAY<STRING>, retrieved_chunk_ids ARRAY<STRING>,
                retrieved_scores ARRAY<DOUBLE>, expected_document_ids ARRAY<STRING>,
                n_sources INT,
                judge_faithfulness INT, judge_context_relevance INT,
                judge_answer_relevance INT, judge_citation_correctness INT,
                judge_answer_correctness INT,
                judge_faithfulness_error STRING, judge_context_relevance_error STRING,
                judge_answer_relevance_error STRING, judge_citation_correctness_error STRING,
                judge_answer_correctness_error STRING,
                recall_at_1 INT, recall_at_3 INT, recall_at_5 INT, recall_at_10 INT,
                precision_at_5 DOUBLE, precision_at_10 DOUBLE, mrr DOUBLE,
                eval_run_config STRING, top_k INT, prompt_version STRING,
                temperature DOUBLE, generation_model STRING,
                eval_dataset_version STRING, mlflow_run_id STRING, eval_timestamp STRING
            )
        """)
        print(f"created {GENERATION_EVAL_TABLE}")
    if not _table_has_columns(GENERATION_EVAL_TABLE, ["recall_at_10", "precision_at_10"]):
        spark.sql(f"ALTER TABLE {GENERATION_EVAL_TABLE} ADD COLUMNS (recall_at_10 INT, precision_at_10 DOUBLE)")

    if not spark.catalog.tableExists(SUMMARY_TABLE):
        spark.sql(f"""
            CREATE TABLE {SUMMARY_TABLE} (
                recall_at_1 DOUBLE, recall_at_3 DOUBLE, recall_at_5 DOUBLE, recall_at_10 DOUBLE,
                precision_at_5 DOUBLE, precision_at_10 DOUBLE, mrr DOUBLE,
                faithfulness DOUBLE, context_relevance DOUBLE, answer_relevance DOUBLE,
                citation_correctness DOUBLE, answer_correctness DOUBLE,
                faithfulness_answered DOUBLE, context_relevance_answered DOUBLE,
                answer_relevance_answered DOUBLE, citation_correctness_answered DOUBLE,
                answer_correctness_answered DOUBLE, judge_error_rate DOUBLE,
                config_name STRING, top_k INT, prompt_version STRING, temperature DOUBLE,
                generation_model STRING, embedding_model STRING, n_questions INT,
                eval_dataset_version STRING, mlflow_run_id STRING, eval_timestamp STRING
            )
        """)
        print(f"created {SUMMARY_TABLE}")
    if not _table_has_columns(SUMMARY_TABLE, ["recall_at_10", "precision_at_10"]):
        spark.sql(f"ALTER TABLE {SUMMARY_TABLE} ADD COLUMNS (recall_at_10 DOUBLE, precision_at_10 DOUBLE)")


def finalize_config_tables(config, rows, metrics, run_id, eval_dataset_version):
    """Persist per-question rows and the summary row for one configuration.

    MERGE on (config_name, question_id) — rerunning a configuration updates
    its rows instead of duplicating them (idempotent per configuration).
    """
    config_name = config["config_name"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = add_uniform_retrieval_metrics(rows, config["top_k"])
    pdf = pd.DataFrame(rows)
    pdf["eval_run_config"] = config_name
    pdf["top_k"] = config["top_k"]
    pdf["prompt_version"] = config["prompt_version"]
    pdf["temperature"] = config["temperature"]
    pdf["generation_model"] = rag_utils.LLM_ENDPOINT
    pdf["eval_dataset_version"] = eval_dataset_version
    pdf["mlflow_run_id"] = run_id
    pdf["eval_timestamp"] = now

    spark.createDataFrame(pdf).createOrReplaceTempView("new_generation_rows")
    spark.sql(f"""
        MERGE INTO {GENERATION_EVAL_TABLE} AS t
        USING new_generation_rows AS s
        ON t.eval_run_config = s.eval_run_config AND t.question_id = s.question_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    print(f"MERGE {GENERATION_EVAL_TABLE}: {len(pdf)} rows for {config_name}")

    summary = {**uniform_metrics(metrics, config["top_k"]),
               "config_name": config_name, "top_k": config["top_k"],
               "prompt_version": config["prompt_version"],
               "temperature": config["temperature"],
               "generation_model": rag_utils.LLM_ENDPOINT,
               "embedding_model": rag_utils.EMBEDDING_MODEL,
               "n_questions": len(rows),
               "eval_dataset_version": eval_dataset_version,
               "mlflow_run_id": run_id,
               "eval_timestamp": now}
    spark.createDataFrame(pd.DataFrame([summary])).createOrReplaceTempView("new_summary_row")
    spark.sql(f"""
        MERGE INTO {SUMMARY_TABLE} AS t
        USING new_summary_row AS s
        ON t.config_name = s.config_name
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    print(f"MERGE {SUMMARY_TABLE}: config {config_name}")


def run_and_persist(config_key, questions_pdf, max_workers=3):
    """Run one configuration end-to-end: pipeline -> MLflow run -> Delta tables.

    Returns (run_id, metrics).
    """
    config = CONFIGS[config_key]
    eval_dataset_version = questions_pdf["eval_dataset_version"].iloc[0]
    config_, rows, metrics, elapsed = run_full_evaluation(config_key, questions_pdf, max_workers)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"full_pipeline_{config['config_name']}") as run:
        run_id = run.info.run_id
        mlflow.set_tags({"stage": "full_pipeline", "config_name": config["config_name"]})
        mlflow.log_params({
            "top_k": config["top_k"],
            "embedding_model": rag_utils.EMBEDDING_MODEL,
            "retrieval_method": "vector_search_ann",
            "generation_model": rag_utils.LLM_ENDPOINT,
            "temperature": config["temperature"],
            "prompt_version": config["prompt_version"],
            "eval_dataset_version": eval_dataset_version,
            "n_questions": len(rows),
            "judge_backend": "mlflow.genai.judges (Databricks hosted judges)",
            "relevance_level": "document",
            "elapsed_seconds": round(elapsed),
        })
        mlflow.log_metrics({k: v for k, v in uniform_metrics(metrics, config["top_k"]).items()
                            if v is not None})
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "config.json"), "w") as f:
            json.dump({**config, "embedding_model": rag_utils.EMBEDDING_MODEL,
                       "generation_model": rag_utils.LLM_ENDPOINT}, f, indent=2)
        pd.DataFrame(rows).drop(columns=["answer"]).to_json(
            os.path.join(tmp, "per_question_results.json"), orient="records", indent=2)
        answers_doc = "\n\n".join(
            f"## {r['question_id']} — {r['question']}\n\n{r['answer']}" for r in rows)
        with open(os.path.join(tmp, "generated_answers.md"), "w") as f:
            f.write(f"# Generated answers — {config['config_name']}\n\n{answers_doc}")
        mlflow.log_artifacts(tmp)

    finalize_config_tables(config, rows, metrics, run_id, eval_dataset_version)
    return run_id, metrics


# ============================================================
# Helpers used by the notebooks
# ============================================================
def load_eval_questions():
    """Load the curated evaluation questions from Delta."""
    return spark.sql(f"SELECT * FROM {EVAL_QUESTIONS_TABLE} ORDER BY question_id").toPandas()


def compare_configs():
    """Return the configuration comparison from the summary table."""
    return spark.sql(f"""
        SELECT config_name, top_k, prompt_version, temperature,
               round(recall_at_1, 3) AS recall_at_1,
               round(recall_at_5, 3) AS recall_at_5,
               round(recall_at_10, 3) AS recall_at_10,
               round(precision_at_5, 3) AS precision_at_5,
               round(precision_at_10, 3) AS precision_at_10,
               round(mrr, 3) AS mrr,
               round(faithfulness, 3) AS faithfulness,
               round(context_relevance, 3) AS context_relevance,
               round(answer_relevance, 3) AS answer_relevance,
               round(citation_correctness, 3) AS citation_correctness,
               round(answer_correctness, 3) AS answer_correctness,
               mlflow_run_id
        FROM {SUMMARY_TABLE}
        ORDER BY top_k, prompt_version
    """).toPandas()


def list_mlflow_runs(stage="full_pipeline"):
    """List evaluation runs recorded in the MLflow experiment."""
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                             filter_string=f"tags.stage = '{stage}'",
                             order_by=["start_time ASC"])
    cols = [c for c in ["tags.config_name", "params.top_k", "params.prompt_version",
                        "params.temperature", "metrics.recall_at_5", "metrics.mrr",
                        "metrics.faithfulness", "metrics.citation_correctness",
                        "metrics.answer_correctness", "run_id"] if c in runs.columns]
    return runs[cols]
