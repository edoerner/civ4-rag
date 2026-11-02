# Sprint 5 — Evaluation of the Civ4 RAG pipeline

This sprint adds a **reproducible evaluation layer** for the RAG pipeline built in Sprints 3–4: a curated question set with ground truth, retrieval metrics, LLM-judge generation metrics, an A/B/C configuration comparison, MLflow tracking, and a validation notebook.

The evaluation layer **reuses** the Sprint 4 implementation (`rag_utils.py`) and only parameterizes `top_k`, `temperature`, and the prompt version — it does not duplicate or modify the pipeline, and it never writes to the Sprint 2–4 tables.

## Assets

| Asset | Purpose |
| --- | --- |
| `rag_eval_utils.py` | Shared evaluation module: metric definitions, judge adapters, full-pipeline runner, MLflow + Delta persistence. (Named `rag_eval_utils` — not `eval_utils` — to avoid a name collision with an unrelated `eval_utils` artifact in the repo.) |
| `01_eval_dataset` | Curated 24-question evaluation set with validated ground truth → `civ4_eval_questions` |
| `02_retrieval_evaluation` | Retrieval-only evaluation at depth 10 → `civ4_retrieval_eval` + MLflow retrieval run |
| `03_generation_evaluation` | Full-pipeline evaluation, 5 LLM judges, baseline Config A → `civ4_generation_eval` / `civ4_evaluation_summary` |
| `04_mlflow_experiments` | A/B/C configuration comparison (Configs B and C) + MLflow run evidence |
| `05_validation` | End-to-end validation checklist (tables, MLflow, judges, traceability, non-regression) |

**Run order:** `01 → 02 → 03 → 04 → 05`. Notebook 05 is read-only and safe to rerun.

## Evaluation dataset

* **24 questions, 2 per source document**, so all 12 corpus documents are represented. Questions were written **from the actual chunk content** of `civ4_chunks` — every question is answerable from the indexed corpus.
* 9 categories: beginner_strategy, economy, city_specialization, combat_mechanics, technology_research, military_strategy, civ_specific, cultural_victory, diplomatic_victory. Difficulties: easy / medium / hard.
* Ground truth: `expected_document_ids` (**document-level**) and `expected_answer` (short factual summary written from the source documents, used by the `answer_correctness` judge).
* Versioned by content fingerprint (`v1-1fd1d9e2`), persisted with overwrite (deterministic).

## Metrics

### Retrieval (notebook 02)

Relevance is **document-level**: a retrieved chunk is relevant if its `document_id` ∈ `expected_document_ids`.

* **Recall@K** — hit rate: 1 if any expected document appears in the top K, else 0
* **Precision@K** — (# top-K chunks from expected documents) / K — a **lower bound** (document-level ground truth)
* **MRR** — mean reciprocal rank of the first relevant result

### Generation (notebooks 03–04)

Five MLflow genai judges (Databricks-hosted judge backend), yes→1 / no→0, averaged over questions:

| Judge | Definition |
| --- | --- |
| `faithfulness` | Answer supported by the retrieved context (anti-hallucination) — built-in `is_grounded` |
| `context_relevance` | Retrieved context relevant to the question — built-in `is_context_relevant` |
| `answer_relevance` | Answer addresses the question — custom judge |
| `citation_correctness` | Cited `[Source N]` labels exist in the retrieved list AND support the attributed claims — custom judge |
| `answer_correctness` | Answer matches the curated expected answer — built-in `is_correct` |

## Results (recorded run, 2026-11-01)

### Retrieval (depth 10)

Recall@1 = 0.875 · Recall@3 = 0.958 · Recall@5 = **1.000** · Recall@10 = 1.000 · Precision@5 = 0.633 · Precision@10 = 0.558 · MRR = 0.927

The retriever places an expected document in the top-1 for 21/24 questions and in the top-5 for all 24.

### Configuration comparison

| Config | top_k | prompt | faithfulness | context_relevance | answer_relevance | citation_correctness | answer_correctness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A (baseline) | 5 | v1 | 1.000 | 1.000 | 1.000 | 0.792 | 0.542 |
| B | 10 | v1 | 1.000 | 1.000 | 1.000 | **0.917** | 0.583 |
| C | 5 | v2 (bullets) | 0.792 | 1.000 | 1.000 | 0.667 | 0.542 |

**Readings** (directional, see limitations):

* Retrieval is strong: top-5 recall is perfect, so most quality headroom is in generation.
* `faithfulness` is 1.0 for prompt v1 (A/B) — the anti-hallucination system prompt works.
* **top_k=10 (B) improves citations** (0.917 vs 0.792) — more context gives the model more citable evidence — at the cost of precision@10 (0.558).
* **Prompt v2 (C) degrades every judge dimension** — concise-bullet formatting hurt faithfulness and citation quality; v1 remains the better prompt.
* `answer_correctness` hovers at 0.54–0.58 for all configs — the yes/no judge against short factual summaries is a strict measure (see limitations).

### MLflow evidence

Experiment `/Users/endoerner@gmail.com/civ4-rag-evaluation`: 1 retrieval run (`a41104b6…`) + 3 full_pipeline runs — A `5bdd9428…`, B `4069828e…`, C `4a1c2601…` — all tagged with config, params, metrics, and per-question answer artifacts.

## Limitations (documented)

1. **Sample size** — 24 questions is enough to demonstrate the methodology and give directional results, not statistically strong conclusions.
2. **Document-level ground truth** — retrieval metrics credit any chunk from an expected document; `precision@K` is therefore a lower bound.
3. **LLM judges are nondeterministic** — repeated runs of the same configuration can produce slightly different scores; config differences under ~0.1 are not conclusive.
4. **Yes/no judges** — partial quality differences within an answer are not captured.
5. **`answer_correctness` strictness** — ground truth is a short factual summary, so the judge measures factual coverage, not style.

## Persistence & reproducibility

* `civ4_eval_questions` / `civ4_retrieval_eval` — overwrite (deterministic: same questions, same embeddings → same metrics)
* `civ4_generation_eval` / `civ4_evaluation_summary` — **idempotent MERGE** on (config, question): rerunning a config updates its rows instead of duplicating them
* Every row carries `eval_run_config`, `eval_dataset_version`, `mlflow_run_id`, and `eval_timestamp` for full traceability
* Sprint 1–4 tables (`civ4_pdf_bronze`, `civ4_chunks`, `civ4_chunks_embeddings`) are read-only inputs — verified unchanged by notebook 05
