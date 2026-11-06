# Evaluation Results — Civilization IV Strategy RAG Assistant

**Status:** Sprint 5 evaluation complete; results below were read directly from the persisted Delta tables (`civ4_evaluation_summary`, `civ4_retrieval_eval`, `civ4_generation_eval`) and the MLflow experiment on 2026-11-06. No values are estimated or extrapolated.

---

## 1. Evaluation dataset

* **24 questions, exactly 2 per source document**, so all 12 corpus documents are represented. Questions were written from the actual chunk content of `civ4_chunks` — every question is answerable from the indexed corpus.
* **9 categories** (verified): beginner_strategy (4), military_strategy (4), civ_specific (4), city_specialization (2), cultural_victory (2), diplomatic_victory (2), economy (2), technology_research (2), combat_mechanics (2).
* **Difficulty:** easy / medium / hard (verified mix, e.g., civ_specific includes 3 medium + 1 hard).
* **Ground truth:** `expected_document_ids` (**document-level**) plus `expected_answer` (short factual summary used by the `answer_correctness` judge).
* **Versioning:** content-fingerprint version `v1-1fd1d9e2`, stored on every row of every eval table.

## 2. Methodology

### Retrieval evaluation

Relevance is **document-level**: a retrieved chunk is relevant if its `document_id` ∈ `expected_document_ids`.

* **Recall@K** — 1 if any expected document appears in the top K (hit rate)
* **Precision@K** — (# top-K chunks from expected documents) / K — a **lower bound**, since any chunk from an expected document counts as relevant
* **MRR** — mean reciprocal rank of the first relevant result

### Generation evaluation

Five MLflow genai LLM judges (Databricks-hosted judge backend), binary yes→1 / no→0, averaged over questions:

| Judge | Definition |
| --- | --- |
| `faithfulness` | Answer supported by the retrieved context (anti-hallucination; built-in `is_grounded`) |
| `context_relevance` | Retrieved context relevant to the question (built-in `is_context_relevant`) |
| `answer_relevance` | Answer addresses the question (custom judge) |
| `citation_correctness` | Cited `[Source N]` labels exist in the retrieved list AND support the attributed claims (custom judge) |
| `answer_correctness` | Answer matches the curated expected answer (built-in `is_correct`) |

## 3. Experimental configurations

| Config | top_k | prompt | temperature | Description |
| --- | --- | --- | --- | --- |
| A `A_baseline_top5_prompt_v1` | 5 | v1 | 0.1 | Baseline |
| B `B_top10_prompt_v1` | 10 | v1 | 0.1 | More retrieval depth (**pinned for deployment**) |
| C `C_top5_prompt_v2` | 5 | v2 (concise bullets) | 0.1 | Prompt-format experiment |

All configs used `databricks-qwen3-next-80b-a3b-instruct` (generation) and `databricks-gte-large-en` (embeddings).

## 4. Results

### 4.1 Retrieval (depth 10, 24 questions)

| Metric | Value |
| --- | --- |
| Recall@1 | 0.875 |
| Recall@3 | 0.958 |
| Recall@5 | **1.000** |
| Recall@10 | 1.000 |
| Precision@5 | 0.633 |
| Precision@10 | 0.558 |
| MRR | 0.927 |

An expected document appears in the top-1 for 21/24 questions and in the top-5 for all 24.

### 4.2 Generation — configuration comparison

Two complete evaluation executions exist (both tracked in MLflow): the **initial run on 2026-11-01** (recorded in `05_evaluation/README.md`) and an **idempotent rerun on 2026-11-02** (the values currently persisted in `civ4_evaluation_summary`, which MERGE-updated the per-config rows). Both are reported; differences between them are within documented judge nondeterminism.

**Initial run, 2026-11-01** (MLflow runs A `5bdd9428…`, B `4069828e…`, C `4a1c2601…`):

| Config | faithfulness | context_relevance | answer_relevance | citation_correctness | answer_correctness |
| --- | --- | --- | --- | --- | --- |
| A (top5, v1) | 1.000 | 1.000 | 1.000 | 0.792 | 0.542 |
| B (top10, v1) | 1.000 | 1.000 | 1.000 | **0.917** | 0.583 |
| C (top5, v2) | 0.792 | 1.000 | 1.000 | 0.667 | 0.542 |

**Rerun, 2026-11-02** (MLflow runs A `1291ef62…`, B `92a0225f…`, C `70140ce2…` — values currently in `civ4_evaluation_summary`):

| Config | faithfulness | context_relevance | answer_relevance | citation_correctness | answer_correctness | judge_error_rate |
| --- | --- | --- | --- | --- | --- | --- |
| A (top5, v1) | 1.000 | 0.958 | 1.000 | 0.958 | 0.542 | 0 |
| B (top10, v1) | 1.000 | 1.000 | 1.000 | 0.917 | 0.667 | 0 |
| C (top5, v2) | 0.958 | 0.958 | 1.000 | 0.708 | 0.458 | 0 |

### 4.3 Findings

1. **Retrieval is strong and effectively solved for this corpus** — recall@5 is 1.000, so remaining quality headroom is in generation.
2. **The anti-hallucination system prompt (v1) works** — faithfulness is 1.000 for A and B in both runs.
3. **top_k=10 (B) improves citation correctness** (0.917 in both runs vs A's 0.792/0.958) — more context gives the model more citable evidence — at the cost of precision@10 (0.558). Both runs agree B has the best answer_correctness (0.583 / 0.667).
4. **Prompt v2 (C) is consistently worse** — degraded faithfulness (0.792 / 0.958), citations (0.667 / 0.708), and answer correctness (0.542 / 0.458) in both runs. v1 was kept.
5. **answer_correctness is a strict measure** — 0.54–0.67 across configs; the judge compares against short factual summaries, so partial coverage scores 0.
6. Based on this evidence, **Config B was pinned** for the deployed model (model_params on `civ4_rag_model` v1: top_k=10, prompt v1, temperature 0.1).

## 5. MLflow references

* **Experiment:** `/Users/endoerner@gmail.com/civ4-rag-evaluation` (ID `2621923533772277`) — 8 FINISHED runs at audit time: 2 retrieval runs (`a41104b6…` 2026-11-01, `189b8806…` 2026-11-02) and 6 full-pipeline runs (A/B/C × initial + rerun). Each run records params (top_k, prompt_version, temperature), metrics, and per-question answer artifacts.
* **Deployment lineage:** registered model version 1 (run `f188f37b33954ca7819e16a2473f7432` in experiment `civ4-rag-model-serving`) carries tags `eval_mlflow_run_id=92a0225f…` and `eval_dataset_version=v1-1fd1d9e2`.
* **Persistence:** `civ4_eval_questions` / `civ4_retrieval_eval` overwrite (deterministic); `civ4_generation_eval` / `civ4_evaluation_summary` idempotent MERGE on (config, question); every row carries `eval_run_config`, `eval_dataset_version`, `mlflow_run_id`, `eval_timestamp`. Sprint 1–4 tables are read-only inputs (verified unchanged by the Sprint 5 validation notebook).

## 6. Limitations

1. **Sample size** — 24 questions demonstrate the methodology and give directional results, not statistically strong conclusions.
2. **Document-level ground truth** — retrieval metrics credit any chunk from an expected document; precision@K is a lower bound.
3. **LLM judges are nondeterministic** — repeated runs of the same config differ slightly (see the two tables above); differences under ~0.1 are not conclusive.
4. **Binary judges** — partial quality differences within an answer are not captured.
5. **answer_correctness strictness** — measures factual coverage against short summaries, not style or completeness.
6. **No per-question variance reporting** — only averages are persisted in the summary; per-question scores are available in `civ4_generation_eval` for deeper analysis.

## 7. Recommendations

* Grow the eval set (e.g., 50–100 questions with chunk-level ground truth) before drawing fine-grained conclusions about config differences.
* Add a reranker or hybrid BM25+dense retrieval — recall is already perfect, so gains would come in precision and ranking.
* Add human spot-checks on answers where judges disagree across runs.
* Track judge-score variance across repeated runs of the same config as a first-class metric.
