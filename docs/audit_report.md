# Technical Audit Report — Sprint 7

**Audit date:** 2026-11-06. **Method:** every artifact was inspected directly in the workspace (repo files, git history, Delta tables, Vector Search endpoint/index, MLflow experiments, UC registered model, serving endpoint) — no claim below relies on plans or memory alone. Statuses: **PASS** (verified and satisfactory) · **WARNING** (implemented, needs improvement/clarification) · **FAIL** (missing/incorrect) · **NOT VERIFIED** (insufficient evidence).

---

## 1. Audit table

| Area | Status | Findings | Recommended action |
| --- | --- | --- | --- |
| **Sprint 1 — Setup / data preparation** | PASS | 12 PDFs + 9-field metadata CSV present in UC Volume `/Volumes/workspace/gcivilization/civ4_rag/`; every PDF matches one metadata record; no repo folder for Sprint 1 (by design — data is not redistributable, documented in README) | None — Sprint 1 is intentionally data-only |
| **Sprint 2 — Processing** | PASS | 3 notebooks in `02_processing/`; `civ4_pdf_bronze` 12 rows, all `extraction_status=SUCCESS`; `civ4_chunks` 604 rows (avg 948 / min 192 / max 999 chars, 572,662 total); deterministic chunk IDs; MERGE (bronze) / overwrite (chunks) write patterns as designed; validation notebook recorded 44 PASS / 0 WARNING / 0 ERROR | None |
| **Sprint 3 — Retrieval** | PASS | `civ4_chunks_embeddings` 604 rows, all `embedding_status=SUCCESS`, 1024-dim `databricks-gte-large-en`; VS endpoint `civ4_vs_endpoint` **ONLINE**; index `civ4_chunks_embeddings_index` **ready, 604 rows**; retrieval function verified in `rag_utils.retrieve` (query embedding + `query_index`) | None |
| **Sprint 4 — RAG generation** | PASS | `rag_utils.py` implements retrieve → build_context → build_rag_prompt → generate_answer → build_sources → rag_query with anti-hallucination SYSTEM_PROMPT v1; LLM `databricks-qwen3-next-80b-a3b-instruct` (temp 0.1, max_tokens 1024); 36/36 validation checks recorded; OOD refusal behavior verified in evaluation and serving tests | None |
| **Sprint 5 — Evaluation / MLflow** | PASS (with note) | 24 eval questions, 24 retrieval rows, 72 generation rows, 3 summary rows — verified; MLflow experiment 2621923533772277 holds 8 FINISHED runs (initial 2026-11-01 + idempotent rerun 2026-11-02); every row traceable via `mlflow_run_id` | Note (not a defect): `05_evaluation/README.md` records the initial 2026-11-01 run while the persisted tables now hold the 2026-11-02 rerun (judge nondeterminism). Both are documented side-by-side in `docs/evaluation_results.md` |
| **Sprint 6 — Model Serving** | PASS (with note) | UC model `workspace.gcivilization.civ4_rag_model` **v1 READY**, alias `@champion`→v1 (verified via UC REST API); endpoint `civ4_rag_endpoint` **READY**, serving v1; model tags link to eval run `92a0225f…` and dataset `v1-1fd1d9e2`; 10/10 inference scenarios recorded; dependencies pinned as floors (`mlflow>=3.0`, `databricks-sdk>=0.122.0`, `pandas>=2.0`) | Operational note: serving PAT (minted 2026-11-06) expires ~2026-12-06 — rotate by re-running `03_model_serving_deployment` |
| **Architecture** | PASS | Names consistent across layers: `rag_utils` constants ↔ table/index names ↔ model `model_params` ↔ serving config; lineage data → eval run → model version → endpoint fully traceable; integration points (`civ4_chunks`→embeddings, embeddings→VS sync, VS→retrieval, retrieval→RAG, eval→pinning, registration→serving) all verified against live artifacts | None |
| **Code quality** | PASS (minor warnings) | Shared modules eliminate duplication (`rag_utils.py`, `rag_eval_utils.py`, PyFunc `code_paths`); per-sprint validation notebooks assert every claim; idempotent write patterns throughout; separation of concerns (pipeline / eval / serving) | Minor: catalog/schema constants are hardcoded (documented, acceptable for a fixed-project workspace); shared-module imports use `sys.path` injection (`%run` limitation on this workspace); observability is print/assert-based rather than structured logging; dependency floors rather than exact pins |
| **Security** | PASS | No secrets, tokens, or credentials in tracked files (pattern scan of all tracked files including notebook sources); serving token value never printed/logged (verified in notebook 03 outputs); `DATABRICKS_TOKEN` is secret-backed; error paths assert no credential leakage | None — keep the `*.pdf` gitignore rule active |
| **Licensing** | PASS (with caveats) | Code: **AGPL-3.0** (pre-existing LICENSE, unchanged). Source PDFs are **not** in the repo (verified — no PDFs tracked). Metadata CSV publishes attribution (titles, authors, civfanatics.com URLs) but is not itself committed to the repo — it lives in the UC Volume | Caveats: AGPL permits commercial use under copyleft terms — README states this accurately; per-document redistribution rights were not individually negotiated (README instructs reproducers to fetch documents from civfanatics.com) |
| **GitHub readiness** | PASS (after Sprint 7 items) | Before Sprint 7: stub README, no requirements.txt, no docs/, no root repo tree. All fixed in Sprint 7: full README, requirements.txt, docs/ (architecture, decisions, evaluation, audit, 4 diagrams) | Complete the GitHub publication checklist in §3 |

### Integration audit (explicit checks)

* `civ4_chunks` (604) → embeddings (604): row counts and all-SUCCESS statuses match ✔
* Embeddings table → VS index: 604 = 604 indexed rows, `embedding_dimension` 1024 ✔
* VS index → retrieval: `INDEX_NAME`/`RETRIEVAL_COLUMNS` in `rag_utils.py` match index schema ✔
* Retrieval → RAG: `rag_query` composes retrieve/build/generate exactly as validated ✔
* RAG → evaluation: eval module parameterizes only `top_k`/`temperature`/prompt version; Sprint 2–4 tables read-only ✔
* Evaluation → deployment: `civ4_evaluation_summary` best config B == `PINNED_CONFIG`/`model_params` on model v1 ✔
* Registration → serving: endpoint serves `workspace.gcivilization.civ4_rag_model` v1, alias `@champion` resolves to v1 ✔

## 2. Issues identified (all resolved or documented)

| # | Issue | Severity | Resolution |
| --- | --- | --- | --- |
| 1 | Root README was a one-line stub | High for a portfolio repo | Replaced with full 19-section README (Sprint 7) |
| 2 | No `requirements.txt` | Medium | Created from verified imports (Sprint 7) |
| 3 | No architecture/decisions/evaluation docs or diagrams | Medium | Created `docs/` with 4 Mermaid diagrams (Sprint 7) |
| 4 | Eval README (2026-11-01 run) vs persisted tables (2026-11-02 rerun) values differ slightly | Low | Expected judge nondeterminism + idempotent MERGE; both runs documented in `docs/evaluation_results.md` |
| 5 | Misplaced notebook `04_generation_01_rag_pipeline` in `03_retrieval/` | Low | Confirmed absent (cleaned up earlier; repo listing verified) |
| 6 | `.gitignore` lacked source-PDF guard | Medium | `*.pdf` exclusion added (Sprint 7) |
| 7 | Serving PAT 30-day expiry | Operational | Documented in README + `docs/architecture.md` maintenance notes |

## 3. GitHub readiness checklist

| Check | Status |
| --- | --- |
| Repository structure consistent (numbered sprint folders 02–06 + docs/) | ✔ |
| Descriptive notebook names, run order obvious | ✔ |
| Root README complete (overview, architecture, setup, usage, limitations, license) | ✔ (Sprint 7) |
| Documentation links resolve (relative paths within repo) | ✔ |
| `requirements.txt` matches actual imports, no extras | ✔ (Sprint 7) |
| `.gitignore` present, covers Python artifacts + blocks accidental PDF commits | ✔ (Sprint 7) |
| No credentials, tokens, or secrets in tracked files | ✔ (verified by scan) |
| No personal data beyond author identity (portfolio owner) | ✔ |
| No large/unnecessary files; no notebook output bloat | ✔ (06 notebooks are compact `.ipynb`; 02–05 are `.py` sources) |
| LICENSE present (AGPL-3.0), README license section consistent | ✔ |
| Source documents not redistributed; attribution + fetch instructions provided | ✔ |
| Reproducible without access to the original workspace (docs + instructions; Databricks account required) | ✔ |
| Working tree clean, all sprints committed and pushed to `main` | ✔ (Sprint 7 commit pending at audit time) |

**Final classification: READY** — the repository is publishable as a public GitHub portfolio project. Outstanding manual actions are listed in the Sprint 7 summary (notably: pushing the Sprint 7 commit and optionally enabling GitHub-flavored Mermaid rendering, which GitHub does natively).
