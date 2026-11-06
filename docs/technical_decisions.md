# Technical Decisions — Civilization IV Strategy RAG Assistant

**Status:** COMPLETE (Sprint 7, 2026-11-06). Each decision below is grounded in the implementation, project history, or workspace artifacts. Where reasoning cannot be established from evidence, it is explicitly labeled **assumption — requires confirmation**.

Legend: 🟢 = evidence in code/docs · 🔵 = evidence in evaluation data · ⚪ = assumption.

---

## TD-001 — Why RAG

* **Topic:** Overall approach for a domain Q&A assistant.
* **Context:** The goal is answering strategy questions about a small, fixed corpus (12 guides, ~573k characters) with verifiable, attributable answers — a portfolio project demonstrating an end-to-end GenAI system.
* **Decision:** Retrieval-Augmented Generation with an explicit retrieval stage, grounded prompts, `[Source N]` citations, and an anti-hallucination system prompt. 🟢
* **Alternatives considered:** Fine-tuning an LLM on the corpus; a pure keyword search engine; an agent with web search. None are documented as formally evaluated — the RAG choice is evident from the design, but the explicit comparison is an **assumption — requires confirmation** if a rationale is cited elsewhere. ⚪
* **Reasoning (from the implementation):** the corpus is small and stable, answers must cite sources (citations are hard-coded into the prompt contract and the API response schema), and RAG lets the same LLM answer only from indexed evidence. Faithfulness 1.0 in evaluation confirms grounding works. 🔵
* **Advantages:** attributable answers; no training cost; corpus updates need only re-chunking + re-sync; demonstrates retrieval, embeddings, and serving skills.
* **Limitations/trade-offs:** quality is bounded by retrieval and the LLM; out-of-domain questions are refused rather than answered.
* **Impact:** every downstream stage (chunking, embeddings, VS, prompts, judges, serving) exists to serve this approach.

## TD-002 — PDF as the initial source format

* **Topic:** Ingestion format.
* **Context:** The best Civilization IV strategy content (civfanatics.com articles) was collected as PDFs.
* **Decision:** 12 PDFs stored in a UC Volume + a sidecar metadata CSV. 🟢
* **Alternatives:** web scraping civfanatics.com HTML (not chosen — **assumption — requires confirmation**: likely avoided because of site terms/fragility; the PDFs were already collected). ⚪
* **Reasoning:** PDFs are stable, self-contained, and match the real-world pattern of document-ingestion pipelines; pypdf extracts their text layers. 🟢
* **Advantages:** reproducible; realistic document-engineering exercise; no runtime dependency on the source site.
* **Limitations:** text extraction depends on PDFs having text layers (all 12 do — extraction_status SUCCESS 🟢); layout artifacts can leak into text (mitigated by cleaning); PDFs are not redistributable in the repo (TD-017).

## TD-003 — Extraction and cleaning approach

* **Topic:** How text is extracted from PDFs.
* **Decision:** `pypdf` per-document extraction inside notebook `02_processing/01_extract_and_clean`, with per-document diagnostics (`page_count`, `character_count`, `extraction_status`, `extraction_error`) and bronze MERGE on `document_id` for idempotency. 🟢
* **Alternatives:** pdfplumber/PyMuPDF, Databricks AI document intelligence; DAB/production OCR — not needed since all PDFs have text layers. Not formally documented as evaluated. ⚪
* **Reasoning:** pypdf is a lightweight pure-Python library, installable via pip on Databricks Free Edition serverless (not pre-installed there — installed per session). 🟢
* **Advantages:** simple, observable (per-doc status), idempotent re-runs.
* **Limitations:** pypdf is slower and less layout-aware than PyMuPDF; diagnostics columns keep failures visible rather than silently dropped. 🟢
* **Impact:** bronze table schema + the Sprint 2 validation suite.

## TD-004 — Chunking strategy

* **Topic:** How documents are split for retrieval.
* **Decision:** Character-based recursive splitting with `chunk_size=1000`, `chunk_overlap=150`, deterministic chunk IDs `{document_id}_chunk_{index:04d}`, overwrite writes (deterministic IDs make MERGE unnecessary). 🟢 Actual corpus: 604 chunks, avg 948, min 192, max 999 chars. 🔵
* **Alternatives:** sentence/token-based splitting, semantic chunking, smaller/larger sizes — larger sizes would exceed the embedding model input window more often; smaller sizes fragment strategy advice across chunks. (Reasoning as recorded — **assumption — requires confirmation** for any exact rationale not in code.) ⚪
* **Reasoning:** 1000 characters keeps related strategy advice in one chunk while staying well inside the embedding model's input limit; 150 overlap preserves continuity at boundaries; deterministic IDs enable idempotent full re-runs. 🟢
* **Advantages:** stable IDs, simple, predictable; validated by 44/44 checks. 🟢
* **Limitations:** character-based splitting can cut mid-sentence; no semantic boundaries; no per-category tuning.

## TD-005 — Metadata design

* **Topic:** What metadata travels with each chunk.
* **Decision:** 9-field metadata CSV (document_id, pdf_filename, title, author, category, game_version, expansion, source_url, publication_date) validated against files, then denormalized onto every chunk and every embedding row; also synced to the Vector Search index so retrieval returns full attribution. 🟢
* **Reasoning:** citations and the API `references` object need title/author/url/category at retrieval time; denormalization avoids joins at inference. 🟢
* **Advantages:** one scan gives the LLM and the client full provenance; document-level evaluation ground truth derives from the same IDs.
* **Limitations:** denormalized metadata inflates storage slightly; schema changes require re-chunking.

## TD-006 — Embedding model choice

* **Topic:** Embedding model.
* **Decision:** `databricks-gte-large-en` (Databricks Foundation Model API), 1024 dimensions, batch size 10 (serverless embedding API limit — larger batches time out at ~0.8 s per batch of 10). 🟢
* **Alternatives:** other Foundation Model APIs (e.g., bge variants) — **assumption — requires confirmation** whether others were benchmarked. ⚪
* **Reasoning:** available in Free Edition via Foundation Model APIs, no external provider, strong English retrieval quality; used for both documents and queries (required: same model for both). 🟢
* **Evidence:** Recall@5 = 1.000 and MRR = 0.927 on the eval set. 🔵
* **Limitations:** English-only; 1024-dim vectors are larger than some alternatives; batch cap lengthens indexing.

## TD-007 — Vector Search architecture

* **Topic:** Vector store + sync.
* **Decision:** Databricks Vector Search: STANDARD endpoint `civ4_vs_endpoint`, DELTA_SYNC index `civ4_chunks_embeddings_index` with **self-managed embeddings** (pipeline computes embeddings into the table; the index syncs them) and TRIGGERED sync. 🟢
* **Alternatives:** managed embeddings inside the index; continuous sync. Self-managed was chosen to control the embedding batch size (serverless cap) and keep embeddings inspectable in Delta. (Reasoning partially recorded — marked **assumption — requires confirmation** where not explicit.) ⚪
* **Advantages:** embeddings are first-class Delta data (re-audit, re-index without re-embedding); TRIGGERED sync gives predictable, auditable index builds.
* **Limitations:** two artifacts to keep in sync; provisioning takes ~7–8 minutes after creation. 🟢
* **Impact:** defines the retrieval API (`WorkspaceClient.vector_search_indexes.query_index` with `query_vector`).

## TD-008 — Retrieval strategy

* **Topic:** Retrieval stage design.
* **Decision:** Single-stage dense retrieval: query → embedding → `query_index(num_results=top_k)`. Default top_k=5 in `rag_utils`; **deployed config pins top_k=10** (Config B). 🟢
* **Alternatives evaluated in Sprint 5:** top_k=5 vs top_k=10 (configs A vs B) — top_k=10 improved citation correctness (0.917 vs 0.792 initial run) at the cost of precision@10 (0.558). 🔵 Hybrid BM25+dense, reranking: not implemented (future improvement).
* **Reasoning:** with recall@5 already 1.000, extra depth adds citable evidence rather than recall. 🔵
* **Limitations:** no reranker, no hybrid search; document-level ground truth makes precision@K a lower bound. 🔵

## TD-009 — LLM selection

* **Topic:** Generation model.
* **Decision:** `databricks-qwen3-next-80b-a3b-instruct` via Foundation Model APIs (`mlflow.deployments` client, `llm/v1/chat`), temperature 0.1, max_tokens 1024. 🟢 The endpoint's description explicitly cites suitability for RAG. 🟢
* **Alternatives:** other Foundation Model endpoints; external providers (paid, out of scope for Free Edition). Not benchmarked head-to-head. ⚪
* **Reasoning:** free-tier available, strong instruction following, low temperature for factual, grounded answers. 🟢
* **Limitations:** judge-measured answer_correctness 0.54–0.67 shows factual coverage headroom; nondeterministic text at temp 0.1. 🔵

## TD-010 — Prompt design

* **Topic:** Prompt architecture.
* **Decision:** 7-instruction system prompt (grounding, no invention, insufficient-context refusal with fixed phrasing, Civ4 terminology, 3–6 paragraphs, `[Source N]` citations, no fabricated citations) + structured context blocks carrying title/IDs/URL per chunk. 🟢 Prompt v2 (concise bullets) was tested in Sprint 5 as config C and **rejected** — it degraded every judge dimension (faithfulness 0.792 initial run). 🔵
* **Advantages:** faithfulness 1.0 for v1 configs; refusal behavior verified for out-of-domain queries. 🔵
* **Limitations:** v1 prose length can exceed what users want in some contexts; prompt is domain-specific English.

## TD-011 — Evaluation methodology

* **Topic:** How the system is evaluated.
* **Decision:** Curated 24-question set (2 per document, 9 categories, 3 difficulties, versioned by content fingerprint `v1-1fd1d9e2`); document-level ground truth; retrieval metrics recall@K/precision@K/MRR; generation scored by 5 binary MLflow genai LLM judges (faithfulness, context_relevance, answer_relevance, citation_correctness, answer_correctness); three configs compared; all runs in MLflow; results MERGEd to Delta. 🟢 Full results: `evaluation_results.md`.
* **Reasoning:** evidence over vibes — the project requires measured performance, and MLflow judges are available in-workspace. 🟢
* **Limitations (documented in the repo):** 24 questions → directional, not statistically strong; document-level ground truth; judge nondeterminism (differences < ~0.1 not conclusive); binary judges; strict answer_correctness. 🔵

## TD-012 — MLflow integration

* **Topic:** Experiment tracking + evaluation.
* **Decision:** Dedicated experiment `civ4-rag-evaluation` (ID 2621923533772277): params, metrics, per-question answer artifacts per run; a second experiment `civ4-rag-model-serving` (ID 481989999099671) for the deployment artifact. Every eval row persists `mlflow_run_id`; the registered model version carries tags linking to its eval run. 🟢
* **Advantages:** full lineage question → run → metrics → model version → endpoint; reruns are idempotent MERGEs. 🟢
* **Limitations:** runs accumulate on rerun by design (8 runs at audit time); no CI/CD trigger.

## TD-013 — Model packaging and registration

* **Topic:** How the pipeline becomes a deployable artifact.
* **Decision:** MLflow PyFunc `Civ4RagModel` with declared I/O signature (`question:string` → `answer:string` + references array), `code_paths` shipping `rag_utils.py` verbatim (zero business-logic duplication), pinned Config B, registered in **Unity Catalog** as `workspace.gcivilization.civ4_rag_model` v1 with alias `@champion`. 🟢
* **Alternatives considered and rejected (documented in 06_model_serving/README.md):** MLflow agent/LangChain flavors (heavier, no benefit for this orchestrator-style pipeline); Databricks Apps (different product). 🟢 The legacy workspace model registry is disabled in this workspace — UC was the only supported registry. 🟢
* **Reasoning:** the RAG "model" is an orchestrator over Vector Search + Foundation Model APIs; PyFunc still yields signature, versioning, and Model Serving. 🟢
* **Limitations:** artifact pins `>=` dependency floors (mlflow>=3.0, databricks-sdk>=0.122.0, pandas>=2.0) rather than exact versions. 🟢

## TD-014 — Model Serving deployment

* **Topic:** Serving.
* **Decision:** Serverless CPU Small endpoint `civ4_rag_endpoint`, scale-to-zero, serving v1 via `@champion`; auth for outbound calls via env vars — `DATABRICKS_HOST` plain, `DATABRICKS_TOKEN` secret-backed (`{{secrets/civ4_rag_serving/model_serving_token}}`, a 30-day PAT minted 2026-11-06, never printed/logged). 🟢 Validated by 10/10 inference scenarios including empty input, malformed payloads, and simulated outages. 🟢
* **Reasoning:** Free Edition supports custom CPU serving with a capped endpoint count; serverless CPU Small + scale-to-zero fits the free tier. 🟢
* **Limitations:** cold start ~1–2 min after idle; monthly token rotation required (PAT expires ~2026-12-06); inference depends on VS + Foundation Model endpoints being up. 🟢

## TD-015 — Databricks Free Edition constraints

* **Topic:** Environment-driven decisions.
* **Decision/evidence (all verified in-workspace):** serverless compute only (R/Scala unsupported); `pypdf` not pre-installed (pip per session); embedding batch cap 10; VS endpoint STANDARD; CPU-only serving, capped endpoint count, no GPU/provisioned throughput; legacy workspace model registry disabled → UC-only; UC model APIs differ from legacy SDK (e.g., `get_latest_versions` unsupported → `search_model_versions`). 🟢
* **Impact:** every sprint stayed inside free-tier services; no external paid provider is used anywhere.

## TD-016 — GitHub repository structure

* **Topic:** Repo organization.
* **Decision:** One numbered folder per sprint (`02_processing` … `06_model_serving`) matching the notebook run order, shared modules co-located with their sprint (`rag_utils.py`, `rag_eval_utils.py`, `rag_model.py`), per-sprint READMEs in 05/06, and a Sprint-7 `docs/` folder for architecture/decisions/evaluation/audit. Sprint 1 has **no repo folder** — its artifacts are data (PDFs + CSV in the UC Volume, not redistributable, documented in the root README instead). 🟢
* **Reasoning:** reproducible run order; a reader follows folders 02→06 exactly as the system was built. 🟢
* **Limitations:** notebooks import shared modules via `sys.path` manipulation (a workaround — `%run` does not work for repo-notebook references on this workspace). 🟢

## TD-017 — Licensing and source-document attribution

* **Topic:** License + copyright of source documents.
* **Decision:** Repository code is licensed **AGPL-3.0** (pre-existing LICENSE, unchanged in Sprint 7). The 12 source PDFs are **not committed** to the repository (verified: no PDFs tracked by git); the metadata CSV (titles, authors, civfanatics.com URLs, dates) provides attribution, and the README documents that source documents must be obtained from civfanatics.com by anyone reproducing the project. 🟢
* **Reasoning:** the CivFanatics articles are copyrighted by their authors; redistribution rights are unclear, so only attribution metadata is published. AGPL-3.0 keeps the code strongly copyleft (source disclosure extends to network use), which aligns with the author's goal of discouraging proprietary reuse. 🟢
* **Implications to state accurately:** AGPL does **not** forbid commercial use — it conditions any use on copyleft compliance (modified source must be offered to network users). Changing the license would require explicit approval and was not done.
* **Limitations:** per-PDF licensing was not individually negotiated; if any author objects to even metadata publication, that record can be trimmed.
