# Sprint 6 — Model Serving | Civilization IV RAG Assistant

**Status: COMPLETE (2026-11-06).** The Civ4 RAG assistant is packaged as a versioned MLflow model, registered in Unity Catalog, deployed on Databricks Model Serving, and exposed through a reproducible REST API with a passing 10/10 end-to-end test suite.

## 1. Deployment architecture

```
                REST POST /api/2.0/serving-endpoints/civ4_rag_endpoint/invocations
                                      │
                                      ▼
            ┌───────────────────────────────────────────────┐
            │  Databricks Model Serving (serverless CPU)    │
            │  endpoint: civ4_rag_endpoint (scale-to-zero)  │
            │                                               │
            │  MLflow PyFunc: workspace.gcivilization.     │
            │  civ4_rag_model v1 (alias @champion)          │
            │  └── rag_model.py  (Civ4RagModel)             │
            │      └── rag_utils.py  (Sprint 4 pipeline)    │
            │          ├── Vector Search  (retrieval)       │◄── secret-backed env vars
            │          └── Foundation Model APIs (LLM +     │    DATABRICKS_HOST / DATABRICKS_TOKEN
            │              embeddings)                      │
            └───────────────────────────────────────────────┘
```

* **Packaging — MLflow PythonModel (PyFunc).** The RAG pipeline is not a conventional predictive model: it is an orchestrator over two external Databricks services. A PyFunc models this correctly while still getting MLflow signature, versioning, and Model Serving deployment for free. The Sprint 4 implementation (`rag_utils.py`) is shipped verbatim inside the artifact via `code_paths` — **zero business-logic duplication**. Alternatives considered and rejected: agent/LangChain MLflow flavors (heavier, no benefit for this pipeline) and Databricks Apps (a different product, not Model Serving).
* **Pinned configuration — Sprint 5 Config B** (`top_k=10`, prompt v1, temperature 0.1), the best validated configuration: faithfulness 1.0, citation 0.917, answer_correctness 0.667 (eval dataset version `v1-1fd1d9e2`).
* **Registration — Unity Catalog Model Registry** (`databricks-uc`), model `workspace.gcivilization.civ4_rag_model`, version 1, alias `champion`.
* **Credentials — secret-based environment variables.** Model Serving containers have no ambient workspace credentials. The served entity receives `DATABRICKS_HOST` (plain, public) and `DATABRICKS_TOKEN` (backed by `{{secrets/civ4_rag_serving/model_serving_token}}`), which both the `databricks-sdk` client and `mlflow.deployments` client use for outbound auth to Vector Search and the Foundation Model APIs.

## 2. Repository layout

```
06_model_serving/
├── rag_model.py                  # Deployable PyFunc (Civ4RagModel) + pinned config + signature
├── 01_model_packaging            # PyFunc validation: save→load→predict round trip, parity, OOD
├── 02_model_registration         # MLflow logging + UC registration with Sprint 5 traceability
├── 03_model_serving_deployment   # Secret scope + PAT, endpoint creation, readiness, verification
├── 04_inference_testing           # 10-scenario test suite incl. live REST inference
└── README.md                      # This file
```

Each notebook is executable independently when its prerequisites are met and contains validation assertions for every claim it makes.

## 3. Prerequisites

* Sprints 1–5 complete: `civ4_chunks` (604 chunks), embeddings table + Vector Search index (`civ4_vs_endpoint` / `civ4_chunks_embeddings_index`), validated RAG pipeline, evaluation evidence in `civ4_evaluation_summary`.
* Foundation Model API endpoints `databricks-qwen3-next-80b-a3b-instruct` (LLM) and `databricks-gte-large-en` (embeddings) READY.
* Unity Catalog permissions on `workspace.gcivilization` (model + table creation).
* Serverless compute for the notebooks (Python; R/Scala unsupported on serverless).

## 4. Deployment workflow

| Step | Notebook | What it does | Result |
| --- | --- | --- | --- |
| 1 | `01_model_packaging` | Validates the PyFunc: prerequisites, signature, structural checks, standalone save→load→predict round trip, retrieval parity, out-of-domain behavior | 6/6 PASS |
| 2 | `02_model_registration` | Logs the model (params, traceability tags, code_paths, signature, example, pip requirements) and registers it in UC | `civ4_rag_model` v1 READY |
| 3 | `03_model_serving_deployment` | Creates secret scope + 30-day PAT, creates/updates the endpoint with secret-backed env vars, waits for READY, verifies config | `civ4_rag_endpoint` READY |
| 4 | `04_inference_testing` | 10-scenario test suite: loading, valid questions, retrieval, generation, structure, references, edge cases, simulated failures, availability, REST end-to-end | 10/10 PASS |

**Traceability:** registered version 1 → MLflow run `f188f37b33954ca7819e16a2473f7432` (experiment `/Users/endoerner@gmail.com/civ4-rag-model-serving`) → tags linking to Sprint 5 eval run `92a0225ff5074439a2882d0c9bb60230`, config `B_top10_prompt_v1`, dataset `v1-1fd1d9e2` (mirrored as tags on the model version itself).

**Re-running:** the notebooks are idempotent. Notebook 02 appends a new model *version* (never a duplicate model); notebook 03 rotates the token and updates the endpoint config; the `champion` alias is re-pointed at the latest version.

## 5. Inference interface (REST API)

**Endpoint name:** `civ4_rag_endpoint`
**URL:** `https://<workspace-host>/api/2.0/serving-endpoints/civ4_rag_endpoint/invocations`
**HTTP method:** `POST`
**Content-Type:** `application/json`

### Authentication

Bearer token (personal access token or OAuth machine-to-machine token) with permission to query the endpoint:

```
Authorization: Bearer <DATABRICKS_API_TOKEN>
```

### Request schema

```json
{
  "dataframe_records": [
    { "question": "<your Civilization IV question>" }
  ]
}
```

`question` is a required string (MLflow input signature: `question: string`). Multiple records are answered in one request (one prediction per record).

### Response schema

```json
{
  "predictions": [
    {
      "answer": "<grounded answer citing [Source N] labels>",
      "references": [
        {
          "document_id": "<stable document identifier>",
          "title": "<document title>",
          "source_url": "<CivFanatics source URL>",
          "author": "<document author>",
          "category": "<document category>"
        }
      ]
    }
  ]
}
```

References are the deduplicated source documents retrieved by Vector Search that the answer is grounded in.

### Example request (curl)

```bash
curl -X POST \
  "https://<workspace-host>/api/2.0/serving-endpoints/civ4_rag_endpoint/invocations" \
  -H "Authorization: Bearer <DATABRICKS_API_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      { "question": "What should a brand-new Civilization IV player focus on first to improve at the game?" }
    ]
  }'
```

### Example response (actual, from the Sprint 6 test run — answer and URLs truncated)

```json
{
  "predictions": [
    {
      "answer": "A brand-new Civilization IV player should first focus on mastering the fundamentals of early-game expansion, resource management, and military preparedness. According to [Source 1], building a strong military early on helps deter AI aggression ...",
      "references": [
        { "document_id": "02_sisiutils_strategy_guide", "title": "Sisiutil's Strategy Guide for Beginners", "source_url": "https://www.civfanatics.com/...", "author": "Sisiutil", "category": "beginner" },
        { "document_id": "01_woundedknights_strategy_guide", "title": "WoundedKnight's Strategy Guide", "source_url": "https://www.civfanatics.com/...", "author": "WoundedKnight", "category": "beginner" },
        { "document_id": "07_early_rush", "title": "The Early Rush", "source_url": "https://www.civfanatics.com/...", "author": "Bismarck", "category": "military" },
        { "document_id": "11_cultural_victory", "title": "Cultural Victory: Notes and Tips", "source_url": "https://www.civfanatics.com/...", "author": "Agerz", "category": "victory_conditions" }
      ]
    }
  ]
}
```

### Example request (Databricks SDK)

```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
resp = w.serving_endpoints.query(
    name="civ4_rag_endpoint",
    dataframe_records=[{"question": "How can I tell if a city site is good for specialization?"}],
)
print(resp.predictions[0])
```

### Error behavior

* **Empty/blank question** → HTTP 200, structured response: `answer` explains that no valid question was provided, `references` is `[]`. No services are called.
* **Malformed payload** (missing `question` field) → HTTP 400, rejected by MLflow input-schema enforcement with a message naming the expected schema.
* **Service outage** (Vector Search or LLM unavailable) → HTTP 500 with an informative error (`RAG pipeline execution failed ... <exception class and message>`). No credentials are ever included in error messages.

## 6. Testing results (notebook 04)

All 10 required scenarios passed on 2026-11-06, using representative questions from the Sprint 5 evaluation dataset (categories: beginner_strategy, city_specialization):

1. Model loading — `models:/workspace.gcivilization.civ4_rag_model@champion`, signature persisted, no-cost predict probe OK
2. Valid question — 2 live invocations (SDK query + raw REST POST)
3. Retrieval — references present, expected documents hit on both questions
4. Generation — grounded answers citing `[Source N]` labels
5. Structure — `answer` + `references` contract held on every response
6. References — hard-asserted, all fields populated
7. Empty/invalid — structured 200 response; malformed payload rejected informatively
8. Service failures — simulated Vector Search and LLM outages produce informative errors without leaking secrets
9. Availability — endpoint READY
10. REST end-to-end — both invocation paths verified

**Parity with the validated pipeline:** the deployed model retrieved *exactly the same document sets* as the Sprint 4 `rag_utils.retrieve(top_k=10)` for both test questions (retrieval is deterministic; answer text at temperature 0.1 is compared structurally, not byte-for-byte). The deployment introduces no retrieval or behavioral drift.

## 7. Known limitations & operational notes

* **Databricks Free Edition serving caps:** custom model serving endpoints are CPU-only, with a cap on the number of active endpoints; no GPU serving and no provisioned throughput. This deployment (CPU `Small`) is within those limits.
* **Scale-to-zero cold start:** the first request after an idle period takes ~1–2 minutes (compute wake-up). Subsequent requests are fast.
* **Token rotation:** the serving PAT expires 30 days after creation (minted 2026-11-06). Re-run `03_model_serving_deployment` to rotate it — the notebook revokes old tokens with the same comment and overwrites the secret.
* **External service dependencies:** inference requires the Vector Search endpoint/index and the two Foundation Model API endpoints to be available; an outage surfaces as an informative HTTP 500.
* **Nondeterministic answer text:** temperature 0.1 generation is not byte-reproducible; retrieval and references are deterministic.
* **Dependency pins:** the artifact requires `mlflow>=3.0`, `databricks-sdk>=0.122.0`, `pandas>=2.0` (validated in this workspace with mlflow 3.12.0 / databricks-sdk 0.122.0 / pandas 2.3.3).
* **Corpus freshness:** the index serves the Sprint 2 corpus (604 chunks / 12 documents). Extending the corpus requires re-running the Sprint 3 sync pipeline and re-registering a new model version if chunk-level behavior must be re-validated.

## 8. Manual configuration / maintenance steps

* Rotate the serving token monthly (re-run notebook 03, credentials cell).
* To deploy a new model version: re-run notebook 02 (new version), then notebook 03 (re-points `champion` and rolls out the endpoint update).
* To tear down: delete the endpoint (`w.serving_endpoints.delete`), the registered model in Unity Catalog, and the secret scope `civ4_rag_serving` (the PAT is revoked automatically if the scope/secret is deleted, or via Tokens settings).
