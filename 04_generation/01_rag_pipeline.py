# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 04_generation / 01_rag_pipeline
# MAGIC
# MAGIC **Purpose:** Implement the complete RAG generation layer — question → retrieval → context → prompt → LLM → grounded answer + source references.
# MAGIC
# MAGIC **Inputs:**
# MAGIC - Vector Search index: `workspace.gcivilization.civ4_chunks_embeddings_index` (Sprint 3)
# MAGIC - Embedding model: `databricks-gte-large-en` (Sprint 3)
# MAGIC
# MAGIC **Outputs:**
# MAGIC - No persisted outputs — this notebook defines the RAG pipeline and demonstrates it with example queries.
# MAGIC
# MAGIC **Dependencies:**
# MAGIC - `databricks.sdk.WorkspaceClient` for Vector Search queries
# MAGIC - `mlflow.deployments` for query-time embedding generation and LLM inference
# MAGIC - LLM endpoint: `databricks-qwen3-next-80b-a3b-instruct` (Databricks Foundation Model API)
# MAGIC
# MAGIC **Pipeline stages:**
# MAGIC 1. User question
# MAGIC 2. Semantic retrieval (reuses Sprint 3 `retrieve()` logic)
# MAGIC 3. Context construction from retrieved chunks
# MAGIC 4. RAG prompt construction (system instructions + context + question)
# MAGIC 5. LLM generation via Databricks Foundation Model API
# MAGIC 6. Structured output: answer + source references
# MAGIC
# MAGIC **Shared module:** All configuration and function definitions live in `rag_utils.py`, imported by both this notebook and `02_validation`. This ensures validation tests the actual implementation.
# MAGIC
# MAGIC **Next notebook:** `02_validation` validates the RAG pipeline behavior and output structure.

# COMMAND ----------

# DBTITLE 1,1. Import shared module
# ============================================================
# Import RAG pipeline functions from shared module
# ============================================================
# All configuration and function definitions live in rag_utils.py
# so that 02_validation can import the same implementation.

import sys, os

# Ensure the module directory is on sys.path
# (Databricks repos may not auto-add the notebook's directory)
for _module_dir in [
    "/Workspace/Repos/endoerner@gmail.com/civ4-rag/04_generation",
    "/Repos/endoerner@gmail.com/civ4-rag/04_generation",
]:
    if os.path.isdir(_module_dir) and _module_dir not in sys.path:
        sys.path.insert(0, _module_dir)
        break

from rag_utils import (
    retrieve, build_context, build_rag_prompt, generate_answer,
    build_sources, rag_query, SYSTEM_PROMPT,
    INDEX_NAME, EMBEDDING_MODEL, LLM_ENDPOINT, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, DEFAULT_TOP_K, RETRIEVAL_COLUMNS,
)

print(f"Index name       : {INDEX_NAME}")
print(f"Embedding model  : {EMBEDDING_MODEL}")
print(f"LLM endpoint     : {LLM_ENDPOINT}")
print(f"LLM max tokens   : {LLM_MAX_TOKENS}")
print(f"LLM temperature   : {LLM_TEMPERATURE}")
print(f"Default top_k    : {DEFAULT_TOP_K}")
print(f"\nFunctions imported from rag_utils:")
print(f"  retrieve, build_context, build_rag_prompt, generate_answer,")
print(f"  build_sources, rag_query")

# COMMAND ----------

# DBTITLE 1,7. End-to-end demonstration
# ============================================================
# End-to-end demonstration
# ============================================================
# Run the complete RAG pipeline on an example Civilization IV question.
# Replace `query` with any question to test the pipeline.

query = "How should I approach an early-game rush in Civilization IV?"

print("=" * 60)
print("  RAG PIPELINE DEMONSTRATION")
print("=" * 60)
print(f"\n  Question: {query}\n")
print("-" * 60)

# Run the pipeline
result = rag_query(query, top_k=DEFAULT_TOP_K)

# Display retrieved chunks
print(f"\n  Retrieved {len(result['retrieved_chunks'])} chunks:")
for i, chunk in enumerate(result["retrieved_chunks"], 1):
    score = chunk.get("score", 0)
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
    print(f"    {i}. [{score_str}] {chunk.get('chunk_id', 'N/A')}")
    print(f"       Title: {chunk.get('title', 'N/A')}")

# Display the answer
print("\n" + "-" * 60)
print("  GENERATED ANSWER:")
print("-" * 60)
print(result["answer"])

# Display sources
print("\n" + "-" * 60)
print("  SOURCES:")
print("-" * 60)
for i, source in enumerate(result["sources"], 1):
    print(f"  [{i}] {source['title']}")
    print(f"      Author:      {source['author']}")
    print(f"      Category:    {source['category']}")
    print(f"      Document ID:  {source['document_id']}")
    print(f"      Source URL:  {source['source_url']}")
    print(f"      Chunks used:  {len(source['chunks'])}")
    for ch in source["chunks"]:
        score = ch.get("score", 0)
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        print(f"        - {ch['chunk_id']} (score: {score_str})")

print("\n" + "=" * 60)
print("  Pipeline complete.")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,8. Out-of-domain demonstration
# ============================================================
# Out-of-domain demonstration
# ============================================================
# Test the pipeline with a question unlikely to have useful context.
# The LLM should acknowledge insufficient evidence rather than fabricate
# a Civilization IV-specific answer.

ood_query = "What is the capital city of France?"

print("=" * 60)
print("  OUT-OF-DOMAIN TEST")
print("=" * 60)
print(f"\n  Question: {ood_query}\n")
print("-" * 60)

ood_result = rag_query(ood_query, top_k=DEFAULT_TOP_K)

print(f"\n  Retrieved {len(ood_result['retrieved_chunks'])} chunks:")
for i, chunk in enumerate(ood_result["retrieved_chunks"], 1):
    score = chunk.get("score", 0)
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
    print(f"    {i}. [{score_str}] {chunk.get('chunk_id', 'N/A')}")
    print(f"       Title: {chunk.get('title', 'N/A')}")

print("\n" + "-" * 60)
print("  GENERATED ANSWER:")
print("-" * 60)
print(ood_result["answer"])

print("\n" + "=" * 60)
print("  Out-of-domain test complete.")
print("=" * 60)

# COMMAND ----------

