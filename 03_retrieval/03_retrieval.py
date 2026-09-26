# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 03_retrieval / 03_retrieval
# MAGIC
# MAGIC **Purpose:** Implement the semantic retrieval layer — a `retrieve(query, top_k)` function that returns relevant Civilization IV chunks from the Vector Search index.
# MAGIC
# MAGIC **Inputs:**
# MAGIC - Vector Search index: `workspace.gcivilization.civ4_chunks_embeddings_index` (created by Notebook 02)
# MAGIC
# MAGIC **Outputs:**
# MAGIC - No persisted outputs — this notebook defines a function and demonstrates it with test queries.
# MAGIC
# MAGIC **Dependencies:**
# MAGIC - `databricks.sdk.WorkspaceClient` for Vector Search queries
# MAGIC - `mlflow.deployments` for query-time embedding generation via `databricks-gte-large-en`
# MAGIC
# MAGIC **Retrieval approach:**
# MAGIC - Self-managed embeddings: the query is embedded with `databricks-gte-large-en` (same model used in Notebook 01), and the resulting vector is passed to `query_index` via `query_vector`.
# MAGIC - `query_text` is not used because the index uses self-managed (pre-computed) embeddings without a model endpoint for query-time embedding.
# MAGIC
# MAGIC **Configuration:** All parameters defined in the Configuration cell below.
# MAGIC
# MAGIC **Next notebook:** `04_validation` validates the complete Sprint 3 pipeline (embeddings + Vector Search + retrieval).

# COMMAND ----------

# DBTITLE 1,1. Configuration
# ============================================================
# Configuration
# ============================================================

CATALOG = "workspace"
SCHEMA = "gcivilization"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.civ4_chunks_embeddings_index"
EMBEDDING_MODEL = "databricks-gte-large-en"

# Columns to retrieve from the Vector Search index
# (must match columns_to_sync from Notebook 02, minus embedding)
RETRIEVAL_COLUMNS = [
    "chunk_id", "chunk_text", "document_id", "title",
    "author", "category", "game_version", "expansion",
    "source_url", "publication_date",
]

print(f"Index name      : {INDEX_NAME}")
print(f"Embedding model : {EMBEDDING_MODEL}")
print(f"Retrieval cols  : {len(RETRIEVAL_COLUMNS)} columns")

# COMMAND ----------

# DBTITLE 1,2. Implement retrieve() function
# ============================================================
# Implement retrieve() function
# ============================================================
# retrieve(query, top_k=5) → list of dicts with chunk metadata + score
#
# Flow:
#   query → embed with databricks-gte-large-en → query_vector → Vector Search → top-K chunks

import mlflow.deployments
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
deploy_client = mlflow.deployments.get_deploy_client("databricks")


def retrieve(query, top_k=5):
    """
    Retrieve top-K relevant Civ4 chunks for a natural-language query.

    Args:
        query:   Natural-language query string.
        top_k:   Number of results to return (default 5).

    Returns:
        List of dicts, each containing:
          chunk_id, chunk_text, document_id, title, author, category,
          game_version, expansion, source_url, publication_date, score
    """
    if not query or not query.strip():
        return []

    # Step 1: Compute query embedding
    response = deploy_client.predict(
        endpoint=EMBEDDING_MODEL,
        inputs={"input": [query]}
    )
    query_vector = response.data[0]["embedding"]

    # Step 2: Query Vector Search index
    results = w.vector_search_indexes.query_index(
        index_name=INDEX_NAME,
        columns=RETRIEVAL_COLUMNS,
        query_vector=query_vector,
        num_results=top_k
    )

    # Step 3: Parse results into list of dicts
    chunks = []
    if results.result and results.result.data_array:
        for row in results.result.data_array:
            chunk = {}
            for i, col in enumerate(RETRIEVAL_COLUMNS):
                chunk[col] = row[i]
            # Similarity score is the last element in each row
            chunk["score"] = row[-1]
            chunks.append(chunk)

    return chunks


print("retrieve() function defined.")
print(f"  Index: {INDEX_NAME}")
print(f"  Default top_k: 5")
print(f"  Columns returned: {RETRIEVAL_COLUMNS} + score")

# COMMAND ----------

# DBTITLE 1,3. Test queries
# ============================================================
# Test queries and display results
# ============================================================
# Representative queries covering key Civ4 strategy topics.

test_queries = [
    "What is a Specialist Economy?",
    "How does city specialization work?",
    "How does combat work in Civilization IV?",
    "How can I execute an early rush?",
    "How can I achieve a Cultural Victory?",
    "How does diplomacy affect victory?",
    "What is the best strategy for playing as Rome?",
]

for query in test_queries:
    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    results = retrieve(query, top_k=5)

    if not results:
        print("  No results returned.")
        continue

    for rank, chunk in enumerate(results, 1):
        score = chunk.get("score", 0)
        if isinstance(score, (int, float)):
            score_str = f"{score:.4f}"
        else:
            score_str = str(score)

        print(f"\n  {rank}. [score: {score_str}] {chunk.get('chunk_id', 'N/A')}")
        print(f"     Title:    {chunk.get('title', 'N/A')}")
        print(f"     Category: {chunk.get('category', 'N/A')}")
        text = chunk.get('chunk_text', '') or ''
        print(f"     Text:     {text[:200].replace(chr(10), ' ')}...")
        print(f"     Source:   {chunk.get('source_url', 'N/A')}")

print("\n" + "=" * 70)
print(f"All {len(test_queries)} test queries executed.")