# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 03_retrieval / 04_validation
# MAGIC
# MAGIC **Purpose:** Validate the complete Sprint 3 retrieval pipeline — embeddings, Vector Search, and retrieval.
# MAGIC
# MAGIC **Inputs:**
# MAGIC - `workspace.gcivilization.civ4_chunks` (Sprint 2 output)
# MAGIC - `workspace.gcivilization.civ4_chunks_embeddings` (Notebook 01 output)
# MAGIC - Vector Search endpoint + index (Notebook 02 output)
# MAGIC
# MAGIC **Outputs:**
# MAGIC - Validation report (printed) — no persisted outputs.
# MAGIC
# MAGIC **Validation dimensions:**
# MAGIC 1. **Embeddings** — coverage, uniqueness, non-null, dimension consistency, model verification
# MAGIC 2. **Vector Search** — endpoint exists, index exists and ready, source table correct, embedding column correct
# MAGIC 3. **Retrieval** — test queries return relevant results with scores, out-of-domain query returns results with lower scores
# MAGIC
# MAGIC **Idempotency:** Safe to rerun — all checks are read-only.
# MAGIC
# MAGIC **Next step:** Sprint 4 — RAG generation layer (answer generation, RAG prompt construction).

# COMMAND ----------

# DBTITLE 1,1. Configuration
# ============================================================
# Configuration
# ============================================================

CATALOG = "workspace"
SCHEMA = "gcivilization"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.civ4_chunks"
EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.civ4_chunks_embeddings"
ENDPOINT_NAME = "civ4_vs_endpoint"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.civ4_chunks_embeddings_index"
EMBEDDING_MODEL = "databricks-gte-large-en"

RETRIEVAL_COLUMNS = [
    "chunk_id", "chunk_text", "document_id", "title",
    "author", "category", "game_version", "expansion",
    "source_url", "publication_date",
]

# Track validation results
validation_results = []

def record_check(name, passed, detail=""):
    """Record a validation check result."""
    status = "PASS" if passed else "FAIL"
    validation_results.append((name, status, detail))
    marker = "✅" if passed else "❌"
    print(f"  {marker} {name}: {status}" + (f" — {detail}" if detail else ""))

print("Configuration loaded. Validation checks will be recorded below.")

# COMMAND ----------

# DBTITLE 1,2. Embedding validation
# ============================================================
# 1. Embedding validation
# ============================================================
# Verifies coverage, uniqueness, non-null, dimensions, model.

print("=" * 55)
print("  EMBEDDING VALIDATION")
print("=" * 55)

df_chunks = spark.table(CHUNKS_TABLE)
df_embeddings = spark.table(EMBEDDINGS_TABLE)

chunks_count = df_chunks.count()
embeddings_count = df_embeddings.count()
print(f"  Chunks table rows    : {chunks_count}")
print(f"  Embeddings table rows: {embeddings_count}")

# Check 1: Coverage — every chunk_id in civ4_chunks should be in embeddings
missing = df_chunks.select("chunk_id").subtract(df_embeddings.select("chunk_id"))
missing_count = missing.count()
record_check("Chunk coverage", missing_count == 0,
            f"{missing_count} chunk(s) missing embeddings")

# Check 2: No extra embeddings (embeddings for chunks not in civ4_chunks)
extra = df_embeddings.select("chunk_id").subtract(df_chunks.select("chunk_id"))
extra_count = extra.count()
record_check("No extra embeddings", extra_count == 0,
            f"{extra_count} extra embedding(s)")

# Check 3: chunk_id uniqueness in embeddings
from pyspark.sql.functions import col
dup_count = df_embeddings.groupBy("chunk_id").count().filter("count > 1").count()
record_check("chunk_id unique", dup_count == 0,
            f"{dup_count} duplicate(s)")

# Check 4: chunk_text populated
empty_text = df_embeddings.filter("chunk_text IS NULL OR chunk_text = ''").count()
record_check("chunk_text populated", empty_text == 0,
            f"{empty_text} empty text(s)")

# Check 5: Embeddings not null
null_emb = df_embeddings.filter("embedding IS NULL").count()
record_check("Embeddings non-null", null_emb == 0,
            f"{null_emb} null embedding(s)")

# Check 6: Embedding dimensions consistent
dim_df = df_embeddings.filter("embedding IS NOT NULL") \
    .select("embedding_dimension").distinct()
dim_values = [row.embedding_dimension for row in dim_df.collect()]
record_check("Dimension consistency", len(dim_values) == 1,
            f"dimensions found: {dim_values}")

# Check 7: Embedding model is databricks-gte-large-en
model_df = df_embeddings.groupBy("embedding_model").count()
models = [(row.embedding_model, row["count"]) for row in model_df.collect()]
model_ok = len(models) == 1 and models[0][0] == EMBEDDING_MODEL
record_check("Embedding model correct", model_ok,
            f"models: {models}")

# Check 8: All embeddings have SUCCESS status
failed_count = df_embeddings.filter("embedding_status != 'SUCCESS'").count()
record_check("All embeddings SUCCESS", failed_count == 0,
            f"{failed_count} non-SUCCESS")

print(f"\n  Embedding checks: {sum(1 for _, s, _ in validation_results if s == 'PASS')}/"
      f"{len(validation_results)} passed")

# COMMAND ----------

# DBTITLE 1,3. Vector Search validation
# ============================================================
# 2. Vector Search validation
# ============================================================
# Verifies endpoint, index, readiness, configuration.

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointStatusState

print("\n" + "=" * 55)
print("  VECTOR SEARCH VALIDATION")
print("=" * 55)

w = WorkspaceClient()

# Check 1: Endpoint exists and is online
try:
    endpoint = w.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
    ep_online = endpoint.endpoint_status.state == EndpointStatusState.ONLINE
    record_check("Endpoint exists", True, f"name={endpoint.name}")
    record_check("Endpoint online", ep_online,
                f"state={endpoint.endpoint_status.state}")
except Exception as e:
    record_check("Endpoint exists", False, str(e))
    endpoint = None

# Check 2: Index exists and is ready
try:
    index = w.vector_search_indexes.get_index(INDEX_NAME)
    record_check("Index exists", True, f"name={index.name}")
    
    index_ready = index.status.ready
    record_check("Index ready", index_ready,
                f"ready={index_ready}, message={index.status.message}")
    
    indexed_rows = index.status.indexed_row_count
    record_check("Index has data", indexed_rows is not None and indexed_rows > 0,
                f"indexed_rows={indexed_rows}")
except Exception as e:
    record_check("Index exists", False, str(e))
    index = None

# Check 3: Index configured against expected source table
if index and index.delta_sync_index_spec:
    spec = index.delta_sync_index_spec
    source_ok = spec.source_table == EMBEDDINGS_TABLE
    record_check("Source table correct", source_ok,
                f"source={spec.source_table}")
    
    # Check 4: Embedding column correct
    if spec.embedding_vector_columns:
        for evc in spec.embedding_vector_columns:
            col_ok = evc.name == "embedding"
            dim_ok = evc.embedding_dimension == 1024
            record_check("Embedding column correct", col_ok,
                        f"column={evc.name}")
            record_check("Embedding dimension correct", dim_ok,
                        f"dimension={evc.embedding_dimension}")
    else:
        record_check("Embedding column correct", False, "no embedding vector columns")
else:
    record_check("Source table correct", False, "no delta sync spec")

# COMMAND ----------

# DBTITLE 1,4. Retrieval validation
# ============================================================
# 3. Retrieval validation
# ============================================================
# Runs test queries and displays results for manual inspection.
# Includes positive cases and one out-of-domain case.

import mlflow.deployments

deploy_client = mlflow.deployments.get_deploy_client("databricks")


def retrieve(query, top_k=5):
    """Retrieve top-K chunks for a query (same logic as Notebook 03)."""
    if not query or not query.strip():
        return []
    response = deploy_client.predict(
        endpoint=EMBEDDING_MODEL,
        inputs={"input": [query]}
    )
    query_vector = response.data[0]["embedding"]
    results = w.vector_search_indexes.query_index(
        index_name=INDEX_NAME,
        columns=RETRIEVAL_COLUMNS,
        query_vector=query_vector,
        num_results=top_k
    )
    chunks = []
    if results.result and results.result.data_array:
        for row in results.result.data_array:
            chunk = {}
            for i, col in enumerate(RETRIEVAL_COLUMNS):
                chunk[col] = row[i]
            chunk["score"] = row[-1]
            chunks.append(chunk)
    return chunks


print("\n" + "=" * 55)
print("  RETRIEVAL VALIDATION")
print("=" * 55)

# --- Positive retrieval cases ---
positive_queries = [
    "What is a Specialist Economy?",
    "How can I achieve a Cultural Victory?",
]

for query in positive_queries:
    print(f"\n  Query: {query}")
    results = retrieve(query, top_k=3)
    if results:
        for rank, chunk in enumerate(results, 1):
            score = chunk.get("score", 0)
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            print(f"    {rank}. [score: {score_str}] {chunk.get('chunk_id', 'N/A')}")
            print(f"       Title: {chunk.get('title', 'N/A')} | Category: {chunk.get('category', 'N/A')}")
            text = (chunk.get('chunk_text', '') or '')[:120].replace(chr(10), ' ')
            print(f"       Text: {text}...")
            print(f"       Source: {chunk.get('source_url', 'N/A')}")
        record_check(f"Query returns results: '{query[:40]}...'", True,
                    f"{len(results)} results, top score: {results[0].get('score', 'N/A')}")
    else:
        record_check(f"Query returns results: '{query[:40]}...'", False, "no results")

# --- Out-of-domain case ---
# This query is unlikely to have useful results in the Civ4 corpus.
# The objective is to verify that results ARE returned with scores,
# so Sprint 4 can implement a relevance threshold.
out_of_domain_query = "How do I build spaceships in Civilization IV?"
print(f"\n  Out-of-domain query: {out_of_domain_query}")
ood_results = retrieve(out_of_domain_query, top_k=3)
if ood_results:
    for rank, chunk in enumerate(ood_results, 1):
        score = chunk.get("score", 0)
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        print(f"    {rank}. [score: {score_str}] {chunk.get('chunk_id', 'N/A')}")
        print(f"       Title: {chunk.get('title', 'N/A')} | Category: {chunk.get('category', 'N/A')}")
    record_check("Out-of-domain returns results with scores", True,
                f"{len(ood_results)} results, top score: {ood_results[0].get('score', 'N/A')}")
else:
    record_check("Out-of-domain returns results with scores", False, "no results")

# COMMAND ----------

# DBTITLE 1,5. Validation summary
# ============================================================
# Validation summary
# ============================================================

pass_count = sum(1 for _, s, _ in validation_results if s == "PASS")
fail_count = sum(1 for _, s, _ in validation_results if s == "FAIL")
total = len(validation_results)

print("\n" + "=" * 55)
print("  SPRINT 3 VALIDATION SUMMARY")
print("=" * 55)
print(f"  Total checks : {total}")
print(f"  Passed      : {pass_count}")
print(f"  Failed      : {fail_count}")
print("=" * 55)

if fail_count > 0:
    print("\n  FAILED CHECKS:")
    for name, status, detail in validation_results:
        if status == "FAIL":
            print(f"    ❌ {name}: {detail}")

if fail_count == 0:
    print("\n  ✅ All validation checks passed!")
    print("  Sprint 3 (Retrieval / Search) is complete.")
    print("  Ready for Sprint 4: RAG generation layer.")
else:
    print(f"\n  ❌ {fail_count} check(s) failed. Review the details above.")

print("\n  Lineage:")
print("    Source PDFs → civ4_chunks → civ4_chunks_embeddings → Vector Search index → retrieval results")