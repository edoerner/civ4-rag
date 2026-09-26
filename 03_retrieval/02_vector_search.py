# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 03_retrieval / 02_vector_search
# MAGIC
# MAGIC **Purpose:** Create the Databricks Vector Search infrastructure (endpoint + Delta Sync index) over `civ4_chunks_embeddings`.
# MAGIC
# MAGIC **Inputs:**
# MAGIC - `workspace.gcivilization.civ4_chunks_embeddings` — Delta table with pre-computed 1024-dim embeddings
# MAGIC
# MAGIC **Outputs:**
# MAGIC - Vector Search endpoint: `civ4_vs_endpoint`
# MAGIC - Vector Search index: `workspace.gcivilization.civ4_chunks_embeddings_index`
# MAGIC
# MAGIC **Dependencies:**
# MAGIC - `databricks.sdk.WorkspaceClient` (pre-installed on serverless compute)
# MAGIC - The `databricks.vector_search.client.VectorSearchClient` is deprecated (renamed to `databricks-ai-search`) and has namespace resolution issues on serverless. This notebook uses the current `WorkspaceClient` SDK API instead.
# MAGIC
# MAGIC **Index type:** Delta Sync with **self-managed embeddings** — we provide the pre-computed `embedding` column from Notebook 01. The index syncs automatically from the Delta table.
# MAGIC
# MAGIC **Synchronization:**
# MAGIC - Pipeline type: `TRIGGERED` — sync is triggered manually via `sync_index()`.
# MAGIC - After new embeddings are added to the Delta table, re-run this notebook (or call `sync_index`) to propagate them to the Vector Search index.
# MAGIC - No external databases or infrastructure are introduced.
# MAGIC
# MAGIC **Idempotency:**
# MAGIC - Endpoint: reused if it already exists.
# MAGIC - Index: reused if it already exists.
# MAGIC - No duplicate endpoints or indexes are created on repeated runs.
# MAGIC
# MAGIC **Configuration:** All parameters defined in the Configuration cell below.
# MAGIC
# MAGIC **Next notebook:** `03_retrieval` implements the `retrieve(query, top_k)` function using the index created here.

# COMMAND ----------

# DBTITLE 1,1. Configuration
# ============================================================
# Configuration
# ============================================================
# All infrastructure names and parameters defined here.
# Uses a civ4-based naming convention.

CATALOG = "workspace"
SCHEMA = "gcivilization"
EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.civ4_chunks_embeddings"

# Vector Search infrastructure
ENDPOINT_NAME = "civ4_vs_endpoint"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.civ4_chunks_embeddings_index"

# Index configuration
PRIMARY_KEY = "chunk_id"
EMBEDDING_COLUMN = "embedding"
EMBEDDING_DIMENSION = 1024  # Verified from Notebook 01 output

# Columns to sync from Delta table to Vector Search index
# (embedding column is specified separately via embedding_vector_columns)
COLUMNS_TO_SYNC = [
    "chunk_id", "chunk_text", "document_id", "title",
    "author", "category", "game_version", "expansion",
    "source_url", "publication_date", "embedding_status",
]

print(f"Source table     : {EMBEDDINGS_TABLE}")
print(f"Endpoint name     : {ENDPOINT_NAME}")
print(f"Index name        : {INDEX_NAME}")
print(f"Primary key       : {PRIMARY_KEY}")
print(f"Embedding column  : {EMBEDDING_COLUMN} ({EMBEDDING_DIMENSION}-dim)")
print(f"Columns to sync   : {len(COLUMNS_TO_SYNC)} columns")

# COMMAND ----------

# DBTITLE 1,2. Create or reuse endpoint
# ============================================================
# Create or reuse Vector Search endpoint
# ============================================================
# Uses a STANDARD endpoint for low-latency semantic search.
# Idempotent: if the endpoint already exists, reuses it.

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import EndpointType, EndpointStatusState

w = WorkspaceClient()

try:
    endpoint = w.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
    ep_state = endpoint.endpoint_status.state
    print(f"Endpoint '{ENDPOINT_NAME}' already exists — state: {ep_state}")
    # If not online, wait for it
    if ep_state != EndpointStatusState.ONLINE:
        print("Waiting for endpoint to come online...")
        endpoint = w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(
            endpoint_name=ENDPOINT_NAME
        )
        print(f"Endpoint now online — state: {endpoint.endpoint_status.state}")
except Exception:
    print(f"Endpoint '{ENDPOINT_NAME}' not found — creating (STANDARD)...")
    endpoint = w.vector_search_endpoints.create_endpoint_and_wait(
        name=ENDPOINT_NAME,
        endpoint_type=EndpointType.STANDARD
    )
    print(f"Endpoint created — state: {endpoint.endpoint_status.state}")

print(f"\nEndpoint summary:")
print(f"  Name : {endpoint.name}")
print(f"  State: {endpoint.endpoint_status.state}")
print(f"  Type : {endpoint.endpoint_type}")

# COMMAND ----------

# DBTITLE 1,3. Prepare source table
# ============================================================
# Prepare source table for Delta Sync
# ============================================================
# Delta Sync indexes require:
#   1. Change Data Feed (CDF) enabled on the source table
#   2. A primary key constraint declared on the primary key column

# Enable CDF
spark.sql(f"""
ALTER TABLE {EMBEDDINGS_TABLE}
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print(f"Change Data Feed enabled on {EMBEDDINGS_TABLE}")

# Add primary key constraint (informational for Delta tables)
try:
    spark.sql(f"""
    ALTER TABLE {EMBEDDINGS_TABLE}
    ADD CONSTRAINT chunk_id_pk PRIMARY KEY ({PRIMARY_KEY})
    """)
    print(f"Primary key constraint added on {PRIMARY_KEY}")
except Exception as e:
    err_msg = str(e).lower()
    if "already exists" in err_msg or "duplicate" in err_msg or "constraint" in err_msg:
        print(f"Primary key constraint already exists on {PRIMARY_KEY}")
    else:
        print(f"Primary key note: {e}")

# Verify table properties
print(f"\nTable properties (CDF-related):")
spark.sql(f"SHOW TBLPROPERTIES {EMBEDDINGS_TABLE}").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,4. Create or reuse index
# ============================================================
# Create or reuse Vector Search index
# ============================================================
# Delta Sync index with self-managed embeddings:
#   - We provide the pre-computed embedding column
#   - The index syncs from the Delta table
#   - Pipeline type TRIGGERED: sync is manual via sync_index()

from databricks.sdk.service.vectorsearch import (
    VectorIndexType, PipelineType,
    DeltaSyncVectorIndexSpecRequest, EmbeddingVectorColumn
)

# Check if index already exists
try:
    index = w.vector_search_indexes.get_index(INDEX_NAME)
    print(f"Index '{INDEX_NAME}' already exists.")
    print(f"  Ready: {index.status.ready}")
    index_exists = True
except Exception:
    print(f"Index '{INDEX_NAME}' not found — creating...")
    index_exists = False

if not index_exists:
    w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=ENDPOINT_NAME,
        primary_key=PRIMARY_KEY,
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=EMBEDDINGS_TABLE,
            embedding_vector_columns=[
                EmbeddingVectorColumn(
                    name=EMBEDDING_COLUMN,
                    embedding_dimension=EMBEDDING_DIMENSION
                )
            ],
            pipeline_type=PipelineType.TRIGGERED,
            columns_to_sync=COLUMNS_TO_SYNC
        )
    )
    print(f"Index '{INDEX_NAME}' creation initiated.")

# COMMAND ----------

# DBTITLE 1,5. Trigger sync and check status
# ============================================================
# Wait for index readiness, trigger sync, verify data
# ============================================================
# The index goes through two phases:
#   1. Provisioning: endpoint allocates resources for the index
#   2. Sync: data from the Delta table is loaded into the index
# We must wait for provisioning to complete before triggering sync.

import time

max_wait = 600  # 10 minutes max
poll_interval = 15  # seconds

# --- Phase 1: Wait for index to be provisioned (ready=True) ---
print("Phase 1: Waiting for index provisioning...")
elapsed = 0
while elapsed < max_wait:
    index = w.vector_search_indexes.get_index(INDEX_NAME)
    ready = index.status.ready
    message = index.status.message or ""
    print(f"  [{elapsed:3d}s] ready={ready}, message={message}")
    
    if ready:
        print("\nIndex is provisioned and ready!")
        break
    
    time.sleep(poll_interval)
    elapsed += poll_interval
else:
    print(f"\nTimeout after {max_wait}s. Index not yet ready.")
    print("The index may still be provisioning. Re-run this cell later.")

# --- Phase 2: Trigger sync and wait for data ---
if index.status.ready:
    print("\nPhase 2: Triggering index sync...")
    w.vector_search_indexes.sync_index(INDEX_NAME)
    print("Sync triggered. Waiting for data to load...")
    
    elapsed = 0
    while elapsed < max_wait:
        index = w.vector_search_indexes.get_index(INDEX_NAME)
        row_count = index.status.indexed_row_count
        message = index.status.message or ""
        print(f"  [{elapsed:3d}s] indexed_rows={row_count}, message={message}")
        
        if row_count is not None and row_count > 0:
            print(f"\nSync complete! {row_count} rows indexed.")
            break
        
        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        print(f"\nTimeout waiting for sync. Current rows: {index.status.indexed_row_count}")
        print("The sync may still be running. Check back later.")

# COMMAND ----------

# DBTITLE 1,6. Summary
# ============================================================
# Summary
# ============================================================

# Final endpoint check
endpoint = w.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
print("=" * 55)
print("  VECTOR SEARCH INFRASTRUCTURE SUMMARY")
print("=" * 55)
print(f"  Endpoint name  : {endpoint.name}")
print(f"  Endpoint state : {endpoint.endpoint_status.state}")
print(f"  Endpoint type  : {endpoint.endpoint_type}")

# Final index check
index = w.vector_search_indexes.get_index(INDEX_NAME)
print(f"  Index name     : {INDEX_NAME}")
print(f"  Index ready    : {index.status.ready}")
print(f"  Indexed rows   : {index.status.indexed_row_count}")
print(f"  Index type     : {index.index_type}")
print(f"  Primary key    : {index.primary_key}")

# Show index spec details
spec = index.delta_sync_index_spec
if spec:
    print(f"  Source table   : {spec.source_table}")
    if spec.embedding_vector_columns:
        for evc in spec.embedding_vector_columns:
            print(f"  Embedding col  : {evc.name} ({evc.embedding_dimension}-dim)")
    print(f"  Pipeline type  : {spec.pipeline_type}")

print("=" * 55)
print("\nNotebook 02 complete. The index is ready for retrieval.")
print("Next: Run 03_retrieval to test semantic search queries.")