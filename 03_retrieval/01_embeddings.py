# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 03_retrieval / 01_embeddings
# MAGIC
# MAGIC **Purpose:** Generate embeddings for the validated `civ4_chunks` table using the `databricks-gte-large-en` model and persist them in a new Delta table.
# MAGIC
# MAGIC **Inputs:**
# MAGIC - `workspace.gcivilization.civ4_chunks` — validated Sprint 2 output
# MAGIC
# MAGIC **Outputs:**
# MAGIC - `workspace.gcivilization.civ4_chunks_embeddings` — Delta table with embeddings and full metadata
# MAGIC
# MAGIC **Dependencies:**
# MAGIC - `mlflow.deployments` (pre-installed on Databricks serverless compute)
# MAGIC - Embedding model: `databricks-gte-large-en` (built-in Foundation Model endpoint)
# MAGIC
# MAGIC **Idempotency:**
# MAGIC - MERGE on `chunk_id` — re-running updates existing rows, inserts new ones.
# MAGIC - The `embedding_model` column detects model changes; chunks embedded with a different model are regenerated.
# MAGIC - Failed embeddings (null embedding) are retried on subsequent runs.
# MAGIC - Chunks with missing/empty `chunk_text` are tracked with `embedding_status = 'SKIPPED'` — never silently discarded.
# MAGIC
# MAGIC **Configuration:** All parameters defined in the Configuration cell below.
# MAGIC
# MAGIC **Next notebook:** `02_vector_search` creates a Vector Search index over `civ4_chunks_embeddings`.

# COMMAND ----------

# DBTITLE 1,1. Configuration
# ============================================================
# Configuration
# ============================================================
# All paths, table names, and parameters defined here.
# Modify these if the workspace layout or model changes.

CATALOG = "workspace"
SCHEMA = "gcivilization"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.civ4_chunks"
EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.civ4_chunks_embeddings"
EMBEDDING_MODEL = "databricks-gte-large-en"
BATCH_SIZE = 10  # Texts per embedding API call (10 is reliable on serverless)

print(f"Source table       : {CHUNKS_TABLE}")
print(f"Embeddings table   : {EMBEDDINGS_TABLE}")
print(f"Embedding model    : {EMBEDDING_MODEL}")
print(f"Batch size         : {BATCH_SIZE}")

# COMMAND ----------

# DBTITLE 1,2. Load and identify chunks
# ============================================================
# Load source chunks and identify what needs (re)embedding
# ============================================================
# Reads civ4_chunks and determines which chunks need new
# embeddings based on: new chunks, model change, or prior failure.

from pyspark.sql.functions import col, length, trim

df_chunks = spark.table(CHUNKS_TABLE)
total_chunks = df_chunks.count()
print(f"Source chunks: {total_chunks}")

# Check if embeddings table already exists
embeddings_exist = spark.catalog.tableExists(EMBEDDINGS_TABLE)

if embeddings_exist:
    df_existing = spark.table(EMBEDDINGS_TABLE)
    existing_count = df_existing.count()
    print(f"Existing embeddings: {existing_count}")
    print("Embedding models in existing table:")
    df_existing.groupBy("embedding_model").count() \
        .orderBy("count", ascending=False).show()
else:
    print("No existing embeddings table — all chunks will be embedded.")

# Build the set of chunks that need (re)embedding
# A chunk needs embedding if:
#   1. It doesn't exist in the embeddings table, OR
#   2. Its embedding_model differs from the current model, OR
#   3. Its embedding is null (previously failed)
df_needs = df_chunks.select(
    "chunk_id", "chunk_text", "document_id", "title",
    "author", "category", "game_version", "expansion",
    "source_url", "publication_date"
)

if embeddings_exist:
    df_existing_keys = spark.table(EMBEDDINGS_TABLE).select(
        col("chunk_id"), col("embedding_model"), col("embedding")
    )
    df_needs = (
        df_needs.alias("c")
        .join(df_existing_keys.alias("e"),
              col("c.chunk_id") == col("e.chunk_id"), "left")
        .where(
            col("e.chunk_id").isNull()
            | (col("e.embedding_model") != EMBEDDING_MODEL)
            | col("e.embedding").isNull()
        )
        .select("c.*")
    )

needs_count = df_needs.count()
print(f"\nChunks needing (re)embedding: {needs_count}")

# Separate valid and invalid (missing/empty) chunk_text
df_valid = df_needs.filter(
    col("chunk_text").isNotNull() & (length(trim(col("chunk_text"))) > 0)
)
df_invalid = df_needs.filter(
    col("chunk_text").isNull() | (length(trim(col("chunk_text"))) == 0)
)

valid_count = df_valid.count()
invalid_count = df_invalid.count()
print(f"  Valid (will embed)  : {valid_count}")
print(f"  Invalid (skip)     : {invalid_count}")

if invalid_count > 0:
    print("\nChunks with missing/empty text:")
    df_invalid.select("chunk_id", "document_id").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,3. Generate embeddings
# ============================================================
# Generate embeddings
# ============================================================
# Calls the databricks-gte-large-en endpoint in batches.
# Captures failures without stopping the pipeline.
# The embedding dimension is determined from actual model output.

import mlflow.deployments
from datetime import datetime, timezone

deploy_client = mlflow.deployments.get_deploy_client("databricks")

all_records = []
all_embeddings = []
embedding_dimension = None
new_success = 0
new_failed = 0

# If no new embeddings needed, try to get dimension from existing table
if embedding_dimension is None and embeddings_exist:
    dim_row = spark.table(EMBEDDINGS_TABLE) \
        .filter("embedding IS NOT NULL") \
        .select("embedding_dimension").first()
    if dim_row:
        embedding_dimension = dim_row.embedding_dimension

# --- Process valid chunks ---
if valid_count > 0:
    valid_rows = df_valid.collect()
    texts = [row.chunk_text for row in valid_rows]

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        try:
            response = deploy_client.predict(
                endpoint=EMBEDDING_MODEL,
                inputs={"input": batch}
            )
            batch_embeddings = [item["embedding"] for item in response.data]

            # Determine embedding dimension from first successful batch
            if embedding_dimension is None and batch_embeddings:
                embedding_dimension = len(batch_embeddings[0])
                print(f"Embedding dimension determined: {embedding_dimension}")

            all_embeddings.extend(batch_embeddings)
            new_success += len(batch)
            print(f"  Batch {batch_num}: {len(batch)} embeddings OK")
        except Exception as e:
            print(f"  Batch {batch_num}: FAILED — {e}")
            all_embeddings.extend([None] * len(batch))
            new_failed += len(batch)

    # Build records for valid chunks
    for idx, row in enumerate(valid_rows):
        emb = all_embeddings[idx]
        all_records.append({
            "chunk_id": row.chunk_id,
            "chunk_text": row.chunk_text,
            "document_id": row.document_id,
            "title": row.title,
            "author": row.author,
            "category": row.category,
            "game_version": row.game_version,
            "expansion": row.expansion,
            "source_url": row.source_url,
            "publication_date": row.publication_date,
            "embedding": emb,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": len(emb) if emb else embedding_dimension,
            "embedding_timestamp": datetime.now(timezone.utc),
            "embedding_status": "SUCCESS" if emb is not None else "FAILED",
        })
else:
    print("No valid chunks need embedding.")

# --- Process skipped chunks (missing/empty text) ---
if invalid_count > 0:
    for row in df_invalid.collect():
        all_records.append({
            "chunk_id": row.chunk_id,
            "chunk_text": row.chunk_text,
            "document_id": row.document_id,
            "title": row.title,
            "author": row.author,
            "category": row.category,
            "game_version": row.game_version,
            "expansion": row.expansion,
            "source_url": row.source_url,
            "publication_date": row.publication_date,
            "embedding": None,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": embedding_dimension,
            "embedding_timestamp": datetime.now(timezone.utc),
            "embedding_status": "SKIPPED",
        })

# --- Validate embedding dimensions ---
if all_embeddings:
    dim_set = set(len(e) for e in all_embeddings if e is not None)
    if len(dim_set) > 1:
        print(f"WARNING: Inconsistent embedding dimensions: {dim_set}")
    elif dim_set:
        print(f"All embeddings have consistent dimension: {dim_set.pop()}")

print(f"\nNew embeddings generated: {new_success}")
print(f"Failed embeddings      : {new_failed}")
print(f"Skipped (no text)      : {invalid_count}")
print(f"Total records to upsert: {len(all_records)}")

# COMMAND ----------

# DBTITLE 1,4. Write to Delta table
# ============================================================
# Write to Delta table (MERGE on chunk_id)
# ============================================================
# Uses MERGE so re-running the notebook updates existing rows
# instead of duplicating them.

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    TimestampType, DateType, ArrayType, FloatType
)

if all_records:
    schema = StructType([
        StructField("chunk_id", StringType(), True),
        StructField("chunk_text", StringType(), True),
        StructField("document_id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("author", StringType(), True),
        StructField("category", StringType(), True),
        StructField("game_version", StringType(), True),
        StructField("expansion", StringType(), True),
        StructField("source_url", StringType(), True),
        StructField("publication_date", DateType(), True),
        StructField("embedding", ArrayType(FloatType()), True),
        StructField("embedding_model", StringType(), True),
        StructField("embedding_dimension", IntegerType(), True),
        StructField("embedding_timestamp", TimestampType(), True),
        StructField("embedding_status", StringType(), True),
    ])

    df_updates = spark.createDataFrame(all_records, schema=schema)

    # Ensure schema exists
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

    # Create table if not exists
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE}
    (
        chunk_id STRING,
        chunk_text STRING,
        document_id STRING,
        title STRING,
        author STRING,
        category STRING,
        game_version STRING,
        expansion STRING,
        source_url STRING,
        publication_date DATE,
        embedding ARRAY<FLOAT>,
        embedding_model STRING,
        embedding_dimension INT,
        embedding_timestamp TIMESTAMP,
        embedding_status STRING
    )
    USING DELTA
    """)

    # MERGE: update existing rows, insert new ones
    df_updates.createOrReplaceTempView("embedding_updates")
    spark.sql(f"""
    MERGE INTO {EMBEDDINGS_TABLE} AS target
    USING embedding_updates AS source
    ON target.chunk_id = source.chunk_id
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
    """)

    print(f"Embeddings table updated: {EMBEDDINGS_TABLE}")
else:
    print("No records to upsert — all embeddings are up to date.")

# COMMAND ----------

# DBTITLE 1,5. Summary statistics
# ============================================================
# Summary statistics
# ============================================================

df_final = spark.table(EMBEDDINGS_TABLE)
final_count = df_final.count()
success_count = df_final.filter("embedding_status = 'SUCCESS'").count()
failed_count = df_final.filter("embedding_status = 'FAILED'").count()
skipped_count = df_final.filter("embedding_status = 'SKIPPED'").count()
null_emb_count = df_final.filter("embedding IS NULL").count()

# Embedding dimension check
dim_df = df_final.filter("embedding IS NOT NULL") \
    .select("embedding_dimension").distinct()
dim_values = [row.embedding_dimension for row in dim_df.collect()]

print("=" * 55)
print("  EMBEDDING SUMMARY — civ4_chunks_embeddings")
print("=" * 55)
print(f"  Input chunks            : {total_chunks}")
print(f"  Chunks needing embed    : {needs_count}")
print(f"  New embeddings generated: {new_success}")
print(f"  Failed embeddings       : {new_failed}")
print(f"  Skipped (no text)       : {skipped_count}")
print(f"  Final embedding count   : {success_count}")
print(f"  Embedding dimension(s)   : {dim_values}")
print(f"  Embedding model         : {EMBEDDING_MODEL}")
print(f"  Null embeddings         : {null_emb_count}")
print("=" * 55)

# Show sample
print(f"\nSample embeddings:")
spark.sql(f"""
SELECT chunk_id, embedding_status, embedding_dimension,
       size(embedding) as actual_dim,
       substring(chunk_text, 1, 80) as text_preview
FROM {EMBEDDINGS_TABLE}
WHERE embedding_status = 'SUCCESS'
LIMIT 5
""").show(truncate=False)