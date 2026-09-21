# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 02_processing / 02_metadata_and_chunking
# MAGIC
# MAGIC Reads the bronze table `civ4_pdf_bronze`, validates and enriches metadata, chunks the text into reproducible segments, and writes the silver-grade `civ4_chunks` Delta table for downstream retrieval.

# COMMAND ----------

# DBTITLE 1,1. Configuration
# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC
# MAGIC Centralizes all path, table, and chunking parameters. Modify these variables if the workspace layout or chunking strategy changes.

# COMMAND ----------

# DBTITLE 1,1. Configuration
# ============================================================
# Configuration
# ============================================================

# --- Unity Catalog ---
CATALOG = "workspace"
SCHEMA = "gcivilization"
BRONZE_TABLE = "civ4_pdf_bronze"
CHUNKS_TABLE = "civ4_chunks"
FULL_BRONZE_NAME = f"{CATALOG}.{SCHEMA}.{BRONZE_TABLE}"
FULL_CHUNKS_NAME = f"{CATALOG}.{SCHEMA}.{CHUNKS_TABLE}"

# --- Metadata CSV (for cross-validation) ---
METADATA_CSV_PATH = "/Volumes/workspace/gcivilization/civ4_rag/civ4_strategy_metadata.csv"

# --- Chunking parameters ---
CHUNK_SIZE = 1000       # Target characters per chunk
CHUNK_OVERLAP = 150     # Overlap characters between consecutive chunks

# --- Validate chunking configuration ---
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        f"Invalid configuration: chunk_overlap ({CHUNK_OVERLAP}) "
        f"must be strictly less than chunk_size ({CHUNK_SIZE})."
    )

print(f"Bronze table   : {FULL_BRONZE_NAME}")
print(f"Chunks table   : {FULL_CHUNKS_NAME}")
print(f"Chunk size     : {CHUNK_SIZE}")
print(f"Chunk overlap  : {CHUNK_OVERLAP}")

# COMMAND ----------

# DBTITLE 1,2. Load Bronze Data
# MAGIC %md
# MAGIC ## 2. Load Bronze Data
# MAGIC
# MAGIC Reads the `civ4_pdf_bronze` Delta table and prints the schema and extraction status distribution.

# COMMAND ----------

# DBTITLE 1,2. Load Bronze Data
# ============================================================
# Load Bronze Data
# ============================================================

df_bronze = spark.table(FULL_BRONZE_NAME)
bronze_count = df_bronze.count()
print(f"Bronze records loaded: {bronze_count}")
print("\nSchema:")
df_bronze.printSchema()

print("\nExtraction status distribution:")
df_bronze.groupBy("extraction_status").count() \
    .orderBy("extraction_status").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,3. Metadata Validation & Enrichment
# MAGIC %md
# MAGIC ## 3. Metadata Validation & Enrichment
# MAGIC
# MAGIC Loads the metadata CSV and cross-checks it against the bronze table. Fills any null metadata fields from the CSV. Handles `publication_date` carefully — unparseable values are kept as NULL rather than silently corrected.

# COMMAND ----------

# DBTITLE 1,3. Metadata Validation & Enrichment
# ============================================================
# Metadata Validation & Enrichment
# ============================================================

from pyspark.sql.functions import col
from datetime import datetime as _dt, date as _date

# --- Load metadata CSV ---
df_metadata = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(METADATA_CSV_PATH)
)
metadata_count = df_metadata.count()
print(f"Metadata CSV records: {metadata_count}")

# --- Cross-check document IDs ---
bronze_ids = {r.document_id for r in df_bronze.select("document_id").collect()}
metadata_ids = {r.document_id for r in df_metadata.select("document_id").collect()}

ids_bronze_only = bronze_ids - metadata_ids
ids_metadata_only = metadata_ids - bronze_ids
print(f"\nDocument IDs in bronze but not in metadata CSV: {len(ids_bronze_only)}")
if ids_bronze_only:
    for did in sorted(ids_bronze_only):
        print(f"  {did}")
print(f"Document IDs in metadata CSV but not in bronze: {len(ids_metadata_only)}")
if ids_metadata_only:
    for did in sorted(ids_metadata_only):
        print(f"  {did}")

# --- Metadata completeness in bronze ---
metadata_fields = ["title", "author", "category", "game_version",
                   "expansion", "source_url", "publication_date"]
print("\n=== Metadata completeness in bronze ===")
for field in metadata_fields:
    null_count = df_bronze.filter(col(field).isNull()).count()
    status = "complete" if null_count == 0 else f"{null_count} null(s)"
    print(f"  {field:20s}: {status}")


# --- Safe date parser for CSV fallback ---
def safe_parse_date(val):
    """Try to convert a value to a Python date. Return None if unparseable."""
    if val is None:
        return None
    if isinstance(val, _date):
        return val
    if isinstance(val, _dt):
        return val.date()
    if isinstance(val, str):
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y"):
            try:
                return _dt.strptime(val.strip(), fmt).date()
            except ValueError:
                continue
    return None  # Cannot parse — keep None rather than inventing a date


# --- Build metadata lookup from CSV for enrichment ---
metadata_lookup = {}
for row in df_metadata.collect():
    metadata_lookup[row.pdf_filename] = {
        "document_id": row.document_id,
        "title": row.title,
        "author": row.author,
        "category": row.category,
        "game_version": row.game_version,
        "expansion": row.expansion,
        "source_url": row.source_url,
        "publication_date": safe_parse_date(row.publication_date),
    }

print("\nMetadata lookup built from CSV.")

# COMMAND ----------

# DBTITLE 1,4. Document-level Validation
# MAGIC %md
# MAGIC ## 4. Document-level Validation
# MAGIC
# MAGIC Validates extraction results before chunking. Documents with `extraction_status != 'SUCCESS'` are skipped — they will not produce chunks.

# COMMAND ----------

# DBTITLE 1,4. Document-level Validation
# ============================================================
# Document-level Validation
# ============================================================

print("=== Document-level validation ===\n")

# Status distribution
print("Extraction status distribution:")
df_bronze.groupBy("extraction_status").count() \
    .orderBy("extraction_status").show(truncate=False)

# Empty text
empty_text_count = df_bronze.filter("character_count = 0").count()
print(f"Documents with empty text: {empty_text_count}")

# Metadata match status
print("\nMetadata match status:")
df_bronze.groupBy("metadata_match_status").count().show(truncate=False)

# Duplicate document_id
dup_ids = df_bronze.groupBy("document_id").count().filter("count > 1")
dup_count = dup_ids.count()
if dup_count > 0:
    print(f"WARNING: {dup_count} duplicate document_id(s) found!")
    dup_ids.show(truncate=False)
else:
    print("No duplicate document_id values.")

# Eligible vs skipped
eligible_count = df_bronze.filter("extraction_status = 'SUCCESS'").count()
skipped_count = bronze_count - eligible_count
print(f"\nDocuments eligible for chunking (SUCCESS): {eligible_count}")
print(f"Documents skipped (non-SUCCESS): {skipped_count}")

if skipped_count > 0:
    print("\nSkipped documents:")
    df_bronze.filter("extraction_status != 'SUCCESS'") \
        .select("document_id", "pdf_filename", "extraction_status",
                "character_count", "extraction_error") \
        .show(truncate=False)

# COMMAND ----------

# DBTITLE 1,5. Chunking Configuration
# MAGIC %md
# MAGIC ## 5. Chunking Configuration
# MAGIC
# MAGIC Defines the chunking function and validates parameters.
# MAGIC
# MAGIC **Strategy:**
# MAGIC - Character-based chunking with configurable size and overlap.
# MAGIC - Prefers natural boundaries (paragraphs, lines, sentences) in the last 20% of each chunk.
# MAGIC - Purely deterministic — same input always produces same output.
# MAGIC - No external dependencies (no `tiktoken`, `langchain`, etc.).
# MAGIC
# MAGIC Parameters `CHUNK_SIZE` and `CHUNK_OVERLAP` are defined in section 1. The overlap must be strictly less than the chunk size to guarantee forward progress.

# COMMAND ----------

# DBTITLE 1,5. Chunking Configuration
# ============================================================
# Chunking Configuration
# ============================================================

def chunk_text(text, chunk_size, chunk_overlap):
    """
    Split text into overlapping chunks, preferring natural boundaries.

    Targets ~chunk_size characters per chunk. Tries to break at
    paragraph (\\n\\n), line (\\n), sentence (. / ! / ?) or word (space)
    boundaries in the last 20% of each chunk. Applies chunk_overlap
    characters of overlap between consecutive chunks.

    Returns a list of string chunks. Deterministic: same input
    always produces the same output.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            # Search for natural break in the last 20% of the target chunk
            search_start = start + int(chunk_size * 0.8)
            for breaker in ["\n\n", "\n", ". ", "! ", "? ", " "]:
                pos = text.rfind(breaker, search_start, end)
                if pos > start:
                    end = pos + len(breaker)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Next chunk starts with overlap
        start = end - chunk_overlap

    return chunks


print("Chunking function defined.")
print(f"  chunk_size    = {CHUNK_SIZE}")
print(f"  chunk_overlap = {CHUNK_OVERLAP}")

# COMMAND ----------

# DBTITLE 1,6. Chunk Documents & Build Chunk Metadata
# MAGIC %md
# MAGIC ## 6. Chunk Documents & Build Chunk Metadata
# MAGIC
# MAGIC Processes eligible documents (`extraction_status = 'SUCCESS'`), chunks their text, and propagates metadata to each chunk.
# MAGIC
# MAGIC **Chunk ID strategy:** `{document_id}_chunk_{chunk_index:04d}` — deterministic and stable across runs.

# COMMAND ----------

# DBTITLE 1,6. Chunk Documents & Build Chunk Metadata
# ============================================================
# Chunk Documents & Build Chunk Metadata
# ============================================================

from datetime import datetime, timezone
from collections import Counter

# Collect eligible documents (SUCCESS only)
eligible_rows = df_bronze.filter("extraction_status = 'SUCCESS'").collect()
print(f"Processing {len(eligible_rows)} eligible documents...")

chunk_records = []
documents_with_zero_chunks = []

for row in eligible_rows:
    doc_id = row["document_id"]
    raw_text = row["raw_text"] or ""

    # Enrich metadata from CSV if any fields are null
    meta = metadata_lookup.get(row["pdf_filename"], {})

    chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)

    if len(chunks) == 0:
        documents_with_zero_chunks.append(doc_id)
        continue

    for idx, chunk_content in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{idx:04d}"
        record = {
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "pdf_filename": row["pdf_filename"],
            "title": row["title"] or meta.get("title"),
            "author": row["author"] or meta.get("author"),
            "category": row["category"] or meta.get("category"),
            "game_version": row["game_version"] or meta.get("game_version"),
            "expansion": row["expansion"] or meta.get("expansion"),
            "source_url": row["source_url"] or meta.get("source_url"),
            "publication_date": row["publication_date"] or meta.get("publication_date"),
            "chunk_index": idx,
            "chunk_text": chunk_content,
            "chunk_character_count": len(chunk_content),
            "chunking_timestamp": datetime.now(timezone.utc),
        }
        chunk_records.append(record)

total_chunks = len(chunk_records)
print(f"\nTotal chunks generated: {total_chunks}")
print(f"Documents with zero chunks: {len(documents_with_zero_chunks)}")
if documents_with_zero_chunks:
    for did in documents_with_zero_chunks:
        print(f"  {did}")

# Chunks per document
chunks_per_doc = Counter(r["document_id"] for r in chunk_records)
print("\nChunks per document:")
for did in sorted(chunks_per_doc):
    print(f"  {did:55s}  {chunks_per_doc[did]:4d} chunks")

# COMMAND ----------

# DBTITLE 1,7. Chunk-level Data Quality Checks
# MAGIC %md
# MAGIC ## 7. Chunk-level Data Quality Checks
# MAGIC
# MAGIC Validates the generated chunks before writing to Delta:
# MAGIC - Total chunk count
# MAGIC - Empty chunk_text
# MAGIC - Duplicate chunk_id
# MAGIC - Duplicate (document_id, chunk_index) pairs
# MAGIC - Metadata propagation to each chunk
# MAGIC - Documents without any chunk

# COMMAND ----------

# DBTITLE 1,7. Chunk-level Data Quality Checks
# ============================================================
# Chunk-level Data Quality Checks
# ============================================================

from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    TimestampType, DateType,
)

# --- Define schema ---
chunks_schema = StructType([
    StructField("document_id", StringType(), True),
    StructField("chunk_id", StringType(), True),
    StructField("pdf_filename", StringType(), True),
    StructField("title", StringType(), True),
    StructField("author", StringType(), True),
    StructField("category", StringType(), True),
    StructField("game_version", StringType(), True),
    StructField("expansion", StringType(), True),
    StructField("source_url", StringType(), True),
    StructField("publication_date", DateType(), True),
    StructField("chunk_index", IntegerType(), True),
    StructField("chunk_text", StringType(), True),
    StructField("chunk_character_count", IntegerType(), True),
    StructField("chunking_timestamp", TimestampType(), True),
])

# --- Create Spark DataFrame ---
if chunk_records:
    df_chunks = spark.createDataFrame(chunk_records, schema=chunks_schema)
else:
    df_chunks = spark.sql("""
        SELECT
            CAST(NULL AS STRING) AS document_id,
            CAST(NULL AS STRING) AS chunk_id,
            CAST(NULL AS STRING) AS pdf_filename,
            CAST(NULL AS STRING) AS title,
            CAST(NULL AS STRING) AS author,
            CAST(NULL AS STRING) AS category,
            CAST(NULL AS STRING) AS game_version,
            CAST(NULL AS STRING) AS expansion,
            CAST(NULL AS STRING) AS source_url,
            CAST(NULL AS DATE) AS publication_date,
            CAST(NULL AS INT) AS chunk_index,
            CAST(NULL AS STRING) AS chunk_text,
            CAST(NULL AS INT) AS chunk_character_count,
            CAST(NULL AS TIMESTAMP) AS chunking_timestamp
        LIMIT 0
    """)

total_chunks_df = df_chunks.count()
print(f"Total chunks in DataFrame: {total_chunks_df}")

# --- DQ: empty chunk_text ---
empty_chunks = df_chunks.filter("chunk_character_count = 0").count()
print(f"Chunks with empty text: {empty_chunks}")

# --- DQ: duplicate chunk_id ---
dup_chunk_ids = df_chunks.groupBy("chunk_id").count().filter("count > 1")
dup_chunk_count = dup_chunk_ids.count()
if dup_chunk_count > 0:
    print(f"WARNING: {dup_chunk_count} duplicate chunk_id(s)!")
    dup_chunk_ids.show(truncate=False)
else:
    print("No duplicate chunk_id values.")

# --- DQ: duplicate (document_id, chunk_index) ---
dup_doc_chunk = df_chunks.groupBy("document_id", "chunk_index").count().filter("count > 1")
dup_doc_chunk_count = dup_doc_chunk.count()
if dup_doc_chunk_count > 0:
    print(f"WARNING: {dup_doc_chunk_count} duplicate (document_id, chunk_index) pair(s)!")
else:
    print("No duplicate (document_id, chunk_index) pairs.")

# --- DQ: metadata propagation ---
required_meta = ["document_id", "pdf_filename", "title", "category", "source_url"]
print("\n=== Metadata propagation check ===")
for field in required_meta:
    null_count = df_chunks.filter(col(field).isNull()).count()
    status = "complete" if null_count == 0 else f"{null_count} null(s)"
    print(f"  {field:20s}: {status}")

# --- DQ: documents without any chunk ---
all_doc_ids = {r.document_id for r in df_bronze.select("document_id").collect()}
chunked_doc_ids = {r.document_id for r in df_chunks.select("document_id").collect()}
docs_without_chunks = all_doc_ids - chunked_doc_ids
print(f"\nDocuments without any chunk: {len(docs_without_chunks)}")
if docs_without_chunks:
    for did in sorted(docs_without_chunks):
        status = df_bronze.filter(f"document_id = '{did}'") \
            .select("extraction_status").collect()[0][0]
        print(f"  {did}  (extraction_status={status})")

# --- DQ: chunk length summary ---
print("\n=== Chunk length summary ===")
df_chunks.selectExpr(
    "MIN(chunk_character_count) AS min_len",
    "MAX(chunk_character_count) AS max_len",
    "ROUND(AVG(chunk_character_count), 1) AS avg_len"
).show()

# COMMAND ----------

# DBTITLE 1,8. Write civ4_chunks
# MAGIC %md
# MAGIC ## 8. Write civ4_chunks
# MAGIC
# MAGIC Writes the chunks to `workspace.gcivilization.civ4_chunks` using **overwrite** mode.
# MAGIC
# MAGIC **Why overwrite (not MERGE)?**
# MAGIC - Chunk IDs are deterministic (`document_id + _chunk_ + index`), so every run regenerates the same set.
# MAGIC - If a document's text changes and produces fewer chunks, MERGE would leave orphan rows from the previous run.
# MAGIC - Overwrite replaces all rows each run, guaranteeing a clean, idempotent result.

# COMMAND ----------

# DBTITLE 1,8. Write civ4_chunks
# ============================================================
# Write civ4_chunks Delta table
# ============================================================

# Ensure schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# Write with overwrite (idempotent: deterministic chunk_ids)
df_chunks.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(FULL_CHUNKS_NAME)

row_count = spark.sql(f"SELECT COUNT(*) AS cnt FROM {FULL_CHUNKS_NAME}").collect()[0][0]
print(f"Chunks table written: {FULL_CHUNKS_NAME}")
print(f"Total rows in table: {row_count}")

# COMMAND ----------

# DBTITLE 1,9. Chunk Statistics
# MAGIC %md
# MAGIC ## 9. Chunk Statistics
# MAGIC
# MAGIC Distribution of chunks per document and overall length statistics for diagnostics.

# COMMAND ----------

# DBTITLE 1,9. Chunk Statistics
# ============================================================
# Chunk Statistics
# ============================================================

print("=== Chunk distribution per document ===")
spark.sql(f"""
SELECT
    document_id,
    COUNT(*) AS number_of_chunks,
    MIN(chunk_character_count) AS min_chunk_length,
    MAX(chunk_character_count) AS max_chunk_length,
    ROUND(AVG(chunk_character_count), 1) AS avg_chunk_length
FROM {FULL_CHUNKS_NAME}
GROUP BY document_id
ORDER BY document_id
""").show(truncate=False)

print("\n=== Overall chunk statistics ===")
spark.sql(f"""
SELECT
    COUNT(*) AS total_chunks,
    MIN(chunk_character_count) AS min_length,
    MAX(chunk_character_count) AS max_length,
    ROUND(AVG(chunk_character_count), 1) AS avg_length
FROM {FULL_CHUNKS_NAME}
""").show(truncate=False)

print("\n=== Sample chunks ===")
spark.sql(f"""
SELECT
    document_id,
    chunk_id,
    chunk_index,
    title,
    category,
    chunk_character_count
FROM {FULL_CHUNKS_NAME}
ORDER BY document_id, chunk_index
LIMIT 20
""").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Next Step
# MAGIC %md
# MAGIC ## Next Step
# MAGIC
# MAGIC The output of this notebook — the `civ4_chunks` Delta table — will be validated by:
# MAGIC
# MAGIC **`02_processing/03_validation`**
# MAGIC
# MAGIC That notebook will verify chunk integrity, metadata completeness, and idempotency.
# MAGIC
# MAGIC After validation, `civ4_chunks` will be consumed by **Sprint 3 — Retrieval / Search** for:
# MAGIC - generating embeddings from `chunk_text`
# MAGIC - creating the Vector Search / AI Search index
# MAGIC - implementing retrieval
# MAGIC
# MAGIC > **Note:** Documents with `extraction_status != 'SUCCESS'` (image-based PDFs without a text layer) were skipped. They will require OCR processing before they can participate in retrieval.