# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 02_processing / 03_validation
# MAGIC
# MAGIC Validation notebook for the `civ4_chunks` Delta table. Performs comprehensive data quality checks across five dimensions — table integrity, chunk integrity, metadata completeness, traceability, and idempotency — to determine whether the dataset is **READY_FOR_SPRINT_3** (retrieval / search) or **NOT_READY_FOR_SPRINT_3**.
# MAGIC
# MAGIC This notebook is **diagnostic only**. It does not modify, correct, or write any data.

# COMMAND ----------

# DBTITLE 1,1. Configuration
# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC
# MAGIC Centralizes table names, expected schema, and the validation results collector. Modify `CATALOG` and `SCHEMA` if the workspace layout changes.

# COMMAND ----------

# DBTITLE 1,1. Configuration
# ============================================================
# Configuration
# ============================================================

# --- Unity Catalog ---
CATALOG = "workspace"
SCHEMA = "gcivilization"
CHUNKS_TABLE = "civ4_chunks"
BRONZE_TABLE = "civ4_pdf_bronze"
FULL_CHUNKS_NAME = f"{CATALOG}.{SCHEMA}.{CHUNKS_TABLE}"
FULL_BRONZE_NAME = f"{CATALOG}.{SCHEMA}.{BRONZE_TABLE}"

# --- Expected schema (minimum required columns) ---
EXPECTED_COLUMNS = [
    "document_id",
    "chunk_id",
    "pdf_filename",
    "title",
    "author",
    "category",
    "game_version",
    "expansion",
    "source_url",
    "publication_date",
    "chunk_index",
    "chunk_text",
    "chunk_character_count",
]

# --- Metadata columns for completeness / consistency checks ---
METADATA_COLUMNS = [
    "pdf_filename",
    "title",
    "author",
    "category",
    "game_version",
    "expansion",
    "source_url",
    "publication_date",
]

# --- Columns with NOT NULL constraint ---
NOT_NULL_COLS = ["document_id", "chunk_id", "chunk_index", "chunk_text"]

# --- Validation results collector ---
validation_results = []


def add_result(check_name, check_category, severity, status, affected_rows, description):
    """Append a validation result row."""
    validation_results.append({
        "check_name": check_name,
        "check_category": check_category,
        "severity": severity,
        "status": status,
        "affected_rows": affected_rows,
        "description": description,
    })


print(f"Chunks table  : {FULL_CHUNKS_NAME}")
print(f"Bronze table  : {FULL_BRONZE_NAME}")
print(f"Expected columns ({len(EXPECTED_COLUMNS)}): {', '.join(EXPECTED_COLUMNS)}")
print("\nValidation framework initialised.")

# COMMAND ----------

# DBTITLE 1,2. Load civ4_chunks
# MAGIC %md
# MAGIC ## 2. Load civ4_chunks
# MAGIC
# MAGIC Reads the `civ4_chunks` Delta table. If the table cannot be loaded, all subsequent checks are skipped and reported as `NOT_VALIDATED`.

# COMMAND ----------

# DBTITLE 1,2. Load civ4_chunks
# ============================================================
# Load civ4_chunks
# ============================================================

from pyspark.sql.functions import (
    col, count, countDistinct, when, lit, length, trim,
    concat, lpad, min as sql_min, max as sql_max,
)

table_loadable = False
df_chunks = None
total_chunks = 0
total_documents = 0
unique_chunks = 0

try:
    df_chunks = spark.table(FULL_CHUNKS_NAME)
    total_chunks = df_chunks.count()
    total_documents = df_chunks.select("document_id").distinct().count()
    unique_chunks = df_chunks.select("chunk_id").distinct().count()
    table_loadable = True
    print(f"Table {FULL_CHUNKS_NAME} loaded successfully.")
    print(f"  Total chunks    : {total_chunks}")
    print(f"  Total documents : {total_documents}")
    print(f"  Unique chunk_ids: {unique_chunks}")
except Exception as e:
    print(f"ERROR: Could not load table {FULL_CHUNKS_NAME}")
    print(f"  Exception: {str(e)}")
    add_result("table_loadable", "Table Integrity", "ERROR", "FAIL", 0,
               f"Table could not be loaded: {str(e)[:200]}")

# COMMAND ----------

# DBTITLE 1,3. Schema and Table Integrity
# MAGIC %md
# MAGIC ## 3. Schema and Table Integrity
# MAGIC
# MAGIC Verifies that the table exists, is loadable, contains records, has all required columns, reasonable data types, and no NULL values in critical columns (`document_id`, `chunk_id`, `chunk_index`, `chunk_text`).

# COMMAND ----------

# DBTITLE 1,3. Schema and Table Integrity
# ============================================================
# 3. Schema and Table Integrity
# ============================================================

if table_loadable:
    print("=== Schema ===")
    df_chunks.printSchema()

    actual_columns = set(df_chunks.columns)
    expected_set = set(EXPECTED_COLUMNS)

    # --- Table exists ---
    add_result("table_exists", "Table Integrity", "PASS", "PASS", 0,
               f"Table {FULL_CHUNKS_NAME} exists and is loadable.")

    # --- Has records ---
    if total_chunks > 0:
        add_result("has_records", "Table Integrity", "PASS", "PASS", total_chunks,
                   f"Table contains {total_chunks} records.")
    else:
        add_result("has_records", "Table Integrity", "ERROR", "FAIL", 0,
                   "Table contains 0 records.")

    # --- Required columns present ---
    missing_columns = expected_set - actual_columns
    extra_columns = actual_columns - expected_set

    if not missing_columns:
        add_result("required_columns_present", "Table Integrity", "PASS", "PASS", 0,
                   f"All {len(EXPECTED_COLUMNS)} required columns are present.")
    else:
        add_result("required_columns_present", "Table Integrity", "ERROR", "FAIL",
                   len(missing_columns),
                   f"Missing columns: {', '.join(sorted(missing_columns))}")

    if extra_columns:
        print(f"\nExtra columns (allowed): {', '.join(sorted(extra_columns))}")

    # --- Critical NOT NULL columns ---
    for c in NOT_NULL_COLS:
        if c in actual_columns:
            null_count = df_chunks.filter(col(c).isNull()).count()
            if null_count == 0:
                add_result(f"{c}_not_null", "Table Integrity", "PASS", "PASS", 0,
                           f"{c} has no NULL values.")
            else:
                add_result(f"{c}_not_null", "Table Integrity", "ERROR", "FAIL",
                           null_count, f"{c} has {null_count} NULL values.")
        else:
            add_result(f"{c}_not_null", "Table Integrity", "ERROR", "FAIL", 0,
                       f"Column {c} not present in schema.")

    # --- Data types reasonable ---
    schema_types = {f.name: str(f.dataType).lower() for f in df_chunks.schema.fields}
    type_expectations = {
        "document_id": "string",
        "chunk_id": "string",
        "pdf_filename": "string",
        "chunk_index": "int",
        "chunk_text": "string",
        "chunk_character_count": "int",
    }
    type_issues = []
    for col_name, exp_type in type_expectations.items():
        if col_name in schema_types:
            if exp_type not in schema_types[col_name]:
                type_issues.append(
                    f"{col_name}: expected {exp_type}, got {schema_types[col_name]}"
                )

    if not type_issues:
        add_result("data_types_reasonable", "Table Integrity", "PASS", "PASS", 0,
                   "All checked columns have reasonable data types.")
    else:
        add_result("data_types_reasonable", "Table Integrity", "ERROR", "FAIL",
                   len(type_issues),
                   f"Type mismatches: {'; '.join(type_issues)}")
else:
    for check in (["table_exists", "has_records", "required_columns_present",
                   "data_types_reasonable"]
                  + [f"{c}_not_null" for c in NOT_NULL_COLS]):
        add_result(check, "Table Integrity", "NOT_VALIDATED", "NOT_VALIDATED", 0,
                   "Cannot check: table not loadable.")

# COMMAND ----------

# DBTITLE 1,4. Chunk Integrity
# MAGIC %md
# MAGIC ## 4. Chunk Integrity
# MAGIC
# MAGIC Validates that individual chunks are consistent:
# MAGIC - Empty chunks (`chunk_text IS NULL` or `TRIM(chunk_text) = ''`)
# MAGIC - `chunk_character_count` consistency with `LENGTH(chunk_text)`
# MAGIC - `chunk_id` uniqueness and determinism (pattern: `{document_id}_chunk_{index:04d}`)
# MAGIC - `chunk_index` NULL, negatives, duplicates per document, and sequence continuity (starting at 0, no gaps)

# COMMAND ----------

# DBTITLE 1,4. Chunk Integrity
# ============================================================
# 4. Chunk Integrity
# ============================================================

if table_loadable:
    actual_columns = set(df_chunks.columns)

    # --- 4.1 Empty chunks ---
    null_text_count = df_chunks.filter(col("chunk_text").isNull()).count()
    blank_text_count = df_chunks.filter(
        col("chunk_text").isNotNull() & (trim(col("chunk_text")) == "")
    ).count()

    if null_text_count == 0:
        add_result("chunk_text_not_null", "Chunk Integrity", "PASS", "PASS", 0,
                   "No NULL chunk_text values.")
    else:
        add_result("chunk_text_not_null", "Chunk Integrity", "ERROR", "FAIL",
                   null_text_count, f"{null_text_count} chunks have NULL chunk_text.")

    if blank_text_count == 0:
        add_result("chunk_text_not_blank", "Chunk Integrity", "PASS", "PASS", 0,
                   "No blank chunk_text values (TRIM = '').")
    else:
        add_result("chunk_text_not_blank", "Chunk Integrity", "ERROR", "FAIL",
                   blank_text_count, f"{blank_text_count} chunks have blank chunk_text.")

    # --- 4.2 Character count consistency ---
    if "chunk_character_count" in actual_columns and "chunk_text" in actual_columns:
        mismatch_count = df_chunks.filter(
            col("chunk_character_count") != length(col("chunk_text"))
        ).count()
        if mismatch_count == 0:
            add_result("char_count_consistency", "Chunk Integrity", "PASS", "PASS", 0,
                       "chunk_character_count matches LENGTH(chunk_text) for all rows.")
        else:
            add_result("char_count_consistency", "Chunk Integrity", "ERROR", "FAIL",
                       mismatch_count,
                       f"{mismatch_count} chunks have mismatched chunk_character_count.")
    else:
        add_result("char_count_consistency", "Chunk Integrity", "NOT_VALIDATED", "NOT_VALIDATED", 0,
                   "Required columns for this check not present.")

    # --- 4.3 chunk_id uniqueness ---
    dup_ids = df_chunks.groupBy("chunk_id").count().filter("count > 1")
    dup_id_count = dup_ids.count()
    if dup_id_count == 0:
        add_result("chunk_id_uniqueness", "Chunk Integrity", "PASS", "PASS", 0,
                   "All chunk_id values are unique.")
    else:
        add_result("chunk_id_uniqueness", "Chunk Integrity", "ERROR", "FAIL",
                   dup_id_count, f"{dup_id_count} duplicate chunk_id values detected.")

    # --- 4.4 chunk_id determinism ---
    # Pattern from 02_metadata_and_chunking: {document_id}_chunk_{chunk_index:04d}
    expected_id = concat(col("document_id"), lit("_chunk_"),
                         lpad(col("chunk_index").cast("string"), 4, "0"))
    non_det_count = df_chunks.filter(col("chunk_id") != expected_id).count()
    if non_det_count == 0:
        add_result("chunk_id_determinism", "Chunk Integrity", "PASS", "PASS", 0,
                   "All chunk_id values follow pattern {document_id}_chunk_{index:04d}.")
    else:
        add_result("chunk_id_determinism", "Chunk Integrity", "ERROR", "FAIL",
                   non_det_count,
                   f"{non_det_count} chunk_id values do not match the expected deterministic pattern.")

    # --- 4.5 chunk_index NULL ---
    null_idx_count = df_chunks.filter(col("chunk_index").isNull()).count()
    if null_idx_count == 0:
        add_result("chunk_index_not_null", "Chunk Integrity", "PASS", "PASS", 0,
                   "No NULL chunk_index values.")
    else:
        add_result("chunk_index_not_null", "Chunk Integrity", "ERROR", "FAIL",
                   null_idx_count, f"{null_idx_count} NULL chunk_index values.")

    # --- 4.6 chunk_index non-negative ---
    neg_idx_count = df_chunks.filter(col("chunk_index") < 0).count()
    if neg_idx_count == 0:
        add_result("chunk_index_non_negative", "Chunk Integrity", "PASS", "PASS", 0,
                   "No negative chunk_index values.")
    else:
        add_result("chunk_index_non_negative", "Chunk Integrity", "ERROR", "FAIL",
                   neg_idx_count, f"{neg_idx_count} negative chunk_index values.")

    # --- 4.7 (document_id, chunk_index) uniqueness ---
    dup_doc_idx = df_chunks.groupBy("document_id", "chunk_index").count().filter("count > 1")
    dup_doc_idx_count = dup_doc_idx.count()
    if dup_doc_idx_count == 0:
        add_result("doc_chunk_index_unique", "Chunk Integrity", "PASS", "PASS", 0,
                   "All (document_id, chunk_index) pairs are unique.")
    else:
        add_result("doc_chunk_index_unique", "Chunk Integrity", "ERROR", "FAIL",
                   dup_doc_idx_count,
                   f"{dup_doc_idx_count} duplicate (document_id, chunk_index) pairs.")

    # --- 4.8 chunk_index sequence per document ---
    # Upstream uses enumerate(chunks), so indices start at 0.
    # For each doc: min=0, max=n-1, no duplicates => contiguous.
    doc_idx_stats = df_chunks.groupBy("document_id").agg(
        count("*").alias("num_chunks"),
        sql_min("chunk_index").alias("min_idx"),
        sql_max("chunk_index").alias("max_idx"),
    ).collect()

    docs_not_at_zero = []
    docs_with_gaps = []
    for row in doc_idx_stats:
        did = row["document_id"]
        n = row["num_chunks"]
        mn = row["min_idx"]
        mx = row["max_idx"]
        if mn is not None and mn != 0:
            docs_not_at_zero.append(f"{did}: min={mn}")
        if mn is not None and mx is not None and (mx - mn + 1) != n:
            docs_with_gaps.append(f"{did}: min={mn}, max={mx}, n={n}")

    if not docs_not_at_zero and not docs_with_gaps:
        add_result("chunk_index_sequence", "Chunk Integrity", "PASS", "PASS", 0,
                   "All documents have chunk_index starting at 0 with no gaps.")
    else:
        issues = docs_not_at_zero + docs_with_gaps
        add_result("chunk_index_sequence", "Chunk Integrity", "WARNING", "WARNING",
                   len(issues),
                   f"{len(docs_not_at_zero)} docs not starting at 0; "
                   f"{len(docs_with_gaps)} docs with gaps. "
                   f"Details: {'; '.join(issues[:5])}")
else:
    for check in ["chunk_text_not_null", "chunk_text_not_blank",
                   "char_count_consistency", "chunk_id_uniqueness",
                   "chunk_id_determinism", "chunk_index_not_null",
                   "chunk_index_non_negative", "doc_chunk_index_unique",
                   "chunk_index_sequence"]:
        add_result(check, "Chunk Integrity", "NOT_VALIDATED", "NOT_VALIDATED", 0,
                   "Cannot check: table not loadable.")

# COMMAND ----------

# DBTITLE 1,5. Metadata Completeness
# MAGIC %md
# MAGIC ## 5. Metadata Completeness
# MAGIC
# MAGIC Checks NULL rates for all metadata fields. A NULL does not automatically constitute an error — it may reflect a legitimately unknown value in the original metadata CSV. A column entirely absent from the schema, however, is reported as an ERROR.

# COMMAND ----------

# DBTITLE 1,5. Metadata Completeness
# ============================================================
# 5. Metadata Completeness
# ============================================================

if table_loadable:
    actual_columns = set(df_chunks.columns)
    print("=== Metadata Completeness ===")

    for field in METADATA_COLUMNS:
        if field not in actual_columns:
            add_result(f"metadata_present_{field}", "Metadata Completeness",
                       "ERROR", "FAIL", 0,
                       f"Column {field} not present in schema.")
            continue

        null_count = df_chunks.filter(col(field).isNull()).count()
        non_null_count = total_chunks - null_count

        if null_count == 0:
            add_result(f"metadata_present_{field}", "Metadata Completeness",
                       "PASS", "PASS", 0,
                       f"{field}: all {total_chunks} chunks populated.")
        elif null_count == total_chunks:
            add_result(f"metadata_present_{field}", "Metadata Completeness",
                       "WARNING", "WARNING", null_count,
                       f"{field}: all {null_count} chunks are NULL "
                       f"(field entirely unpopulated).")
        else:
            add_result(f"metadata_present_{field}", "Metadata Completeness",
                       "WARNING", "WARNING", null_count,
                       f"{field}: {null_count}/{total_chunks} chunks are NULL.")

        print(f"  {field:20s}: {non_null_count}/{total_chunks} populated, {null_count} null")
else:
    for field in METADATA_COLUMNS:
        add_result(f"metadata_present_{field}", "Metadata Completeness",
                   "NOT_VALIDATED", "NOT_VALIDATED", 0,
                   "Cannot check: table not loadable.")

# COMMAND ----------

# DBTITLE 1,6. Document-level Validation
# MAGIC %md
# MAGIC ## 6. Document-level Validation
# MAGIC
# MAGIC Generates per-document statistics (chunk count, min / max / average chunk length) and detects outliers:
# MAGIC - Documents with a single chunk (WARNING — limited retrieval granularity)
# MAGIC - Documents present in bronze but absent from chunks (WARNING — likely non-SUCCESS extraction)
# MAGIC - Unusually high or low chunk counts (reported as diagnostics, not errors)

# COMMAND ----------

# DBTITLE 1,6. Document-level Validation
# ============================================================
# 6. Document-level Validation
# ============================================================

if table_loadable:
    print("=== Document-level Validation ===")

    # Per-document statistics
    doc_stats = spark.sql(f"""
    SELECT
        document_id,
        pdf_filename,
        title,
        category,
        COUNT(*) AS number_of_chunks,
        MIN(chunk_character_count) AS min_chunk_length,
        MAX(chunk_character_count) AS max_chunk_length,
        ROUND(AVG(chunk_character_count), 1) AS avg_chunk_length
    FROM {FULL_CHUNKS_NAME}
    GROUP BY document_id, pdf_filename, title, category
    ORDER BY document_id
    """)

    doc_stats.show(truncate=False)

    doc_stats_rows = doc_stats.collect()

    # Documents with single chunk
    single_chunk_docs = [r for r in doc_stats_rows if r["number_of_chunks"] == 1]
    if not single_chunk_docs:
        add_result("single_chunk_documents", "Document-level", "PASS", "PASS", 0,
                   "No documents with a single chunk.")
    else:
        add_result("single_chunk_documents", "Document-level", "WARNING", "WARNING",
                   len(single_chunk_docs),
                   f"{len(single_chunk_docs)} document(s) have only 1 chunk. "
                   f"These may have limited retrieval granularity.")

    # Documents in bronze but not in chunks
    bronze_available = False
    try:
        spark.table(FULL_BRONZE_NAME)
        bronze_available = True
    except Exception:
        pass

    if bronze_available:
        bronze_ids = {r.document_id for r in spark.sql(
            f"SELECT DISTINCT document_id FROM {FULL_BRONZE_NAME}"
        ).collect()}
        chunk_ids_set = {r["document_id"] for r in doc_stats_rows}
        docs_without_chunks = bronze_ids - chunk_ids_set

        if not docs_without_chunks:
            add_result("docs_without_chunks", "Document-level", "PASS", "PASS", 0,
                       f"All {len(bronze_ids)} bronze documents have chunks in civ4_chunks.")
        else:
            add_result("docs_without_chunks", "Document-level", "WARNING", "WARNING",
                       len(docs_without_chunks),
                       f"{len(docs_without_chunks)} document(s) in bronze but not in chunks. "
                       f"These may have extraction_status != SUCCESS.")
            for did in sorted(docs_without_chunks):
                print(f"  {did}")

        print(f"\nBronze documents: {len(bronze_ids)}")
        print(f"Chunks documents: {len(chunk_ids_set)}")
    else:
        add_result("docs_without_chunks", "Document-level", "NOT_VALIDATED", "NOT_VALIDATED", 0,
                   "Bronze table not available for cross-checking.")

    # Chunk count distribution summary
    chunk_counts = [r["number_of_chunks"] for r in doc_stats_rows]
    if chunk_counts:
        avg_chunks = sum(chunk_counts) / len(chunk_counts)
        print(f"\nChunks per document: min={min(chunk_counts)}, "
              f"max={max(chunk_counts)}, avg={avg_chunks:.1f}")
else:
    add_result("single_chunk_documents", "Document-level",
               "NOT_VALIDATED", "NOT_VALIDATED", 0,
               "Cannot check: table not loadable.")
    add_result("docs_without_chunks", "Document-level",
               "NOT_VALIDATED", "NOT_VALIDATED", 0,
               "Cannot check: table not loadable.")

# COMMAND ----------

# DBTITLE 1,7. Metadata Consistency
# MAGIC %md
# MAGIC ## 7. Metadata Consistency within Documents
# MAGIC
# MAGIC For each `document_id`, verifies that metadata fields (title, author, category, etc.) have only one distinct value across all its chunks. Multiple distinct values for a field within the same document indicate a problem during metadata enrichment. This check treats NULL as a distinct value, so a mix of NULL and non-NULL for the same field is also flagged.

# COMMAND ----------

# DBTITLE 1,7. Metadata Consistency
# ============================================================
# 7. Metadata Consistency within Documents
# ============================================================

if table_loadable:
    print("=== Metadata Consistency within Documents ===")
    actual_columns = set(df_chunks.columns)

    for field in METADATA_COLUMNS:
        if field not in actual_columns:
            add_result(f"metadata_consistency_{field}", "Metadata Consistency",
                       "NOT_VALIDATED", "NOT_VALIDATED", 0,
                       f"Column {field} not present in schema.")
            continue

        # Count distinct values per document_id, treating NULL as a distinct value
        inconsistent = df_chunks.groupBy("document_id").agg(
            countDistinct(
                when(col(field).isNull(), lit("__NULL__")).otherwise(col(field))
            ).alias("distinct_vals")
        ).filter("distinct_vals > 1")

        inconsistent_count = inconsistent.count()
        if inconsistent_count == 0:
            add_result(f"metadata_consistency_{field}", "Metadata Consistency",
                       "PASS", "PASS", 0,
                       f"{field} is consistent within each document.")
        else:
            add_result(f"metadata_consistency_{field}", "Metadata Consistency",
                       "ERROR", "FAIL", inconsistent_count,
                       f"{inconsistent_count} document(s) have inconsistent "
                       f"{field} across chunks.")
            print(f"  Inconsistent {field}:")
            inconsistent.select("document_id", "distinct_vals").show(truncate=False)
else:
    for field in METADATA_COLUMNS:
        add_result(f"metadata_consistency_{field}", "Metadata Consistency",
                   "NOT_VALIDATED", "NOT_VALIDATED", 0,
                   "Cannot check: table not loadable.")

# COMMAND ----------

# DBTITLE 1,8. Traceability Checks
# MAGIC %md
# MAGIC ## 8. Traceability Checks
# MAGIC
# MAGIC Validates the chain: **PDF → pdf_filename → document_id → chunk_id → chunk_text**.
# MAGIC - `pdf_filename` is present on every chunk
# MAGIC - `document_id` ↔ `pdf_filename` is a 1:1 mapping (no document split across PDFs or vice versa)
# MAGIC - If `civ4_pdf_bronze` is available, cross-checks document coverage between bronze and chunks

# COMMAND ----------

# DBTITLE 1,8. Traceability Checks
# ============================================================
# 8. Traceability Checks
# ============================================================

if table_loadable:
    print("=== Traceability Checks ===")
    actual_columns = set(df_chunks.columns)

    # --- pdf_filename present ---
    if "pdf_filename" in actual_columns:
        null_pdf = df_chunks.filter(col("pdf_filename").isNull()).count()
        if null_pdf == 0:
            add_result("pdf_filename_present", "Traceability", "PASS", "PASS", 0,
                       "All chunks have pdf_filename.")
        else:
            add_result("pdf_filename_present", "Traceability", "ERROR", "FAIL",
                       null_pdf, f"{null_pdf} chunks have NULL pdf_filename.")
    else:
        add_result("pdf_filename_present", "Traceability", "ERROR", "FAIL", 0,
                   "Column pdf_filename not present in schema.")

    # --- 1:1 document_id <-> pdf_filename ---
    if "document_id" in actual_columns and "pdf_filename" in actual_columns:
        doc_pdf = df_chunks.select("document_id", "pdf_filename").distinct()

        # Multiple pdf_filename per document_id
        multi_pdf = doc_pdf.groupBy("document_id").count().filter("count > 1")
        multi_pdf_count = multi_pdf.count()
        if multi_pdf_count == 0:
            add_result("doc_to_pdf_1to1", "Traceability", "PASS", "PASS", 0,
                       "Each document_id maps to exactly one pdf_filename.")
        else:
            add_result("doc_to_pdf_1to1", "Traceability", "ERROR", "FAIL",
                       multi_pdf_count,
                       f"{multi_pdf_count} document_id(s) map to multiple "
                       f"pdf_filename values.")

        # Multiple document_id per pdf_filename
        multi_doc = doc_pdf.groupBy("pdf_filename").count().filter("count > 1")
        multi_doc_count = multi_doc.count()
        if multi_doc_count == 0:
            add_result("pdf_to_doc_1to1", "Traceability", "PASS", "PASS", 0,
                       "Each pdf_filename maps to exactly one document_id.")
        else:
            add_result("pdf_to_doc_1to1", "Traceability", "ERROR", "FAIL",
                       multi_doc_count,
                       f"{multi_doc_count} pdf_filename(s) map to multiple "
                       f"document_id values.")
    else:
        add_result("doc_to_pdf_1to1", "Traceability", "NOT_VALIDATED", "NOT_VALIDATED", 0,
                   "Required columns not present.")
        add_result("pdf_to_doc_1to1", "Traceability", "NOT_VALIDATED", "NOT_VALIDATED", 0,
                   "Required columns not present.")

    # --- Cross-check with bronze if available ---
    if bronze_available:
        bronze_ids = {r.document_id for r in spark.sql(
            f"SELECT DISTINCT document_id FROM {FULL_BRONZE_NAME}"
        ).collect()}
        chunk_ids_set = {r.document_id for r in df_chunks.select("document_id").distinct().collect()}

        in_bronze_only = bronze_ids - chunk_ids_set
        in_chunks_only = chunk_ids_set - bronze_ids

        if not in_bronze_only and not in_chunks_only:
            add_result("bronze_coverage", "Traceability", "PASS", "PASS", 0,
                       f"All {len(bronze_ids)} bronze documents are represented "
                       f"in civ4_chunks.")
        else:
            parts = []
            if in_bronze_only:
                parts.append(f"{len(in_bronze_only)} in bronze but not in chunks")
            if in_chunks_only:
                parts.append(f"{len(in_chunks_only)} in chunks but not in bronze")
            add_result("bronze_coverage", "Traceability", "WARNING", "WARNING",
                       len(in_bronze_only) + len(in_chunks_only),
                       "; ".join(parts))

        print(f"\nBronze documents: {len(bronze_ids)}")
        print(f"Chunks documents: {len(chunk_ids_set)}")
        if in_bronze_only:
            print(f"  In bronze but not in chunks: {len(in_bronze_only)}")
            for did in sorted(in_bronze_only):
                print(f"    {did}")
    else:
        add_result("bronze_coverage", "Traceability", "NOT_VALIDATED", "NOT_VALIDATED", 0,
                   f"Bronze table {FULL_BRONZE_NAME} not available for cross-checking.")
        print("\nBronze table not available for cross-checking.")
else:
    for check in ["pdf_filename_present", "doc_to_pdf_1to1",
                   "pdf_to_doc_1to1", "bronze_coverage"]:
        add_result(check, "Traceability", "NOT_VALIDATED", "NOT_VALIDATED", 0,
                   "Cannot check: table not loadable.")

# COMMAND ----------

# DBTITLE 1,9. Duplicate Detection
# MAGIC %md
# MAGIC ## 9. Duplicate Detection
# MAGIC
# MAGIC Detects and classifies duplicates by severity:
# MAGIC - **ERROR**: duplicate `chunk_id` — violates uniqueness
# MAGIC - **ERROR**: duplicate `(document_id, chunk_index)` — violates uniqueness
# MAGIC - **ERROR**: same `pdf_filename` with different `document_id` — duplicate document
# MAGIC - **WARNING**: identical `chunk_text` within the same document — may be a natural consequence of overlap

# COMMAND ----------

# DBTITLE 1,9. Duplicate Detection
# ============================================================
# 9. Duplicate Detection
# ============================================================

if table_loadable:
    print("=== Duplicate Detection ===")

    # --- chunk_id duplicates ---
    dup_ids = df_chunks.groupBy("chunk_id").count().filter("count > 1")
    dup_id_count = dup_ids.count()
    if dup_id_count == 0:
        add_result("dup_chunk_id", "Duplicate Detection", "PASS", "PASS", 0,
                   "No duplicate chunk_id values.")
    else:
        add_result("dup_chunk_id", "Duplicate Detection", "ERROR", "FAIL",
                   dup_id_count, f"{dup_id_count} duplicate chunk_id values.")

    # --- (document_id, chunk_index) duplicates ---
    dup_doc_idx = df_chunks.groupBy("document_id", "chunk_index").count().filter("count > 1")
    dup_doc_idx_count = dup_doc_idx.count()
    if dup_doc_idx_count == 0:
        add_result("dup_doc_chunk_index", "Duplicate Detection", "PASS", "PASS", 0,
                   "No duplicate (document_id, chunk_index) pairs.")
    else:
        add_result("dup_doc_chunk_index", "Duplicate Detection", "ERROR", "FAIL",
                   dup_doc_idx_count,
                   f"{dup_doc_idx_count} duplicate (document_id, chunk_index) pairs.")

    # --- Exact chunk_text duplicates within same document ---
    text_dups = df_chunks.groupBy("document_id", "chunk_text").agg(
        count(lit(1)).alias("cnt")
    ).filter("cnt > 1")

    text_dup_count = text_dups.count()
    if text_dup_count == 0:
        add_result("dup_text_within_doc", "Duplicate Detection", "PASS", "PASS", 0,
                   "No exact text duplicates within the same document.")
    else:
        add_result("dup_text_within_doc", "Duplicate Detection", "WARNING", "WARNING",
                   text_dup_count,
                   f"{text_dup_count} case(s) of identical chunk_text within the "
                   f"same document (may be caused by overlap).")

    # --- Duplicate documents (same pdf_filename, different document_id) ---
    doc_pdf = df_chunks.select("document_id", "pdf_filename").distinct()
    dup_pdfs = doc_pdf.groupBy("pdf_filename").count().filter("count > 1")
    dup_pdf_count = dup_pdfs.count()
    if dup_pdf_count == 0:
        add_result("dup_documents", "Duplicate Detection", "PASS", "PASS", 0,
                   "No duplicate documents (same pdf_filename with different document_id).")
    else:
        add_result("dup_documents", "Duplicate Detection", "ERROR", "FAIL",
                   dup_pdf_count,
                   f"{dup_pdf_count} pdf_filename(s) associated with multiple "
                   f"document_id(s).")
else:
    for check in ["dup_chunk_id", "dup_doc_chunk_index",
                   "dup_text_within_doc", "dup_documents"]:
        add_result(check, "Duplicate Detection", "NOT_VALIDATED", "NOT_VALIDATED", 0,
                   "Cannot check: table not loadable.")

# COMMAND ----------

# DBTITLE 1,10. Idempotency / Reproducibility Checks
# MAGIC %md
# MAGIC ## 10. Idempotency / Reproducibility Checks
# MAGIC
# MAGIC Idempotency means re-running `02_metadata_and_chunking` with the same input produces identical output. This notebook can verify **observable preconditions** for idempotency without re-running the upstream notebook:
# MAGIC - `chunk_id` is deterministic (follows `{document_id}_chunk_{index:04d}` pattern)
# MAGIC - `chunk_id` is unique
# MAGIC - `(document_id, chunk_index)` is unique
# MAGIC
# MAGIC If all preconditions pass, the status is **PARTIALLY VALIDATED** — full proof requires re-running the upstream notebook.

# COMMAND ----------

# DBTITLE 1,10. Idempotency / Reproducibility Checks
# ============================================================
# 10. Idempotency / Reproducibility Checks
# ============================================================

if table_loadable:
    print("=== Idempotency / Reproducibility ===")

    # Gather results from idempotency-relevant checks already computed
    idempotency_checks = [
        r for r in validation_results
        if r["check_name"] in ("chunk_id_determinism",
                               "chunk_id_uniqueness",
                               "doc_chunk_index_unique")
    ]

    all_pass = all(r["status"] == "PASS" for r in idempotency_checks)
    any_not_validated = any(r["status"] == "NOT_VALIDATED" for r in idempotency_checks)

    if all_pass:
        add_result("idempotency_properties", "Idempotency", "PASS", "PASS", 0,
                   "Observable idempotency properties verified: deterministic chunk_ids, "
                   "unique chunk_ids, unique (document_id, chunk_index). "
                   "Full idempotency requires re-running the upstream notebook.")
    elif any_not_validated:
        add_result("idempotency_properties", "Idempotency", "WARNING", "WARNING", 0,
                   "Some idempotency properties could not be validated. "
                   "Partial validation only.")
    else:
        add_result("idempotency_properties", "Idempotency", "ERROR", "FAIL", 0,
                   "Idempotency properties not satisfied. See chunk_id_determinism, "
                   "chunk_id_uniqueness, doc_chunk_index_unique.")

    print("Properties verified:")
    print("  1. chunk_id deterministic pattern ({document_id}_chunk_{index:04d})")
    print("  2. chunk_id uniqueness")
    print("  3. (document_id, chunk_index) uniqueness")
    print()
    print("Note: Full idempotency (re-running produces identical output)")
    print("cannot be demonstrated without re-running 02_metadata_and_chunking.")
    print("The properties above are necessary preconditions for idempotency.")
    print("Status: PARTIALLY VALIDATED")
else:
    add_result("idempotency_properties", "Idempotency",
               "NOT_VALIDATED", "NOT_VALIDATED", 0,
               "Cannot check: table not loadable.")

# COMMAND ----------

# DBTITLE 1,11. Chunk Size Diagnostics
# MAGIC %md
# MAGIC ## 11. Chunk Size Diagnostics
# MAGIC
# MAGIC Reports distribution statistics for `chunk_character_count`: min, max, average, median, and percentiles. The upstream configuration used `CHUNK_SIZE=1000` and `CHUNK_OVERLAP=150` — these are configuration values, not strict requirements. Chunks at the beginning or end of a document may naturally be smaller. No pass/fail is assigned; this section is purely diagnostic.

# COMMAND ----------

# DBTITLE 1,11. Chunk Size Diagnostics
# ============================================================
# 11. Chunk Size Diagnostics
# ============================================================

if table_loadable:
    print("=== Chunk Size Diagnostics ===")
    print("(Reference: upstream chunk_size=1000, chunk_overlap=150)")
    print("(These are configuration values, not strict requirements for every chunk.)")
    print()

    # Basic statistics with percentiles
    spark.sql(f"""
    SELECT
        COUNT(*) AS total_chunks,
        MIN(chunk_character_count) AS min_len,
        MAX(chunk_character_count) AS max_len,
        ROUND(AVG(chunk_character_count), 1) AS avg_len,
        PERCENTILE(chunk_character_count, 0.5) AS median,
        PERCENTILE(chunk_character_count, 0.25) AS p25,
        PERCENTILE(chunk_character_count, 0.75) AS p75,
        PERCENTILE(chunk_character_count, 0.90) AS p90,
        PERCENTILE(chunk_character_count, 0.95) AS p95,
        PERCENTILE(chunk_character_count, 0.99) AS p99
    FROM {FULL_CHUNKS_NAME}
    """).show(truncate=False)

    # Distribution by bucket
    print("=== Chunk length distribution ===")
    spark.sql(f"""
    SELECT
        CASE
            WHEN chunk_character_count < 100   THEN '000-099'
            WHEN chunk_character_count < 500   THEN '100-499'
            WHEN chunk_character_count < 800   THEN '500-799'
            WHEN chunk_character_count < 1000  THEN '800-999'
            WHEN chunk_character_count < 1200  THEN '1000-1199'
            WHEN chunk_character_count >= 1200 THEN '1200+'
        END AS length_bucket,
        COUNT(*) AS chunk_count
    FROM {FULL_CHUNKS_NAME}
    GROUP BY length_bucket
    ORDER BY length_bucket
    """).show(truncate=False)
else:
    print("Cannot compute: table not loadable.")

# COMMAND ----------

# DBTITLE 1,12. Validation Results
# MAGIC %md
# MAGIC ## 12. Validation Results
# MAGIC
# MAGIC Aggregates all check results into a single DataFrame, sorted by severity (errors first, then warnings, then pass).

# COMMAND ----------

# DBTITLE 1,12. Validation Results
# ============================================================
# 12. Validation Results
# ============================================================

from pyspark.sql.types import (
    StructType, StructField, StringType, LongType,
)

results_schema = StructType([
    StructField("check_name", StringType(), True),
    StructField("check_category", StringType(), True),
    StructField("severity", StringType(), True),
    StructField("status", StringType(), True),
    StructField("affected_rows", LongType(), True),
    StructField("description", StringType(), True),
])

df_results = spark.createDataFrame(validation_results, schema=results_schema)

# Severity ordering for display
severity_order = (
    when(col("severity") == "ERROR", 0)
    .when(col("severity") == "WARNING", 1)
    .when(col("severity") == "PASS", 2)
    .otherwise(3)
)

print("=== Validation Results by Severity ===")
df_results.groupBy("severity").count().orderBy(severity_order).show(truncate=False)

print("\n=== All Validation Results ===")
df_results.orderBy(severity_order, "check_category", "check_name").show(100, truncate=80)

# COMMAND ----------

# DBTITLE 1,13. Overall Readiness
# MAGIC %md
# MAGIC ## 13. Overall Readiness
# MAGIC
# MAGIC Calculates the final readiness status based on the presence of ERROR-level findings:
# MAGIC - **READY_FOR_SPRINT_3** — no errors (warnings alone do not block)
# MAGIC - **NOT_READY_FOR_SPRINT_3** — at least one ERROR
# MAGIC
# MAGIC The status is derived exclusively from the checks executed above. No values are hardcoded.

# COMMAND ----------

# DBTITLE 1,13. Overall Readiness
# ============================================================
# 13. Overall Readiness
# ============================================================

error_count = sum(1 for r in validation_results if r["severity"] == "ERROR")
warning_count = sum(1 for r in validation_results if r["severity"] == "WARNING")
pass_count = sum(1 for r in validation_results if r["severity"] == "PASS")
not_validated_count = sum(1 for r in validation_results if r["severity"] == "NOT_VALIDATED")

overall_status = "READY_FOR_SPRINT_3" if error_count == 0 else "NOT_READY_FOR_SPRINT_3"

print("=" * 50)
print("CIV4 RAG — SPRINT 2 VALIDATION")
print("=" * 50)
print(f"Input table:")
print(f"  {FULL_CHUNKS_NAME}")
print(f"Documents: {total_documents}")
print(f"Chunks:    {total_chunks}")
print(f"PASS:      {pass_count}")
print(f"WARNING:   {warning_count}")
print(f"ERROR:     {error_count}")
if not_validated_count > 0:
    print(f"NOT_VALIDATED: {not_validated_count}")
print()
print(f"Overall status:")
print(f"  {overall_status}")
print("=" * 50)

# Show errors first
if error_count > 0:
    print("\n=== FAILED Checks (ERROR) ===")
    for r in validation_results:
        if r["severity"] == "ERROR":
            print(f"  [{r['check_category']}] {r['check_name']}")
            print(f"    {r['description']}")
            print(f"    Affected rows: {r['affected_rows']}")
            print()

# Then warnings
if warning_count > 0:
    print("=== Warnings ===")
    for r in validation_results:
        if r["severity"] == "WARNING":
            print(f"  [{r['check_category']}] {r['check_name']}: {r['description']}")

# COMMAND ----------

# DBTITLE 1,14. Summary / Next Step
# MAGIC %md
# MAGIC ## 14. Summary / Next Step
# MAGIC
# MAGIC ### Validation Summary
# MAGIC
# MAGIC This notebook validated the `civ4_chunks` Delta table across five dimensions:
# MAGIC 1. **Table integrity** — schema, columns, data types, NOT NULL constraints
# MAGIC 2. **Chunk integrity** — empty chunks, character count consistency, chunk_id determinism, chunk_index sequence
# MAGIC 3. **Metadata completeness** — NULL rates for all metadata fields
# MAGIC 4. **Traceability** — document_id ↔ pdf_filename 1:1 mapping, bronze cross-check
# MAGIC 5. **Idempotency** — deterministic chunk_ids, uniqueness of (document_id, chunk_index)
# MAGIC
# MAGIC The overall status is determined solely by the presence of ERROR-level findings. WARNINGs do not block Sprint 3.
# MAGIC
# MAGIC ### Next Step
# MAGIC
# MAGIC The `civ4_chunks` table has been validated for use in **Sprint 3 — Retrieval / Search**.
# MAGIC
# MAGIC In Sprint 3, `chunk_text` will be used to:
# MAGIC 1. Generate embeddings
# MAGIC 2. Create / configure the Vector Search / AI Search index
# MAGIC 3. Implement retrieval
# MAGIC
# MAGIC > **Note:** No retrieval, embeddings, or Vector Search functionality is implemented in this notebook. Corrections to data issues (if any) should be made in `02_processing/02_metadata_and_chunking`, not here.