# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 02_processing / 01_extract_and_clean
# MAGIC
# MAGIC **Objective:** Discover PDFs of Civilization IV strategy guides, extract their text, apply minimal cleaning, associate each PDF with its metadata, and produce a Bronze Delta table (`civ4_pdf_bronze`) that is reproducible and traceable.
# MAGIC
# MAGIC **Input:**
# MAGIC - PDF files: `/Volumes/workspace/gcivilization/civ4_rag/pdfs/`
# MAGIC - Metadata CSV: `/Volumes/workspace/gcivilization/civ4_rag/civ4_strategy_metadata.csv`
# MAGIC
# MAGIC **Output:**
# MAGIC - Delta table: `workspace.gcivilization.civ4_pdf_bronze`
# MAGIC
# MAGIC **Key principles:**
# MAGIC - Conservative text cleaning — preserve all information for retrieval.
# MAGIC - Explicit extraction status for every PDF (SUCCESS / EMPTY / SUSPICIOUS / FAILED).
# MAGIC - `document_id` from the CSV is the canonical identifier.
# MAGIC - Idempotent via MERGE on `document_id`.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC This notebook is part of the **Civ4 RAG** project (Sprint 2 — Processing).
# MAGIC It does **not** perform embeddings, vector search, retrieval, or generation.

# COMMAND ----------

# DBTITLE 1,1. Configuration
# MAGIC %md
# MAGIC ## 1. Configuration
# MAGIC
# MAGIC Centralizes all path and parameter definitions. Modify these variables if the workspace layout changes.
# MAGIC
# MAGIC **Dependency note:** `pypdf` is a pure-Python library (no system-level requirements) used for PDF text extraction. It is not pre-installed in Databricks Free Edition but can be installed via `pip`. The flag `INSTALL_PYPDF` controls automatic installation.

# COMMAND ----------

# DBTITLE 1,1. Configuration
# ============================================================
# Configuration
# ============================================================
# All paths and table names are defined here as variables.
# Modify these if the workspace layout changes.

# --- Unity Catalog destination ---
CATALOG = "workspace"
SCHEMA = "gcivilization"
BRONZE_TABLE = "civ4_pdf_bronze"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{BRONZE_TABLE}"

# --- Source paths (Unity Catalog Volumes) ---
PDF_DIR = "/Volumes/workspace/gcivilization/civ4_rag/pdfs/"
METADATA_CSV_PATH = "/Volumes/workspace/gcivilization/civ4_rag/civ4_strategy_metadata.csv"

# --- Extraction parameters ---
MIN_CHAR_THRESHOLD = 100    # Texts shorter than this are flagged as SUSPICIOUS
INSTALL_PYPDF = True         # Attempt pip install if pypdf is not available

print(f"Bronze table target : {FULL_TABLE_NAME}")
print(f"PDF source directory: {PDF_DIR}")
print(f"Metadata CSV path   : {METADATA_CSV_PATH}")
print(f"Min char threshold  : {MIN_CHAR_THRESHOLD}")

# COMMAND ----------

# DBTITLE 1,2. Load metadata
# MAGIC %md
# MAGIC ## 2. Load metadata
# MAGIC
# MAGIC Reads the CSV into a Spark DataFrame and checks for duplicate `document_id` and `pdf_filename` values.

# COMMAND ----------

# DBTITLE 1,2. Load metadata
# ============================================================
# Load metadata CSV
# ============================================================
# Reads the metadata CSV into a Spark DataFrame and performs
# basic quality checks (duplicate document_id / pdf_filename).

df_metadata = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(METADATA_CSV_PATH)
)

metadata_count = df_metadata.count()
print(f"Metadata records loaded: {metadata_count}")

# --- DQ: duplicate document_id ---
dup_doc_ids = (
    df_metadata.groupBy("document_id").count().filter("count > 1")
)
dup_doc_count = dup_doc_ids.count()
if dup_doc_count > 0:
    print(f"WARNING: {dup_doc_count} duplicate document_id(s) found:")
    dup_doc_ids.show(truncate=False)
else:
    print("No duplicate document_id values.")

# --- DQ: duplicate pdf_filename ---
dup_pdfs = (
    df_metadata.groupBy("pdf_filename").count().filter("count > 1")
)
dup_pdf_count = dup_pdfs.count()
if dup_pdf_count > 0:
    print(f"WARNING: {dup_pdf_count} duplicate pdf_filename(s) found:")
    dup_pdfs.show(truncate=False)
else:
    print("No duplicate pdf_filename values.")

print("\nMetadata schema:")
df_metadata.printSchema()

# COMMAND ----------

# DBTITLE 1,3. Discover PDFs
# MAGIC %md
# MAGIC ## 3. Discover PDFs
# MAGIC
# MAGIC Lists every `.pdf` file inside the configured volume directory using the local filesystem path. UC Volumes are accessible at `/Volumes/...` from compute nodes.

# COMMAND ----------

# DBTITLE 1,3. Discover PDFs
# ============================================================
# Discover PDFs
# ============================================================
# Lists every .pdf file inside the configured volume directory.
# Uses the local filesystem path (Databricks mounts UC volumes
# at /Volumes/...).

import os

pdf_files_on_disk = sorted([
    f for f in os.listdir(PDF_DIR)
    if f.lower().endswith(".pdf")
])

pdf_count = len(pdf_files_on_disk)
print(f"PDFs discovered: {pdf_count}")
for f in pdf_files_on_disk:
    filepath = os.path.join(PDF_DIR, f)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  {f}  ({size_mb:.2f} MB)")

# COMMAND ----------

# DBTITLE 1,4. Match PDFs with metadata
# MAGIC %md
# MAGIC ## 4. Match PDFs with metadata
# MAGIC
# MAGIC Joins the discovered PDFs with metadata records on `pdf_filename`. Identifies:
# MAGIC - PDFs without a matching metadata entry.
# MAGIC - Metadata entries without a corresponding PDF on disk.

# COMMAND ----------

# DBTITLE 1,4. Match PDFs with metadata
# ============================================================
# Match PDFs with metadata
# ============================================================
# Builds a unified list: every PDF on disk is checked against
# the metadata DataFrame. Records without matching metadata
# are still included (metadata_match_status = "NO_METADATA").

# Collect metadata pdf_filenames for set operations
metadata_pdfs = {
    row.pdf_filename for row in df_metadata.select("pdf_filename").collect()
}
disk_pdfs = set(pdf_files_on_disk)

pdfs_with_metadata = metadata_pdfs & disk_pdfs
pdfs_without_metadata = disk_pdfs - metadata_pdfs
metadata_without_pdf = metadata_pdfs - disk_pdfs

print(f"PDFs with metadata match : {len(pdfs_with_metadata)}")
print(f"PDFs without metadata    : {len(pdfs_without_metadata)}")
print(f"Metadata without PDF     : {len(metadata_without_pdf)}")

if pdfs_without_metadata:
    print("\nPDFs without metadata:")
    for f in sorted(pdfs_without_metadata):
        print(f"  {f}")

if metadata_without_pdf:
    print("\nMetadata entries without corresponding PDF:")
    for f in sorted(metadata_without_pdf):
        print(f"  {f}")

# Build a lookup dict for fast access during extraction
metadata_lookup = {
    row.pdf_filename: {
        "document_id": row.document_id,
        "title": row.title,
        "author": row.author,
        "category": row.category,
        "game_version": row.game_version,
        "expansion": row.expansion,
        "source_url": row.source_url,
        "publication_date": row.publication_date,
    }
    for row in df_metadata.collect()
}

# COMMAND ----------

# DBTITLE 1,5. Extract text
# MAGIC %md
# MAGIC ## 5. Extract text
# MAGIC
# MAGIC Uses `pypdf` (pure-Python, installable via pip) to extract text from each PDF.
# MAGIC
# MAGIC - Captures exceptions without stopping the pipeline.
# MAGIC - Detects image-based PDFs (no text layer, only embedded images) and notes them in `extraction_error`.
# MAGIC - Records `page_count` and `character_count` for every document.
# MAGIC
# MAGIC **Status values:**
# MAGIC | Status | Meaning |
# MAGIC |---|---|
# MAGIC | `SUCCESS` | Text extracted, length >= `MIN_CHAR_THRESHOLD` |
# MAGIC | `EMPTY` | Text extracted but length == 0 (image-based PDF?) |
# MAGIC | `SUSPICIOUS` | Text extracted but length < `MIN_CHAR_THRESHOLD` |
# MAGIC | `FAILED` | Exception during extraction |

# COMMAND ----------

# DBTITLE 1,5. Extract text
# ============================================================
# Extract text from PDFs
# ============================================================
# Dependency: pypdf (pure-Python, no system dependencies).
# Not pre-installed in Databricks Free Edition.
# If INSTALL_PYPDF is True, attempts pip install automatically.
# If INSTALL_PYPDF is False and pypdf is missing, raises ImportError.

import io
import re
import os
import subprocess
import importlib
from datetime import datetime, timezone

# --- Ensure pypdf is available ---
try:
    importlib.import_module("pypdf")
except ImportError:
    if INSTALL_PYPDF:
        print("pypdf not found — installing via pip...")
        subprocess.run(["pip", "install", "pypdf", "--quiet"], check=True)
        print("pypdf installed successfully.")
    else:
        raise ImportError(
            "pypdf is required for PDF text extraction but is not installed. "
            "Set INSTALL_PYPDF = True or run: pip install pypdf"
        )

import pypdf

# --- Extraction function ---
def extract_pdf(filepath):
    """
    Extract text from a PDF file using pypdf.
    Returns dict with: raw_text, page_count, character_count,
    extraction_status, extraction_error, has_images.
    """
    result = {
        "raw_text": "",
        "page_count": 0,
        "character_count": 0,
        "extraction_status": "FAILED",
        "extraction_error": None,
        "has_images": False,
    }
    try:
        with open(filepath, "rb") as f:
            pdf_bytes = f.read()
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        result["page_count"] = len(reader.pages)

        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            # Detect image-based pages (no text layer)
            if "/Resources" in page:
                resources = page["/Resources"]
                if "/XObject" in resources:
                    xobjects = resources["/XObject"]
                    for _name, obj in xobjects.items():
                        resolved = obj.get_object() if hasattr(obj, "get_object") else obj
                        if resolved.get("/Subtype") == "/Image":
                            result["has_images"] = True
                            break

        raw_text = "\n".join(text_parts).strip()
        result["raw_text"] = raw_text
        result["character_count"] = len(raw_text)

        if result["character_count"] == 0:
            result["extraction_status"] = "EMPTY"
            if result["has_images"]:
                result["extraction_error"] = (
                    "No text layer detected; PDF contains images. "
                    "OCR may be required for full extraction."
                )
            else:
                result["extraction_error"] = (
                    "No text extracted and no images detected."
                )
        elif result["character_count"] < MIN_CHAR_THRESHOLD:
            result["extraction_status"] = "SUSPICIOUS"
            result["extraction_error"] = (
                f"Extracted text is very short ({result['character_count']} chars), "
                f"below threshold of {MIN_CHAR_THRESHOLD}."
            )
        else:
            result["extraction_status"] = "SUCCESS"

    except Exception as e:
        result["extraction_status"] = "FAILED"
        result["extraction_error"] = f"{type(e).__name__}: {str(e)}"

    return result

# --- Process all PDFs ---
extraction_records = []
for pdf_filename in pdf_files_on_disk:
    filepath = os.path.join(PDF_DIR, pdf_filename)
    meta = metadata_lookup.get(pdf_filename, {})

    ext = extract_pdf(filepath)

    record = {
        "document_id": meta.get("document_id", pdf_filename.replace(".pdf", "")),
        "pdf_filename": pdf_filename,
        "file_path": filepath,
        "raw_text": ext["raw_text"],
        "ingestion_timestamp": datetime.now(timezone.utc),
        "extraction_status": ext["extraction_status"],
        "page_count": ext["page_count"],
        "character_count": ext["character_count"],
        "extraction_error": ext["extraction_error"],
        "metadata_match_status": "MATCHED" if pdf_filename in metadata_pdfs else "NO_METADATA",
        # Carry metadata forward for traceability
        "title": meta.get("title"),
        "author": meta.get("author"),
        "category": meta.get("category"),
        "game_version": meta.get("game_version"),
        "expansion": meta.get("expansion"),
        "source_url": meta.get("source_url"),
        "publication_date": meta.get("publication_date"),
    }
    extraction_records.append(record)

print(f"Extraction complete: {len(extraction_records)} records processed")
for rec in extraction_records:
    print(f"  {rec['pdf_filename']:50s}  status={rec['extraction_status']:10s}  "
          f"pages={rec['page_count']:3d}  chars={rec['character_count']:7d}")

# COMMAND ----------

# DBTITLE 1,6. Minimal text cleaning
# MAGIC %md
# MAGIC ## 6. Minimal text cleaning
# MAGIC
# MAGIC Applies conservative cleaning that **preserves all content**:
# MAGIC - Normalizes line endings (`\r\n`, `\r` → `\n`).
# MAGIC - Collapses runs of 3+ blank lines into 2.
# MAGIC - Strips leading/trailing whitespace.
# MAGIC
# MAGIC No lowercasing, no removal of game terms, numbers, or URLs — those transformations belong in later stages.

# COMMAND ----------

# DBTITLE 1,6. Minimal text cleaning
# ============================================================
# Minimal text cleaning
# ============================================================
# Conservative cleaning that preserves all content:
#   1. Normalize \r\n and \r to \n
#   2. Collapse runs of 3+ blank lines into 2
#   3. Strip trailing/leading whitespace on the whole text
#
# Deliberately NOT applied (reserved for later stages):
#   - lowercasing
#   - removal of game-specific terms
#   - removal of numbers
#   - removal of URLs
#   - aggressive whitespace normalization within lines


def minimal_clean(text):
    """Apply conservative text cleaning without losing information."""
    if not text:
        return text
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


# Apply cleaning to all records
cleaned_count = 0
for rec in extraction_records:
    original_len = rec["character_count"]
    rec["raw_text"] = minimal_clean(rec["raw_text"])
    rec["character_count"] = len(rec["raw_text"])
    if rec["character_count"] != original_len:
        cleaned_count += 1

print(f"Minimal cleaning applied to {cleaned_count} record(s) "
      f"(character counts updated).")

# COMMAND ----------

# DBTITLE 1,7. Data quality checks
# MAGIC %md
# MAGIC ## 7. Data quality checks
# MAGIC
# MAGIC Validates extraction results before writing to Delta:
# MAGIC - Status distribution (SUCCESS / EMPTY / SUSPICIOUS / FAILED).
# MAGIC - Empty and suspiciously short texts.
# MAGIC - Metadata match coverage.
# MAGIC - Duplicate `document_id` in bronze data.
# MAGIC - Detailed listing of problematic records.

# COMMAND ----------

# DBTITLE 1,7. Data quality checks
# ============================================================
# Data quality checks
# ============================================================
# Validates the extraction results before writing to Delta.

# --- Build Spark DataFrame from extraction records ---
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    TimestampType, DateType,
)

schema = StructType([
    StructField("document_id", StringType(), True),
    StructField("pdf_filename", StringType(), True),
    StructField("file_path", StringType(), True),
    StructField("raw_text", StringType(), True),
    StructField("ingestion_timestamp", TimestampType(), True),
    StructField("extraction_status", StringType(), True),
    StructField("page_count", IntegerType(), True),
    StructField("character_count", IntegerType(), True),
    StructField("extraction_error", StringType(), True),
    StructField("metadata_match_status", StringType(), True),
    StructField("title", StringType(), True),
    StructField("author", StringType(), True),
    StructField("category", StringType(), True),
    StructField("game_version", StringType(), True),
    StructField("expansion", StringType(), True),
    StructField("source_url", StringType(), True),
    StructField("publication_date", DateType(), True),
])

df_bronze = spark.createDataFrame(extraction_records, schema=schema)

# --- DQ: counts by status ---
print("=== Extraction status distribution ===")
df_bronze.groupBy("extraction_status").count() \
    .orderBy("extraction_status").show(truncate=False)

# --- DQ: empty texts ---
empty_count = df_bronze.filter("character_count = 0").count()
print(f"Records with empty text: {empty_count}")

# --- DQ: suspiciously short texts ---
suspicious_count = df_bronze.filter(
    f"character_count > 0 AND character_count < {MIN_CHAR_THRESHOLD}"
).count()
print(f"Records with suspiciously short text: {suspicious_count}")

# --- DQ: metadata match ---
print("\n=== Metadata match status ===")
df_bronze.groupBy("metadata_match_status").count().show(truncate=False)

# --- DQ: duplicate document_id in bronze ---
dup_bronze = df_bronze.groupBy("document_id").count().filter("count > 1")
dup_bronze_count = dup_bronze.count()
if dup_bronze_count > 0:
    print(f"WARNING: {dup_bronze_count} duplicate document_id(s) in bronze data!")
    dup_bronze.show(truncate=False)
else:
    print("No duplicate document_id values in bronze data.")

# --- DQ: detail of problematic records ---
print("\n=== Problematic records (non-SUCCESS) ===")
df_bronze.filter("extraction_status != 'SUCCESS'") \
    .select("document_id", "pdf_filename", "extraction_status",
            "page_count", "character_count", "extraction_error") \
    .show(truncate=False)

# COMMAND ----------

# DBTITLE 1,8. Write Bronze Delta table
# MAGIC %md
# MAGIC ## 8. Write Bronze Delta table
# MAGIC
# MAGIC Writes the extraction results to `workspace.gcivilization.civ4_pdf_bronze` using **MERGE** on `document_id`.
# MAGIC
# MAGIC **Why MERGE (not overwrite)?**
# MAGIC - Idempotent: re-running updates existing rows instead of duplicating.
# MAGIC - Preserves the table across runs; new PDFs are inserted automatically.
# MAGIC - Simpler than tracking which rows changed.
# MAGIC
# MAGIC The table is created with `CREATE TABLE IF NOT EXISTS` on first run.

# COMMAND ----------

# DBTITLE 1,8. Write Bronze Delta table
# ============================================================
# Write Bronze Delta table
# ============================================================
# Strategy: MERGE (upsert) keyed on document_id.
#
# MERGE is chosen over overwrite because:
#   - It is idempotent: re-running the notebook updates existing
#     rows instead of duplicating them.
#   - If new PDFs are added in the future, they are inserted
#     automatically.
#   - It avoids the complexity of manual deduplication logic.
#
# The table is created if it does not exist.

# Ensure schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# Create table if not exists with explicit schema
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME}
(
    document_id STRING,
    pdf_filename STRING,
    file_path STRING,
    raw_text STRING,
    ingestion_timestamp TIMESTAMP,
    extraction_status STRING,
    page_count INT,
    character_count INT,
    extraction_error STRING,
    metadata_match_status STRING,
    title STRING,
    author STRING,
    category STRING,
    game_version STRING,
    expansion STRING,
    source_url STRING,
    publication_date DATE
)
USING DELTA
""")

# Register the DataFrame as a temp view for MERGE
df_bronze.createOrReplaceTempView("bronze_updates")

# MERGE: update existing rows, insert new ones
spark.sql(f"""
MERGE INTO {FULL_TABLE_NAME} AS target
USING bronze_updates AS source
ON target.document_id = source.document_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

print(f"Bronze table updated: {FULL_TABLE_NAME}")

# COMMAND ----------

# DBTITLE 1,9. Summary statistics
# MAGIC %md
# MAGIC ## 9. Summary statistics
# MAGIC
# MAGIC Prints a consolidated summary of the extraction run and previews the bronze table.

# COMMAND ----------

# DBTITLE 1,9. Summary statistics
# ============================================================
# Summary statistics
# ============================================================

total_records = df_bronze.count()
success_count = df_bronze.filter("extraction_status = 'SUCCESS'").count()
empty_count = df_bronze.filter("extraction_status = 'EMPTY'").count()
failed_count = df_bronze.filter("extraction_status = 'FAILED'").count()
suspicious_count = df_bronze.filter("extraction_status = 'SUSPICIOUS'").count()
total_chars = df_bronze.agg({"character_count": "sum"}).collect()[0][0] or 0

print("=" * 55)
print("  EXTRACTION SUMMARY — civ4_pdf_bronze")
print("=" * 55)
print(f"  Metadata records      : {metadata_count}")
print(f"  PDFs discovered        : {pdf_count}")
print(f"  Total records in bronze: {total_records}")
print(f"  Successful extraction  : {success_count}")
print(f"  Empty extraction       : {empty_count}")
print(f"  Suspicious extraction  : {suspicious_count}")
print(f"  Failed extraction      : {failed_count}")
print(f"  PDFs without metadata  : {len(pdfs_without_metadata)}")
print(f"  Metadata without PDF   : {len(metadata_without_pdf)}")
print(f"  Total extracted chars  : {total_chars:,}")
print("=" * 55)

# Show final table preview
print(f"\nTable preview: {FULL_TABLE_NAME}")
spark.sql(f"""
SELECT
    document_id,
    pdf_filename,
    extraction_status,
    page_count,
    character_count
FROM {FULL_TABLE_NAME}
ORDER BY document_id
""").show(truncate=False)

# Show total row count
row_count = spark.sql(f"SELECT COUNT(*) AS cnt FROM {FULL_TABLE_NAME}").collect()[0][0]
print(f"Total rows in {FULL_TABLE_NAME}: {row_count}")

# COMMAND ----------

# DBTITLE 1,Next Step
# MAGIC %md
# MAGIC ## Next Step
# MAGIC
# MAGIC The output of this notebook — the `civ4_pdf_bronze` Delta table — will be consumed by:
# MAGIC
# MAGIC **`02_processing/02_metadata_and_chunking`**
# MAGIC
# MAGIC That notebook will:
# MAGIC - read the bronze table;
# MAGIC - apply chunking strategies to `raw_text`;
# MAGIC - enrich chunks with metadata fields already preserved here;
# MAGIC - produce a silver-grade table of chunks ready for embedding.
# MAGIC
# MAGIC > **Note:** PDFs with `extraction_status = 'EMPTY'` (image-based PDFs without a text layer)
# MAGIC > will require OCR processing before they can participate in retrieval. This should be
# MAGIC > addressed in a future enhancement.

# COMMAND ----------

