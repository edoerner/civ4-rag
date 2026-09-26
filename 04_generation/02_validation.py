# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 04_generation / 02_validation
# MAGIC
# MAGIC **Purpose:** Validate the RAG generation pipeline — retrieval integration, context construction, prompt construction, LLM generation, source references, and insufficient-context behavior.
# MAGIC
# MAGIC **Inputs:**
# MAGIC - Shared module: `04_generation/rag_utils.py` (imported by both notebooks)
# MAGIC - Vector Search index: `workspace.gcivilization.civ4_chunks_embeddings_index` (Sprint 3)
# MAGIC - LLM endpoint: `databricks-qwen3-next-80b-a3b-instruct` (Databricks Foundation Model API)
# MAGIC
# MAGIC **Outputs:**
# MAGIC - Validation report (printed) — no persisted outputs.
# MAGIC
# MAGIC **Validation dimensions:**
# MAGIC 1. **Retrieval integration** — query produces chunks with expected fields, top_k respected
# MAGIC 2. **Context construction** — context contains chunk text, metadata preserved, chunks distinguishable
# MAGIC 3. **Prompt construction** — question included, context included, system instructions present
# MAGIC 4. **Generation** — LLM response is non-empty, original query preserved
# MAGIC 5. **Sources** — output contains source references, URLs match retrieved metadata, no fabricated URLs
# MAGIC 6. **Insufficient-context behavior** — out-of-domain query does not produce a confident Civ4-specific answer
# MAGIC
# MAGIC **Idempotency:** Safe to rerun — all checks are read-only.
# MAGIC
# MAGIC **Approach:** The validation notebook imports functions from `rag_utils.py` — the same shared module used by `01_rag_pipeline` — so tests validate the actual implementation, not a copy.

# COMMAND ----------

# DBTITLE 1,1. Setup
# ============================================================
# Setup: import RAG pipeline functions and validation helper
# ============================================================
# Functions are imported from rag_utils.py — the same module used
# by 01_rag_pipeline — so validation tests the actual implementation.

import sys, os

# Ensure the module directory is on sys.path
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

# Validation tracking
validation_results = []

def record_check(name, passed, detail=""):
    """Record a validation check result."""
    status = "PASS" if passed else "FAIL"
    validation_results.append((name, status, detail))
    marker = "✅" if passed else "❌"
    print(f"  {marker} {name}: {status}" + (f" — {detail}" if detail else ""))

print("Validation setup complete.")
print(f"  Index: {INDEX_NAME}")
print(f"  LLM: {LLM_ENDPOINT}")
print(f"  Functions imported from rag_utils (same as 01_rag_pipeline)")

# COMMAND ----------

# DBTITLE 1,2. Retrieval integration
# ============================================================
# 1. Retrieval integration
# ============================================================
# Verifies that a user query produces retrieved chunks with
# the expected fields and that top_k is respected.

print("=" * 55)
print("  RETRIEVAL INTEGRATION")
print("=" * 55)

test_query = "What is a Specialist Economy?"

# Test 1: Query returns results
test_chunks = retrieve(test_query, top_k=5)
record_check("Query returns results", len(test_chunks) > 0,
            f"{len(test_chunks)} chunks returned")

# Test 2: Expected fields present in first chunk
if test_chunks:
    expected_fields = RETRIEVAL_COLUMNS + ["score"]
    missing_fields = [f for f in expected_fields if f not in test_chunks[0]]
    record_check("Expected fields present", len(missing_fields) == 0,
                f"missing: {missing_fields}" if missing_fields else "all fields present")

    # Test 3: chunk_text is non-empty
    empty_text_count = sum(1 for c in test_chunks if not c.get("chunk_text"))
    record_check("chunk_text non-empty", empty_text_count == 0,
                f"{empty_text_count} empty text(s)")

    # Test 4: score is numeric
    score_numeric = all(isinstance(c.get("score"), (int, float)) for c in test_chunks)
    record_check("Scores are numeric", score_numeric,
                f"top score: {test_chunks[0].get('score', 'N/A')}")

# Test 5: top_k is respected
test_chunks_3 = retrieve(test_query, top_k=3)
record_check("top_k=3 returns 3 results", len(test_chunks_3) == 3,
            f"returned {len(test_chunks_3)} results")

test_chunks_1 = retrieve(test_query, top_k=1)
record_check("top_k=1 returns 1 result", len(test_chunks_1) == 1,
            f"returned {len(test_chunks_1)} results")

# Test 6: Empty query returns empty list
empty_result = retrieve("")
record_check("Empty query returns empty list", len(empty_result) == 0,
            f"returned {len(empty_result)} results")

# COMMAND ----------

# DBTITLE 1,3. Context construction
# ============================================================
# 2. Context construction
# ============================================================
# Verifies that build_context() produces a structured context
# with chunk text, metadata, and distinguishable sources.

print("\n" + "=" * 55)
print("  CONTEXT CONSTRUCTION")
print("=" * 55)

context = build_context(test_chunks)

# Test 1: Context is non-empty
record_check("Context non-empty", len(context) > 0,
            f"{len(context)} characters")

# Test 2: Contains chunk text from retrieved chunks
has_text = any(c.get("chunk_text", "") in context for c in test_chunks)
record_check("Context contains chunk text", has_text,
            "chunk text found in context")

# Test 3: Source metadata preserved (title, source_url, document_id, chunk_id)
has_title = any(c.get("title", "") in context for c in test_chunks if c.get("title"))
record_check("Context contains titles", has_title, "titles found")

has_url = any(c.get("source_url", "") in context for c in test_chunks if c.get("source_url"))
record_check("Context contains source URLs", has_url, "URLs found")

has_doc_id = any(c.get("document_id", "") in context for c in test_chunks if c.get("document_id"))
record_check("Context contains document IDs", has_doc_id, "document IDs found")

has_chunk_id = any(c.get("chunk_id", "") in context for c in test_chunks if c.get("chunk_id"))
record_check("Context contains chunk IDs", has_chunk_id, "chunk IDs found")

# Test 4: Individual chunks are distinguishable ([Source N] markers)
source_markers = context.count("[Source ")
record_check("Chunks are distinguishable", source_markers == len(test_chunks),
            f"{source_markers} [Source N] markers for {len(test_chunks)} chunks")

# Test 5: Empty list returns "No relevant context"
empty_context = build_context([])
record_check("Empty chunks returns fallback message",
            "No relevant context" in empty_context,
            "fallback message present")

# COMMAND ----------

# DBTITLE 1,4. Prompt construction
# ============================================================
# 3. Prompt construction
# ============================================================
# Verifies that build_rag_prompt() includes the question,
# retrieved context, and system instructions.

print("\n" + "=" * 55)
print("  PROMPT CONSTRUCTION")
print("=" * 55)

sys_prompt, user_prompt = build_rag_prompt(test_query, test_chunks)

# Test 1: System prompt is non-empty
record_check("System prompt non-empty", len(sys_prompt) > 0,
            f"{len(sys_prompt)} characters")

# Test 2: System prompt contains anti-hallucination instructions
has_anti_halluc = "do not invent" in sys_prompt.lower() or "do not fabricate" in sys_prompt.lower()
record_check("Anti-hallucination instructions present", has_anti_halluc,
            "grounding instructions found")

# Test 3: System prompt references context
has_context_ref = "CONTEXT" in sys_prompt or "context" in sys_prompt.lower()
record_check("System prompt references context", has_context_ref,
            "context reference found")

# Test 4: User prompt contains the question
record_check("Question in user prompt", test_query in user_prompt,
            "question found")

# Test 5: User prompt contains the context
has_context_in_prompt = "CONTEXT:" in user_prompt
record_check("Context in user prompt", has_context_in_prompt,
            "CONTEXT section found")

# Test 6: User prompt contains QUESTION section
has_question_section = "QUESTION:" in user_prompt
record_check("Question section in user prompt", has_question_section,
            "QUESTION section found")

# Test 7: Source metadata available for grounding (in the context within user_prompt)
has_source_metadata = "Source URL:" in user_prompt and "Title:" in user_prompt
record_check("Source metadata available in prompt", has_source_metadata,
            "title and source_url present in context")

# COMMAND ----------

# DBTITLE 1,5. Generation
# ============================================================
# 4. Generation
# ============================================================
# Verifies that the LLM produces a non-empty response
# and that the result preserves the original query.

print("\n" + "=" * 55)
print("  GENERATION")
print("=" * 55)

# Use a simple query for the generation test
gen_query = "How can I achieve a Cultural Victory?"
gen_result = rag_query(gen_query, top_k=3)

# Test 1: Answer is non-empty
answer = gen_result.get("answer", "")
record_check("Answer is non-empty", len(answer) > 0,
            f"{len(answer)} characters")

# Test 2: Answer is a string
record_check("Answer is a string", isinstance(answer, str),
            f"type={type(answer).__name__}")

# Test 3: Result preserves original query
record_check("Query preserved in result", gen_result.get("query") == gen_query,
            f"query='{gen_result.get('query', 'N/A')}'")

# Test 4: Result contains retrieved_chunks
has_chunks = "retrieved_chunks" in gen_result and len(gen_result["retrieved_chunks"]) > 0
record_check("Result contains retrieved chunks", has_chunks,
            f"{len(gen_result.get('retrieved_chunks', []))} chunks")

# Test 5: Answer contains source references ([Source N] pattern)
has_source_refs = "[Source" in answer or "source" in answer.lower()
record_check("Answer references sources", has_source_refs,
            "source references found in answer")

# COMMAND ----------

# DBTITLE 1,6. Sources
# ============================================================
# 5. Sources
# ============================================================
# Verifies that the output contains source references,
# that URLs originate from retrieved metadata (not fabricated),
# and that sources are deduplicated by document.

print("\n" + "=" * 55)
print("  SOURCES")
print("=" * 55)

sources = gen_result.get("sources", [])
retrieved = gen_result.get("retrieved_chunks", [])

# Test 1: Sources list is non-empty
record_check("Sources list non-empty", len(sources) > 0,
            f"{len(sources)} source(s)")

# Test 2: Each source has required fields
required_source_fields = ["title", "source_url", "document_id", "chunk_id"]
# Note: chunk_id is inside the 'chunks' sub-list, not at the top level of the source dict
# The top-level source dict has: document_id, title, source_url, author, category, chunks
top_level_fields = ["document_id", "title", "source_url"]
all_fields_present = all(all(f in s for f in top_level_fields) for s in sources)
record_check("Source fields present", all_fields_present,
            f"top-level fields: {top_level_fields}")

# Test 3: URLs match retrieved metadata (not fabricated)
retrieved_urls = set(c.get("source_url", "") for c in retrieved)
source_urls = set(s.get("source_url", "") for s in sources)
fabricated_urls = source_urls - retrieved_urls
record_check("No fabricated URLs", len(fabricated_urls) == 0,
            f"{len(fabricated_urls)} fabricated URL(s)" if fabricated_urls else "all URLs match retrieved metadata")

# Test 4: URLs are from CivFanatics (not invented domains)
all_civfanatics = all("civfanatics.com" in s.get("source_url", "") for s in sources)
record_check("URLs from CivFanatics", all_civfanatics,
            "all URLs contain civfanatics.com")

# Test 5: Sources are deduplicated by document_id
source_doc_ids = [s.get("document_id") for s in sources]
unique_doc_ids = set(source_doc_ids)
is_deduplicated = len(source_doc_ids) == len(unique_doc_ids)
record_check("Sources deduplicated by document", is_deduplicated,
            f"{len(source_doc_ids)} sources, {len(unique_doc_ids)} unique documents")

# Test 6: Each source contains chunk-level traceability
has_chunk_info = all(len(s.get("chunks", [])) > 0 for s in sources)
record_check("Sources contain chunk-level info", has_chunk_info,
            "each source has chunks sub-list")

# COMMAND ----------

# DBTITLE 1,7. Insufficient-context behavior
# ============================================================
# 6. Insufficient-context behavior
# ============================================================
# Tests that an out-of-domain query (unrelated to Civ4) does not
# produce a confident Civilization IV-specific fabricated answer.
# The LLM may answer general-knowledge questions from its training
# data, but it should NOT present Civ4 strategy content as relevant
# to the unrelated question.

print("\n" + "=" * 55)
print("  INSUFFICIENT-CONTEXT BEHAVIOR")
print("=" * 55)

ood_query = "What is the capital city of France?"
ood_result = rag_query(ood_query, top_k=3)

ood_answer = ood_result.get("answer", "")
ood_chunks = ood_result.get("retrieved_chunks", [])

# Test 1: Out-of-domain query still retrieves chunks (Vector Search always returns results)
record_check("OOD query retrieves chunks", len(ood_chunks) > 0,
            f"{len(ood_chunks)} chunks retrieved")

# Test 2: Answer acknowledges the context is about Civ4, not the question topic
# The LLM should recognize that the retrieved context is about Civilization IV
# strategy and does not address the out-of-domain question.
answer_lower = ood_answer.lower()
acknowledges_context_mismatch = (
    "civilization" in answer_lower or
    "context" in answer_lower or
    "does not address" in answer_lower or
    "not about" in answer_lower or
    "strategy" in answer_lower or
    "not relevant" in answer_lower or
    "unrelated" in answer_lower or
    "outside" in answer_lower or
    "real-world" in answer_lower or
    "insufficient" in answer_lower or
    "don't have enough" in answer_lower or
    "not enough information" in answer_lower
)
record_check("Answer acknowledges context mismatch", acknowledges_context_mismatch,
            "answer recognizes context does not address the question")

# Test 3: Answer does not present Civ4 strategy as the answer to the OOD question
# The answer should not start with Civ4 strategy advice as if it answers
# "What is the capital of France?"
starts_with_civ4_advice = answer_lower.strip().startswith((
    "build", "research", "train", "produce", "attack",
    "rush", "focus", "prioritize", "choose"
))
record_check("Does not present Civ4 advice as answer", not starts_with_civ4_advice,
            "answer does not start with Civ4 strategy advice")

# Print the OOD answer for manual inspection
print(f"\n  OOD Query: {ood_query}")
print(f"  OOD Answer (first 500 chars):\n  {ood_answer[:500]}")

# COMMAND ----------

# DBTITLE 1,8. Validation summary
# ============================================================
# Validation summary
# ============================================================

pass_count = sum(1 for _, s, _ in validation_results if s == "PASS")
fail_count = sum(1 for _, s, _ in validation_results if s == "FAIL")
total = len(validation_results)

print("\n" + "=" * 55)
print("  SPRINT 4 VALIDATION SUMMARY")
print("=" * 55)
print(f"  Total checks : {total}")
print(f"  Passed       : {pass_count}")
print(f"  Failed       : {fail_count}")
print("=" * 55)

if fail_count > 0:
    print("\n  FAILED CHECKS:")
    for name, status, detail in validation_results:
        if status == "FAIL":
            print(f"    ❌ {name}: {detail}")

if fail_count == 0:
    print("\n  ✅ All validation checks passed!")
    print("  Sprint 4 (RAG / Generation) is complete.")
    print("  The RAG pipeline produces grounded answers with source references.")
else:
    print(f"\n  ❌ {fail_count} check(s) failed. Review the details above.")

print("\n  Pipeline:")
print("    Question → retrieve() → build_context() → build_rag_prompt() → generate_answer() → answer + sources")
print("\n  Lineage:")
print("    PDFs → civ4_chunks → civ4_chunks_embeddings → Vector Search → retrieval → RAG prompt → LLM → grounded answer")