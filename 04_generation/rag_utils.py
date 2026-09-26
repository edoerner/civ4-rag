"""
rag_utils.py — Shared RAG pipeline functions and configuration.

This module is imported by both 01_rag_pipeline and 02_validation
to avoid code duplication and ensure validation tests the actual
implementation.

Pipeline: question → retrieve() → build_context() → build_rag_prompt()
          → generate_answer() → answer + sources
"""

import mlflow.deployments
from databricks.sdk import WorkspaceClient

# ============================================================
# Configuration
# ============================================================
# Retrieval configuration reused from Sprint 3.
# LLM configuration added in Sprint 4.

CATALOG = "workspace"
SCHEMA = "gcivilization"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.civ4_chunks_embeddings_index"
EMBEDDING_MODEL = "databricks-gte-large-en"

# Columns retrieved from the Vector Search index
# (must match columns_to_sync from Sprint 3 Notebook 02, minus embedding)
RETRIEVAL_COLUMNS = [
    "chunk_id", "chunk_text", "document_id", "title",
    "author", "category", "game_version", "expansion",
    "source_url", "publication_date",
]

# LLM configuration (Sprint 4)
# databricks-qwen3-next-80b-a3b-instruct is a Databricks Foundation Model
# whose description explicitly cites retrieval-augmented generation.
LLM_ENDPOINT = "databricks-qwen3-next-80b-a3b-instruct"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.1  # Low temperature for factual, grounded answers

# RAG pipeline defaults
DEFAULT_TOP_K = 5

# ============================================================
# Clients
# ============================================================

w = WorkspaceClient()
deploy_client = mlflow.deployments.get_deploy_client("databricks")

# ============================================================
# Retrieval — reuse Sprint 3 retrieve() logic
# ============================================================

def retrieve(query, top_k=DEFAULT_TOP_K):
    """
    Retrieve top-K relevant Civ4 chunks for a natural-language query.

    Args:
        query:  Natural-language query string.
        top_k:  Number of results to return (default 5).

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
            chunk["score"] = row[-1]
            chunks.append(chunk)

    return chunks

# ============================================================
# Context construction
# ============================================================

def build_context(retrieved_chunks):
    """
    Build a structured context string from retrieved chunks.

    Each chunk is formatted as:
        [Source N]
        Title: ...
        Document ID: ...
        Chunk ID: ...
        Source URL: ...
        Content: ...

    Args:
        retrieved_chunks: List of chunk dicts from retrieve().

    Returns:
        Formatted context string for the LLM prompt.
    """
    if not retrieved_chunks:
        return "No relevant context was retrieved."

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        part = f"[Source {i}]\n"
        part += f"Title: {chunk.get('title', 'N/A')}\n"
        part += f"Document ID: {chunk.get('document_id', 'N/A')}\n"
        part += f"Chunk ID: {chunk.get('chunk_id', 'N/A')}\n"
        part += f"Source URL: {chunk.get('source_url', 'N/A')}\n"
        part += f"Content:\n{chunk.get('chunk_text', '')}\n"
        context_parts.append(part)

    return "\n".join(context_parts)

# ============================================================
# Prompt construction
# ============================================================

SYSTEM_PROMPT = """\
You are a Civilization IV strategy advisor. Your task is to answer the user's \
question about Civilization IV strategy using ONLY the provided context from \
retrieved strategy-guide documents.

Instructions:
1. Answer the question using the information in the CONTEXT section below.
2. Prioritize information from the retrieved documents. Do not invent facts \
   that are not supported by the context.
3. If the context does not contain sufficient information to answer the \
   question, clearly state: "I don't have enough information in the available \
   context to fully answer this question." Do not fabricate a Civilization \
   IV-specific answer.
4. Preserve important Civilization IV terminology (e.g., Specialist Economy, \
   Cottage Economy, warmongering, cultural victory, etc.).
5. Provide a concise but useful answer. Aim for 3-6 paragraphs.
6. Reference the sources you used by their [Source N] label, e.g., \
   "According to [Source 2]..."
7. Do not fabricate source titles, URLs, or citations. Only reference \
   sources that appear in the CONTEXT section.
"""


def build_rag_prompt(query, retrieved_chunks):
    """
    Build the complete RAG prompt (system + context + question).

    Args:
        query:            The user's question string.
        retrieved_chunks: List of chunk dicts from retrieve().

    Returns:
        Tuple of (system_prompt, user_prompt) strings for the LLM.
    """
    context = build_context(retrieved_chunks)

    user_prompt = f"""\
CONTEXT:
{context}

QUESTION:
{query}
"""

    return SYSTEM_PROMPT, user_prompt

# ============================================================
# LLM generation
# ============================================================

def generate_answer(system_prompt, user_prompt):
    """
    Generate an answer using the configured LLM endpoint.

    Args:
        system_prompt: System instructions string.
        user_prompt:   User message (context + question) string.

    Returns:
        Generated answer string.
    """
    response = deploy_client.predict(
        endpoint=LLM_ENDPOINT,
        inputs={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE
        }
    )

    # Extract the generated text from the OpenAI-format response
    answer = response["choices"][0]["message"]["content"]
    return answer

# ============================================================
# Source references
# ============================================================

def build_sources(retrieved_chunks):
    """
    Build a deduplicated list of source references from retrieved chunks.

    Multiple chunks from the same document are grouped into one source entry,
    but chunk-level information is retained internally for traceability.

    Args:
        retrieved_chunks: List of chunk dicts from retrieve().

    Returns:
        List of source dicts with:
          document_id, title, source_url, author, category, chunks
        where `chunks` is a list of {chunk_id, score} for traceability.
    """
    sources_by_doc = {}
    for chunk in retrieved_chunks:
        doc_id = chunk.get("document_id", "unknown")
        if doc_id not in sources_by_doc:
            sources_by_doc[doc_id] = {
                "document_id": doc_id,
                "title": chunk.get("title", "N/A"),
                "source_url": chunk.get("source_url", "N/A"),
                "author": chunk.get("author", "N/A"),
                "category": chunk.get("category", "N/A"),
                "chunks": []
            }
        sources_by_doc[doc_id]["chunks"].append({
            "chunk_id": chunk.get("chunk_id", "N/A"),
            "score": chunk.get("score", None)
        })

    return list(sources_by_doc.values())

# ============================================================
# RAG pipeline — ties everything together
# ============================================================

def rag_query(query, top_k=DEFAULT_TOP_K):
    """
    Execute a complete RAG query: retrieve → build context → generate answer.

    Args:
        query:  The user's question string.
        top_k:  Number of chunks to retrieve (default 5).

    Returns:
        Dict with:
          query:            The original question.
          answer:           The LLM-generated answer.
          sources:          List of source references (deduplicated by document).
          retrieved_chunks: List of all retrieved chunk dicts (for traceability).
    """
    # Stage 1: Retrieval
    retrieved_chunks = retrieve(query, top_k=top_k)

    # Stage 2: Prompt construction
    system_prompt, user_prompt = build_rag_prompt(query, retrieved_chunks)

    # Stage 3: LLM generation
    answer = generate_answer(system_prompt, user_prompt)

    # Stage 4: Build source references
    sources = build_sources(retrieved_chunks)

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks
    }