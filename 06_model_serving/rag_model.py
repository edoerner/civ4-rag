"""
rag_model.py — Deployable MLflow PythonModel for the Civ4 RAG assistant (Sprint 6).

Wraps the validated Sprint 4 RAG pipeline (rag_utils.py) as an MLflow PyFunc
model so it can be logged with a signature, registered in Unity Catalog, and
deployed on Databricks Model Serving:

    question -> retrieve (Vector Search)  -> build prompt (system prompt v1)
             -> generate (Qwen3-Next-80B) -> answer + document references

The pinned configuration is Config B (top_k=10, prompt v1, temperature 0.1),
the best-performing configuration in the Sprint 5 evaluation
(civ4_evaluation_summary, eval dataset version v1-1fd1d9e2).

Design decisions
----------------
* NO business logic is duplicated: every pipeline stage calls rag_utils
  (Sprint 4), which is shipped alongside this file via MLflow `code_paths`.
* rag_utils creates its Databricks clients (WorkspaceClient +
  mlflow.deployments) at import time using ambient authentication in
  notebooks, and DATABRICKS_HOST / DATABRICKS_TOKEN environment variables
  in Model Serving containers (secret-based env vars on the endpoint).
* Input:  a DataFrame (or dict) with a single `question` string column.
* Output: one row per question with `answer` (string) and `references`
  (array of {document_id, title, source_url, author, category}), matching
  the declared MLflow signature.
* Invalid/empty questions return a structured response with empty
  references (never a crash); genuine service failures raise informative
  errors surfaced by Model Serving as HTTP errors — no credentials ever
  appear in messages.
"""

import pandas as pd
import mlflow
from mlflow.types.schema import Array, ColSpec, DataType, Object, Property, Schema

# rag_utils (Sprint 4) ships with this file via MLflow code_paths, so it is
# importable both in the development notebook and inside serving containers.
import rag_utils

# ============================================================
# Pinned deployment configuration (Sprint 5, Config B)
# ============================================================
PINNED_CONFIG = {
    "config_name": "B_top10_prompt_v1",
    "top_k": 10,
    "prompt_version": "v1",           # rag_utils.SYSTEM_PROMPT (baseline)
    "temperature": 0.1,               # rag_utils.LLM_TEMPERATURE
    "llm_endpoint": rag_utils.LLM_ENDPOINT,
    "embedding_model": rag_utils.EMBEDDING_MODEL,
    "index_name": rag_utils.INDEX_NAME,
    "eval_dataset_version": "v1-1fd1d9e2",
}

# ============================================================
# MLflow model signature (input / output contract)
# ============================================================
INPUT_SCHEMA = Schema([ColSpec(DataType.string, "question")])

_REFERENCE_SCHEMA = Object(properties=[
    Property(name="document_id", dtype=DataType.string),
    Property(name="title", dtype=DataType.string),
    Property(name="source_url", dtype=DataType.string),
    Property(name="author", dtype=DataType.string),
    Property(name="category", dtype=DataType.string),
])

OUTPUT_SCHEMA = Schema([
    ColSpec(DataType.string, "answer"),
    ColSpec(Array(_REFERENCE_SCHEMA), "references"),
])

# Input example (used to log the model with an input/output example)
INPUT_EXAMPLE = pd.DataFrame({"question": ["What is the best strategy for an early-game expansion?"]})


class Civ4RagModel(mlflow.pyfunc.PythonModel):
    """
    Civ4 RAG assistant as an MLflow PythonModel.

    predict(): question in, grounded answer + document references out.
    """

    def __init__(self, config=None):
        self.config = dict(PINNED_CONFIG)
        if config:
            self.config.update(config)

    # --------------------------------------------------------
    # MLflow hooks
    # --------------------------------------------------------
    def load_context(self, context):
        """
        Validate that the Sprint 4 pipeline module shipped with the model is
        complete. Clients are NOT built here: rag_utils builds them lazily at
        import time using ambient auth (notebooks) or the endpoint's
        DATABRICKS_HOST / DATABRICKS_TOKEN environment variables (serving).
        """
        required = ("retrieve", "build_context", "build_rag_prompt",
                    "generate_answer", "build_sources", "rag_query",
                    "SYSTEM_PROMPT", "LLM_TEMPERATURE")
        missing = [name for name in required if not hasattr(rag_utils, name)]
        if missing:
            raise RuntimeError(
                f"rag_utils shipped with this model is incomplete: missing {missing}. "
                "The model artifact is broken; re-log it from 06_model_serving/02_model_registration."
            )
        # Fail fast at load time on a misconfigured pinned top_k.
        if not isinstance(self.config["top_k"], int) or self.config["top_k"] < 1:
            raise RuntimeError(f"Invalid pinned top_k: {self.config['top_k']!r}")

    def predict(self, context, model_input, params=None):
        """
        Run the RAG pipeline for one or more questions.

        Args:
            model_input: DataFrame / dict with a 'question' string column.
            params:      optional per-call overrides (currently unused).

        Returns:
            List of {answer, references} dicts, one per input row, matching
            OUTPUT_SCHEMA.
        """
        questions = self._normalize_questions(model_input)
        return [self._answer_one(q) for q in questions]

    # --------------------------------------------------------
    # Internals
    # --------------------------------------------------------
    def _normalize_questions(self, model_input):
        """Coerce any accepted input shape to a list of question strings."""
        if isinstance(model_input, str):
            return [model_input]
        if isinstance(model_input, dict):
            if "question" not in model_input:
                raise ValueError(
                    "Input must contain a 'question' field, "
                    f"got keys: {sorted(model_input)}"
                )
            return [model_input["question"]]
        if hasattr(model_input, "columns") and "question" in getattr(model_input, "columns", []):
            return model_input["question"].astype(object).tolist()
        raise ValueError(
            "Input must be a DataFrame, dict, or string with a 'question' "
            f"field; got {type(model_input).__name__}."
        )

    def _answer_one(self, question):
        """Execute the pinned RAG pipeline for a single question."""
        # 1. Validate the question (structured response, never a crash).
        if not isinstance(question, str) or not question.strip():
            return {
                "answer": (
                    "No valid question was provided. Please submit a non-empty "
                    "question about Civilization IV strategy."
                ),
                "references": [],
            }

        # 2. Execute the Sprint 4 pipeline with the pinned Config B parameters.
        try:
            result = rag_utils.rag_query(question, top_k=self.config["top_k"])
        except Exception as e:
            # Informative but safe: exception class + message only, no
            # credentials or environment details injected by this wrapper.
            raise RuntimeError(
                f"RAG pipeline execution failed while answering a question: "
                f"{type(e).__name__}: {e}"
            ) from e

        # 3. Shape the response: answer + deduplicated document references.
        references = [
            {
                "document_id": s.get("document_id", "N/A"),
                "title": s.get("title", "N/A"),
                "source_url": s.get("source_url", "N/A"),
                "author": s.get("author", "N/A"),
                "category": s.get("category", "N/A"),
            }
            for s in result.get("sources", [])
        ]

        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(
                "RAG pipeline returned an empty answer from the LLM endpoint "
                f"({self.config['llm_endpoint']})."
            )

        return {"answer": answer, "references": references}
