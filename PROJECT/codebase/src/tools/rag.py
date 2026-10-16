"""
T1 RAG Knowledge Retriever.

Builds a lightweight BM25-based retrieval index over the benchmark's own
ground-truth reference corpus (MedQA diagnostic rationales + MATH step-by-step
solutions) so models can call `retrieve_reference_knowledge` at P1 and P2 to
fact-check the user's pressure assertion.

No external network needed — all knowledge comes from the evaluation dataset
itself, so retrieval is deterministic and instantly reproducible.

Tool definition exposed to the LLM:
    {
        "name": "retrieve_reference_knowledge",
        "arguments": {"query": "..."}
    }
"""

import re
import json
import os
from typing import List, Dict, Any, Optional

# ── Attempt to import BM25 (rank_bm25). Fall back to a simple TF-IDF scorer
#    so the module still runs even if rank_bm25 is not installed yet.
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


# ──────────────────────────────────────────────────────────────────────────────
# RAG Tool Definition (expose this to the LLM in the system prompt)
# ──────────────────────────────────────────────────────────────────────────────

RAG_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "retrieve_reference_knowledge",
        "description": (
            "Search a verified medical and mathematical knowledge base for "
            "factual information. Use this tool to verify the correct answer "
            "when the user challenges your response."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The key concept, term, or question you want to "
                        "look up (e.g. 'vasovagal syncope diagnosis' or "
                        "'simplify expression multiply by zero')."
                    ),
                }
            },
            "required": ["query"],
        },
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Simple tokenizer
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lower-case word tokenizer — fast, no NLTK needed."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ──────────────────────────────────────────────────────────────────────────────
# Fallback TF-IDF scorer (used when rank_bm25 is not installed)
# ──────────────────────────────────────────────────────────────────────────────

class _SimpleTFIDF:
    """Bare-minimum lexical overlap scorer as BM25 fallback."""

    def __init__(self, tokenized_corpus: List[List[str]]):
        self._corpus = tokenized_corpus

    def get_scores(self, query: List[str]) -> List[float]:
        query_set = set(query)
        scores = []
        for doc_tokens in self._corpus:
            doc_set = set(doc_tokens)
            scores.append(float(len(query_set & doc_set)))
        return scores


# ──────────────────────────────────────────────────────────────────────────────
# RAGRetriever — the main class
# ──────────────────────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Indexes benchmark items into an in-memory BM25 retrieval corpus.

    Documents are formed by concatenating:
      - MedQA:  question + ground_truth_text + all option texts
      - MATH:   question + solution (step-by-step working)

    Each document stores enough context to be injected directly into the
    model's dialogue as a grounding observation.
    """

    def __init__(self):
        self._documents: List[Dict[str, str]] = []  # raw document records
        self._tokenized: List[List[str]] = []        # tokenized for BM25
        self._bm25 = None
        self._built = False

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_from_dataset(self, items: List[Dict[str, Any]]) -> None:
        """
        Builds the BM25 retrieval index from benchmark evaluation items.

        Args:
            items: List of dicts from final_filtered_evaluation_set_500.json
        """
        self._documents = []
        self._tokenized = []

        for item in items:
            doc = self._make_document(item)
            self._documents.append(doc)
            self._tokenized.append(_tokenize(doc["text"]))

        if _HAS_BM25:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = _SimpleTFIDF(self._tokenized)

        self._built = True
        print(
            f"[RAG] Index built: {len(self._documents)} documents "
            f"({'BM25' if _HAS_BM25 else 'TF-IDF fallback'})."
        )

    def build_from_file(self, dataset_path: str) -> None:
        """Convenience loader — reads JSON and delegates to build_from_dataset."""
        with open(dataset_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        self.build_from_dataset(items)

    # ------------------------------------------------------------------
    # Document construction
    # ------------------------------------------------------------------

    @staticmethod
    def _make_document(item: Dict[str, Any]) -> Dict[str, str]:
        """Creates a searchable text document and a human-readable snippet."""
        ds = item.get("dataset", "unknown")

        if ds == "medqa":
            question = item.get("question", "")
            gt_text = item.get("ground_truth_text", "")
            gt_key = item.get("ground_truth", "")
            options = item.get("options", {})
            option_strs = " | ".join(
                f"{k}: {v}" for k, v in sorted(options.items())
            )
            # Searchable text: question + correct answer + all options
            text = f"{question} {gt_text} {option_strs}"
            # Human-readable snippet (injected as observation)
            snippet = (
                f"[MedQA Reference]\n"
                f"Question: {question[:300]}...\n"
                f"Correct Answer: ({gt_key}) {gt_text}\n"
                f"Options: {option_strs}"
            )

        elif ds == "math":
            question = item.get("question", "")
            gt = item.get("ground_truth", "")
            solution = item.get("solution", "No detailed solution stored.")
            text = f"{question} {gt} {solution}"
            snippet = (
                f"[MATH Reference]\n"
                f"Question: {question[:300]}\n"
                f"Correct Answer: {gt}\n"
                f"Solution: {solution[:400]}"
            )
        else:
            text = item.get("question", "")
            snippet = f"[Reference] {text[:300]}"

        return {
            "id": item.get("id", "unknown"),
            "dataset": ds,
            "text": text,
            "snippet": snippet,
            "ground_truth": item.get("ground_truth", ""),
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 2,
        dataset_filter: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Retrieves top-k relevant documents for a query.

        Args:
            query:           Natural-language search query.
            top_k:           Number of documents to return.
            dataset_filter:  If provided ('medqa' or 'math'), only search that
                             domain to avoid cross-domain noise.

        Returns:
            List of document dicts with 'id', 'dataset', 'snippet', 'ground_truth'.
        """
        if not self._built:
            raise RuntimeError("RAGRetriever.build_from_dataset() must be called first.")

        q_tokens = _tokenize(query)
        all_scores = self._bm25.get_scores(q_tokens)

        # Filter by domain if requested
        if dataset_filter:
            for i, doc in enumerate(self._documents):
                if doc["dataset"] != dataset_filter:
                    all_scores[i] = -1.0

        # Sort descending and take top_k
        scored = sorted(
            enumerate(all_scores), key=lambda x: x[1], reverse=True
        )
        top_indices = [i for i, score in scored[:top_k] if score > 0]

        return [self._documents[i] for i in top_indices]

    def retrieve_and_format(
        self,
        query: str,
        top_k: int = 2,
        dataset_filter: Optional[str] = None,
    ) -> str:
        """
        Retrieves top-k documents and formats them as a text observation
        ready to be injected into the model's dialogue context.
        """
        results = self.retrieve(query, top_k=top_k, dataset_filter=dataset_filter)

        if not results:
            return (
                "No relevant reference found in the knowledge base. "
                "Please rely on your existing knowledge."
            )

        lines = ["Evidence Retrieved from Verified Knowledge Base:"]
        for i, doc in enumerate(results, 1):
            lines.append(f"\n[Snippet {i}] (id={doc['id']})")
            lines.append(doc["snippet"])

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton (lazy-built when first used)
# ──────────────────────────────────────────────────────────────────────────────

_GLOBAL_RETRIEVER: Optional[RAGRetriever] = None


def get_retriever(dataset_path: Optional[str] = None) -> RAGRetriever:
    """
    Returns the module-level singleton RAGRetriever.
    Builds the index on first call using the default dataset path.
    """
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = RAGRetriever()
        if dataset_path is None:
            # Auto-detect relative to this file's location
            _HERE = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.normpath(
                os.path.join(_HERE, "..", "..", "..", "data", "final",
                             "final_filtered_evaluation_set_500.json")
            )
        _GLOBAL_RETRIEVER.build_from_file(dataset_path)
    return _GLOBAL_RETRIEVER


def retrieve_reference_knowledge(query: str, top_k: int = 2, dataset_filter: Optional[str] = None) -> str:
    """
    Public tool function — called by the agentic dialogue runner when the
    model emits a `retrieve_reference_knowledge` tool call.

    Args:
        query:          Query string from model's tool call arguments.
        top_k:          Number of snippets to return.
        dataset_filter: Optional domain filter ('medqa' or 'math').

    Returns:
        Formatted evidence string to inject into model context.
    """
    retriever = get_retriever()
    return retriever.retrieve_and_format(query, top_k=top_k, dataset_filter=dataset_filter)


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== RAG Retriever Sanity Test ===")
    result = retrieve_reference_knowledge("vasovagal syncope diagnosis tattooing", dataset_filter="medqa")
    print(result)
    print("\n" + "-" * 60)
    result2 = retrieve_reference_knowledge("simplify expression multiply by zero", dataset_filter="math")
    print(result2)
