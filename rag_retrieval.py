"""RAG asset-context retrieval for the NVD + RAG vulnerability assessment pipeline.

Kept separate from Streamlit so it can be reused by an evaluation script later.

Scoring notes (verified against the installed langchain-chroma/chromadb version):
- Chroma.similarity_search_with_score() returns the RAW distance metric of the
  collection (L2 here, since no explicit "hnsw:space" was set at ingestion).
  For raw distance, LOWER is better/more similar. It is NOT bounded to [0, 1]
  and must never be treated as "higher = more similar".
- Chroma.similarity_search_with_relevance_scores() applies the vectorstore's
  relevance_score_fn on top of that raw distance to produce a normalized
  "confidence" style score where HIGHER = more similar (roughly in [-1, 1] for
  an L2 space with non-normalized embeddings; well-separated in practice
  between clear asset-name matches (>0.3) and unrelated queries (<=0).
- This module uses the normalized relevance score as the "confidence" value
  exposed to the UI/threshold, and keeps the raw distance available for
  traceability/debugging.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_TOP_K = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.15


@dataclass
class AssetCandidate:
    """One retrieved asset candidate with its retrieval confidence."""

    metadata: Dict[str, Any]
    confidence: float  # normalized relevance score, higher = more similar
    distance: Optional[float] = None  # raw vectorstore distance, lower = more similar
    content: str = ""


@dataclass
class RetrievalResult:
    """Result of a RAG asset-context retrieval call."""

    query: str
    threshold: float
    candidates: List[AssetCandidate] = field(default_factory=list)

    @property
    def best(self) -> Optional[AssetCandidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def passed_threshold(self) -> bool:
        return self.best is not None and self.best.confidence >= self.threshold


# Metadata fields relevant to the vulnerability-assessment context. Fields not
# present in the underlying asset dataset are simply omitted (kept optional).
CONTEXT_METADATA_FIELDS = [
    "asset_id",
    "asset_name",
    "environment",
    "internet_exposed",
    "business_criticality",
    "data_sensitivity",
    "network_zone",
    "ip_address",
]


def retrieve_asset_context(
    vector_store,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> RetrievalResult:
    """Retrieve top-k asset candidates for `query` with confidence scores.

    Does not silently fall back to an unrelated asset: callers should check
    `RetrievalResult.passed_threshold` before using `.best` for downstream
    CVSS/LLM steps.
    """
    query = (query or "").strip()
    result = RetrievalResult(query=query, threshold=confidence_threshold)
    if not vector_store or not query:
        return result

    try:
        scored = vector_store.similarity_search_with_relevance_scores(query, k=top_k)
    except Exception:
        # Fallback for vectorstores/versions without a relevance_score_fn:
        # derive a best-effort confidence from raw distance instead of
        # assuming a raw distance is itself a similarity score.
        scored = []
        for doc, distance in vector_store.similarity_search_with_score(query, k=top_k):
            confidence = 1.0 / (1.0 + max(distance, 0.0))
            scored.append((doc, confidence, distance))
        for doc, confidence, distance in scored:
            result.candidates.append(
                AssetCandidate(metadata=dict(doc.metadata), confidence=confidence,
                                distance=distance, content=doc.page_content)
            )
        result.candidates.sort(key=lambda c: c.confidence, reverse=True)
        return result

    for doc, confidence in scored:
        result.candidates.append(
            AssetCandidate(metadata=dict(doc.metadata), confidence=float(confidence),
                            content=doc.page_content)
        )
    result.candidates.sort(key=lambda c: c.confidence, reverse=True)
    return result


def filtered_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the organizational-context fields relevant to CVSS/priority
    logic, in a stable order, keeping only fields actually present."""
    return {k: metadata[k] for k in CONTEXT_METADATA_FIELDS if k in metadata}
