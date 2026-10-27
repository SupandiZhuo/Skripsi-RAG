"""Asset lookup and separately scoped organizational document retrieval.

Only similarity_search_with_relevance_scores is used; failures never silently
switch to a different distance transformation. Scores are retrieval relevance,
not calibrated confidence or proof of asset identity. Organizational documents
use a dedicated cosine collection and must match the selected asset exactly.
"""

import hashlib
import json
import math
from datetime import date

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

DEFAULT_TOP_K = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.15

# Runtime extension field for editable software inventory (Phase B1).
SOFTWARE_INVENTORY_FIELD = "software_inventory_json"


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


# Metadata fields relevant to the vulnerability-assessment context (Asset
# Schema v2). Fields not present in the underlying asset document are simply
# omitted (kept optional), so v1 documents keep working while v2 documents
# expose the full evidence: identity, placement, business function, software
# inventory (CPE applicability), declared CVSS Environmental Requirement
# baseline (CR/IR/AR), and provenance/freshness.
CONTEXT_METADATA_FIELDS = [
    "asset_id",
    "asset_name",
    "description",
    "inventory_verified",
    "ip_address",
    "network_zone",
    "business_criticality",
    "data_sensitivity",
    "environment",
    "internet_exposed",
    "asset_type",
    "owner",
    "business_function",
    "vendor",
    "software_name",
    "software_version",
    "cpe",
    "data_classification",
    "confidentiality_requirement",
    "integrity_requirement",
    "availability_requirement",
    "software_inventory_json",
    "source",
    "source_reference",
    "source_id",
    "source_type",
    "source_version",
    "collected_at",
    "verified_by_role",
    "verified_at",
    "last_updated",
    "is_synthetic",
    "schema_version",
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

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not math.isfinite(confidence_threshold):
        raise ValueError("confidence_threshold must be finite")
    scored = vector_store.similarity_search_with_relevance_scores(query, k=top_k)
    for doc, confidence in scored:
        confidence = float(confidence)
        if not math.isfinite(confidence):
            raise ValueError("Vector store returned a non-finite relevance score")
        result.candidates.append(
            AssetCandidate(metadata=dict(doc.metadata), confidence=confidence,
                            content=doc.page_content)
        )
    result.candidates.sort(key=lambda c: c.confidence, reverse=True)
    return result


def filtered_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the organizational-context fields relevant to CVSS/priority
    logic, in a stable order, keeping only fields actually present."""
    return {k: metadata[k] for k in CONTEXT_METADATA_FIELDS if k in metadata}


def list_all_assets(vector_store) -> List[Dict[str, Any]]:
    """Return all asset metadata records from the vector store.

    Tries the public `get()` API first, then falls back to the underlying
    Chroma collection for older langchain-chroma versions. Missing/None
    metadata entries are skipped.
    """
    if vector_store is None:
        return []
    try:
        data = vector_store.get()
    except Exception:
        try:
            data = vector_store._collection.get()
        except Exception:
            return []
    if not isinstance(data, dict):
        return []
    metadatas = data.get("metadatas") or []
    return [dict(m) for m in metadatas if m]


def _document_id(asset_id: str) -> str:
    return f"asset:{asset_id}"


def _collection(vector_store) -> Any:
    """Return the underlying Chroma collection object."""
    return getattr(vector_store, "_collection", vector_store)


def get_asset_by_id(vector_store, asset_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single asset's metadata by its asset_id."""
    if vector_store is None or not asset_id:
        return None
    collection = _collection(vector_store)
    doc_id = _document_id(asset_id)
    try:
        data = collection.get(ids=[doc_id], include=["metadatas"])
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    ids = data.get("ids") or []
    metadatas = data.get("metadatas") or []
    if ids and metadatas and ids[0] == doc_id and isinstance(metadatas[0], Mapping):
        metadata = dict(metadatas[0])
        return metadata if metadata.get("asset_id") == asset_id else None
    return None


def update_asset_metadata(
    vector_store,
    asset_id: str,
    metadata_updates: Mapping[str, Any],
) -> bool:
    """Merge metadata, regenerate searchable text, and re-embed atomically per ID.

    The public LangChain update API uses the same configured embedding function
    as ingestion. Never update only raw Chroma metadata and leave stale vectors.
    """
    if vector_store is None or not asset_id or not isinstance(metadata_updates, Mapping):
        return False
    if "asset_id" in metadata_updates and metadata_updates["asset_id"] != asset_id:
        return False
    existing = get_asset_by_id(vector_store, asset_id)
    if existing is None:
        return False
    merged = dict(existing)
    merged.update(metadata_updates)
    try:
        from ingest_assets import build_v2_document

        document = build_v2_document(merged)
        # Keep runtime extension fields, including the editable software inventory.
        document.metadata.update(merged)
        vector_store.update_documents(ids=[_document_id(asset_id)], documents=[document])
        saved = _collection(vector_store).get(
            ids=[_document_id(asset_id)], include=["metadatas", "documents"]
        )
        return (
            saved.get("ids") == [_document_id(asset_id)]
            and bool(saved.get("metadatas"))
            and saved["metadatas"][0] == document.metadata
            and saved.get("documents") == [document.page_content]
        )
    except Exception:
        return False


CONTEXT_COLLECTION_NAME = "organizational_context"
DOCUMENT_SCORING_METHOD = "chroma_cosine_relevance"
DOCUMENT_PROVENANCE_FIELDS = (
    "asset_id", "source_id", "document_id", "source", "source_reference",
    "last_updated", "is_synthetic", "chunk_id", "chunk_index", "content_sha256",
    "source_type", "source_version", "collected_at", "verified_by_role", "verified_at",
)


def context_chunk_id(asset_id: str, source_id: str, document_id: str, chunk_index: int) -> str:
    """Stable identity separate from the content hash (updates keep their ID)."""
    key = json.dumps([asset_id, source_id, document_id], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"context:{digest}:{chunk_index}"


def document_provenance_error(metadata: Mapping[str, Any], content: str) -> Optional[str]:
    """Validate attribution/integrity, not authenticity of a source's claims."""
    if not isinstance(metadata, Mapping) or not isinstance(content, str) or not content.strip():
        return "invalid_provenance"
    for key in ("asset_id", "source_id", "document_id", "source", "source_reference", "last_updated"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            return "invalid_provenance"
        if value.casefold() in {"unknown", "none", "null", "n/a"}:
            return "invalid_provenance"
    try:
        parsed_date = date.fromisoformat(metadata["last_updated"])
        if parsed_date.isoformat() != metadata["last_updated"]:
            return "invalid_provenance"
    except (ValueError, TypeError):
        return "invalid_provenance"
    if not (isinstance(metadata.get("is_synthetic"), bool) or metadata.get("is_synthetic") in ("true", "false")):
        return "invalid_provenance"
    index = metadata.get("chunk_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return "invalid_provenance"
    expected_id = context_chunk_id(metadata["asset_id"], metadata["source_id"], metadata["document_id"], index)
    if metadata.get("chunk_id") != expected_id:
        return "invalid_provenance"
    if metadata.get("content_sha256") != hashlib.sha256(content.encode("utf-8")).hexdigest():
        return "content_hash_mismatch"
    return None


@dataclass
class DocumentEvidence:
    evidence_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: Optional[float]
    accepted: bool = False
    rejection_reason: Optional[str] = None
    scoring_method: str = DOCUMENT_SCORING_METHOD

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class DocumentRetrievalResult:
    query: str
    asset_id: str
    threshold: float
    accepted: List[DocumentEvidence] = field(default_factory=list)
    rejected: List[DocumentEvidence] = field(default_factory=list)
    scoring_method: str = DOCUMENT_SCORING_METHOD
    error: Optional[str] = None

    @property
    def candidates(self) -> List[DocumentEvidence]:
        return self.accepted + self.rejected

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def retrieve_document_context(
    vector_store,
    query: str,
    asset_id: str,
    top_k: int = 3,
    relevance_threshold: float = 0.15,
) -> DocumentRetrievalResult:
    """Retrieve scoped source chunks and retain acceptance/rejection evidence.

    No fallback to asset metadata or another score transformation. An empty
    accepted list means no usable document evidence; callers must show that
    explicitly. Storage failures are surfaced in ``error``, never hidden.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if isinstance(relevance_threshold, bool) or not isinstance(relevance_threshold, (int, float)) or not math.isfinite(relevance_threshold) or not 0 <= relevance_threshold <= 1:
        raise ValueError("relevance_threshold must be finite and between 0 and 1")
    if not isinstance(query, str) or not isinstance(asset_id, str):
        raise ValueError("query and asset_id must be strings")
    if asset_id != asset_id.strip():
        raise ValueError("asset_id must not contain surrounding whitespace")
    result = DocumentRetrievalResult(query=query.strip(), asset_id=asset_id, threshold=relevance_threshold)
    if not query.strip() or not asset_id:
        result.error = "A non-empty query and selected asset_id are required"
        return result
    if vector_store is None:
        result.error = "Organizational document collection unavailable"
        return result
    try:
        collection = getattr(vector_store, "_collection", None)
        if collection is not None:
            if getattr(collection, "name", None) != CONTEXT_COLLECTION_NAME:
                raise ValueError("Wrong organizational document collection")
            if (getattr(collection, "metadata", None) or {}).get("hnsw:space") != "cosine":
                raise ValueError("Organizational document collection must use cosine relevance")
        scored = vector_store.similarity_search_with_relevance_scores(
            query.strip(), k=top_k, filter={"asset_id": asset_id}
        )
        seen = set()
        for doc, score in scored:
            metadata = dict(doc.metadata or {})
            content = doc.page_content
            try:
                relevance = float(score) if not isinstance(score, bool) else float("nan")
            except (ValueError, TypeError, OverflowError):
                relevance = float("nan")
            reason = None
            if not math.isfinite(relevance) or not -1 <= relevance <= 1:
                reason = "invalid_score"
            elif metadata.get("asset_id") != asset_id:
                reason = "asset_id_mismatch"
            else:
                reason = document_provenance_error(metadata, content)
            evidence_id = metadata.get("chunk_id", "")
            if not reason and evidence_id in seen:
                reason = "duplicate_evidence"
            if not reason and relevance < relevance_threshold:
                reason = "below_relevance_threshold"
            if not reason:
                seen.add(evidence_id)
            evidence = DocumentEvidence(
                evidence_id=evidence_id, content=content, metadata=metadata,
                relevance_score=relevance if math.isfinite(relevance) else None,
                accepted=reason is None, rejection_reason=reason,
            )
            (result.rejected if reason else result.accepted).append(evidence)
        result.accepted.sort(key=lambda item: item.relevance_score if item.relevance_score is not None else -math.inf, reverse=True)
    except Exception as exc:
        # Do not retain partial accepted evidence if retrieval/auditing failed.
        for evidence in result.accepted:
            evidence.accepted = False
            evidence.rejection_reason = "retrieval_failed"
            result.rejected.append(evidence)
        result.accepted = []
        result.error = f"Document retrieval failed ({type(exc).__name__}): {exc}"
    return result
