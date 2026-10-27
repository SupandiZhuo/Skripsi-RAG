"""Document retrieval evaluation, independent of the assessment batch runner.

Run with ``python -m evaluation.evaluate_retrieval`` (or this file directly).
Labels JSONL, one explicitly annotated query per line::

    {"scenario_id":"s1", "query_id":"q1", "query":"dependency evidence",
     "asset_id":"asset-A", "split":"validation", "relevant_document_ids":["doc-1"]}

An explicit empty relevant_document_ids list denotes an annotated negative query;
missing labels are errors, never inferred negatives. Document IDs are scoped to
asset_id. Do not use this example as research ground truth.

Offline ranked JSONL rows repeat scenario_id/query_id/query/asset_id and contain
``candidates`` (in descending score order) and optional ``error``. Each candidate
requires document_id, asset_id, score, score_method; optional rank must equal its
1-based input position. Optional rejection_reason preserves provenance failures;
only below_relevance_threshold is recomputed. Duplicate document chunks collapse
to the first eligible occurrence. All original candidates remain in the report.

Ranking metrics are macro averages over positive queries, including failures as
zero; Precision@K divides by K even when fewer documents are returned. MRR uses
the entire supplied ranking, not only K (live depth is candidate_k). Threshold
accepted_precision is document-micro precision within K; correct_positive_recall
is the fraction of positive queries with >=1 accepted relevant document; negative
rejection rate is the fraction of negative queries with no accepted documents.
Retrieval errors cannot count as successful negative rejections. Zero denominators
produce JSON null. Scores are not calibrated probabilities.

Threshold selection is validation-only: maximize correct_positive_recall subject
to minimum accepted_precision, then negative_rejection_rate, accepted_precision,
and finally the higher threshold. Test runs require an explicit frozen threshold.
Split checks prevent scenarios and identical normalized query/asset pairs from
crossing validation/test within the supplied label file; independence across
separate label files remains the investigator's responsibility.

Live mode requires --chroma-path and --embedding-config JSON containing a local
model_name directory plus optional model_kwargs/encode_kwargs. Model downloads
are disabled. The existing collection is obtained with get_collection, never
created, ingested, updated or deleted. Chroma PersistentClient can nevertheless
perform internal housekeeping writes: use a separately prepared read replica if
filesystem-level immutability is required. No live libraries load in offline mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


LIVE_SCORE_METHOD = "chroma_cosine_relevance"
COLLECTION = "organizational_context"
IDENTITY = ("scenario_id", "query_id")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


def _number(value: Any, name: str, low: float = 0, high: float = 1) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    return float(value)


def _key(row: dict) -> tuple[str, str]:
    return (_text(row.get("scenario_id"), "scenario_id"),
            _text(row.get("query_id"), "query_id"))


def _jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def _validate_labels(rows: list[dict]) -> None:
    seen, scenarios, queries = set(), {}, {}
    if not rows:
        raise ValueError("At least one labeled query is required")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each label must be an object")
        key = _key(row)
        if key in seen:
            raise ValueError(f"Duplicate scenario/query identity: {key}")
        seen.add(key)
        query, asset = _text(row.get("query"), "query"), _text(row.get("asset_id"), "asset_id")
        split = row.get("split")
        if split not in ("validation", "test"):
            raise ValueError(f"{key}: split must explicitly be validation or test")
        relevant = row.get("relevant_document_ids")
        if not isinstance(relevant, list):
            raise ValueError(f"{key}: explicit relevant_document_ids list required")
        for document_id in relevant:
            _text(document_id, "relevant_document_ids entry")
        if len(set(relevant)) != len(relevant):
            raise ValueError(f"{key}: duplicate relevant document IDs")
        query_key = (asset, " ".join(query.casefold().split()))
        for registry, token in ((scenarios, row["scenario_id"]), (queries, query_key)):
            if token in registry and registry[token] != split:
                raise ValueError(f"Split leakage: scenario or normalized asset/query appears across splits ({key})")
            registry[token] = split


def load_labels(path: str | Path) -> list[dict]:
    rows = _jsonl(path)
    _validate_labels(rows)
    return rows


def load_rankings(path: str | Path) -> list[dict]:
    return _jsonl(path)


def _indexed_rankings(rows: list[dict], labels: list[dict], score_method: str) -> dict:
    known = {_key(row): row for row in labels}
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each ranking must be an object")
        key = _key(row)
        if key not in known:
            raise ValueError(f"Unlabeled ranking identity: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate ranking identity: {key}")
        for field in ("query", "asset_id"):
            if row.get(field) != known[key][field]:
                raise ValueError(f"{key}: ranking {field} does not match label")
        error = row.get("error")
        if error is not None:
            _text(error, "error")
        if not isinstance(row.get("candidates"), list):
            raise ValueError(f"{key}: candidates must be a list (empty is allowed)")
        previous_score = math.inf
        for rank, candidate in enumerate(row["candidates"], 1):
            if not isinstance(candidate, dict):
                raise ValueError(f"{key}: candidate must be an object")
            _text(candidate.get("document_id"), "document_id")
            _text(candidate.get("asset_id"), "candidate asset_id")
            if candidate.get("score_method") != score_method:
                raise ValueError(f"{key}: missing/mixed score_method; expected {score_method!r}")
            if "score" not in candidate:
                raise ValueError(f"{key}: candidate score is required (null marks unavailable)")
            if "rank" in candidate and (type(candidate["rank"]) is not int or candidate["rank"] != rank):
                raise ValueError(f"{key}: candidate rank must equal its 1-based input position")
            reason = candidate.get("rejection_reason")
            if reason is not None:
                _text(reason, "rejection_reason")
            score = candidate["score"]
            try:
                valid_score = _number(score, "score", -1, 1)
            except ValueError:
                continue  # Recorded as invalid_score, never accepted or ranked.
            if valid_score > previous_score:
                raise ValueError(f"{key}: candidates must be in descending score order")
            previous_score = valid_score
        indexed[key] = row
    return indexed


def _fraction(numerator: float, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def evaluate(labels: list[dict], rankings: list[dict], *, split: str,
             top_k: int, threshold: float, score_method: str) -> dict:
    """Pure offline evaluation; input validation never guesses missing labels."""
    _validate_labels(labels)
    if split not in ("validation", "test"):
        raise ValueError("Explicit validation/test split required")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    threshold = _number(threshold, "threshold")
    _text(score_method, "score_method")
    indexed = _indexed_rankings(rankings, labels, score_method)
    selected = [row for row in labels if row["split"] == split]
    if not selected:
        raise ValueError(f"No labels for split {split!r}")
    sums = dict(hit_at_1=0, hit_at_k=0, recall_at_k=0, precision_at_k=0, mrr=0)
    positives = negatives = correct_queries = rejected_negatives = accepted_count = correct_accepted = errors = 0
    output = []
    for label in selected:
        key = _key(label)
        retrieval = indexed.get(key)
        error = retrieval.get("error") if retrieval is not None else "missing_ranked_input"
        errors += bool(error)
        relevant = set(label["relevant_document_ids"])
        seen, documents, candidate_rows = set(), [], []
        for input_rank, source in enumerate(retrieval["candidates"] if retrieval is not None else [], 1):
            candidate = dict(source)
            score = candidate["score"]
            reason = None
            try:
                score = _number(score, "score", -1, 1)
            except ValueError:
                score, reason = None, "invalid_score"
            if error:
                reason = "retrieval_failed"
            elif candidate["asset_id"] != label["asset_id"]:
                reason = "asset_id_mismatch"
            elif candidate.get("rejection_reason") not in (None, "below_relevance_threshold"):
                reason = candidate["rejection_reason"]
            document_id = candidate["document_id"]
            document_rank = None
            if not reason:
                if document_id in seen:
                    reason = "duplicate_document"
                else:
                    seen.add(document_id)
                    documents.append({"document_id": document_id, "score": score})
                    document_rank = len(documents)
            accepted = bool(document_rank and document_rank <= top_k and score is not None and score >= threshold)
            candidate.update(input_rank=input_rank, document_rank=document_rank, score=score,
                             relevant=candidate["asset_id"] == label["asset_id"] and document_id in relevant,
                             accepted=accepted,
                             excluded_reason=reason,
                             threshold_rejection=("below_relevance_threshold" if document_rank and score is not None and score < threshold
                                                  else "outside_top_k" if document_rank and document_rank > top_k else None))
            candidate_rows.append(candidate)
        top = documents[:top_k]
        accepted = [row["document_id"] for row in top if row["score"] >= threshold]
        hits = len({row["document_id"] for row in top} & relevant)
        first_rank = next((rank for rank, row in enumerate(documents, 1) if row["document_id"] in relevant), None)
        per_query = {
            "hit_at_1": int(bool(documents) and documents[0]["document_id"] in relevant) if relevant else None,
            "hit_at_k": int(hits > 0) if relevant else None,
            "recall_at_k": hits / len(relevant) if relevant else None,
            "precision_at_k": hits / top_k if relevant else None,
            "mrr": (1 / first_rank if first_rank else 0) if relevant else None,
        }
        if relevant:
            positives += 1
            for name, value in per_query.items():
                sums[name] += value
            correct_queries += bool(set(accepted) & relevant)
        else:
            negatives += 1
            rejected_negatives += not error and not accepted
        accepted_count += len(accepted)
        correct_accepted += len(set(accepted) & relevant)
        output.append({**{field: label[field] for field in (*IDENTITY, "query", "asset_id", "split")},
                       "relevant_document_ids": label["relevant_document_ids"], "error": error,
                       "candidates": candidate_rows, "document_ranking": documents,
                       "first_relevant_rank": first_rank, "accepted_document_ids": accepted,
                       "metrics": per_query})
    metrics = {name: _fraction(value, positives) for name, value in sums.items()}
    metrics.update(accepted_precision=_fraction(correct_accepted, accepted_count),
                   correct_positive_recall=_fraction(correct_queries, positives),
                   negative_rejection_rate=_fraction(rejected_negatives, negatives))
    return {
        "schema_version": 1,
        "config": {"split": split, "top_k": top_k, "threshold": threshold, "score_method": score_method,
                   "ranking_population": "positive_queries_including_errors",
                   "precision_denominator_per_query": top_k,
                   "mrr_depth": "all_eligible_unique_documents_in_supplied_ranking",
                   "document_identity": "asset_id + document_id",
                   "acceptance": "eligible_unique_document_in_top_k_with_score_gte_threshold",
                   "threshold_precision_unit": "accepted_documents_micro",
                   "correct_positive_recall_unit": "positive_queries_with_any_correct_acceptance",
                   "negative_rejection_errors_count_as_success": False,
                   "zero_denominator": None},
        "counts": {"queries": len(selected), "positive_queries": positives, "negative_queries": negatives,
                   "retrieval_errors": errors, "accepted_documents": accepted_count,
                   "correct_accepted_documents": correct_accepted},
        "metrics": metrics, "queries": output,
    }


def select_threshold(labels: list[dict], rankings: list[dict], *, split: str,
                     top_k: int, score_method: str, thresholds: list[float],
                     min_accepted_precision: float) -> dict:
    """Return validation sweep metadata; test labels never participate in tuning."""
    if split != "validation":
        raise ValueError("Threshold selection is allowed only on validation, never test")
    minimum = _number(min_accepted_precision, "min_accepted_precision")
    if not thresholds:
        raise ValueError("An explicit nonempty threshold grid is required")
    thresholds = sorted({_number(value, "grid threshold") for value in thresholds})
    sweep, feasible = [], []
    for threshold in thresholds:
        report = evaluate(labels, rankings, split=split, top_k=top_k, threshold=threshold, score_method=score_method)
        metrics = report["metrics"]
        precision = metrics["accepted_precision"]["value"]
        recall = metrics["correct_positive_recall"]["value"]
        eligible = precision is not None and precision >= minimum and recall is not None
        row = {"threshold": threshold, "metrics": metrics, "counts": report["counts"], "feasible": eligible}
        sweep.append(row)
        if eligible:
            rejection = metrics["negative_rejection_rate"]["value"]
            feasible.append(((recall, rejection if rejection is not None else -1, precision, threshold), threshold))
    if not feasible:
        raise ValueError("No validation threshold meets minimum accepted precision with defined positive-query recall")
    return {"selection_split": "validation", "selected_threshold": max(feasible)[1],
            "min_accepted_precision": minimum,
            "objective": "max correct_positive_recall; tie: negative_rejection_rate, accepted_precision, threshold",
            "sweep": sweep, "test_evaluated_for_selection": False}


def _live_rankings(labels: list[dict], *, split: str, path: str,
                   embedding_config: str, candidate_k: int) -> tuple[list[dict], dict]:
    """Lazy, existing-collection-only Chroma adapter for the shared retriever."""
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir() or not (directory / "chroma.sqlite3").is_file():
        raise ValueError("--chroma-path must contain an existing Chroma database")
    config = json.loads(Path(embedding_config).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) - {"model_name", "model_kwargs", "encode_kwargs"}:
        raise ValueError("Embedding config supports model_name, model_kwargs, encode_kwargs only")
    model_name = _text(config.get("model_name"), "embedding model_name")
    model_path = Path(model_name).expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError("Live embedding model_name must name an existing local model directory; downloads disabled")
    model_kwargs, encode_kwargs = config.get("model_kwargs", {}), config.get("encode_kwargs", {})
    if not isinstance(model_kwargs, dict) or not isinstance(encode_kwargs, dict):
        raise ValueError("model_kwargs and encode_kwargs must be objects")
    if model_kwargs.get("trust_remote_code"):
        raise ValueError("trust_remote_code is not permitted")
    model_kwargs = {**model_kwargs, "local_files_only": True, "trust_remote_code": False}
    import chromadb
    from langchain_huggingface import HuggingFaceEmbeddings
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rag_retrieval import retrieve_document_context

    client = chromadb.PersistentClient(path=str(directory))
    collection = client.get_collection(name=COLLECTION, embedding_function=None)
    if (collection.metadata or {}).get("hnsw:space") != "cosine":
        raise ValueError("Existing organizational_context collection must use cosine distance")
    embeddings = HuggingFaceEmbeddings(model_name=str(model_path), model_kwargs=model_kwargs,
                                      encode_kwargs=encode_kwargs)

    class ExistingCollectionStore:
        _collection = collection

        def similarity_search_with_relevance_scores(self, query, k, filter):
            result = collection.query(query_embeddings=[embeddings.embed_query(query)],
                                      n_results=k, where=filter,
                                      include=["documents", "metadatas", "distances"])
            return [(SimpleNamespace(page_content=text, metadata=metadata), 1.0 - distance)
                    for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0],
                                                        result["distances"][0])]

    store, rows = ExistingCollectionStore(), []
    for label in labels:
        if label["split"] != split:
            continue
        result = retrieve_document_context(store, label["query"], label["asset_id"],
                                           top_k=candidate_k, relevance_threshold=0)
        candidates = []
        for evidence in result.candidates:
            metadata = evidence.metadata
            # Missing identity is an audited retrieval failure, never guessed from chunk IDs.
            if not metadata.get("document_id") or not metadata.get("asset_id"):
                result.error = result.error or "candidate_missing_document_or_asset_identity"
                continue
            candidates.append({"document_id": metadata["document_id"], "asset_id": metadata["asset_id"],
                               "score": evidence.relevance_score, "score_method": evidence.scoring_method,
                               "rejection_reason": evidence.rejection_reason,
                               "evidence_id": evidence.evidence_id, "metadata": metadata})
        candidates.sort(key=lambda row: row["score"] if row["score"] is not None else -math.inf, reverse=True)
        rows.append({**{field: label[field] for field in (*IDENTITY, "query", "asset_id")},
                     "candidates": candidates, "error": result.error})
    return rows, {"embedding_model": str(model_path), "embedding_config": {
        "model_name": str(model_path), "model_kwargs": model_kwargs, "encode_kwargs": encode_kwargs},
        "chroma_path": str(directory), "collection": COLLECTION,
        "collection_metadata": collection.metadata, "candidate_k": candidate_k,
        "score_transform": "1 - cosine_distance", "collection_operations": "get_collection/query only",
        "filesystem_read_only_guaranteed": False}


def _fingerprint(path: str | Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ranked-input", help="Offline ranked candidates JSONL; no embedding imports")
    source.add_argument("--chroma-path", help="Existing persistent Chroma directory; no collection creation")
    parser.add_argument("--embedding-config", help="Live only: JSON config with local model_name path")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--score-method", required=True, help="Explicit score scale; must match every candidate")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, help="Live chunk retrieval depth, >= top-k; default top-k")
    parser.add_argument("--threshold", type=float, help="Explicit frozen threshold; required unless selecting")
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--threshold-grid", help="Validation-only comma-separated numeric thresholds")
    parser.add_argument("--min-accepted-precision", type=float, default=.9)
    parser.add_argument("--output", required=True, help="Report JSON; existing files are not overwritten")
    args = parser.parse_args(argv)
    if args.select_threshold and args.split != "validation":
        parser.error("--select-threshold is validation-only; test requires a frozen --threshold")
    if args.select_threshold and (not args.threshold_grid or args.threshold is not None):
        parser.error("Selection requires --threshold-grid and forbids --threshold")
    if not args.select_threshold and (args.threshold is None or args.threshold_grid is not None):
        parser.error("Evaluation requires --threshold; --threshold-grid requires --select-threshold")
    if args.chroma_path and (not args.embedding_config or args.score_method != LIVE_SCORE_METHOD):
        parser.error(f"Live retrieval requires --embedding-config and --score-method {LIVE_SCORE_METHOD}")
    if args.ranked_input and (args.embedding_config or args.candidate_k is not None):
        parser.error("--embedding-config and --candidate-k are live-only")
    try:
        if args.top_k < 1:
            raise ValueError("--top-k must be positive")
        if not args.select_threshold:
            _number(args.threshold, "threshold")
        thresholds = [float(value) for value in str(args.threshold_grid).split(",")] if args.select_threshold else []
        for value in thresholds:
            _number(value, "grid threshold")
        _number(args.min_accepted_precision, "min_accepted_precision")
        candidate_k = args.candidate_k if args.candidate_k is not None else args.top_k
        if candidate_k < args.top_k:
            raise ValueError("--candidate-k must be >= --top-k")
        output = Path(args.output)
        if output.exists():
            raise ValueError("Output already exists; choose a new report path (no overwrite)")
        labels = load_labels(args.labels)
        if not any(row["split"] == args.split for row in labels):
            raise ValueError(f"No labels for split {args.split!r}")
        inputs = {"labels": _fingerprint(args.labels)}
        if args.ranked_input:
            rankings, provenance = load_rankings(args.ranked_input), {"mode": "offline"}
            inputs["ranked_input"] = _fingerprint(args.ranked_input)
        else:
            inputs["embedding_config"] = _fingerprint(args.embedding_config)
            rankings, provenance = _live_rankings(labels, split=args.split, path=args.chroma_path,
                                                  embedding_config=args.embedding_config, candidate_k=candidate_k)
            provenance["mode"] = "live"
        selection = None
        if args.select_threshold:
            selection = select_threshold(labels, rankings, split=args.split, top_k=args.top_k,
                                         score_method=args.score_method, thresholds=thresholds,
                                         min_accepted_precision=args.min_accepted_precision)
        threshold = _number(selection["selected_threshold"] if selection else args.threshold, "threshold")
        report = evaluate(labels, rankings, split=args.split, top_k=args.top_k,
                          threshold=threshold, score_method=args.score_method)
        report["config"].update(provenance, inputs=inputs)
        report["threshold_selection"] = selection
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(f"Wrote {output}: {report['counts']['queries']} queries, {report['counts']['retrieval_errors']} retrieval errors")
        return 1 if report["counts"]["retrieval_errors"] else 0
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
