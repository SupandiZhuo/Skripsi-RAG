"""Offline fixtures below are synthetic unit cases, not thesis ground truth."""
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate_retrieval import (
    evaluate, load_labels, load_rankings, main, select_threshold,
)


METHOD = "fixture_cosine"


def label(query_id="q1", relevant=None, split="validation", scenario_id="s1"):
    return {
        "scenario_id": scenario_id, "query_id": query_id, "query": query_id,
        "asset_id": "asset-A", "split": split,
        "relevant_document_ids": ["d1", "d2"] if relevant is None else relevant,
    }


def candidate(doc, score, asset="asset-A", **kwargs):
    return dict(document_id=doc, asset_id=asset, score=score,
                score_method=METHOD, **kwargs)


def ranking(item, candidates, error=None):
    return {key: item[key] for key in ("scenario_id", "query_id", "query", "asset_id")} | {
        "candidates": candidates, "error": error,
    }


class RetrievalEvaluationTests(unittest.TestCase):
    def report(self, labels, rankings, **kwargs):
        return evaluate(labels, rankings, split="validation", top_k=2,
                        threshold=0.5, score_method=METHOD, **kwargs)

    def test_document_metrics_deduplicate_chunks_and_keep_full_mrr(self):
        item = label()
        result = self.report([item], [ranking(item, [
            candidate("x", .9), candidate("x", .8), candidate("d1", .7),
            candidate("d2", .6),
        ])])
        metrics = result["metrics"]
        self.assertEqual(metrics["hit_at_1"]["value"], 0)
        self.assertEqual(metrics["hit_at_k"]["value"], 1)
        self.assertEqual(metrics["recall_at_k"]["value"], .5)
        self.assertEqual(metrics["precision_at_k"]["value"], .5)
        self.assertEqual(metrics["mrr"]["value"], .5)
        self.assertEqual(result["queries"][0]["first_relevant_rank"], 2)
        self.assertEqual(len(result["queries"][0]["candidates"]), 4)

    def test_mrr_beyond_k_is_not_truncated(self):
        item = label(relevant=["d1"])
        result = self.report([item], [ranking(item, [
            candidate("x", .9), candidate("y", .8), candidate("d1", .7),
        ])])
        self.assertAlmostEqual(result["metrics"]["mrr"]["value"], 1 / 3)
        self.assertEqual(result["metrics"]["hit_at_k"]["value"], 0)

    def test_wrong_asset_and_invalid_scores_do_not_count(self):
        item = label(relevant=["d1"])
        result = self.report([item], [ranking(item, [
            candidate("d1", .9, asset="asset-B"), candidate("d1", None),
            candidate("x", .6),
        ])])
        self.assertEqual(result["metrics"]["hit_at_k"]["value"], 0)
        rows = result["queries"][0]["candidates"]
        self.assertEqual(rows[0]["excluded_reason"], "asset_id_mismatch")
        self.assertFalse(rows[0]["relevant"])
        self.assertEqual(rows[1]["excluded_reason"], "invalid_score")

    def test_threshold_metrics_have_explicit_document_and_query_denominators(self):
        positive, negative = label(relevant=["d1"]), label("q2", relevant=[])
        result = self.report([positive, negative], [
            ranking(positive, [candidate("d1", .5), candidate("x", .2)]),
            ranking(negative, [candidate("x", .8)]),
        ])
        metrics = result["metrics"]
        self.assertEqual(metrics["accepted_precision"], {
            "numerator": 1, "denominator": 2, "value": .5,
        })
        self.assertEqual(metrics["correct_positive_recall"]["value"], 1)
        self.assertEqual(metrics["negative_rejection_rate"]["value"], 0)
        self.assertEqual(metrics["precision_at_k"]["value"], .5)
        self.assertEqual(metrics["precision_at_k"]["denominator"], 1)

    def test_missing_rows_and_retrieval_errors_are_not_successful_rejections(self):
        positive, negative = label(), label("q2", relevant=[])
        result = self.report([positive, negative], [
            ranking(negative, [], error="storage unavailable"),
        ])
        self.assertEqual(result["counts"]["retrieval_errors"], 2)
        self.assertEqual(result["metrics"]["negative_rejection_rate"]["value"], 0)
        self.assertEqual(result["metrics"]["hit_at_k"]["value"], 0)
        self.assertIsNone(result["metrics"]["accepted_precision"]["value"])

    def test_retrieval_error_discards_partial_candidates(self):
        item = label()
        result = self.report([item], [ranking(item, [candidate("d1", .9)], error="failed")])
        self.assertEqual(result["metrics"]["hit_at_k"]["value"], 0)
        self.assertEqual(result["queries"][0]["candidates"][0]["excluded_reason"], "retrieval_failed")

    def test_provenance_rejection_is_not_overridden_by_threshold_selection(self):
        item = label()
        result = self.report([item], [ranking(item, [
            candidate("d1", .9, rejection_reason="invalid_provenance"),
            candidate("d2", .8, rejection_reason="below_relevance_threshold"),
        ])])
        self.assertEqual(result["metrics"]["accepted_precision"]["value"], 1)
        self.assertEqual(result["queries"][0]["accepted_document_ids"], ["d2"])

    def test_mixed_score_methods_and_mismatched_query_are_errors(self):
        item = label()
        mixed = candidate("d1", .9)
        mixed["score_method"] = "other"
        with self.assertRaises(ValueError):
            self.report([item], [ranking(item, [mixed])])
        row = ranking(item, [])
        row["query"] = "different"
        with self.assertRaises(ValueError):
            self.report([item], [row])

    def test_selection_is_validation_only_and_test_uses_fixed_threshold(self):
        item = label(relevant=["d1"])
        rows = [ranking(item, [candidate("d1", .8), candidate("x", .4)])]
        selected = select_threshold([item], rows, split="validation", top_k=2,
                                    score_method=METHOD, thresholds=[.3, .7, .9],
                                    min_accepted_precision=1)
        self.assertEqual(selected["selected_threshold"], .7)
        with self.assertRaises(ValueError):
            select_threshold([label(split="test")], [], split="test", top_k=2,
                             score_method=METHOD, thresholds=[.5], min_accepted_precision=.9)
        with self.assertRaises(ValueError):
            select_threshold([item], rows, split="validation", top_k=2,
                             score_method=METHOD, thresholds=[.9], min_accepted_precision=1)

    def test_labels_require_independent_explicit_judgments_and_unique_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            invalid_rows = []
            missing = label()
            del missing["relevant_document_ids"]
            invalid_rows.append([missing])
            invalid_rows.append([label(relevant="d1")])
            invalid_rows.append([label(relevant=["d1", "d1"])])
            invalid_rows.append([label(), label()])
            invalid_rows.append([label(), label("q2", split="test")])
            invalid_rows.append([label(), label("q2", split="test", scenario_id="s2") | {"query": "q1"}])
            for rows in invalid_rows:
                path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    load_labels(path)

    def test_input_rank_order_is_preserved_and_validated(self):
        item = label()
        with self.assertRaises(ValueError):
            self.report([item], [ranking(item, [candidate("d1", .2), candidate("x", .9)])])
        with self.assertRaises(ValueError):
            self.report([item], [ranking(item, [candidate("d1", .9, rank=2)])])

    def test_undefined_metrics_are_null_and_invalid_configuration_rejected(self):
        item = label(relevant=[])
        result = self.report([item], [ranking(item, [])])
        self.assertIsNone(result["metrics"]["mrr"]["value"])
        self.assertEqual(result["metrics"]["negative_rejection_rate"]["value"], 1)
        for kwargs in ({"top_k": 0}, {"threshold": float("nan")}, {"threshold": True}):
            config = dict(split="validation", top_k=2, threshold=.5, score_method=METHOD)
            config.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate([item], [ranking(item, [])], **config)

    def test_cli_offline_writes_full_report_without_live_dependencies(self):
        item = label()
        with tempfile.TemporaryDirectory() as tmp:
            labels, ranked, output = (Path(tmp) / name for name in ("labels.jsonl", "ranked.jsonl", "report.json"))
            labels.write_text(json.dumps(item) + "\n", encoding="utf-8")
            ranked.write_text(json.dumps(ranking(item, [candidate("d1", .9)])) + "\n", encoding="utf-8")
            self.assertEqual(len(load_rankings(ranked)), 1)
            status = main(["--labels", str(labels), "--ranked-input", str(ranked),
                           "--output", str(output), "--split", "validation",
                           "--score-method", METHOD, "--top-k", "2", "--threshold", ".5"])
            self.assertEqual(status, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["config"]["mode"], "offline")
            self.assertEqual(report["queries"][0]["candidates"][0]["score_method"], METHOD)
            self.assertIn("sha256", report["config"]["inputs"]["labels"])

    def test_cli_rejects_threshold_tuning_on_test_split(self):
        with self.assertRaises(SystemExit):
            main(["--labels", "unused", "--ranked-input", "unused", "--output", "unused",
                  "--split", "test", "--score-method", METHOD, "--select-threshold",
                  "--threshold-grid", ".1,.5"])

    def test_validation_selection_does_not_use_test_outcomes(self):
        validation = label(relevant=["d1"])
        test = label("q2", relevant=[], split="test", scenario_id="s2")
        rows = [ranking(validation, [candidate("d1", .6)]),
                ranking(test, [candidate("x", .9)])]
        selected = select_threshold([validation, test], rows, split="validation", top_k=2,
                                    score_method=METHOD, thresholds=[.5, .8], min_accepted_precision=1)
        self.assertEqual(selected["selected_threshold"], .5)
        self.assertEqual(selected["sweep"][0]["counts"]["queries"], 1)
        self.assertIsNone(selected["sweep"][0]["metrics"]["negative_rejection_rate"]["value"])

    def test_unlabeled_and_duplicate_ranking_rows_are_rejected(self):
        item = label()
        for rows in ([ranking(label("unknown"), [])],
                     [ranking(item, []), ranking(item, [])]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.report([item], rows)

    def test_cli_preserves_existing_output_and_reports_retrieval_failures(self):
        item = label()
        with tempfile.TemporaryDirectory() as tmp:
            labels, ranked, output = (Path(tmp) / name for name in ("labels.jsonl", "ranked.jsonl", "report.json"))
            labels.write_text(json.dumps(item) + "\n", encoding="utf-8")
            ranked.write_text(json.dumps(ranking(item, [], error="unavailable")) + "\n", encoding="utf-8")
            args = ["--labels", str(labels), "--ranked-input", str(ranked),
                    "--output", str(output), "--split", "validation",
                    "--score-method", METHOD, "--threshold", ".5"]
            self.assertEqual(main(args), 1)
            original = output.read_text(encoding="utf-8")
            with self.assertRaises(SystemExit):
                main(args)
            self.assertEqual(output.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
