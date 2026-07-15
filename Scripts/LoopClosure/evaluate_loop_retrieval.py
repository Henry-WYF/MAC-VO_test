from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def pair_id(query: dict[str, Any], candidate: dict[str, Any]) -> str:
    return f"{int(query['loop_frame_idx'])}:{int(candidate['loop_frame_idx'])}"


def freeze(review_path: Path, output: Path) -> None:
    review = load_json(review_path)
    items = review.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("review file contains no items")
    allowed = {"true", "false", "uncertain"}
    seen: set[str] = set()
    for item in items:
        current_pair = str(item.get("pair_id"))
        if current_pair in seen:
            raise ValueError(f"duplicate reviewed pair {current_pair}")
        seen.add(current_pair)
        if item.get("label") not in allowed:
            raise ValueError(f"pair {current_pair} has not been assigned a valid review label")
    frozen = dict(review)
    frozen["frozen"] = True
    frozen["review_source_sha256"] = sha256(review_path)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, indent=2, allow_nan=False), encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="utf-8"
    )


def query_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if int(payload.get("schema_version", -1)) not in {1, 2}:
        raise ValueError("unsupported queries schema")
    result: dict[int, dict[str, Any]] = {}
    for query in payload.get("queries", []):
        loop_idx = int(query["loop_frame_idx"])
        if loop_idx in result:
            raise ValueError(f"duplicate query loop frame {loop_idx}")
        result[loop_idx] = query
    return result


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def filtered_candidates(query: dict[str, Any], threshold: float | None) -> list[dict[str, Any]]:
    candidates = list(query.get("candidates", []))
    if threshold is not None:
        candidates = [item for item in candidates if float(item["score"]) >= threshold]
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["sensor_frame_idx"])))
    return candidates[:10]


def compute_metrics(
    queries: dict[int, dict[str, Any]], fixed_query_ids: list[int],
    labels: dict[str, str], threshold: float | None,
) -> dict[str, Any]:
    true_universe = {
        pair for pair, label in labels.items()
        if label == "true" and int(pair.split(":", 1)[0]) in fixed_query_ids
    }
    returned_labeled: set[str] = set()
    false_count = 0
    no_candidate_queries = 0
    ranked: dict[int, list[str]] = {}
    for query_id in fixed_query_ids:
        query = queries[query_id]
        candidates = filtered_candidates(query, threshold)
        if not candidates:
            no_candidate_queries += 1
        ranked[query_id] = [pair_id(query, candidate) for candidate in candidates]
        for current_pair in ranked[query_id]:
            label = labels[current_pair]
            if label in {"true", "false"}:
                returned_labeled.add(current_pair)
            if label == "false":
                false_count += 1

    tp = sum(labels[pair] == "true" for pair in returned_labeled)
    fp = sum(labels[pair] == "false" for pair in returned_labeled)
    fn = len(true_universe - returned_labeled)
    precision = safe_ratio(tp, tp + fp)
    pair_recall = safe_ratio(tp, tp + fn)
    f1 = None
    if precision is not None and pair_recall is not None:
        f1 = (
            0.0 if precision + pair_recall == 0.0
            else 2.0 * precision * pair_recall / (precision + pair_recall)
        )

    positive_queries = sorted({int(pair.split(":", 1)[0]) for pair in true_universe})
    query_recall: dict[str, float | None] = {}
    for k in (1, 5, 10):
        hits = sum(
            any(labels[current_pair] == "true" for current_pair in ranked[query_id][:k])
            for query_id in positive_queries
        )
        query_recall[str(k)] = safe_ratio(hits, len(positive_queries))
    return {
        "threshold": threshold,
        "fixed_query_count": len(fixed_query_ids),
        "reviewed_true_pair_count": len(true_universe),
        "positive_query_count": len(positive_queries),
        "evaluation_set_sufficient": len(positive_queries) >= 10 and len(true_universe) >= 20,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_micro": precision,
        "pair_recall_micro": pair_recall,
        "f1_micro": f1,
        "query_recall_at_k": query_recall,
        "false_candidates_total": false_count,
        "false_candidates_per_fixed_query": safe_ratio(false_count, len(fixed_query_ids)),
        "no_candidate_queries": no_candidate_queries,
    }


def common_query_ids(
    custom: dict[int, dict[str, Any]], dbow2: dict[int, dict[str, Any]], labels: dict[str, str],
) -> list[int]:
    fixed: list[int] = []
    for query_id in sorted(set(custom) & set(dbow2)):
        lhs, rhs = custom[query_id], dbow2[query_id]
        lhs_empty = bool(lhs.get("empty_descriptors", int(lhs.get("orb_features", 0)) == 0))
        rhs_empty = bool(rhs.get("empty_descriptors", int(rhs.get("orb_features", 0)) == 0))
        if lhs_empty != rhs_empty:
            raise ValueError(f"backend descriptor state differs for query {query_id}")
        lhs_eligible = lhs.get("eligible_database_entries")
        rhs_eligible = rhs.get("eligible_database_entries")
        if lhs_eligible != rhs_eligible:
            raise ValueError(f"backend eligible database size differs for query {query_id}")
        if lhs_empty or lhs_eligible is None or int(lhs_eligible) <= 0:
            continue
        candidate_pairs = {
            pair_id(query, candidate)
            for query in (lhs, rhs) for candidate in query.get("candidates", [])
        }
        if all(current_pair in labels for current_pair in candidate_pairs):
            fixed.append(query_id)
    return fixed


def evaluate(
    labels_path: Path, custom_path: Path, dbow2_path: Path,
    output: Path, frozen_threshold: float | None,
) -> None:
    frozen = load_json(labels_path)
    if frozen.get("frozen") is not True:
        raise ValueError("labels have not been frozen")
    if frozen.get("review_protocol_version") != "macvo-loop-overlap-review-v1":
        raise ValueError("unsupported loop-overlap review protocol")
    if frozen.get("pair_match_policy") != "exact loop_frame_idx pair":
        raise ValueError("formal retrieval metrics require exact cached-frame pair labels")
    expected = frozen.get("input_sha256", {})
    if expected.get("custom_raw_queries") != sha256(custom_path):
        raise ValueError("custom raw queries do not match the frozen review input")
    if expected.get("dbow2_raw_queries") != sha256(dbow2_path):
        raise ValueError("DBoW2 raw queries do not match the frozen review input")
    labels = {str(item["pair_id"]): str(item["label"]) for item in frozen["items"]}
    custom = query_map(load_json(custom_path))
    dbow2 = query_map(load_json(dbow2_path))
    fixed_ids = common_query_ids(custom, dbow2, labels)
    custom_metrics = compute_metrics(custom, fixed_ids, labels, None)

    selected_threshold = frozen_threshold
    threshold_search: list[dict[str, Any]] | None = None
    if selected_threshold is None:
        thresholds = {0.0}
        thresholds.update(
            float(candidate["score"])
            for query_id in fixed_ids for candidate in dbow2[query_id].get("candidates", [])
            if math.isfinite(float(candidate["score"]))
        )
        threshold_search = [compute_metrics(dbow2, fixed_ids, labels, value) for value in sorted(thresholds)]
        eligible = [item for item in threshold_search if item["f1_micro"] is not None]
        if eligible:
            best = max(eligible, key=lambda item: (
                item["f1_micro"],
                item["query_recall_at_k"]["10"] if item["query_recall_at_k"]["10"] is not None else -1.0,
                item["precision_micro"] if item["precision_micro"] is not None else -1.0,
                item["threshold"],
            ))
            selected_threshold = float(best["threshold"])
    dbow2_metrics = compute_metrics(dbow2, fixed_ids, labels, selected_threshold)
    payload = {
        "schema_version": 1,
        "recall_scope": "Recall within the frozen reviewed overlap evaluation set",
        "pair_match_policy": "exact loop_frame_idx pair",
        "labels_sha256": sha256(labels_path),
        "custom_raw_sha256": sha256(custom_path),
        "dbow2_raw_sha256": sha256(dbow2_path),
        "fixed_query_ids": fixed_ids,
        "selected_dbow2_threshold": selected_threshold,
        "custom": custom_metrics,
        "dbow2": dbow2_metrics,
        "threshold_search": threshold_search,
        "per_dataset_promotion_checks": {
            "evaluation_set_sufficient": custom_metrics["evaluation_set_sufficient"],
            "query_recall_at_10_not_lower": (
                None if custom_metrics["query_recall_at_k"]["10"] is None or dbow2_metrics["query_recall_at_k"]["10"] is None
                else dbow2_metrics["query_recall_at_k"]["10"] >= custom_metrics["query_recall_at_k"]["10"]
            ),
            "false_candidates_per_query_strictly_lower": (
                None if custom_metrics["false_candidates_per_fixed_query"] is None or dbow2_metrics["false_candidates_per_fixed_query"] is None
                else dbow2_metrics["false_candidates_per_fixed_query"] < custom_metrics["false_candidates_per_fixed_query"]
            ),
        },
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze or evaluate MAC-VO loop-overlap labels.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--review", required=True, type=Path)
    freeze_parser.add_argument("--output", required=True, type=Path)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--labels", required=True, type=Path)
    evaluate_parser.add_argument("--custom-queries", required=True, type=Path)
    evaluate_parser.add_argument("--dbow2-queries", required=True, type=Path)
    evaluate_parser.add_argument("--output", required=True, type=Path)
    evaluate_parser.add_argument("--threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        freeze(args.review, args.output)
    else:
        if args.threshold is not None and (
            not math.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0
        ):
            raise ValueError("threshold must be finite and in [0,1]")
        evaluate(args.labels, args.custom_queries, args.dbow2_queries, args.output, args.threshold)


if __name__ == "__main__":
    main()
