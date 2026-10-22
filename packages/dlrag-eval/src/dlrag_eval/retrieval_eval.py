"""Evaluate retrieval quality against section headers with normalized matching."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from dlrag_index.hybrid import HybridSearchEngine


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_index(artifacts: Path) -> HybridSearchEngine:
    return HybridSearchEngine.load(artifacts)


def normalize_header(header: str | None) -> str:
    """Strip markdown escape characters, whitespace, and lowercase for robust matching."""
    if not header:
        return ""
    cleaned = header.replace("\\", "").strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def is_match(retrieved_header: str, ground_truth_headers: set[str]) -> bool:
    norm_retrieved = normalize_header(retrieved_header)
    for gt in ground_truth_headers:
        norm_gt = normalize_header(gt)
        if norm_gt and (norm_gt == norm_retrieved or norm_gt in norm_retrieved):
            return True
    return False


def rank_of_first_relevant(ordered_headers: list[str], relevant_set: set[str]) -> int | None:
    for rank, h in enumerate(ordered_headers, start=1):
        if is_match(h, relevant_set):
            return rank
    return None


def evaluate(
    index: Any,
    benchmark: list[dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []

    # Create a fast lookup for chunk_id -> section_header
    # using the chunks stored on the index.
    chunk_to_header = {
        chunk.chunk_id: chunk.section_header
        for chunk in index.chunks
    }

    for item in benchmark:
        query = item["query"]
        relevant_headers = set(item.get("relevant_section_ids", []))

        # Retrieve the bundle containing fused hits and independent pools
        bundle = index.search_with_pools(query, top_k=top_k)

        # 1. Extract fused headers directly from the SearchHit objects
        retrieved_headers = [
            h.section_header for h in bundle.hits if h.section_header is not None
        ]

        # 2. Extract true independent BM25 top-K headers using the lookup
        bm25_headers = [
            chunk_to_header.get(chunk_id)
            for chunk_id, _score in bundle.bm25_pool[:top_k]
            if chunk_to_header.get(chunk_id) is not None
        ]

        # 3. Extract true independent Dense top-K headers using the lookup
        dense_headers = [
            chunk_to_header.get(chunk_id)
            for chunk_id, _score in bundle.dense_pool[:top_k]
            if chunk_to_header.get(chunk_id) is not None
        ]

        # Evaluate the first relevant rank for all three strategies independently
        bm25_first = rank_of_first_relevant(bm25_headers, relevant_headers)
        dense_first = rank_of_first_relevant(dense_headers, relevant_headers)
        fused_first = rank_of_first_relevant(retrieved_headers, relevant_headers)

        has_hit = fused_first is not None
        has_bm25_hit = bm25_first is not None
        has_dense_hit = dense_first is not None

        # Attribution reflects what the two retrievers independently found --
        # NOT whether the fused top-k happened to surface it. A doc BM25
        # ranks #1 can still miss the fused top-k if RRF prefers candidates
        # both retrievers agreed on; that's a fusion loss, not a BM25 miss,
        # and the two need different fixes (retrieval quality vs. rrf_k /
        # candidate_pool tuning).
        if has_bm25_hit and has_dense_hit:
            attribution = "both"
        elif has_bm25_hit:
            attribution = "bm25_only"
        elif has_dense_hit:
            attribution = "dense_only"
        else:
            attribution = "neither"

        # Separately: did fusion actually deliver what a retriever found?
        lost_to_fusion = (has_bm25_hit or has_dense_hit) and not has_hit

        # Calculate MRR and Recall
        mrr = 1.0 / fused_first if fused_first else 0.0
        bm25_mrr = 1.0 / bm25_first if bm25_first else 0.0
        dense_mrr = 1.0 / dense_first if dense_first else 0.0
        recall = 1.0 if has_hit else 0.0

        results.append({
            "id": item.get("id"),
            "query": query,
            "first_relevant_rank": fused_first,
            "bm25_first_relevant_rank": bm25_first,
            "dense_first_relevant_rank": dense_first,
            "attribution": attribution,
            "lost_to_fusion": lost_to_fusion,
            f"recall@{top_k}": recall,
            "mrr": mrr,
            "bm25_mrr@k": bm25_mrr,
            "dense_mrr@k": dense_mrr,
            "retrieved_headers": retrieved_headers,
        })

    n = len(results)
    if n == 0:
        return {}, []

    mean = lambda key: sum(r[key] for r in results) / n

    metrics: dict[str, Any] = {
        f"recall@{top_k}": mean(f"recall@{top_k}"),
        "mrr": mean("mrr"),
        "bm25_mrr@k": mean("bm25_mrr@k"),
        "dense_mrr@k": mean("dense_mrr@k"),
        "dense_uplift_mrr": mean("mrr") - mean("bm25_mrr@k"),
        "bm25_uplift_mrr": mean("mrr") - mean("dense_mrr@k"),
    }

    counts = {"both": 0, "bm25_only": 0, "dense_only": 0, "neither": 0}
    for r in results:
        counts[r["attribution"]] += 1

    metrics["attribution_counts"] = counts
    metrics["attribution_share"] = {k: round(v / n, 4) for k, v in counts.items()}
    metrics["lost_to_fusion_count"] = sum(1 for r in results if r["lost_to_fusion"])

    return metrics, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval performance.")
    parser.add_argument("--artifacts", type=Path, default=Path("data/rag_artifacts"))
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark/benchmark.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    index = load_index(args.artifacts)
    benchmark = load_json(args.benchmark)

    metrics, results = evaluate(index, benchmark, args.top_k)

    print("\n=== Retrieval Evaluation ===")
    print(f"Benchmark: {len(results)} queries | Top-k: {args.top_k}")
    for name in sorted(metrics):
        val = metrics[name]
        print(f"{name:<22} {val:.4f}" if isinstance(val, float) else f"{name:<22} {val}")

    print("\n=== Per-query results ===")
    for r in results:
        fmt = lambda x: str(x) if x is not None else "MISS"
        flag = " <-- LOST TO FUSION" if r["lost_to_fusion"] else ""
        print(
            f"{r['id']:<5} fus={fmt(r['first_relevant_rank']):<5} "
            f"bm25={fmt(r['bm25_first_relevant_rank']):<5} "
            f"dense={fmt(r['dense_first_relevant_rank']):<5} "
            f"[{r['attribution']:<10}] {r['query']}{flag}"
        )


if __name__ == "__main__":
    main()
