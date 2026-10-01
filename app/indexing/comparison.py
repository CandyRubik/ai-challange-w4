"""Compact, per-answer comparison of raw and active document retrieval."""

from __future__ import annotations

from .rag import RetrievalSettings, RetrievalTrace
from .store import DocumentIndex, SearchHit


def _source(hit: SearchHit, relevance: float | None = None) -> dict:
    return {
        "chunk_id": hit.metadata["chunk_id"],
        "section": hit.metadata["section"],
        "page": hit.metadata["page_start"],
        "source": hit.metadata["source"],
        "similarity": round(hit.score, 4),
        "relevance": round(relevance, 4) if relevance is not None else None,
    }


def retrieval_comparison(
    index: DocumentIndex, trace: RetrievalTrace,
    settings: RetrievalSettings, strategy: str,
) -> dict:
    """Show the exact active selection beside the raw top-K for this question."""
    raw_query = trace.original_question[:500]
    raw_hits = (
        list(trace.selected)
        if not settings.rewrite and not settings.filter_relevance
        else index.search(raw_query, strategy=strategy, k=settings.final_k)
    )
    scores = {
        hit.metadata["chunk_id"]: score
        for hit, score in zip(trace.candidates, trace.relevance_scores)
    }
    selected_ids = {hit.metadata["chunk_id"] for hit in trace.selected}
    rejected = []
    for i, hit in enumerate(trace.candidates):
        if hit.metadata["chunk_id"] in selected_ids:
            continue
        score = trace.relevance_scores[i] if trace.relevance_scores else None
        if trace.lexical_matches and not trace.lexical_matches[i]:
            reason = "identifier"
        elif score is not None and score < settings.min_relevance_score:
            reason = "threshold"
        else:
            reason = "top_k"
        rejected.append({**_source(hit, score), "reason": reason})
    return {
        "strategy": strategy,
        "raw": {
            "query": raw_query,
            "candidate_count": len(raw_hits),
            "hits": [_source(hit) for hit in raw_hits],
        },
        "active": {
            "query": trace.search_query,
            "rewrite": settings.rewrite,
            "filter_relevance": settings.filter_relevance,
            "candidate_k": settings.candidate_k,
            "candidate_count": len(trace.candidates),
            "final_k": settings.final_k,
            "threshold": settings.min_relevance_score if settings.filter_relevance else None,
            "hits": [_source(hit, scores.get(hit.metadata["chunk_id"]))
                     for hit in trace.selected],
            "rejected": rejected,
        },
    }
