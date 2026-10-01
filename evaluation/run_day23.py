"""Compare raw retrieval, query rewrite, relevance filtering, and their combination.

Usage: python -m evaluation.run_day23 --output data/day23-evaluation.json
Loads .env externally. Use --retrieval-only to inspect retrieval without LLM answers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from app.agents.agent import Agent
from app.indexing.rag import DocumentRag, RetrievalSettings, RetrievalTrace
from app.indexing.rerank import LocalCrossEncoderReranker
from app.indexing.store import DocumentIndex, SearchHit
from app.providers.deepseek import DeepSeekProvider
from app.services.chat_sessions import ChatSessionService
from app.storage.chat_sessions import SQLiteChatSessionRepository


QUESTIONS_PATH = Path(__file__).with_name("day22-questions.json")
NEGATIVES_PATH = Path(__file__).with_name("day23-hard-negatives.json")
DEFAULT_OUTPUT = Path("data/day23-evaluation.json")


def variants(threshold: float) -> dict[str, RetrievalSettings]:
    return {
        "raw": RetrievalSettings(
            rewrite=False, filter_relevance=False, candidate_k=3, final_k=3,
        ),
        "rewrite": RetrievalSettings(
            filter_relevance=False, candidate_k=3, final_k=3,
        ),
        "filter": RetrievalSettings(
            rewrite=False, candidate_k=10, final_k=3,
            min_relevance_score=threshold,
        ),
        "enhanced": RetrievalSettings(
            candidate_k=10, final_k=3, min_relevance_score=threshold,
        ),
    }


def expected_hit(case: dict, hit: SearchHit) -> bool:
    return (
        hit.metadata["section"] in case["sections"]
        and hit.metadata["page_start"] in case["pdf_pages"]
    )


def retrieval_record(case: dict, trace: RetrievalTrace) -> dict:
    selected_ids = {hit.metadata["chunk_id"] for hit in trace.selected}
    candidates = [
        {
            "chunk_id": hit.metadata["chunk_id"],
            "section": hit.metadata["section"],
            "pdf_page": hit.metadata["page_start"],
            "similarity": round(hit.score, 4),
            "relevance": round(trace.relevance_scores[i], 4) if trace.relevance_scores else None,
            "original_relevance": round(trace.original_scores[i], 4) if trace.original_scores else None,
            "contextual_relevance": round(trace.contextual_scores[i], 4) if trace.contextual_scores else None,
            "identifier_match": trace.lexical_matches[i] if trace.lexical_matches else None,
            "selected": hit.metadata["chunk_id"] in selected_ids,
        }
        for i, hit in enumerate(trace.candidates)
    ]
    return {
        "search_query": trace.search_query,
        "required_terms": list(trace.required_terms),
        "candidates": candidates,
        "selected_ids": [hit.metadata["chunk_id"] for hit in trace.selected],
        "expected_in_candidates": any(expected_hit(case, hit) for hit in trace.candidates)
        if case["sections"] else None,
        "expected_in_selected": any(expected_hit(case, hit) for hit in trace.selected)
        if case["sections"] else None,
        "empty_for_negative": not trace.selected if not case["sections"] else None,
    }


def run(
    output: Path, strategy: str = "structure", *, threshold: float = 0.07,
    retrieval_only: bool = False,
) -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions += json.loads(NEGATIVES_PATH.read_text(encoding="utf-8"))
    index = DocumentIndex()
    manifest = index.ensure_built()
    reranker = LocalCrossEncoderReranker()
    query_model = DeepSeekProvider(thinking_enabled=False)
    answer_model = DeepSeekProvider()
    settings = variants(threshold)
    results = []

    with TemporaryDirectory(prefix="day23-rag-") as temporary_directory:
        repository = SQLiteChatSessionRepository(Path(temporary_directory) / "chat.sqlite3")
        for case in questions:
            modes = {}
            for name, config in settings.items():
                traces: list[RetrievalTrace] = []
                rag = DocumentRag(
                    index, query_model, reranker=reranker, settings=config,
                    on_retrieval=traces.append,
                )
                answer = None
                if name in {"raw", "enhanced"} and not retrieval_only:
                    service = ChatSessionService(
                        repository, Agent(answer_model), document_rag=rag,
                        record_retrieval_comparison=False,
                    )
                    session = service.create()
                    started = monotonic()
                    response = service.send(session.id, case["question"], strategy, "rag")
                    answer = {
                        "text": response.assistant_message.content,
                        "sources": response.assistant_message.sources,
                        "seconds": round(monotonic() - started, 2),
                    }
                else:
                    rag.retrieve(case["question"], [], strategy, force=True)
                if len(traces) != 1:
                    raise RuntimeError("Ожидался один поиск на вариант")
                modes[name] = retrieval_record(case, traces[0])
                if answer is not None:
                    modes[name]["answer"] = answer
            results.append({**case, "modes": modes})
            print(f"{case['id']}/{len(questions)} complete", flush=True)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "answer_model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "embedding_model": manifest["model"],
        "reranker_model": reranker.model_name,
        "index_version": manifest["version"],
        "pdf_sha256": hashlib.sha256(index.pdf_path.read_bytes()).hexdigest(),
        "strategy": strategy,
        "threshold": threshold,
        "top_k_before": 10,
        "top_k_after": 3,
        "retrieval_only": retrieval_only,
        "method": "Fresh empty chat per generated answer; raw and enhanced answers use the same Agent and service. Rewrite and filter ablations measure retrieval only.",
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strategy", choices=("structure", "fixed"), default="structure")
    parser.add_argument("--threshold", type=float, default=0.07)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    report = run(args.output, args.strategy, threshold=args.threshold,
                 retrieval_only=args.retrieval_only)
    positives = sum(bool(case["sections"]) for case in report["cases"])
    negatives = len(report["cases"]) - positives
    for name in variants(args.threshold):
        found = sum(case["modes"][name]["expected_in_selected"] is True
                    for case in report["cases"])
        rejected = sum(case["modes"][name]["empty_for_negative"] is True
                       for case in report["cases"])
        print(f"{name}: expected source {found}/{positives}; empty negative {rejected}/{negatives}")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
