"""Evaluate grounded RAG answers on ten new questions through the chat service.

Usage: python -m evaluation.run_day24 --output data/day24-evaluation.json
Load .env before running. Each question uses a fresh chat session.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import monotonic

from app.agents.agent import Agent
from app.indexing.grounding import MAX_EXCERPT_CHARS
from app.indexing.rag import DocumentRag, RetrievalSettings, RetrievalTrace
from app.indexing.rerank import LocalCrossEncoderReranker
from app.indexing.store import DocumentIndex
from app.providers.deepseek import DeepSeekProvider
from app.services.chat_sessions import ChatSessionService
from app.storage.chat_sessions import SQLiteChatSessionRepository


QUESTIONS_PATH = Path(__file__).with_name("day24-questions.json")
DEFAULT_OUTPUT = Path("data/day24-evaluation.json")


def run(output: Path, strategy: str = "structure", *, threshold: float = 0.07) -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if len(questions) != 10:
        raise ValueError("Контрольный набор должен содержать ровно 10 вопросов")
    index = DocumentIndex()
    manifest = index.ensure_built()
    reranker = LocalCrossEncoderReranker()
    query_model = DeepSeekProvider(thinking_enabled=False)
    answer_model = DeepSeekProvider()
    settings = RetrievalSettings(min_relevance_score=threshold)
    traces: list[RetrievalTrace] = []
    rag = DocumentRag(index, query_model, reranker=reranker,
                      settings=settings, on_retrieval=traces.append)
    results = []

    with TemporaryDirectory(prefix="day24-rag-") as temporary_directory:
        service = ChatSessionService(
            SQLiteChatSessionRepository(Path(temporary_directory) / "chat.sqlite3"),
            Agent(answer_model), document_rag=rag,
            record_retrieval_comparison=False,
        )
        for case in questions:
            traces.clear()
            session = service.create()
            started = monotonic()
            response = service.send(session.id, case["question"], strategy, "rag")
            elapsed = round(monotonic() - started, 2)
            if len(traces) != 1:
                raise RuntimeError("Ожидался один поиск на вопрос")
            trace = traces[0]
            message = response.assistant_message
            scores = {
                hit.metadata["chunk_id"]: score
                for hit, score in zip(trace.candidates, trace.relevance_scores)
            }
            selected = [{
                "chunk_id": hit.metadata["chunk_id"],
                "section": hit.metadata["section"],
                "page": hit.metadata["page_start"],
                "relevance": round(scores[hit.metadata["chunk_id"]], 4),
                "excerpt": hit.metadata["text"][:MAX_EXCERPT_CHARS],
            } for hit in trace.selected]
            positive = bool(case["sections"])
            selected_by_id = {item["chunk_id"]: item for item in selected}
            source_ids = {item["chunk_id"] for item in message.sources}
            citation_ids = {item["chunk_id"] for item in message.citations}
            marker_ids = {item.lower() for item in re.findall(
                r"\[DOC:([a-z]+-\d{4})\]", message.content, flags=re.I,
            )}
            automatic_pass = (
                message.grounding_status == "answered"
                and bool(message.sources) and bool(message.citations)
                and source_ids == citation_ids == marker_ids
                and all(
                    item["chunk_id"] in selected_by_id
                    and item["section"] == selected_by_id[item["chunk_id"]]["section"]
                    for item in message.sources
                )
                and all(
                    item["chunk_id"] in selected_by_id
                    and item["quote"] in selected_by_id[item["chunk_id"]]["excerpt"]
                    for item in message.citations
                )
            ) if positive else (
                message.grounding_status == "insufficient_context"
                and not message.sources and not message.citations
                and "не знаю" in message.content.casefold()
                and "уточните" in message.content.casefold()
            )
            results.append({
                **case, "search_query": trace.search_query,
                "selected": selected,
                "answer": message.content, "sources": message.sources,
                "citations": message.citations,
                "grounding_status": message.grounding_status,
                "automatic_pass": automatic_pass,
                "seconds": elapsed,
            })
            print(f"{case['id']}/10 complete: {message.grounding_status}", flush=True)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "answer_model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "embedding_model": manifest["model"],
        "reranker_model": reranker.model_name,
        "index_version": manifest["version"],
        "pdf_sha256": hashlib.sha256(index.pdf_path.read_bytes()).hexdigest(),
        "strategy": strategy,
        "threshold": threshold,
        "method": "Fresh empty chat per answer; exact quotes validated against excerpts supplied to the model; semantic support requires manual review.",
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
    args = parser.parse_args()
    report = run(args.output, args.strategy, threshold=args.threshold)
    count = sum(case["automatic_pass"] for case in report["cases"])
    print(f"Automatic checks: {count}/10; saved to {args.output}")


if __name__ == "__main__":
    main()
