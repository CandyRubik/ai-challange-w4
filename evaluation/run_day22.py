"""Run paired, isolated plain/RAG answers through the production chat service.

Usage: python -m evaluation.run_day22 --output data/day22-evaluation.json
The process needs DEEPSEEK_API_KEY; load .env before running it.
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
from app.indexing.rag import DocumentRag, RetrievalSettings
from app.indexing.store import DocumentIndex, SearchHit
from app.providers.deepseek import DeepSeekProvider
from app.services.chat_sessions import ChatSessionService
from app.storage.chat_sessions import SQLiteChatSessionRepository


QUESTIONS_PATH = Path(__file__).with_name("day22-questions.json")
DEFAULT_OUTPUT = Path("data/day22-evaluation.json")


class RecordingIndex:
    """Capture the exact query and hits used by a chat response."""

    def __init__(self, index: DocumentIndex) -> None:
        self.index = index
        self.query: str | None = None
        self.hits: list[SearchHit] = []

    def ensure_built(self) -> dict:
        return self.index.ensure_built()

    def search(self, query: str, strategy: str, k: int) -> list[SearchHit]:
        self.query = query
        self.hits = self.index.search(query, strategy, k)
        return self.hits

    def reset(self) -> None:
        self.query = None
        self.hits = []


def run(output: Path, strategy: str) -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    index = DocumentIndex()
    manifest = index.ensure_built()
    recording_index = RecordingIndex(index)
    answer_model = DeepSeekProvider()
    query_model = DeepSeekProvider(thinking_enabled=False)
    rag = DocumentRag(
        recording_index, query_model,
        settings=RetrievalSettings(
            preserve_terms=False, filter_relevance=False, candidate_k=3,
        ),
    )

    with TemporaryDirectory(prefix="day22-rag-") as temporary_directory:
        repository = SQLiteChatSessionRepository(Path(temporary_directory) / "chat.sqlite3")
        service = ChatSessionService(
            repository, Agent(answer_model), document_rag=rag,
            record_retrieval_comparison=False,
        )
        results = []
        for case in questions:
            answers = {}
            for mode in ("plain", "rag"):
                recording_index.reset()
                session = service.create()
                started = monotonic()
                response = service.send(session.id, case["question"], strategy, mode)
                answers[mode] = {
                    "answer": response.assistant_message.content,
                    "sources": response.assistant_message.sources,
                    "seconds": round(monotonic() - started, 2),
                }
                if mode == "rag":
                    answers[mode]["search_query"] = recording_index.query
                    answers[mode]["top_hits"] = [
                        {
                            "chunk_id": hit.metadata["chunk_id"],
                            "section": hit.metadata["section"],
                            "pdf_page": hit.metadata["page_start"],
                            "score": round(hit.score, 3),
                        }
                        for hit in recording_index.hits
                    ]
                    answers[mode]["expected_source_in_top_3"] = any(
                        hit.metadata["section"] in case["sections"]
                        and hit.metadata["page_start"] in case["pdf_pages"]
                        for hit in recording_index.hits
                    ) if case["sections"] else None
                else:
                    assert recording_index.query is None
            results.append({**case, "answers": answers})
            print(f"{case['id']}/10 complete", flush=True)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "answer_model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "embedding_model": manifest["model"],
        "index_version": manifest["version"],
        "pdf_sha256": hashlib.sha256(index.pdf_path.read_bytes()).hexdigest(),
        "strategy": strategy,
        "method": "Fresh empty chat per answer; same Agent and ChatSessionService; no profile, memory, or product invariants.",
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strategy", choices=("structure", "fixed"), default="structure")
    args = parser.parse_args()
    report = run(args.output, args.strategy)
    hits = sum(case["answers"]["rag"]["expected_source_in_top_3"] is True for case in report["cases"])
    print(f"Saved {len(report['cases'])} paired answers to {args.output}; expected source in top 3: {hits}/9")


if __name__ == "__main__":
    main()
