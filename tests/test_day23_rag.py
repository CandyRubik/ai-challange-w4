from __future__ import annotations

import json
from pathlib import Path

from app.agents.agent import Agent
from app.indexing.rag import DocumentRag, RetrievalSettings
from app.indexing.store import SearchHit
from app.services.chat_sessions import ChatSessionService
from app.storage.chat_sessions import SQLiteChatSessionRepository


def hit(chunk_id: str, text: str) -> SearchHit:
    return SearchHit({
        "chunk_id": chunk_id, "section": "6.2 Example", "text": text,
        "source": "https://example.test/book.pdf", "title": "Test book",
        "page_start": 1, "page_end": 1,
    }, 0.9)


class CandidateIndex:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.searches: list[tuple[str, str, int]] = []

    def ensure_built(self) -> dict:
        return {"version": "test"}

    def search(self, query: str, strategy: str, k: int) -> list[SearchHit]:
        self.searches.append((query, strategy, k))
        return self.hits[:k]


class RewriteModel:
    def generate_json(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
        return json.dumps({"query": "executor framework thread pools"})


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.questions: list[str] = []

    def score(self, question: str, hits: list[SearchHit]) -> list[float]:
        self.questions.append(question)
        return self.scores[:len(hits)]


def test_reranker_filters_and_limits_candidates_using_original_question() -> None:
    candidates = [
        hit("structure-0000", "An unrelated thread pool example."),
        hit("structure-0001", "Timer has one worker thread."),
        hit("structure-0002", "Timer cannot handle a long-running task."),
        hit("structure-0003", "Timer is not recommended for periodic tasks."),
    ]
    index = CandidateIndex(candidates)
    scorer = FakeReranker([0.99, 0.42, 0.8, 0.05])
    traces = []
    rag = DocumentRag(
        index, RewriteModel(), reranker=scorer,
        settings=RetrievalSettings(candidate_k=4, final_k=2, min_relevance_score=0.1),
        on_retrieval=traces.append,
    )
    selected = rag.retrieve("Что сказано про Timer?", [], "structure", force=True)
    assert [item.metadata["chunk_id"] for item in selected] == ["structure-0002", "structure-0001"]
    assert scorer.questions == [
        "Что сказано про Timer?",
        "Что сказано про Timer?\nexecutor framework thread pools Timer",
    ]
    assert index.searches == [("executor framework thread pools Timer", "structure", 4)]
    assert traces[0].lexical_matches == (False, True, True, True)
    assert rag.cited_sources("[DOC:structure-0000] [DOC:structure-0002]", selected) == [
        {"chunk_id": "structure-0002", "source": "https://example.test/book.pdf",
         "title": "Test book", "section": "6.2 Example", "page_start": 1, "page_end": 1},
    ]


def test_no_relevant_hits_yield_explicit_no_evidence_answer(tmp_path: Path) -> None:
    class AnswerModel(RewriteModel):
        def generate(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
            assert "NO_RELEVANT_DOCUMENT_EXCERPTS" in messages[0]["content"]
            return "В найденных фрагментах нет подтверждения."

    model = AnswerModel()
    index = CandidateIndex([hit("structure-0000", "Executor runs tasks.")])
    rag = DocumentRag(
        index, model, reranker=FakeReranker([0.95]),
        settings=RetrievalSettings(candidate_k=1, final_k=1, min_relevance_score=0.08),
    )
    service = ChatSessionService(
        SQLiteChatSessionRepository(tmp_path / "chat.sqlite3"), Agent(model),
        document_rag=rag,
    )
    session = service.create()
    response = service.send(session.id, "Есть ли в книге Kotlin coroutines?", mode="rag")
    assert response.assistant_message.content == "В найденных фрагментах нет подтверждения."
    assert response.assistant_message.sources == []
    comparison = response.assistant_message.retrieval
    assert [item["chunk_id"] for item in comparison["raw"]["hits"]] == ["structure-0000"]
    assert comparison["active"]["hits"] == []
    assert comparison["active"]["rejected"][0]["reason"] == "identifier"
    assert service.get(session.id).messages[-1].retrieval == comparison
    assert index.searches[0][0].endswith("Kotlin coroutines")


def test_rag_message_persists_raw_vs_selected_sources(tmp_path: Path) -> None:
    class AnswerModel(RewriteModel):
        def generate(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
            return "Timer has one worker [DOC:structure-0001]."

    index = CandidateIndex([
        hit("structure-0000", "Unrelated text."),
        hit("structure-0001", "Timer has one worker."),
    ])
    rag = DocumentRag(
        index, AnswerModel(), reranker=FakeReranker([0.01, 0.8]),
        settings=RetrievalSettings(candidate_k=2, final_k=1, min_relevance_score=0.07),
    )
    database = tmp_path / "chat.sqlite3"
    service = ChatSessionService(SQLiteChatSessionRepository(database), Agent(AnswerModel()),
                                 document_rag=rag)
    session = service.create()
    response = service.send(session.id, "Что сказано про Timer?", mode="rag")

    comparison = response.assistant_message.retrieval
    assert comparison["raw"]["hits"][0]["chunk_id"] == "structure-0000"
    assert comparison["active"]["query"].endswith("Timer")
    assert comparison["active"]["hits"][0]["chunk_id"] == "structure-0001"
    assert comparison["active"]["rejected"][0]["chunk_id"] == "structure-0000"
    assert response.assistant_message.sources[0]["chunk_id"] == "structure-0001"
    assert SQLiteChatSessionRepository(database).get(session.id).messages[-1].retrieval == comparison


def test_legacy_baseline_uses_raw_question_and_skips_reranker() -> None:
    index = CandidateIndex([hit("structure-0000", "Any text")])
    rag = DocumentRag(
        index, RewriteModel(),
        settings=RetrievalSettings(
            rewrite=False, filter_relevance=False, candidate_k=1, final_k=1,
        ),
    )
    assert rag.retrieve("Исходный вопрос", [], "structure", force=True)
    assert index.searches == [("Исходный вопрос", "structure", 1)]
