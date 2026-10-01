from __future__ import annotations

import json
from pathlib import Path

from app.agents.agent import Agent
from app.indexing.rag import DocumentRag, RetrievalSettings
from app.indexing.store import SearchHit
from app.invariants import SQLiteInvariantRepository
from app.services.chat_sessions import ChatSessionService
from app.storage.chat_sessions import SQLiteChatSessionRepository


def hit(text: str = "Executor manages submitted tasks.") -> SearchHit:
    return SearchHit({
        "chunk_id": "structure-0001", "source": "https://example.test/book.pdf",
        "title": "Test book", "section": "6.2 Executor", "page_start": 6,
        "page_end": 6, "text": text,
    }, 0.9)


class Index:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits

    def ensure_built(self) -> dict:
        return {"version": "test"}

    def search(self, query: str, strategy: str, k: int) -> list[SearchHit]:
        return self.hits[:k]


class Scorer:
    def __init__(self, score: float) -> None:
        self.value = score

    def score(self, question: str, hits: list[SearchHit]) -> list[float]:
        return [self.value] * len(hits)


class Model:
    def __init__(self, *outputs: dict) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def generate_json(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
        assert "DOCUMENT_EXCERPTS" in messages[0]["content"]
        self.calls += 1
        return json.dumps(self.outputs.pop(0))

    def generate(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
        raise AssertionError("Строгий RAG должен вызывать JSON-генерацию")


def draft(chunk_id: str = "structure-0001", quote: str = "Executor manages submitted tasks.") -> dict:
    return {"status": "answered", "claims": [{
        "text": "Executor выполняет отправленные задачи.",
        "evidence": [{"chunk_id": chunk_id, "quote": quote}],
    }]}


def service(tmp_path: Path, model: Model, *, score: float = 0.9,
            text: str = "Executor manages submitted tasks.",
            invariants: bool = False) -> ChatSessionService:
    database = tmp_path / "chat.sqlite3"
    rag = DocumentRag(
        Index([hit(text)]), model, reranker=Scorer(score),
        settings=RetrievalSettings(
            rewrite=False, candidate_k=1, final_k=1, min_relevance_score=0.5,
        ),
    )
    return ChatSessionService(
        SQLiteChatSessionRepository(database), Agent(model),
        invariant_repository=SQLiteInvariantRepository(database) if invariants else None,
        document_rag=rag, record_retrieval_comparison=False,
    )


def test_answer_exposes_and_persists_verified_quote_and_source(tmp_path: Path) -> None:
    model = Model(draft())
    chat = service(tmp_path, model)
    session = chat.create()

    answer = chat.send(session.id, "Что делает Executor?", mode="rag").assistant_message

    assert answer.grounding_status == "answered"
    assert answer.content == "Executor выполняет отправленные задачи. [DOC:structure-0001]"
    assert answer.sources[0]["source"] == "https://example.test/book.pdf"
    assert answer.sources[0]["section"] == "6.2 Executor"
    assert answer.citations == [{
        "chunk_id": "structure-0001", "quote": "Executor manages submitted tasks.",
    }]
    assert chat.get(session.id).messages[-1].citations == answer.citations


def test_below_threshold_abstains_without_answer_model(tmp_path: Path) -> None:
    model = Model()
    chat = service(tmp_path, model, score=0.1)
    session = chat.create()

    answer = chat.send(session.id, "Что делает Executor?", mode="rag").assistant_message

    assert answer.grounding_status == "insufficient_context"
    assert "Не знаю" in answer.content and "уточните" in answer.content
    assert answer.sources == answer.citations == []
    assert model.calls == 0


def test_invalid_quote_is_retried_then_rejected(tmp_path: Path) -> None:
    model = Model(draft(quote="Executor invents unsupported details."), draft())
    chat = service(tmp_path, model)
    session = chat.create()

    answer = chat.send(session.id, "Что делает Executor?", mode="rag").assistant_message

    assert model.calls == 2
    assert answer.grounding_status == "answered"
    assert answer.citations[0]["quote"] == "Executor manages submitted tasks."


def test_repeated_invalid_chunk_fails_closed(tmp_path: Path) -> None:
    model = Model(draft(chunk_id="structure-9999"), draft(chunk_id="structure-9999"))
    chat = service(tmp_path, model)
    session = chat.create()

    answer = chat.send(session.id, "Что делает Executor?", mode="rag").assistant_message

    assert model.calls == 2
    assert answer.grounding_status == "grounding_failed"
    assert answer.sources == answer.citations == []
    assert "Не могу подтвердить" in answer.content


def test_model_can_abstain_when_retrieved_text_does_not_answer(tmp_path: Path) -> None:
    model = Model({"status": "unknown", "claims": []})
    chat = service(tmp_path, model)
    session = chat.create()

    answer = chat.send(session.id, "Какой срок указан?", mode="rag").assistant_message

    assert answer.grounding_status == "insufficient_context"
    assert answer.sources == answer.citations == []


def test_quote_keeps_source_case_when_answer_is_uppercased(tmp_path: Path) -> None:
    model = Model(draft())
    chat = service(tmp_path, model, invariants=True)
    session = chat.create()

    answer = chat.send(session.id, "Что делает Executor?", mode="rag").assistant_message

    assert "EXECUTOR ВЫПОЛНЯЕТ" in answer.content
    assert answer.citations[0]["quote"] == "Executor manages submitted tasks."


def test_sentence_limit_retries_a_grounded_answer(tmp_path: Path) -> None:
    too_long = {"status": "answered", "claims": [draft()["claims"][0]] * 4}
    model = Model(too_long, draft())
    chat = service(tmp_path, model, invariants=True)
    session = chat.create()

    answer = chat.send(session.id, "Что делает Executor?", mode="rag").assistant_message

    assert model.calls == 2
    assert answer.grounding_status == "answered"
    assert len(answer.sources) == len(answer.citations) == 1


def test_quote_uses_exact_excerpt_whitespace(tmp_path: Path) -> None:
    model = Model(draft(quote="Executor manages submitted tasks."))
    chat = service(tmp_path, model, text="Executor manages\nsubmitted tasks.")
    session = chat.create()

    answer = chat.send(session.id, "Что делает Executor?", mode="rag").assistant_message

    assert answer.citations[0]["quote"] == "Executor manages\nsubmitted tasks."


def test_quote_cannot_come_from_text_hidden_by_excerpt_limit(tmp_path: Path) -> None:
    unseen = "Hidden evidence beyond the excerpt."
    model = Model(draft(quote=unseen), draft(quote=unseen))
    chat = service(tmp_path, model, text="Executor " + "x" * 1800 + unseen)
    session = chat.create()

    answer = chat.send(session.id, "Что делает Executor?", mode="rag").assistant_message

    assert answer.grounding_status == "grounding_failed"
    assert answer.citations == answer.sources == []
