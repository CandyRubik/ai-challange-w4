from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import pytest
from fastapi.testclient import TestClient

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")

from app.indexing.chunking import fixed_chunks, structured_chunks
from app.indexing.corpus import Page
from app.indexing.rag import DocumentRag, RetrievalSettings
from app.indexing.store import DocumentIndex, DocumentIndexError
from app.indexing.store import SearchHit
from app.agents.agent import Agent
from app.main import app, get_chat_session_service
from app.services.chat_sessions import ChatSessionService
from app.storage.chat_sessions import SQLiteChatSessionRepository


LEGACY_SETTINGS = RetrievalSettings(filter_relevance=False, candidate_k=3)


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False, verbose: bool = False) -> list[int]:
        return [sum((i + 1) * ord(char) for i, char in enumerate(word)) for word in text.split()]

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(token) for token in tokens)


class FakeEmbedder:
    tokenizer = WordTokenizer()

    def encode(self, sentences: list[str], *, normalize_embeddings: bool, batch_size: int) -> np.ndarray:
        result = []
        for sentence in sentences:
            words = sentence.lower().split()
            vector = np.asarray([words.count("executor"), words.count("future"), len(words) / 100], dtype="float32")
            vector /= np.linalg.norm(vector)
            result.append(vector)
        return np.asarray(result, dtype="float32")


class Router:
    def generate_json(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
        return json.dumps({"search_book": True, "query": "Executor"})


class UnrelatedRouter:
    def generate_json(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
        return json.dumps({"search_book": False, "query": ""})


class RewriteOnlyRouter:
    def generate_json(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
        return json.dumps({"search_book": False, "query": "travel quotes time budget"})


class ChatModel(Router):
    def generate_json(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
        if "DOCUMENT_EXCERPTS" in messages[0]["content"]:
            return json.dumps({"status": "answered", "claims": [{
                "text": "Executor manages tasks.",
                "evidence": [{"chunk_id": "structure-0000", "quote": "Executor manages tasks."}],
            }]})
        return super().generate_json(messages=messages, max_tokens=max_tokens)

    def generate(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
        assert "DOCUMENT_EXCERPTS" in messages[0]["content"]
        return "Executor manages tasks [DOC:structure-0000]."


class ReadyIndex:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def status(self) -> dict:
        return {"index": {"version": "test"}}

    def ensure_built(self) -> dict:
        return {"version": "test"}

    def search(self, query: str, strategy: str, k: int) -> list[SearchHit]:
        assert strategy == "structure"
        self.queries.append(query)
        return [SearchHit({
            "chunk_id": "structure-0000", "source": "https://jcip.net/jcip-sample.pdf",
            "title": "Java Concurrency in Practice", "section": "6.1 Executor Framework",
            "page_start": 3, "page_end": 3, "text": "Executor manages tasks.",
        }, 0.9)]


def test_chunking_retains_metadata_and_section_boundaries() -> None:
    pages = [
        Page(1, "6.1 Executor Framework\n" + "executor " * 330, "6.1 Executor Framework"),
        Page(2, "6.2 Future Tasks\n" + "future " * 125, "6.2 Future Tasks"),
    ]
    fixed = fixed_chunks(pages, WordTokenizer())
    structure = structured_chunks(pages, WordTokenizer())
    assert fixed[0].page_start == 1
    assert any(chunk.page_start == 1 and chunk.page_end == 2 for chunk in fixed)
    assert all(chunk.page_start == chunk.page_end for chunk in structure)
    assert {chunk.section for chunk in structure} == {"6.1 Executor Framework", "6.2 Future Tasks"}
    assert sum(chunk.token_count for chunk in structure) == sum(len(page.text.split()) - 3 for page in pages)


def test_index_persists_two_strategies_and_rag_sources(tmp_path: Path, monkeypatch) -> None:
    pages = [Page(number, "6.1 Executor Framework\n" + "executor future " * 160,
                  "6.1 Executor Framework") for number in range(1, 21)]
    extracted = []
    def extract(_: Path) -> list[Page]:
        extracted.append(True)
        return pages
    monkeypatch.setattr("app.indexing.store.extract_pages", extract)
    pdf = tmp_path / "sample.pdf"
    pdf.touch()
    root = tmp_path / "index"
    index = DocumentIndex(root, pdf, FakeEmbedder())
    rag = DocumentRag(index, Router(), settings=LEGACY_SETTINGS)
    assert index.status()["state"] == "missing"
    assert rag.retrieve("What is Executor?", [], "structure")
    manifest = index.status()["index"]
    assert manifest["pages"] == 20
    assert set(manifest["strategies"]) == {"fixed", "structure"}
    assert rag.retrieve("What is Executor?", [], "fixed")
    assert len(extracted) == 1
    reopened = DocumentIndex(root, pdf, FakeEmbedder())
    hits = reopened.search("Executor", "structure")
    assert len(hits) == 3
    assert hits[0].metadata["source"].startswith("https://")
    rag = DocumentRag(reopened, Router(), settings=LEGACY_SETTINGS)
    routed = rag.retrieve("What is Executor?", [], "structure")
    assert routed
    answer = f"Executor is described in [DOC:{routed[0].metadata['chunk_id']}]."
    sources = rag.cited_sources(answer, routed)
    assert len(sources) == 1
    assert sources[0]["page_start"] >= 1
    assert rag.cited_sources("No book claim", routed) == []
    assert len(rag.cited_sources("[DOC:STRUCTURE-0000]", routed)) == 1

    repository = SQLiteChatSessionRepository(tmp_path / "chat.sqlite3")
    session = repository.create()
    repository.append_exchange(session.id, "What is Executor?", answer, sources=sources)
    assert repository.get(session.id).messages[-1].sources[0]["chunk_id"] == sources[0]["chunk_id"]


def test_chat_uses_rag_and_persists_citations(tmp_path: Path) -> None:
    repository = SQLiteChatSessionRepository(tmp_path / "chat.sqlite3")
    model = ChatModel()
    service = ChatSessionService(repository, Agent(model), document_rag=DocumentRag(ReadyIndex(), model, settings=LEGACY_SETTINGS))
    session = service.create()
    sent = service.send(session.id, "What is Executor?", mode="rag")
    assert sent.assistant_message.sources[0]["chunk_id"] == "structure-0000"
    assert sent.assistant_message.citations == [
        {"chunk_id": "structure-0000", "quote": "Executor manages tasks."},
    ]
    assert sent.assistant_message.grounding_status == "answered"
    assert sent.assistant_message.mode == "rag"
    assert sent.assistant_message.retrieval["raw"]["hits"]
    assert service.get(session.id).messages[-1].sources[0]["page_start"] == 3
    assert service.get(session.id).messages[-1].citations == sent.assistant_message.citations
    assert service.get(session.id).messages[-1].mode == "rag"


def test_plain_mode_skips_router_and_index_and_persists_mode(tmp_path: Path) -> None:
    class PlainModel:
        def generate(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
            assert "DOCUMENT_EXCERPTS" not in messages[0]["content"]
            return "Ответ без документа"

    index = DocumentIndex(tmp_path / "index", tmp_path / "missing.pdf", FakeEmbedder())
    service = ChatSessionService(
        SQLiteChatSessionRepository(tmp_path / "chat.sqlite3"),
        Agent(PlainModel()), document_rag=DocumentRag(index, Router(), settings=LEGACY_SETTINGS),
    )
    session = service.create()
    sent = service.send(session.id, "Что сказано в книге?", mode="plain")
    assert sent.assistant_message.mode == "plain"
    assert sent.assistant_message.sources == []
    assert sent.assistant_message.retrieval is None
    assert service.get(session.id).messages[-1].mode == "plain"
    assert index.status()["state"] == "missing"


def test_explicit_rag_searches_even_when_router_declines() -> None:
    index = ReadyIndex()
    hits = DocumentRag(index, RewriteOnlyRouter(), settings=LEGACY_SETTINGS).retrieve(
        "Как ограничивают ожидание?", [], "structure", force=True,
    )
    assert hits
    assert index.queries == ["travel quotes time budget"]


def test_explicit_rag_falls_back_to_question_if_query_json_is_invalid() -> None:
    class BrokenRouter:
        def generate_json(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
            return "not json"

    index = ReadyIndex()
    assert DocumentRag(index, BrokenRouter(), settings=LEGACY_SETTINGS).retrieve(
        "Executor question", [], "structure", force=True,
    )
    assert index.queries == ["Executor question"]


def test_chat_api_exposes_two_modes_and_rejects_unknown_mode(tmp_path: Path) -> None:
    class DualModeModel(Router):
        def generate_json(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
            if "DOCUMENT_EXCERPTS" in messages[0]["content"]:
                return json.dumps({"status": "answered", "claims": [{
                    "text": "Книга описывает Executor.",
                    "evidence": [{"chunk_id": "structure-0000", "quote": "Executor manages tasks."}],
                }]})
            return super().generate_json(messages=messages, max_tokens=max_tokens)

        def generate(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
            if "DOCUMENT_EXCERPTS" in messages[0]["content"]:
                return "Книга описывает Executor [DOC:structure-0000]."
            return "Общий ответ без книги."

    model = DualModeModel()
    service = ChatSessionService(
        SQLiteChatSessionRepository(tmp_path / "chat.sqlite3"),
        Agent(model), document_rag=DocumentRag(ReadyIndex(), model, settings=LEGACY_SETTINGS),
    )
    session = service.create()
    app.dependency_overrides[get_chat_session_service] = lambda: service
    try:
        client = TestClient(app)
        plain = client.post(
            f"/api/chat/sessions/{session.id}/messages",
            json={"content": "Что такое Executor?"},
        )
        rag = client.post(
            f"/api/chat/sessions/{session.id}/messages",
            json={"content": "А что сказано в книге?", "mode": "rag"},
        )
        invalid = client.post(
            f"/api/chat/sessions/{session.id}/messages",
            json={"content": "Вопрос", "mode": "unknown"},
        )
        history = client.get(f"/api/chat/sessions/{session.id}")
    finally:
        app.dependency_overrides.clear()

    assert plain.status_code == rag.status_code == 200
    assert plain.json()["assistant_message"]["mode"] == "plain"
    assert plain.json()["assistant_message"]["sources"] == []
    assert rag.json()["assistant_message"]["mode"] == "rag"
    assert rag.json()["assistant_message"]["retrieval"]["active"]["hits"]
    assert len(rag.json()["assistant_message"]["sources"]) == 1
    assert rag.json()["assistant_message"]["citations"][0]["quote"] == "Executor manages tasks."
    assert invalid.status_code == 422
    assert [message["mode"] for message in history.json()["messages"]] == [
        None, "plain", None, "rag",
    ]


def test_unrelated_question_does_not_build_index(tmp_path: Path) -> None:
    index = DocumentIndex(tmp_path / "index", tmp_path / "missing.pdf", FakeEmbedder())
    rag = DocumentRag(index, UnrelatedRouter())
    assert rag.retrieve("Какая погода завтра?", [], "structure") == []
    assert index.status()["state"] == "missing"
    assert "NO_RELEVANT_DOCUMENT_EXCERPTS" in rag.context([])


def test_first_relevant_question_reports_missing_pdf(tmp_path: Path) -> None:
    index = DocumentIndex(tmp_path / "index", tmp_path / "missing.pdf", FakeEmbedder())
    with pytest.raises(DocumentIndexError, match="PDF не найден"):
        DocumentRag(index, Router()).retrieve("What does the book say?", [], "structure")
    assert index.status()["state"] == "error"


def test_concurrent_first_queries_share_one_build(tmp_path: Path, monkeypatch) -> None:
    index = DocumentIndex(tmp_path / "index", tmp_path / "sample.pdf", FakeEmbedder())
    count = 0
    lock = Lock()

    def build() -> dict:
        nonlocal count
        with lock:
            count += 1
        time.sleep(0.05)
        index.root.mkdir()
        manifest = {"version": "once"}
        (index.root / "active.json").write_text(json.dumps(manifest))
        return manifest

    monkeypatch.setattr(index, "build", build)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: index.ensure_built(), range(2)))
    assert results == [{"version": "once"}, {"version": "once"}]
    assert count == 1
