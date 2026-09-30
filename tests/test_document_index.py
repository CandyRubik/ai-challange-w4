from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("faiss")

from app.indexing.chunking import fixed_chunks, structured_chunks
from app.indexing.corpus import Page
from app.indexing.rag import DocumentRag
from app.indexing.store import DocumentIndex, DocumentIndexError
from app.indexing.store import SearchHit
from app.agents.agent import Agent
from app.services.chat_sessions import ChatSessionService
from app.storage.chat_sessions import SQLiteChatSessionRepository


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


class ChatModel(Router):
    def generate(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
        assert "DOCUMENT_EXCERPTS" in messages[0]["content"]
        return "Executor manages tasks [DOC:structure-0000]."


class ReadyIndex:
    def status(self) -> dict:
        return {"index": {"version": "test"}}

    def ensure_built(self) -> dict:
        return {"version": "test"}

    def search(self, query: str, strategy: str, k: int) -> list[SearchHit]:
        assert strategy == "structure"
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
    rag = DocumentRag(index, Router())
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
    rag = DocumentRag(reopened, Router())
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
    service = ChatSessionService(repository, Agent(model), document_rag=DocumentRag(ReadyIndex(), model))
    session = service.create()
    sent = service.send(session.id, "What is Executor?")
    assert sent.assistant_message.sources[0]["chunk_id"] == "structure-0000"
    assert service.get(session.id).messages[-1].sources[0]["page_start"] == 3


def test_unrelated_question_does_not_build_index(tmp_path: Path) -> None:
    index = DocumentIndex(tmp_path / "index", tmp_path / "missing.pdf", FakeEmbedder())
    rag = DocumentRag(index, UnrelatedRouter())
    assert rag.retrieve("Какая погода завтра?", [], "structure") == []
    assert index.status()["state"] == "missing"
    assert rag.context([]) == ""


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
