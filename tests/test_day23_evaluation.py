from __future__ import annotations

import json
from pathlib import Path

from app.indexing.store import SearchHit
from evaluation import run_day23


def test_day23_compares_four_retrieval_variants_and_isolated_answers(
    tmp_path: Path, monkeypatch,
) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{
        "id": 1, "question": "Что делает Executor?", "expected": "Запускает задачи",
        "sections": ["6.2 Executor"], "pdf_pages": [1],
    }]))
    negatives = tmp_path / "negatives.json"
    negatives.write_text(json.dumps([{
        "id": 2, "question": "Есть ли Kotlin coroutines?", "expected": "Нет подтверждения",
        "sections": [], "pdf_pages": [],
    }]))
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"sample")
    prompts = []

    class FakeProvider:
        def __init__(self, *, thinking_enabled: bool = True) -> None:
            pass

        def generate_json(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
            return json.dumps({"query": "Executor"})

        def generate(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
            prompts.append(messages)
            assert len(messages) == 2
            if "NO_RELEVANT_DOCUMENT_EXCERPTS" in messages[0]["content"]:
                return "В найденных фрагментах нет подтверждения."
            return "Executor runs tasks [DOC:structure-0000]."

    class FakeIndex:
        pdf_path = pdf

        def ensure_built(self) -> dict:
            return {"model": "test", "version": "test"}

        def search(self, query: str, strategy: str, k: int) -> list[SearchHit]:
            return [SearchHit({
                "chunk_id": "structure-0000", "section": "6.2 Executor",
                "page_start": 1, "page_end": 1, "text": "Executor runs tasks.",
                "source": "https://example.test/book.pdf", "title": "Test book",
            }, 0.9)]

    class FakeReranker:
        model_name = "test-reranker"

        def score(self, question: str, hits: list[SearchHit]) -> list[float]:
            return [0.9] * len(hits)

    monkeypatch.setattr(run_day23, "QUESTIONS_PATH", questions)
    monkeypatch.setattr(run_day23, "NEGATIVES_PATH", negatives)
    monkeypatch.setattr(run_day23, "DocumentIndex", FakeIndex)
    monkeypatch.setattr(run_day23, "LocalCrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(run_day23, "DeepSeekProvider", FakeProvider)
    report = run_day23.run(tmp_path / "report.json")

    assert len(prompts) == 4
    assert set(report["cases"][0]["modes"]) == {"raw", "rewrite", "filter", "enhanced"}
    assert report["cases"][0]["modes"]["enhanced"]["expected_in_selected"] is True
    assert report["cases"][1]["modes"]["raw"]["empty_for_negative"] is False
    assert report["cases"][1]["modes"]["enhanced"]["empty_for_negative"] is True
    assert report["cases"][1]["modes"]["enhanced"]["answer"]["sources"] == []
    assert json.loads((tmp_path / "report.json").read_text())["top_k_before"] == 10
