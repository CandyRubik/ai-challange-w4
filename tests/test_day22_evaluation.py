from __future__ import annotations

import json
from pathlib import Path

from app.indexing.store import SearchHit
from evaluation import run_day22


def test_paired_evaluation_uses_fresh_context_and_records_search(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"public sample")
    answer_prompts: list[list[dict]] = []

    class FakeProvider:
        def __init__(self, *, thinking_enabled: bool = True) -> None:
            pass

        def generate_json(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
            return json.dumps({"query": "English search query"})

        def generate(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
            answer_prompts.append(messages)
            assert len(messages) == 2  # A fresh, empty chat for every answer.
            if "DOCUMENT_EXCERPTS" in messages[0]["content"]:
                return "Ответ из книги [DOC:structure-0000]."
            return "Ответ без книги."

    class FakeIndex:
        pdf_path = pdf

        def ensure_built(self) -> dict:
            return {"model": "fake-embedder", "version": "fake-version"}

        def search(self, query: str, strategy: str, k: int) -> list[SearchHit]:
            assert query == "English search query"
            assert strategy == "structure" and k == 3
            return [SearchHit({
                "chunk_id": "structure-0000",
                "source": "https://example.test/sample.pdf",
                "title": "Sample",
                "section": "6.1.1 Executing tasks sequentially",
                "page_start": 2,
                "page_end": 2,
                "text": "Sequential tasks block the next request.",
            }, 0.9)]

    monkeypatch.setattr(run_day22, "DeepSeekProvider", FakeProvider)
    monkeypatch.setattr(run_day22, "DocumentIndex", FakeIndex)
    output = tmp_path / "results.json"
    report = run_day22.run(output, "structure")

    assert len(report["cases"]) == 10
    assert len(answer_prompts) == 20
    assert report["cases"][0]["answers"]["plain"]["sources"] == []
    assert report["cases"][0]["answers"]["rag"]["search_query"] == "English search query"
    assert report["cases"][0]["answers"]["rag"]["expected_source_in_top_3"] is True
    assert report["cases"][9]["answers"]["rag"]["expected_source_in_top_3"] is None
    assert json.loads(output.read_text(encoding="utf-8"))["pdf_sha256"] == report["pdf_sha256"]
