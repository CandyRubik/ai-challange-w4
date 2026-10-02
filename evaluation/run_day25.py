"""Run two long conversations through the real Day 25 RAG chat.

Load .env first, then: python -m evaluation.run_day25
The report contains answers, sources, state snapshots, and automatic checks.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import monotonic

from app.agents.agent import Agent
from app.indexing.rag import DocumentRag, RetrievalSettings, RetrievalTrace
from app.indexing.rerank import LocalCrossEncoderReranker
from app.indexing.store import DocumentIndex
from app.providers.deepseek import DeepSeekProvider
from app.rag_chat.service import GROUNDING_SYSTEM_PROMPT, RagChatService
from app.rag_chat.state import StateChange, TurnDecision, TurnInterpreter
from app.rag_chat.store import SQLiteRagChatRepository


SCENARIOS_PATH = Path(__file__).with_name("day25-scenarios.json")
DEFAULT_OUTPUT = Path(__file__).with_name("day25-evaluation.json")
OFFLINE_OUTPUT = Path(__file__).with_name("day25-offline-evaluation.json")


class OfflineInterpreter:
    """Fixed interpretations for the published synthetic scenario messages."""

    def __init__(self, scenarios: list[dict]) -> None:
        self._cases = iter((scenario, position, case)
                           for scenario in scenarios
                           for position, case in enumerate(scenario["turns"], start=1))

    def interpret(self, message: str, state: object, recent: list[dict]) -> TurnDecision:
        scenario, position, case = next(self._cases)
        if message != case["message"]:
            raise ValueError("Offline scenario order changed")
        if case["expect"] != "statement":
            return TurnDecision(
                kind="question",
                question_scope="state" if case["expect"] == "state" else "document",
                search_question=case["query"],
            )
        if position == 1:
            return TurnDecision(kind="statement", goal_action="set", goal={
                "executor": "Объяснить начинающим Java-разработчикам переход от ThreadPerTaskWebServer к Executor",
                "renderer": "Разобраться, как FutureRenderer и CompletionService влияют на отзывчивость отображения веб-страницы",
            }[scenario["id"]])
        if position in {2, 7}:
            group = "length" if scenario["id"] == "executor" else "audience"
            value = (
                "кратко" if position == 2 else "подробно"
            ) if scenario["id"] == "executor" else (
                "начинающие студенты" if position == 2 else "опытные Java-разработчики"
            )
            return TurnDecision(kind="statement", changes=[StateChange(
                group="constraint", action="set", key=group, value=value,
            )])
        term, value = (
            ("старый сервер", "ThreadPerTaskWebServer")
            if scenario["id"] == "executor" else
            ("изображения", "картинки, которые загрузчик скачивает для страницы")
        )
        return TurnDecision(kind="statement", changes=[StateChange(
            group="term", action="set", key=term, value=value,
        )])


class OfflineExtractiveModel:
    """Emit a short exact excerpt; only proves wiring and citation validation."""

    def generate_json(self, *, messages: list[dict], max_tokens: int = 2000) -> str:
        excerpts = json.loads(messages[0]["content"].split("DOCUMENT_EXCERPTS:\n", 1)[1])
        if not excerpts:
            return json.dumps({"status": "unknown", "claims": []})
        excerpt = excerpts[0]
        quote = excerpt["text"][:160].strip()
        return json.dumps({"status": "answered", "claims": [{
            "text": "В найденном фрагменте сказано: «" + " ".join(quote.split()) + "».",
            "evidence": [{"chunk_id": excerpt["id"], "quote": quote}],
        }]}, ensure_ascii=False)


def _check(turn: dict, case: dict, traces: list[RetrievalTrace]) -> dict[str, bool]:
    expected = case["expect"]
    questions = {"document", "unsupported", "state"}
    searched = bool(traces) if expected in questions else not traces
    source_ids = {item["chunk_id"] for item in turn["sources"] if item["kind"] == "document"}
    citation_ids = {item["chunk_id"] for item in turn["citations"]}
    marker_ids = set(re.findall(r"\[DOC:([^]]+)\]", turn["answer"]))
    if expected == "document":
        answer = (turn["status"] == "done" and turn["grounding_status"] == "answered"
                  and bool(source_ids) and source_ids == citation_ids == marker_ids)
    elif expected == "unsupported":
        answer = (turn["status"] == "done"
                  and turn["grounding_status"] == "insufficient_context"
                  and not turn["sources"] and not turn["citations"])
    elif expected == "state":
        answer = (turn["status"] == "done" and turn["grounding_status"] == "state"
                  and bool(turn["sources"])
                  and all(item["kind"] == "message" for item in turn["sources"]))
    else:
        answer = (turn["status"] == "done" and turn["kind"] == "statement"
                  and bool(turn["sources"])
                  and all(item["kind"] == "message" for item in turn["sources"]))
    return {"searched_as_expected": searched, "answer_as_expected": answer}


def run(output: Path = DEFAULT_OUTPUT, *, offline: bool = False) -> dict:
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if len(scenarios) != 2 or any(not 10 <= len(item["turns"]) <= 15 for item in scenarios):
        raise ValueError("Ожидаются два сценария по 10–15 сообщений")

    index = DocumentIndex()
    manifest = index.ensure_built()
    reranker = LocalCrossEncoderReranker()
    fast_model = None if offline else DeepSeekProvider(thinking_enabled=False)
    traces: list[RetrievalTrace] = []
    rag = DocumentRag(
        index, fast_model, reranker=reranker,
        settings=replace(RetrievalSettings.from_env(), rewrite=False),
        on_retrieval=traces.append,
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline_scripted" if offline else "live_deepseek",
        "answer_model": "offline-extractive" if offline else os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "embedding_model": manifest["model"],
        "reranker_model": reranker.model_name,
        "index_version": manifest["version"],
        "pdf_sha256": hashlib.sha256(index.pdf_path.read_bytes()).hexdigest(),
        "history_turns": 4,
        "method": (
            "Offline scripted interpretation and extractive answer model over the real local index and reranker. This checks orchestration, persistence, retrieval, and citation plumbing, not LLM answer quality."
            if offline else
            "Two persistent 12-turn conversations with DeepSeek; each question runs retrieval. Document citations are validated against retrieved excerpts by the service. Semantic completeness needs manual review."
        ),
        "scenarios": [],
    }
    with TemporaryDirectory(prefix="day25-rag-") as temporary_directory:
        repository = SQLiteRagChatRepository(Path(temporary_directory) / "chat.sqlite3")
        interpreter = OfflineInterpreter(scenarios) if offline else TurnInterpreter(fast_model)
        answer_model = (OfflineExtractiveModel() if offline
                        else DeepSeekProvider(thinking_enabled=False))
        chat = RagChatService(repository, interpreter, rag,
                              Agent(answer_model, system_prompt=GROUNDING_SYSTEM_PROMPT,
                                    max_tokens=3_000), history_turns=4)
        for scenario in scenarios:
            session = chat.create()
            rows = []
            report["scenarios"].append({
                "id": scenario["id"], "title": scenario["title"],
                "session_id": session.id, "turns": rows,
            })
            for position, case in enumerate(scenario["turns"], start=1):
                traces.clear()
                started = monotonic()
                turn = chat.send(session.id, case["message"])
                # Provider failures remain visible and retryable in the report.
                if turn.status == "failed":
                    turn = chat.retry(session.id, turn.id)
                elapsed = round(monotonic() - started, 2)
                snapshot = chat.get(session.id)
                row = {
                    "position": position, "user": case["message"],
                    "expected": case["expect"],
                    "turn": turn.model_dump(mode="json"),
                    "search_queries": [trace.search_query for trace in traces],
                    "selected_chunk_ids": list(dict.fromkeys(
                        hit.metadata["chunk_id"]
                        for trace in traces for hit in trace.selected
                    )),
                    "state": snapshot.state.model_dump(),
                    "checks": _check(turn.model_dump(mode="json"), case, traces),
                    "seconds": elapsed,
                }
                rows.append(row)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
                print(f"{scenario['id']} {position}/{len(scenario['turns'])}: "
                      f"{turn.grounding_status or turn.status}, "
                      f"{row['checks']}", flush=True)
                if turn.status != "done":
                    raise RuntimeError("Последний ход не завершился после повторения")
            final_state = chat.get(session.id).state
            report["scenarios"][-1]["final_state"] = final_state.model_dump()
            report["scenarios"][-1]["goal_preserved"] = bool(
                final_state.goal and scenario["goal_contains"] in final_state.goal.value
            )
            report["scenarios"][-1]["correction_preserved"] = (
                any(scenario["constraint_final_contains"].casefold() in item.value.casefold()
                    for item in final_state.constraints)
            )
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offline", action="store_true", help="Use scripted local models")
    args = parser.parse_args()
    output = args.output or (OFFLINE_OUTPUT if args.offline else DEFAULT_OUTPUT)
    report = run(output, offline=args.offline)
    checks = [check for scenario in report["scenarios"]
              for row in scenario["turns"] for check in row["checks"].values()]
    goals = [scenario["goal_preserved"] and scenario["correction_preserved"]
             for scenario in report["scenarios"]]
    print(f"Automatic checks: {sum(checks)}/{len(checks)}; "
          f"task state: {sum(goals)}/2; saved to {output}")


if __name__ == "__main__":
    main()
