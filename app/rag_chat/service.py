"""One user turn through task memory, retrieval, grounding, and durable history."""

from __future__ import annotations

import logging
from threading import Lock

from ..agents.agent import Agent, AgentMessage, AgentOutputError
from ..indexing.grounding import GroundedAnswer, GroundingError, validate_grounded_answer
from ..indexing.rag import DocumentRag
from ..indexing.store import SearchHit
from ..memory.context import MemoryContext
from .models import RagSession, RagSessionSummary, RagTaskState, RagTurn, StateFact
from .state import InterpretationError, TurnDecision, TurnInterpreter, apply_decision
from .store import RagTurnConflict, SQLiteRagChatRepository


logger = logging.getLogger(__name__)


class RagChatService:
    """The dedicated chat's public interface; one session is serialized per turn."""

    def __init__(
        self, repository: SQLiteRagChatRepository, interpreter: TurnInterpreter,
        rag: DocumentRag, agent: Agent, *, history_turns: int = 4,
    ) -> None:
        self._repository = repository
        self._interpreter = interpreter
        self._rag = rag
        self._agent = agent
        self._history_turns = history_turns
        self._locks: dict[str, Lock] = {}
        self._locks_guard = Lock()

    def _lock(self, session_id: str) -> Lock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, Lock())

    def create(self) -> RagSession:
        return self._repository.create()

    def list(self) -> list[RagSessionSummary]:
        return self._repository.list()

    def get(self, session_id: str) -> RagSession:
        return self._repository.get(session_id)

    def delete(self, session_id: str) -> None:
        with self._lock(session_id):
            self._repository.delete(session_id)

    def send(self, session_id: str, content: str) -> RagTurn:
        if not content.strip() or len(content) > 12_000:
            raise ValueError("Сообщение должно содержать от 1 до 12000 символов")
        with self._lock(session_id):
            session = self._repository.get(session_id)
            if session.turns and session.turns[-1].status != "done":
                raise RagTurnConflict("Повторите последний ответ перед новым сообщением")
            turn = self._repository.start_turn(session_id, content)
            return self._process(session, turn)

    def retry(self, session_id: str, turn_id: str) -> RagTurn:
        with self._lock(session_id):
            session = self._repository.get(session_id)
            turn = next((item for item in session.turns if item.id == turn_id), None)
            if turn is None:
                raise RagTurnConflict("Сообщение не принадлежит этому RAG-чату")
            if turn != session.turns[-1] or turn.status not in {"pending", "failed"}:
                raise RagTurnConflict("Повторить можно только последнее сообщение без ответа")
            return self._process(session, turn)

    def _process(self, session: RagSession, turn: RagTurn) -> RagTurn:
        try:
            decision = self._repository.decision(turn.id)
            if decision is None:
                decision = self._interpreter.interpret(
                    turn.content, session.state, self._recent_payload(session),
                )
                state = apply_decision(session.state, decision, turn.id)
                turn = self._repository.save_decision(
                    turn.id, decision, state, session.state.revision,
                )
            else:
                state = session.state
            answer, sources, citations, grounding_status = self._answer(
                session, turn, decision, state,
            )
            return self._repository.finish_turn(
                turn.id, answer, sources=sources, citations=citations,
                grounding_status=grounding_status,
            )
        except Exception as error:
            logger.warning("RAG chat turn failed: %s", type(error).__name__, exc_info=True)
            return self._repository.fail_turn(turn.id, self._public_error(error))

    @staticmethod
    def _public_error(error: Exception) -> str:
        if isinstance(error, InterpretationError):
            return str(error)
        return "Не удалось ответить. Повторите попытку."

    def _recent_payload(self, session: RagSession) -> list[dict]:
        return [
            {"user": turn.content, "assistant": turn.answer}
            for turn in session.turns if turn.status == "done"
        ][-self._history_turns:]

    def _context(self, session: RagSession) -> list[AgentMessage]:
        context: list[AgentMessage] = []
        for turn in [item for item in session.turns if item.status == "done"][-self._history_turns:]:
            context.append({"role": "user", "content": turn.content})
            context.append({"role": "assistant", "content": turn.answer})
        return context

    @staticmethod
    def _memory(state: RagTaskState) -> MemoryContext:
        entries = []
        if state.goal is not None:
            entries.append({"category": "goal", "content": state.goal.value})
        for category, facts in (
            ("constraint", state.constraints),
            ("term", state.terms),
            ("clarification", state.clarifications),
        ):
            entries.extend(
                {"category": category, "content": f"{fact.key}: {fact.value}"}
                for fact in facts
            )
        return MemoryContext(working=tuple(entries))

    @staticmethod
    def _message_source(fact: StateFact) -> dict:
        return {
            "kind": "message", "message_id": fact.source_message_id,
            "title": f"Сообщение пользователя · {fact.key}",
        }

    def _state_answer(
        self, decision: TurnDecision, state: RagTaskState,
    ) -> tuple[str, list[dict]]:
        if decision.question_scope == "goal":
            if state.goal is None:
                return "Цель пока не зафиксирована.", []
            return f"Цель: {state.goal.value}", [self._message_source(state.goal)]
        facts = ([state.goal] if state.goal is not None else []) + [
            *state.constraints, *state.terms, *state.clarifications,
        ]
        if not facts:
            return "Договорённости пока не зафиксированы.", []
        lines = [f"{fact.key}: {fact.value}" for fact in facts]
        sources = list({
            fact.source_message_id: self._message_source(fact) for fact in facts
        }.values())
        return "\n".join(lines), sources

    def _document_answer(
        self, session: RagSession, turn: RagTurn, state: RagTaskState,
        hits: list[SearchHit],
    ) -> GroundedAnswer:
        if not hits:
            return GroundedAnswer.unknown()
        for retry in (False, True):
            try:
                raw = self._agent.respond_json(
                    self._context(session), turn.content,
                    memory=self._memory(state),
                    document_context=self._rag.grounded_prompt(hits, retry=retry),
                )
                return validate_grounded_answer(raw, hits)
            except (AgentOutputError, GroundingError):
                if retry:
                    return GroundedAnswer.failed()
        return GroundedAnswer.failed()

    @staticmethod
    def _named_terms(query: str) -> tuple[str, ...]:
        # Prefix a space so a leading identifier is not treated as an ordinary
        # sentence-initial word by the shared retrieval helper.
        return tuple(term for term in DocumentRag._required_terms(" " + query)
                     if len(term) > 2 and term.casefold() not in {
                         "compare", "describe", "summarize", "discuss", "define",
                     })

    def _retrieve(self, decision: TurnDecision) -> list[SearchHit]:
        hits = self._rag.retrieve(
            decision.search_question, [], "structure", force=True,
        )
        if hits or decision.question_scope != "document":
            return hits
        terms = self._named_terms(decision.search_question)
        if len(terms) < 2:
            return hits
        # The existing lexical filter requires every named identifier in one
        # chunk. A comparison may need passages from separate sections.
        collected: dict[str, SearchHit] = {}
        for term in terms[:3]:
            for hit in self._rag.retrieve(term, [], "structure", force=True):
                collected.setdefault(hit.metadata["chunk_id"], hit)
        return list(collected.values())[:self._rag.settings.final_k * 2]

    def _answer(
        self, session: RagSession, turn: RagTurn, decision: TurnDecision,
        state: RagTaskState,
    ) -> tuple[str, list[dict], list[dict], str]:
        if decision.kind == "clarification_needed":
            return decision.clarification, [self._message_source(StateFact(
                key="уточнение", value=turn.content, source_message_id=turn.id,
            ))], [], "clarification_needed"
        if decision.kind == "statement":
            content = (
                "Уточнение сохранено."
                if state.revision > session.state.revision else "Принято."
            )
            return content, [self._message_source(StateFact(
                key="уточнение", value=turn.content, source_message_id=turn.id,
            ))], [], "statement"

        # Every question searches the indexed corpus, including questions about
        # the user's own agreements. Only relevant evidence appears in the answer.
        hits = self._retrieve(decision)
        if decision.question_scope in {"goal", "state"}:
            content, sources = self._state_answer(decision, state)
            return content, sources, [], "state"
        grounded = self._document_answer(session, turn, state, hits)
        content = grounded.content
        if grounded.status == "answered":
            available = " ".join(
                hit.metadata["section"] + " " + hit.metadata["text"]
                for hit in hits
            ).casefold()
            missing = [term for term in self._named_terms(decision.search_question)
                       if term.casefold() not in available]
            if missing:
                content += (
                    "\nПо темам " + ", ".join(missing)
                    + " в найденных фрагментах нет подтверждения."
                )
        return (
            content,
            [{"kind": "document", **source} for source in grounded.sources],
            grounded.citations,
            grounded.status,
        )
