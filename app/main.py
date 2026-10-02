from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .agents.agent import Agent, AgentInputError, AgentOutputError
from .indexing.rag import DocumentRag, RetrievalSettings
from .indexing.rerank import LocalCrossEncoderReranker
from .indexing.corpus import DEFAULT_PDF
from .indexing.store import DEFAULT_INDEX_DIR, DocumentIndex, DocumentIndexError
from .invariants import (
    InvariantSnapshot, InvariantUpdateRequest, InvariantSettingsConflict,
    SQLiteInvariantRepository,
)
from .memory.extractor import MemoryExtractor
from .orchestration.profile_interviewer import ProfileInterviewer
from .state.task import TaskConflict
from .providers.deepseek import (
    DeepSeekProvider,
    LlmConfigurationError,
    LlmRequestError,
)
from .rag_chat.models import RagSendRequest, RagSession, RagSessionSummary, RagTurn
from .rag_chat.service import GROUNDING_SYSTEM_PROMPT, RagChatService
from .rag_chat.state import TurnInterpreter
from .rag_chat.store import RagChatNotFound, RagTurnConflict, SQLiteRagChatRepository
from .schemas import (
    ChatSendRequest,
    ChatSendResponse,
    ChatSession,
    ChatSessionCreateRequest,
    ChatSessionSummary,
    MemoryCreateRequest,
    MemoryEntry,
    MemorySnapshot,
    UserProfile,
    UserProfileCreateRequest,
    UserProfileUpdateRequest,
    TaskActionRequest, TaskStartRequest,
)
from .services.chat_sessions import (
    ChatSessionService,
)
from .storage.chat_sessions import (
    ChatSessionNotFound,
    DEFAULT_CHAT_DB_PATH,
    SQLiteChatSessionRepository,
)
from .memory.service import (
    MemoryNotFound,
    MemoryService,
    MemoryValidationError,
    SQLiteMemoryRepository,
)
from .orchestration.profiles import (
    DEFAULT_PROFILE_ID,
    ProfileDeletionError,
    ProfileNotFound,
    ProfileService,
    SQLiteProfileRepository,
)


def _allowed_origins() -> list[str]:
    configured_origins = os.getenv("FRONTEND_ORIGINS")
    if not configured_origins:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]
    return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]


app = FastAPI(title="AI Challenge W4 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["DELETE", "GET", "POST", "PUT"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def get_profile_repository() -> SQLiteProfileRepository:
    database_path = os.getenv("CHAT_DB_PATH") or str(DEFAULT_CHAT_DB_PATH)
    return SQLiteProfileRepository(database_path)


@lru_cache(maxsize=1)
def get_chat_repository() -> SQLiteChatSessionRepository:
    database_path = os.getenv("CHAT_DB_PATH") or str(DEFAULT_CHAT_DB_PATH)
    get_profile_repository()
    return SQLiteChatSessionRepository(database_path)


@lru_cache(maxsize=1)
def get_memory_repository() -> SQLiteMemoryRepository:
    database_path = os.getenv("CHAT_DB_PATH") or str(DEFAULT_CHAT_DB_PATH)
    get_chat_repository()
    return SQLiteMemoryRepository(database_path)


def get_chat_session_service() -> ChatSessionService:
    provider = DeepSeekProvider()
    agent = Agent(provider)
    memory_extractor = MemoryExtractor(provider)
    return ChatSessionService(
        get_chat_repository(),
        agent,
        get_memory_repository(),
        memory_extractor,
        get_profile_repository(),
        ProfileInterviewer(provider),
        invariant_repository=get_invariant_repository(),
        document_rag=DocumentRag(
            get_document_index(), DeepSeekProvider(thinking_enabled=False),
            reranker=get_document_reranker(), settings=RetrievalSettings.from_env(),
        ),
    )


@lru_cache(maxsize=1)
def get_rag_chat_service() -> RagChatService:
    database_path = os.getenv("CHAT_DB_PATH") or str(DEFAULT_CHAT_DB_PATH)
    fast_model = DeepSeekProvider(thinking_enabled=False)
    return RagChatService(
        SQLiteRagChatRepository(database_path),
        TurnInterpreter(fast_model),
        DocumentRag(
            get_document_index(), fast_model,
            reranker=get_document_reranker(),
            settings=replace(RetrievalSettings.from_env(), rewrite=False),
        ),
        Agent(DeepSeekProvider(thinking_enabled=False),
              system_prompt=GROUNDING_SYSTEM_PROMPT, max_tokens=3_000),
    )


@lru_cache(maxsize=1)
def get_document_index() -> DocumentIndex:
    root = Path(os.getenv("DOCUMENT_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
    pdf = Path(os.getenv("DOCUMENT_PDF_PATH", str(DEFAULT_PDF)))
    return DocumentIndex(root=root, pdf_path=pdf)


@lru_cache(maxsize=1)
def get_document_reranker() -> LocalCrossEncoderReranker:
    return LocalCrossEncoderReranker()


@app.get("/api/document-index/status")
def document_index_status(index: DocumentIndex = Depends(get_document_index)) -> dict:
    settings = RetrievalSettings.from_env()
    return {
        **index.status(),
        "retrieval": {
            "candidate_k": settings.candidate_k,
            "final_k": settings.final_k,
            "min_relevance_score": settings.min_relevance_score,
            "reranker_model": get_document_reranker().model_name,
        },
    }


@lru_cache(maxsize=1)
def get_invariant_repository() -> SQLiteInvariantRepository:
    database_path = os.getenv("CHAT_DB_PATH") or str(DEFAULT_CHAT_DB_PATH)
    return SQLiteInvariantRepository(database_path)


@app.get("/api/invariants", response_model=InvariantSnapshot)
def get_invariants(
    repository: SQLiteInvariantRepository = Depends(get_invariant_repository),
) -> InvariantSnapshot:
    return repository.get()


@app.put("/api/invariants", response_model=InvariantSnapshot)
def update_invariants(
    request: InvariantUpdateRequest,
    repository: SQLiteInvariantRepository = Depends(get_invariant_repository),
) -> InvariantSnapshot:
    try:
        return repository.update(request)
    except InvariantSettingsConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from None


def get_memory_service() -> MemoryService:
    return MemoryService(
        get_memory_repository(),
        get_chat_repository(),
        get_profile_repository(),
    )


def get_profile_service() -> ProfileService:
    return ProfileService(get_profile_repository())


@app.get("/api/health")
def health() -> dict[str, bool | str]:
    return {
        "status": "ok",
        "deepseek_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
    }


@app.post("/api/chat/sessions", response_model=ChatSession, status_code=201)
def create_chat_session(
    request: ChatSessionCreateRequest | None = None,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSession:
    try:
        return service.create(request.profile_id if request else DEFAULT_PROFILE_ID)
    except ProfileNotFound:
        raise HTTPException(status_code=404, detail="Профиль не найден") from None


@app.get("/api/chat/sessions", response_model=list[ChatSessionSummary])
def list_chat_sessions(
    profile_id: str | None = None,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> list[ChatSessionSummary]:
    try:
        return service.list(profile_id)
    except ProfileNotFound:
        raise HTTPException(status_code=404, detail="Профиль не найден") from None


@app.delete("/api/chat/sessions", status_code=204)
def clear_chat_sessions(
    profile_id: str | None = None,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> Response:
    try:
        service.clear(profile_id)
    except ProfileNotFound:
        raise HTTPException(status_code=404, detail="Профиль не найден") from None
    return Response(status_code=204)


@app.get("/api/profiles", response_model=list[UserProfile])
def list_profiles(
    service: ProfileService = Depends(get_profile_service),
) -> list[UserProfile]:
    return service.list()


@app.post("/api/profiles", response_model=UserProfile, status_code=201)
def create_profile(
    request: UserProfileCreateRequest,
    service: ProfileService = Depends(get_profile_service),
) -> UserProfile:
    return service.create(request)


@app.post("/api/profiles/auto", response_model=UserProfile, status_code=201)
def create_automatic_profile(
    service: ProfileService = Depends(get_profile_service),
) -> UserProfile:
    return service.create_auto()


@app.put("/api/profiles/{profile_id}", response_model=UserProfile)
def update_profile(
    profile_id: str,
    request: UserProfileUpdateRequest,
    service: ProfileService = Depends(get_profile_service),
) -> UserProfile:
    try:
        return service.update(profile_id, request)
    except ProfileNotFound:
        raise HTTPException(status_code=404, detail="Профиль не найден") from None


@app.delete("/api/profiles/{profile_id}", status_code=204)
def delete_profile(
    profile_id: str,
    service: ProfileService = Depends(get_profile_service),
) -> Response:
    try:
        service.delete(profile_id)
    except ProfileNotFound:
        raise HTTPException(status_code=404, detail="Профиль не найден") from None
    except ProfileDeletionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    return Response(status_code=204)


@app.get("/api/memory", response_model=MemorySnapshot)
def get_memory(
    session_id: str | None = None,
    profile_id: str | None = None,
    service: MemoryService = Depends(get_memory_service),
) -> MemorySnapshot:
    try:
        return service.snapshot(session_id, profile_id)
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Чат не найден") from None
    except ProfileNotFound:
        raise HTTPException(status_code=404, detail="Профиль не найден") from None
    except MemoryValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@app.post("/api/memory", response_model=MemoryEntry, status_code=201)
def create_memory(
    request: MemoryCreateRequest,
    service: MemoryService = Depends(get_memory_service),
) -> MemoryEntry:
    try:
        return service.create(request)
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Чат не найден") from None
    except ProfileNotFound:
        raise HTTPException(status_code=404, detail="Профиль не найден") from None
    except MemoryValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@app.delete("/api/memory/{layer}/{memory_id}", status_code=204)
def delete_memory(
    layer: Literal["working", "long_term"],
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
) -> Response:
    try:
        service.delete(layer, memory_id)
    except MemoryNotFound:
        raise HTTPException(status_code=404, detail="Запись памяти не найдена") from None
    return Response(status_code=204)


@app.get("/api/chat/sessions/{session_id}", response_model=ChatSession)
def get_chat_session(
    session_id: str,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSession:
    try:
        return service.get(session_id)
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Чат не найден") from None
    except ProfileNotFound:
        raise HTTPException(status_code=404, detail="Профиль не найден") from None


@app.post(
    "/api/chat/sessions/{session_id}/messages",
    response_model=ChatSendResponse,
)
def send_chat_message(
    session_id: str,
    request: ChatSendRequest,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSendResponse:
    try:
        return service.send(session_id, request.content, request.strategy, request.mode)
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Чат не найден") from None
    except ProfileNotFound:
        raise HTTPException(status_code=404, detail="Профиль не найден") from None
    except TaskConflict as error:
        raise HTTPException(status_code=409, detail=error.detail) from None
    except AgentInputError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except (AgentOutputError, LlmRequestError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from None
    except LlmConfigurationError:
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY не задан") from None
    except DocumentIndexError as error:
        raise HTTPException(status_code=503, detail=str(error)) from None


@app.post("/api/chat/sessions/{session_id}/task", response_model=ChatSession, status_code=201)
def start_task(
    session_id: str,
    request: TaskStartRequest,
    response: Response,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSession:
    try:
        session = service.start_task(session_id, request.task)
        if session.task is None:
            response.status_code = 200
        return session
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Чат не найден") from None
    except TaskConflict as error:
        raise HTTPException(status_code=409, detail=error.detail) from None


@app.post("/api/chat/sessions/{session_id}/task/actions", response_model=ChatSession)
def task_action(
    session_id: str,
    request: TaskActionRequest,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSession:
    try:
        return service.task_action(session_id, request)
    except ChatSessionNotFound:
        raise HTTPException(status_code=404, detail="Чат не найден") from None
    except TaskConflict as error:
        raise HTTPException(status_code=409, detail=error.detail) from None
    except (AgentOutputError, LlmRequestError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from None
    except LlmConfigurationError:
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY не задан") from None


@app.post("/api/rag-chat/sessions", response_model=RagSession, status_code=201)
def create_rag_chat(service: RagChatService = Depends(get_rag_chat_service)) -> RagSession:
    return service.create()


@app.get("/api/rag-chat/sessions", response_model=list[RagSessionSummary])
def list_rag_chats(
    service: RagChatService = Depends(get_rag_chat_service),
) -> list[RagSessionSummary]:
    return service.list()


@app.get("/api/rag-chat/sessions/{session_id}", response_model=RagSession)
def get_rag_chat(
    session_id: str, service: RagChatService = Depends(get_rag_chat_service),
) -> RagSession:
    try:
        return service.get(session_id)
    except RagChatNotFound:
        raise HTTPException(status_code=404, detail="RAG-чат не найден") from None


@app.delete("/api/rag-chat/sessions/{session_id}", status_code=204)
def delete_rag_chat(
    session_id: str, service: RagChatService = Depends(get_rag_chat_service),
) -> Response:
    try:
        service.delete(session_id)
    except RagChatNotFound:
        raise HTTPException(status_code=404, detail="RAG-чат не найден") from None
    return Response(status_code=204)


@app.post("/api/rag-chat/sessions/{session_id}/turns", response_model=RagTurn)
def send_rag_chat_turn(
    session_id: str, request: RagSendRequest,
    service: RagChatService = Depends(get_rag_chat_service),
) -> RagTurn:
    try:
        return service.send(session_id, request.content)
    except RagChatNotFound:
        raise HTTPException(status_code=404, detail="RAG-чат не найден") from None
    except RagTurnConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@app.post("/api/rag-chat/sessions/{session_id}/turns/{turn_id}/retry", response_model=RagTurn)
def retry_rag_chat_turn(
    session_id: str, turn_id: str,
    service: RagChatService = Depends(get_rag_chat_service),
) -> RagTurn:
    try:
        return service.retry(session_id, turn_id)
    except RagChatNotFound:
        raise HTTPException(status_code=404, detail="RAG-чат не найден") from None
    except RagTurnConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from None


STATIC_ROOT = Path(__file__).resolve().parents[1] / "static"
app.mount("/", StaticFiles(directory=STATIC_ROOT, html=True), name="static")
