const sessionsView = document.querySelector("#sessions");
const messagesView = document.querySelector("#messages");
const chatTitle = document.querySelector("#chat-title");
const statusView = document.querySelector("#status");
const composer = document.querySelector("#composer");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const newButton = document.querySelector("#new-chat");
let currentSession = null;
let busy = false;

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "Не удалось выполнить запрос");
  }
  return response.status === 204 ? null : response.json();
}

function element(tag, className = "", content = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = content;
  return node;
}

function sourceLink(source) {
  const link = element("a");
  if (source.kind === "message") {
    link.href = `#turn-${source.message_id}`;
    link.textContent = source.title || "Сообщение пользователя";
    link.addEventListener("click", () => {
      const target = document.getElementById(`turn-${source.message_id}`);
      target?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  } else {
    link.href = `${source.source}#page=${source.page_start}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = `${source.section} · стр. ${source.page_start} · ${source.chunk_id}`;
  }
  return link;
}

function renderSources(turn) {
  const box = element("div", "source-box");
  box.append(element("strong", "", "Источники"));
  if (turn.sources.length) {
    const list = element("div", "source-list");
    list.append(...turn.sources.map(sourceLink));
    box.append(list);
  } else {
    const message = turn.status === "done"
      ? "Нет подтверждённых источников."
      : "Поиск источников не завершён.";
    box.append(element("span", "no-source", message));
  }
  if (turn.citations.length) {
    const quotes = element("div", "quote-list");
    for (const citation of turn.citations) {
      quotes.append(element("blockquote", "", `«${citation.quote}»`));
    }
    box.append(quotes);
  }
  return box;
}

async function retry(turn) {
  if (busy || !currentSession) return;
  setBusy(true, "Повторяю ответ…");
  try {
    await api(`/api/rag-chat/sessions/${currentSession.id}/turns/${turn.id}/retry`, { method: "POST" });
    await openSession(currentSession.id);
  } catch (error) {
    statusView.textContent = error.message;
  } finally {
    setBusy(false);
  }
}

function renderMessages() {
  messagesView.replaceChildren();
  if (!currentSession?.turns.length) {
    const empty = element("div", "empty");
    empty.append(element("span", "empty-mark", "✦"));
    empty.append(element("h2", "", "Спросите по книге"));
    empty.append(element("p", "", "Сформулируйте цель, уточните ограничения или сразу задайте вопрос. История, память задачи и источники сохраняются в этом чате."));
    messagesView.append(empty);
    return;
  }
  for (const turn of currentSession.turns) {
    const pair = element("div", "turn");
    const user = element("article", "bubble user");
    user.id = `turn-${turn.id}`;
    user.append(element("span", "bubble-label", `ВЫ · ${turn.position}`));
    user.append(element("p", "answer-text", turn.content));
    pair.append(user);

    const assistant = element("article", "bubble assistant");
    assistant.append(element("span", "bubble-label", "RAG-ЧАТ"));
    const message = turn.status === "failed"
      ? turn.error || "Не удалось ответить."
      : turn.status === "pending" ? "Готовлю ответ…" : turn.answer;
    assistant.append(element("p", "answer-text", message));
    assistant.append(renderSources(turn));
    if (turn.status !== "done" && turn === currentSession.turns.at(-1)) {
      const button = element("button", "retry-button", "Повторить ответ");
      button.type = "button";
      button.addEventListener("click", () => retry(turn));
      assistant.append(button);
    }
    pair.append(assistant);
    messagesView.append(pair);
  }
  messagesView.scrollTop = messagesView.scrollHeight;
}

function renderFact(target, fact) {
  const container = document.querySelector(target);
  container.replaceChildren();
  const facts = Array.isArray(fact) ? fact : fact ? [fact] : [];
  if (!facts.length) {
    container.append(element("p", "state-empty", "Пока не зафиксировано"));
    return;
  }
  for (const item of facts) {
    const card = element("div", "fact");
    if (item.key !== "goal") card.append(element("strong", "", item.key));
    card.append(element("p", "", item.value));
    card.append(sourceLink({
      kind: "message", message_id: item.source_message_id,
      title: "Показать исходное сообщение →",
    }));
    container.append(card);
  }
}

function renderState() {
  const state = currentSession?.state || {
    goal: null, constraints: [], terms: [], clarifications: [], revision: 0,
  };
  document.querySelector("#state-revision").textContent = `v${state.revision}`;
  renderFact("#state-goal", state.goal);
  renderFact("#state-constraints", state.constraints);
  renderFact("#state-terms", state.terms);
  renderFact("#state-clarifications", state.clarifications);
}

async function refreshSessions() {
  const sessions = await api("/api/rag-chat/sessions");
  sessionsView.replaceChildren();
  for (const session of sessions) {
    const button = element("button", `session-link${currentSession?.id === session.id ? " active" : ""}`, session.title);
    button.type = "button";
    button.addEventListener("click", () => openSession(session.id));
    sessionsView.append(button);
  }
  return sessions;
}

async function openSession(id) {
  currentSession = await api(`/api/rag-chat/sessions/${id}`);
  localStorage.setItem("ragChatSessionId", id);
  chatTitle.textContent = currentSession.title;
  renderMessages();
  renderState();
  updateComposer();
  await refreshSessions();
}

async function createSession() {
  const session = await api("/api/rag-chat/sessions", { method: "POST" });
  await openSession(session.id);
  input.focus();
}

function setBusy(value, statusText = "Готово") {
  busy = value;
  newButton.disabled = value;
  statusView.textContent = statusText;
  updateComposer();
}

function updateComposer() {
  const unresolved = currentSession?.turns.at(-1)?.status !== "done"
    && Boolean(currentSession?.turns.length);
  input.disabled = busy || unresolved;
  sendButton.disabled = busy || unresolved;
  if (!busy && unresolved) {
    statusView.textContent = "Повторите последний ответ перед новым сообщением";
  }
}

async function send(content) {
  if (busy) return;
  setBusy(true, "Ищу контекст и готовлю ответ…");
  try {
    if (!currentSession) await createSession();
    await api(`/api/rag-chat/sessions/${currentSession.id}/turns`, {
      method: "POST", body: JSON.stringify({ content }),
    });
    input.value = "";
    await openSession(currentSession.id);
  } catch (error) {
    statusView.textContent = error.message;
  } finally {
    setBusy(false);
    input.focus();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (content) void send(content);
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    composer.requestSubmit();
  }
});
newButton.addEventListener("click", () => {
  if (!busy) void createSession().catch((error) => { statusView.textContent = error.message; });
});

async function initialize() {
  try {
    const sessions = await refreshSessions();
    const requested = new URLSearchParams(window.location.search).get("session");
    const saved = localStorage.getItem("ragChatSessionId");
    const selected = sessions.find((session) => session.id === requested)
      || sessions.find((session) => session.id === saved) || sessions[0];
    if (selected) await openSession(selected.id);
    else await createSession();
    const index = await api("/api/document-index/status");
    document.querySelector("#index-status").textContent = index.state === "ready"
      ? `${index.index.pages} стр. · индекс готов`
      : "Индекс создастся при первом вопросе";
  } catch (error) {
    statusView.textContent = error.message;
  }
}

void initialize();
