(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const SESSION_KEY = "university-agent.recent-sessions.v1";
  const MAX_RECENT = 12;
  const views = ["chat", "knowledge", "learning", "settings"];
  const codeBlocks = [];
  let sessionId = null;
  let busy = false;
  let toastTimer = null;
  let currentView = "chat";
  let systemCache = null;

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[character]);
  }

  function inlineMarkdown(value) {
    const inlineCode = [];
    let text = value.replace(/`([^`]+)`/g, (_, code) => {
      const token = `@@INLINECODE${inlineCode.length}@@`;
      inlineCode.push(`<code>${code}</code>`);
      return token;
    });
    text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, label, url) =>
      `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`
    );
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return text.replace(/@@INLINECODE(\d+)@@/g, (_, index) => inlineCode[Number(index)] || "");
  }

  function highlightCode(source, language) {
    const escaped = escapeHtml(source);
    const lang = String(language || "").toLowerCase();
    const keywords = lang.includes("py")
      ? "and|as|async|await|class|def|elif|else|except|False|for|from|if|import|in|is|lambda|None|not|or|pass|raise|return|True|try|while|with|yield"
      : lang.includes("json") ? "true|false|null"
      : "const|let|var|function|return|async|await|class|new|if|else|for|while|try|catch|throw|import|from|export|default|true|false|null|undefined|def|pass|None|True|False";
    const tokenPattern = new RegExp(`("(?:\\\\.|[^"\\\\])*"|'(?:\\\\.|[^'\\\\])*'|\\b(?:${keywords})\\b|\\b\\d+(?:\\.\\d+)?\\b|(?:^|\\s)(?:#|//).*?$)`, "gm");
    return escaped.replace(tokenPattern, (token) => {
      let type = "number";
      if (/^("|')/.test(token)) type = "string";
      else if (/^(#|\/\/)/.test(token) || /\s(#|\/\/)/.test(token)) type = "comment";
      else if (new RegExp(`^(${keywords})$`).test(token)) type = "keyword";
      return `<span class="syntax-${type}">${token}</span>`;
    });
  }

  // Small, dependency-free Markdown renderer. All source HTML is escaped first;
  // only elements generated here are inserted into the page.
  function renderMarkdown(source) {
    codeBlocks.length = 0;
    let text = String(source ?? "").replace(/```([^\n`]*)\n([\s\S]*?)```/g, (_, language, code) => {
      const index = codeBlocks.push(code) - 1;
      const token = `@@BLOCKCODE${index}@@`;
      return `\n${token}\n`;
    });
    text = escapeHtml(text);
    const lines = text.split(/\r?\n/);
    const output = [];
    let i = 0;

    const cells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
    const isTableRule = (line) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
    const isSpecial = (line) => !line.trim() || /^@@BLOCKCODE\d+@@$/.test(line.trim()) || /^#{1,3}\s/.test(line) || /^>\s?/.test(line) || /^[-*]\s+/.test(line) || /^\d+\.\s+/.test(line) || /^---+$/.test(line) || (i + 1 < lines.length && line.includes("|") && isTableRule(lines[i + 1]));

    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i += 1; continue; }
      const blockCode = line.trim().match(/^@@BLOCKCODE(\d+)@@$/);
      if (blockCode) {
        const index = Number(blockCode[1]);
        const raw = codeBlocks[index] ?? "";
        const langMatch = source.match(/```([^\n`]*)\n[\s\S]*?```/g);
        const codeLang = langMatch?.[index]?.match(/^```([^\n`]*)/)?.[1]?.trim() || "code";
        output.push(`<div class="code-block"><div class="code-toolbar"><span>${escapeHtml(codeLang || "code")}</span><button type="button" data-copy-code="${index}">Copy code</button></div><pre><code>${highlightCode(raw.replace(/\n$/, ""), codeLang)}</code></pre></div>`);
        i += 1; continue;
      }
      if (/^---+$/.test(line)) { output.push("<hr>"); i += 1; continue; }
      const heading = line.match(/^(#{1,3})\s+(.*)$/);
      if (heading) { const level = heading[1].length; output.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`); i += 1; continue; }
      if (/^>\s?/.test(line)) {
        const quote = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) quote.push(lines[i++].replace(/^>\s?/, ""));
        output.push(`<blockquote>${inlineMarkdown(quote.join("<br>"))}</blockquote>`); continue;
      }
      if (i + 1 < lines.length && line.includes("|") && isTableRule(lines[i + 1])) {
        const headers = cells(line); i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(cells(lines[i++]));
        output.push(`<table><thead><tr>${headers.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${headers.map((_, index) => `<td>${inlineMarkdown(row[index] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
        continue;
      }
      if (/^[-*]\s+/.test(line) || /^\d+\.\s+/.test(line)) {
        const ordered = /^\d+\.\s+/.test(line); const tag = ordered ? "ol" : "ul"; const items = [];
        const pattern = ordered ? /^\d+\.\s+/ : /^[-*]\s+/;
        while (i < lines.length && pattern.test(lines[i])) items.push(`<li>${inlineMarkdown(lines[i++].replace(pattern, ""))}</li>`);
        output.push(`<${tag}>${items.join("")}</${tag}>`); continue;
      }
      const paragraph = [line]; i += 1;
      while (i < lines.length && !isSpecial(lines[i])) paragraph.push(lines[i++]);
      output.push(`<p>${inlineMarkdown(paragraph.join("<br>"))}</p>`);
    }
    return output.join("");
  }

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    let data;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) data = await response.json();
    else data = await response.text();
    if (!response.ok) {
      const message = typeof data === "object" && data?.detail ? data.detail : `Request failed (${response.status})`;
      throw new Error(message);
    }
    return data;
  }

  function showToast(message, isError = false) {
    const toast = $("#toast");
    toast.textContent = message;
    toast.classList.toggle("error", isError);
    toast.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("visible"), 3400);
  }

  function getRecent() {
    try { return JSON.parse(localStorage.getItem(SESSION_KEY) || "[]").filter((item) => item && item.id); }
    catch { return []; }
  }

  function setRecent(items) {
    try { localStorage.setItem(SESSION_KEY, JSON.stringify(items.slice(0, MAX_RECENT))); } catch { /* Browser storage may be disabled. */ }
    renderRecent();
  }

  function rememberSession(id, title, mode) {
    const existing = getRecent().filter((item) => item.id !== id);
    existing.unshift({ id, title: String(title).slice(0, 80), mode, updated: Date.now() });
    setRecent(existing);
  }

  function renderRecent() {
    const list = $("#recent-list");
    list.replaceChildren();
    const recent = getRecent();
    $("#empty-recent").hidden = recent.length > 0;
    recent.forEach((item) => {
      const button = document.createElement("button");
      button.className = `recent-session${item.id === sessionId ? " active" : ""}`;
      button.type = "button";
      button.textContent = item.title || "New conversation";
      button.title = item.title || "New conversation";
      button.addEventListener("click", () => loadSession(item.id));
      list.append(button);
    });
  }

  function setView(view) {
    if (!views.includes(view)) return;
    currentView = view;
    views.forEach((name) => { $(`#${name}-view`).hidden = name !== view; });
    $$(".nav-item[data-view]").forEach((button) => {
      const active = button.dataset.view === view;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
    });
    const titles = { chat: "Chat", knowledge: "Knowledge", learning: "Learning", settings: "System & approvals" };
    $("#page-title").textContent = titles[view];
    closeSidebar();
    if (view === "knowledge") loadDocuments();
    if (view === "learning") loadLearning();
    if (view === "settings") { loadSystemDetails(); loadApprovals(); }
  }

  function openSidebar() {
    $("#sidebar").classList.add("open");
    $("#mobile-scrim").hidden = false;
    $("#menu-toggle").setAttribute("aria-expanded", "true");
  }

  function closeSidebar() {
    $("#sidebar").classList.remove("open");
    $("#mobile-scrim").hidden = true;
    $("#menu-toggle").setAttribute("aria-expanded", "false");
  }

  function showWelcome(show) {
    $("#welcome").hidden = !show;
    $("#message-list").hidden = show;
  }

  function addMessage(role, content, citations = [], meta = {}) {
    showWelcome(false);
    const list = $("#message-list");
    const article = document.createElement("article");
    article.className = `message ${role}${meta.error ? " error" : ""}`;
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = role === "user" ? "YOU" : "UA";
    const body = document.createElement("div");
    body.className = "message-body";

    if (role === "assistant") {
      const label = document.createElement("div");
      label.className = "message-meta";
      label.textContent = meta.pending ? "REQUEST SENT · WAITING FOR RESPONSE" : meta.error ? "REQUEST ERROR" : `UNIVERSITY AGENT${meta.model ? ` · ${meta.provider || "local"} / ${meta.model}` : ""}`;
      body.append(label);
    }

    const contentNode = document.createElement("div");
    contentNode.className = "rendered-content";
    if (role === "user") contentNode.textContent = content;
    else contentNode.innerHTML = renderMarkdown(content);
    body.append(contentNode);

    if (role === "assistant" && citations?.length) body.append(renderCitations(citations));
    if (role === "assistant" && !meta.pending) {
      const actions = document.createElement("div");
      actions.className = "message-copy";
      const copy = document.createElement("button");
      copy.type = "button"; copy.textContent = "Copy response";
      copy.addEventListener("click", () => copyText(content, copy));
      actions.append(copy); body.append(actions);
    }
    if (role === "user") { article.append(body, avatar); } else { article.append(avatar, body); }
    list.append(article);
    $("#conversation").scrollTop = $("#conversation").scrollHeight;
    return article;
  }

  function renderCitations(citations) {
    const container = document.createElement("div");
    container.className = "citation-list";
    container.setAttribute("aria-label", "Retrieved source citations");
    citations.forEach((citation, index) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "citation-chip";
      const file = citation.filename || "Source document";
      const page = citation.page_number ? ` · p.${citation.page_number}` : "";
      chip.textContent = `[${citation.index || index + 1}] ${file}${page}`;
      const detail = document.createElement("div");
      detail.className = "citation-detail";
      detail.hidden = true;
      const section = citation.heading ? ` · ${citation.heading}` : "";
      const meta = document.createElement("div");
      meta.textContent = `${file}${page}${section}`;
      detail.append(meta);
      if (citation.snippet) {
        const snippet = document.createElement("p"); snippet.className = "citation-snippet"; snippet.textContent = citation.snippet; detail.append(snippet);
      }
      if (citation.chunk_id) {
        const id = document.createElement("code"); id.textContent = `Chunk ${citation.chunk_id}`; detail.append(id);
      }
      chip.addEventListener("click", () => { detail.hidden = !detail.hidden; });
      container.append(chip, detail);
    });
    return container;
  }

  async function copyText(value, button) {
    try {
      await navigator.clipboard.writeText(value);
      const prior = button.textContent; button.textContent = "Copied"; setTimeout(() => { button.textContent = prior; }, 1200);
    } catch { showToast("Clipboard access is unavailable in this browser.", true); }
  }

  function newChat() {
    sessionId = null;
    $("#message-list").replaceChildren();
    showWelcome(true);
    renderRecent();
    $("#message-input").focus();
    setView("chat");
  }

  async function loadSession(id) {
    sessionId = id;
    $("#message-list").replaceChildren();
    showWelcome(false);
    renderRecent();
    setView("chat");
    const loading = addMessage("assistant", "Loading this conversation…", [], { pending: true });
    try {
      const history = await api(`/api/chat/history/${encodeURIComponent(id)}`);
      loading.remove();
      if (!history.length) { showWelcome(true); showToast("No saved messages were found for that conversation.", true); return; }
      history.forEach((item) => addMessage(item.role === "user" ? "user" : "assistant", item.content || "", item.citations || [], { model: item.mode, provider: "saved" }));
    } catch (error) {
      loading.remove(); addMessage("assistant", `Could not load this conversation: ${error.message}`, [], { error: true });
    }
  }

  function setBusy(value) {
    busy = value;
    const button = $("#send-button");
    button.disabled = value;
    button.setAttribute("aria-busy", String(value));
    button.querySelector("span:first-child").textContent = value ? "Waiting" : "Send";
    $("#message-input").disabled = value;
  }

  async function sendMessage(event) {
    event?.preventDefault();
    if (busy) return;
    const input = $("#message-input");
    const message = input.value.trim();
    if (!message) { input.focus(); return; }
    const mode = $("#mode-select").value;
    const requestSession = sessionId || crypto.randomUUID();
    sessionId = requestSession;
    addMessage("user", message);
    input.value = ""; resizeInput();
    rememberSession(requestSession, message, mode);
    const pending = addMessage("assistant", "The request has been sent. Waiting for the model response…", [], { pending: true });
    setBusy(true);
    try {
      const response = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, session_id: requestSession, mode, use_rag: true, is_private: true })
      });
      pending.remove();
      rememberSession(response.session_id || requestSession, message, mode);
      if (sessionId === requestSession) addMessage("assistant", response.reply || "", response.citations || [], { model: response.model, provider: response.provider });
    } catch (error) {
      pending.remove();
      if (sessionId === requestSession) addMessage("assistant", `I couldn't complete that request. ${error.message}`, [], { error: true });
    } finally { setBusy(false); }
  }

  function resizeInput() {
    const input = $("#message-input"); input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
  }

  function setConnection(status, text) {
    const dot = $("#status-dot"); dot.className = `status-dot ${status}`;
    $("#status-label").textContent = text;
  }

  async function loadHealth() {
    try {
      const health = await api("/api/health");
      systemCache = health;
      if (health.ollama_available) setConnection("online", "Local model available");
      else setConnection("offline", "Local model unavailable");
    } catch { systemCache = null; setConnection("offline", "System status unavailable"); }
  }

  async function loadDocuments() {
    const list = $("#document-list");
    list.innerHTML = '<div class="loading-row">Loading your library…</div>';
    $("#library-note").textContent = "";
    try {
      const docs = await api("/api/documents");
      list.replaceChildren();
      if (!docs.length) { list.innerHTML = '<div class="empty-row">No documents indexed yet. Add a course file above to get started.</div>'; return; }
      docs.forEach((doc) => {
        const card = document.createElement("article"); card.className = "document-card";
        const icon = document.createElement("div"); icon.className = "file-icon"; icon.textContent = (doc.filename || "file").split(".").pop().slice(0, 4);
        const details = document.createElement("div");
        const name = document.createElement("div"); name.className = "document-name"; name.textContent = doc.filename || "Unnamed document"; name.title = name.textContent;
        const meta = document.createElement("div"); meta.className = "document-meta"; meta.textContent = [doc.subject, `${doc.total_pages || 1} page${doc.total_pages === 1 ? "" : "s"}`].filter(Boolean).join(" · ");
        details.append(name, meta);
        const count = document.createElement("div"); count.className = "document-count"; count.textContent = `${doc.total_chunks || 0} chunks`;
        card.append(icon, details, count); list.append(card);
      });
    } catch (error) { list.innerHTML = `<div class="empty-row error-text">${escapeHtml(error.message)}</div>`; }
  }

  async function uploadDocument(file) {
    if (!file) return;
    const allowed = /\.(pdf|docx?|txt|md)$/i.test(file.name);
    if (!allowed) { showToast("Choose a PDF, DOCX, TXT or Markdown file.", true); return; }
    showToast(`Uploading and indexing ${file.name}…`);
    const body = new FormData(); body.append("file", file);
    try {
      const result = await api("/api/documents/upload", { method: "POST", body });
      showToast(`${result.filename} is indexed and ready to use.`);
      await loadDocuments();
    } catch (error) { showToast(error.message, true); }
    finally { $("#document-input").value = ""; $("#library-file-input").value = ""; }
  }

  async function loadLearning() {
    const stats = $("#learning-stats");
    const topics = $("#topic-list");
    const tasks = $("#task-list");
    stats.innerHTML = '<div class="loading-row">Loading your progress…</div>';
    try {
      const [progress, taskRows] = await Promise.all([api("/api/memory/progress"), api("/api/memory/tasks")]);
      stats.innerHTML = "";
      const statItems = [
        [progress.total_quizzes_taken ?? 0, "Quizzes completed"],
        [`${Math.round((progress.average_score || 0) * 100)}%`, "Average quiz score"],
        [progress.total_correct ?? 0, "Correct answers"]
      ];
      statItems.forEach(([value, label]) => {
        const card = document.createElement("div"); card.className = "stat-card";
        const number = document.createElement("div"); number.className = "stat-value"; number.textContent = value;
        const caption = document.createElement("div"); caption.className = "stat-label"; caption.textContent = label;
        card.append(number, caption); stats.append(card);
      });
      topics.replaceChildren();
      const topicRows = progress.topic_breakdown || [];
      if (!topicRows.length) topics.innerHTML = '<div class="empty-row">No topics recorded yet.</div>';
      topicRows.forEach((topic) => {
        const row = document.createElement("div"); row.className = "topic-row";
        const dot = document.createElement("span"); dot.className = "topic-indicator";
        const name = document.createElement("span"); name.className = "topic-name"; name.textContent = topic.name || "Topic";
        const level = document.createElement("span"); level.className = "topic-level"; level.textContent = `${Math.round((topic.mastery_level || 0) * 100)}%`;
        row.append(dot, name, level); topics.append(row);
      });
      tasks.replaceChildren();
      if (!taskRows.length) tasks.innerHTML = '<div class="empty-row">No study tasks recorded yet.</div>';
      taskRows.forEach((task) => {
        const row = document.createElement("label"); row.className = `task-row${task.status === "completed" ? " done" : ""}`;
        const checkbox = document.createElement("input"); checkbox.className = "task-check"; checkbox.type = "checkbox"; checkbox.checked = task.status === "completed"; checkbox.setAttribute("aria-label", `Mark ${task.title} ${checkbox.checked ? "pending" : "completed"}`);
        checkbox.addEventListener("change", () => updateTask(task, checkbox.checked));
        const nameWrap = document.createElement("span"); nameWrap.className = "task-name"; nameWrap.textContent = task.title || "Study task";
        const meta = document.createElement("small"); meta.className = "task-meta"; meta.textContent = [task.deadline, task.priority].filter(Boolean).join(" · "); nameWrap.append(meta);
        row.append(checkbox, nameWrap); tasks.append(row);
      });
    } catch (error) {
      stats.innerHTML = `<div class="empty-row error-text">${escapeHtml(error.message)}</div>`;
      topics.replaceChildren(); tasks.replaceChildren();
    }
  }

  async function updateTask(task, completed) {
    try {
      await api(`/api/memory/tasks/${encodeURIComponent(task.id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: completed ? "completed" : "pending" }) });
      await loadLearning();
    } catch (error) { showToast(error.message, true); await loadLearning(); }
  }

  async function loadSystemDetails() {
    const container = $("#system-details"); container.innerHTML = '<div class="loading-row">Checking services…</div>';
    try {
      const [health, status] = await Promise.all([api("/api/health"), api("/api/system/status")]);
      systemCache = health;
      setConnection(health.ollama_available ? "online" : "offline", health.ollama_available ? "Local model available" : "Local model unavailable");
      const embedding = status.embedding || {};
      const index = status.vector_index || {};
      const rows = [
        ["Ollama service", health.ollama_available ? "Available" : "Unavailable", health.ollama_available],
        ["Chat model", health.local_model || status.active_local_model || "Unknown", null],
        ["Embedding provider", `${embedding.provider || "unknown"} · ${embedding.model || "unknown"}`, null],
        ["Embedding type", embedding.is_neural ? "Semantic vectors" : "Lexical / hash fallback", embedding.is_neural],
        ["Vector index", `${index.total_documents ?? 0} documents · ${index.total_chunks ?? 0} chunks`, null],
        ["Cloud routing", status.cloud_routing_enabled ? "Enabled" : "Disabled", null]
      ];
      container.replaceChildren();
      rows.forEach(([label, value, positive]) => {
        const row = document.createElement("div"); row.className = "system-row";
        const name = document.createElement("span"); name.textContent = label;
        const result = document.createElement("span"); result.className = `system-value${positive === null ? "" : positive ? " online" : " offline"}`; result.textContent = value;
        row.append(name, result); container.append(row);
      });
    } catch (error) { container.innerHTML = `<div class="empty-row error-text">${escapeHtml(error.message)}</div>`; }
  }

  async function loadApprovals() {
    const container = $("#approval-list"); container.innerHTML = '<div class="loading-row">Loading approvals…</div>';
    try {
      const approvals = await api("/api/security/approvals");
      $("#approval-count").textContent = String(approvals.length);
      container.replaceChildren();
      if (!approvals.length) { container.innerHTML = '<div class="empty-row">No pending actions need your decision.</div>'; return; }
      approvals.forEach((approval) => {
        const card = document.createElement("article"); card.className = "approval-card";
        const title = document.createElement("div"); title.className = "approval-title"; title.textContent = approval.action || "Action approval";
        const description = document.createElement("div"); description.className = "approval-description"; description.textContent = approval.description || JSON.stringify(approval.params || {});
        const actions = document.createElement("div"); actions.className = "approval-actions";
        const approve = document.createElement("button"); approve.className = "approve-button"; approve.type = "button"; approve.textContent = "Approve";
        const reject = document.createElement("button"); reject.className = "reject-button"; reject.type = "button"; reject.textContent = "Reject";
        approve.addEventListener("click", () => decideApproval(approval.id, "approve"));
        reject.addEventListener("click", () => decideApproval(approval.id, "reject"));
        actions.append(approve, reject); card.append(title, description, actions); container.append(card);
      });
    } catch (error) { $("#approval-count").textContent = "!"; container.innerHTML = `<div class="empty-row error-text">${escapeHtml(error.message)}</div>`; }
  }

  async function decideApproval(id, decision) {
    try {
      await api(`/api/security/approvals/${encodeURIComponent(id)}/${decision}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
      showToast(decision === "approve" ? "Action approved." : "Action rejected.");
      await loadApprovals();
    } catch (error) { showToast(error.message, true); await loadApprovals(); }
  }

  function init() {
    $("#composer").addEventListener("submit", sendMessage);
    $("#message-input").addEventListener("input", resizeInput);
    $("#message-input").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(); }
    });
    $("#new-chat").addEventListener("click", newChat);
    $$(".nav-item[data-view]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
    $$(".prompt-card").forEach((button) => button.addEventListener("click", () => { $("#message-input").value = button.dataset.prompt; resizeInput(); $("#message-input").focus(); }));
    $("#menu-toggle").addEventListener("click", () => $("#sidebar").classList.contains("open") ? closeSidebar() : openSidebar());
    $("#mobile-scrim").addEventListener("click", closeSidebar);
    $("#status-details").addEventListener("click", () => setView("settings"));
    $("#refresh-status").addEventListener("click", () => { loadHealth(); loadSystemDetails(); loadApprovals(); });
    $("#refresh-documents").addEventListener("click", loadDocuments);
    $("#document-input").addEventListener("change", (event) => uploadDocument(event.target.files?.[0]));
    $("#library-file-input").addEventListener("change", (event) => uploadDocument(event.target.files?.[0]));
    document.addEventListener("click", (event) => {
      const copy = event.target.closest("[data-copy-code]");
      if (copy) copyText(codeBlocks[Number(copy.dataset.copyCode)] || "", copy);
    });
    document.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); newChat(); }
      if (event.key === "Escape") closeSidebar();
    });
    renderRecent();
    loadHealth();
    setInterval(loadHealth, 60000);
    const recent = getRecent();
    if (recent.length) loadSession(recent[0].id);
    else showWelcome(true);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
