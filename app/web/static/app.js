(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const SESSION_KEY = "university-agent.recent-sessions.v1";
  const MAX_RECENT = 20;
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
    // Citation badges: [1], [2], [1, 2]
    text = text.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, nums) => {
      const cleanNums = nums.trim();
      return `<button type="button" class="citation-badge" data-citation-ref="${cleanNums}">[${cleanNums}]</button>`;
    });
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

  // Small, dependency-free Markdown renderer with safe HTML escaping
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

  // --- REASONING ACCORDION COMPONENT ---
  function renderReasoningAccordion(reasoningPath) {
    if (!reasoningPath || !String(reasoningPath).trim()) return null;
    const details = document.createElement("details");
    details.className = "reasoning-accordion";
    const summary = document.createElement("summary");
    summary.innerHTML = '<span class="accordion-icon" aria-hidden="true">💭</span> <span>View AI Reasoning Process</span>';
    const content = document.createElement("div");
    content.className = "reasoning-content";
    content.textContent = String(reasoningPath).trim();
    details.append(summary, content);
    return details;
  }

  // --- GROUNDED SOURCES CARD COMPONENT ---
  function renderSourcesCard(sources) {
    if (!sources || !sources.length) return null;
    const card = document.createElement("div");
    card.className = "sources-card";
    const header = document.createElement("div");
    header.className = "sources-header";
    header.innerHTML = '<span class="sources-icon" aria-hidden="true">📚</span> <strong>Sources Used</strong>';
    const list = document.createElement("div");
    list.className = "sources-list";

    sources.forEach((s, idx) => {
      const item = document.createElement("div");
      item.className = "source-item";
      const num = s.index || (idx + 1);
      item.id = `source-item-${num}`;

      const metaParts = [];
      if (s.page_number) metaParts.push(`Page ${s.page_number}`);
      if (s.heading) metaParts.push(s.heading);
      const metaStr = metaParts.length ? ` · ${metaParts.join(" — ")}` : "";

      item.innerHTML = `
        <div class="source-top">
          <span class="source-index">[${num}]</span>
          <span class="source-filename">${escapeHtml(s.filename || "Document")}</span>
          <span class="source-meta">${escapeHtml(metaStr)}</span>
        </div>
        ${s.snippet ? `<p class="source-snippet">${escapeHtml(s.snippet)}</p>` : ""}
      `;
      list.append(item);
    });

    card.append(header, list);
    return card;
  }

  // --- HUMAN-IN-THE-LOOP SECURITY APPROVAL CARD COMPONENT ---
  function renderApprovalCard(approvalInfo, onDecide = null) {
    const card = document.createElement("div");
    card.className = "approval-card chat-approval-card";
    const approvalId = approvalInfo.approval_id || approvalInfo.id || "";
    card.dataset.approvalId = approvalId;

    const title = document.createElement("div");
    title.className = "approval-title";
    title.innerHTML = '⚠️ <strong>Security Approval Required</strong>';

    const actionDesc = document.createElement("div");
    actionDesc.className = "approval-description";
    const actionName = approvalInfo.action || "Tool operation";
    const msg = approvalInfo.message || `The agent requested execution of '${actionName}'.`;
    actionDesc.textContent = msg;

    const actions = document.createElement("div");
    actions.className = "approval-actions";

    const approveBtn = document.createElement("button");
    approveBtn.type = "button";
    approveBtn.className = "approve-button";
    approveBtn.textContent = "Approve & Execute";

    const rejectBtn = document.createElement("button");
    rejectBtn.type = "button";
    rejectBtn.className = "reject-button";
    rejectBtn.textContent = "Reject";

    const statusMsg = document.createElement("div");
    statusMsg.className = "approval-status-message";
    statusMsg.hidden = true;

    approveBtn.addEventListener("click", async () => {
      approveBtn.disabled = true;
      rejectBtn.disabled = true;
      try {
        await api(`/api/security/approve/${encodeURIComponent(approvalId)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reason: "User approved in chat interface" })
        });
        statusMsg.hidden = false;
        statusMsg.className = "approval-status-message success";
        statusMsg.textContent = "✓ Action approved. Operation authorized.";
        actions.remove();
        showToast("Action approved.");
        if (onDecide) onDecide("approved", approvalId);
      } catch (err) {
        approveBtn.disabled = false;
        rejectBtn.disabled = false;
        showToast(err.message, true);
      }
    });

    rejectBtn.addEventListener("click", async () => {
      approveBtn.disabled = true;
      rejectBtn.disabled = true;
      try {
        await api(`/api/security/reject/${encodeURIComponent(approvalId)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reason: "User rejected in chat interface" })
        });
        statusMsg.hidden = false;
        statusMsg.className = "approval-status-message rejected";
        statusMsg.textContent = "✕ Action rejected. Operation cancelled.";
        actions.remove();
        showToast("Action rejected.");
        if (onDecide) onDecide("rejected", approvalId);
      } catch (err) {
        approveBtn.disabled = false;
        rejectBtn.disabled = false;
        showToast(err.message, true);
      }
    });

    actions.append(approveBtn, rejectBtn);
    card.append(title, actionDesc, actions, statusMsg);
    return card;
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
      const modelInfo = meta.model ? ` · ${meta.provider || "local"} / ${meta.model}` : "";
      label.innerHTML = `<span>UNIVERSITY AGENT${escapeHtml(modelInfo)}</span>`;
      if (meta.intent) {
        const intentClean = String(meta.intent).toUpperCase();
        const intentCls = intentClean.toLowerCase().includes("tutor") ? "tutoring"
          : intentClean.toLowerCase().includes("summar") ? "summarization"
          : intentClean.toLowerCase().includes("assign") ? "assignment"
          : intentClean.toLowerCase().includes("cod") ? "coding" : "";
        label.innerHTML += `<span class="intent-pill ${intentCls}">${escapeHtml(intentClean)}</span>`;
      }
      body.append(label);
    }

    // Reasoning accordion (placed above main answer if present)
    const reasoningPath = meta.reasoning_path || meta.reasoning;
    if (role === "assistant" && reasoningPath) {
      const accordion = renderReasoningAccordion(reasoningPath);
      if (accordion) body.append(accordion);
    }

    const contentNode = document.createElement("div");
    contentNode.className = "rendered-content";
    if (role === "user") contentNode.textContent = content;
    else contentNode.innerHTML = renderMarkdown(content);
    body.append(contentNode);

    // Approval card if status is approval_required
    if (meta.status === "approval_required" || meta.approval_required) {
      const approvalData = meta.approval_required || meta;
      const approvalCard = renderApprovalCard(approvalData);
      body.append(approvalCard);
    }

    // Grounded citations & sources used card
    const allSources = meta.used_sources || citations || [];
    if (role === "assistant" && allSources?.length) {
      const sourcesCard = renderSourcesCard(allSources);
      if (sourcesCard) body.append(sourcesCard);
    }

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

  // Session history loading from database or fallback cache
  async function loadSession(id) {
    sessionId = id;
    $("#message-list").replaceChildren();
    showWelcome(false);
    renderRecent();
    setView("chat");
    const loading = addMessage("assistant", "Loading this conversation…", [], { pending: true });
    try {
      let history;
      try {
        history = await api(`/api/memory/sessions/${encodeURIComponent(id)}`);
      } catch {
        // Fallback for chat history
        history = await api(`/api/chat/history/${encodeURIComponent(id)}`);
      }
      loading.remove();
      if (!history || !history.length) {
        showWelcome(true);
        showToast("No saved messages were found for that conversation.", true);
        return;
      }
      history.forEach((item) => {
        const role = item.role === "user" ? "user" : "assistant";
        const meta = {
          model: item.mode,
          provider: "saved",
          reasoning_path: item.metadata?.reasoning_path,
          intent: item.metadata?.intent,
          used_sources: item.citations || item.metadata?.used_sources,
          status: item.metadata?.status,
          approval_required: item.metadata?.approval_required
        };
        addMessage(role, item.content || "", item.citations || [], meta);
      });
    } catch (error) {
      loading.remove();
      addMessage("assistant", `Could not load this conversation: ${error.message}`, [], { error: true });
    }
  }

  async function loadRecentSessions() {
    try {
      const serverSessions = await api("/api/memory/sessions?limit=20");
      if (Array.isArray(serverSessions) && serverSessions.length) {
        const local = getRecent();
        const mergedMap = new Map();
        serverSessions.forEach((s) => {
          mergedMap.set(s.session_id, {
            id: s.session_id,
            title: s.title || "Untitled Session",
            mode: s.session_type || "tutor",
            updated: s.updated_at ? new Date(s.updated_at).getTime() : Date.now()
          });
        });
        local.forEach((l) => {
          if (!mergedMap.has(l.id)) mergedMap.set(l.id, l);
        });
        setRecent([...mergedMap.values()]);
      } else {
        renderRecent();
      }
    } catch {
      renderRecent();
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

  // --- REAL-TIME SSE CHAT STREAMING CLIENT ---
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

    showWelcome(false);
    const list = $("#message-list");
    const assistantArticle = document.createElement("article");
    assistantArticle.className = "message assistant";
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "UA";
    const body = document.createElement("div");
    body.className = "message-body";

    const label = document.createElement("div");
    label.className = "message-meta";
    label.textContent = "THINKING & STREAMING · WAITING FOR RESPONSE";
    body.append(label);

    const contentNode = document.createElement("div");
    contentNode.className = "rendered-content";
    contentNode.innerHTML = '<span class="loading-cursor">…</span>';
    body.append(contentNode);

    assistantArticle.append(avatar, body);
    list.append(assistantArticle);
    $("#conversation").scrollTop = $("#conversation").scrollHeight;

    setBusy(true);

    try {
      let accumulatedText = "";
      let finalEvent = null;

      // Real-time SSE streaming connection to POST /api/chat
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          session_id: requestSession,
          mode,
          use_rag: true,
          is_private: false,
          stream: true
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errObj;
        try { errObj = JSON.parse(errorText); } catch { errObj = { detail: errorText }; }
        throw new Error(errObj.detail || `Request failed (${response.status})`);
      }

      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("text/event-stream") && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop();

          for (const part of parts) {
            const lines = part.split("\n");
            for (const rawLine of lines) {
              const line = rawLine.trim();
              if (!line.startsWith("data:")) continue;
              const dataStr = line.slice(5).trim();
              if (!dataStr || dataStr === "[DONE]") continue;

              try {
                const event = JSON.parse(dataStr);
                if (event.error) {
                  throw new Error(event.error);
                }
                if (event.token) {
                  accumulatedText += event.token;
                  contentNode.innerHTML = renderMarkdown(accumulatedText);
                  $("#conversation").scrollTop = $("#conversation").scrollHeight;
                }
                if (event.done) {
                  finalEvent = event;
                }
                if (event.status === "approval_required" || event.approval_required) {
                  finalEvent = event;
                }
              } catch (jsonErr) {
                // Ignore partial JSON parse errors
              }
            }
          }
        }
      } else {
        // Fallback for non-streaming environments using api("/api/chat")
        const jsonRes = await response.json();
        accumulatedText = jsonRes.reply || "";
        finalEvent = jsonRes;
      }

      // Finalize the rendered assistant message
      const reasoningPath = finalEvent?.metadata?.reasoning_path || finalEvent?.reasoning_path || "";
      const usedSources = finalEvent?.metadata?.used_sources || finalEvent?.used_sources || finalEvent?.citations || [];
      const intent = finalEvent?.metadata?.intent || finalEvent?.intent || "";
      const isApprovalRequired = finalEvent?.status === "approval_required" || Boolean(finalEvent?.approval_required);

      label.innerHTML = `<span>UNIVERSITY AGENT${finalEvent?.model ? ` · ${finalEvent?.provider || "local"} / ${finalEvent?.model}` : ""}</span>`;
      if (intent) {
        const intentClean = String(intent).toUpperCase();
        const intentCls = intentClean.toLowerCase().includes("tutor") ? "tutoring"
          : intentClean.toLowerCase().includes("summar") ? "summarization"
          : intentClean.toLowerCase().includes("assign") ? "assignment"
          : intentClean.toLowerCase().includes("cod") ? "coding" : "";
        label.innerHTML += `<span class="intent-pill ${intentCls}">${escapeHtml(intentClean)}</span>`;
      }

      // Insert reasoning accordion above content if present
      if (reasoningPath) {
        const accordion = renderReasoningAccordion(reasoningPath);
        if (accordion) body.insertBefore(accordion, contentNode);
      }

      contentNode.innerHTML = renderMarkdown(accumulatedText || finalEvent?.reply || "");

      // Render security approval card if required
      if (isApprovalRequired) {
        const approvalData = finalEvent.approval_required || finalEvent;
        const approvalCard = renderApprovalCard(approvalData);
        body.append(approvalCard);
      }

      // Render grounded source cards if citations exist
      if (usedSources.length) {
        const sourcesCard = renderSourcesCard(usedSources);
        if (sourcesCard) body.append(sourcesCard);
      }

      // Add copy action
      const actions = document.createElement("div");
      actions.className = "message-copy";
      const copy = document.createElement("button");
      copy.type = "button"; copy.textContent = "Copy response";
      copy.addEventListener("click", () => copyText(accumulatedText || finalEvent?.reply || "", copy));
      actions.append(copy); body.append(actions);

      rememberSession(finalEvent?.session_id || requestSession, message, mode);
      loadRecentSessions();
    } catch (error) {
      label.textContent = "REQUEST ERROR";
      contentNode.textContent = `I couldn't complete that request. ${error.message}`;
      assistantArticle.classList.add("error");
    } finally {
      setBusy(false);
    }
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

  let cachedSubjects = [];
  let currentTaskFilter = "all";
  let activeQuiz = null;

  async function loadSubjectOptions() {
    try {
      cachedSubjects = await api("/api/subjects");
      const docFilter = $("#doc-subject-filter");
      const topicFilter = $("#topic-subject-filter");
      [docFilter, topicFilter].forEach((sel) => {
        if (!sel) return;
        const currentVal = sel.value;
        sel.replaceChildren();
        const allOpt = document.createElement("option");
        allOpt.value = "";
        allOpt.textContent = "All Subjects";
        sel.appendChild(allOpt);
        cachedSubjects.forEach((sub) => {
          const opt = document.createElement("option");
          opt.value = sub.id || sub.name;
          opt.textContent = `${sub.name}${sub.code ? " (" + sub.code + ")" : ""}`;
          sel.appendChild(opt);
        });
        sel.value = currentVal;
      });
    } catch (_) {}
  }

  async function loadDocuments(subjectId = "") {
    const list = $("#document-list");
    list.innerHTML = '<div class="loading-row">Loading your library…</div>';
    $("#library-note").textContent = "";
    try {
      loadSubjectOptions();
      const endpoint = subjectId ? `/api/documents?subject_id=${encodeURIComponent(subjectId)}` : "/api/documents";
      const docs = await api(endpoint);
      list.replaceChildren();
      if (!docs.length) {
        list.innerHTML = '<div class="empty-row">No documents found. Add a course file above to get started.</div>';
        return;
      }
      docs.forEach((doc) => {
        const card = document.createElement("article");
        card.className = "document-card";
        const icon = document.createElement("div");
        icon.className = "file-icon";
        icon.textContent = (doc.filename || "file").split(".").pop().slice(0, 4);

        const details = document.createElement("div");
        details.style.flex = "1";
        details.style.minWidth = "0";

        const name = document.createElement("div");
        name.className = "document-name";
        name.textContent = doc.filename || "Unnamed document";
        name.title = name.textContent;

        const meta = document.createElement("div");
        meta.className = "document-meta";
        const parts = [];
        if (doc.subject) parts.push(`Subject: ${doc.subject}`);
        parts.push(`${doc.total_pages || 1} page${doc.total_pages === 1 ? "" : "s"}`);
        if (doc.status) parts.push(`Status: ${doc.status}`);
        meta.textContent = parts.join(" · ");
        details.append(name, meta);

        const count = document.createElement("div");
        count.className = "document-count";
        count.textContent = `${doc.total_chunks || 0} chunks`;

        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.className = "doc-delete-btn";
        delBtn.title = "Delete document from library";
        delBtn.textContent = "Delete";
        const docId = doc.document_id || doc.id || doc.filename;
        delBtn.addEventListener("click", async (e) => {
          e.stopPropagation();
          if (!confirm(`Are you sure you want to delete ${doc.filename || 'this document'}?`)) return;
          try {
            await api(`/api/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
            showToast(`Document ${doc.filename || docId} deleted.`);
            loadDocuments($("#doc-subject-filter") ? $("#doc-subject-filter").value : "");
          } catch (err) {
            showToast(err.message, true);
          }
        });

        card.append(icon, details, count, delBtn);
        list.append(card);
      });
    } catch (error) {
      list.innerHTML = `<div class="empty-row error-text">${escapeHtml(error.message)}</div>`;
    }
  }

  async function uploadDocument(file) {
    if (!file) return;
    const zone = $("#upload-zone"); zone.classList.add("uploading");
    showToast("Processing and indexing document locally…");
    const formData = new FormData(); formData.append("file", file);
    try {
      const result = await api("/api/documents/upload", { method: "POST", body: formData });
      showToast(`Indexed ${result.filename} (${result.total_chunks} chunks).`);
      loadDocuments($("#doc-subject-filter") ? $("#doc-subject-filter").value : "");
    } catch (error) { showToast(error.message, true); }
    finally { zone.classList.remove("uploading"); $("#document-input").value = ""; $("#library-file-input").value = ""; }
  }

  async function loadTopics(subjectId = "") {
    const topicContainer = $("#topic-list");
    topicContainer.innerHTML = '<div class="loading-row">Loading topics…</div>';
    try {
      await loadSubjectOptions();
      let topics = [];
      if (subjectId) {
        topics = await api(`/api/subjects/${encodeURIComponent(subjectId)}/topics`);
      } else {
        topics = await api("/api/memory/topics");
      }
      topicContainer.replaceChildren();
      if (!topics || !topics.length) {
        topicContainer.innerHTML = '<div class="empty-row">No topics tracked yet.</div>';
        return;
      }
      topics.forEach((topic) => {
        const row = document.createElement("div");
        row.className = "topic-row";

        const indicator = document.createElement("span");
        indicator.className = "topic-indicator";

        const content = document.createElement("div");
        content.style.flex = "1";
        content.style.minWidth = "0";

        const topRow = document.createElement("div");
        topRow.style.display = "flex";
        topRow.style.justifyContent = "space-between";
        topRow.style.alignItems = "center";

        const name = document.createElement("div");
        name.className = "topic-name";
        name.textContent = topic.name || "Topic";

        const masteryPct = Math.round((topic.mastery_level || 0) * 100);
        const level = document.createElement("div");
        level.className = "topic-level";
        level.textContent = `Mastery ${masteryPct}%`;
        topRow.append(name, level);

        const masteryTrack = document.createElement("div");
        masteryTrack.className = "mastery-container";
        const track = document.createElement("div");
        track.className = "mastery-track";
        const fill = document.createElement("div");
        fill.className = "mastery-fill";
        fill.style.width = `${Math.min(100, Math.max(0, masteryPct))}%`;
        track.appendChild(fill);
        masteryTrack.appendChild(track);

        content.append(topRow, masteryTrack);

        const quizBtn = document.createElement("button");
        quizBtn.type = "button";
        quizBtn.className = "quiz-launch-btn";
        quizBtn.textContent = "Take Quiz";
        quizBtn.dataset.topicId = topic.id;
        quizBtn.dataset.topicName = topic.name || "";
        quizBtn.dataset.subjectId = topic.subject_id || "";
        quizBtn.addEventListener("click", () => {
          openQuizModal(topic.id, topic.name, topic.subject_id);
        });

        row.append(indicator, content, quizBtn);
        topicContainer.append(row);
      });
    } catch (error) {
      topicContainer.innerHTML = `<div class="empty-row error-text">${escapeHtml(error.message)}</div>`;
    }
  }

  async function loadTasks(filter = currentTaskFilter) {
    currentTaskFilter = filter;
    const taskContainer = $("#task-list");
    taskContainer.innerHTML = '<div class="loading-row">Loading tasks…</div>';

    $$("#task-filter-tabs .tab-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.taskFilter === filter);
    });

    let endpoint = "/api/tasks";
    if (filter === "upcoming") endpoint = "/api/tasks/upcoming";
    else if (filter === "overdue") endpoint = "/api/tasks/overdue";

    try {
      const tasks = await api(endpoint);
      taskContainer.replaceChildren();
      if (!tasks.length) {
        taskContainer.innerHTML = `<div class="empty-row">No ${filter === "all" ? "" : filter + " "}tasks found.</div>`;
        return;
      }
      tasks.forEach((task) => {
        const row = document.createElement("div");
        const isDone = task.completed || task.status === "completed";
        row.className = `task-row${isDone ? " done" : ""}`;

        const check = document.createElement("input");
        check.type = "checkbox";
        check.className = "task-check";
        check.checked = Boolean(isDone);
        const taskId = task.id || task.task_id;

        check.addEventListener("change", async () => {
          try {
            await api(`/api/tasks/${taskId}`, {
              method: "PATCH",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ status: check.checked ? "completed" : "pending" })
            });
            row.classList.toggle("done", check.checked);
          } catch (err) {
            check.checked = !check.checked;
            showToast(err.message, true);
          }
        });

        const content = document.createElement("div");
        content.className = "task-content";
        content.style.flex = "1";
        content.style.minWidth = "0";

        const name = document.createElement("div");
        name.className = "task-name";
        name.textContent = task.title || "Untitled task";

        if (task.priority) {
          const priorityBadge = document.createElement("span");
          priorityBadge.className = `badge-priority badge-${task.priority.toLowerCase()}`;
          priorityBadge.textContent = task.priority;
          name.appendChild(priorityBadge);
        }

        const meta = document.createElement("span");
        meta.className = "task-meta";
        const metaParts = [];
        if (task.deadline || task.due_date) metaParts.push(`Due ${task.deadline || task.due_date}`);
        if (task.subject_id) metaParts.push(task.subject_id);
        meta.textContent = metaParts.join(" · ") || "No deadline";
        content.append(name, meta);

        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.className = "task-delete-btn";
        delBtn.title = "Delete task";
        delBtn.textContent = "✕";
        delBtn.addEventListener("click", async () => {
          try {
            await api(`/api/tasks/${taskId}`, { method: "DELETE" });
            showToast("Task deleted.");
            loadTasks(currentTaskFilter);
          } catch (err) {
            showToast(err.message, true);
          }
        });

        row.append(check, content, delBtn);
        taskContainer.append(row);
      });
    } catch (error) {
      taskContainer.innerHTML = `<div class="empty-row error-text">${escapeHtml(error.message)}</div>`;
    }
  }

  async function createTask(event) {
    if (event) event.preventDefault();
    const titleInput = $("#task-title-input");
    const dateInput = $("#task-due-date");
    const title = titleInput.value.trim();
    if (!title) return;
    try {
      await api("/api/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: title,
          deadline: dateInput.value || null,
          priority: "medium"
        })
      });
      titleInput.value = "";
      dateInput.value = "";
      showToast("Academic task scheduled.");
      loadTasks(currentTaskFilter);
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function loadGoals() {
    const goalContainer = $("#goal-list");
    if (!goalContainer) return;
    goalContainer.innerHTML = '<div class="loading-row">Loading goals…</div>';
    try {
      const goals = await api("/api/goals");
      goalContainer.replaceChildren();
      if (!goals.length) {
        goalContainer.innerHTML = '<div class="empty-row">No study goals set. Add one above!</div>';
        return;
      }
      goals.forEach((goal) => {
        const card = document.createElement("div");
        card.className = "goal-card";

        const header = document.createElement("div");
        header.className = "goal-header";
        const title = document.createElement("div");
        title.className = "goal-title";
        title.textContent = goal.title || "Untitled goal";

        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.className = "task-delete-btn";
        delBtn.title = "Delete goal";
        delBtn.textContent = "✕";
        delBtn.addEventListener("click", async () => {
          try {
            await api(`/api/goals/${goal.id}`, { method: "DELETE" });
            showToast("Goal removed.");
            loadGoals();
          } catch (err) {
            showToast(err.message, true);
          }
        });
        header.append(title, delBtn);

        const meta = document.createElement("div");
        meta.className = "goal-meta";
        const parts = [];
        if (goal.target_date) parts.push(`Target: ${goal.target_date}`);
        if (goal.subject_id) parts.push(goal.subject_id);
        meta.textContent = parts.join(" · ");

        const track = document.createElement("div");
        track.className = "mastery-track";
        track.style.marginTop = "6px";
        const fill = document.createElement("div");
        fill.className = "mastery-fill goal-fill";
        let pct = Math.round(goal.progress_percent || 0);
        fill.style.width = `${pct}%`;
        track.appendChild(fill);

        const controls = document.createElement("div");
        controls.className = "goal-controls";

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "0";
        slider.max = "100";
        slider.value = String(pct);
        slider.className = "goal-slider";

        const pctLabel = document.createElement("span");
        pctLabel.className = "goal-pct-label";
        pctLabel.textContent = `${pct}%`;

        slider.addEventListener("change", async () => {
          const newVal = parseFloat(slider.value);
          try {
            await api(`/api/goals/${goal.id}/progress`, {
              method: "PATCH",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ progress_percent: newVal })
            });
            pct = newVal;
            fill.style.width = `${pct}%`;
            pctLabel.textContent = `${pct}%`;
          } catch (err) {
            slider.value = String(pct);
            showToast(err.message, true);
          }
        });

        slider.addEventListener("input", () => {
          pctLabel.textContent = `${slider.value}%`;
          fill.style.width = `${slider.value}%`;
        });

        const quickBtn = document.createElement("button");
        quickBtn.type = "button";
        quickBtn.className = "goal-btn";
        quickBtn.textContent = "+25%";
        quickBtn.addEventListener("click", async () => {
          const newVal = Math.min(100, pct + 25);
          try {
            await api(`/api/goals/${goal.id}/progress`, {
              method: "PATCH",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ progress_percent: newVal })
            });
            pct = newVal;
            slider.value = String(pct);
            fill.style.width = `${pct}%`;
            pctLabel.textContent = `${pct}%`;
          } catch (err) {
            showToast(err.message, true);
          }
        });

        controls.append(slider, pctLabel, quickBtn);
        card.append(header, meta, track, controls);
        goalContainer.append(card);
      });
    } catch (error) {
      goalContainer.innerHTML = `<div class="empty-row error-text">${escapeHtml(error.message)}</div>`;
    }
  }

  async function createGoal(event) {
    if (event) event.preventDefault();
    const titleInput = $("#goal-title-input");
    const dateInput = $("#goal-target-date");
    const title = titleInput.value.trim();
    if (!title) return;
    try {
      await api("/api/goals", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: title,
          target_date: dateInput.value || null,
          status: "in_progress",
          progress_percent: 0.0
        })
      });
      titleInput.value = "";
      dateInput.value = "";
      showToast("Study goal saved.");
      loadGoals();
    } catch (err) {
      showToast(err.message, true);
    }
  }

  async function openQuizModal(topicId, topicName, subjectId) {
    const modal = $("#quiz-modal");
    const title = $("#quiz-modal-title");
    const loading = $("#quiz-loading");
    const questionsContainer = $("#quiz-questions-container");
    const resultsContainer = $("#quiz-results");
    const submitBtn = $("#quiz-submit-btn");

    title.textContent = `Quiz: ${topicName || "Topic Mastery"}`;
    modal.hidden = false;
    loading.hidden = false;
    questionsContainer.replaceChildren();
    questionsContainer.hidden = true;
    resultsContainer.replaceChildren();
    resultsContainer.hidden = true;
    submitBtn.hidden = false;
    submitBtn.disabled = true;
    submitBtn.textContent = "Submit Answers";

    try {
      const payload = {
        topic_id: topicId,
        topic_name: topicName,
        subject_id: subjectId || topicId,
        num_questions: 3,
        hide_answers: true
      };
      const quizData = await api("/api/agent/quiz", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      activeQuiz = {
        topic_id: topicId,
        topic_name: topicName,
        quiz_data: quizData,
        answers: {}
      };

      const questions = quizData.questions || [];
      if (!questions.length) {
        loading.hidden = true;
        questionsContainer.hidden = false;
        questionsContainer.innerHTML = '<div class="empty-row">No questions generated. Please ensure course materials are indexed.</div>';
        return;
      }

      questions.forEach((q, idx) => {
        const card = document.createElement("div");
        card.className = "quiz-question-card";

        const qNum = document.createElement("div");
        qNum.className = "quiz-q-num";
        qNum.textContent = `Question ${idx + 1} of ${questions.length}`;

        const qText = document.createElement("div");
        qText.className = "quiz-q-text";
        qText.textContent = q.question_text || q.question || "";

        const optContainer = document.createElement("div");
        optContainer.className = "quiz-options";

        const options = q.options || [];
        options.forEach((opt) => {
          const label = document.createElement("label");
          label.className = "quiz-option-label";

          const radio = document.createElement("input");
          radio.type = "radio";
          radio.name = `quiz_q_${idx}`;
          radio.value = opt;

          radio.addEventListener("change", () => {
            activeQuiz.answers[String(idx)] = opt;
            card.querySelectorAll(".quiz-option-label").forEach((l) => l.classList.remove("selected"));
            label.classList.add("selected");
          });

          const optSpan = document.createElement("span");
          optSpan.textContent = opt;

          label.append(radio, optSpan);
          optContainer.append(label);
        });

        card.append(qNum, qText, optContainer);
        questionsContainer.append(card);
      });

      loading.hidden = true;
      questionsContainer.hidden = false;
      submitBtn.disabled = false;
    } catch (error) {
      loading.hidden = true;
      questionsContainer.hidden = false;
      questionsContainer.innerHTML = `<div class="empty-row error-text">Failed to generate quiz: ${escapeHtml(error.message)}</div>`;
    }
  }

  async function submitQuiz() {
    if (!activeQuiz) return;
    const submitBtn = $("#quiz-submit-btn");
    const questionsContainer = $("#quiz-questions-container");
    const resultsContainer = $("#quiz-results");

    submitBtn.disabled = true;
    submitBtn.textContent = "Grading…";

    try {
      const evalPayload = {
        topic_id: activeQuiz.topic_id,
        answers: activeQuiz.answers,
        quiz_data: activeQuiz.quiz_data
      };

      const result = await api("/api/agent/quiz/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(evalPayload)
      });

      questionsContainer.hidden = true;
      resultsContainer.hidden = false;
      resultsContainer.replaceChildren();

      const summaryCard = document.createElement("div");
      summaryCard.className = "quiz-summary-card";

      const scorePct = Math.round((result.score || 0) * 100);
      const scoreVal = document.createElement("div");
      scoreVal.className = "quiz-score-val";
      scoreVal.textContent = `${scorePct}% (${result.correct_count} / ${result.total_questions} correct)`;

      const feedback = document.createElement("div");
      feedback.className = "quiz-feedback";
      feedback.textContent = result.summary_feedback || (scorePct >= 70 ? "Great job!" : "Keep studying and try again!");

      const masteryUpdate = document.createElement("div");
      masteryUpdate.className = "quiz-mastery-update";
      const newMastery = Math.round((result.mastery_level || 0) * 100);
      masteryUpdate.textContent = `★ Topic Mastery Updated: ${newMastery}% (via Exponential Moving Average)`;

      summaryCard.append(scoreVal, feedback, masteryUpdate);
      resultsContainer.append(summaryCard);

      const evals = result.evaluations || [];
      evals.forEach((ev, idx) => {
        const item = document.createElement("div");
        item.className = `quiz-eval-item ${ev.is_correct ? "correct" : "incorrect"}`;

        const header = document.createElement("div");
        header.className = "eval-header";

        const qTitle = document.createElement("strong");
        qTitle.textContent = `Q${idx + 1}: ${ev.question_text}`;

        const badge = document.createElement("span");
        badge.className = `eval-badge ${ev.is_correct ? "pass" : "fail"}`;
        badge.textContent = ev.is_correct ? "✓ Correct" : "✗ Needs Review";

        header.append(qTitle, badge);

        const ansRow = document.createElement("div");
        ansRow.style.fontSize = "11px";
        ansRow.style.marginTop = "4px";
        ansRow.innerHTML = `Your answer: <em>${escapeHtml(ev.student_answer || "None")}</em> · Correct: <strong>${escapeHtml(ev.correct_answer || "")}</strong>`;

        item.append(header, ansRow);

        if (ev.explanation) {
          const exp = document.createElement("div");
          exp.className = "eval-explanation";
          exp.textContent = ev.explanation;
          item.append(exp);
        }

        if (ev.source_chunk_id) {
          const cit = document.createElement("span");
          cit.className = "eval-citation";
          cit.textContent = `Source chunk: ${ev.source_chunk_id}`;
          item.append(cit);
        }

        resultsContainer.append(item);
      });

      submitBtn.hidden = true;
      showToast(`Quiz completed! Score: ${scorePct}%. Mastery: ${newMastery}%.`);

      loadTopics($("#topic-subject-filter") ? $("#topic-subject-filter").value : "");
      loadLearning();
    } catch (error) {
      showToast(`Evaluation failed: ${error.message}`, true);
      submitBtn.disabled = false;
      submitBtn.textContent = "Submit Answers";
    }
  }

  function closeQuizModal() {
    $("#quiz-modal").hidden = true;
    activeQuiz = null;
  }

  async function loadLearning() {
    const statsContainer = $("#learning-stats");
    statsContainer.innerHTML = '<div class="loading-row">Loading your progress…</div>';

    try {
      const progress = await api("/api/memory/progress");
      statsContainer.replaceChildren();
      const stats = [
        ["Total study sessions", progress.total_sessions ?? 0],
        ["Indexed documents", progress.total_documents ?? 0],
        ["Open academic tasks", progress.total_tasks ?? 0],
        ["Recorded quiz results", progress.total_quizzes ?? 0]
      ];
      stats.forEach(([label, value]) => {
        const card = document.createElement("div"); card.className = "stat-card";
        const val = document.createElement("div"); val.className = "stat-value"; val.textContent = String(value);
        const lbl = document.createElement("div"); lbl.className = "stat-label"; lbl.textContent = label;
        card.append(val, lbl); statsContainer.append(card);
      });
    } catch (error) { statsContainer.innerHTML = `<div class="empty-row error-text">${escapeHtml(error.message)}</div>`; }

    loadTopics($("#topic-subject-filter") ? $("#topic-subject-filter").value : "");
    loadTasks(currentTaskFilter);
    loadGoals();
  }

  async function loadSystemDetails() {
    const container = $("#system-details");
    container.innerHTML = '<div class="loading-row">Checking services…</div>';
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
        ["Cloud routing", status.cloud_routing_enabled ? "Enabled" : "Disabled", status.cloud_routing_enabled]
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
      await api(`/api/security/approve/${encodeURIComponent(id)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reason: `User ${decision}d via settings` }) });
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
    $("#refresh-documents").addEventListener("click", () => loadDocuments($("#doc-subject-filter") ? $("#doc-subject-filter").value : ""));
    $("#document-input").addEventListener("change", (event) => uploadDocument(event.target.files?.[0]));
    $("#library-file-input").addEventListener("change", (event) => uploadDocument(event.target.files?.[0]));

    // Quiz modal controls
    const quizClose = $("#quiz-modal-close");
    if (quizClose) quizClose.addEventListener("click", closeQuizModal);
    const quizCancel = $("#quiz-cancel-btn");
    if (quizCancel) quizCancel.addEventListener("click", closeQuizModal);
    const quizSubmit = $("#quiz-submit-btn");
    if (quizSubmit) quizSubmit.addEventListener("click", submitQuiz);
    const quizModal = $("#quiz-modal");
    if (quizModal) {
      quizModal.addEventListener("click", (e) => {
        if (e.target === quizModal) closeQuizModal();
      });
    }

    // Subject & Task filters
    const docSubFilter = $("#doc-subject-filter");
    if (docSubFilter) {
      docSubFilter.addEventListener("change", (e) => loadDocuments(e.target.value));
    }
    const topicSubFilter = $("#topic-subject-filter");
    if (topicSubFilter) {
      topicSubFilter.addEventListener("change", (e) => loadTopics(e.target.value));
    }
    $$("#task-filter-tabs .tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => loadTasks(btn.dataset.taskFilter));
    });

    // Task & Goal creation forms
    const taskForm = $("#new-task-form");
    if (taskForm) taskForm.addEventListener("submit", createTask);
    const goalForm = $("#new-goal-form");
    if (goalForm) goalForm.addEventListener("submit", createGoal);

    // Citation badge click & code copying
    document.addEventListener("click", (event) => {
      const badge = event.target.closest(".citation-badge");
      if (badge) {
        const ref = badge.dataset.citationRef;
        if (ref) {
          const firstNum = ref.split(",")[0].trim();
          const target = document.getElementById(`source-item-${firstNum}`);
          if (target) {
            target.scrollIntoView({ behavior: "smooth", block: "center" });
            target.classList.add("highlighted");
            setTimeout(() => target.classList.remove("highlighted"), 1800);
          }
        }
      }
      const copy = event.target.closest("[data-copy-code]");
      if (copy) copyText(codeBlocks[Number(copy.dataset.copyCode)] || "", copy);
    });

    document.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); newChat(); }
      if (event.key === "Escape") { closeSidebar(); closeQuizModal(); }
    });

    loadRecentSessions();
    loadHealth();
    setInterval(loadHealth, 60000);
    const recent = getRecent();
    if (recent.length) loadSession(recent[0].id);
    else showWelcome(true);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
