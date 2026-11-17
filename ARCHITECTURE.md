# 🏛 System Architecture & Technical Specifications

This document outlines the architectural principles, data flow models, database schemas, security mechanisms, model routing strategies, and RAG retrieval pipeline powering the **Local University AI Agent**.

---

## 1. Architectural Philosophy & Layered Model

The University AI Agent is architected with a strict layered design adhering to separation of concerns, deterministic lifecycle management, and privacy-first local computation:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. Presentation Layer                           │
│  • Single-Page Web Interface (Vanilla ES6, CSS3, SSE Client, Modals)   │
│  • OpenAI-Compatible REST API (/v1/models, /v1/chat/completions)       │
│  • Canonical Domain API (/api/chat, /api/documents, /api/agent/quiz)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                        2. Orchestration Layer                          │
│  • AcademicAgent (Stateful Conversation Loop, Intent Classifier)       │
│  • ModelRouter (Complexity Heuristic, Privacy Fencing, Cloud Failover) │
│  • Tool Orchestrator (Schema Formatting, Side-Effect Pause/Resume)     │
└─────────┬─────────────────────────┬──────────────────────────┬─────────┘
          │                         │                          │
┌─────────▼──────────────┐ ┌────────▼──────────────┐ ┌─────────▼─────────┐
│   3. Security Domain   │ │     4. RAG Domain     │ │ 5. Memory Domain  │
│ • SecurityValidator    │ │ • AcademicChunker     │ │ • DatabaseManager │
│ • CodeSandbox (AST)    │ │ • PyMuPDF & docx Parser│ │ • WAL SQLite Pool │
│ • Approval Token Store │ │ • Hybrid Vector Store │ │ • Migration Engine│
│ • Forensic AuditLogger │ │ • Citation Grounding  │ │ • Mastery EMA Calc│
└────────────────────────┘ └───────────────────────┘ └───────────────────┘
```

### Dependency Injection & Service Lifecycle
All services are created via the `create_service_container()` factory in `app/containers.py` and attached directly to `FastAPI.state.services`. FastAPI endpoints consume services using the `get_services(request: Request)` dependency injector in `app/api/deps.py`, guaranteeing zero global state pollution during concurrent testing.

The application lifespan context manager in `app/main.py` enforces clean startup checks and graceful termination hooks:
- **Startup:** Ensures workspace directory trees exist, initializes SQLite migrations synchronously, verifies vector store collections, and checks local Ollama health.
- **Shutdown:** Safely flushes fallback lexical indices, terminates ChromaDB client threads, and executes an explicit SQLite WAL truncation checkpoint (`PRAGMA wal_checkpoint(TRUNCATE);`) to prevent dirty-state sidecars upon process termination.

---

## 2. Database & Storage Architecture

The persistence model utilizes a dual storage strategy: SQLite for relational metadata and conversation memory, combined with ChromaDB (and a zero-dependency lexical fallback) for high-dimensional vector search.

### A. Relational Schema & Migration Engine (`app/memory/database.py`)
Database migrations run deterministically inside an isolated sync transaction on startup, managed by table `schema_migrations`:

```sql
-- Migration 001: Initial Core Schema
CREATE TABLE IF NOT EXISTS subjects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    code TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS topics (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    mastery_level REAL DEFAULT 0.0 CHECK(mastery_level >= 0.0 AND mastery_level <= 1.0),
    notes TEXT,
    last_reviewed TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject_id, name)
);

CREATE TABLE IF NOT EXISTS quiz_results (
    id TEXT PRIMARY KEY,
    topic_id TEXT REFERENCES topics(id) ON DELETE CASCADE,
    score REAL NOT NULL CHECK(score >= 0.0 AND score <= 1.0),
    total_questions INTEGER NOT NULL CHECK(total_questions > 0),
    question TEXT,
    user_answer TEXT,
    correct_answer TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS academic_tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    subject_id TEXT REFERENCES subjects(id) ON DELETE SET NULL,
    deadline TIMESTAMP,
    priority TEXT DEFAULT 'medium' CHECK(priority IN ('low', 'medium', 'high')),
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'completed')),
    completed INTEGER DEFAULT 0 CHECK(completed IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS study_goals (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    target_date TIMESTAMP,
    status TEXT DEFAULT 'in_progress' CHECK(status IN ('in_progress', 'completed')),
    progress_percent REAL DEFAULT 0.0 CHECK(progress_percent >= 0.0 AND progress_percent <= 100.0),
    subject_id TEXT REFERENCES subjects(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    mode TEXT DEFAULT 'general',
    title TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    reasoning_path TEXT,
    sources_used TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Migration 003: Physical Documents Catalog
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    file_type TEXT NOT NULL CHECK(file_type IN ('pdf', 'docx', 'txt', 'md')),
    file_size_bytes INTEGER NOT NULL CHECK(file_size_bytes >= 0),
    subject_id TEXT REFERENCES subjects(id) ON DELETE SET NULL,
    title TEXT,
    page_count INTEGER NOT NULL DEFAULT 1 CHECK(page_count >= 1),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(chunk_count >= 0),
    status TEXT NOT NULL DEFAULT 'indexed' CHECK(status IN ('pending', 'processing', 'indexed', 'failed')),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Hardened SQLite Pragmas:
Connections apply performance and crash-safety pragmas on every checkout:
- `PRAGMA foreign_keys = ON;`: Enforces relational referential integrity.
- `PRAGMA journal_mode = WAL;`: Write-Ahead Logging allows concurrent readers alongside a writer.
- `PRAGMA synchronous = NORMAL;`: Optimizes disk flushes while maintaining durability.
- `PRAGMA busy_timeout = 5000;`: Prevents lock contention errors during multi-worker access.

### B. Vector Store & Fallback Index (`app/knowledge/vector_store.py`)
- **ChromaDB Mode**: Persisted at `CHROMA_DIR` using Cosine Distance (`metadata={"hnsw:space": "cosine"}`). Stores chunk texts, document IDs, page numbers, subject codes, and section headers.
- **Fallback Lexical Index**: In resource-constrained or headless environments where ChromaDB is unavailable, the vector store automatically falls back to `fallback_index.json`. It computes deterministic lexical hashes and BM25 token frequencies, ensuring the system remains completely functional without C++ native dependencies.

---

## 3. Security Architecture & Sandboxing

Security operations are centralized within `app/security/validator.py`, `app/security/sandbox.py`, and `app/security/audit.py`.

```
                  ┌────────────────────────────────────────┐
                  │          Inbound Action Call           │
                  └───────────────────┬────────────────────┘
                                      │
                   Is Action Safe / Non-Mutating?
                                     / \
                                   YES  NO
                                   /     \
             ┌────────────────────▼─┐   ┌─▼────────────────────────┐
             │ Execute Immediately  │   │ Require Approval Token   │
             │ (e.g. read_file, ls) │   │ (e.g. rm, write, exec)   │
             └──────────────────────┘   └─────────────┬────────────┘
                                                      │
                                           Token Valid & Unused?
                                                     / \
                                                   YES  NO
                                                   /     \
                  ┌───────────────────────────────▼┐   ┌──▼────────────────┐
                  │ Consume Token (Single-Use)     │   │ Halt & Return     │
                  │ Execute in Isolated Subprocess │   │ approval_required │
                  └───────────────┬────────────────┘   └───────────────────┘
                                  │
                  ┌───────────────▼────────────────┐
                  │ Log Action to audit.log        │
                  └────────────────────────────────┘
```

### A. Path Traversal Protection
All filesystem operations (reading course documents, saving notes, creating folders) validate paths against authorized boundaries using `ActionValidator.is_path_safe(requested_path)`:
```python
resolved = Path(requested_path).resolve()
base = Path(settings.DOCUMENTS_DIR).resolve()
if not resolved.is_relative_to(base):
    raise SecurityError("Path traversal breakout detected outside authorized directory.")
```
Any attempt to reference parent paths (`../../etc/passwd`, `~/.ssh`) immediately triggers an error and logs a security violation.

### B. Command Whitelisting & Shell Injection Defenses
Terminal tool requests pass through `ActionValidator.is_terminal_action_allowed(command)`:
- **Whitelisted Binaries**: Strictly permits `python`, `node`, `ls`, `cat`, `echo`, `grep`, `ollama`, `pytest`, `npm run`.
- **Explicit Blocklist**: Instantly rejects `rm -rf`, `sudo`, `chmod`, `chown`, `mkfs`, `dd`, or any path targeting `/etc/`, `/var/`, or `~/.ssh`.
- **Injection Sanitization**: Rejects shell metacharacters (`;`, `&&`, `||`, `|`, `>`, `<`) unless explicitly approved via human verification.

### C. AST Code Execution Sandbox (`app/security/sandbox.py`)
When the model or user requests Python code execution:
1. The code is parsed into an Abstract Syntax Tree (`ast.parse()`).
2. An AST NodeVisitor inspects all `Import` and `ImportFrom` nodes. Any import of prohibited system packages (`os`, `sys`, `subprocess`, `socket`, `shutil`, `pty`, `pathlib`) raises a `SecurityError`.
3. The script is executed inside an isolated subprocess with a strict 5.0-second timeout, capturing stdout and stderr into in-memory buffers with max length truncations.

### D. Single-Use Cryptographic Approval Tokens
Destructive actions generate an approval challenge:
- Tokens are cryptographically bound to the specific `action`, target `parameters`, and a 5-minute TTL.
- Once approved via the UI or `POST /api/security/approve/{token}`, the token is marked consumed immediately, preventing replay attacks.
- Every approval, rejection, and execution is appended to `logs/audit.log` with ISO-8601 timestamps and caller metadata.

---

## 4. Hybrid Model Routing Architecture

The `ModelRouter` (`app/providers/router.py`) determines whether an academic query should be resolved by the local Ollama instance or forwarded to an external cloud model (e.g. Gemini 2.5 Flash):

```
                        ┌───────────────────────────────┐
                        │      Incoming Student Query   │
                        └───────────────┬───────────────┘
                                        │
                         Contains Private RAG Chunks?
                                       / \
                                     YES  NO
                                     /     \
             ┌──────────────────────▼──┐  ┌─▼──────────────────────────┐
             │ FORCE LOCAL OLLAMA      │  │ Cloud Enabled in Settings? │
             │ Privacy guardrail active│  └──────────────┬─────────────┘
             └─────────────────────────┘                 │
                                                        / \
                                                      YES  NO
                                                      /     \
                 ┌───────────────────────────────────▼─┐   ┌─▼──────────────┐
                 │ Evaluate Complexity Heuristic       │   │ Local Ollama   │
                 │ (Math proofs, long coding, tokens)  │   └────────────────┘
                 └───────────────────┬─────────────────┘
                                     │
                             Is High Complexity?
                                    / \
                                  YES  NO
                                  /     \
        ┌────────────────────────▼─┐   ┌─▼──────────────┐
        │ Route to Cloud Provider  │   │ Local Ollama   │
        │ Failover to Local on err │   └────────────────┘
        └──────────────────────────┘
```

### Privacy Guardrails for Course Materials
If a student query retrieves indexed documents containing academic content, the router **strictly enforces local inference** (`route.provider = "local"`). Confidential lecture slides, assignments, and notes are never leaked to external third-party cloud APIs.

---

## 5. RAG Pipeline & Retrieval Architecture

The RAG engine (`app/knowledge/`) is engineered for high accuracy, grounded source citations, and zero hallucinations.

```
[Raw Document: PDF/DOCX/TXT]
          │
          ▼
[DocumentParser] ──> Preserves page numbers, headings, tables
          │
          ▼
[AcademicChunker] ──> 500-token sliding window + 100-token overlap
          │
          ▼
[Deterministic Hash ID] ──> doc_hash + chunk_index
          │
          ▼
[Embedding Engine & Vector Store] ──> Dense vector + Lexical BM25 terms
          │
   (Query Phase)
          │
          ▼
[Hybrid Retrieval] ──> Score = 0.7 * Semantic_Norm + 0.3 * Lexical
          │
          ▼
[Threshold Filter] ──> Score >= RAG_SIMILARITY_THRESHOLD (0.35)
          │
          ▼
[Citation Validator] ──> Maps inline [1] to verified DocumentChunk
```

### A. Structure-Aware Document Parsing (`app/knowledge/document_parser.py`)
- **PDF**: Uses PyMuPDF (`fitz`) to extract text while tracking physical page numbers, font sizes, and structural headings (`#`, `##`).
- **Word (.docx)**: Uses `python-docx` to extract both paragraph text and structured tabular data.
- **Plain Text / Markdown**: Preserves markdown hierarchy and code blocks.

### B. Anisotropy-Compensated Hybrid Retrieval
Cosine similarity in raw embedding spaces often suffers from vector anisotropy (narrow cones of representation). The vector store remediates this by normalizing cosine distance:
$$\text{SemanticNorm} = \frac{\cos(\theta) + 1.0}{2.0}$$
The final rank score fuses semantic and lexical scores:
$$\text{FinalScore} = 0.7 \cdot \text{SemanticNorm} + 0.3 \cdot \text{LexicalScore}$$

### C. Citation Verification & Prompt Injection Fencing
- **Prompt Injection Defense**: Retrieved chunks are injected inside explicit XML boundaries (`<rag_context>...</rag_context>`), instruct-fencing external instructions from overriding the agent's system prompt.
- **Citation Validator**: The agent post-processes raw LLM responses, matching bracketed references `[1]`, `[2]` against actual chunks in the context window. Hallucinated numbers are stripped before the response reaches the student.

---

## 6. Real-Time Chat Streaming & Reasoning Extraction

When streaming is enabled (`stream=true`), `AcademicAgent.process_query_stream()` yields sequential Server-Sent Event (SSE) chunks formatted as JSON:

```json
data: {"type": "reasoning", "delta": "Let us break down the mathematical definition..."}
data: {"type": "token", "delta": "A graph is bipartite if and only if..."}
data: {"type": "meta", "intent": "tutor", "model": "qwen2.5:1.5b", "used_sources": [...]}
data: [DONE]
```

- Tokens within `<think>` and `</think>` tags are segregated in real-time into the `reasoning` channel, enabling the UI to render collapsible reasoning accordions.
- The `[DONE]` marker terminates the stream cleanly, matching standard OpenAI chunk conventions.

---

## 7. Mastery Progression Calculation (Exponential Moving Average)

Student topic mastery is not a static score, but an evolving Exponential Moving Average (EMA) updated whenever a quiz is graded (`app/memory/database.py`):

$$\text{Mastery}_{t} = (1 - \alpha) \cdot \text{Mastery}_{t-1} + \alpha \cdot \text{Score}_{t}$$

- **Smoothing Factor ($\alpha$)**: Defaults to `0.20`.
- **Clamping**: Mastery is strictly clamped between `[0.0, 1.0]`.
- **Significance**: Early mistakes can be overcome with repeated practice, accurately modeling the learning curve.
