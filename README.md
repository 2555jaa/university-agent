# Local University AI Agent

A privacy-first, modular, hybrid AI agent built for university students, personal tutoring, academic research, and safe PC assistance on Linux.

Designed specifically for resource-constrained environments (Intel Core i5, 8GB RAM, integrated graphics), this system optimizes for low memory footprint, zero unnecessary background bloat, and offline-first private document processing.

---

## 🏛 System Architecture & Capabilities

```
┌─────────────────────────────────────────────────────────────┐
│                 User Interfaces                             │
│   • Built-in Web UI (http://127.0.0.1:8000)                 │
│   • Open WebUI Chat (OpenAI API /v1/chat/completions)       │
│   • REST API / OpenAPI Swagger (/docs)                      │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
┌──────────────▼──────────────────────────────▼───────────────┐
│                     FastAPI Agent Backend                   │
│   • University Tutor Mode     • Academic Research Mode      │
│   • Assignment Planner        • Quiz & Mastery Evaluator    │
└──────────────┬──────────────┬───────────────┬───────────────┘
               │              │               │
┌──────────────▼────┐  ┌──────▼──────┐  ┌─────▼───────────────┐
│  Hybrid Router    │  │ RAG Engine  │  │ SQLite Memory       │
│ • Local: Ollama   │  │ • PyMuPDF   │  │ • Subjects & Topics │
│   (Qwen 1.5B/1.7B)│  │ • docx/txt  │  │ • Mastery Progress  │
│ • Cloud: Optional │  │ • Vector DB │  │ • Tasks & Deadlines │
│   (Gemini/OpenAI) │  │ • Citations │  │ • Chat Sessions     │
└───────────────────┘  └─────────────┘  └─────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                  Restricted PC Tools & Security             │
│ • Safe File Reader     • Material Organizer                 │
│ • Whitelisted Shell    • Human-in-the-Loop Approval Manager │
│ • Append-Only Forensic Audit Logging (audit.log)            │
└─────────────────────────────────────────────────────────────┘
```

### Key Capabilities

1. **University Tutor**:
   - Socratic step-by-step concept instruction.
   - Adaptive explanations matched to student mastery level.
   - Dynamic practice quiz generation and mistake evaluation.
   - Persistent mastery score recalculation.

2. **Academic Research & Grounded RAG**:
   - Ingestion of PDF, Word (.docx), and plain text (.txt, .md).
   - Intelligent sliding-window chunking with page number retention.
   - Strict source attribution `[Source: <filename>, Page: <page>]`.
   - Zero hallucination policy: unsupported claims are explicitly rejected.

3. **Persistent SQLite Memory**:
   - Survives application restarts.
   - Tracks subjects, topics, quiz scores, academic tasks, deadlines, and conversation history.

4. **Hybrid AI Model Routing**:
   - Local Ollama inference for private course documents, tutoring, and summarization.
   - Cloud AI routing (Gemini / OpenAI) for advanced research, complex proofs, and code debugging.
   - Guardrail: Confidential and private academic documents are strictly restricted to local execution.

5. **Restricted PC Assistance & Security**:
   - Whitelisted inspection commands only (`ls`, `stat`, `date`, `pwd`, `wc`, `head`, `tail`).
   - High-risk operations (file deletion, arbitrary shell commands) require explicit human approval tokens.
   - All tool calls and security decisions recorded to `audit.log`.

---

## 💻 Hardware & Environment Constraints

- **Detected OS**: Ubuntu 26.04 LTS (x86_64)
- **CPU**: Intel Core i5-1035G1 (4 cores, 8 threads)
- **RAM**: ~7 GiB (Optimized for < 2GB available headroom)
- **GPU**: Intel Iris Plus Graphics (CPU inference mode)
- **Local Model**: Qwen 2.5 1.5B (~986 MB in-memory)

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure Python 3.12+ (or 3.14) and Ollama are installed:
```bash
# Verify Ollama
ollama --version

# Verify Python
python3 --version
```

### 2. Configure Environment
Copy the example environment file:
```bash
cp .env.example .env
```

The default `APP_ENV=development` preserves the project's existing `./data`
location. For isolated test runs, use `APP_ENV=testing`; this selects a dedicated
temporary data root and rejects paths overlapping development or production data.
Production defaults to `~/.local/share/university-agent`, or can use an explicit
`DATA_DIR`. Environment variables override `.env` values.

Edit `.env` if you want to enable cloud models (optional):
```ini
OLLAMA_BASE_URL=http://127.0.0.1:11434
LOCAL_MODEL=qwen2.5:1.5b

# Optional Cloud API
CLOUD_ENABLED=false
# Set CLOUD_API_KEY only in your private .env when cloud access is enabled.
CLOUD_MODEL=gemini-2.5-flash
```

### 3. Launch Ollama & Pull Local Model
```bash
ollama serve &
ollama pull qwen2.5:1.5b
```

### 4. Run the Agent Backend
```bash
./scripts/run_agent.sh
```
The server will start on `http://127.0.0.1:8000`:
- **Web Interface**: `http://127.0.0.1:8000/`
- **Interactive API Documentation**: `http://127.0.0.1:8000/docs`
- **OpenAI Compatible API**: `http://127.0.0.1:8000/v1`

---

## 🌐 Connecting Open WebUI

Open WebUI can be connected to the agent using the OpenAI API standard:

1. Launch Open WebUI (via Docker or local package):
   ```bash
   docker run -d -p 3000:8080 -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 -e OPENAI_API_KEY=local --add-host=host.docker.internal:host-gateway --name open-webui --restart always ghcr.io/open-webui/open-webui:main
   ```
2. Navigate to `http://localhost:3000`.
3. In Open WebUI Settings → Connections → OpenAI API:
   - URL: `http://localhost:8000/v1` (or `http://host.docker.internal:8000/v1` in Docker)
   - Key: `local`
4. The models `university-tutor` and `university-research` will appear automatically in the dropdown!

---

## 📚 Document Ingestion & RAG Usage

Ingest your course notes or syllabus via CLI:
```bash
# Ingest single PDF or DOCX file
.venv/bin/python scripts/ingest_documents.py /path/to/lecture_notes.pdf "Computer Science"

# Ingest entire folder
.venv/bin/python scripts/ingest_documents.py ./data/documents "Physics"
```

Or ingest via API:
```bash
curl -X POST http://127.0.0.1:8000/api/documents/upload \
  -F "file=@lecture1.pdf" \
  -F "subject=Calculus"
```

---

## 🧪 Running Verification Tests

Run the full automated test suite covering security, memory, RAG, and providers:
```bash
.venv/bin/pytest -v
```

All 22 unit and integration tests verify:
- Path traversal defenses and approval workflows
- SQLite topic mastery and quiz progress persistence
- PDF, DOCX, and TXT parsing with PyMuPDF and python-docx
- Vector similarity search and citation grounding
- Hybrid local/cloud routing and fallback resilience
- OpenAI-compatible endpoints for Open WebUI
