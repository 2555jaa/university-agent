#  Local University AI Agent

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688.svg)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57.svg)](https://sqlite.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **local-first, privacy-respecting academic copilot and tutoring agent** designed for university students, educators, and researchers.

Optimized specifically for resource-constrained consumer hardware (e.g. Intel Core i5, 8GB RAM, integrated graphics), the agent delivers grounded Socratic tutoring, accurate document research, automated practice quizzes with exponential moving average (EMA) mastery tracking, and safe sandboxed system execution with zero telemetry or data leakage.

---

## 🏛 System Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Interfaces & Clients                       │
│   • Built-in Reactive Web UI (SSE Streaming, Badges, Modals)           │
│   • Open WebUI Integration (Standard OpenAI /v1/chat/completions)      │
│   • REST API / Interactive Swagger Documentation (/docs)               │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
┌───────────────────▼────────────────────────────────▼───────────────────┐
│                          FastAPI Agent Backend                         │
│   • Tutor Mode (Socratic)            • Research Mode (Citations)       │
│   • Assignment Guardrail Planner     • Interactive Quiz Evaluator      │
└─────────┬───────────────────┬───────────────────┬──────────────────────┘
          │                   │                   │
┌─────────▼─────────┐ ┌───────▼─────────┐ ┌───────▼──────────────────────┐
│   Hybrid Router   │ │   RAG Pipeline  │ │  SQLite Memory & Progress    │
│ • Local: Ollama   │ │ • PyMuPDF/DOCX  │ │ • Schema Migrations (v1-v3)  │
│   (Qwen / Llama)  │ │ • Hybrid Search │ │ • Topic Mastery EMA Tracking │
│ • Cloud: Failover │ │ • Citations     │ │ • Academic Tasks & Deadlines │
│   (Gemini/OpenAI) │ │ • XML Fencing   │ │ • Study Goals Progress       │
└───────────────────┘ └─────────────────┘ └──────────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────────────────┐
│                    Restricted Sandbox & Tool Security                  │
│ • AST Code Execution Sandbox (Timeout & Import Isolation)              │
│ • Strict Terminal Command Whitelisting & Shell Injection Prevention    │
│ • Path Traversal Defenses (`is_relative_to` Base Enclosure)            │
│ • Human-in-the-Loop Approval Tokens (Single-Use, Cryptographic Hash)   │
│ • Append-Only Forensic Audit Logging (`logs/audit.log`)                │
└────────────────────────────────────────────────────────────────────────┘
```

---

##  Key Features

1. **Real-Time SSE Streaming Chat & Collapsible Reasoning Paths**:
   - Streams assistant tokens via Server-Sent Events (`POST /api/chat` and `/v1/chat/completions`).
   - Automatically extracts `<think>...</think>` tokens and renders them inside interactive collapsible reasoning accordions in the UI.

2. **Grounded Academic RAG & Accurate Citation Badges**:
   - Ingests **PDF** (PyMuPDF `fitz`), **Word** (`.docx`), **Markdown**, and **Plain Text**.
   - Preserves page numbers, section headers, tables, and theorems.
   - Hybrid Search combining dense vector cosine similarity (ChromaDB) and lexical BM25-style keyword search with anisotropy normalization.
   - Renders numbered citation badges (`[1]`, `[2]`) that smoothly scroll to source cards containing exact filenames, page numbers, and preview text.
   - Unsupported or unverified queries trigger strict threshold rejection rather than model hallucination.

3. **Interactive Practice Quizzes & Mastery Tracking**:
   - Generates multi-question diagnostic quizzes on any tracked subject or topic.
   - Interactive quiz modal allowing option selection and instant grading.
   - Evaluates answers against ground truth course chunks and updates topic mastery using an **Exponential Moving Average (EMA)** (`α = 0.2`):
     $$\text{Mastery}_{t} = (1 - \alpha) \cdot \text{Mastery}_{t-1} + \alpha \cdot \text{Score}$$
   - Live mastery progress bars displayed on the student dashboard.

4. **Academic Task Management & Study Goals**:
   - Track assignments, reading lists, and project milestones with priority badges (`High`, `Medium`, `Low`).
   - Filter views for **Upcoming** tasks (due within 7 days) and **Overdue** tasks.
   - Goal progress tracking with interactive sliders and quick-increment completion buttons.

5. **Hardened Security & Sandboxed Tool Execution**:
   - **AST Validation**: Code execution evaluates Python syntax trees, blocking forbidden system imports (`os`, `sys`, `subprocess`, `socket`, `shutil`).
   - **Command Whitelisting**: Strictly restricts terminal actions to safe developer utilities (`python`, `node`, `ls`, `cat`, `echo`, `grep`, `ollama`, `pytest`) while rejecting shell metacharacters (`;`, `&&`, `|`, `>`, `<`).
   - **Path Traversal Protection**: Enforces canonical path verification (`Path.resolve().is_relative_to()`) to lock operations within authorized workspace boundaries.
   - **Human-in-the-Loop Approvals**: High-risk actions generate single-use, cryptographically verified approval tokens requiring user confirmation in the chat interface.

6. **Open WebUI & OpenAI Standard Compatibility**:
   - Serves `GET /v1/models` and `POST /v1/chat/completions` (streaming & non-streaming).
   - Exposes virtual academic persona models:
     - `university-agent`: Default auto-routing intent
     - `university-agent-tutor`: Socratic tutoring persona
     - `university-agent-research`: Deep research with citations
     - `university-agent-assignment`: Assignment breakdown and planning

7. **Production Resilience & Auto-Restart**:
   - Production ASGI runner with worker pooling, crash watchdog, and auto-restart loop (`scripts/run_production.sh`).
   - Graceful shutdown lifecycle: flushes pending index writes, terminates ChromaDB subsystems, and triggers SQLite WAL checkpointing (`PRAGMA wal_checkpoint(TRUNCATE);`).
   - Production Docker container and `systemd` service unit template included.

---

##  Hardware & System Requirements

- **Operating System**: Linux (Ubuntu 22.04+ / Debian 12+ recommended; macOS and WSL2 compatible).
- **Python**: Version 3.12+ (tested up to 3.14).
- **RAM**: Minimum 4 GB; 8 GB recommended for local LLM inference.
- **Local LLM Engine**: [Ollama](https://ollama.com/) (e.g. `qwen2.5:3b`, `qwen2.5:7b`, or `llama3.2:3b`).

---

##  Quickstart Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-username/university-agent.git
cd university-agent

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Settings

Copy the example environment configuration:

```bash
cp .env.example .env
```

Review or edit `.env` parameters:

```ini
APP_ENV=development
LOG_LEVEL=INFO

# Ollama local inference
OLLAMA_BASE_URL=http://127.0.0.1:11434
LOCAL_MODEL=qwen2.5:3b

# Optional Cloud API Failover (Gemini / OpenAI)
CLOUD_ENABLED=false
CLOUD_API_KEY=
CLOUD_MODEL=gemini-2.5-flash

# Storage Paths (defaults to ./data in development)
WORKSPACE_DIR=./data/workspace
DOCUMENTS_DIR=./data/documents
CHROMA_DIR=./data/chroma
SQLITE_PATH=./data/university_agent.db
```

### 3. Launch Ollama & Pull Local Model

```bash
# Start Ollama service (if not already running as a system daemon)
ollama serve &

# Pull the lightweight academic local model
ollama pull qwen2.5:3b
```

### 4. Seed Academic Workspace (Optional)

Seed sample course syllabi, subjects, discrete math topics, and initial tasks:

```bash
.venv/bin/python scripts/seed_database.py
```

### 5. Run the Application

#### Development Mode (with hot-reload):
```bash
./scripts/run_agent.sh
```

#### Production Mode (with worker group & auto-restart watchdog):
```bash
./scripts/run_production.sh
```

The service is available at:
- **Interactive Web Interface**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger / OpenAPI Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Diagnostic System Health**: [http://127.0.0.1:8000/api/system/health](http://127.0.0.1:8000/api/system/health)
- **Open WebUI Endpoint**: `http://127.0.0.1:8000/v1`

---

##  Connecting Open WebUI

To use the agent inside [Open WebUI](https://github.com/open-webui/open-webui):

1. Start Open WebUI:
   ```bash
   docker run -d -p 3000:8080 \
     -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
     -e OPENAI_API_KEY=local \
     --add-host=host.docker.internal:host-gateway \
     --name open-webui \
     ghcr.io/open-webui/open-webui:main
   ```
2. Navigate to `http://localhost:3000`.
3. In **Settings → Connections → OpenAI API**:
   - **URL**: `http://host.docker.internal:8000/v1` (or `http://localhost:8000/v1` for local installs)
   - **API Key**: `local`
4. The academic personas (`university-agent`, `university-agent-tutor`, `university-agent-research`) will appear directly in your model selection dropdown.

---

##  Example API Usage

### 1. Check System Health & Diagnostics
```bash
curl -s http://127.0.0.1:8000/api/system/health | jq .
```

### 2. Send Chat Query (Non-Streaming)
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain how virtual memory paging handles page faults.",
    "mode": "tutor",
    "stream": false
  }' | jq .
```

### 3. Stream Chat Query via SSE
```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are discrete probability distributions?",
    "stream": true
  }'
```

### 4. Ingest Course Document
```bash
curl -X POST http://127.0.0.1:8000/api/documents/upload \
  -F "file=@./data/documents/discrete_math.txt" \
  -F "subject=Mathematics"
```

### 5. Generate & Evaluate a Topic Quiz
```bash
# 1. Generate Quiz (answers hidden)
curl -X POST http://127.0.0.1:8000/api/agent/quiz \
  -H "Content-Type: application/json" \
  -d '{
    "topic_name": "Virtual Memory",
    "num_questions": 3,
    "hide_answers": true
  }' | jq .

# 2. Evaluate Submitted Answers & Update Mastery EMA
curl -X POST http://127.0.0.1:8000/api/agent/quiz/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "topic_id": "top_vm_101",
    "answers": {"0": "Paging", "1": "TLB", "2": "Thrashing"},
    "quiz_data": { ... }
  }' | jq .
```

---

##  Testing & Verification

Run the full automated test suite containing unit, integration, RAG, and security tests:

```bash
.venv/bin/pytest -v
```

All 198 automated tests run against an isolated sandbox environment ensuring zero modifications to production `./data` files:
- Architecture & dependency injection isolation
- SQLite migrations, pragmas, and atomic rollback
- PyMuPDF and python-docx parsing integrity
- Semantic + BM25 hybrid retrieval and threshold filtering
- AST sandboxing and command injection prevention
- Real-time SSE streaming and OpenAI compatibility
- Interactive quiz grading and EMA mastery calculations

---

##  License

This project is licensed under the MIC License. See [LICENSE](LICENSE) for details.
