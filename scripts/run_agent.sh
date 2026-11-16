#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================================="
echo "      🎓 Starting Local University AI Agent Backend       "
echo "=========================================================="

# 1. Check Python virtual environment
if [ ! -d ".venv" ]; then
    echo "Virtual environment (.venv) not found. Creating..."
    if command -v /tmp/uv >/dev/null 2>&1; then
        /tmp/uv venv .venv
        /tmp/uv pip install -r requirements.txt
    elif command -v uv >/dev/null 2>&1; then
        uv venv .venv
        uv pip install -r requirements.txt
    else
        python3 -m venv .venv
        .venv/bin/pip install -r requirements.txt
    fi
fi

# 2. Check if Ollama is running
echo "Checking Ollama connectivity on 127.0.0.1:11434..."
if curl -s http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    echo "✓ Ollama is online and responsive."
else
    echo "⚠ Ollama is not running on 127.0.0.1:11434."
    echo "  Starting Ollama daemon in background..."
    ollama serve >/dev/null 2>&1 &
    sleep 3
fi

# 3. Start FastAPI Application
echo "Starting University AI Agent on http://127.0.0.1:8000..."
echo "Interactive Web Dashboard: http://127.0.0.1:8000/"
echo "OpenAPI Swagger Docs:      http://127.0.0.1:8000/docs"
echo "Open WebUI Compatible API: http://127.0.0.1:8000/v1"
echo "=========================================================="

exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
