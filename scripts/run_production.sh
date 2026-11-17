#!/usr/bin/env bash
# ==============================================================================
# Production Runner for University AI Agent with Auto-Restart & Health Watchdog
# ==============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Production configuration defaults
export APP_ENV="${APP_ENV:-production}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-4}"
BACKOFF_SECONDS="${BACKOFF_SECONDS:-2}"
MAX_RAPID_CRASHES="${MAX_RAPID_CRASHES:-10}"

echo "========================================================================"
echo "      🚀 Launching Local University AI Agent in Production Mode         "
echo "========================================================================"
echo "Environment: $APP_ENV"
echo "Listening:   http://$HOST:$PORT"
echo "Workers:     $WORKERS"
echo "PID:         $$"
echo "========================================================================"

# Virtual environment validation
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_UVICORN="$PROJECT_ROOT/.venv/bin/uvicorn"

if [ ! -x "$VENV_PYTHON" ] || [ ! -x "$VENV_UVICORN" ]; then
    echo "ERROR: Virtual environment not found or uvicorn not installed at $VENV_UVICORN." >&2
    exit 1
fi

CHILD_PID=0
SHUTDOWN_SIGNALED=0

cleanup() {
    SHUTDOWN_SIGNALED=1
    echo ""
    echo "[$ (date -u +"%Y-%m-%dT%H:%M:%SZ")] Received termination signal. Initiating graceful shutdown..."
    if [ "$CHILD_PID" -ne 0 ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        echo "Sending SIGTERM to worker group (PID: $CHILD_PID)..."
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    echo "[$ (date -u +"%Y-%m-%dT%H:%M:%SZ")] University AI Agent stopped cleanly."
    exit 0
}

trap cleanup SIGINT SIGTERM SIGHUP

crash_count=0
last_crash_time=0

while [ "$SHUTDOWN_SIGNALED" -eq 0 ]; do
    echo "[$ (date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting ASGI server with $WORKERS worker(s)..."

    "$VENV_UVICORN" app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --lifespan on \
        --no-access-log &

    CHILD_PID=$!
    set +e
    wait "$CHILD_PID"
    EXIT_CODE=$?
    set -e

    if [ "$SHUTDOWN_SIGNALED" -ne 0 ]; then
        break
    fi

    NOW=$(date +%s)
    if [ $((NOW - last_crash_time)) -lt 60 ]; then
        crash_count=$((crash_count + 1))
    else
        crash_count=1
    fi
    last_crash_time=$NOW

    echo "[$ (date -u +"%Y-%m-%dT%H:%M:%SZ")] WARNING: Server process exited unexpectedly with code $EXIT_CODE."
    echo "[$ (date -u +"%Y-%m-%dT%H:%M:%SZ")] Rapid crash count: $crash_count / $MAX_RAPID_CRASHES."

    if [ "$crash_count" -ge "$MAX_RAPID_CRASHES" ]; then
        echo "[$ (date -u +"%Y-%m-%dT%H:%M:%SZ")] FATAL: Exceeded maximum consecutive rapid crashes. Halting auto-restart." >&2
        exit 1
    fi

    echo "[$ (date -u +"%Y-%m-%dT%H:%M:%SZ")] Sleeping for ${BACKOFF_SECONDS}s before automatic restart..."
    sleep "$BACKOFF_SECONDS"
done
