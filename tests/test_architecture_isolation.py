"""Tests verifying zero import-time side effects and isolated dependency injection."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, Settings, get_settings
from app.main import create_app


def test_import_time_has_zero_filesystem_side_effects(tmp_path):
    """Importing any core app module must create ZERO files or directories."""
    isolated_root = tmp_path / "ghost_root"
    assert not isolated_root.exists()

    env = os.environ.copy()
    env.update({
        "APP_ENV": "testing",
        "DATA_DIR": str(isolated_root),
        "PYTHONPATH": str(PROJECT_ROOT)
    })

    test_code = """
import app.main
import app.api.routes
import app.memory.database
import app.knowledge.vector_store
import app.security.audit
import app.security.validator
import app.containers
"""

    result = subprocess.run(
        [sys.executable, "-c", test_code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30
    )

    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
    assert not isolated_root.exists(), f"Import-time side effect detected! {isolated_root} was created."


def test_create_app_with_isolated_settings_wiring(tmp_path):
    """create_app wires all services to the isolated custom settings."""
    isolated_dir = tmp_path / "isolated_app_test"
    custom_settings = Settings(APP_ENV="testing", DATA_DIR=isolated_dir, _env_file=None)

    app_instance = create_app(custom_settings)
    services = app_instance.state.services

    assert services.settings.DATA_DIR == isolated_dir.resolve()
    assert services.db.db_path == (isolated_dir / "university_agent.db").resolve()
    assert services.vector_store.persist_dir == (isolated_dir / "chroma").resolve()
    assert services.audit_logger.log_path == (isolated_dir / "audit.log").resolve()
    assert services.rag_service.store is services.vector_store
    assert services.agent.db is services.db
    assert services.agent.rag is services.rag_service
    assert services.agent.router is services.router


def test_openapi_schema_has_canonical_routes_without_collisions():
    """Canonical routes use /api prefix and root aliases do not cause collisions in OpenAPI schema."""
    app_instance = create_app()
    schema = app_instance.openapi()
    paths = schema.get("paths", {})

    # OpenAPI schema should contain /api routes
    assert "/api/health" in paths
    assert "/api/chat" in paths
    assert "/api/documents" in paths
    assert "/api/system/status" in paths

    # Unprefixed root aliases must have include_in_schema=False so they do not collide
    assert "/health" not in paths
    assert "/chat" not in paths


@pytest.mark.asyncio
async def test_request_id_tracing_middleware(async_client):
    """Every response includes an X-Request-ID, echoing custom header if provided."""
    # 1. Without client header: auto-generate UUID4
    res = await async_client.get("/api/health")
    assert res.status_code == 200
    req_id = res.headers.get("x-request-id")
    assert req_id is not None
    assert len(req_id) > 10

    # 2. With client header: echo back client-supplied ID
    custom_id = "custom-req-trace-987654"
    res_custom = await async_client.get("/api/health", headers={"X-Request-ID": custom_id})
    assert res_custom.status_code == 200
    assert res_custom.headers.get("x-request-id") == custom_id


@pytest.mark.asyncio
async def test_standardized_error_handling(async_client, test_app):
    """Validation (422), HTTP (404), and 500 errors return standardized JSON schema with request_id."""
    # 1. 422 RequestValidationError (POST /api/chat with invalid body)
    res_422 = await async_client.post("/api/chat", json={"invalid_field": 123})
    assert res_422.status_code == 422
    data_422 = res_422.json()
    assert "detail" in data_422  # FastAPI backward compatibility
    assert "error" in data_422
    assert data_422["error"]["code"] == "validation_error"
    assert "request_id" in data_422["error"]
    assert res_422.headers.get("x-request-id") == data_422["error"]["request_id"]

    # 2. 404 StarletteHTTPException
    res_404 = await async_client.get("/api/nonexistent-route-for-testing")
    assert res_404.status_code == 404
    data_404 = res_404.json()
    assert "detail" in data_404
    assert "error" in data_404
    assert data_404["error"]["code"] == "http_404"
    assert "request_id" in data_404["error"]
    assert res_404.headers.get("x-request-id") == data_404["error"]["request_id"]

    # 3. 500 Unhandled runtime error with DEBUG=False: no stack trace or sensitive path leakage
    @test_app.get("/api/test-trigger-500", include_in_schema=False)
    async def trigger_500():
        raise RuntimeError("simulated critical crash at /private/secrets/db.key")

    client_sync = TestClient(test_app, raise_server_exceptions=False)
    res_500 = client_sync.get("/api/test-trigger-500", headers={"X-Request-ID": "trace-500-test"})
    assert res_500.status_code == 500
    data_500 = res_500.json()
    assert data_500["error"]["code"] == "internal_server_error"
    assert data_500["error"]["request_id"] == "trace-500-test"
    # Ensure no internal path leaks in message when DEBUG=False
    assert "/private/secrets" not in data_500["error"]["message"]
    assert "traceback" not in data_500["error"]


@pytest.mark.asyncio
async def test_honest_health_and_system_status_reporting(async_client):
    """Health and status endpoints honestly report component states."""
    res_health = await async_client.get("/api/health")
    assert res_health.status_code == 200
    h_data = res_health.json()

    assert h_data["environment"] == "testing"
    assert h_data["database"]["status"] == "ok"
    assert "path" in h_data["database"]
    assert h_data["vector_store"]["status"] == "ok"
    assert h_data["vector_store"]["mode"] in ("fallback_json", "chromadb")
    assert "total_chunks" in h_data["vector_store"]
    assert "total_documents" in h_data["vector_store"]
    assert "provider" in h_data["embeddings"]
    assert "is_neural" in h_data["embeddings"]
    assert "ollama" in h_data["llm_providers"]
    assert "cloud" in h_data["llm_providers"]

    res_sys = await async_client.get("/api/system/status")
    assert res_sys.status_code == 200
    s_data = res_sys.json()
    assert s_data["environment"] == "testing"
    assert s_data["database"]["status"] == "ok"
    assert s_data["vector_store"]["status"] == "ok"
    assert "llm_providers" in s_data
    assert "memory" in s_data
