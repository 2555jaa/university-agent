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
