"""Global pytest configuration and sandbox isolation fixtures."""

import hashlib
import os
from pathlib import Path
from typing import Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from app.config import PROJECT_ROOT, Settings
from app.containers import ServiceContainer, create_service_container
from app.main import app, create_app


def snapshot_data_dir() -> Dict[Path, str]:
    """Calculate SHA-256 digests for all files in the development ./data directory."""
    data_dir = PROJECT_ROOT / "data"
    hashes: Dict[Path, str] = {}
    if not data_dir.exists():
        return hashes
    for p in data_dir.rglob("*"):
        if p.is_file():
            # Skip transient files
            if "tmp" in p.name or p.name.endswith(".tmp"):
                continue
            hashes[p] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


@pytest.fixture(autouse=True)
def isolated_sandbox(monkeypatch, tmp_path):
    """Guarantees every test runs in an isolated ephemeral directory with zero ./data side effects."""
    # 1. Snapshot real ./data before test
    pre_snapshot = snapshot_data_dir()

    # 2. Setup isolated testing directory and settings
    test_data = tmp_path / "sandbox_data"
    test_data.mkdir(parents=True, exist_ok=True)
    test_settings = Settings(APP_ENV="testing", DATA_DIR=test_data, _env_file=None)
    test_settings.ensure_directories()

    # 3. Create isolated service container and initialize database schema
    container = create_service_container(test_settings)
    container.db.init_db_sync()

    # 4. Monkeypatch global singletons across modules
    for mod_name in (
        "app.config",
        "app.memory.database",
        "app.knowledge.vector_store",
        "app.knowledge.rag_service",
        "app.agent.orchestrator",
        "app.providers.router",
        "app.security.audit",
        "app.security.validator",
        "app.knowledge.embeddings",
        "app.api.routes",
    ):
        monkeypatch.setattr(f"{mod_name}.settings", test_settings, raising=False)

    monkeypatch.setattr("app.memory.database.db", container.db)
    monkeypatch.setattr("app.knowledge.vector_store.vector_store", container.vector_store)
    monkeypatch.setattr("app.knowledge.rag_service.rag_service", container.rag_service)
    monkeypatch.setattr("app.agent.orchestrator.academic_agent", container.agent)
    monkeypatch.setattr("app.providers.router.router", container.router)
    monkeypatch.setattr("app.security.audit.audit_logger", container.audit_logger)
    monkeypatch.setattr("app.security.validator.security_validator", container.security_validator)
    monkeypatch.setattr("app.knowledge.embeddings.embedding_manager", container.embedding_manager)

    # In app.agent.orchestrator module globals:
    monkeypatch.setattr("app.agent.orchestrator.db", container.db, raising=False)
    monkeypatch.setattr("app.agent.orchestrator.rag_service", container.rag_service, raising=False)
    monkeypatch.setattr("app.agent.orchestrator.router", container.router, raising=False)
    monkeypatch.setattr("app.agent.orchestrator.audit_logger", container.audit_logger, raising=False)
    monkeypatch.setattr("app.agent.orchestrator.settings", test_settings, raising=False)

    # In app.knowledge.rag_service and vector_store:
    monkeypatch.setattr("app.knowledge.rag_service.vector_store", container.vector_store, raising=False)
    monkeypatch.setattr("app.knowledge.rag_service.default_db", container.db, raising=False)
    monkeypatch.setattr("app.knowledge.rag_service.embedding_manager", container.embedding_manager, raising=False)
    monkeypatch.setattr("app.knowledge.rag_service.audit_logger", container.audit_logger, raising=False)
    monkeypatch.setattr("app.knowledge.vector_store.embedding_manager", container.embedding_manager, raising=False)

    # In app.api.routes module globals:
    monkeypatch.setattr("app.api.routes.db", container.db, raising=False)
    monkeypatch.setattr("app.api.routes.vector_store", container.vector_store, raising=False)
    monkeypatch.setattr("app.api.routes.rag_service", container.rag_service, raising=False)
    monkeypatch.setattr("app.api.routes.academic_agent", container.agent, raising=False)
    monkeypatch.setattr("app.api.routes.security_validator", container.security_validator, raising=False)
    monkeypatch.setattr("app.api.routes.audit_logger", container.audit_logger, raising=False)
    monkeypatch.setattr("app.api.routes.embedding_manager", container.embedding_manager, raising=False)

    # In app.main singleton app:
    monkeypatch.setattr(app.state, "services", container, raising=False)

    # In app.tools & app.services:
    monkeypatch.setattr("app.tools.filesystem_tools.security_validator", container.security_validator, raising=False)
    monkeypatch.setattr("app.tools.filesystem_tools.audit_logger", container.audit_logger, raising=False)
    monkeypatch.setattr("app.tools.terminal_tools.security_validator", container.security_validator, raising=False)
    monkeypatch.setattr("app.tools.terminal_tools.audit_logger", container.audit_logger, raising=False)
    monkeypatch.setattr("app.tools.browser_tools.audit_logger", container.audit_logger, raising=False)
    try:
        from app.services.file_service import file_service
        monkeypatch.setattr(file_service, "_validator", container.security_validator, raising=False)
        monkeypatch.setattr(file_service, "_audit_logger", container.audit_logger, raising=False)
    except ImportError:
        pass

    yield container

    # 5. Post-test verification that real ./data was untouched
    post_snapshot = snapshot_data_dir()
    for path, pre_hash in pre_snapshot.items():
        assert path.exists(), f"CRITICAL: Test deleted development file: {path}"
        post_hash = post_snapshot.get(path)
        assert post_hash == pre_hash, f"CRITICAL: Test mutated development file: {path} (pre: {pre_hash}, post: {post_hash})"
    for path in post_snapshot:
        assert path in pre_snapshot, f"CRITICAL: Test created new file in development data: {path}"


@pytest.fixture
def test_settings(isolated_sandbox) -> Settings:
    """Return the active isolated Settings instance."""
    return isolated_sandbox.settings


@pytest.fixture
def test_container(isolated_sandbox) -> ServiceContainer:
    """Return the active isolated ServiceContainer."""
    return isolated_sandbox


@pytest.fixture
def test_app(test_settings) -> FastAPI:
    """Return an isolated FastAPI application instance."""
    return create_app(test_settings)


@pytest.fixture
def client(test_app):
    """Sync TestClient for FastAPI endpoints."""
    with TestClient(test_app) as c:
        yield c


@pytest.fixture
async def async_client(test_app):
    """AsyncClient for FastAPI endpoints."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
