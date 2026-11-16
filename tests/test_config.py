"""Configuration tests; all paths are isolated from the project data directory."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings, get_settings


def test_environment_defaults_are_separate(tmp_path):
    base = tmp_path / "project"
    dev = Settings(APP_ENV="development", BASE_DIR=base, _env_file=None)
    test = Settings(APP_ENV="testing", BASE_DIR=base, _env_file=None)
    prod = Settings(APP_ENV="production", BASE_DIR=base, _env_file=None)

    assert dev.DATA_DIR == base / "data"
    assert test.DATA_DIR != dev.DATA_DIR
    assert test.SQLITE_PATH != dev.SQLITE_PATH
    assert test.DOCUMENTS_DIR != dev.DOCUMENTS_DIR
    assert test.CHROMA_DIR != dev.CHROMA_DIR
    assert prod.DATA_DIR == Path.home() / ".local/share/university-agent"
    assert prod.DATA_DIR != dev.DATA_DIR


def test_testing_rejects_development_and_production_paths(tmp_path):
    dev_path = tmp_path / "project" / "data"
    prod_path = Path.home() / ".local/share/university-agent"
    for unsafe in (dev_path, prod_path, dev_path / "subdir"):
        with pytest.raises(ValidationError, match="isolated"):
            Settings(APP_ENV="testing", BASE_DIR=tmp_path / "project", DATA_DIR=unsafe, _env_file=None)


def test_testing_rejects_subpath_overrides_pointing_to_data(tmp_path):
    base = tmp_path / "project"
    unsafe_doc = base / "data" / "documents"
    unsafe_sqlite = base / "data" / "university_agent.db"
    unsafe_chroma = base / "data" / "chroma"
    unsafe_audit = base / "data" / "audit.log"

    for kwargs in (
        {"DOCUMENTS_DIR": unsafe_doc},
        {"SQLITE_PATH": unsafe_sqlite},
        {"CHROMA_DIR": unsafe_chroma},
        {"AUDIT_LOG_PATH": unsafe_audit},
    ):
        with pytest.raises(ValidationError, match="isolated"):
            Settings(APP_ENV="testing", BASE_DIR=base, _env_file=None, **kwargs)


def test_invalid_environment_is_rejected():
    with pytest.raises(ValidationError, match="APP_ENV"):
        Settings(APP_ENV="staging", _env_file=None)
    with pytest.raises(ValidationError, match="APP_ENV"):
        Settings(APP_ENV="prod", _env_file=None)
    with pytest.raises(ValidationError, match="APP_ENV"):
        Settings(APP_ENV="", _env_file=None)


def test_missing_required_cloud_key_is_rejected():
    with pytest.raises(ValidationError, match="CLOUD_API_KEY"):
        Settings(CLOUD_ENABLED=True, CLOUD_API_KEY=None, _env_file=None)


def test_environment_overrides_are_applied(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "isolated"))
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:12345")
    monkeypatch.setenv("LOCAL_MODEL", "test-model")
    config = Settings(_env_file=None)
    assert config.DATA_DIR == (tmp_path / "isolated").resolve()
    assert config.OLLAMA_BASE_URL == "http://localhost:12345"
    assert config.LOCAL_MODEL == "test-model"


def test_secret_is_not_exposed_by_repr_or_validation_error():
    secret = "private-test-token"
    config = Settings(CLOUD_API_KEY=secret, _env_file=None)
    assert secret not in repr(config)
    with pytest.raises(ValidationError) as err:
        Settings(CLOUD_ENABLED=True, CLOUD_API_KEY=None, _env_file=None)
    assert secret not in str(err.value)


def test_constructing_settings_has_no_filesystem_side_effects(tmp_path):
    base = tmp_path / "not-created"
    config = Settings(APP_ENV="development", BASE_DIR=base, _env_file=None)
    assert config.DATA_DIR == base / "data"
    assert not base.exists()


def test_get_settings_factory(tmp_path):
    base = tmp_path / "factory_test"
    # Default factory call
    cfg1 = get_settings(BASE_DIR=base, _env_file=None)
    assert cfg1.APP_ENV == "development"
    assert cfg1.DATA_DIR == base / "data"

    # Environment override
    cfg2 = get_settings(env_override="testing", BASE_DIR=base, _env_file=None)
    assert cfg2.APP_ENV == "testing"
    assert cfg2.DATA_DIR != base / "data"

    cfg3 = get_settings(env_override="production", BASE_DIR=base, _env_file=None)
    assert cfg3.APP_ENV == "production"

    # Field overrides
    cfg4 = get_settings(env_override="testing", PORT=9090, RAG_TOP_K=8, BASE_DIR=base, _env_file=None)
    assert cfg4.PORT == 9090
    assert cfg4.RAG_TOP_K == 8


def test_numeric_ranges_and_constraints(tmp_path):
    base = tmp_path / "ranges"

    # PORT valid [1, 65535]
    Settings(PORT=1, BASE_DIR=base, _env_file=None)
    Settings(PORT=65535, BASE_DIR=base, _env_file=None)
    with pytest.raises(ValidationError, match="PORT"):
        Settings(PORT=0, BASE_DIR=base, _env_file=None)
    with pytest.raises(ValidationError, match="PORT"):
        Settings(PORT=65536, BASE_DIR=base, _env_file=None)

    # RAG_TOP_K >= 1
    Settings(RAG_TOP_K=1, BASE_DIR=base, _env_file=None)
    with pytest.raises(ValidationError, match="RAG_TOP_K"):
        Settings(RAG_TOP_K=0, BASE_DIR=base, _env_file=None)
    with pytest.raises(ValidationError, match="RAG_TOP_K"):
        Settings(RAG_TOP_K=-2, BASE_DIR=base, _env_file=None)

    # RAG_SIMILARITY_THRESHOLD [0.0, 1.0]
    Settings(RAG_SIMILARITY_THRESHOLD=0.0, BASE_DIR=base, _env_file=None)
    Settings(RAG_SIMILARITY_THRESHOLD=1.0, BASE_DIR=base, _env_file=None)
    with pytest.raises(ValidationError, match="RAG_SIMILARITY_THRESHOLD"):
        Settings(RAG_SIMILARITY_THRESHOLD=-0.01, BASE_DIR=base, _env_file=None)
    with pytest.raises(ValidationError, match="RAG_SIMILARITY_THRESHOLD"):
        Settings(RAG_SIMILARITY_THRESHOLD=1.01, BASE_DIR=base, _env_file=None)

    # RAG_HYBRID_ALPHA [0.0, 1.0]
    Settings(RAG_HYBRID_ALPHA=0.0, BASE_DIR=base, _env_file=None)
    Settings(RAG_HYBRID_ALPHA=1.0, BASE_DIR=base, _env_file=None)
    with pytest.raises(ValidationError, match="RAG_HYBRID_ALPHA"):
        Settings(RAG_HYBRID_ALPHA=-0.1, BASE_DIR=base, _env_file=None)
    with pytest.raises(ValidationError, match="RAG_HYBRID_ALPHA"):
        Settings(RAG_HYBRID_ALPHA=1.1, BASE_DIR=base, _env_file=None)

    # MAX_UPLOAD_SIZE_MB > 0
    Settings(MAX_UPLOAD_SIZE_MB=1, BASE_DIR=base, _env_file=None)
    with pytest.raises(ValidationError, match="MAX_UPLOAD_SIZE_MB"):
        Settings(MAX_UPLOAD_SIZE_MB=0, BASE_DIR=base, _env_file=None)
    with pytest.raises(ValidationError, match="MAX_UPLOAD_SIZE_MB"):
        Settings(MAX_UPLOAD_SIZE_MB=-5, BASE_DIR=base, _env_file=None)

    # ACTION_TIMEOUT_SECONDS > 0
    Settings(ACTION_TIMEOUT_SECONDS=1, BASE_DIR=base, _env_file=None)
    with pytest.raises(ValidationError, match="ACTION_TIMEOUT_SECONDS"):
        Settings(ACTION_TIMEOUT_SECONDS=0, BASE_DIR=base, _env_file=None)
    with pytest.raises(ValidationError, match="ACTION_TIMEOUT_SECONDS"):
        Settings(ACTION_TIMEOUT_SECONDS=-10, BASE_DIR=base, _env_file=None)


def test_ensure_directories_creates_on_demand(tmp_path):
    isolated = tmp_path / "demand_test"
    config = Settings(APP_ENV="testing", DATA_DIR=isolated, _env_file=None)

    # Before ensure_directories, directories must not exist
    assert not isolated.exists()
    assert not config.DOCUMENTS_DIR.exists()
    assert not config.CHROMA_DIR.exists()
    assert not config.SQLITE_PATH.parent.exists()
    assert not config.AUDIT_LOG_PATH.parent.exists()

    # Call ensure_directories
    config.ensure_directories()

    # Now directories must exist
    assert isolated.exists()
    assert config.DOCUMENTS_DIR.exists()
    assert config.CHROMA_DIR.exists()
    assert config.SQLITE_PATH.parent.exists()
    assert config.AUDIT_LOG_PATH.parent.exists()


def test_config_import_and_application_import_in_testing_environment(tmp_path):
    isolated = tmp_path / "isolated-app-import"
    development_db = PROJECT_ROOT / "data" / "university_agent.db"
    before = development_db.stat().st_mtime_ns if development_db.exists() else None
    env = os.environ.copy()
    env.update({"APP_ENV": "testing", "DATA_DIR": str(isolated), "PYTHONPATH": str(PROJECT_ROOT)})
    code = f"""
from pathlib import Path
isolated_path = Path({repr(str(isolated))})
import app.config
from app.main import app
assert app.title
assert not isolated_path.exists(), "importing app.main must not create DATA_DIR"
from fastapi.testclient import TestClient
with TestClient(app) as client:
    res = client.get('/api/health')
    assert res.status_code == 200
assert isolated_path.exists(), "lifespan startup must create isolated DATA_DIR"
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env,
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert isolated.exists()
    after = development_db.stat().st_mtime_ns if development_db.exists() else None
    assert after == before
