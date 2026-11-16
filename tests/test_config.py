"""Configuration tests; all paths are isolated from the project data directory."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings


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


def test_invalid_environment_is_rejected():
    with pytest.raises(ValidationError, match="APP_ENV"):
        Settings(APP_ENV="staging", _env_file=None)


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


def test_config_import_and_application_import_in_testing_environment(tmp_path):
    isolated = tmp_path / "isolated-app-import"
    development_db = PROJECT_ROOT / "data" / "university_agent.db"
    before = development_db.stat().st_mtime_ns if development_db.exists() else None
    env = os.environ.copy()
    env.update({"APP_ENV": "testing", "DATA_DIR": str(isolated), "PYTHONPATH": str(PROJECT_ROOT)})
    code = "import app.config; from app.main import app; assert app.title"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env,
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert isolated.exists()
    after = development_db.stat().st_mtime_ns if development_db.exists() else None
    assert after == before
