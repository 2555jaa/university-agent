"""Central application settings with explicit development, testing and production paths."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _inside(path: Path, parent: Path) -> bool:
    """Return whether path is parent or a descendant of parent."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class Settings(BaseSettings):
    """Validated configuration. Constructing settings never creates files/directories."""

    APP_ENV: Literal["development", "testing", "production"] = "development"
    APP_NAME: str = "Local University AI Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: Union[bool, str] = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    BASE_DIR: Path = PROJECT_ROOT
    DATA_DIR: Optional[Path] = None
    DOCUMENTS_DIR: Optional[Path] = None
    CHROMA_DIR: Optional[Path] = None
    SQLITE_PATH: Optional[Path] = None
    AUDIT_LOG_PATH: Optional[Path] = None

    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    LOCAL_MODEL: str = "qwen2.5:1.5b"
    LOCAL_FALLBACK_MODEL: str = "qwen2.5:0.5b"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_PROVIDER: Literal["auto", "ollama", "deterministic_hash"] = "auto"

    RAG_SIMILARITY_THRESHOLD: float = 0.45
    RAG_MIN_SEMANTIC_ONLY_SCORE: float = 0.50
    RAG_TOP_K: int = 4
    RAG_HYBRID_SEARCH: bool = True
    RAG_HYBRID_ALPHA: float = 0.65

    CLOUD_ENABLED: bool = False
    CLOUD_PROVIDER: Literal["gemini", "openai"] = "gemini"
    CLOUD_API_KEY: Optional[str] = Field(default=None, repr=False)
    CLOUD_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    CLOUD_MODEL: str = "gemini-2.5-flash"

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    RESTRICT_TO_LOCAL: bool = True
    ALLOWED_DIRECTORIES: list[str] = Field(default_factory=lambda: ["data", "workspace", "documents"])
    MAX_UPLOAD_SIZE_MB: int = 25
    ACTION_TIMEOUT_SECONDS: int = 30

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_flag(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "debug")
        return False

    @model_validator(mode="after")
    def resolve_and_validate_paths(self) -> "Settings":
        base = self.BASE_DIR.expanduser().resolve()
        object.__setattr__(self, "BASE_DIR", base)
        development_data = base / "data"
        production_data = Path.home() / ".local" / "share" / "university-agent"
        testing_data = Path(tempfile.gettempdir()) / f"university-agent-tests-{getattr(os, 'getuid', lambda: 0)()}"

        defaults = {
            "development": development_data,
            "testing": testing_data,
            "production": production_data,
        }
        data_dir = (self.DATA_DIR or defaults[self.APP_ENV]).expanduser().resolve()

        if self.APP_ENV == "testing":
            protected = (development_data.resolve(), production_data.resolve())
            if any(_inside(data_dir, root) or _inside(root, data_dir) for root in protected):
                raise ValueError("Testing DATA_DIR must be isolated from development and production data")

        derived = {
            "DATA_DIR": data_dir,
            "DOCUMENTS_DIR": data_dir / "documents",
            "CHROMA_DIR": data_dir / "chroma",
            "SQLITE_PATH": data_dir / "university_agent.db",
            "AUDIT_LOG_PATH": data_dir / "audit.log",
        }
        for name, default in derived.items():
            override = getattr(self, name)
            value = override.expanduser().resolve() if override is not None else default.resolve()
            if self.APP_ENV == "testing":
                for root in (development_data.resolve(), production_data.resolve()):
                    if _inside(value, root) or _inside(root, value):
                        raise ValueError(f"Testing {name} must be isolated from development and production data")
            object.__setattr__(self, name, value)

        if self.CLOUD_ENABLED and not self.CLOUD_API_KEY:
            raise ValueError("CLOUD_API_KEY is required when CLOUD_ENABLED=true")
        if self.PORT < 1 or self.PORT > 65535:
            raise ValueError("PORT must be between 1 and 65535")
        if self.MAX_UPLOAD_SIZE_MB < 1:
            raise ValueError("MAX_UPLOAD_SIZE_MB must be positive")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        hide_input_in_errors=True,
    )

    def ensure_directories(self) -> None:
        """Create configured runtime directories when application startup requests it."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# Kept for existing modules. No filesystem side effects occur at import time.
settings = Settings()
