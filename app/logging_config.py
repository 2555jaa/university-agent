"""Structured logging configuration and request tracing for University AI Agent."""

import logging
from contextvars import ContextVar
from typing import Optional

# Request ID context variable for tracing logs across async tasks
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class RequestIDFilter(logging.Filter):
    """Logging filter that injects the current async request ID into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get() or "-"
        return True


def setup_logging(log_level: str = "INFO") -> None:
    """Configure centralized structured logging format and levels."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s [req:%(request_id)s]: %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(RequestIDFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers on re-initialization
    if not any(isinstance(h, logging.StreamHandler) and any(isinstance(f, RequestIDFilter) for f in h.filters) for h in root_logger.handlers):
        root_logger.handlers.clear()
        root_logger.addHandler(handler)

    # Set external libraries to warning to reduce noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
