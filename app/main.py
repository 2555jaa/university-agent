"""FastAPI Application entrypoint and app factory for Local University AI Agent."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, settings, get_settings
from app.containers import ServiceContainer, create_service_container
from app.api.routes import router as api_router
from app.providers.ollama_provider import OllamaProvider
from app.logging_config import setup_logging, request_id_ctx

setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger("university_agent.main")


def create_app(custom_settings: Optional[Settings] = None) -> FastAPI:
    """Application factory constructing an isolated FastAPI application with dependency injection."""
    s = custom_settings or settings
    setup_logging(s.LOG_LEVEL)
    container = create_service_container(s)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        """Application startup and shutdown lifecycle."""
        logger.info(f"Starting University AI Agent backend ({s.APP_ENV} mode)...")
        s.ensure_directories()
        container.db.init_db_sync()

        # Check local Ollama connectivity
        ollama_ok = await OllamaProvider(settings_obj=s).is_available()
        if ollama_ok:
            logger.info(f"Ollama connected successfully at {s.OLLAMA_BASE_URL}")
        else:
            logger.warning(f"Ollama is currently unreachable at {s.OLLAMA_BASE_URL}")

        yield

        logger.info("Shutting down University AI Agent backend...")

    app_instance = FastAPI(
        title=s.APP_NAME,
        version=s.APP_VERSION,
        description="Local University AI Agent: Personal Tutor, Academic Assistant, RAG Knowledge Base, and Restricted PC Tools.",
        lifespan=lifespan
    )

    # Attach service container to app state
    app_instance.state.services = container

    # Request ID and tracing middleware
    @app_instance.middleware("http")
    async def request_tracing_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_ctx.set(req_id)
        request.state.request_id = req_id
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(f"{request.method} {request.url.path} completed {response.status_code} in {duration_ms}ms")
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            request_id_ctx.reset(token)

    # Standardized Exception Handlers
    @app_instance.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or str(uuid.uuid4())
        raw_errors = exc.errors()
        encoded_errors = jsonable_encoder(raw_errors)
        logger.warning(f"Request validation error on {request.method} {request.url.path}: {encoded_errors}")
        return JSONResponse(
            status_code=422,
            headers={"X-Request-ID": req_id},
            content={
                "detail": encoded_errors,
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "details": encoded_errors,
                    "request_id": req_id
                }
            }
        )

    @app_instance.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        req_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or str(uuid.uuid4())
        logger.warning(f"HTTP exception {exc.status_code} on {request.method} {request.url.path}: {exc.detail}")
        headers = {"X-Request-ID": req_id}
        if getattr(exc, "headers", None):
            headers.update(exc.headers)
        return JSONResponse(
            status_code=exc.status_code,
            headers=headers,
            content={
                "detail": exc.detail,
                "error": {
                    "code": f"http_{exc.status_code}",
                    "message": str(exc.detail),
                    "request_id": req_id
                }
            }
        )

    @app_instance.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or str(uuid.uuid4())
        logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
        msg = "Internal Server Error"
        content = {
            "detail": msg,
            "error": {
                "code": "internal_server_error",
                "message": str(exc) if s.DEBUG else "An unexpected error occurred.",
                "request_id": req_id
            }
        }
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            headers={"X-Request-ID": req_id},
            content=content
        )

    # Bind CORS strictly to authorized local origins only
    app_instance.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:8080",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8080"
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Canonical API routing with /api prefix documented in OpenAPI schema
    app_instance.include_router(api_router, prefix="/api")
    # Root alias router without duplicate OpenAPI schema collisions (supports Open WebUI /v1/...)
    app_instance.include_router(api_router, include_in_schema=False)

    web_dir = Path(__file__).resolve().parent / "web"
    static_dir = web_dir / "static"
    if static_dir.exists():
        app_instance.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app_instance.get("/", include_in_schema=False)
    async def root():
        """Serve the lightweight local chat interface."""
        index_file = web_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"status": "ok", "app": s.APP_NAME}

    return app_instance


# Default application instance for uvicorn app.main:app
app = create_app()

__all__ = ["create_app", "app"]
