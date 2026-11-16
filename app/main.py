"""FastAPI Application entrypoint and app factory for Local University AI Agent."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, settings, get_settings
from app.containers import ServiceContainer, create_service_container
from app.api.routes import router as api_router
from app.providers.ollama_provider import OllamaProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("university_agent.main")


def create_app(custom_settings: Optional[Settings] = None) -> FastAPI:
    """Application factory constructing an isolated FastAPI application with dependency injection."""
    s = custom_settings or settings
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
