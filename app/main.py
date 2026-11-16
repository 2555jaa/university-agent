"""FastAPI Application entrypoint for Local University AI Agent."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api.routes import router as api_router
from app.memory.database import db
from app.providers.ollama_provider import OllamaProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("university_agent.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown life cycle."""
    logger.info("Starting University AI Agent backend...")
    settings.ensure_directories()
    db.init_db_sync()

    # Check local Ollama connectivity
    ollama_ok = await OllamaProvider().is_available()
    if ollama_ok:
        logger.info(f"Ollama connected successfully at {settings.OLLAMA_BASE_URL}")
    else:
        logger.warning(f"Ollama is currently unreachable at {settings.OLLAMA_BASE_URL}")

    yield

    logger.info("Shutting down University AI Agent backend...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Local University AI Agent: Personal Tutor, Academic Assistant, RAG Knowledge Base, and Restricted PC Tools.",
    lifespan=lifespan
)

# Bind CORS strictly to authorized local origins only
app.add_middleware(
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

app.include_router(api_router, prefix="/api")
app.include_router(api_router)  # Also expose root /v1 for Open WebUI OpenAI API compatibility


WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")


@app.get("/")
async def root():
    """Serve the lightweight local chat interface."""
    return FileResponse(WEB_DIR / "index.html")
