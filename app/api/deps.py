"""FastAPI dependencies for accessing injected services."""

from fastapi import Request

from app.containers import ServiceContainer, get_default_container
from app.config import Settings
from app.memory.database import DatabaseManager
from app.knowledge.vector_store import VectorStore
from app.knowledge.embeddings import EmbeddingManager
from app.knowledge.rag_service import RAGService
from app.agent.orchestrator import AcademicAgent
from app.providers.router import ModelRouter
from app.security.audit import AuditLogger
from app.security.validator import SecurityValidator


def get_services(request: Request) -> ServiceContainer:
    """Retrieve the application's service container from request.app.state."""
    services = getattr(request.app.state, "services", None)
    if services is not None:
        return services
    return get_default_container()


def get_settings(request: Request) -> Settings:
    return get_services(request).settings


def get_db(request: Request) -> DatabaseManager:
    return get_services(request).db


def get_vector_store(request: Request) -> VectorStore:
    return get_services(request).vector_store


def get_embedding_manager(request: Request) -> EmbeddingManager:
    return get_services(request).embedding_manager


def get_rag_service(request: Request) -> RAGService:
    return get_services(request).rag_service


def get_agent(request: Request) -> AcademicAgent:
    return get_services(request).agent


def get_router(request: Request) -> ModelRouter:
    return get_services(request).router


def get_audit_logger(request: Request) -> AuditLogger:
    return get_services(request).audit_logger


def get_security_validator(request: Request) -> SecurityValidator:
    return get_services(request).security_validator
