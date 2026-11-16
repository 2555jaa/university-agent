"""Service container for dependency injection and lifecycle management."""

from dataclasses import dataclass
from typing import Optional

from app.config import Settings, settings
from app.memory.database import DatabaseManager, db
from app.knowledge.vector_store import VectorStore, vector_store
from app.knowledge.embeddings import EmbeddingManager, embedding_manager
from app.knowledge.rag_service import RAGService, rag_service
from app.agent.orchestrator import AcademicAgent, academic_agent
from app.providers.router import ModelRouter, router
from app.security.audit import AuditLogger, audit_logger
from app.security.validator import SecurityValidator, security_validator


@dataclass
class ServiceContainer:
    """Holds all configured application services."""
    settings: Settings
    db: DatabaseManager
    vector_store: VectorStore
    embedding_manager: EmbeddingManager
    rag_service: RAGService
    agent: AcademicAgent
    router: ModelRouter
    audit_logger: AuditLogger
    security_validator: SecurityValidator


def create_service_container(custom_settings: Optional[Settings] = None) -> ServiceContainer:
    """Factory creating an isolated ServiceContainer for a specific Settings configuration."""
    s = custom_settings or settings
    audit = AuditLogger(log_path=s.AUDIT_LOG_PATH, settings_obj=s)
    validator = SecurityValidator(settings_obj=s)
    database = DatabaseManager(db_path=s.SQLITE_PATH, settings_obj=s)
    embeddings = EmbeddingManager(settings_obj=s)
    store = VectorStore(persist_dir=s.CHROMA_DIR, settings_obj=s, embedding_mgr=embeddings)
    model_router = ModelRouter(settings_obj=s)
    rag = RAGService(store=store, audit=audit, settings_obj=s)
    agent = AcademicAgent(
        router_instance=model_router,
        rag_instance=rag,
        db_instance=database,
        audit=audit,
        settings_obj=s
    )

    return ServiceContainer(
        settings=s,
        db=database,
        vector_store=store,
        embedding_manager=embeddings,
        rag_service=rag,
        agent=agent,
        router=model_router,
        audit_logger=audit,
        security_validator=validator
    )


def get_default_container() -> ServiceContainer:
    """Return a container backed by the default module instances."""
    return ServiceContainer(
        settings=settings,
        db=db,
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        rag_service=rag_service,
        agent=academic_agent,
        router=router,
        audit_logger=audit_logger,
        security_validator=security_validator
    )
