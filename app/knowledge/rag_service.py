"""RAG service orchestrating document ingestion, retrieval, heading-aware citation generation, and idempotent reindexing."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.config import settings
from app.knowledge.document_parser import document_parser
from app.knowledge.chunker import academic_chunker
from app.knowledge.vector_store import vector_store, SearchResult
from app.security.audit import audit_logger

logger = logging.getLogger("university_agent.knowledge.rag")


class IngestionResult(BaseModel):
    document_id: str
    filename: str
    file_type: str
    subject: Optional[str]
    total_pages: int
    total_chunks: int
    total_characters: int
    sha256: str = ""
    status: str = "success"


class GroundedContext(BaseModel):
    context_text: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    has_relevant_info: bool = False


class RAGService:
    """Manages document ingestion, citation-grounded retrieval, and query formatting."""

    def __init__(
        self,
        parser: Optional[Any] = None,
        chunker: Optional[Any] = None,
        store: Optional[Any] = None,
        audit: Optional[Any] = None,
        settings_obj: Optional[Any] = None
    ):
        self.settings = settings_obj or settings
        self.parser = parser or document_parser
        self.chunker = chunker or academic_chunker
        self.store = store or vector_store
        self.audit = audit or audit_logger
        self._ingest_lock = asyncio.Semaphore(1)

    async def ingest_document(self, file_path: Path, subject: Optional[str] = None) -> IngestionResult:
        """Parse, chunk, and index an academic file into the vector store."""
        async with self._ingest_lock:
            logger.info(f"Starting ingestion of {file_path.name} (Subject: {subject})")
            parsed = self.parser.parse_document(file_path, subject=subject)
            chunks = self.chunker.chunk_document(parsed)
            indexed_count = await self.store.add_chunks(chunks, file_sha256=parsed.sha256)

        self.audit.log_event(
            event_type="document_ingestion",
            action="ingest_file",
            details={
                "filename": parsed.filename,
                "document_id": parsed.document_id,
                "subject": subject,
                "pages": len(parsed.pages),
                "chunks": indexed_count,
                "sha256": parsed.sha256
            },
            risk_level="low",
            status="success"
        )

        return IngestionResult(
            document_id=parsed.document_id,
            filename=parsed.filename,
            file_type=parsed.file_type,
            subject=subject,
            total_pages=len(parsed.pages),
            total_chunks=indexed_count,
            total_characters=parsed.total_characters,
            sha256=parsed.sha256
        )

    async def retrieve_grounded_context(
        self,
        query: str,
        subject: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> GroundedContext:
        """Retrieve relevant passages and format them with explicit, heading-aware citation markers."""
        results: List[SearchResult] = await self.store.search(
            query=query,
            top_k=top_k,
            subject=subject,
            score_threshold=score_threshold
        )

        if not results:
            return GroundedContext(
                context_text="No relevant academic documents found in local knowledge base.",
                citations=[],
                has_relevant_info=False
            )

        context_blocks = []
        citations = []

        for i, res in enumerate(results, 1):
            ref_tag = f"[{i}]"
            heading_info = f" § {res.heading}" if res.heading else ""
            context_blocks.append(
                f"{ref_tag} Source: {res.filename} | Page {res.page_number}{heading_info} | Score: {res.score}\n"
                f"{res.text.strip()}"
            )
            citations.append({
                "index": i,
                "chunk_id": res.chunk_id,
                "filename": res.filename,
                "page_number": res.page_number,
                "heading": res.heading,
                "subject": res.subject,
                "score": res.score,
                "semantic_score": res.semantic_score,
                "lexical_score": res.lexical_score,
                "snippet": res.text[:200]
            })

        formatted_context = "\n\n---\n\n".join(context_blocks)
        return GroundedContext(
            context_text=formatted_context,
            citations=citations,
            has_relevant_info=True
        )

    async def reindex_all_documents(self, force: bool = False) -> Dict[str, Any]:
        """Safely reindex all documents stored in the documents directory with idempotency."""
        doc_dir = settings.DOCUMENTS_DIR
        if not doc_dir.exists():
            return {"status": "no_documents_directory", "reindexed": 0, "skipped": 0, "failed": 0}

        reindexed = 0
        skipped = 0
        failed = 0
        details = []

        allowed_exts = {".pdf", ".docx", ".doc", ".txt", ".md"}
        files = [p for p in doc_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts and p.stat().st_size > 0]

        for file_path in files:
            try:
                import hashlib
                current_sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
                existing_entry = self.store._document_registry.get(file_path.name)

                if not force and existing_entry and existing_entry.get("sha256") == current_sha256:
                    skipped += 1
                    details.append({"filename": file_path.name, "action": "skipped_unchanged"})
                    continue

                res = await self.ingest_document(file_path, subject=existing_entry.get("subject") if existing_entry else None)
                reindexed += 1
                details.append({
                    "filename": file_path.name,
                    "action": "reindexed",
                    "chunks": res.total_chunks,
                    "pages": res.total_pages
                })
            except Exception as e:
                failed += 1
                logger.error(f"Reindexing failed for {file_path.name}: {e}")
                details.append({"filename": file_path.name, "action": "failed", "error": str(e)})

        return {
            "status": "success",
            "total_files": len(files),
            "reindexed": reindexed,
            "skipped": skipped,
            "failed": failed,
            "details": details
        }


rag_service = RAGService()
