"""RAG service orchestrating document ingestion, retrieval, catalog synchronization, safe deletion, and failure recovery."""

import asyncio
from enum import Enum
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from app.config import settings
from app.knowledge.document_parser import document_parser, DocumentParseError
from app.knowledge.chunker import academic_chunker
from app.knowledge.vector_store import vector_store, SearchResult
from app.security.audit import audit_logger, AuditLogger
from app.memory.database import db as default_db

logger = logging.getLogger("university_agent.knowledge.rag")


class ContextStatus(str, Enum):
    FOUND = "found"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_DOCUMENTS = "no_documents"
    ERROR = "error"


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
    is_duplicate: bool = False
    error_message: Optional[str] = None
    doc_category: Optional[str] = None
    course_code: Optional[str] = None


class GroundedContext(BaseModel):
    context_text: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    has_relevant_info: bool = False
    status: ContextStatus = ContextStatus.NO_DOCUMENTS
    threshold_applied: Optional[float] = None


class RAGService:
    """Manages document ingestion, citation-grounded retrieval, catalog sync, deletion, and failure recovery."""

    def __init__(
        self,
        parser: Optional[Any] = None,
        chunker: Optional[Any] = None,
        store: Optional[Any] = None,
        audit: Optional[Any] = None,
        db: Optional[Any] = None,
        settings_obj: Optional[Any] = None
    ):
        self.settings = settings_obj or settings
        self.parser = parser or document_parser
        self.chunker = chunker or academic_chunker
        self.store = store or vector_store
        if audit is not None:
            self.audit = audit
        elif settings_obj is not None:
            self.audit = AuditLogger(settings_obj=self.settings)
        else:
            self.audit = audit_logger
        self.db = db if db is not None else default_db
        self._ingest_lock = asyncio.Semaphore(1)

    async def ingest_document(
        self,
        file_path: Path,
        subject: Optional[str] = None,
        force_reindex: bool = False
    ) -> IngestionResult:
        """Parse, chunk, and index an academic file into the vector store and synchronize with SQLite catalog."""
        async with self._ingest_lock:
            p = Path(file_path).resolve()
            logger.info(f"Starting ingestion of {p.name} (Subject: {subject}, force: {force_reindex})")

            # Step 1: Validation & Parsing
            try:
                parsed = self.parser.parse_document(p, subject=subject)
            except DocumentParseError as e:
                logger.warning(f"Document parsing failed for {p.name}: {e}")
                if self.db:
                    try:
                        file_bytes = p.read_bytes() if p.exists() and p.is_file() else b""
                        f_hash = hashlib.sha256(file_bytes).hexdigest() if file_bytes else "0" * 64
                        await self.db.upsert_document(
                            doc_id=str(uuid.uuid4()),
                            filename=p.name,
                            file_path=str(p),
                            file_hash=f_hash,
                            file_type=p.suffix.lstrip(".").lower() or "txt",
                            file_size_bytes=len(file_bytes),
                            subject_id=subject,
                            page_count=1,
                            chunk_count=0,
                            status="failed",
                            error_message=str(e)
                        )
                    except Exception:
                        pass
                raise

            file_hash = parsed.file_hash or parsed.sha256
            filename = parsed.filename
            file_type = parsed.file_type
            page_count = len(parsed.pages)
            file_size_bytes = parsed.file_size_bytes

            # Step 2: Check SQLite Documents Catalog & VectorStore for Duplicate / Same-Filename
            existing_by_hash = await self.db.get_document_by_hash(file_hash) if self.db else None
            existing_by_fn = await self.db.get_document_by_filename(filename) if self.db else None
            store_entry = self.store._document_registry.get(filename)

            # Exact Duplicate Detection:
            if not force_reindex:
                is_duplicate = False
                existing_doc_id = None
                existing_chunks = 0
                existing_pages = page_count

                if existing_by_hash and existing_by_hash.get("status") == "indexed":
                    if store_entry and store_entry.get("sha256") == file_hash:
                        is_duplicate = True
                        existing_doc_id = existing_by_hash["id"]
                        existing_chunks = existing_by_hash.get("chunk_count", store_entry.get("chunk_count", 0))
                        existing_pages = existing_by_hash.get("page_count", page_count)

                if is_duplicate:
                    logger.info(f"Document {filename} ({file_hash[:8]}) is already indexed. Skipping re-embedding.")
                    return IngestionResult(
                        document_id=existing_doc_id or parsed.document_id,
                        filename=filename,
                        file_type=file_type,
                        subject=existing_by_hash.get("subject_id") if existing_by_hash else subject,
                        total_pages=existing_pages,
                        total_chunks=existing_chunks,
                        total_characters=parsed.total_characters,
                        sha256=file_hash,
                        status="already_indexed",
                        is_duplicate=True,
                        doc_category=parsed.doc_category,
                        course_code=parsed.course_code
                    )

            # Step 3: Same-Filename Content Update / Replacement:
            if existing_by_fn and existing_by_fn.get("file_hash") != file_hash:
                logger.info(f"Document {filename} content modified on disk (replacing old hash {existing_by_fn.get('file_hash')[:8]} with {file_hash[:8]}).")
                self.store.delete_document_chunks(filename)
                target_doc_id = existing_by_fn["id"]
            elif store_entry and store_entry.get("sha256") != file_hash:
                self.store.delete_document_chunks(filename)
                target_doc_id = store_entry.get("document_id", parsed.document_id)
            elif force_reindex:
                self.store.delete_document_chunks(filename)
                target_doc_id = existing_by_fn["id"] if existing_by_fn else parsed.document_id
            else:
                target_doc_id = existing_by_fn["id"] if existing_by_fn else parsed.document_id

            # Step 4: Track status transitions in SQLite documents catalog
            if self.db:
                await self.db.upsert_document(
                    doc_id=target_doc_id,
                    filename=filename,
                    file_path=str(p),
                    file_hash=file_hash,
                    file_type=file_type,
                    file_size_bytes=file_size_bytes,
                    subject_id=subject,
                    title=parsed.title,
                    page_count=page_count,
                    chunk_count=0,
                    status="processing",
                    error_message=None
                )

            # Step 5: Chunking & Indexing with Failure Recovery
            try:
                chunks = self.chunker.chunk_document(parsed)
                for ch in chunks:
                    ch.document_id = target_doc_id

                indexed_count = await self.store.add_chunks(chunks, file_sha256=file_hash)

                if self.db:
                    await self.db.upsert_document(
                        doc_id=target_doc_id,
                        filename=filename,
                        file_path=str(p),
                        file_hash=file_hash,
                        file_type=file_type,
                        file_size_bytes=file_size_bytes,
                        subject_id=subject,
                        title=parsed.title,
                        page_count=page_count,
                        chunk_count=indexed_count,
                        status="indexed",
                        error_message=None
                    )
            except Exception as e:
                logger.error(f"Processing/indexing failed for {filename}: {e}")
                # Rollback partially written chunks from VectorStore so no orphan chunks remain
                self.store.delete_document_chunks(filename)
                if self.db:
                    try:
                        await self.db.upsert_document(
                            doc_id=target_doc_id,
                            filename=filename,
                            file_path=str(p),
                            file_hash=file_hash,
                            file_type=file_type,
                            file_size_bytes=file_size_bytes,
                            subject_id=subject,
                            title=parsed.title,
                            page_count=page_count,
                            chunk_count=0,
                            status="failed",
                            error_message=str(e)
                        )
                    except Exception:
                        pass
                raise

            self.audit.log_event(
                event_type="document_ingestion",
                action="ingest_file",
                details={
                    "filename": filename,
                    "document_id": target_doc_id,
                    "subject": subject,
                    "pages": page_count,
                    "chunks": indexed_count,
                    "sha256": file_hash,
                    "doc_category": parsed.doc_category
                },
                risk_level="low",
                status="success"
            )

            return IngestionResult(
                document_id=target_doc_id,
                filename=filename,
                file_type=file_type,
                subject=subject,
                total_pages=page_count,
                total_chunks=indexed_count,
                total_characters=parsed.total_characters,
                sha256=file_hash,
                status="success",
                is_duplicate=False,
                doc_category=parsed.doc_category,
                course_code=parsed.course_code
            )

    async def delete_document(
        self,
        doc_id_or_filename: str,
        delete_physical_file: bool = False
    ) -> Dict[str, Any]:
        """Safely delete document chunks from VectorStore, metadata from SQLite catalog, and optionally disk."""
        target = doc_id_or_filename.strip()
        if not target:
            raise ValueError("Document ID or filename cannot be empty.")

        async with self._ingest_lock:
            doc_record = None
            if self.db:
                doc_record = await self.db.get_document(target)
                if not doc_record:
                    doc_record = await self.db.get_document_by_filename(target)

            filename = doc_record["filename"] if doc_record else target
            doc_id = doc_record["id"] if doc_record else target

            # Remove from VectorStore
            chunks_removed = self.store.delete_document_chunks(filename)
            if doc_id != filename:
                chunks_removed += self.store.delete_document_chunks(doc_id)

            # Remove from SQLite catalog
            db_deleted = False
            if self.db:
                db_deleted = await self.db.delete_document_record(doc_id)
                if not db_deleted and filename != doc_id:
                    db_deleted = await self.db.delete_document_record(filename)

            # Optionally remove physical file with path traversal defense
            physical_deleted = False
            if delete_physical_file:
                doc_dir = self.settings.DOCUMENTS_DIR.resolve()
                p = (doc_dir / filename).resolve()
                if p.is_relative_to(doc_dir) and p.exists() and p.is_file():
                    p.unlink()
                    physical_deleted = True
                elif doc_record and doc_record.get("file_path"):
                    rp = Path(doc_record["file_path"]).resolve()
                    if rp.is_relative_to(doc_dir) and rp.exists() and rp.is_file():
                        rp.unlink()
                        physical_deleted = True

            self.audit.log_event(
                event_type="document_deletion",
                action="delete_document",
                details={
                    "target": target,
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunks_removed": chunks_removed,
                    "db_deleted": db_deleted,
                    "physical_deleted": physical_deleted
                },
                risk_level="medium" if delete_physical_file else "low",
                status="success"
            )

            return {
                "ok": True,
                "document_id": doc_id,
                "filename": filename,
                "chunks_removed": chunks_removed,
                "db_deleted": db_deleted,
                "physical_file_deleted": physical_deleted
            }

    async def retry_failed_documents(self) -> Dict[str, Any]:
        """Retry ingestion for any documents in the SQLite catalog marked with status='failed'."""
        if not self.db:
            return {"total_retried": 0, "succeeded": 0, "failed": 0, "details": []}

        failed_docs = await self.db.list_documents(status="failed")
        succeeded = 0
        failed = 0
        details = []

        for rec in failed_docs:
            fn = rec["filename"]
            fp = Path(rec["file_path"])
            if not fp.exists():
                fp = self.settings.DOCUMENTS_DIR / fn

            if not fp.exists() or not fp.is_file():
                failed += 1
                details.append({"filename": fn, "status": "failed", "error": "Physical file not found"})
                continue

            try:
                res = await self.ingest_document(fp, subject=rec.get("subject_id"), force_reindex=True)
                succeeded += 1
                details.append({"filename": fn, "status": "indexed", "chunks": res.total_chunks})
            except Exception as e:
                failed += 1
                details.append({"filename": fn, "status": "failed", "error": str(e)})

        return {
            "total_retried": len(failed_docs),
            "succeeded": succeeded,
            "failed": failed,
            "details": details
        }

    async def retrieve_grounded_context(
        self,
        query: str,
        subject: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        hybrid: Optional[bool] = None
    ) -> GroundedContext:
        """Retrieve relevant passages and format them with explicit, heading-aware citation markers.

        Enforces strict relevance thresholds to prevent hallucination.
        Returns structured ContextStatus (FOUND, INSUFFICIENT_EVIDENCE, NO_DOCUMENTS).
        """
        threshold_applied = score_threshold if score_threshold is not None else getattr(self.settings, "RAG_SIMILARITY_THRESHOLD", 0.45)
        results: List[SearchResult] = await self.store.search(
            query=query,
            top_k=top_k,
            subject=subject,
            score_threshold=score_threshold,
            filters=filters,
            hybrid=hybrid
        )

        if not results:
            total_in_index = self.store.get_index_status().get("total_chunks", 0)
            if total_in_index == 0:
                return GroundedContext(
                    context_text="No academic documents are indexed in the local knowledge base.",
                    citations=[],
                    has_relevant_info=False,
                    status=ContextStatus.NO_DOCUMENTS,
                    threshold_applied=threshold_applied
                )
            return GroundedContext(
                context_text="I don't have enough information in your notes to answer this. No relevant passages met the required similarity threshold.",
                citations=[],
                has_relevant_info=False,
                status=ContextStatus.INSUFFICIENT_EVIDENCE,
                threshold_applied=threshold_applied
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
            has_relevant_info=True,
            status=ContextStatus.FOUND,
            threshold_applied=threshold_applied
        )

    async def reindex_all(self, force_re_embed: bool = False) -> Dict[str, Any]:
        """Completely rebuild the vector store from SQLite documents catalog and physical documents."""
        async with self._ingest_lock:
            doc_dir = self.settings.DOCUMENTS_DIR
            logger.info(f"Starting full reindex_all job (force_re_embed={force_re_embed}, docs_dir={doc_dir})...")

            # 1. Clear VectorStore & Rebuild collection
            if hasattr(self.store, "clear"):
                self.store.clear()
            else:
                if getattr(self.store, "_chroma_client", None) is not None:
                    try:
                        self.store._chroma_client.delete_collection(self.store.COLLECTION_NAME)
                    except Exception:
                        pass
                    self.store._collection = self.store._chroma_client.create_collection(
                        name=self.store.COLLECTION_NAME,
                        metadata={"hnsw:space": "cosine"}
                    )
                self.store._fallback_chunks.clear()
                self.store._fallback_vectors.clear()
                self.store._document_registry.clear()

            # 2. Fetch all document catalog records from SQLite
            catalog_docs = await self.db.list_documents() if self.db else []
            processed_filenames = set()
            reindexed_count = 0
            missing_count = 0
            failed_count = 0
            total_chunks = 0
            details = []

            # 3. Process catalog records
            for doc in catalog_docs:
                filename = doc["filename"]
                processed_filenames.add(filename)
                doc_id = doc["id"]
                subject = doc.get("subject_id")

                file_path = None
                if doc.get("file_path"):
                    candidate = Path(doc["file_path"]).resolve()
                    if candidate.exists() and candidate.is_file():
                        file_path = candidate

                if not file_path and doc_dir.exists():
                    candidate = (doc_dir / filename).resolve()
                    if candidate.exists() and candidate.is_file():
                        file_path = candidate

                if not file_path:
                    # Missing physical file on disk
                    missing_count += 1
                    logger.warning(f"Reindex: physical file missing for catalog record '{filename}' (ID: {doc_id}).")
                    if self.db:
                        await self.db.upsert_document(
                            doc_id=doc_id,
                            filename=filename,
                            file_path=str(doc.get("file_path") or (doc_dir / filename)),
                            file_hash=doc.get("file_hash", "0" * 64),
                            file_type=doc.get("file_type", "unknown"),
                            file_size_bytes=doc.get("file_size_bytes", 0),
                            subject_id=subject,
                            title=doc.get("title"),
                            page_count=doc.get("page_count", 0),
                            chunk_count=0,
                            status="failed",
                            error_message="File missing on disk"
                        )
                    details.append({
                        "filename": filename,
                        "document_id": doc_id,
                        "status": "failed",
                        "error": "File missing on disk"
                    })
                    continue

                try:
                    parsed = self.parser.parse_document(file_path, subject=subject)
                    file_hash = parsed.file_hash or parsed.sha256
                    chunks = self.chunker.chunk_document(parsed)
                    for ch in chunks:
                        ch.document_id = doc_id

                    indexed_chunks = await self.store.add_chunks(chunks, file_sha256=file_hash)
                    total_chunks += indexed_chunks
                    reindexed_count += 1

                    if self.db:
                        await self.db.upsert_document(
                            doc_id=doc_id,
                            filename=filename,
                            file_path=str(file_path),
                            file_hash=file_hash,
                            file_type=parsed.file_type,
                            file_size_bytes=parsed.file_size_bytes,
                            subject_id=subject,
                            title=parsed.title,
                            page_count=len(parsed.pages),
                            chunk_count=indexed_chunks,
                            status="indexed",
                            error_message=None
                        )
                    details.append({
                        "filename": filename,
                        "document_id": doc_id,
                        "status": "reindexed",
                        "chunks": indexed_chunks,
                        "pages": len(parsed.pages)
                    })
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Reindex error parsing/embedding '{filename}': {e}")
                    if self.db:
                        await self.db.upsert_document(
                            doc_id=doc_id,
                            filename=filename,
                            file_path=str(file_path),
                            file_hash=doc.get("file_hash", "0" * 64),
                            file_type=doc.get("file_type", "unknown"),
                            file_size_bytes=doc.get("file_size_bytes", 0),
                            subject_id=subject,
                            title=doc.get("title"),
                            page_count=doc.get("page_count", 0),
                            chunk_count=0,
                            status="failed",
                            error_message=str(e)
                        )
                    details.append({
                        "filename": filename,
                        "document_id": doc_id,
                        "status": "failed",
                        "error": str(e)
                    })

            # 4. Check for physical files in DOCUMENTS_DIR not in catalog
            if doc_dir.exists():
                allowed_exts = {".pdf", ".docx", ".doc", ".txt", ".md"}
                for p in doc_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in allowed_exts and p.name not in processed_filenames:
                        try:
                            res = await self.ingest_document(p, force_reindex=True)
                            reindexed_count += 1
                            total_chunks += res.total_chunks
                            details.append({
                                "filename": p.name,
                                "document_id": res.document_id,
                                "status": "indexed_new",
                                "chunks": res.total_chunks,
                                "pages": res.total_pages
                            })
                        except Exception as e:
                            failed_count += 1
                            details.append({
                                "filename": p.name,
                                "status": "failed",
                                "error": str(e)
                            })

            self.audit.log_event(
                event_type="document_reindex",
                action="reindex_all",
                details={
                    "total_catalog": len(catalog_docs),
                    "reindexed": reindexed_count,
                    "missing": missing_count,
                    "failed": failed_count,
                    "total_chunks": total_chunks
                },
                risk_level="medium",
                status="success"
            )

            return {
                "ok": True,
                "status": "completed",
                "total_catalog_documents": len(catalog_docs),
                "reindexed": reindexed_count,
                "missing": missing_count,
                "failed": failed_count,
                "total_chunks": total_chunks,
                "details": details
            }

    async def reindex_all_documents(self, force: bool = False) -> Dict[str, Any]:
        """Backward-compatible alias for reindex_all."""
        return await self.reindex_all(force_re_embed=force)


rag_service = RAGService()
