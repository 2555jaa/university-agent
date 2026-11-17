"""Vector Store for academic document chunks supporting ChromaDB persistent storage, metadata filtering, and index migration."""

import os
import json
import math
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.knowledge.chunker import DocumentChunk
from app.knowledge.embeddings import (
    embedding_manager,
    BaseEmbeddingProvider,
    EmbeddingInfo,
    STOPWORDS,
    ModelMigrationError,
)

logger = logging.getLogger("university_agent.knowledge.vector_store")


@dataclass
class SearchResult(DocumentChunk):
    score: float = 0.0
    semantic_score: float = 0.0
    lexical_score: float = 0.0

    @property
    def final_score(self) -> float:
        """Alias for combined hybrid relevance score."""
        return self.score

    def citation(self) -> str:
        """Standard academic citation string with section/heading awareness."""
        subj = f"[{self.subject}] " if self.subject else ""
        head = f" § {self.heading}" if self.heading else ""
        return f"{subj}{self.filename} (Page {self.page_number}{head})"


def _build_chroma_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Translate dictionary of filter key-values into ChromaDB where syntax."""
    if not filters:
        return None
    clauses = []
    for k, v in filters.items():
        if v is not None:
            clauses.append({k: v})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class VectorStore:
    """Production vector database managing neural document embeddings, ChromaDB persistence, and metadata filtering."""

    INDEX_VERSION = 2
    COLLECTION_NAME = "university_agent_chunks"

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        settings_obj: Optional[Any] = None,
        embedding_mgr: Optional[Any] = None,
        lazy_init: bool = False
    ):
        self.settings = settings_obj or settings
        self.persist_dir = persist_dir or self.settings.CHROMA_DIR
        if embedding_mgr is not None:
            self.embedding_manager = embedding_mgr
        elif settings_obj is not None:
            from app.knowledge.embeddings import EmbeddingManager
            self.embedding_manager = EmbeddingManager(settings_obj=self.settings)
        else:
            self.embedding_manager = embedding_manager
        self._chroma_client = None
        self._collection = None
        self._chroma_initialized = False

        # In-memory document & vector storage (synchronized cache)
        self._fallback_chunks: Dict[str, DocumentChunk] = {}
        self._fallback_vectors: Dict[str, List[float]] = {}
        self._document_registry: Dict[str, Dict[str, Any]] = {}
        self._index_metadata: Dict[str, Any] = {
            "index_version": self.INDEX_VERSION,
            "embedding_provider": "unknown",
            "embedding_model": "unknown",
            "embedding_dimension": 0,
            "vector_dimension": 0,
            "is_neural": False,
            "created_at": time.time(),
            "updated_at": time.time()
        }

        self._fallback_file = self.persist_dir / "fallback_index.json"

        # Initialize ChromaDB persistent client if directory already exists and not lazy
        if not lazy_init and self.persist_dir.exists():
            self._init_chroma(create_dirs=False)
            if self._fallback_file.exists():
                self._load_fallback_index()

    @property
    def chroma_client(self):
        if not self._chroma_initialized:
            self._init_chroma(create_dirs=True)
        return self._chroma_client

    @chroma_client.setter
    def chroma_client(self, val):
        self._chroma_client = val

    @property
    def collection(self):
        if not self._chroma_initialized:
            self._init_chroma(create_dirs=True)
        return self._collection

    @collection.setter
    def collection(self, val):
        self._collection = val

    def _init_chroma(self, create_dirs: bool = False) -> None:
        """Initialize ChromaDB PersistentClient and collection."""
        if self._chroma_initialized and self._collection is not None:
            return
        if not self.persist_dir.exists() and not create_dirs:
            return
        self._chroma_initialized = True
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir.resolve()),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB PersistentClient initialized with collection '{self.COLLECTION_NAME}'.")
            self._restore_from_chroma()
        except Exception as e:
            env = getattr(self.settings, "APP_ENV", "development").lower()
            if env in ("development", "production"):
                logger.error(f"Failed to initialize ChromaDB PersistentClient in {env} mode: {e}")
                raise RuntimeError(f"ChromaDB initialization failed in {env} environment: {e}") from e
            logger.info(f"ChromaDB not available in testing environment ({e}); using embedded index.")

    def _restore_from_chroma(self) -> None:
        """Synchronize in-memory registries from persistent ChromaDB collection."""
        if self.collection is None:
            return
        try:
            total = self.collection.count()
            if total == 0:
                return
            data = self.collection.get(include=["documents", "metadatas", "embeddings"])
            c_ids = data.get("ids", [])
            c_docs = data.get("documents", [])
            c_metas = data.get("metadatas", [])
            c_vecs = data.get("embeddings", [])
            if c_vecs is None:
                c_vecs = []
            for i, (cid, doc, meta) in enumerate(zip(c_ids, c_docs, c_metas)):
                meta = meta or {}
                sub = meta.get("subject_id") or meta.get("subject")
                ch = DocumentChunk(
                    chunk_id=cid,
                    document_id=meta.get("document_id", ""),
                    filename=meta.get("filename", ""),
                    subject=sub,
                    page_number=int(meta.get("page_number", 1)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc or "",
                    heading=meta.get("heading") or None,
                    metadata=meta
                )
                self._fallback_chunks[cid] = ch
                if i < len(c_vecs) and c_vecs[i] is not None:
                    self._fallback_vectors[cid] = [float(x) for x in c_vecs[i]]
                fn = ch.filename
                if fn and fn not in self._document_registry:
                    self._document_registry[fn] = {
                        "filename": fn,
                        "document_id": ch.document_id,
                        "subject": sub,
                        "sha256": meta.get("file_hash", ""),
                        "chunk_count": 0,
                        "total_pages": int(meta.get("page_number", 1)),
                        "indexed_at": time.time()
                    }
                if fn:
                    self._document_registry[fn]["chunk_count"] += 1

            if c_metas and c_metas[0]:
                first_meta = c_metas[0]
                stored_dim = int(first_meta.get("embedding_dimension", 0))
                stored_model = str(first_meta.get("embedding_model", "unknown"))
                if stored_dim:
                    self._index_metadata["embedding_dimension"] = stored_dim
                    self._index_metadata["vector_dimension"] = stored_dim
                if stored_model != "unknown":
                    self._index_metadata["embedding_model"] = stored_model

            logger.info(f"Restored {len(self._fallback_chunks)} chunks from ChromaDB collection.")
        except Exception as e:
            logger.warning(f"Failed to restore chunks from ChromaDB: {e}")

    def clear(self) -> None:
        """Purge all chunks and vectors, rebuilding the persistent ChromaDB collection from scratch."""
        if self._chroma_client is not None:
            try:
                self._chroma_client.delete_collection(self.COLLECTION_NAME)
            except Exception:
                pass
            self._collection = self._chroma_client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        self._fallback_chunks.clear()
        self._fallback_vectors.clear()
        self._document_registry.clear()
        self._index_metadata.update({
            "embedding_provider": "unknown",
            "embedding_model": "unknown",
            "embedding_dimension": 0,
            "vector_dimension": 0,
            "is_neural": False,
            "updated_at": time.time()
        })
        if self._fallback_file.exists():
            try:
                self._fallback_file.unlink()
            except Exception:
                pass
        logger.info(f"VectorStore cleared; collection '{self.COLLECTION_NAME}' rebuilt.")

    def _backup_index(self) -> Optional[Path]:
        """Create a timestamped backup of the current index file."""
        try:
            timestamp = int(time.time())
            backup_path = self.persist_dir / f"fallback_index.json.bak.{timestamp}"
            if self._fallback_file.exists():
                backup_path.write_bytes(self._fallback_file.read_bytes())
            else:
                backup_path.write_text(json.dumps(self._index_metadata))
            logger.info(f"Created index backup at: {backup_path.name}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create index backup: {e}")
            return None

    def _load_fallback_index(self) -> None:
        """Load fallback vector index and metadata from disk."""
        if not self._fallback_file.exists():
            return
        try:
            with open(self._fallback_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._index_metadata = data.get("metadata", self._index_metadata)
            self._document_registry = data.get("documents", {})

            for item in data.get("chunks", []):
                ch = DocumentChunk(
                    chunk_id=item["chunk_id"],
                    document_id=item["document_id"],
                    filename=item["filename"],
                    subject=item.get("subject"),
                    page_number=item.get("page_number", 1),
                    chunk_index=item.get("chunk_index", 0),
                    text=item["text"],
                    heading=item.get("heading") or item.get("metadata", {}).get("heading"),
                    metadata=item.get("metadata", {})
                )
                self._fallback_chunks[ch.chunk_id] = ch

            self._fallback_vectors = data.get("vectors", {})
            logger.info(
                f"Loaded fallback vector index: {len(self._fallback_chunks)} chunks, "
                f"model: {self._index_metadata.get('embedding_model')}, "
                f"dim: {self._index_metadata.get('embedding_dimension')}."
            )
        except Exception as e:
            logger.error(f"Failed to load fallback index: {e}")

    def _save_fallback_index(self) -> None:
        """Persist vector index and metadata to disk atomically."""
        try:
            chunks_data = [
                {
                    "chunk_id": ch.chunk_id,
                    "document_id": ch.document_id,
                    "filename": ch.filename,
                    "subject": ch.subject,
                    "page_number": ch.page_number,
                    "chunk_index": ch.chunk_index,
                    "text": ch.text,
                    "heading": ch.heading,
                    "metadata": ch.metadata
                }
                for ch in self._fallback_chunks.values()
            ]

            payload = {
                "metadata": {
                    **self._index_metadata,
                    "index_version": self.INDEX_VERSION,
                    "updated_at": time.time(),
                    "total_chunks": len(self._fallback_chunks),
                    "total_documents": len(self._document_registry)
                },
                "documents": self._document_registry,
                "chunks": chunks_data,
                "vectors": self._fallback_vectors
            }

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            temp_file = self.persist_dir / f"fallback_index.tmp.{os.getpid()}"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_file, self._fallback_file)
        except Exception as e:
            logger.error(f"Failed to save fallback index: {e}")

    async def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Generate embedding vector using the active provider."""
        provider = await (self.embedding_manager or embedding_manager).get_provider()
        return await provider.get_embedding(text, is_query=is_query)

    async def ensure_index_compatibility(self) -> bool:
        """Check if stored vectors are compatible with the active embedding provider."""
        provider = await (self.embedding_manager or embedding_manager).get_provider()
        info = provider.get_info()

        stored_dim = self._index_metadata.get("vector_dimension") or self._index_metadata.get("embedding_dimension", 0)
        stored_model = self._index_metadata.get("embedding_model", "unknown")

        existing_count = self.collection.count() if self.collection is not None else len(self._fallback_chunks)
        if existing_count == 0 or stored_dim == 0:
            self._index_metadata.update({
                "embedding_provider": info.provider,
                "embedding_model": info.model,
                "embedding_dimension": info.dimension,
                "vector_dimension": info.dimension,
                "is_neural": info.is_neural
            })
            if self.collection is None:
                self._save_fallback_index()
            return True

        if stored_dim != info.dimension or (stored_model not in ("unknown", "") and stored_model != info.model):
            return False

        return True

    async def migrate_index(self, target_provider: Optional[BaseEmbeddingProvider] = None) -> bool:
        """Safely re-embed all indexed chunks using the target embedding provider."""
        if target_provider is None:
            target_provider = await (self.embedding_manager or embedding_manager).get_provider()

        info = target_provider.get_info()
        logger.info(f"Starting index migration to provider '{info.provider}' (model: '{info.model}', dim: {info.dimension})...")

        # 1. Backup existing index before migration
        self._backup_index()

        # 2. Gather all chunks
        chunks_list = list(self._fallback_chunks.values())
        if not chunks_list and self.collection is not None and self.collection.count() > 0:
            self._restore_from_chroma()
            chunks_list = list(self._fallback_chunks.values())

        if not chunks_list:
            self._index_metadata.update({
                "embedding_provider": info.provider,
                "embedding_model": info.model,
                "embedding_dimension": info.dimension,
                "vector_dimension": info.dimension,
                "is_neural": info.is_neural,
                "migrated_at": time.time()
            })
            return True

        new_vectors: Dict[str, List[float]] = {}
        failed = 0

        for ch in chunks_list:
            try:
                vec = await target_provider.get_embedding(ch.text, is_query=False)
                new_vectors[ch.chunk_id] = vec
            except Exception as e:
                logger.error(f"Failed to re-embed chunk {ch.chunk_id} ({ch.filename}): {e}")
                failed += 1

        if failed > 0 and len(new_vectors) == 0:
            logger.error("Index migration failed completely; keeping previous index.")
            return False

        # 3. Update ChromaDB collection if active
        if self.collection is not None:
            if self._chroma_client is not None:
                try:
                    self._chroma_client.delete_collection(self.COLLECTION_NAME)
                except Exception:
                    pass
                self._collection = self._chroma_client.create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )

            ids = [ch.chunk_id for ch in chunks_list if ch.chunk_id in new_vectors]
            vecs = [new_vectors[ch.chunk_id] for ch in chunks_list if ch.chunk_id in new_vectors]
            docs = [ch.text for ch in chunks_list if ch.chunk_id in new_vectors]
            metas = []
            for ch in chunks_list:
                if ch.chunk_id in new_vectors:
                    sub = ch.subject or ch.metadata.get("subject_id") or ch.metadata.get("subject") or "general"
                    m = {
                        "subject_id": str(sub),
                        "subject": str(sub),
                        "document_id": str(ch.document_id),
                        "filename": str(ch.filename),
                        "file_hash": str(ch.metadata.get("file_hash", "") or ch.metadata.get("sha256", "")),
                        "page_number": int(ch.page_number),
                        "heading": str(ch.heading or ""),
                        "chunk_index": int(ch.chunk_index),
                        "doc_category": str(ch.metadata.get("doc_category") or "lecture_notes"),
                        "embedding_model": str(info.model),
                        "embedding_dimension": int(info.dimension)
                    }
                    metas.append(m)
            self.collection.upsert(ids=ids, embeddings=vecs, documents=docs, metadatas=metas)

        # 4. Update vectors and metadata
        self._fallback_vectors = new_vectors
        self._index_metadata.update({
            "embedding_provider": info.provider,
            "embedding_model": info.model,
            "embedding_dimension": info.dimension,
            "vector_dimension": info.dimension,
            "is_neural": info.is_neural,
            "migrated_at": time.time()
        })

        self._save_fallback_index()

        logger.info(f"Successfully migrated index ({len(new_vectors)} chunks re-embedded).")
        return True

    def delete_document_chunks(self, document_id: str) -> int:
        """Remove chunks for a document_id or filename from ChromaDB, fallback store, and registry."""
        count = 0
        if self.collection is not None:
            try:
                self.collection.delete(where={"document_id": document_id})
                count += 1
            except Exception as e:
                logger.debug(f"Chroma delete by document_id failed: {e}")

            try:
                self.collection.delete(where={"filename": document_id})
                count += 1
            except Exception as e:
                logger.debug(f"Chroma delete by filename failed: {e}")

        removed_cids = [cid for cid, ch in self._fallback_chunks.items() if ch.filename == document_id or ch.document_id == document_id]
        for cid in removed_cids:
            self._fallback_chunks.pop(cid, None)
            self._fallback_vectors.pop(cid, None)

        self._document_registry.pop(document_id, None)
        matching_keys = [k for k, v in self._document_registry.items() if v.get("document_id") == document_id]
        for k in matching_keys:
            self._document_registry.pop(k, None)

        if removed_cids:
            self._save_fallback_index()

        return len(removed_cids) if removed_cids else count

    async def add_chunks(self, chunks: List[DocumentChunk], file_sha256: Optional[str] = None) -> int:
        """Index document chunks into ChromaDB collection with automatic deduplication and active embeddings."""
        if not chunks:
            return 0

        provider = await (self.embedding_manager or embedding_manager).get_provider()
        info = provider.get_info()

        mgr = self.embedding_manager or embedding_manager
        active_dim = getattr(mgr, "dimension", info.dimension) or info.dimension
        active_model = getattr(mgr, "model_name", info.model) or info.model

        stored_dim = self._index_metadata.get("vector_dimension") or self._index_metadata.get("embedding_dimension", 0)
        stored_model = self._index_metadata.get("embedding_model", "unknown")

        existing_count = self.collection.count() if self.collection is not None else len(self._fallback_chunks)
        if existing_count > 0 and stored_dim > 0:
            if stored_dim != active_dim:
                raise ModelMigrationError(
                    f"Vector index dimensionality lock error: existing index has dimension {stored_dim}d (model: '{stored_model}'), "
                    f"but active embedding configuration is {active_dim}d (model: '{active_model}'). "
                    "Cannot insert chunks into index with mismatched dimensions. "
                    "Please reindex all documents via `await vector_store.migrate_index()` or restore the original EMBEDDING_MODEL."
                )
            if stored_model not in ("unknown", "") and stored_model != active_model:
                raise ModelMigrationError(
                    f"Vector index model lock error: existing index was built with '{stored_model}' ({stored_dim}d), "
                    f"but active embedding configuration is '{active_model}' ({active_dim}d). "
                    "Cannot insert chunks into index built with a different model. "
                    "Please reindex all documents via `await vector_store.migrate_index()` or restore the original EMBEDDING_MODEL."
                )

        # Update metadata to reflect current provider
        self._index_metadata.update({
            "embedding_provider": info.provider,
            "embedding_model": info.model,
            "embedding_dimension": info.dimension,
            "vector_dimension": info.dimension,
            "is_neural": info.is_neural
        })

        # Deduplicate: remove any prior chunks for these files/docs before re-indexing
        distinct_files = {ch.filename for ch in chunks if ch.filename}
        distinct_doc_ids = {ch.document_id for ch in chunks if ch.document_id}
        for fn in distinct_files:
            self.delete_document_chunks(fn)
        for did in distinct_doc_ids:
            if did not in distinct_files:
                self.delete_document_chunks(did)

        # Register document metadata & checksum
        for ch in chunks:
            fn = ch.filename
            if fn not in self._document_registry:
                self._document_registry[fn] = {
                    "filename": fn,
                    "document_id": ch.document_id,
                    "subject": ch.subject,
                    "sha256": file_sha256 or ch.metadata.get("sha256", ""),
                    "chunk_count": 0,
                    "total_pages": ch.metadata.get("total_pages", ch.page_number),
                    "indexed_at": time.time()
                }
            self._document_registry[fn]["chunk_count"] += 1

        # Generate embeddings in batch
        texts = [ch.text for ch in chunks]
        vectors = await provider.get_embeddings_batch(texts, is_query=False)

        # Insert into ChromaDB collection
        if self.collection is not None:
            ids = [ch.chunk_id for ch in chunks]
            metadatas = []
            for ch in chunks:
                sub = ch.subject or ch.metadata.get("subject_id") or ch.metadata.get("subject") or "general"
                meta: Dict[str, Any] = {
                    "subject_id": str(sub),
                    "subject": str(sub),
                    "document_id": str(ch.document_id),
                    "filename": str(ch.filename),
                    "file_hash": str(file_sha256 or ch.metadata.get("sha256", "") or ""),
                    "page_number": int(ch.page_number),
                    "heading": str(ch.heading or ""),
                    "chunk_index": int(ch.chunk_index),
                    "doc_category": str(ch.metadata.get("doc_category") or "lecture_notes"),
                    "embedding_model": str(info.model),
                    "embedding_dimension": int(info.dimension)
                }
                for k, v in ch.metadata.items():
                    if k not in meta and isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                metadatas.append(meta)

            MAX_BATCH = 500
            for i in range(0, len(chunks), MAX_BATCH):
                b_ids = ids[i : i + MAX_BATCH]
                b_vecs = vectors[i : i + MAX_BATCH]
                b_docs = texts[i : i + MAX_BATCH]
                b_metas = metadatas[i : i + MAX_BATCH]
                self.collection.upsert(
                    ids=b_ids,
                    embeddings=b_vecs,
                    documents=b_docs,
                    metadatas=b_metas
                )

        # Maintain in-memory chunks
        for ch, vec in zip(chunks, vectors):
            self._fallback_chunks[ch.chunk_id] = ch
            self._fallback_vectors[ch.chunk_id] = vec

        self._save_fallback_index()

        return len(chunks)

    def _compute_lexical_score(self, query_tokens: List[str], text: str) -> float:
        """Compute normalized BM25-like lexical term saturation score."""
        if not query_tokens:
            return 0.0
        doc_tokens = [w for w in re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', text.lower()) if w not in STOPWORDS]
        if not doc_tokens:
            return 0.0

        score = 0.0
        doc_len = len(doc_tokens)
        avg_len = 50.0

        for qt in query_tokens:
            tf = doc_tokens.count(qt)
            if tf > 0:
                score += tf / (tf + 1.2 * (0.25 + 0.75 * (doc_len / avg_len)))

        return min(1.0, score / max(1.0, len(query_tokens)))

    async def search(
        self,
        query: Optional[Any] = None,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        subject: Optional[str] = None,
        document_id: Optional[str] = None,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        hybrid: Optional[bool] = None,
        alpha: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for relevant academic chunks using ChromaDB with metadata filtering and hybrid scoring.

        Supports query as string, query_text parameter, query_embedding, or both.
        Supports metadata filters (e.g. {'subject_id': 'CS201'}).
        Supports explicit hybrid flag and alpha blending parameter.
        """
        if query_text is None:
            if isinstance(query, str):
                query_text = query
        elif query is None:
            query = query_text

        if isinstance(query, (list, tuple)) and query_embedding is None:
            query_embedding = list(query)

        k = top_k or getattr(self.settings, "RAG_TOP_K", 4)
        use_hybrid = getattr(self.settings, "RAG_HYBRID_SEARCH", True) if hybrid is None else bool(hybrid)
        alpha_val = getattr(self.settings, "RAG_HYBRID_ALPHA", 0.65) if alpha is None else float(alpha)

        provider = await (self.embedding_manager or embedding_manager).get_provider()
        is_neural = self._index_metadata.get("is_neural", False)

        default_thresh = getattr(self.settings, "RAG_SIMILARITY_THRESHOLD", 0.45)
        if score_threshold is not None:
            if is_neural and score_threshold < default_thresh:
                threshold = default_thresh
            else:
                threshold = score_threshold
        else:
            if is_neural:
                threshold = default_thresh
            elif default_thresh != 0.45:
                threshold = default_thresh
            else:
                threshold = 0.22

        if query_embedding is None:
            if not query_text:
                return []
            query_embedding = await provider.get_embedding(query_text, is_query=True)

        query_tokens = [w for w in re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', query_text.lower()) if w not in STOPWORDS] if query_text else []

        # Authorized ChromaDB retrieval
        if self.collection is not None and self.collection.count() > 0:
            active_filters: Dict[str, Any] = dict(filters) if filters else {}
            if subject and "subject_id" not in active_filters and "subject" not in active_filters:
                active_filters["subject_id"] = subject
            if document_id and "document_id" not in active_filters:
                active_filters["document_id"] = document_id

            where_clause = _build_chroma_where(active_filters)
            n_results = min(max(k * 3, 10), self.collection.count())

            try:
                chroma_res = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_clause,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                logger.warning(f"ChromaDB query failed with where={where_clause}: {e}. Retrying without where.")
                chroma_res = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )

            ids = chroma_res.get("ids", [[]])[0]
            docs = chroma_res.get("documents", [[]])[0]
            metas = chroma_res.get("metadatas", [[]])[0]
            dists = chroma_res.get("distances", [[]])[0]

            candidates: List[Tuple[float, float, float, SearchResult]] = []

            for cid, doc_text, meta, dist in zip(ids, docs, metas, dists):
                meta = meta or {}
                sem_sim = max(0.0, min(1.0, 1.0 - float(dist)))
                lex_sim = self._compute_lexical_score(query_tokens, doc_text) if query_tokens else 0.0

                min_semantic_only = getattr(self.settings, "RAG_MIN_SEMANTIC_ONLY_SCORE", 0.50)
                if lex_sim == 0.0 and is_neural and sem_sim < min_semantic_only:
                    continue

                if use_hybrid and query_tokens:
                    comb_score = alpha_val * sem_sim + (1.0 - alpha_val) * lex_sim
                else:
                    comb_score = sem_sim

                if comb_score >= threshold:
                    sub = meta.get("subject_id") or meta.get("subject")
                    sr = SearchResult(
                        chunk_id=cid,
                        document_id=meta.get("document_id", ""),
                        filename=meta.get("filename", ""),
                        subject=sub,
                        page_number=int(meta.get("page_number", 1)),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        text=doc_text,
                        score=round(comb_score, 3),
                        heading=meta.get("heading") or None,
                        semantic_score=round(sem_sim, 3),
                        lexical_score=round(lex_sim, 3),
                        metadata=meta
                    )
                    candidates.append((comb_score, sem_sim, lex_sim, sr))

            candidates.sort(key=lambda x: x[0], reverse=True)
            return [sr for _, _, _, sr in candidates[:k]]

        # Fallback in-memory search
        candidates_fallback: List[Tuple[float, float, float, DocumentChunk]] = []
        for cid, ch in self._fallback_chunks.items():
            if subject and ch.subject and ch.subject.lower() != subject.lower():
                continue
            if document_id and ch.document_id != document_id:
                continue
            if filters:
                mismatch = False
                for fk, fv in filters.items():
                    if fk in ("subject", "subject_id"):
                        if ch.subject != fv and ch.metadata.get(fk) != fv:
                            mismatch = True
                            break
                    elif ch.metadata.get(fk) != fv and getattr(ch, fk, None) != fv:
                        mismatch = True
                        break
                if mismatch:
                    continue

            ch_vec = self._fallback_vectors.get(cid)
            if not ch_vec or len(ch_vec) != len(query_embedding):
                continue

            sem_sim = max(0.0, sum(q * c for q, c in zip(query_embedding, ch_vec)))
            lex_sim = self._compute_lexical_score(query_tokens, ch.text) if query_tokens else 0.0

            min_semantic_only = getattr(self.settings, "RAG_MIN_SEMANTIC_ONLY_SCORE", 0.50)
            if lex_sim == 0.0 and is_neural and sem_sim < min_semantic_only:
                continue

            if use_hybrid and query_tokens:
                combined_score = alpha_val * sem_sim + (1.0 - alpha_val) * lex_sim
            else:
                combined_score = sem_sim

            if combined_score >= threshold:
                candidates_fallback.append((combined_score, sem_sim, lex_sim, ch))

        candidates_fallback.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResult] = []
        for comb_score, sem_score, lex_score, ch in candidates_fallback[:k]:
            results.append(SearchResult(
                chunk_id=ch.chunk_id,
                document_id=ch.document_id,
                filename=ch.filename,
                subject=ch.subject,
                page_number=ch.page_number,
                chunk_index=ch.chunk_index,
                text=ch.text,
                score=round(comb_score, 3),
                heading=ch.heading,
                semantic_score=round(sem_score, 3),
                lexical_score=round(lex_score, 3),
                metadata=ch.metadata
            ))

        return results

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        """List distinct documents indexed in the vector store with metadata and checksums."""
        docs: List[Dict[str, Any]] = []
        for fn, info in self._document_registry.items():
            docs.append({
                "filename": fn,
                "document_id": info.get("document_id", ""),
                "subject": info.get("subject", ""),
                "sha256": info.get("sha256", ""),
                "total_chunks": info.get("chunk_count", 0),
                "total_pages": info.get("total_pages", 1),
                "indexed_at": info.get("indexed_at", 0)
            })
        return docs

    def get_index_status(self) -> Dict[str, Any]:
        """Return diagnostic status of the vector index."""
        backend_name = "chromadb" if self.collection is not None else "fallback_json"
        total_chunks = self.collection.count() if self.collection is not None else len(self._fallback_chunks)
        return {
            "mode": backend_name,
            "backend": backend_name,
            "status": "ok",
            "collection_name": self.collection.name if self.collection is not None else None,
            "index_version": self._index_metadata.get("index_version", self.INDEX_VERSION),
            "embedding_provider": self._index_metadata.get("embedding_provider", "unknown"),
            "embedding_model": self._index_metadata.get("embedding_model", "unknown"),
            "embedding_dimension": self._index_metadata.get("embedding_dimension", 0),
            "vector_dimension": self._index_metadata.get("vector_dimension", 0),
            "is_neural": self._index_metadata.get("is_neural", False),
            "total_documents": len(self._document_registry),
            "total_chunks": total_chunks,
            "collection_size": total_chunks,
            "updated_at": self._index_metadata.get("updated_at", 0),
            "hybrid_search_enabled": getattr(self.settings, "RAG_HYBRID_SEARCH", True),
            "similarity_threshold": getattr(self.settings, "RAG_SIMILARITY_THRESHOLD", 0.35),
        }

    def close(self) -> None:
        """Safely flush and release vector store resources upon application shutdown."""
        try:
            if hasattr(self, "_save_fallback_index") and self._fallback_file:
                self._save_fallback_index()
            if self._chroma_client is not None:
                if hasattr(self._chroma_client, "_system") and hasattr(self._chroma_client._system, "stop"):
                    self._chroma_client._system.stop()
                self._chroma_client = None
                self._collection = None
                self._chroma_initialized = False
            logger.info("Vector store resources flushed and closed successfully.")
        except Exception as e:
            logger.warning(f"Error during vector store shutdown: {e}")


vector_store = VectorStore(lazy_init=True)
