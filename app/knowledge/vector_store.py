"""Vector Store for academic document chunks supporting neural embeddings, hybrid retrieval, and index migration."""

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
from app.knowledge.embeddings import embedding_manager, BaseEmbeddingProvider, EmbeddingInfo, STOPWORDS

logger = logging.getLogger("university_agent.knowledge.vector_store")


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    subject: Optional[str]
    page_number: int
    text: str
    score: float
    metadata: Dict[str, Any]
    heading: Optional[str] = None
    semantic_score: float = 0.0
    lexical_score: float = 0.0

    def citation(self) -> str:
        """Standard academic citation string with section/heading awareness."""
        subj = f"[{self.subject}] " if self.subject else ""
        head = f" § {self.heading}" if self.heading else ""
        return f"{subj}{self.filename} (Page {self.page_number}{head})"


class VectorStore:
    """Production vector database managing neural document embeddings, hybrid retrieval, and index migration."""

    INDEX_VERSION = 2

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or settings.CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = None
        self.collection = None
        self._init_chroma()

        # In-memory document & vector storage
        self._fallback_chunks: Dict[str, DocumentChunk] = {}
        self._fallback_vectors: Dict[str, List[float]] = {}
        self._document_registry: Dict[str, Dict[str, Any]] = {}
        self._index_metadata: Dict[str, Any] = {
            "index_version": self.INDEX_VERSION,
            "embedding_provider": "unknown",
            "embedding_model": "unknown",
            "embedding_dimension": 0,
            "is_neural": False,
            "created_at": time.time(),
            "updated_at": time.time()
        }

        self._fallback_file = self.persist_dir / "fallback_index.json"
        self._load_fallback_index()

    def _init_chroma(self) -> None:
        """Attempt to initialize ChromaDB PersistentClient if available."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir.resolve()),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="academic_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized successfully.")
        except Exception as e:
            logger.info(f"ChromaDB not available in current environment ({e}); using embedded index.")

    def _backup_index(self) -> Optional[Path]:
        """Create a timestamped backup of the current index file."""
        if not self._fallback_file.exists():
            return None
        try:
            timestamp = int(time.time())
            backup_path = self.persist_dir / f"fallback_index.json.bak.{timestamp}"
            backup_path.write_bytes(self._fallback_file.read_bytes())
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
                f"Loaded vector index: {len(self._fallback_chunks)} chunks, "
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

            # Atomic write via temporary file
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
        provider = await embedding_manager.get_provider()
        return await provider.get_embedding(text, is_query=is_query)

    async def ensure_index_compatibility(self) -> bool:
        """Check if stored vectors are compatible with the active embedding provider.

        If incompatible (e.g. switching from 256d hash to 768d nomic-embed-text),
        migrates by re-embedding all stored chunks.
        """
        provider = await embedding_manager.get_provider()
        info = provider.get_info()

        stored_dim = self._index_metadata.get("embedding_dimension", 0)
        stored_model = self._index_metadata.get("embedding_model", "unknown")

        if not self._fallback_chunks:
            # Empty index, just update metadata
            self._index_metadata.update({
                "embedding_provider": info.provider,
                "embedding_model": info.model,
                "embedding_dimension": info.dimension,
                "is_neural": info.is_neural
            })
            self._save_fallback_index()
            return True

        if stored_dim != info.dimension or stored_model != info.model:
            logger.warning(
                f"Index compatibility mismatch: stored index uses '{stored_model}' ({stored_dim}d), "
                f"but active provider is '{info.model}' ({info.dimension}d). Initiating automatic safe migration..."
            )
            return await self.migrate_index(provider)

        return True

    async def migrate_index(self, target_provider: Optional[BaseEmbeddingProvider] = None) -> bool:
        """Safely re-embed all indexed chunks using the target embedding provider."""
        if target_provider is None:
            target_provider = await embedding_manager.get_provider()

        info = target_provider.get_info()
        logger.info(f"Starting index migration to provider '{info.provider}' (model: '{info.model}', dim: {info.dimension})...")

        # 1. Backup existing index before migration
        self._backup_index()

        # 2. Re-embed all chunks with the target provider
        new_vectors: Dict[str, List[float]] = {}
        chunks_list = list(self._fallback_chunks.values())
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

        # 3. Update vectors and metadata
        self._fallback_vectors = new_vectors
        self._index_metadata.update({
            "embedding_provider": info.provider,
            "embedding_model": info.model,
            "embedding_dimension": info.dimension,
            "is_neural": info.is_neural,
            "migrated_at": time.time()
        })

        self._save_fallback_index()
        logger.info(f"Successfully migrated index ({len(new_vectors)} chunks re-embedded).")
        return True

    def delete_document_chunks(self, filename: str) -> int:
        """Remove existing chunks for a filename to prevent duplicates and enable clean re-ingestion."""
        removed_cids = [cid for cid, ch in self._fallback_chunks.items() if ch.filename == filename]
        for cid in removed_cids:
            self._fallback_chunks.pop(cid, None)
            self._fallback_vectors.pop(cid, None)

        self._document_registry.pop(filename, None)

        if self.collection is not None and removed_cids:
            try:
                self.collection.delete(where={"filename": filename})
            except Exception as e:
                logger.debug(f"ChromaDB delete failed ({e})")

        if removed_cids:
            self._save_fallback_index()
        return len(removed_cids)

    async def add_chunks(self, chunks: List[DocumentChunk], file_sha256: Optional[str] = None) -> int:
        """Index document chunks into vector database with automatic deduplication and active embeddings."""
        if not chunks:
            return 0

        # Ensure index compatibility before adding
        await self.ensure_index_compatibility()

        provider = await embedding_manager.get_provider()
        info = provider.get_info()

        # Update metadata to reflect current provider
        self._index_metadata.update({
            "embedding_provider": info.provider,
            "embedding_model": info.model,
            "embedding_dimension": info.dimension,
            "is_neural": info.is_neural
        })

        # Deduplicate: remove any prior chunks for these files before re-indexing
        distinct_files = {ch.filename for ch in chunks}
        for fn in distinct_files:
            self.delete_document_chunks(fn)

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

        # Add chunks and generate embeddings
        for ch in chunks:
            self._fallback_chunks[ch.chunk_id] = ch
            vec = await provider.get_embedding(ch.text, is_query=False)
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
        avg_len = 50.0  # reference chunk token length

        for qt in query_tokens:
            tf = doc_tokens.count(qt)
            if tf > 0:
                # Saturation curve
                score += tf / (tf + 1.2 * (0.25 + 0.75 * (doc_len / avg_len)))

        return min(1.0, score / max(1.0, len(query_tokens)))

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        subject: Optional[str] = None,
        document_id: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for relevant academic chunks using hybrid dense semantic and sparse lexical scoring.

        SCORING SEMANTICS:
        - `sem_sim` (Semantic Score): Cosine similarity in range [0.0, 1.0] between L2-normalized query
          and chunk embeddings (sum(q_i * c_i)). Higher indicates closer semantic alignment.
          NOTE: This is NOT a distance metric (e.g., cosine distance is 1.0 - sem_sim).
        - `lex_sim` (Lexical Score): Normalized BM25-like token saturation score in range [0.0, 1.0].
        - `combined_score` (Composite Score): Convex combination (alpha * sem_sim + (1 - alpha) * lex_sim).
          Higher indicates higher overall relevance.

        RETRIEVAL DEFECT & ANISOTROPY MITIGATION:
        Dense neural text embeddings (including `nomic-embed-text`) exhibit anisotropy, where unrelated
        English sentences share a nonzero background similarity floor of ~0.30 - 0.42 (e.g. 0.386).
        To prevent false-positive retrieval on unrelated queries:
        1. When `lex_sim == 0.0` (zero lexical overlap), candidates must satisfy a strict semantic
           threshold (RAG_MIN_SEMANTIC_ONLY_SCORE = 0.50) to filter out baseline noise.
        2. When `is_neural=True`, any explicit `score_threshold` below the neural noise floor
           (settings.RAG_SIMILARITY_THRESHOLD = 0.45) is calibrated to the neural threshold, preventing
           legacy hash-based thresholds (such as 0.22) from admitting background noise.
        """
        k = top_k or getattr(settings, "RAG_TOP_K", 4)
        use_hybrid = getattr(settings, "RAG_HYBRID_SEARCH", True)
        alpha = getattr(settings, "RAG_HYBRID_ALPHA", 0.65)

        # Ensure vector compatibility
        await self.ensure_index_compatibility()

        provider = await embedding_manager.get_provider()
        is_neural = self._index_metadata.get("is_neural", False)

        default_thresh = getattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.45)
        if score_threshold is not None:
            if is_neural and score_threshold < default_thresh:
                # Calibrate legacy or uncalibrated threshold to neural minimum
                threshold = default_thresh
            else:
                threshold = score_threshold
        else:
            threshold = default_thresh if is_neural else min(default_thresh, 0.22)

        query_vec = await provider.get_embedding(query, is_query=True)

        query_tokens = [w for w in re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', query.lower()) if w not in STOPWORDS]

        candidates: List[Tuple[float, float, float, DocumentChunk]] = []

        for cid, ch in self._fallback_chunks.items():
            if subject and ch.subject and ch.subject.lower() != subject.lower():
                continue
            if document_id and ch.document_id != document_id:
                continue

            ch_vec = self._fallback_vectors.get(cid)
            if not ch_vec or len(ch_vec) != len(query_vec):
                continue

            # 1. Dense Semantic Cosine Similarity: <q, c> for L2-normalized vectors
            sem_sim = max(0.0, sum(q * c for q, c in zip(query_vec, ch_vec)))

            # 2. Sparse Lexical Token Score
            lex_sim = self._compute_lexical_score(query_tokens, ch.text)

            # Cosine similarity has a nonzero unrelated-text baseline in neural models (~0.30 - 0.42).
            # When no query token overlaps the chunk, require stronger semantic evidence.
            min_semantic_only = getattr(settings, "RAG_MIN_SEMANTIC_ONLY_SCORE", 0.50)
            if lex_sim == 0.0 and is_neural and sem_sim < min_semantic_only:
                continue

            # 3. Hybrid Score Combination
            if use_hybrid:
                combined_score = alpha * sem_sim + (1.0 - alpha) * lex_sim
            else:
                combined_score = sem_sim

            if combined_score >= threshold:
                candidates.append((combined_score, sem_sim, lex_sim, ch))

        candidates.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResult] = []
        for comb_score, sem_score, lex_score, ch in candidates[:k]:
            results.append(SearchResult(
                chunk_id=ch.chunk_id,
                document_id=ch.document_id,
                filename=ch.filename,
                subject=ch.subject,
                page_number=ch.page_number,
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
        return {
            "index_version": self._index_metadata.get("index_version", self.INDEX_VERSION),
            "embedding_provider": self._index_metadata.get("embedding_provider", "unknown"),
            "embedding_model": self._index_metadata.get("embedding_model", "unknown"),
            "embedding_dimension": self._index_metadata.get("embedding_dimension", 0),
            "is_neural": self._index_metadata.get("is_neural", False),
            "total_documents": len(self._document_registry),
            "total_chunks": len(self._fallback_chunks),
            "updated_at": self._index_metadata.get("updated_at", 0),
            "hybrid_search_enabled": getattr(settings, "RAG_HYBRID_SEARCH", True),
            "similarity_threshold": getattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.35),
            "min_semantic_only_score": getattr(settings, "RAG_MIN_SEMANTIC_ONLY_SCORE", 0.50)
        }


vector_store = VectorStore()
