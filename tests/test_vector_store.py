"""Tests for ChromaDB PersistentClient integration, metadata filtering, and collection persistence."""

import pytest
from unittest.mock import AsyncMock, patch

from app.config import Settings
from app.knowledge.chunker import DocumentChunk
from app.knowledge.vector_store import VectorStore, SearchResult
from app.knowledge.embeddings import DeterministicHashEmbeddingProvider, EmbeddingManager


@pytest.mark.asyncio
async def test_chromadb_initialization_and_collection_creation(tmp_path):
    """Verify ChromaDB PersistentClient initializes cleanly and creates the university_agent_chunks collection."""
    test_settings = Settings(APP_ENV="testing", CHROMA_DIR=tmp_path / "chroma")
    mgr = EmbeddingManager(settings_obj=test_settings)
    store = VectorStore(persist_dir=tmp_path / "chroma", settings_obj=test_settings, embedding_mgr=mgr)

    assert store.chroma_client is not None
    assert store.collection is not None
    assert store.collection.name == "university_agent_chunks"
    assert store.get_index_status()["backend"] == "chromadb"
    assert store.get_index_status()["collection_name"] == "university_agent_chunks"


@pytest.mark.asyncio
async def test_chromadb_ingestion_with_full_metadata(tmp_path):
    """Verify chunks are stored in ChromaDB with full metadata payload."""
    test_settings = Settings(APP_ENV="testing", CHROMA_DIR=tmp_path / "chroma")
    mgr = EmbeddingManager(settings_obj=test_settings)
    hash_prov = DeterministicHashEmbeddingProvider(dimension=256)
    store = VectorStore(persist_dir=tmp_path / "chroma", settings_obj=test_settings, embedding_mgr=mgr)

    chunks = [
        DocumentChunk(
            chunk_id="chunk-os-1",
            document_id="doc-os-101",
            filename="os_paging.pdf",
            subject="Operating Systems",
            page_number=12,
            chunk_index=3,
            text="Virtual memory and page table entries translate virtual addresses to physical frames.",
            heading="Page Table Translation",
            metadata={"doc_category": "lecture_notes", "custom_flag": True}
        )
    ]

    with patch.object(mgr, "get_provider", AsyncMock(return_value=hash_prov)):
        inserted = await store.add_chunks(chunks, file_sha256="abc123def456")

    assert inserted == 1
    assert store.collection.count() == 1

    # Verify directly from ChromaDB collection
    data = store.collection.get(ids=["chunk-os-1"], include=["metadatas", "documents"])
    assert len(data["ids"]) == 1
    assert data["documents"][0] == chunks[0].text

    meta = data["metadatas"][0]
    assert meta["subject_id"] == "Operating Systems"
    assert meta["document_id"] == "doc-os-101"
    assert meta["filename"] == "os_paging.pdf"
    assert meta["file_hash"] == "abc123def456"
    assert meta["page_number"] == 12
    assert meta["heading"] == "Page Table Translation"
    assert meta["chunk_index"] == 3
    assert meta["doc_category"] == "lecture_notes"


@pytest.mark.asyncio
async def test_chromadb_metadata_filtering_restricts_results(tmp_path):
    """Verify searching with subject_id filter strictly restricts results to matching subjects, rejecting closer vectors from other subjects."""
    test_settings = Settings(APP_ENV="testing", CHROMA_DIR=tmp_path / "chroma")
    mgr = EmbeddingManager(settings_obj=test_settings)
    store = VectorStore(persist_dir=tmp_path / "chroma", settings_obj=test_settings, embedding_mgr=mgr)

    # We mock get_provider to return fixed orthogonal vectors:
    # vector_math aligns with query [1.0, 0.0]
    # vector_cs aligns with query [0.0, 1.0]
    query_vec = [1.0, 0.0]
    chunk_math = DocumentChunk(
        chunk_id="chunk-math",
        document_id="doc-math-1",
        filename="calculus.txt",
        subject="Mathematics",
        page_number=1,
        chunk_index=0,
        text="Eigenvalues and eigenvectors in linear algebra.",
        metadata={"subject_id": "Mathematics"}
    )
    chunk_cs = DocumentChunk(
        chunk_id="chunk-cs",
        document_id="doc-cs-1",
        filename="algorithms.txt",
        subject="Computer Science",
        page_number=1,
        chunk_index=0,
        text="Graph algorithms and Dijkstra shortest path.",
        metadata={"subject_id": "Computer Science"}
    )

    class MockProvider:
        def get_info(self):
            from app.knowledge.embeddings import EmbeddingInfo
            return EmbeddingInfo(provider="mock", model="mock-model", dimension=2, is_neural=False)

        async def get_embeddings_batch(self, texts, is_query=False):
            return [[1.0, 0.0] if "linear" in t else [0.0, 1.0] for t in texts]

        async def get_embedding(self, text, is_query=False):
            return [1.0, 0.0]

    mock_prov = MockProvider()
    with patch.object(mgr, "get_provider", AsyncMock(return_value=mock_prov)):
        await store.add_chunks([chunk_math, chunk_cs])

    assert store.collection.count() == 2

    with patch.object(mgr, "get_provider", AsyncMock(return_value=mock_prov)):
        # 1. Search with query [1.0, 0.0] without filters: chunk-math has score ~1.0
        all_results = await store.search(query_embedding=query_vec, top_k=2)
        assert len(all_results) >= 1
        assert all_results[0].chunk_id == "chunk-math"

        # 2. Search with filter: subject_id = 'Computer Science'
        # Even though chunk-math is closer, it MUST be excluded by the filter
        cs_results = await store.search(
            query_embedding=query_vec,
            top_k=2,
            filters={"subject_id": "Computer Science"},
            score_threshold=0.0
        )
        assert len(cs_results) == 1
        assert cs_results[0].chunk_id == "chunk-cs"
        assert cs_results[0].subject == "Computer Science"

        # 3. Search with non-matching filter returns empty
        empty_results = await store.search(
            query_embedding=query_vec,
            top_k=2,
            filters={"subject_id": "Physics"}
        )
        assert len(empty_results) == 0


@pytest.mark.asyncio
async def test_chromadb_delete_document_chunks_purges_only_target(tmp_path):
    """Verify delete_document_chunks removes exactly the chunks matching the document_id and leaves other documents intact."""
    test_settings = Settings(APP_ENV="testing", CHROMA_DIR=tmp_path / "chroma")
    mgr = EmbeddingManager(settings_obj=test_settings)
    hash_prov = DeterministicHashEmbeddingProvider(dimension=256)
    store = VectorStore(persist_dir=tmp_path / "chroma", settings_obj=test_settings, embedding_mgr=mgr)

    chunks = [
        DocumentChunk(
            chunk_id="chunk-docA-1",
            document_id="doc-A",
            filename="docA.txt",
            subject="CS",
            page_number=1,
            chunk_index=0,
            text="First chunk of doc A."
        ),
        DocumentChunk(
            chunk_id="chunk-docA-2",
            document_id="doc-A",
            filename="docA.txt",
            subject="CS",
            page_number=2,
            chunk_index=1,
            text="Second chunk of doc A."
        ),
        DocumentChunk(
            chunk_id="chunk-docB-1",
            document_id="doc-B",
            filename="docB.txt",
            subject="Math",
            page_number=1,
            chunk_index=0,
            text="Chunk of doc B."
        ),
    ]

    with patch.object(mgr, "get_provider", AsyncMock(return_value=hash_prov)):
        await store.add_chunks(chunks)

    assert store.collection.count() == 3

    # Delete doc-A by document_id
    store.delete_document_chunks("doc-A")

    assert store.collection.count() == 1
    remaining = store.collection.get()
    assert remaining["ids"] == ["chunk-docB-1"]

    # Delete doc-B by filename
    store.delete_document_chunks("docB.txt")
    assert store.collection.count() == 0


@pytest.mark.asyncio
async def test_chromadb_persistence_reload_across_restart(tmp_path):
    """Verify that a second VectorStore instance loading from the same directory restores chunks from ChromaDB."""
    chroma_dir = tmp_path / "chroma_persist"
    test_settings = Settings(APP_ENV="testing", CHROMA_DIR=chroma_dir)
    mgr = EmbeddingManager(settings_obj=test_settings)
    hash_prov = DeterministicHashEmbeddingProvider(dimension=256)

    # Instance 1: Add chunk
    store1 = VectorStore(persist_dir=chroma_dir, settings_obj=test_settings, embedding_mgr=mgr)
    chunk = DocumentChunk(
        chunk_id="persist-c1",
        document_id="persist-doc-1",
        filename="persistent_notes.txt",
        subject="Security",
        page_number=1,
        chunk_index=0,
        text="Access control matrices enforce authorization policies across security domains."
    )
    with patch.object(mgr, "get_provider", AsyncMock(return_value=hash_prov)):
        await store1.add_chunks([chunk])

    assert store1.collection.count() == 1

    # Instance 2: Reload from same path (simulating restart)
    store2 = VectorStore(persist_dir=chroma_dir, settings_obj=test_settings, embedding_mgr=mgr)
    assert store2.collection.count() == 1
    assert "persist-c1" in store2._fallback_chunks
    assert "persistent_notes.txt" in store2._document_registry

    with patch.object(mgr, "get_provider", AsyncMock(return_value=hash_prov)):
        res = await store2.search(query="authorization security policies", top_k=1)
        assert len(res) == 1
        assert res[0].chunk_id == "persist-c1"
        assert isinstance(res[0], DocumentChunk)
        assert isinstance(res[0], SearchResult)


@pytest.mark.asyncio
async def test_hybrid_scoring_elevates_exact_keyword_matches_over_loose_semantic(tmp_path):
    """Verify that hybrid search with alpha blending elevates a chunk with exact acronym/keyword matches over a chunk with higher semantic score but zero lexical match."""
    test_settings = Settings(
        APP_ENV="testing",
        CHROMA_DIR=tmp_path / "chroma",
        RAG_HYBRID_SEARCH=True,
        RAG_HYBRID_ALPHA=0.65
    )
    mgr = EmbeddingManager(settings_obj=test_settings)
    store = VectorStore(persist_dir=tmp_path / "chroma", settings_obj=test_settings, embedding_mgr=mgr)

    # Chunk 1: High semantic relevance in mock embeddings, but zero mention of "RAID" acronym
    chunk1 = DocumentChunk(
        chunk_id="chunk-generic-storage",
        document_id="doc-storage-1",
        filename="storage_systems.txt",
        subject="Computer Architecture",
        page_number=5,
        chunk_index=0,
        text="Persistent non-volatile hardware drives and magnetic disks store block-level digital records."
    )
    # Chunk 2: Moderate semantic relevance, but contains exact acronym "RAID" and keyword "architecture"
    chunk2 = DocumentChunk(
        chunk_id="chunk-raid-acronym",
        document_id="doc-raid-1",
        filename="raid_arrays.txt",
        subject="Computer Architecture",
        page_number=2,
        chunk_index=0,
        text="RAID architecture organizes multiple physical drives into redundant arrays with striping."
    )

    class MockHybridProvider:
        def get_info(self):
            from app.knowledge.embeddings import EmbeddingInfo
            return EmbeddingInfo(provider="mock", model="mock-neural", dimension=2, is_neural=False)

        async def get_embeddings_batch(self, texts, is_query=False):
            res = []
            for t in texts:
                if "RAID" in t:
                    res.append([0.60, 0.80])  # Cosine similarity with query [1.0, 0.0] = 0.60
                else:
                    res.append([0.75, 0.6614])  # Cosine similarity with query [1.0, 0.0] = 0.75
            return res

        async def get_embedding(self, text, is_query=False):
            return [1.0, 0.0]

    mock_prov = MockHybridProvider()
    with patch.object(mgr, "get_provider", AsyncMock(return_value=mock_prov)):
        await store.add_chunks([chunk1, chunk2])

    query = "RAID architecture"

    with patch.object(mgr, "get_provider", AsyncMock(return_value=mock_prov)):
        # 1. Pure semantic search (hybrid=False)
        results_no_hybrid = await store.search(query=query, top_k=2, hybrid=False)
        assert len(results_no_hybrid) == 2
        # Without hybrid, higher semantic similarity (Chunk 1: 0.75) beats Chunk 2 (0.60)
        assert results_no_hybrid[0].chunk_id == "chunk-generic-storage"
        assert results_no_hybrid[0].semantic_score == 0.75
        assert results_no_hybrid[1].chunk_id == "chunk-raid-acronym"

        # 2. Hybrid search (hybrid=True, alpha=0.65)
        results_hybrid = await store.search(query=query, top_k=2, hybrid=True)
        assert len(results_hybrid) == 2
        # With hybrid, exact acronym "RAID" gives chunk2 a strong lexical score (>0),
        # boosting its combined final_score above chunk1!
        assert results_hybrid[0].chunk_id == "chunk-raid-acronym"
        assert results_hybrid[0].lexical_score > results_hybrid[1].lexical_score
        assert results_hybrid[0].score > results_hybrid[1].score
        assert results_hybrid[0].final_score == results_hybrid[0].score


@pytest.mark.asyncio
async def test_vector_store_strict_threshold_filtering(tmp_path):
    """Verify that chunks with similarity below threshold are strictly dropped."""
    test_settings = Settings(APP_ENV="testing", CHROMA_DIR=tmp_path / "chroma")
    mgr = EmbeddingManager(settings_obj=test_settings)
    store = VectorStore(persist_dir=tmp_path / "chroma", settings_obj=test_settings, embedding_mgr=mgr)

    chunk = DocumentChunk(
        chunk_id="chunk-thresh-1",
        document_id="doc-1",
        filename="notes.txt",
        subject="CS",
        page_number=1,
        chunk_index=0,
        text="Computer networks use packet switching to route datagrams."
    )

    class MockThreshProvider:
        def get_info(self):
            from app.knowledge.embeddings import EmbeddingInfo
            return EmbeddingInfo(provider="mock", model="mock-neural", dimension=2, is_neural=True)

        async def get_embeddings_batch(self, texts, is_query=False):
            return [[0.60, 0.80] for _ in texts]

        async def get_embedding(self, text, is_query=False):
            return [1.0, 0.0]

    mock_prov = MockThreshProvider()
    with patch.object(mgr, "get_provider", AsyncMock(return_value=mock_prov)):
        await store.add_chunks([chunk])

    with patch.object(mgr, "get_provider", AsyncMock(return_value=mock_prov)):
        # 1. Search with threshold higher than candidate's combined score (0.85) -> dropped
        res_high = await store.search(query="packet switching routing", top_k=2, score_threshold=0.85)
        assert len(res_high) == 0

        # 2. Search with lower threshold (0.40) -> retained
        res_low = await store.search(query="packet switching routing", top_k=2, score_threshold=0.40)
        assert len(res_low) == 1
        assert res_low[0].chunk_id == "chunk-thresh-1"

