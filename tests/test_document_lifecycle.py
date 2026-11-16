"""Comprehensive tests for document lifecycle management, catalog sync, duplicate detection, replacement, and failure recovery."""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from app.knowledge.document_parser import document_parser
from app.knowledge.chunker import academic_chunker
from app.knowledge.rag_service import RAGService
from app.main import create_app


@pytest.mark.asyncio
async def test_full_ingestion_and_catalog_sync(test_container, tmp_path):
    """Verify ingesting a valid document synchronizes both VectorStore and the SQLite documents catalog."""
    rag = test_container.rag_service
    db = test_container.db
    store = test_container.vector_store

    doc_file = tmp_path / "distributed_systems.txt"
    doc_file.write_text("Byzantine Fault Tolerance ensures consensus in untrusted distributed networks.")

    res = await rag.ingest_document(doc_file, subject="Distributed Systems")
    assert res.status == "success"
    assert res.is_duplicate is False
    assert res.total_chunks >= 1
    assert res.total_pages == 1
    assert len(res.sha256) == 64

    # Verify VectorStore state
    assert doc_file.name in store._document_registry
    registry_entry = store._document_registry[doc_file.name]
    assert registry_entry["sha256"] == res.sha256
    assert registry_entry["chunk_count"] == res.total_chunks

    # Verify SQLite documents catalog state
    db_doc = await db.get_document(res.document_id)
    assert db_doc is not None
    assert db_doc["filename"] == doc_file.name
    assert db_doc["file_hash"] == res.sha256
    assert db_doc["status"] == "indexed"
    assert db_doc["chunk_count"] == res.total_chunks
    assert db_doc["page_count"] == 1
    assert db_doc["error_message"] is None


@pytest.mark.asyncio
async def test_exact_duplicate_detection_skips_reembedding(test_container, tmp_path):
    """Verify that ingesting identical content skips re-embedding and creates zero duplicate chunks."""
    rag = test_container.rag_service
    store = test_container.vector_store

    doc_file = tmp_path / "algorithms.txt"
    doc_file.write_text("Dijkstra's algorithm finds shortest paths in non-negative weighted graphs.")

    # Ingestion 1
    res1 = await rag.ingest_document(doc_file, subject="Algorithms")
    assert res1.is_duplicate is False
    assert res1.status == "success"
    initial_chunk_count = len(store._fallback_chunks)

    # Ingestion 2: exact same content
    res2 = await rag.ingest_document(doc_file, subject="Algorithms")
    assert res2.is_duplicate is True
    assert res2.status == "already_indexed"
    assert res2.document_id == res1.document_id
    # Total chunks in vector store must be identical (no duplication)
    assert len(store._fallback_chunks) == initial_chunk_count

    # Ingestion 3: force_reindex=True overrides duplicate skip
    res3 = await rag.ingest_document(doc_file, subject="Algorithms", force_reindex=True)
    assert res3.is_duplicate is False
    assert res3.status == "success"
    assert len(store._fallback_chunks) == initial_chunk_count


@pytest.mark.asyncio
async def test_document_content_update_replaces_old_chunks(test_container, tmp_path):
    """Verify that updating a document's content replaces old chunks and updates catalog metadata."""
    rag = test_container.rag_service
    db = test_container.db
    store = test_container.vector_store

    doc_file = tmp_path / "database_notes.txt"

    # Version 1: Relational
    doc_file.write_text("Relational databases use tuple relational calculus and relational algebra queries.")
    res1 = await rag.ingest_document(doc_file, subject="Databases")
    assert res1.status == "success"

    # Verify Version 1 is searchable
    search_v1 = await store.search("relational algebra queries")
    assert len(search_v1) >= 1
    assert "tuple relational calculus" in search_v1[0].text

    # Version 2: Modified on disk with new content
    doc_file.write_text("NoSQL document stores use JSON documents, key-value mappings, and eventual consistency.")
    res2 = await rag.ingest_document(doc_file, subject="Databases")
    assert res2.status == "success"
    assert res2.sha256 != res1.sha256

    # Verify Version 1 text is completely gone
    search_old = await store.search("tuple relational calculus")
    assert len(search_old) == 0 or "tuple relational calculus" not in search_old[0].text

    # Verify Version 2 text is searchable
    search_v2 = await store.search("NoSQL document stores")
    assert len(search_v2) >= 1
    assert "JSON documents" in search_v2[0].text

    # Verify SQLite catalog updated
    catalog_entry = await db.get_document(res2.document_id)
    assert catalog_entry is not None
    assert catalog_entry["file_hash"] == res2.sha256
    assert catalog_entry["file_hash"] != res1.sha256
    assert catalog_entry["status"] == "indexed"


@pytest.mark.asyncio
async def test_safe_deletion_coordination(test_container, tmp_path):
    """Verify safe document deletion coordinates across VectorStore, SQLite, and optional disk removal."""
    rag = test_container.rag_service
    db = test_container.db
    store = test_container.vector_store

    # Create file inside DOCUMENTS_DIR so disk deletion is allowed
    doc_file = test_container.settings.DOCUMENTS_DIR / "networks.txt"
    doc_file.write_text("The sliding window protocol regulates packet transmission flow control in TCP.")

    res = await rag.ingest_document(doc_file, subject="Networking")
    doc_id = res.document_id

    # Confirm present in store and db
    assert doc_file.name in store._document_registry
    assert await db.get_document(doc_id) is not None

    # Delete without physical file removal
    del_res = await rag.delete_document(doc_id, delete_physical_file=False)
    assert del_res["ok"] is True
    assert del_res["chunks_removed"] >= 1
    assert del_res["db_deleted"] is True
    assert del_res["physical_file_deleted"] is False
    assert doc_file.exists()  # File still preserved on disk

    # Verify completely unsearchable
    search_res = await store.search("sliding window protocol")
    assert len(search_res) == 0
    assert await db.get_document(doc_id) is None
    assert doc_file.name not in store._document_registry

    # Re-ingest then delete with delete_physical_file=True
    res_re = await rag.ingest_document(doc_file, subject="Networking")
    del_res_disk = await rag.delete_document(res_re.document_id, delete_physical_file=True)
    assert del_res_disk["physical_file_deleted"] is True
    assert not doc_file.exists()


@pytest.mark.asyncio
async def test_failure_recovery_and_orphan_cleanup(test_container, tmp_path):
    """Verify that failure during chunk indexing marks status='failed', cleans partial chunks, and recovers on retry."""
    rag = test_container.rag_service
    db = test_container.db
    store = test_container.vector_store

    doc_file = test_container.settings.DOCUMENTS_DIR / "operating_systems.txt"
    doc_file.write_text("Deadlock detection algorithms search for cycles in resource allocation graphs.")

    # 1. Simulate failure during add_chunks
    with patch.object(store, "add_chunks", side_effect=RuntimeError("Simulated vector index failure")):
        with pytest.raises(RuntimeError, match="Simulated vector index failure"):
            await rag.ingest_document(doc_file, subject="OS")

    # 2. Verify SQLite catalog recorded failure cleanly
    db_doc = await db.get_document_by_filename(doc_file.name)
    assert db_doc is not None
    assert db_doc["status"] == "failed"
    assert "Simulated vector index failure" in (db_doc["error_message"] or "")

    # 3. Verify zero orphan chunks exist in VectorStore
    assert doc_file.name not in store._document_registry
    orphan_chunks = [ch for ch in store._fallback_chunks.values() if ch.filename == doc_file.name]
    assert len(orphan_chunks) == 0

    # 4. Recover: retry failed documents
    retry_report = await rag.retry_failed_documents()
    assert retry_report["succeeded"] == 1
    assert retry_report["failed"] == 0

    # 5. Verify document is now indexed and searchable
    db_doc_recovered = await db.get_document_by_filename(doc_file.name)
    assert db_doc_recovered["status"] == "indexed"
    assert db_doc_recovered["chunk_count"] >= 1

    search_res = await store.search("resource allocation graphs")
    assert len(search_res) >= 1
    assert "Deadlock detection" in search_res[0].text


@pytest.mark.asyncio
async def test_document_api_delete_and_details_endpoints(async_client, test_container):
    """Verify HTTP endpoints GET /api/documents/{doc_id} and DELETE /api/documents/{doc_id}."""
    rag = test_container.rag_service
    doc_file = test_container.settings.DOCUMENTS_DIR / "api_test.txt"
    doc_file.write_text("REST API contract testing for document lifecycle endpoints.")

    res = await rag.ingest_document(doc_file, subject="API")
    doc_id = res.document_id

    # 1. GET /api/documents/{doc_id}
    get_res = await async_client.get(f"/api/documents/{doc_id}")
    assert get_res.status_code == 200
    doc_data = get_res.json()
    assert doc_data["id"] == doc_id
    assert doc_data["filename"] == "api_test.txt"
    assert doc_data["status"] == "indexed"

    # 2. DELETE /api/documents/{doc_id}
    del_res = await async_client.delete(f"/api/documents/{doc_id}?delete_file=true")
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["ok"] is True
    assert del_data["db_deleted"] is True
    assert del_data["physical_file_deleted"] is True

    # 3. Subsequent GET returns 404
    get_after = await async_client.get(f"/api/documents/{doc_id}")
    assert get_after.status_code == 404
