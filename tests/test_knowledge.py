"""Unit tests for document parser, chunker, vector store, and RAG retrieval."""

from pathlib import Path
import pytest

from app.knowledge.document_parser import DocumentParser
from app.knowledge.chunker import AcademicChunker
from app.knowledge.vector_store import VectorStore
from app.knowledge.rag_service import RAGService


@pytest.fixture
def temp_knowledge_dir(tmp_path):
    p = tmp_path / "knowledge_test"
    p.mkdir()
    return p


def test_txt_parsing(temp_knowledge_dir):
    """Test text document parsing and page estimation."""
    sample_txt = temp_knowledge_dir / "physics.txt"
    sample_txt.write_text("Newton's First Law: An object remains at rest unless acted upon by a net force.")

    parser = DocumentParser()
    parsed = parser.parse_document(sample_txt, subject="Physics")

    assert parsed.filename == "physics.txt"
    assert parsed.subject == "Physics"
    assert len(parsed.pages) == 1
    assert "Newton's First Law" in parsed.pages[0].text


def test_academic_chunker():
    """Test intelligent overlapping chunking."""
    chunker = AcademicChunker(chunk_size=100, chunk_overlap=20)
    text = (
        "Quantum mechanics is a fundamental theory in physics. "
        "It describes the physical properties of nature at the scale of atoms. "
        "Unlike classical physics, energy and momentum are often restricted to discrete values."
    )
    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) > 0


@pytest.mark.asyncio
async def test_vector_store_indexing_and_search(temp_knowledge_dir):
    """Test adding chunks and retrieving with semantic similarity and metadata filtering."""
    store = VectorStore(persist_dir=temp_knowledge_dir)
    parser = DocumentParser()
    chunker = AcademicChunker(chunk_size=200, chunk_overlap=30)

    # Ingest document 1: Operating Systems
    os_file = temp_knowledge_dir / "os_notes.txt"
    os_file.write_text("A deadlock is a situation where a set of processes are blocked because each holds a resource.")
    parsed_os = parser.parse_document(os_file, subject="Computer Science")
    chunks_os = chunker.chunk_document(parsed_os)
    await store.add_chunks(chunks_os)

    # Ingest document 2: Chemistry
    chem_file = temp_knowledge_dir / "chem_notes.txt"
    chem_file.write_text("An exothermic reaction releases energy through light or heat into its surrounding environment.")
    parsed_chem = parser.parse_document(chem_file, subject="Chemistry")
    chunks_chem = chunker.chunk_document(parsed_chem)
    await store.add_chunks(chunks_chem)

    # Search for deadlock
    results = await store.search(query="deadlock processes blocked", top_k=2)
    assert len(results) >= 1
    assert "deadlock" in results[0].text.lower()
    assert results[0].filename == "os_notes.txt"
    assert "Page 1" in results[0].citation()

    # Search with subject filter
    chem_results = await store.search(query="energy reaction", subject="Chemistry", top_k=2)
    assert len(chem_results) >= 1
    assert chem_results[0].filename == "chem_notes.txt"


@pytest.mark.asyncio
async def test_reingestion_deduplication(temp_knowledge_dir):
    """Verify that re-ingesting an updated file replaces existing chunks rather than duplicating them."""
    store = VectorStore(persist_dir=temp_knowledge_dir)
    parser = DocumentParser()
    chunker = AcademicChunker(chunk_size=200, chunk_overlap=30)

    doc_file = temp_knowledge_dir / "syllabus.txt"
    doc_file.write_text("Syllabus v1: Course overview and initial grading policy.")
    parsed1 = parser.parse_document(doc_file, subject="CS")
    chunks1 = chunker.chunk_document(parsed1)
    await store.add_chunks(chunks1)

    initial_count = len(store._fallback_chunks)
    assert initial_count >= 1

    # Re-ingest updated version with same filename
    doc_file.write_text("Syllabus v2: Updated final exam weighting and schedule.")
    parsed2 = parser.parse_document(doc_file, subject="CS")
    chunks2 = chunker.chunk_document(parsed2)
    await store.add_chunks(chunks2)

    after_count = len(store._fallback_chunks)
    # Total chunk count should not have multiplied
    assert after_count == initial_count

    # Search should return updated content
    res = await store.search("final exam weighting", top_k=1)
    assert len(res) >= 1
    assert "Syllabus v2" in res[0].text


@pytest.mark.asyncio
async def test_irrelevant_query_threshold_rejection(temp_knowledge_dir):
    """Verify that queries with no relevance to indexed documents return zero citations."""
    store = VectorStore(persist_dir=temp_knowledge_dir)
    parser = DocumentParser()
    chunker = AcademicChunker(chunk_size=200, chunk_overlap=30)

    doc_file = temp_knowledge_dir / "botany.txt"
    doc_file.write_text("Photosynthesis in vascular plants converts carbon dioxide into glucose.")
    parsed = parser.parse_document(doc_file, subject="Biology")
    await store.add_chunks(chunker.chunk_document(parsed))

    # Query with completely unrelated terms
    unrelated_res = await store.search("quantum cryptography and blockchain hashing", score_threshold=0.22)
    assert len(unrelated_res) == 0

