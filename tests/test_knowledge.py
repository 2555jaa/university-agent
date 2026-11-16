"""Unit tests for document parser, chunker, vector store, and RAG retrieval."""

from pathlib import Path
import pytest

from app.knowledge.document_parser import DocumentParser, ParsedDocument, DocumentPage
from app.knowledge.chunker import AcademicChunker, classify_academic_document, detect_course_code
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


def test_academic_document_classification_and_course_code():
    """Verify classification into 5 academic categories and course code detection."""
    chunker = AcademicChunker()

    # 1. Syllabus
    syl_text = """CS201: Data Structures Syllabus
Instructor: Dr. Alice
Office Hours: MWF 2-4pm
Course Description: Advanced algorithms and data structures.
Grading Policy: Midterm 30%, Final Project 40%, Homework 30%
Weekly Schedule:
Week 1: Algorithmic Complexity
Week 2: Self-Balancing Trees
"""
    assert chunker.classify_document(ParsedDocument(
        document_id="d1", filename="cs201_syllabus.pdf", file_type="pdf", subject="CS",
        pages=[DocumentPage(page_number=1, text=syl_text)], title="CS201 Syllabus"
    )) == "syllabus"
    assert chunker.detect_course_code(syl_text) == "CS201"

    # 2. Exam or Assignment
    exam_text = """MT 204 Midterm Examination - Fall Semester
Time Allowed: 3 Hours
Total Marks: 100 Points
Instructions to Candidates: Answer all questions.
Question 1 (25 Marks): Compute the eigenvalues of the following matrix.
Question 2 (25 Marks): Solve the system of differential equations.
"""
    assert chunker.classify_document(ParsedDocument(
        document_id="d2", filename="mt204_midterm.pdf", file_type="pdf", subject="Math",
        pages=[DocumentPage(page_number=1, text=exam_text)], title="MT 204 Exam"
    )) == "exam_or_assignment"
    assert chunker.detect_course_code(exam_text) == "MT 204"

    # 3. Reference or Formula Sheet
    ref_text = """Physics Constants and Formula Sheet
Speed of light c = 2.998 x 10^8 m/s
Planck constant h = 6.626 x 10^-34 J*s
Universal gravitational constant G = 6.674 x 10^-11 N*m^2/kg^2
Summary of Equations:
E = mc^2, F = G*m1*m2/r^2
"""
    assert chunker.classify_document(ParsedDocument(
        document_id="d3", filename="physics_formula_sheet.pdf", file_type="pdf", subject="Physics",
        pages=[DocumentPage(page_number=1, text=ref_text)], title="Physics Formula Sheet"
    )) == "reference_or_formula_sheet"

    # 4. Research Paper
    paper_text = """Scalable Byzantine Agreement Protocols
Abstract
Byzantine agreement is a foundational primitive for fault-tolerant state machine replication.
In this paper, we propose an asynchronous consensus protocol.
1. Introduction
Decentralized systems require consensus under partial synchrony...
2. Methodology
We model adversarial behavior using stochastic processes...
References
[1] Castro and Liskov. Practical Byzantine Fault Tolerance. OSDI 1999.
"""
    assert chunker.classify_document(ParsedDocument(
        document_id="d4", filename="byzantine_consensus.pdf", file_type="pdf", subject="CS",
        pages=[DocumentPage(page_number=1, text=paper_text)], title="Scalable Byzantine Agreement"
    )) == "research_paper"

    # 5. Lecture Notes with AI-301 course code
    lec_text = """AI-301 Lecture 4: Heuristic Search
Chapter 3: Informed Search and Exploration
Definition 3.1: An admissible heuristic never overestimates the true cost to reach the goal.
Theorem 3.2: A* search using an admissible heuristic is tree-optimal.
Proof: By induction on the path cost of expanded frontier nodes.
"""
    assert chunker.classify_document(ParsedDocument(
        document_id="d5", filename="ai301_lecture4.pdf", file_type="pdf", subject="AI",
        pages=[DocumentPage(page_number=1, text=lec_text)], title="AI-301 Lecture 4"
    )) == "lecture_notes"
    assert chunker.detect_course_code(lec_text) == "AI-301"


def test_structure_aware_chunking_tables_and_theorems():
    """Verify that structure-aware chunking keeps tables, theorems, and page boundaries intact."""
    chunker = AcademicChunker(chunk_size=400, chunk_overlap=0)

    # Page 1 has an introduction and a Markdown table
    page1_text = """CS201 Data Structures
Overview of course modules.

| Week | Module | Assessment |
| --- | --- | --- |
| 1 | Asymptotic Notation | Problem Set 1 |
| 2 | Linked Lists & Stacks | Quiz 1 |
| 3 | Binary Search Trees | Problem Set 2 |
"""

    # Page 2 has formal academic blocks
    page2_text = """Definition 2.1: A binary search tree is a binary tree where left child < node <= right child.

Theorem 2.2: Searching in a balanced binary search tree requires O(log n) comparisons.

Proof: The maximum depth of a balanced binary tree with n nodes is bounded by ceil(log2(n + 1)).
"""

    doc = ParsedDocument(
        document_id="doc-struct-1",
        filename="cs201_syllabus.pdf",
        file_type="pdf",
        subject="Computer Science",
        pages=[
            DocumentPage(page_number=1, text=page1_text, heading="1. Course Overview"),
            DocumentPage(page_number=2, text=page2_text, heading="2. Binary Search Trees"),
        ],
        title="CS201 Data Structures Syllabus",
        file_hash="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2

    # Verify Page 1 chunks
    p1_chunks = [ch for ch in chunks if ch.page_number == 1]
    assert len(p1_chunks) >= 1
    for ch in p1_chunks:
        assert ch.page_number == 1
        assert "Course Overview" in ch.heading

    # Verify the table is intact
    table_chunk = next(ch for ch in chunks if "| Week | Module | Assessment |" in ch.text)
    assert "| 1 | Asymptotic Notation | Problem Set 1 |" in table_chunk.text
    assert "| 3 | Binary Search Trees | Problem Set 2 |" in table_chunk.text
    assert table_chunk.metadata["has_tables"] is True
    assert table_chunk.metadata["doc_category"] == "syllabus"
    assert table_chunk.metadata["detected_course_code"] == "CS201"

    # Verify Page 2 chunks
    p2_chunks = [ch for ch in chunks if ch.page_number == 2]
    assert len(p2_chunks) >= 1
    for ch in p2_chunks:
        assert ch.page_number == 2
        assert "Binary Search Trees" in ch.heading
        assert ch.metadata["has_equations_or_theorems"] is True

    # Verify large table splitting repeats headers cleanly
    large_table = """| ID | Metric | Value |
| --- | --- | --- |
| 1 | Latency | 10ms |
| 2 | Throughput | 1000qps |
| 3 | CPU Utilization | 45% |
| 4 | Memory Footprint | 256MB |
| 5 | Cache Hit Ratio | 98.5% |
| 6 | Network Ingress | 10MB/s |
| 7 | Network Egress | 12MB/s |
"""
    small_chunker = AcademicChunker(chunk_size=120, chunk_overlap=0)
    split_chunks = small_chunker.chunk_text(large_table)
    assert len(split_chunks) >= 2
    for sc in split_chunks:
        assert "| ID | Metric | Value |" in sc
        assert "| --- | --- |" in sc


def test_deterministic_chunk_identity_and_metadata():
    """Verify chunking the same ParsedDocument twice produces identical chunk_ids, ordering, and metadata."""
    chunker = AcademicChunker(chunk_size=300, chunk_overlap=50)

    doc = ParsedDocument(
        document_id="doc-fixed-uuid",
        filename="it101_operating_systems.pdf",
        file_type="pdf",
        subject="IT",
        pages=[
            DocumentPage(
                page_number=1,
                text="IT 101 Introduction to Operating Systems.\nKernel architecture manages hardware abstractions.",
                heading="Chapter 1: Intro"
            ),
            DocumentPage(
                page_number=2,
                text="Process scheduling algorithms include Round Robin, FCFS, and Priority Scheduling.",
                heading="Chapter 2: Scheduling"
            )
        ],
        title="IT 101 Notes",
        file_hash="9876543210fedcba9876543210fedcba9876543210fedcba9876543210fedcba"
    )

    run1 = chunker.chunk_document(doc)
    run2 = chunker.chunk_document(doc)

    assert len(run1) == len(run2)
    for i in range(len(run1)):
        ch1, ch2 = run1[i], run2[i]
        assert ch1.chunk_id == ch2.chunk_id
        assert ch1.chunk_index == ch2.chunk_index
        assert ch1.page_number == ch2.page_number
        assert ch1.text == ch2.text
        assert ch1.heading == ch2.heading
        assert ch1.metadata == ch2.metadata
        assert ch1.doc_category == ch2.doc_category
        assert ch1.token_estimate == ch2.token_estimate
        assert ch1.token_estimate > 0


