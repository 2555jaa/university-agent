"""Comprehensive tests for PDF, DOCX, TXT, and Markdown parsing, heading hierarchy, and corrupt/scanned detection."""

from pathlib import Path
import pytest
try:
    import pymupdf as fitz
except ImportError:
    import fitz
import docx

from app.knowledge.document_parser import (
    DocumentParser,
    DocumentPage,
    ParsedDocument,
    DocumentParseError,
    UnsupportedFileFormatError,
    EmptyDocumentError,
    CorruptDocumentError,
    ScannedDocumentError,
    document_parser,
)


def test_real_pdf_generation_and_parsing(tmp_path):
    """Generate a valid binary PDF document, parse it, and verify extracted text."""
    pdf_path = tmp_path / "calculus_lecture.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 72), "Calculus I: The Fundamental Theorem of Calculus connects differentiation with integration.")
    page2 = doc.new_page()
    page2.insert_text((50, 72), "Page 2: Derivatives represent instantaneous rates of change.")
    doc.save(str(pdf_path))
    doc.close()

    parsed = document_parser.parse_document(pdf_path, subject="Mathematics")
    assert parsed.filename == "calculus_lecture.pdf"
    assert parsed.subject == "Mathematics"
    assert len(parsed.pages) == 2
    assert "Fundamental Theorem" in parsed.pages[0].text
    assert "rates of change" in parsed.pages[1].text.lower()
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[1].page_number == 2


def test_real_docx_generation_and_parsing(tmp_path):
    """Generate a valid DOCX document, parse it, and verify extracted text."""
    docx_path = tmp_path / "ai_ethics.docx"
    doc = docx.Document()
    doc.add_heading("AI Ethics in Healthcare", level=1)
    doc.add_paragraph("Autonomous diagnostic systems must uphold transparency, beneficence, and non-maleficence.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Principle"
    table.cell(0, 1).text = "Definition"
    table.cell(1, 0).text = "Autonomy"
    table.cell(1, 1).text = "Respect patient decision-making rights."
    doc.save(str(docx_path))

    parsed = document_parser.parse_document(docx_path, subject="Philosophy")
    assert parsed.filename == "ai_ethics.docx"
    assert len(parsed.pages) >= 1
    assert "Autonomous diagnostic systems" in parsed.pages[0].text
    assert "Autonomy" in parsed.pages[0].text


@pytest.mark.asyncio
async def test_pdf_end_to_end_rag_ingestion(tmp_path, test_container):
    """End-to-end ingestion of a PDF into the RAG knowledge system with citation retrieval."""
    pdf_path = tmp_path / "operating_systems.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Virtual Memory: Paging divides process address spaces into fixed-size frames.")
    doc.save(str(pdf_path))
    doc.close()

    result = await test_container.rag_service.ingest_document(pdf_path, subject="Computer Science")
    assert result.status == "success"
    assert result.total_pages == 1
    assert result.total_chunks >= 1

    # Query RAG
    context = await test_container.rag_service.retrieve_grounded_context("What does paging do in virtual memory?", subject="Computer Science")
    assert context.has_relevant_info is True
    assert len(context.citations) >= 1
    assert context.citations[0]["filename"] == "operating_systems.pdf"
    assert context.citations[0]["page_number"] == 1


# --- MULTI-PAGE PDF & HEADING PRESERVATION TESTS ---


def test_multipage_pdf_page_number_and_heading_preservation(tmp_path):
    """Verify exact 1-based page numbers and heading inheritance across multi-page PDFs."""
    pdf_path = tmp_path / "distributed_systems.pdf"
    doc = fitz.open()

    # Page 1
    p1 = doc.new_page()
    p1.insert_text((50, 50), "1. Foundations of Distributed Systems", fontsize=16)
    p1.insert_text((50, 80), "A distributed system consists of autonomous nodes communicating over a network.", fontsize=11)

    # Page 2
    p2 = doc.new_page()
    p2.insert_text((50, 50), "1.1 Consensus and Byzantine Fault Tolerance", fontsize=16)
    p2.insert_text((50, 80), "Consensus requires agreement among nodes even in the presence of arbitrary failures.", fontsize=11)

    # Page 3: continuation without a new heading
    p3 = doc.new_page()
    p3.insert_text((50, 50), "Practical Byzantine Fault Tolerance achieves consensus with 3f+1 replicas.", fontsize=11)

    doc.save(str(pdf_path))
    doc.close()

    parsed = document_parser.parse_file(pdf_path, subject="Computer Science")
    assert parsed.page_count == 3
    assert len(parsed.pages) == 3
    assert parsed.file_type == "pdf"
    assert parsed.file_size_bytes > 0
    assert len(parsed.file_hash) == 64
    assert parsed.title != ""

    # Page 1 assertions
    assert parsed.pages[0].page_number == 1
    assert "1. Foundations of Distributed Systems" in parsed.pages[0].heading
    assert "autonomous nodes" in parsed.pages[0].text

    # Page 2 assertions
    assert parsed.pages[1].page_number == 2
    assert "1.1 Consensus" in parsed.pages[1].heading
    assert "arbitrary failures" in parsed.pages[1].text

    # Page 3 assertions: inherits active heading
    assert parsed.pages[2].page_number == 3
    assert "1.1 Consensus" in parsed.pages[2].heading
    assert "Practical Byzantine Fault Tolerance" in parsed.pages[2].text


# --- DOCX TABLE & HEADING HIERARCHY TESTS ---


def test_docx_paragraphs_headings_and_table_extraction(tmp_path):
    """Verify DOCX parser extracts interleaved paragraphs, heading hierarchy, and tables in Markdown format."""
    docx_path = tmp_path / "database_systems_syllabus.docx"
    doc = docx.Document()

    doc.add_heading("CS301 Database Systems Syllabus", level=1)
    doc.add_paragraph("Welcome to Database Systems. Below is the grading criteria and course breakdown:")

    doc.add_heading("Grading Scheme", level=2)
    doc.add_paragraph("Coursework and examination weighting:")

    table = doc.add_table(rows=3, cols=2)
    table.cell(0, 0).text = "Assessment Component"
    table.cell(0, 1).text = "Weight"
    table.cell(1, 0).text = "Midterm Examination"
    table.cell(1, 1).text = "30%"
    table.cell(2, 0).text = "Final Project"
    table.cell(2, 1).text = "70%"

    doc.add_heading("Course Policies", level=2)
    doc.add_paragraph("Late submissions will be penalized by 10% per day.")

    doc.save(str(docx_path))

    parsed = document_parser.parse_file(docx_path, subject="Computer Science")
    assert parsed.file_type == "docx"
    assert len(parsed.pages) >= 1

    full_text = "\n\n".join(p.text for p in parsed.pages)
    # Check that Markdown table was constructed
    assert "| Assessment Component | Weight |" in full_text
    assert "| --- | --- |" in full_text
    assert "| Midterm Examination | 30% |" in full_text
    assert "| Final Project | 70% |" in full_text
    assert "Late submissions will be penalized" in full_text


# --- MARKDOWN & TXT HEADING HIERARCHY TESTS ---


def test_markdown_and_txt_heading_hierarchy_preservation(tmp_path):
    """Verify Markdown and plain text preserve ATX headings and numbered sections."""
    # 1. Markdown test
    md_path = tmp_path / "operating_systems.md"
    md_content = """# Operating Systems Architecture

Overview of modern kernel architecture and abstractions.

## Memory Management

Virtual memory isolates address spaces and maps virtual to physical pages.

### Page Replacement Algorithms

LRU and Clock algorithms minimize page fault frequency under memory pressure.
"""
    md_path.write_text(md_content, encoding="utf-8")

    parsed_md = document_parser.parse_file(md_path)
    assert parsed_md.file_type == "md"
    assert len(parsed_md.pages) == 3
    assert parsed_md.pages[0].heading == "Operating Systems Architecture"
    assert parsed_md.pages[1].heading == "Memory Management"
    assert parsed_md.pages[2].heading == "Page Replacement Algorithms"
    assert "Virtual memory isolates" in parsed_md.pages[1].text

    # 2. Numbered academic text test
    txt_path = tmp_path / "networking.txt"
    txt_content = """1. Introduction to Computer Networks
Computer networks interconnect heterogeneous devices.

1.1 Transmission Control Protocol
TCP guarantees in-order, reliable byte stream delivery.

1.2 User Datagram Protocol
UDP provides connectionless, lightweight datagram transfer.
"""
    txt_path.write_text(txt_content, encoding="utf-8")

    parsed_txt = document_parser.parse_file(txt_path)
    assert parsed_txt.file_type == "txt"
    assert len(parsed_txt.pages) == 3
    assert "1. Introduction" in parsed_txt.pages[0].heading
    assert "1.1 Transmission" in parsed_txt.pages[1].heading
    assert "1.2 User Datagram" in parsed_txt.pages[2].heading


# --- CORRUPT, EMPTY & INVALID FILE DETECTION TESTS ---


def test_corrupt_files_and_empty_files_raise_explicit_errors(tmp_path):
    """Verify that corrupt files, 0-byte files, and disguised binaries raise explicit custom errors."""
    # 1. 0-byte file
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")
    with pytest.raises(EmptyDocumentError, match="empty \\(0 bytes\\)"):
        document_parser.parse_file(empty_pdf)

    # 2. Corrupt PDF header
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"NOT_A_PDF_HEADER_JUST_GARBAGE_BYTES")
    with pytest.raises(CorruptDocumentError, match="missing '%PDF-' header signature"):
        document_parser.parse_file(corrupt_pdf)

    # 3. Corrupt DOCX (invalid zip)
    corrupt_docx = tmp_path / "corrupt.docx"
    corrupt_docx.write_bytes(b"PK\x03\x04INVALID_ZIP_ARCHIVE_DATA_TRUNCATED")
    with pytest.raises(CorruptDocumentError, match="Corrupted DOCX archive"):
        document_parser.parse_file(corrupt_docx)

    # 4. Binary file disguised as TXT
    binary_txt = tmp_path / "disguised.txt"
    binary_txt.write_bytes(b"Hello world\x00\x01\x02\x03\xffBinaryData")
    with pytest.raises(CorruptDocumentError, match="binary null bytes"):
        document_parser.parse_file(binary_txt)

    # 5. Unsupported file format
    tar_file = tmp_path / "archive.tar.gz"
    tar_file.write_bytes(b"random-archive-content")
    with pytest.raises(UnsupportedFileFormatError, match="Unsupported file format"):
        document_parser.parse_file(tar_file)

    # 6. Non-existent file
    with pytest.raises(FileNotFoundError):
        document_parser.parse_file(tmp_path / "non_existent.pdf")

    # 7. Whitespace-only text file
    whitespace_txt = tmp_path / "blank.txt"
    whitespace_txt.write_text("   \n\n\t   \n  ", encoding="utf-8")
    with pytest.raises(EmptyDocumentError, match="contains no readable text"):
        document_parser.parse_file(whitespace_txt)


# --- SCANNED PDF & OCR HOOK TESTS ---


def test_scanned_pdf_detection_and_ocr_hook(tmp_path):
    """Verify that image-only PDFs raise ScannedDocumentError and execute OCR hook when configured."""
    scanned_pdf = tmp_path / "scanned_receipt.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Insert an image with zero text layer
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 80, 80), 1)
    page.insert_image(fitz.Rect(50, 50, 150, 150), pixmap=pix)
    doc.save(str(scanned_pdf))
    doc.close()

    # Case 1: Without OCR hook -> raises ScannedDocumentError
    with pytest.raises(ScannedDocumentError, match="contains scanned images/drawings but no extractable text layer"):
        document_parser.parse_file(scanned_pdf)

    # Case 2: With OCR hook -> succeeds and extracts OCR text
    def mock_ocr(file_path: Path):
        return [
            DocumentPage(
                page_number=1,
                text="Scanned Course Syllabus: IT Security and Cryptography",
                heading="Course Syllabus"
            )
        ]

    parsed = document_parser.parse_file(scanned_pdf, ocr_hook=mock_ocr)
    assert parsed.page_count == 1
    assert "IT Security and Cryptography" in parsed.pages[0].text
    assert parsed.pages[0].heading == "Course Syllabus"
