"""PDF and DOCX document parsing tests."""

from pathlib import Path
import pytest
import fitz  # PyMuPDF
import docx  # python-docx

from app.knowledge.document_parser import document_parser
from app.knowledge.rag_service import rag_service


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
async def test_pdf_end_to_end_rag_ingestion(tmp_path):
    """End-to-end ingestion of a PDF into the RAG knowledge system with citation retrieval."""
    pdf_path = tmp_path / "operating_systems.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Virtual Memory: Paging divides process address spaces into fixed-size frames.")
    doc.save(str(pdf_path))
    doc.close()

    result = await rag_service.ingest_document(pdf_path, subject="Computer Science")
    assert result.status == "success"
    assert result.total_pages == 1
    assert result.total_chunks >= 1

    # Query RAG
    context = await rag_service.retrieve_grounded_context("What does paging do in virtual memory?", subject="Computer Science")
    assert context.has_relevant_info is True
    assert len(context.citations) >= 1
    assert context.citations[0]["filename"] == "operating_systems.pdf"
    assert context.citations[0]["page_number"] == 1
