"""Document parser for academic documents (PDF, DOCX, TXT, MD) with heading detection and checksums."""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
import logging
import re

logger = logging.getLogger("university_agent.knowledge.parser")


@dataclass
class DocumentPage:
    page_number: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    heading: Optional[str] = None


@dataclass
class ParsedDocument:
    document_id: str
    filename: str
    file_type: str
    subject: Optional[str]
    pages: List[DocumentPage]
    total_characters: int = 0
    sha256: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """Parses PDF, DOCX, TXT, and Markdown files extracting text with page-level metadata and headings."""

    @staticmethod
    def _extract_heading_from_text(text: str) -> Optional[str]:
        """Detect section headers or title lines in text blocks."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:5]:
            # Markdown heading
            md_match = re.match(r'^#{1,4}\s+(.+)$', line)
            if md_match:
                return md_match.group(1).strip()
            # Explicit section / chapter pattern
            sec_match = re.match(r'^(?:Chapter\s+\d+|Section\s+\d+|\d+\.\d+)\s*:?\s*(.+)$', line, re.IGNORECASE)
            if sec_match:
                return sec_match.group(0).strip()
            # Short uppercase or title-like line without terminal punctuation
            if len(line) < 60 and not line.endswith((".", ",", ";")) and (line.isupper() or line.istitle()):
                if len(line.split()) <= 8:
                    return line
        return None

    @classmethod
    def parse_pdf(cls, file_path: Path) -> List[DocumentPage]:
        """Extract text page by page from PDF using PyMuPDF (fitz) with heading detection."""
        pages = []
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            current_heading: Optional[str] = None

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text").strip()
                if text:
                    detected_heading = cls._extract_heading_from_text(text)
                    if detected_heading:
                        current_heading = detected_heading

                    pages.append(DocumentPage(
                        page_number=page_idx + 1,
                        text=text,
                        heading=current_heading,
                        metadata={
                            "total_pages": len(doc),
                            "heading": current_heading
                        }
                    ))
            doc.close()
        except ImportError:
            logger.warning("PyMuPDF not installed, attempting fallback PDF text extraction.")
            raw_text = file_path.read_text(errors="ignore").strip()
            heading = cls._extract_heading_from_text(raw_text)
            pages.append(DocumentPage(
                page_number=1,
                text=raw_text,
                heading=heading,
                metadata={"warning": "PyMuPDF missing, extracted as raw stream", "heading": heading}
            ))
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            raise
        return pages

    @classmethod
    def parse_docx(cls, file_path: Path) -> List[DocumentPage]:
        """Extract text and headings from Word .docx file using python-docx."""
        pages = []
        try:
            import docx
            doc = docx.Document(str(file_path))
            
            # Extract paragraphs while tracking active headings
            sections: List[Dict[str, Any]] = []
            current_heading = "Overview"
            current_paras = []

            for p in doc.paragraphs:
                txt = p.text.strip()
                if not txt:
                    continue
                style_name = p.style.name if p.style else ""
                if style_name.startswith("Heading") or (len(txt) < 80 and txt.isupper()):
                    if current_paras:
                        sections.append({"heading": current_heading, "text": "\n\n".join(current_paras)})
                        current_paras = []
                    current_heading = txt
                else:
                    current_paras.append(txt)

            if current_paras:
                sections.append({"heading": current_heading, "text": "\n\n".join(current_paras)})

            # Extract tables
            table_texts = []
            for table in doc.tables:
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_data:
                        table_texts.append(" | ".join(row_data))
            if table_texts and sections:
                sections[-1]["text"] += "\n\nTables:\n" + "\n".join(table_texts)

            if not sections:
                return []

            # Assign sections into logical pages
            page_idx = 1
            for sec in sections:
                sec_text = sec["text"]
                chunk_size = 2000
                for i in range(0, len(sec_text), chunk_size):
                    pages.append(DocumentPage(
                        page_number=page_idx,
                        text=sec_text[i:i + chunk_size],
                        heading=sec["heading"],
                        metadata={"section": page_idx, "heading": sec["heading"]}
                    ))
                    page_idx += 1
        except ImportError:
            logger.warning("python-docx not installed, reading file with binary fallback.")
            pages.append(DocumentPage(page_number=1, text=""))
        except Exception as e:
            logger.error(f"Failed to parse DOCX {file_path}: {e}")
            raise
        return pages

    @classmethod
    def parse_txt(cls, file_path: Path) -> List[DocumentPage]:
        """Extract text and headings from plain text or markdown files."""
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return []

        chunk_size = 2000
        pages = []
        current_heading: Optional[str] = None

        for i in range(0, len(text), chunk_size):
            slice_text = text[i:i + chunk_size]
            detected = cls._extract_heading_from_text(slice_text)
            if detected:
                current_heading = detected
            page_num = (i // chunk_size) + 1
            pages.append(DocumentPage(
                page_number=page_num,
                text=slice_text,
                heading=current_heading,
                metadata={"heading": current_heading}
            ))
        return pages

    def parse_document(self, file_path: Path, subject: Optional[str] = None) -> ParsedDocument:
        """Parse any supported file format and return structured document representation with sha256 checksum."""
        suffix = file_path.suffix.lower()
        doc_id = str(uuid.uuid4())

        # Compute document checksum
        file_bytes = file_path.read_bytes()
        file_sha256 = hashlib.sha256(file_bytes).hexdigest()

        if suffix == ".pdf":
            pages = self.parse_pdf(file_path)
            file_type = "pdf"
        elif suffix in [".docx", ".doc"]:
            pages = self.parse_docx(file_path)
            file_type = "docx"
        elif suffix in [".txt", ".md", ".csv"]:
            pages = self.parse_txt(file_path)
            file_type = "txt"
        else:
            raise ValueError(f"Unsupported file format: '{suffix}'. Supported: .pdf, .docx, .txt, .md")

        total_chars = sum(len(p.text) for p in pages)
        return ParsedDocument(
            document_id=doc_id,
            filename=file_path.name,
            file_type=file_type,
            subject=subject,
            pages=pages,
            total_characters=total_chars,
            sha256=file_sha256,
            metadata={
                "path": str(file_path.resolve()),
                "sha256": file_sha256,
                "file_size": len(file_bytes)
            }
        )


document_parser = DocumentParser()
