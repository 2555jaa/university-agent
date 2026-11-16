"""Document parser for academic documents (PDF, DOCX, TXT, MD) with heading detection, table extraction, and corrupt/scanned detection."""

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional
import unicodedata
import uuid
import warnings
import zipfile

logger = logging.getLogger("university_agent.knowledge.parser")


# --- CUSTOM EXCEPTIONS ---


class DocumentParseError(ValueError):
    """Base exception for all document parsing errors."""
    pass


class UnsupportedFileFormatError(DocumentParseError):
    """Raised when file extension or format is not supported."""
    pass


class EmptyDocumentError(DocumentParseError):
    """Raised when a document is empty or yields zero meaningful text."""
    pass


class CorruptDocumentError(DocumentParseError):
    """Raised when a document file is corrupted or fails structural validation."""
    pass


class ScannedDocumentError(DocumentParseError):
    """Raised when a PDF contains scanned images/drawings but no extractable text layer."""
    pass


# --- DATA SCHEMAS ---


@dataclass
class DocumentPage:
    page_number: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    heading: Optional[str] = None

    @property
    def content(self) -> str:
        """Alias for text for consistent access across components."""
        return self.text


@dataclass
class ParsedDocument:
    document_id: str
    filename: str
    file_type: str
    subject: Optional[str]
    pages: List[DocumentPage]
    total_characters: int = 0
    sha256: str = ""
    file_path: str = ""
    file_size_bytes: int = 0
    file_hash: str = ""
    title: str = ""
    page_count: int = 0
    word_count: int = 0
    char_count: int = 0
    doc_category: Optional[str] = None
    course_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.file_hash and self.sha256:
            self.file_hash = self.sha256
        elif not self.sha256 and self.file_hash:
            self.sha256 = self.file_hash

        if not self.char_count and self.total_characters:
            self.char_count = self.total_characters
        elif not self.total_characters and self.char_count:
            self.total_characters = self.char_count

        if not self.page_count:
            self.page_count = len(self.pages)

        if not self.word_count and self.pages:
            self.word_count = sum(len(p.text.split()) for p in self.pages)

    @property
    def sections(self) -> List[DocumentPage]:
        """Alias for pages for section-based document pipelines."""
        return self.pages


# --- DOCUMENT PARSER ---


class DocumentParser:
    """Parses PDF, DOCX, TXT, and Markdown files with heading hierarchy, table extraction, and corrupt/scanned detection."""

    def __init__(self, ocr_hook: Optional[Callable[[Path], List[DocumentPage]]] = None):
        self.ocr_hook = ocr_hook

    @classmethod
    def validate_file(cls, file_path: Path) -> Path:
        """Validate file existence, size, extension, and magic byte integrity."""
        p = Path(file_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Document file does not exist: {p}")
        if not p.is_file():
            raise DocumentParseError(f"Path is not a regular file: {p}")

        size = p.stat().st_size
        if size == 0:
            raise EmptyDocumentError(f"Document file '{p.name}' is empty (0 bytes).")

        suffix = p.suffix.lower()
        if suffix not in (".pdf", ".docx", ".doc", ".txt", ".md"):
            raise UnsupportedFileFormatError(
                f"Unsupported file format '{suffix}'. Supported formats: .pdf, .docx, .txt, .md"
            )

        # Inspect initial byte signatures
        try:
            with open(p, "rb") as f:
                head = f.read(2048)
        except Exception as e:
            raise CorruptDocumentError(f"Cannot read file header for '{p.name}': {e}")

        if suffix == ".pdf":
            if b"%PDF-" not in head:
                raise CorruptDocumentError(
                    f"File '{p.name}' is not a valid PDF: missing '%PDF-' header signature."
                )
        elif suffix in (".docx", ".doc"):
            if not head.startswith(b"PK\x03\x04"):
                raise CorruptDocumentError(
                    f"File '{p.name}' is not a valid DOCX file: missing PK zip signature."
                )
            try:
                with zipfile.ZipFile(p, "r") as zf:
                    namelist = zf.namelist()
                    if not any("document.xml" in name for name in namelist) and "[Content_Types].xml" not in namelist:
                        raise CorruptDocumentError(f"Corrupt DOCX archive '{p.name}': missing document structures.")
            except zipfile.BadZipFile as e:
                raise CorruptDocumentError(f"Corrupted DOCX archive '{p.name}': {e}")
        elif suffix in (".txt", ".md"):
            if b"\x00" in head:
                raise CorruptDocumentError(
                    f"File '{p.name}' contains binary null bytes and cannot be parsed as text/markdown."
                )

        return p

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize unicode ligatures, fix hyphenated line wraps, and normalize whitespace."""
        if not text:
            return ""
        # Unicode normalization (NFKC decomposes ligatures like fi, fl)
        norm = unicodedata.normalize("NFKC", text)
        # Fix hyphenated words broken across line wraps: e.g., "differen-\ntiation" -> "differentiation"
        norm = re.sub(r'(\b[A-Za-z]+)-\n([a-z]+)\b', r'\1\2', norm)
        # Replace non-breaking spaces and uniformize newlines
        norm = norm.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
        # Strip excessive trailing whitespace per line while preserving paragraph breaks
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in norm.split("\n")]
        cleaned = "\n".join(lines)
        # Collapse 3+ newlines into 2 (paragraphs)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    @classmethod
    def _extract_heading_from_text(cls, text: str) -> Optional[str]:
        """Detect section headers, markdown headings, or structural title lines."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:8]:
            # Markdown heading (# to ######)
            md_match = re.match(r"^#{1,6}\s+(.+)$", line)
            if md_match:
                return md_match.group(1).strip()
            # Numbered academic headings: "1. Introduction", "1.1 Background", "Chapter 1: Foundations"
            sec_match = re.match(
                r"^(?:Chapter\s+\d+|Section\s+\d+|\d+(?:\.\d+)*)\s*[:.\-–—]?\s*(.+)$",
                line,
                re.IGNORECASE,
            )
            if sec_match:
                return line.strip()
            # Short uppercase or title-case line without terminal punctuation
            if len(line) < 80 and not line.endswith((".", ",", ";", ":", "?", "!")):
                if line.isupper() or line.istitle():
                    if len(line.split()) <= 10:
                        return line.strip()
        return None

    @classmethod
    def parse_pdf(cls, file_path: Path, ocr_hook: Optional[Callable] = None) -> List[DocumentPage]:
        """Extract text page by page from PDF using PyMuPDF (fitz) with heading detection and scanned PDF detection."""
        cls.validate_file(file_path)
        pages: List[DocumentPage] = []
        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz

            try:
                doc = fitz.open(str(file_path))
            except Exception as e:
                raise CorruptDocumentError(f"Failed to open corrupted PDF '{file_path.name}': {e}")

            total_pages = len(doc)
            if total_pages == 0:
                doc.close()
                raise EmptyDocumentError(f"PDF '{file_path.name}' has 0 pages.")

            current_heading: str = "Overview"
            raw_page_texts: List[str] = []
            total_images_or_drawings = 0

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_num = page_idx + 1

                images_count = len(page.get_images())
                drawings_count = len(page.get_drawings())
                total_images_or_drawings += (images_count + drawings_count)

                raw_text = page.get_text("text")
                cleaned = cls.clean_text(raw_text)
                raw_page_texts.append(cleaned)

                # Heuristic: inspect font sizes and bold blocks in dict representation
                page_heading: Optional[str] = None
                try:
                    page_dict = page.get_text("dict")
                    max_size = 0.0
                    heading_candidate = ""
                    for block in page_dict.get("blocks", []):
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line.get("spans", []):
                                    stext = span.get("text", "").strip()
                                    ssize = float(span.get("size", 0.0))
                                    if ssize > max_size and len(stext) > 2 and len(stext.split()) <= 12:
                                        if not stext.endswith((".", ",", ";")):
                                            max_size = ssize
                                            heading_candidate = stext
                    if max_size >= 13.0 and heading_candidate:
                        page_heading = heading_candidate
                except Exception:
                    pass

                if not page_heading and cleaned:
                    detected = cls._extract_heading_from_text(cleaned)
                    if detected:
                        page_heading = detected

                if page_heading:
                    current_heading = page_heading

                if cleaned:
                    pages.append(DocumentPage(
                        page_number=page_num,
                        text=cleaned,
                        heading=current_heading,
                        metadata={
                            "page_number": page_num,
                            "total_pages": total_pages,
                            "heading": current_heading,
                            "char_count": len(cleaned),
                            "word_count": len(cleaned.split())
                        }
                    ))

            doc.close()

            total_chars = sum(len(t) for t in raw_page_texts)
            if total_chars < 15:
                # Scanned or image-only PDF detection
                if total_images_or_drawings > 0:
                    hook = ocr_hook or getattr(cls, "_global_ocr_hook", None)
                    if hook and callable(hook):
                        try:
                            hook_res = hook(file_path)
                            if isinstance(hook_res, list) and all(isinstance(x, DocumentPage) for x in hook_res):
                                return hook_res
                            elif isinstance(hook_res, list) and all(isinstance(x, str) for x in hook_res):
                                return [
                                    DocumentPage(page_number=i + 1, text=cls.clean_text(t), heading="Scanned Content")
                                    for i, t in enumerate(hook_res)
                                    if cls.clean_text(t)
                                ]
                            elif isinstance(hook_res, str) and hook_res.strip():
                                return [DocumentPage(page_number=1, text=cls.clean_text(hook_res), heading="Scanned Content")]
                        except Exception as e:
                            logger.warning(f"OCR hook failed on {file_path.name}: {e}")

                    raise ScannedDocumentError(
                        f"PDF '{file_path.name}' contains scanned images/drawings but no extractable text layer. "
                        f"Configure an OCR hook (e.g. pytesseract) to process scanned documents."
                    )
                else:
                    raise EmptyDocumentError(
                        f"PDF '{file_path.name}' has {total_pages} page(s) but contains no extractable text."
                    )

            if not pages:
                raise EmptyDocumentError(f"PDF '{file_path.name}' contains no readable text.")

            return pages
        except (DocumentParseError, FileNotFoundError):
            raise
        except ImportError:
            logger.warning("PyMuPDF not installed, attempting fallback text extraction.")
            raw_text = file_path.read_text(errors="ignore").strip()
            if not raw_text:
                raise EmptyDocumentError(f"PDF '{file_path.name}' contains no readable text.")
            heading = cls._extract_heading_from_text(raw_text) or "Overview"
            pages.append(DocumentPage(
                page_number=1,
                text=raw_text,
                heading=heading,
                metadata={"warning": "PyMuPDF missing, extracted as raw stream", "heading": heading}
            ))
            return pages
        except Exception as e:
            logger.error(f"Unexpected error parsing PDF {file_path}: {e}")
            raise CorruptDocumentError(f"Failed to parse PDF {file_path.name}: {e}")

    @staticmethod
    def _format_docx_table_as_markdown(table) -> str:
        """Convert a python-docx Table into clean GitHub Flavored Markdown table format."""
        rows_data = []
        for row in table.rows:
            row_cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
            # Deduplicate repeated adjacent cells caused by merged columns in python-docx
            deduped = []
            for c in row_cells:
                if not deduped or c != deduped[-1] or c == "":
                    deduped.append(c)
            if any(deduped):
                rows_data.append(deduped)

        if not rows_data:
            return ""

        max_cols = max(len(r) for r in rows_data)
        if max_cols == 0:
            return ""

        padded_rows = []
        for r in rows_data:
            padded = r + [""] * (max_cols - len(r))
            padded_rows.append(padded)

        header = padded_rows[0]
        md_lines = ["| " + " | ".join(header) + " |"]
        md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for row in padded_rows[1:]:
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)

    @classmethod
    def parse_docx(cls, file_path: Path) -> List[DocumentPage]:
        """Extract text and tables from Word .docx file preserving heading hierarchy and section layout."""
        cls.validate_file(file_path)
        pages: List[DocumentPage] = []
        try:
            import docx
            from docx.text.paragraph import Paragraph
            from docx.table import Table

            try:
                doc = docx.Document(str(file_path))
            except Exception as e:
                raise CorruptDocumentError(f"Failed to open corrupted DOCX archive '{file_path.name}': {e}")

            sections: List[Dict[str, Any]] = []
            current_heading = "Overview"
            current_content_blocks: List[str] = []

            for child in doc.element.body:
                if child.tag.endswith("p"):
                    p = Paragraph(child, doc)
                    txt = cls.clean_text(p.text)
                    if not txt:
                        continue
                    style_name = p.style.name if p.style else ""
                    is_heading_style = any(style_name.startswith(h) for h in ("Heading", "Title", "Subtitle"))
                    is_numbered = bool(re.match(r'^(?:Chapter\s+\d+|Section\s+\d+|\d+(?:\.\d+)*)\s*[:.\-–—]?\s*(.+)$', txt, re.IGNORECASE))
                    is_short_title = len(txt) < 80 and not txt.endswith((".", ",", ";")) and (txt.isupper() or txt.istitle()) and len(txt.split()) <= 10

                    if is_heading_style or is_numbered or (is_short_title and len(current_content_blocks) > 0):
                        if current_content_blocks:
                            sections.append({"heading": current_heading, "blocks": list(current_content_blocks)})
                            current_content_blocks = []
                        current_heading = txt
                    else:
                        current_content_blocks.append(txt)

                elif child.tag.endswith("tbl"):
                    tbl = Table(child, doc)
                    md_table = cls._format_docx_table_as_markdown(tbl)
                    if md_table:
                        current_content_blocks.append(md_table)

            if current_content_blocks:
                sections.append({"heading": current_heading, "blocks": list(current_content_blocks)})

            if not sections:
                raise EmptyDocumentError(f"DOCX document '{file_path.name}' contains no readable text or tables.")

            total_text_chars = sum(sum(len(b) for b in s["blocks"]) for s in sections)
            if total_text_chars == 0:
                raise EmptyDocumentError(f"DOCX document '{file_path.name}' contains no readable text.")

            page_idx = 1
            for sec in sections:
                sec_text = "\n\n".join(sec["blocks"]).strip()
                if not sec_text:
                    continue
                # Split large sections into logical pages of 2000 characters
                chunk_size = 2000
                for i in range(0, len(sec_text), chunk_size):
                    chunk_txt = sec_text[i:i + chunk_size].strip()
                    if chunk_txt:
                        pages.append(DocumentPage(
                            page_number=page_idx,
                            text=chunk_txt,
                            heading=sec["heading"],
                            metadata={
                                "page_number": page_idx,
                                "heading": sec["heading"],
                                "char_count": len(chunk_txt),
                                "word_count": len(chunk_txt.split())
                            }
                        ))
                        page_idx += 1

            if not pages:
                raise EmptyDocumentError(f"DOCX document '{file_path.name}' contains no readable text.")

            return pages
        except (DocumentParseError, FileNotFoundError):
            raise
        except ImportError:
            logger.warning("python-docx not installed, returning empty document page.")
            raise DocumentParseError("python-docx is required to parse DOCX files.")
        except Exception as e:
            logger.error(f"Failed to parse DOCX {file_path}: {e}")
            raise CorruptDocumentError(f"Failed to parse DOCX {file_path.name}: {e}")

    @classmethod
    def parse_txt(cls, file_path: Path) -> List[DocumentPage]:
        """Extract text and headings from plain text or markdown files with encoding fallback."""
        cls.validate_file(file_path)
        raw_bytes = file_path.read_bytes()

        decoded_text = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                decoded_text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            raise CorruptDocumentError(f"Unable to decode text file '{file_path.name}' with supported encodings.")

        cleaned = cls.clean_text(decoded_text)
        if not cleaned:
            raise EmptyDocumentError(f"Text file '{file_path.name}' contains no readable text.")

        lines = cleaned.split("\n")
        sections: List[Dict[str, Any]] = []
        current_heading = "Overview"
        current_lines: List[str] = []

        for line in lines:
            trimmed = line.strip()
            md_match = re.match(r"^#{1,6}\s+(.+)$", trimmed)
            sec_match = re.match(r"^(?:Chapter\s+\d+|Section\s+\d+|\d+(?:\.\d+)*)\s*[:.\-–—]?\s*(.+)$", trimmed, re.IGNORECASE)

            if md_match:
                if current_lines:
                    sections.append({"heading": current_heading, "text": "\n".join(current_lines).strip()})
                    current_lines = []
                current_heading = md_match.group(1).strip()
            elif sec_match:
                if current_lines:
                    sections.append({"heading": current_heading, "text": "\n".join(current_lines).strip()})
                    current_lines = []
                current_heading = trimmed
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({"heading": current_heading, "text": "\n".join(current_lines).strip()})

        pages: List[DocumentPage] = []
        page_idx = 1
        chunk_size = 2000

        for sec in sections:
            stext = sec["text"]
            if not stext:
                continue
            for i in range(0, len(stext), chunk_size):
                chunk_txt = stext[i:i + chunk_size].strip()
                if chunk_txt:
                    pages.append(DocumentPage(
                        page_number=page_idx,
                        text=chunk_txt,
                        heading=sec["heading"],
                        metadata={
                            "page_number": page_idx,
                            "heading": sec["heading"],
                            "char_count": len(chunk_txt),
                            "word_count": len(chunk_txt.split())
                        }
                    ))
                    page_idx += 1

        if not pages:
            raise EmptyDocumentError(f"File '{file_path.name}' contains no readable text.")

        return pages

    def parse_file(
        self,
        file_path: Path,
        subject: Optional[str] = None,
        ocr_hook: Optional[Callable] = None
    ) -> ParsedDocument:
        """Parse any supported file format and return structured document representation with checksums and metadata."""
        p = self.validate_file(file_path)
        suffix = p.suffix.lower()
        effective_ocr = ocr_hook or self.ocr_hook

        if suffix == ".pdf":
            pages = self.parse_pdf(p, ocr_hook=effective_ocr)
            file_type = "pdf"
        elif suffix in (".docx", ".doc"):
            pages = self.parse_docx(p)
            file_type = "docx"
        elif suffix in (".txt", ".md", ".csv"):
            pages = self.parse_txt(p)
            file_type = "md" if suffix == ".md" else "txt"
        else:
            raise UnsupportedFileFormatError(
                f"Unsupported file format '{suffix}'. Supported: .pdf, .docx, .txt, .md"
            )

        if not pages:
            raise EmptyDocumentError(f"Document '{p.name}' yielded zero extractable pages.")

        file_bytes = p.read_bytes()
        file_sha256 = hashlib.sha256(file_bytes).hexdigest()
        size_bytes = p.stat().st_size

        char_count = sum(len(pg.text) for pg in pages)
        word_count = sum(len(pg.text.split()) for pg in pages)

        # Infer title from PDF metadata, first heading, or filename stem
        title = ""
        if suffix == ".pdf":
            try:
                try:
                    import pymupdf as fitz
                except ImportError:
                    import fitz
                d = fitz.open(str(p))
                t = d.metadata.get("title")
                if t and len(t.strip()) > 2 and "untitled" not in t.lower():
                    title = t.strip()
                d.close()
            except Exception:
                pass

        if not title and pages and pages[0].heading and pages[0].heading not in ("Overview", "General"):
            title = pages[0].heading

        if not title:
            title = p.stem.replace("_", " ").replace("-", " ").title()

        # Document classification & course code hints
        try:
            from app.knowledge.chunker import classify_academic_document, detect_course_code
            full_text = "\n\n".join(pg.text for pg in pages)
            doc_category = classify_academic_document(full_text, filename=p.name, title=title)
            course_code = detect_course_code(full_text, filename=p.name, title=title)
        except Exception:
            doc_category = "lecture_notes"
            course_code = None

        doc_id = str(uuid.uuid4())
        return ParsedDocument(
            document_id=doc_id,
            filename=p.name,
            file_type=file_type,
            subject=subject,
            pages=pages,
            total_characters=char_count,
            sha256=file_sha256,
            file_path=str(p.resolve()),
            file_size_bytes=size_bytes,
            file_hash=file_sha256,
            title=title,
            page_count=len(pages),
            word_count=word_count,
            char_count=char_count,
            doc_category=doc_category,
            course_code=course_code,
            metadata={
                "filename": p.name,
                "path": str(p.resolve()),
                "file_type": file_type,
                "file_size": size_bytes,
                "sha256": file_sha256,
                "title": title,
                "page_count": len(pages),
                "word_count": word_count,
                "char_count": char_count,
                "doc_category": doc_category,
                "detected_course_code": course_code
            }
        )

    def parse_document(
        self,
        file_path: Path,
        subject: Optional[str] = None,
        ocr_hook: Optional[Callable] = None
    ) -> ParsedDocument:
        """Alias for parse_file for backwards compatibility."""
        return self.parse_file(file_path, subject=subject, ocr_hook=ocr_hook)


document_parser = DocumentParser()
