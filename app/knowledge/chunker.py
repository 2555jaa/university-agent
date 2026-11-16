"""Intelligent academic structure-aware chunker with document classification and metadata extraction."""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

from app.knowledge.document_parser import ParsedDocument, DocumentPage


# --- ACADEMIC CLASSIFICATION & METADATA HELPERS ---


def classify_academic_document(text: str, filename: str = "", title: str = "") -> str:
    """Classify academic document into syllabus, exam_or_assignment, reference_or_formula_sheet, research_paper, or lecture_notes."""
    combined_header = f"{filename} {title} {text[:3000]}".lower()
    full_lower = text.lower()

    # 1. Syllabus signals
    syllabus_kw = [
        "syllabus", "course outline", "grading policy", "grading scheme", "grading criteria",
        "learning outcomes", "course objectives", "weekly schedule", "office hours",
        "prerequisites", "course description", "instructor information", "course policies",
        "grade breakdown", "evaluation scheme", "required textbook"
    ]
    s_score = sum(3 for kw in syllabus_kw if kw in combined_header) + sum(1 for kw in syllabus_kw if kw in full_lower)
    if any(k in filename.lower() or k in title.lower() for k in ("syllabus", "course_outline")):
        s_score += 10

    # 2. Exam or Assignment signals
    exam_kw = [
        "exam", "midterm", "final exam", "quiz", "test", "assignment", "homework",
        "problem set", "past paper", "total points", "max marks", "total marks", "due date",
        "submission deadline", "time allowed", "instructions to candidates",
        "answer all questions", "question 1", "question 2", "marks]"
    ]
    e_score = sum(3 for kw in exam_kw if kw in combined_header) + sum(1 for kw in exam_kw if kw in full_lower)
    if any(k in filename.lower() or k in title.lower() for k in ("exam", "quiz", "assignment", "homework", "problem_set", "midterm")):
        e_score += 10

    # 3. Reference or Formula Sheet signals
    ref_kw = [
        "formula sheet", "cheat sheet", "reference sheet", "constants", "physical constants",
        "mathematical formulas", "table of integrals", "periodic table", "summary of formulas",
        "summary of equations", "standard values", "conversion factors", "trigonometric identities"
    ]
    r_score = sum(3 for kw in ref_kw if kw in combined_header) + sum(1 for kw in ref_kw if kw in full_lower)
    if any(k in filename.lower() or k in title.lower() for k in ("formula", "cheat_sheet", "constants", "reference_sheet")):
        r_score += 10

    # 4. Research Paper signals
    paper_kw = [
        "abstract\n", "abstract:\n", "methodology", "literature review", "related work",
        "experimental results", "references\n", "bibliography\n", "ieee", "acm", "arxiv:", "doi:"
    ]
    p_score = sum(3 for kw in paper_kw if kw in combined_header) + sum(1 for kw in paper_kw if kw in full_lower)
    if ("abstract" in combined_header and "references" in full_lower) or "arxiv:" in full_lower or "doi:" in full_lower:
        p_score += 8

    # 5. Lecture Notes signals
    lecture_kw = [
        "lecture", "chapter", "slide", "lecture notes", "module", "lesson",
        "theorem", "proof", "definition"
    ]
    l_score = sum(2 for kw in lecture_kw if kw in combined_header) + sum(1 for kw in lecture_kw if kw in full_lower)
    if any(k in filename.lower() or k in title.lower() for k in ("lecture", "notes", "chapter", "slide")):
        l_score += 5

    scores = {
        "syllabus": s_score,
        "exam_or_assignment": e_score,
        "reference_or_formula_sheet": r_score,
        "research_paper": p_score,
        "lecture_notes": l_score,
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 3 else "lecture_notes"


def detect_course_code(text: str, filename: str = "", title: str = "") -> Optional[str]:
    """Extract academic course code pattern (e.g. CS201, MT 204, AI-301, IT 101)."""
    target = f"{filename} {title} {text[:2000]}"
    pattern = re.compile(r'\b([A-Z]{2,5}\s*[-_]?\s*\d{3,4}[A-Z]?)\b')
    match = pattern.search(target)
    if match:
        code = match.group(1).strip()
        return re.sub(r'\s+', ' ', code)
    return None


def extract_academic_features(text: str) -> Dict[str, bool]:
    """Detect structure elements: tables, code blocks, equations/theorems."""
    has_tables = bool(re.search(r'\|[^\n]+\|\n\|[\s\-:|]+\|', text))
    has_code = "```" in text
    has_theorems = bool(
        re.search(r'\b(Theorem|Lemma|Definition|Proof|Corollary|Proposition|Example)\b', text, re.IGNORECASE)
        or '$' in text
        or r'\(' in text
        or r'\[' in text
    )
    return {
        "has_tables": has_tables,
        "has_code_blocks": has_code,
        "has_equations_or_theorems": has_theorems,
    }


def is_noise_or_footer(text: str) -> bool:
    """Filter out empty text or standalone page number footers."""
    stripped = text.strip()
    if not stripped:
        return True
    if re.match(r'^(?:page\s+\d+(?:\s+of\s+\d+)?|-?\s*\d+\s*-?|\d+\s*/\s*\d+|\d+\s+of\s+\d+)$', stripped, re.IGNORECASE):
        return True
    return False


# --- DATA SCHEMAS ---


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    subject: Optional[str]
    page_number: int
    chunk_index: int
    text: str
    heading: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_category: str = "lecture_notes"
    token_estimate: int = 0
    word_count: int = 0
    char_count: int = 0

    def __post_init__(self):
        if not self.char_count and self.text:
            self.char_count = len(self.text)
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())
        if not self.token_estimate and self.word_count:
            self.token_estimate = max(1, int(self.word_count * 1.3))


# --- ACADEMIC CHUNKER ---


class AcademicChunker:
    """Splits academic texts into structure-aware overlapping chunks preserving tables, code, and theorems."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def classify_document(doc: ParsedDocument) -> str:
        """Classify document based on filename, title, and page contents."""
        full_text = "\n\n".join(p.text for p in doc.pages)
        return classify_academic_document(full_text, filename=doc.filename, title=doc.title)

    @staticmethod
    def detect_course_code(text: str, filename: str = "", title: str = "") -> Optional[str]:
        return detect_course_code(text, filename=filename, title=title)

    @staticmethod
    def _extract_atomic_blocks(text: str) -> List[Tuple[str, str]]:
        """Partition text into atomic blocks: tables, code blocks, and prose paragraphs."""
        if not text or not text.strip():
            return []

        spans: List[Tuple[int, int, str]] = []

        # 1. Markdown tables
        for m in re.finditer(r'(?:^|\n)(\|[^\n]+\|\n\|[\s\-:|]+\|\n(?:\|[^\n]+\|\n?)+)', text, re.MULTILINE):
            spans.append((m.start(1), m.end(1), "table"))

        # 2. Fenced code blocks
        for m in re.finditer(r'(```[^\n]*\n[\s\S]*?```)', text, re.MULTILINE):
            spans.append((m.start(1), m.end(1), "code"))

        spans.sort(key=lambda x: x[0])
        non_overlapping: List[Tuple[int, int, str]] = []
        last_end = 0
        for s, e, btype in spans:
            if s >= last_end:
                non_overlapping.append((s, e, btype))
                last_end = e

        blocks: List[Tuple[str, str]] = []
        cursor = 0
        for s, e, btype in non_overlapping:
            if s > cursor:
                prose = text[cursor:s].strip()
                if prose:
                    for p in re.split(r'\n\s*\n', prose):
                        p_clean = p.strip()
                        if p_clean and not is_noise_or_footer(p_clean):
                            blocks.append(("prose", p_clean))
            block_content = text[s:e].strip()
            if block_content:
                blocks.append((btype, block_content))
            cursor = e

        if cursor < len(text):
            prose = text[cursor:].strip()
            if prose:
                for p in re.split(r'\n\s*\n', prose):
                    p_clean = p.strip()
                    if p_clean and not is_noise_or_footer(p_clean):
                        blocks.append(("prose", p_clean))

        return blocks

    @staticmethod
    def _split_table(table_text: str, max_size: int) -> List[str]:
        """Split a large table across row boundaries repeating the header row on continuations."""
        lines = [l.strip() for l in table_text.strip().splitlines() if l.strip()]
        if len(lines) <= 2:
            return [table_text]

        header = lines[:2]
        header_str = "\n".join(header)
        rows = lines[2:]

        table_chunks = []
        current_rows = []
        current_len = len(header_str)

        for row in rows:
            row_len = len(row) + 1
            if current_rows and (current_len + row_len > max_size):
                table_chunks.append(header_str + "\n" + "\n".join(current_rows))
                current_rows = [row]
                current_len = len(header_str) + row_len
            else:
                current_rows.append(row)
                current_len += row_len

        if current_rows:
            table_chunks.append(header_str + "\n" + "\n".join(current_rows))

        return table_chunks

    @staticmethod
    def _split_sentences(text: str, max_size: int) -> List[str]:
        """Split prose text by sentence boundaries without breaking words."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return [text] if text.strip() else []

        subchunks = []
        curr = ""
        for s in sentences:
            if len(curr) + len(s) + 1 <= max_size:
                curr = f"{curr} {s}" if curr else s
            else:
                if curr:
                    subchunks.append(curr)
                if len(s) > max_size:
                    words = s.split()
                    w_curr = ""
                    for w in words:
                        if len(w_curr) + len(w) + 1 <= max_size:
                            w_curr = f"{w_curr} {w}" if w_curr else w
                        else:
                            if w_curr:
                                subchunks.append(w_curr)
                            w_curr = w
                    curr = w_curr
                else:
                    curr = s

        if curr:
            subchunks.append(curr)
        return subchunks

    @staticmethod
    def _split_lines(text: str, max_size: int) -> List[str]:
        """Split code blocks or preformatted text by lines."""
        lines = text.splitlines()
        chunks = []
        curr: List[str] = []
        curr_len = 0
        for l in lines:
            if curr_len + len(l) + 1 <= max_size:
                curr.append(l)
                curr_len += len(l) + 1
            else:
                if curr:
                    chunks.append("\n".join(curr))
                curr = [l]
                curr_len = len(l) + 1
        if curr:
            chunks.append("\n".join(curr))
        return chunks if chunks else [text]

    def chunk_text(self, text: str) -> List[str]:
        """Split text cleanly into structure-aware chunks respecting tables, code, and sentences."""
        if not text or len(text.strip()) <= self.chunk_size:
            cleaned = text.strip()
            return [cleaned] if cleaned and not is_noise_or_footer(cleaned) else []

        atomic_blocks = self._extract_atomic_blocks(text)
        if not atomic_blocks:
            return []

        raw_chunks: List[str] = []
        current_chunk_blocks: List[str] = []
        current_chunk_len = 0

        for btype, bcontent in atomic_blocks:
            b_len = len(bcontent)
            sep_len = 2 if current_chunk_blocks else 0

            if current_chunk_len + b_len + sep_len <= self.chunk_size:
                current_chunk_blocks.append(bcontent)
                current_chunk_len += b_len + sep_len
            else:
                if current_chunk_blocks:
                    raw_chunks.append("\n\n".join(current_chunk_blocks).strip())
                    current_chunk_blocks = []
                    current_chunk_len = 0

                if b_len <= self.chunk_size:
                    current_chunk_blocks.append(bcontent)
                    current_chunk_len = b_len
                else:
                    # Oversized atomic block
                    if btype == "table":
                        subs = self._split_table(bcontent, self.chunk_size)
                    elif btype == "code":
                        subs = self._split_lines(bcontent, self.chunk_size)
                    else:
                        subs = self._split_sentences(bcontent, self.chunk_size)

                    for sub in subs[:-1]:
                        raw_chunks.append(sub)
                    if subs:
                        current_chunk_blocks.append(subs[-1])
                        current_chunk_len = len(subs[-1])

        if current_chunk_blocks:
            raw_chunks.append("\n\n".join(current_chunk_blocks).strip())

        # Apply overlap where appropriate between prose chunks
        overlapped_chunks: List[str] = []
        for i, ch in enumerate(raw_chunks):
            if i > 0 and self.chunk_overlap > 0:
                prev = raw_chunks[i - 1]
                # Only apply overlap if neither chunk is a table or code block
                if not ch.startswith("|") and not prev.startswith("|") and not ch.startswith("```"):
                    prev_tail = prev[-self.chunk_overlap:]
                    combined = f"...{prev_tail} {ch}"
                    overlapped_chunks.append(combined)
                else:
                    overlapped_chunks.append(ch)
            else:
                overlapped_chunks.append(ch)

        return [c for c in overlapped_chunks if c.strip() and not is_noise_or_footer(c)]

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        """Chunk all pages of a parsed document and assign source metadata, classification, and active heading."""
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0
        current_heading: Optional[str] = None

        full_text = "\n\n".join(p.text for p in doc.pages)
        doc_category = getattr(doc, "doc_category", None) or self.classify_document(doc)
        course_code = getattr(doc, "course_code", None) or self.detect_course_code(full_text, doc.filename, doc.title)
        features = extract_academic_features(full_text)

        doc.metadata.update({
            "doc_category": doc_category,
            "detected_course_code": course_code,
            **features,
        })

        for page in doc.pages:
            if page.heading:
                current_heading = page.heading

            page_text_chunks = self.chunk_text(page.text)
            for ch_text in page_text_chunks:
                if not ch_text.strip() or is_noise_or_footer(ch_text):
                    continue

                # Detect heading within chunk or inherit from page
                md_match = re.search(r'^(#{1,6}\s+[^\n]+)', ch_text, re.MULTILINE)
                sec_match = re.search(r'^(?:Chapter\s+\d+|Section\s+\d+|\d+(?:\.\d+)*)\s*[:.\-–—]?\s*([^\n]+)', ch_text, re.MULTILINE | re.IGNORECASE)

                if md_match:
                    chunk_heading = md_match.group(1).lstrip('#').strip()
                    current_heading = chunk_heading
                elif sec_match:
                    chunk_heading = sec_match.group(0).strip()
                    current_heading = chunk_heading
                else:
                    chunk_heading = current_heading or "Overview"

                # Deterministic chunk ID using UUID5
                seed = f"{doc.file_hash or doc.sha256 or doc.document_id}#chunk_{global_chunk_idx}"
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

                ch_features = extract_academic_features(ch_text)
                word_count = len(ch_text.split())
                char_count = len(ch_text)
                token_estimate = max(1, int(word_count * 1.3))

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    filename=doc.filename,
                    subject=doc.subject,
                    page_number=page.page_number,
                    chunk_index=global_chunk_idx,
                    text=ch_text,
                    heading=chunk_heading,
                    metadata={
                        **page.metadata,
                        "document_id": doc.document_id,
                        "filename": doc.filename,
                        "file_hash": doc.file_hash or doc.sha256,
                        "file_type": doc.file_type,
                        "subject": doc.subject,
                        "doc_category": doc_category,
                        "detected_course_code": course_code,
                        "page_number": page.page_number,
                        "chunk_index": global_chunk_idx,
                        "heading": chunk_heading,
                        "section_heading": chunk_heading,
                        "has_tables": ch_features["has_tables"],
                        "has_code_blocks": ch_features["has_code_blocks"],
                        "has_equations_or_theorems": ch_features["has_equations_or_theorems"],
                        "token_estimate": token_estimate,
                        "word_count": word_count,
                        "char_count": char_count,
                        "total_pages": len(doc.pages),
                        "sha256": doc.file_hash or doc.sha256,
                    },
                    doc_category=doc_category,
                    token_estimate=token_estimate,
                    word_count=word_count,
                    char_count=char_count,
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1

        return all_chunks


academic_chunker = AcademicChunker()
