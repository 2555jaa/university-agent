"""Intelligent document chunker for academic RAG indexing with heading awareness."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid
import re

from app.knowledge.document_parser import ParsedDocument


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


class AcademicChunker:
    """Splits academic texts into semantically coherent overlapping chunks with heading awareness."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split text cleanly across paragraph and sentence boundaries."""
        if not text or len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # Split by double newlines (paragraphs) first, then sentences
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself is longer than chunk_size, split by sentences
                if len(para) > self.chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    sub_chunk = ""
                    for sent in sentences:
                        if len(sub_chunk) + len(sent) + 1 <= self.chunk_size:
                            sub_chunk = f"{sub_chunk} {sent}" if sub_chunk else sent
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = sent
                    if sub_chunk:
                        current_chunk = sub_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap where appropriate
        overlapped_chunks = []
        for i, ch in enumerate(chunks):
            if i > 0 and self.chunk_overlap > 0:
                prev_tail = chunks[i - 1][-self.chunk_overlap:]
                combined = f"...{prev_tail} {ch}"
                overlapped_chunks.append(combined)
            else:
                overlapped_chunks.append(ch)

        return overlapped_chunks

    def chunk_document(self, doc: ParsedDocument) -> List[DocumentChunk]:
        """Chunk all pages of a parsed document and assign source metadata and active heading."""
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0
        current_heading: Optional[str] = None

        for page in doc.pages:
            if page.heading:
                current_heading = page.heading

            page_text_chunks = self.chunk_text(page.text)
            for ch_text in page_text_chunks:
                # Check if chunk text contains its own markdown heading
                md_match = re.search(r'^(#{1,4}\s+[^\n]+)', ch_text, re.MULTILINE)
                chunk_heading = md_match.group(1).lstrip('#').strip() if md_match else current_heading

                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=doc.document_id,
                    filename=doc.filename,
                    subject=doc.subject,
                    page_number=page.page_number,
                    chunk_index=global_chunk_idx,
                    text=ch_text,
                    heading=chunk_heading,
                    metadata={
                        **page.metadata,
                        "file_type": doc.file_type,
                        "total_pages": len(doc.pages),
                        "heading": chunk_heading,
                        "sha256": doc.sha256
                    }
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1

        return all_chunks


academic_chunker = AcademicChunker()
