"""Extract text from PDF files for RAG indexing."""

import os
import re
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class PdfChunk:
    source_pdf: str
    page_start: int
    page_end: int
    text: str


def extract_pdf_text(pdf_path: str) -> list[tuple[int, str]]:
    """Return list of (page_number_1based, text) for each page."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i + 1, text))
    return pages


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def chunk_pdf_pages(
    pdf_path: str,
    pages_per_chunk: int = 8,
    min_chars: int = 200,
) -> list[PdfChunk]:
    """Group PDF pages into chunks suitable for vector embedding."""
    page_texts = extract_pdf_text(pdf_path)
    if not page_texts:
        return []

    chunks: list[PdfChunk] = []
    batch: list[tuple[int, str]] = []

    def _flush():
        if not batch:
            return
        combined = _clean_text("\n\n".join(t for _, t in batch))
        if len(combined) >= min_chars:
            chunks.append(
                PdfChunk(
                    source_pdf=os.path.abspath(pdf_path),
                    page_start=batch[0][0],
                    page_end=batch[-1][0],
                    text=combined,
                )
            )
        batch.clear()

    for page_num, text in page_texts:
        batch.append((page_num, text))
        if len(batch) >= pages_per_chunk:
            _flush()
    _flush()
    return chunks


def chunks_to_markdown(chunk: PdfChunk, pdf_basename: str) -> str:
    return (
        f"<!-- extracted from {pdf_basename} pages {chunk.page_start}-{chunk.page_end} -->\n\n"
        f"# {pdf_basename} (pages {chunk.page_start}–{chunk.page_end})\n\n"
        f"{chunk.text}\n"
    )
