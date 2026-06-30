#!/usr/bin/env python3
"""
Extract PDF text into chunked Markdown for RAG indexing.

Usage:
    python scripts/digitize_pdf.py style_samples/fastapi_tutorial.pdf
    python scripts/digitize_pdf.py style_samples/*.pdf
    python scripts/digitize_pdf.py --all --index
"""

import os

# Before chromadb / protobuf (fixes Python 3.13 descriptor error).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.pdf_extractor import chunk_pdf_pages, chunks_to_markdown

STYLE_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "style_samples")
EXTRACTED_DIR = os.path.join(STYLE_SAMPLES_DIR, "extracted")


def sanitize_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^\w\-]+", "_", stem).strip("_") or "document"


def digitize_pdf(pdf_path: str, pages_per_chunk: int = 8) -> list[str]:
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        print(f"Skip (not found): {pdf_path}")
        return []

    basename = sanitize_stem(pdf_path)
    out_dir = os.path.join(EXTRACTED_DIR, basename)
    os.makedirs(out_dir, exist_ok=True)

    chunks = chunk_pdf_pages(pdf_path, pages_per_chunk=pages_per_chunk)
    if not chunks:
        print(f"  Warning: no extractable text in {pdf_path} (scanned PDF? use OCR on images)")
        return []

    written = []
    pdf_name = os.path.basename(pdf_path)
    for i, chunk in enumerate(chunks):
        fname = f"chunk_{chunk.page_start:03d}_{chunk.page_end:03d}.md"
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(chunks_to_markdown(chunk, pdf_name))
        written.append(out_path)

    print(f"  {pdf_name}: {len(chunks)} chunk(s) -> {out_dir}/")
    return written


def main():
    parser = argparse.ArgumentParser(description="Extract PDFs to style_samples/extracted/")
    parser.add_argument("pdfs", nargs="*", help="PDF file paths")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every PDF in style_samples/",
    )
    parser.add_argument(
        "--pages-per-chunk",
        type=int,
        default=8,
        help="Pages per markdown chunk (default: 8)",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="After extraction, re-index knowledge RAG",
    )
    args = parser.parse_args()

    paths = list(args.pdfs)
    if args.all:
        paths += glob.glob(os.path.join(STYLE_SAMPLES_DIR, "*.pdf"))

    if not paths:
        parser.error("Provide PDF path(s) or use --all")

    all_written = []
    for pdf in paths:
        print(f"Extracting: {pdf}")
        all_written.extend(digitize_pdf(pdf, pages_per_chunk=args.pages_per_chunk))

    print(f"\nDone. Wrote {len(all_written)} markdown chunk(s).")

    if args.index:
        from dotenv import load_dotenv

        load_dotenv()
        try:
            from core.rag import index_knowledge_collection, index_style_collection

            k = index_knowledge_collection()
            s = index_style_collection()
            print(f"RAG indexed: knowledge={k}, style={s} new document(s).")
        except Exception as exc:
            print(f"\nIndexing failed: {exc}")
            print("Fix deps: python -m pip install \"protobuf>=3.20,<4\"")
            print("Notes still work — primary reference loads from disk without ChromaDB.")
            sys.exit(1)


if __name__ == "__main__":
    main()
