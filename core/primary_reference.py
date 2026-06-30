"""
Primary reference document — canonical format for notes (content + structure).

Default: style_samples/fastapi_tutorial.pdf → style_samples/extracted/fastapi_tutorial/
"""

import glob
import os
import re

STYLE_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "style_samples")
EXTRACTED_DIR = os.path.join(STYLE_SAMPLES_DIR, "extracted")

PRIMARY_REFERENCE = os.getenv("PRIMARY_REFERENCE", "fastapi_tutorial")
PRIMARY_REFERENCE_PDF = os.getenv(
    "PRIMARY_REFERENCE_PDF",
    os.path.join(STYLE_SAMPLES_DIR, f"{PRIMARY_REFERENCE}.pdf"),
)
PRIMARY_EXTRACTED_DIR = os.path.join(EXTRACTED_DIR, PRIMARY_REFERENCE)
PRIMARY_REF_K = int(os.getenv("PRIMARY_REF_K", "3"))
PRIMARY_ANCHOR_CHARS = int(os.getenv("PRIMARY_ANCHOR_CHARS", "2500"))


def primary_reference_name() -> str:
    return PRIMARY_REFERENCE


def primary_pdf_path() -> str:
    return os.path.abspath(PRIMARY_REFERENCE_PDF)


def primary_extracted_dir() -> str:
    return os.path.abspath(PRIMARY_EXTRACTED_DIR)


def is_primary_source(path: str) -> bool:
    path = os.path.abspath(path)
    return primary_extracted_dir() in path or primary_pdf_path() == path


def load_primary_chunks() -> list[dict]:
    """Load all extracted markdown chunks for the primary reference."""
    chunk_dir = primary_extracted_dir()
    if not os.path.isdir(chunk_dir):
        return []

    chunks = []
    for path in sorted(glob.glob(os.path.join(chunk_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            chunks.append({
                "content": content,
                "source": os.path.abspath(path),
                "basename": os.path.basename(path),
            })
    return chunks


def _score_chunk(query: str, text: str) -> float:
    """Simple keyword overlap — works without ChromaDB."""
    stop = {
        "the", "a", "an", "and", "or", "to", "in", "on", "for", "of", "is", "it",
        "this", "that", "with", "as", "be", "are", "from", "by", "at",
    }
    q = {w for w in re.findall(r"[a-z0-9_]+", query.lower()) if len(w) > 2 and w not in stop}
    if not q:
        return 0.0
    t = text.lower()
    hits = sum(1 for w in q if w in t)
    return hits / len(q)


def retrieve_primary_references(query: str, k: int | None = None) -> str:
    """
    Retrieve top-k chunks from the primary reference by keyword relevance.
    Always available (disk-based, no embedding model required).
    """
    k = k or PRIMARY_REF_K
    chunks = load_primary_chunks()
    if not chunks:
        return ""

    scored = sorted(
        (( _score_chunk(query, c["content"]), c) for c in chunks),
        key=lambda x: x[0],
        reverse=True,
    )
    top = [c for score, c in scored if score > 0][:k]
    if not top:
        top = [c for _, c in scored[:k]]

    parts = []
    for i, chunk in enumerate(top, 1):
        excerpt = chunk["content"][:2000]
        parts.append(f"### Primary ref {i} ({chunk['basename']})\n\n{excerpt}")

    return (
        f"## PRIMARY REFERENCE — `{PRIMARY_REFERENCE}` (match this format, depth, and code style)\n\n"
        "This is the user's canonical notes document. Replicate its:\n"
        "- Section flow: concept → example → code → explanation → validation/pitfall\n"
        "- Code block size and annotation level\n"
        "- Heading style and bullet density\n"
        "- How definitions and API examples are presented\n\n"
        + "\n\n---\n\n".join(parts)
    )


def get_primary_anchor_excerpt() -> str:
    """Fixed anchor excerpt from the first substantive chunk for synthesize step."""
    chunks = load_primary_chunks()
    if not chunks:
        return ""

    for chunk in chunks:
        text = chunk["content"]
        if len(text) > 500 and ("Example" in text or "from fastapi" in text):
            return text[:PRIMARY_ANCHOR_CHARS]

    return chunks[0]["content"][:PRIMARY_ANCHOR_CHARS]


def primary_reference_status() -> dict:
    chunks = load_primary_chunks()
    return {
        "name": PRIMARY_REFERENCE,
        "pdf": primary_pdf_path(),
        "pdf_exists": os.path.isfile(primary_pdf_path()),
        "extracted_dir": primary_extracted_dir(),
        "chunk_count": len(chunks),
        "ready": len(chunks) > 0,
    }
