"""Feedback loop: user-edited notes improve future generations."""

import os
import shutil
from datetime import datetime

from core.rag import index_approved_note, index_knowledge_collection, index_style_collection

APPROVED_DIR = os.path.join(os.path.dirname(__file__), "..", "notes", "approved")


def submit_feedback(edited_md: str, title: str, source_path: str | None = None) -> str:
    """
    Save user-edited notes and re-index into knowledge RAG.
    Returns path to saved approved note.
    """
    os.makedirs(APPROVED_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:60]
    filename = f"{safe_title}_{timestamp}.md"
    out_path = os.path.join(APPROVED_DIR, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(edited_md.strip())
        f.write("\n")

    if source_path and os.path.isfile(source_path):
        shutil.copy2(source_path, out_path + ".orig")

    index_approved_note(out_path)
    index_knowledge_collection()
    print(f"Feedback saved and indexed: {out_path}")
    return out_path


def submit_feedback_from_file(path: str) -> str:
    """Index an existing edited markdown file as approved feedback."""
    path = os.path.abspath(path)
    index_approved_note(path)
    index_knowledge_collection()
    return path
