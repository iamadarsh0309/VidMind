"""Structured notes representation (notes.json) and renderers."""

import json
import re
from typing import Optional

from pydantic import BaseModel, Field


class CodeBlock(BaseModel):
    language: str = "text"
    code: str
    valid: bool = True
    validation_note: Optional[str] = None


class NoteSection(BaseModel):
    heading: str
    content_md: str = ""
    code_blocks: list[CodeBlock] = Field(default_factory=list)
    interview_point: bool = False
    diagram_hint: Optional[str] = None


class NoteDocument(BaseModel):
    title: str
    topic: str = "software-engineering"
    introduction: str = ""
    sections: list[NoteSection] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)
    takeaways: list[str] = Field(default_factory=list)
    modules: list[str] = Field(default_factory=list)


def parse_markdown_to_document(title: str, notes_md: str, modules: list[str] | None = None) -> NoteDocument:
    """Best-effort parse of synthesized Markdown into NoteDocument."""
    lines = notes_md.splitlines()
    introduction = ""
    sections: list[NoteSection] = []
    glossary: dict[str, str] = {}
    takeaways: list[str] = []
    current_heading = ""
    current_lines: list[str] = []
    mode = "body"

    def _flush_section():
        nonlocal current_heading, current_lines
        if current_heading:
            content = "\n".join(current_lines).strip()
            code_blocks = _extract_code_blocks(content)
            sections.append(
                NoteSection(
                    heading=current_heading,
                    content_md=content,
                    code_blocks=code_blocks,
                    interview_point="interview" in current_heading.lower(),
                )
            )
        current_heading = ""
        current_lines = []

    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            continue
        if line.strip() == "## Glossary":
            _flush_section()
            mode = "glossary"
            continue
        if line.strip() == "## Key Takeaways":
            _flush_section()
            mode = "takeaways"
            continue
        if line.startswith("## Table of Contents"):
            mode = "toc"
            continue
        if mode == "toc":
            continue
        if mode == "glossary":
            m = re.match(r"^[-*]\s+\*\*(.+?)\*\*[:\s]+(.+)$", line.strip())
            if m:
                glossary[m.group(1)] = m.group(2)
            continue
        if mode == "takeaways":
            if line.strip().startswith(("-", "*")):
                takeaways.append(line.strip().lstrip("-* ").strip())
            continue
        if line.startswith("## "):
            _flush_section()
            current_heading = line[3:].strip()
            mode = "body"
            if not introduction and sections == [] and not current_heading:
                pass
            continue
        if not introduction and not current_heading and line.strip() and mode == "body":
            introduction += line + "\n"
            continue
        current_lines.append(line)

    _flush_section()
    intro_clean = introduction.strip()
    if sections and not intro_clean:
        intro_clean = sections[0].content_md[:400]

    return NoteDocument(
        title=title,
        introduction=intro_clean,
        sections=sections,
        glossary=glossary,
        takeaways=takeaways,
        modules=modules or [],
    )


def _extract_code_blocks(content: str) -> list[CodeBlock]:
    blocks = []
    for match in re.finditer(r"```(\w*)\n(.*?)```", content, re.DOTALL):
        lang = match.group(1) or "text"
        code = match.group(2).strip()
        blocks.append(CodeBlock(language=lang, code=code))
    return blocks


def document_to_markdown(doc: NoteDocument) -> str:
    parts = [f"# {doc.title}", "", doc.introduction.strip(), ""]

    if doc.modules:
        parts.append("## Course Modules")
        for m in doc.modules:
            parts.append(f"- {m}")
        parts.append("")

    parts.append("## Table of Contents")
    for sec in doc.sections:
        anchor = sec.heading.lower().replace(" ", "-")
        parts.append(f"- [{sec.heading}](#{anchor})")
    parts.append("")

    for sec in doc.sections:
        parts.append(f"## {sec.heading}")
        parts.append(sec.content_md)
        parts.append("")

    if doc.glossary:
        parts.append("## Glossary")
        for term, definition in doc.glossary.items():
            parts.append(f"- **{term}**: {definition}")
        parts.append("")

    if doc.takeaways:
        parts.append("## Key Takeaways")
        for t in doc.takeaways:
            parts.append(f"- {t}")

    return "\n".join(parts).strip() + "\n"


def save_notes_json(doc: NoteDocument, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(), f, indent=2, ensure_ascii=False)
    return path


def load_notes_json(path: str) -> NoteDocument:
    with open(path, encoding="utf-8") as f:
        return NoteDocument(**json.load(f))
