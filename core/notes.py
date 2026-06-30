import os
import re

from markdown import markdown
from xhtml2pdf import pisa
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.code_validator import validate_code_in_text
from core.course_planner import CourseModule, attach_code_to_modules, plan_course
from core.llm import get_llm
from core.notes_schema import NoteDocument, document_to_markdown, parse_markdown_to_document, save_notes_json
from core.primary_reference import retrieve_primary_references
from core.rag import retrieve_style_references
from core.style import build_style_context
from core.transcriber import TranscriptSegment


NOTES_DIR = "notes/"
os.makedirs(NOTES_DIR, exist_ok=True)

DOMAIN_PROMPT = (
    "You write study notes exclusively for **software engineering, programming, "
    "and computer science** topics. Do not produce finance, medical, or unrelated content."
)

SECTION_NOTES_BASE_PROMPT = """You are an expert technical note-taker for software engineering content.

Your **primary reference** is the user's canonical tutorial document (FastAPI tutorial format).
Match its structure: concept explanation → runnable code example → output/behavior → validation rules or pitfalls.

Produce **deep, exhaustive Markdown study notes** for this section.

Rules:
- Output **only Markdown**. No preamble.
- Use `##` for major topics and `###` for sub-topics (no top-level `#`).
- Follow the primary reference's pattern: short intro paragraph, then Example, then code block, then explanation.
- Explain every technical term inline on first use (parentheses or italics).
- Include complete but minimal runnable code snippets like the reference (imports + app setup + endpoint).
- Use bullet lists for validation rules, operators, and step-by-step flows.
- Never invent content not in the transcript/code/primary reference.
- End dense sections with a brief recap when the reference does.

When on-screen code is provided:
- Merge OCR code with reference-style formatting.
- Prefer minimal runnable snippets over full file dumps."""


SYNTHESIZE_NOTES_BASE_PROMPT = """Merge section notes into one polished software engineering study document.

Structure:
# <Video Title>
Introduction (2-4 sentences)

## Table of Contents
## <Topics...>
## Glossary
## Key Takeaways

Preserve all detail. Deduplicate overlaps. Output only Markdown."""


PDF_CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.55; color: #1f2328; }
h1 { font-size: 24pt; color: #0b3d91; border-bottom: 2px solid #0b3d91; padding-bottom: 6pt; }
h2 { font-size: 16pt; color: #0b3d91; border-bottom: 1px solid #d0d7de; margin-top: 18pt; }
h3 { font-size: 13pt; color: #24292f; margin-top: 14pt; }
p { margin: 0 0 8pt 0; }
ul, ol { margin: 0 0 10pt 0; padding-left: 18pt; }
strong { color: #0b3d91; }
code { font-family: Courier, monospace; font-size: 9.5pt; background: #f6f8fa; color: #b91c1c; padding: 1pt 3pt; }
pre { font-family: Courier, monospace; font-size: 9.5pt; background: #f6f8fa; padding: 8pt; border: 1px solid #d0d7de; }
blockquote { border-left: 3px solid #0b3d91; background: #f6f8fa; padding: 6pt 10pt; }
table { border-collapse: collapse; width: 100%; font-size: 10pt; }
th { background: #0b3d91; color: white; padding: 5pt 7pt; }
td { padding: 5pt 7pt; border: 1px solid #d0d7de; }
"""


def _build_section_system_prompt(query_text: str = "") -> str:
    parts = [DOMAIN_PROMPT, SECTION_NOTES_BASE_PROMPT]
    style = build_style_context()
    if style:
        parts.append(style)
    if query_text:
        primary_refs = retrieve_primary_references(query_text, k=3)
        if primary_refs:
            parts.append(primary_refs)
        style_refs = retrieve_style_references(query_text, k=1)
        if style_refs and style_refs not in (primary_refs or ""):
            parts.append(style_refs)
    return "\n\n".join(parts)


def _build_synthesize_system_prompt() -> str:
    parts = [DOMAIN_PROMPT, SYNTHESIZE_NOTES_BASE_PROMPT]
    style = build_style_context()
    if style:
        parts.append("Match the user's note style:\n\n" + style)
    return "\n\n".join(parts)


def _split_transcript(transcript: str, chunk_size: int = 4000, overlap: int = 300) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(transcript)


def _section_notes_chain(query_text: str = ""):
    system = _build_section_system_prompt(query_text)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Transcript section:\n\n{text}\n{code_block}"),
    ])

    def _prepare(inputs: dict) -> dict:
        code = inputs.get("code", "").strip()
        code_block = ""
        if code:
            code_block = f"\nOn-screen code (OCR):\n```\n{code}\n```\n"
        return {"text": inputs["text"], "code_block": code_block}

    return RunnableLambda(_prepare) | prompt | get_llm(temperature=0.2, max_tokens=4096) | StrOutputParser()


def _synthesize_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", _build_synthesize_system_prompt()),
        ("human", "Video title: {title}\n\nSection notes (in order):\n\n{text}"),
    ])
    return prompt | get_llm(temperature=0.2, max_tokens=4096) | StrOutputParser()


def _generate_module_notes(module: CourseModule) -> str:
    section = {"text": module.transcript, "code": module.code}
    query = module.transcript[:500]
    chain = _section_notes_chain(query_text=query)
    header = f"<!-- module: {module.title} -->\n"
    return header + chain.invoke(section)


def generate_notes(
    transcript: str,
    title: str,
    enriched_chunks: list[dict] | None = None,
    segments: list[TranscriptSegment] | None = None,
    use_course_planner: bool = True,
) -> tuple[str, NoteDocument]:
    """Map-reduce notes generation. Returns (markdown, NoteDocument)."""
    module_titles: list[str] = []

    if use_course_planner:
        modules = plan_course(transcript, segments=segments)
        if enriched_chunks:
            modules = attach_code_to_modules(modules, enriched_chunks)
        module_titles = [m.title for m in modules]

        print(f"Generating notes for {len(modules)} module(s)...")
        section_notes = []
        for i, mod in enumerate(modules):
            print(f"  - Module {i + 1}/{len(modules)}: {mod.title}")
            section_notes.append(_generate_module_notes(mod))
    else:
        if enriched_chunks:
            sections = [{"text": c.get("text", ""), "code": c.get("code", "")} for c in enriched_chunks]
        else:
            sections = [{"text": t, "code": ""} for t in _split_transcript(transcript)]

        print(f"Generating notes for {len(sections)} section(s)...")
        section_notes = []
        for i, section in enumerate(sections):
            print(f"  - Section {i + 1}/{len(sections)}")
            query = section.get("text", "")[:500]
            chain = _section_notes_chain(query_text=query)
            section_notes.append(chain.invoke(section))

    combined = "\n\n---\n\n".join(section_notes)
    print("Synthesizing final document...")
    notes_md = _synthesize_chain().invoke({"title": title, "text": combined})
    notes_md = validate_code_in_text(notes_md)

    doc = parse_markdown_to_document(title, notes_md, modules=module_titles)
    return notes_md, doc


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\n\r\t]+', "", name).strip().strip(".")
    return name[:120] or "notes"


def _markdown_to_pdf(notes_md: str, output_path: str) -> None:
    body_html = markdown(notes_md, extensions=["fenced_code", "tables", "sane_lists", "toc"])
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{PDF_CSS}</style></head>
<body>{body_html}</body></html>"""
    with open(output_path, "wb") as f:
        result = pisa.CreatePDF(src=full_html, dest=f, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"xhtml2pdf failed with {result.err} error(s)")


def save_notes(
    notes_md: str,
    title: str,
    output_dir: str = NOTES_DIR,
    doc: NoteDocument | None = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    base = _sanitize_filename(title)
    md_path = os.path.join(output_dir, base + ".md")
    pdf_path = os.path.join(output_dir, base + ".pdf")
    json_path = os.path.join(output_dir, base + ".json")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(notes_md)

    if doc is None:
        doc = parse_markdown_to_document(title, notes_md)
    save_notes_json(doc, json_path)

    _markdown_to_pdf(notes_md, pdf_path)
    return {"md": md_path, "pdf": pdf_path, "json": json_path}


def create_notes(
    transcript: str,
    title: str,
    output_dir: str = NOTES_DIR,
    enriched_chunks: list[dict] | None = None,
    segments: list[TranscriptSegment] | None = None,
    use_course_planner: bool = True,
) -> dict:
    notes_md, doc = generate_notes(
        transcript,
        title,
        enriched_chunks=enriched_chunks,
        segments=segments,
        use_course_planner=use_course_planner,
    )
    paths = save_notes(notes_md, title, output_dir=output_dir, doc=doc)
    print(f"Notes saved:\n  - {paths['md']}\n  - {paths['pdf']}\n  - {paths['json']}")
    return paths
