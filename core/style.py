"""Load and build personalized note-style prompts from style guide + few-shot samples."""

import glob
import os

from core.primary_reference import get_primary_anchor_excerpt, primary_reference_status

STYLE_DIR = os.path.join(os.path.dirname(__file__), "..", "style")
STYLE_GUIDE_PATH = os.path.join(STYLE_DIR, "style_guide.md")
STYLE_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "style_samples")
FEWSHOT_GLOB = os.path.join(STYLE_SAMPLES_DIR, "fewshot_*.md")

MAX_FEWSHOT_CHARS = 6000


def load_style_guide() -> str:
    if os.path.isfile(STYLE_GUIDE_PATH):
        with open(STYLE_GUIDE_PATH, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def load_fewshot_examples() -> str:
    paths = sorted(glob.glob(FEWSHOT_GLOB))
    if not paths:
        return ""

    parts = []
    total = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            continue
        block = f"### Example: {os.path.basename(path)}\n\n{content}"
        if total + len(block) > MAX_FEWSHOT_CHARS:
            break
        parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(parts)


def build_style_context() -> str:
    """Combined style guide + primary reference anchor + few-shot block."""
    sections = []
    status = primary_reference_status()
    if status["ready"]:
        sections.append(
            f"## Canonical reference document: `{status['name']}`\n\n"
            "All generated notes must match the structure, depth, code-example style, "
            "and section flow of this primary reference (FastAPI tutorial format).\n\n"
            + get_primary_anchor_excerpt()
        )

    guide = load_style_guide()
    if guide:
        sections.append("## Note-writing style guide\n\n" + guide)

    fewshot = load_fewshot_examples()
    if fewshot:
        sections.append(
            "## Supplementary format examples\n\n" + fewshot
        )

    return "\n\n".join(sections)


STYLE_EXTRACTION_PROMPT = """You are analyzing a user's handwritten study notes (digitized via OCR).

Produce a **style guide** in Markdown that another AI can follow to write notes in this person's voice and format.

Cover:
1. Heading depth and naming habits (## vs ###, title casing, topic labels)
2. Bullet vs numbered lists — when each is used
3. How jargon and acronyms are explained (parentheses, italics, inline definitions)
4. Abbreviations, symbols, emphasis patterns the author uses
5. Typical section order (e.g. concept → example → pitfall → recap)
6. How code snippets are presented (size, annotations, language tags)
7. Use of tables, blockquotes, callouts
8. What the author omits (e.g. never copies full paragraphs, skips filler)
9. Tone (terse vs verbose, first person or not, use of "Pitfall" / "Takeaway" labels)

Output only the style guide Markdown. No preamble."""


def extract_style_guide_from_samples(samples_text: str, llm) -> str:
    """One-time LLM pass to build style_guide.md from digitized notes."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", STYLE_EXTRACTION_PROMPT),
        ("human", "Digitized note samples:\n\n{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": samples_text[:12000]})
