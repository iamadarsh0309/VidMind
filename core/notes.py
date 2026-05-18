import os
import re

from markdown import markdown
from xhtml2pdf import pisa
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter


NOTES_DIR = "notes/"
os.makedirs(NOTES_DIR, exist_ok=True)


SECTION_NOTES_SYSTEM_PROMPT = """You are an expert technical note-taker. You will receive one section of a video transcript.

Produce **deep, exhaustive Markdown study notes** for this section. The goal is that a reader who has never seen the video can fully understand and learn from the notes alone.

Rules:
- Output **only Markdown**. No preamble. No closing remarks. No "Here are the notes".
- Use `##` for major topics in this section and `###` for sub-topics. Do **not** use a top-level `#` heading — that is reserved for the final document.
- Cover every concept, claim, example, command, name, number, version, and reference that appears in the section. Do not summarize away detail.
- **Every technical term, acronym, tool, framework, protocol, or piece of jargon must be explained inline the first time it appears**, in parentheses or italics. Example: "They route traffic via *Envoy (a high-performance L7 proxy from Lyft)* to ..."
- Use fenced code blocks (```lang) for any code, commands, configs, or URLs.
- Use tables when comparing options or listing structured pairs (name → meaning, before → after).
- Use bullet lists for steps, properties, and enumerations. Use short paragraphs for explanations.
- Preserve the speaker's order of ideas. Quote distinctive phrases sparingly in `*italics*` when they aid memory.
- Never invent content. If something is unclear in the transcript, omit it.
- Be thorough. It is fine for one section's notes to be long if the section is dense."""


SYNTHESIZE_NOTES_SYSTEM_PROMPT = """You are an expert technical editor. You will receive several Markdown note sections, each generated from a consecutive part of the same video transcript, in order.

Merge them into a single, polished, **comprehensive Markdown study document**.

Output structure (use exactly these levels):

# <Video Title>
A 2-4 sentence introduction describing what the video covers, who it is for, and the main themes.

## Table of Contents
A bulleted list linking to each major section using Markdown anchors.

## <First major topic>
... full notes ...

## <Second major topic>
... full notes ...

(continue for all topics)

## Glossary
A definition list for every technical term, acronym, tool, and framework mentioned in the document. One short, clear definition per item.

## Key Takeaways
A bulleted list of 6-12 things a reader should walk away knowing or able to do.

Rules:
- Preserve **all** content from the input sections. Do not compress, summarize, or drop detail — your job is to organize, deduplicate, and polish, not to shorten.
- Merge overlapping or repeated points across sections into one place.
- Keep every inline explanation of jargon. Add more if any terms are still unexplained.
- Use fenced code blocks, tables, and bullets as appropriate. Keep paragraphs short.
- Output **only Markdown**. No preamble."""


PDF_CSS = """
@page {
    size: A4;
    margin: 2cm 2cm 2cm 2cm;
    @frame footer {
        -pdf-frame-content: footerContent;
        bottom: 1cm;
        margin-left: 2cm;
        margin-right: 2cm;
        height: 1cm;
    }
}

body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1f2328;
}

h1 {
    font-size: 24pt;
    color: #0b3d91;
    margin-top: 0;
    margin-bottom: 16pt;
    padding-bottom: 6pt;
    border-bottom: 2px solid #0b3d91;
}

h2 {
    font-size: 16pt;
    color: #0b3d91;
    margin-top: 18pt;
    margin-bottom: 8pt;
    padding-bottom: 3pt;
    border-bottom: 1px solid #d0d7de;
}

h3 {
    font-size: 13pt;
    color: #24292f;
    margin-top: 14pt;
    margin-bottom: 6pt;
}

h4 {
    font-size: 11.5pt;
    color: #24292f;
    margin-top: 10pt;
    margin-bottom: 4pt;
}

p {
    margin: 0 0 8pt 0;
    text-align: justify;
}

ul, ol {
    margin: 0 0 10pt 0;
    padding-left: 18pt;
}

li {
    margin-bottom: 4pt;
}

strong { color: #0b3d91; }
em { color: #57606a; }

code {
    font-family: Courier, monospace;
    font-size: 9.5pt;
    background-color: #f6f8fa;
    color: #b91c1c;
    padding: 1pt 3pt;
    border-radius: 3pt;
}

pre {
    font-family: Courier, monospace;
    font-size: 9.5pt;
    background-color: #f6f8fa;
    color: #1f2328;
    padding: 8pt;
    border: 1px solid #d0d7de;
    border-radius: 4pt;
    margin: 8pt 0;
}

pre code {
    background-color: transparent;
    color: #1f2328;
    padding: 0;
}

blockquote {
    border-left: 3px solid #0b3d91;
    background-color: #f6f8fa;
    color: #57606a;
    padding: 6pt 10pt;
    margin: 8pt 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 10pt 0;
    font-size: 10pt;
}

th {
    background-color: #0b3d91;
    color: white;
    padding: 5pt 7pt;
    text-align: left;
    border: 1px solid #0b3d91;
}

td {
    padding: 5pt 7pt;
    border: 1px solid #d0d7de;
    vertical-align: top;
}

a { color: #0b3d91; text-decoration: none; }

hr {
    border: 0;
    border-top: 1px solid #d0d7de;
    margin: 12pt 0;
}
"""


def _llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
        max_tokens=4096,
    )


def _split_transcript(transcript: str, chunk_size: int = 4000, overlap: int = 300) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(transcript)


def _section_notes_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SECTION_NOTES_SYSTEM_PROMPT),
        ("human", "Transcript section:\n\n{text}"),
    ])
    return RunnableLambda(lambda x: {"text": x}) | prompt | _llm() | StrOutputParser()


def _synthesize_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIZE_NOTES_SYSTEM_PROMPT),
        ("human", "Video title: {title}\n\nSection notes (in order):\n\n{text}"),
    ])
    return prompt | _llm() | StrOutputParser()


def generate_notes(transcript: str, title: str) -> str:
    """Map-reduce: per-section deep notes -> synthesized final Markdown document."""
    sections = _split_transcript(transcript)
    print(f"Generating notes for {len(sections)} section(s)...")

    section_chain = _section_notes_chain()
    section_notes = []
    for i, section in enumerate(sections):
        print(f"  - Section {i + 1}/{len(sections)}")
        section_notes.append(section_chain.invoke(section))

    combined = "\n\n---\n\n".join(section_notes)
    print("Synthesizing final document...")
    return _synthesize_chain().invoke({"title": title, "text": combined})


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\n\r\t]+', "", name).strip().strip(".")
    return name[:120] or "notes"


def _markdown_to_pdf(notes_md: str, output_path: str) -> None:
    body_html = markdown(
        notes_md,
        extensions=["fenced_code", "tables", "sane_lists", "toc"],
    )
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>{PDF_CSS}</style>
</head>
<body>
{body_html}
<div id="footerContent" style="font-size:8pt; color:#57606a; text-align:center;">
    Page <pdf:pagenumber> of <pdf:pagecount>
</div>
</body>
</html>"""

    with open(output_path, "wb") as f:
        result = pisa.CreatePDF(src=full_html, dest=f, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"xhtml2pdf failed with {result.err} error(s)")


def save_notes(notes_md: str, title: str, output_dir: str = NOTES_DIR) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    base = _sanitize_filename(title)
    md_path = os.path.join(output_dir, base + ".md")
    pdf_path = os.path.join(output_dir, base + ".pdf")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(notes_md)

    _markdown_to_pdf(notes_md, pdf_path)
    return {"md": md_path, "pdf": pdf_path}


def create_notes(transcript: str, title: str, output_dir: str = NOTES_DIR) -> dict:
    notes_md = generate_notes(transcript, title)
    paths = save_notes(notes_md, title, output_dir=output_dir)
    print(f"Notes saved:\n  - {paths['md']}\n  - {paths['pdf']}")
    return paths
