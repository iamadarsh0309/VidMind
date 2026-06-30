# Note Style Guide

**Primary reference:** `style_samples/fastapi_tutorial.pdf` (canonical format for all generated notes)

Generated notes must follow the FastAPI tutorial document's pattern:

## Structure (from primary reference)
- Topic title → short concept paragraph
- **Example** section with explanation before code
- Full runnable code block (`from fastapi import FastAPI` + decorator + handler)
- Explanation of URL behavior, parameters (path vs query), and expected output
- Validation rules as bullet lists (`gt`, `ge`, `lt`, `le` or equivalent)
- JSON error response examples when relevant

## Headings
- Numbered chapter-style topics in the reference map to `##` in generated notes
- Sub-concepts use `###`

## Code
- Complete minimal snippets with imports — not fragments
- Show decorator, function signature, and return value
- Include try/access URLs like `http://localhost:8000/docs` when the video mentions Swagger/OpenAPI

## Lists
- Bullet points for validation operators, prerequisites, audience
- Numbered steps only for sequential setup procedures

## Tone
- Tutorial voice: explanatory but concise
- Third person / instructional ("Enter the URL...", "It is possible to...")
- Define terms on first use (path parameter, query parameter, etc.)

## What to omit
- Copyright/disclaimer boilerplate from the reference PDF
- Verbatim transcript filler
- Content unrelated to software engineering
