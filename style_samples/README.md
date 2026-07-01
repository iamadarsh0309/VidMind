# Style Samples

## Primary reference (canonical)

**`fastapi_tutorial.pdf`** is the main reference for how all notes should look — structure, depth, code examples, and flow.

```bash
# Extract + index (run once after adding/updating the PDF)
python scripts/digitize_pdf.py style_samples/fastapi_tutorial.pdf --index
```

Extracted chunks are written to `style_samples/extracted/fastapi_tutorial/` locally (gitignored). Run the command above after placing your PDF — they are **not** committed to the repo.

Configure in `.env`:
```
PRIMARY_REFERENCE=fastapi_tutorial
```

---

## Handwritten notes (optional, supplementary)

Place scans in `scans/` and run:

```bash
brew install tesseract
python scripts/digitize_notes.py style_samples/scans/*.jpg
```

Mark polished few-shot examples as `fewshot_*.md`.

---

## Other PDFs

Additional PDFs go in `style_samples/` and are indexed as supplementary knowledge:

```bash
python scripts/digitize_pdf.py style_samples/other_topic.pdf --index
```

They do **not** replace the primary reference unless you change `PRIMARY_REFERENCE` in `.env`.
