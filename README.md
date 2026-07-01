# VidMind

**Turn software-engineering video tutorials into personalized study notes — in your format.**

Paste a YouTube URL or local video file. VidMind transcribes the audio, optionally captures on-screen code, and generates structured notes (Markdown, PDF, JSON) styled after your reference documents.

Runs **free on Apple Silicon** with Ollama + Whisper. No API keys required for the default setup.

[![GitHub](https://img.shields.io/github/stars/iamadarsh0309/ai-video-agent?style=social)](https://github.com/iamadarsh0309/ai-video-agent)

---

## What it does

| Step | What happens |
|------|----------------|
| 1. **Ingest** | Download YouTube audio/video or read a local file |
| 2. **Transcribe** | Whisper (English) or Sarvam (Hinglish) with timestamps |
| 3. **Capture code** | Sample video frames → OCR / vision model for IDE screenshots |
| 4. **Plan** | Split long courses into logical modules |
| 5. **Generate** | Map-reduce LLM notes using your style + primary reference |
| 6. **Export** | `.md`, `.pdf`, and structured `notes.json` |

---

## Quick start (5 minutes)

### 1. Install system dependencies (macOS)

```bash
brew install ollama ffmpeg tesseract deno
ollama pull mistral
```

- **ollama** — local LLM (free)
- **ffmpeg** — audio/video processing
- **tesseract** — OCR for on-screen code
- **deno** — JavaScript runtime for YouTube downloads (yt-dlp)

### 2. Install Python dependencies

```bash
git clone https://github.com/iamadarsh0309/ai-video-agent.git
cd ai-video-agent

uv venv && uv pip install -r requirements.txt --python .venv/bin/python
# or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

cp .env.example .env
```

### 3. Run the app

```bash
source .venv/bin/activate
streamlit run app.py
```

Paste a YouTube URL → click **Generate notes** → download from the **Downloads** tab.

---

## Configuration (`.env`)

Copy `.env.example` to `.env`. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `auto` | `ollama` (free, local), `mistral` (API), or `auto` (Mistral with Ollama fallback) |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |
| `MISTRAL_API_KEY` | — | Optional; leave empty to use Ollama only |
| `WHISPER_MODEL` | `small` | `tiny` / `base` / `small` / `medium` (larger = slower, more accurate) |
| `PRIMARY_REFERENCE` | `fastapi_tutorial` | Your canonical notes format (see below) |
| `FRAME_INTERVAL_SECONDS` | `45` | How often to sample video frames for code OCR |
| `COURSE_PLANNER_THRESHOLD` | `12000` | Transcript length (chars) before module splitting |

**Recommended for long videos (3+ hours):**

```env
LLM_PROVIDER=ollama
WHISPER_MODEL=small
```

---

## Using the Streamlit UI

```bash
streamlit run app.py
```

### Sidebar options

| Option | When to use |
|--------|-------------|
| **Transcription language** | `english` → Whisper. `hinglish` → Sarvam API (needs `SARVAM_API_KEY`) |
| **Capture on-screen code** | ON for coding tutorials. OFF for talks/slides (faster) |
| **Course planner** | ON for videos > ~20 min. Splits into modules before note generation |
| **Software engineering only** | OFF for general tech talks (e.g. LLM deep dives) |
| **Frame interval** | Lower = more frames, better code capture, slower |

### Output tabs

- **Notes** — generated Markdown study notes
- **Summary** — high-level video summary
- **Transcript** — full transcript text
- **Downloads** — `.md`, `.pdf`, `.json`
- **Feedback** — edit notes and re-index to improve future generations

---

## CLI usage

### Basic pipeline

Edit `test.py` with your URL, then:

```bash
python test.py
```

### Resume after interruption

Long videos are checkpointed automatically. If the run stops (crash, rate limit, Ctrl+C):

```bash
# List saved sessions
python scripts/resume_pipeline.py --list

# Resume latest (skip re-transcription)
python scripts/resume_pipeline.py --latest --no-code

# Resume a specific checkpoint
python scripts/resume_pipeline.py downloads/checkpoints/<file>.json --no-code
```

**Flags:**

| Flag | Purpose |
|------|---------|
| `--latest` | Use most recent checkpoint |
| `--no-code` | Skip video frame OCR (recommended for non-coding videos) |
| `--no-domain` | Skip software-engineering content filter |
| `--reuse-chunks` | Reuse existing `downloads/*_chunk_*.wav` files |

Checkpoints are saved in `downloads/checkpoints/`. Transcripts are also written to `downloads/transcripts/`.

---

## Personalizing note style

VidMind matches **your** note format using a primary reference document and optional style samples.

### Primary reference (required for best results)

1. Place your reference PDF in `style_samples/` (e.g. `fastapi_tutorial.pdf`)
2. Extract and index it:

```bash
python scripts/digitize_pdf.py style_samples/fastapi_tutorial.pdf --index
```

3. Set in `.env`:

```env
PRIMARY_REFERENCE=fastapi_tutorial
```

Generated notes follow: **concept → example → code → explanation → validation rules**.

> Notes work without ChromaDB — primary reference chunks load from disk in `style_samples/extracted/`.

### Handwritten notes (optional)

```bash
# OCR scans → markdown
python scripts/digitize_notes.py style_samples/scans/*.jpg

# Extract writing style guide
python scripts/extract_style_guide.py
```

See [`style_samples/README.md`](style_samples/README.md) for details.

### Feedback loop

Edit generated notes in the Streamlit **Feedback** tab, or:

```bash
python scripts/submit_feedback.py notes/my_edited_notes.md --in-place
```

Approved edits are indexed into the knowledge RAG collection for future runs.

---

## Scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/digitize_pdf.py` | PDF → markdown chunks + optional ChromaDB index |
| `scripts/digitize_notes.py` | Handwritten scan OCR → `style_samples/*.md` |
| `scripts/extract_style_guide.py` | LLM extracts `style/style_guide.md` from samples |
| `scripts/submit_feedback.py` | Index edited notes for learning |
| `scripts/resume_pipeline.py` | Resume from checkpoint (transcription or notes) |
| `scripts/fix_chroma_deps.py` | Fix protobuf/ChromaDB on Python 3.13 |

---

## Architecture

```
YouTube URL / local file
        │
        ├── audio_processor.py   → download, convert, chunk (10 min WAVs)
        ├── transcriber.py       → Whisper / Sarvam → timestamped segments
        └── video_processor.py   → ffmpeg frames → OCR / VLM (moondream)
        │
        ▼
   checkpoint.py               → save/resume per-chunk transcription + notes
        │
        ▼
   course_planner.py            → split long transcripts into modules
        │
        ▼
   notes.py (map-reduce)
        ├── primary_reference.py → disk-based retrieval (no DB required)
        ├── rag.py               → ChromaDB style + knowledge collections
        └── style.py             → style guide + few-shot examples
        │
        ▼
   notes.json → Markdown → PDF (xhtml2pdf)
```

---

## Project structure

```
├── app.py                      # Streamlit UI
├── test.py                     # CLI runner
├── core/
│   ├── pipeline.py             # End-to-end orchestration
│   ├── transcriber.py          # Whisper + Sarvam dual-engine
│   ├── checkpoint.py           # Resume transcription & notes
│   ├── notes.py                # Map-reduce note generation
│   ├── course_planner.py       # Module splitting for long videos
│   ├── code_aligner.py         # Align OCR code to transcript timestamps
│   ├── primary_reference.py    # Canonical reference retrieval
│   ├── rag.py                  # ChromaDB style + knowledge RAG
│   ├── llm.py                  # Ollama / Mistral factory with fallback
│   └── ...
├── utils/
│   ├── audio_processor.py      # YouTube download + WAV chunking
│   ├── video_processor.py      # Frame sampling + OCR + VLM
│   └── pdf_extractor.py        # PDF text extraction
├── style_samples/              # Reference PDFs, few-shot examples
├── style/                      # Extracted style guide
└── scripts/                    # CLI utilities
```

---

## LLM backends

| Backend | Cost | Setup |
|---------|------|-------|
| **Ollama** | Free, local | `brew install ollama && ollama pull mistral` |
| **Mistral API** | Paid, rate-limited | Set `MISTRAL_API_KEY` in `.env` |

For videos with 10+ LLM calls (course planner + modules + synthesis), use:

```env
LLM_PROVIDER=ollama
```

With `LLM_PROVIDER=auto`, Mistral is tried first and Ollama is used automatically on rate-limit errors.

---

## Troubleshooting

### Installation

**`ModuleNotFoundError: langchain_ollama`**
```bash
uv pip install langchain-ollama --python .venv/bin/python
```

**ChromaDB protobuf error (Python 3.13)**
```bash
python scripts/fix_chroma_deps.py
```

**YouTube download warning (No JavaScript runtime)**
```bash
brew install deno
```

### Runtime

**Whisper `FP16 is not supported on CPU`** — harmless on Mac Air; transcription still works.

**Mistral `429 Rate limit exceeded`** — set `LLM_PROVIDER=ollama` in `.env`, or wait and resume:
```bash
python scripts/resume_pipeline.py --latest --no-code
```

**OCR / code capture fails**
```bash
brew install tesseract
uv pip install pytesseract Pillow --python .venv/bin/python
```
Or disable code capture in the sidebar / use `--no-code`.

**Domain check failed** — uncheck "Software engineering only" for general tech talks.

**Streamlit log spam** — `.streamlit/config.toml` disables file watching. Install `torchvision` if needed:
```bash
uv pip install torchvision --python .venv/bin/python
```

**Re-transcribing on resume** — ensure checkpoint exists:
```bash
python scripts/resume_pipeline.py --list
```
Use `--latest` to load the saved transcript without re-running Whisper.

---

## Requirements

- **Python** 3.11+ (tested on 3.13)
- **macOS** Apple Silicon recommended (M-series)
- **Disk** ~2 GB for Whisper model + Ollama model
- **RAM** 8 GB minimum; 16 GB recommended for long videos

---

## License

MIT

## Author

[Adarsh Pandey](https://github.com/iamadarsh0309)
