# VidMind

**Personalized, code-aware video notes for software engineering tutorials.**

Paste a YouTube URL (or local video file) and get study notes in **your** format — structured like your reference docs, with on-screen code captured from coding tutorials, exported as Markdown, PDF, and JSON.

Free to run locally on Apple Silicon (M-series) with Ollama + Whisper.

[![GitHub](https://img.shields.io/github/stars/iamadarsh0309/ai-video-agent?style=social)](https://github.com/iamadarsh0309/ai-video-agent)

---

## Features

| Feature | Description |
|---------|-------------|
| **Transcription** | Local Whisper (English) or Sarvam API (Hinglish) with timestamps |
| **Code capture** | Video frame sampling + OCR + optional Ollama vision (`moondream`) |
| **Primary reference** | Match notes to your canonical PDF (default: FastAPI tutorial) |
| **Course planner** | Split long courses into modules before note generation |
| **Style RAG** | ChromaDB retrieval for writing patterns and few-shot examples |
| **Knowledge RAG** | Topic content from PDF extracts and approved notes |
| **Feedback loop** | Edit notes → re-index → improve future generations |
| **Exports** | `.md`, `.pdf`, `.json` (`notes.json` schema) |
| **UI** | Streamlit app with download and feedback tabs |

---

## Architecture

```
YouTube / video file
        │
        ├── Whisper (transcript + timestamps)
        ├── ffmpeg frames → OCR / VLM (on-screen code)
        │
        ▼
Course planner (long videos)
        │
        ▼
Note generator (Ollama / Mistral)
   ↑          ↑
Style RAG   Primary reference (fastapi_tutorial.pdf)
   ↑
Knowledge RAG
        │
        ▼
notes.json → Markdown → PDF
```

---

## Requirements

### System (macOS)

```bash
brew install ollama ffmpeg tesseract
ollama pull mistral
ollama pull moondream   # optional, better IDE code OCR
```

### Python

- Python 3.11+ (tested on 3.13)
- [uv](https://github.com/astral-sh/uv) or `pip`

---

## Installation

```bash
git clone https://github.com/iamadarsh0309/ai-video-agent.git
cd ai-video-agent

# Create venv and install deps
uv venv
uv pip install -r requirements.txt --python .venv/bin/python

# Or with pip
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

cp .env.example .env
# Edit .env if using Mistral API or Sarvam
```

### Fix ChromaDB / protobuf (Python 3.13)

If indexing fails with a protobuf descriptor error:

```bash
python scripts/fix_chroma_deps.py
# or
python -m pip install "protobuf>=3.20,<4"
```

The app sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` automatically.

---

## Primary reference (your note format)

Place your canonical reference PDF in `style_samples/` (e.g. `fastapi_tutorial.pdf`), then:

```bash
python scripts/digitize_pdf.py style_samples/fastapi_tutorial.pdf --index
```

This extracts text to `style_samples/extracted/` and indexes it for RAG.  
Configure in `.env`:

```env
PRIMARY_REFERENCE=fastapi_tutorial
```

Generated notes follow that document's structure: **concept → example → code → explanation → validation rules**.

See [`style_samples/README.md`](style_samples/README.md) for handwritten notes and additional PDFs.

---

## Usage

### Streamlit UI (recommended)

```bash
source .venv/bin/activate
streamlit run app.py
```

Paste a YouTube URL, enable **Capture on-screen code** for tutorials, click **Generate notes**.

### CLI

```bash
python test.py
```

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/digitize_pdf.py` | Extract PDF → markdown chunks + optional RAG index |
| `scripts/digitize_notes.py` | OCR handwritten scans → `style_samples/*.md` |
| `scripts/extract_style_guide.py` | LLM extracts `style/style_guide.md` from primary ref |
| `scripts/submit_feedback.py` | Index your edited notes for learning |
| `scripts/fix_chroma_deps.py` | Fix protobuf/ChromaDB on Python 3.13 |

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `mistral` | Local LLM via Ollama |
| `MISTRAL_API_KEY` | — | Optional; uses Mistral API instead of Ollama |
| `WHISPER_MODEL` | `small` | Whisper model size |
| `PRIMARY_REFERENCE` | `fastapi_tutorial` | Canonical notes PDF stem name |
| `FRAME_INTERVAL_SECONDS` | `45` | Video frame sampling interval |
| `COURSE_PLANNER_THRESHOLD` | `12000` | Chars before course planner activates |

Full list: [`.env.example`](.env.example)

---

## Project structure

```
├── app.py                 # Streamlit UI
├── test.py                # CLI pipeline runner
├── core/
│   ├── pipeline.py        # End-to-end orchestration
│   ├── transcriber.py     # Whisper / Sarvam
│   ├── notes.py           # Map-reduce note generation
│   ├── primary_reference.py
│   ├── rag.py             # ChromaDB style + knowledge
│   ├── course_planner.py
│   └── ...
├── utils/
│   ├── audio_processor.py
│   ├── video_processor.py # Frame OCR + VLM
│   └── pdf_extractor.py
├── style_samples/         # Reference PDFs, few-shot examples
├── style/                 # style_guide.md
└── scripts/
```

---

## LLM backends

| Backend | Cost | When |
|---------|------|------|
| **Ollama** (`mistral`) | Free, local | Default — no API key needed |
| **Mistral API** | Paid | Set `MISTRAL_API_KEY` in `.env` |

---

## Troubleshooting

**`ModuleNotFoundError: langchain_ollama`**
```bash
uv pip install langchain-ollama --python .venv/bin/python
```

**ChromaDB protobuf error** — run `python scripts/fix_chroma_deps.py`

**`pip: command not found`** — use `python -m pip` or `uv pip` with your venv Python

**Notes work without ChromaDB** — primary reference loads from disk in `style_samples/extracted/`

**Domain check failed** — disable "Software engineering only" in Streamlit sidebar for non-CS videos

---

## License

MIT (add a LICENSE file if you intend to open-source formally)

## Author

[Adarsh Pandey](https://github.com/iamadarsh0309)
