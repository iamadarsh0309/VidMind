"""Save/resume pipeline progress (per-chunk transcription + post-transcript checkpoint)."""

import hashlib
import json
import os
import re
from datetime import datetime, timezone

from core.transcriber import TranscriptSegment, TranscriptionResult

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "downloads", "checkpoints")
TRANSCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "downloads", "transcripts")
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "downloads")


def sort_chunks_natural(paths: list[str]) -> list[str]:
    def key(p: str) -> int:
        m = re.search(r"_chunk_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    return sorted((os.path.abspath(p) for p in paths), key=key)


def session_key_from_chunks(chunks: list[str]) -> str:
    """Stable id from audio chunk filenames (same for URL or local wav path)."""
    if not chunks:
        return "session"
    first = os.path.basename(chunks[0])
    if "_chunk_" in first:
        return first.rsplit("_chunk_", 1)[0]
    return os.path.splitext(first)[0]


def _slug(key: str) -> str:
    base = key.strip().rstrip("/")
    if len(base) > 80:
        base = hashlib.md5(base.encode()).hexdigest()[:12]
    return re.sub(r"[^\w\-]+", "_", base)[:80] or "session"


def checkpoint_path_for_key(session_key: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, _slug(session_key) + ".json")


def transcript_txt_path(session_key: str) -> str:
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    return os.path.join(TRANSCRIPT_DIR, _slug(session_key) + ".txt")


def _segment_to_dict(seg: TranscriptSegment) -> dict:
    return {"start": seg.start, "end": seg.end, "text": seg.text}


def _segment_from_dict(d: dict) -> TranscriptSegment:
    return TranscriptSegment(start=d["start"], end=d["end"], text=d["text"])


def save_chunk_progress(
    session_key: str,
    source: str,
    language: str,
    chunks: list[str],
    completed_through: int,
    segments: list[TranscriptSegment],
    text_parts: list[str],
) -> str:
    existing = find_checkpoint_for_chunks(chunks)
    path = existing["_path"] if existing else checkpoint_path_for_key(session_key)
    text = " ".join(text_parts).strip()
    status = "transcribing" if completed_through < len(chunks) - 1 else "transcribed"

    data = {
        "session_key": session_key,
        "source": source,
        "language": language,
        "chunks": sort_chunks_natural(chunks),
        "completed_chunk_index": completed_through,
        "text_parts": text_parts,
        "text": text,
        "segments": [_segment_to_dict(s) for s in segments],
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(transcript_txt_path(session_key), "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

    return path


def save_transcribed(
    session_key: str,
    source: str,
    language: str,
    transcription: TranscriptionResult,
    chunks: list[str] | None = None,
) -> str:
    existing = find_checkpoint_for_chunks(chunks) if chunks else None
    path = existing["_path"] if existing else checkpoint_path_for_key(session_key)

    prior: dict = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            prior = json.load(f)

    data = {
        **prior,
        "session_key": session_key,
        "source": source,
        "language": language,
        "text": transcription.text,
        "segments": [_segment_to_dict(s) for s in transcription.segments],
        "status": prior.get("status") if prior.get("notes_progress") else "transcribed",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if chunks:
        data["chunks"] = sort_chunks_natural(chunks)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(transcript_txt_path(session_key), "w", encoding="utf-8") as f:
        f.write(transcription.text)
        f.write("\n")

    print(f"Checkpoint saved: {path}")
    return path


def save_notes_progress(
    checkpoint_path: str,
    *,
    modules: list[dict] | None = None,
    section_notes: list[str] | None = None,
    completed_module_index: int | None = None,
    notes_md: str | None = None,
    status: str = "notes_generating",
) -> None:
    """Persist per-module notes progress into the session checkpoint."""
    path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    progress = data.get("notes_progress", {})
    if modules is not None:
        progress["modules"] = modules
    if section_notes is not None:
        progress["section_notes"] = section_notes
    if completed_module_index is not None:
        progress["completed_module_index"] = completed_module_index
    if notes_md is not None:
        progress["notes_md"] = notes_md

    progress["status"] = status
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["notes_progress"] = progress
    data["status"] = status
    data["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_notes_progress(checkpoint_path: str) -> dict | None:
    data = load_checkpoint(path=checkpoint_path)
    if not data:
        return None
    return data.get("notes_progress")


def resolve_session_key(data: dict, source: str = "") -> str:
    if data.get("session_key"):
        return data["session_key"]
    if data.get("chunks"):
        return session_key_from_chunks(data["chunks"])
    return _slug(source or data.get("source", "session"))


def _chunks_match(a: list[str], b: list[str]) -> bool:
    return sort_chunks_natural(a) == sort_chunks_natural(b)


def find_checkpoint_for_chunks(chunks: list[str]) -> dict | None:
    """Find checkpoint matching these audio chunk paths."""
    if not chunks:
        return None

    abs_chunks = sort_chunks_natural(chunks)

    key = session_key_from_chunks(chunks)
    path = checkpoint_path_for_key(key)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if sort_chunks_natural(data.get("chunks", [])) == abs_chunks:
            data["_path"] = path
            return data

    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    for name in os.listdir(CHECKPOINT_DIR):
        if not name.endswith(".json"):
            continue
        full = os.path.join(CHECKPOINT_DIR, name)
        with open(full, encoding="utf-8") as f:
            data = json.load(f)
        if sort_chunks_natural(data.get("chunks", [])) == abs_chunks:
            data["_path"] = full
            return data
    return None


def load_checkpoint(source: str | None = None, path: str | None = None) -> dict | None:
    if path:
        p = os.path.abspath(path)
    elif source:
        p = checkpoint_path_for_key(source)
    else:
        return None
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def list_checkpoints() -> list[dict]:
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    out = []
    for name in sorted(os.listdir(CHECKPOINT_DIR)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(CHECKPOINT_DIR, name), encoding="utf-8") as f:
            data = json.load(f)
        data["_path"] = os.path.join(CHECKPOINT_DIR, name)
        out.append(data)
    return out


def transcription_from_checkpoint(data: dict) -> TranscriptionResult | None:
    if data.get("text"):
        text = data["text"]
    elif data.get("text_parts"):
        text = " ".join(data["text_parts"]).strip()
    else:
        return None
    if not text:
        return None
    segments = [_segment_from_dict(s) for s in data.get("segments", [])]
    return TranscriptionResult(text=text, segments=segments)


def is_fully_transcribed(data: dict, num_chunks: int) -> bool:
    if data.get("status") == "transcribed" and data.get("text"):
        return True
    if data.get("completed_chunk_index", -1) >= num_chunks - 1 and data.get("text"):
        return True
    return False
