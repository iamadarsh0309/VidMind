import os
import requests
import whisper
from dataclasses import dataclass, field
from pydub import AudioSegment

# Sarvam's sync STT-translate API rejects audio longer than 30s.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
  text: str
  segments: list[TranscriptSegment] = field(default_factory=list)


def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def _chunk_offset_seconds(chunk_path: str, chunk_index: int, chunk_minutes: int = 10) -> float:
    """Estimate absolute start time for a chunk from its index."""
    return chunk_index * chunk_minutes * 60


def transcribe_chunk_whisper(chunk_path: str, time_offset: float = 0.0) -> TranscriptionResult:
    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe")

    segments = []
    for seg in result.get("segments", []):
        segments.append(
            TranscriptSegment(
                start=time_offset + float(seg["start"]),
                end=time_offset + float(seg["end"]),
                text=seg["text"].strip(),
            )
        )

    if not segments and result.get("text"):
        segments.append(
            TranscriptSegment(start=time_offset, end=time_offset, text=result["text"].strip())
        )

    return TranscriptionResult(text=result["text"].strip(), segments=segments)


def _send_to_sarvam(piece_path: str) -> str:
    headers = {"api-subscription-key": SARVAM_API_KEY}
    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\nSarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str, time_offset: float = 0.0) -> TranscriptionResult:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start_ms in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start_ms: start_ms + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  -> Sarvam piece {i + 1}/{total_pieces} ...")
            piece_text = _send_to_sarvam(piece_path)
            full_text += piece_text + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    text = full_text.strip()
    piece_duration = SARVAM_PIECE_SECONDS
    segments = []
    # Sarvam has no per-segment timestamps; approximate one segment per piece
    for i in range(total_pieces):
        start = time_offset + i * piece_duration
        end = time_offset + min((i + 1) * piece_duration, len(audio) / 1000.0)
        segments.append(TranscriptSegment(start=start, end=end, text=""))

    if segments:
        segments[-1] = TranscriptSegment(
            start=segments[0].start,
            end=time_offset + len(audio) / 1000.0,
            text=text,
        )
    elif text:
        segments.append(
            TranscriptSegment(start=time_offset, end=time_offset + len(audio) / 1000.0, text=text)
        )

    return TranscriptionResult(text=text, segments=segments)


def transcribe_chunk(
    chunk_path: str,
    language: str = "english",
    time_offset: float = 0.0,
) -> TranscriptionResult:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path, time_offset=time_offset)
    return transcribe_chunk_whisper(chunk_path, time_offset=time_offset)


def transcribe_all(
    chunks: list,
    language: str = "english",
    chunk_minutes: int = 10,
) -> TranscriptionResult:
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    all_segments: list[TranscriptSegment] = []
    full_text_parts: list[str] = []

    for i, chunk in enumerate(chunks):
        offset = _chunk_offset_seconds(chunk, i, chunk_minutes)
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        result = transcribe_chunk(chunk, language=language, time_offset=offset)
        full_text_parts.append(result.text)
        all_segments.extend(result.segments)

    print("Transcription complete.")
    return TranscriptionResult(
        text=" ".join(full_text_parts).strip(),
        segments=all_segments,
    )


def transcribe_all_text(chunks: list, language: str = "english") -> str:
    """Backward-compatible helper returning plain text only."""
    return transcribe_all(chunks, language=language).text
