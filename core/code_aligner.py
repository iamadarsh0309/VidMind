"""Align on-screen code OCR captures with timestamped transcript segments."""

from dataclasses import dataclass

from core.transcriber import TranscriptSegment
from utils.video_processor import CodeCapture


@dataclass
class EnrichedSegment:
    start: float
    end: float
    text: str
    code: str


def _segment_at_time(segments: list[TranscriptSegment], t: float) -> TranscriptSegment | None:
    for seg in segments:
        if seg.start <= t <= seg.end:
            return seg
    best = None
    best_dist = float("inf")
    for seg in segments:
        mid = (seg.start + seg.end) / 2
        dist = abs(mid - t)
        if dist < best_dist:
            best_dist = dist
            best = seg
    return best


def _normalize_code(code: str) -> str:
    return " ".join(code.split())


def align_code_to_segments(
    segments: list[TranscriptSegment],
    code_captures: list[CodeCapture],
) -> list[EnrichedSegment]:
    """
    Attach OCR code to the transcript segment whose time window overlaps each frame.
    Merges consecutive segments that share the same text window.
    """
    code_by_segment: dict[tuple[float, float], list[str]] = {}

    for capture in code_captures:
        seg = _segment_at_time(segments, capture.time_seconds)
        if seg is None:
            continue
        key = (seg.start, seg.end)
        code_by_segment.setdefault(key, []).append(capture.code)

    enriched: list[EnrichedSegment] = []
    seen_keys: set[tuple[float, float]] = set()

    for seg in segments:
        key = (seg.start, seg.end)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        codes = code_by_segment.get(key, [])
        deduped: list[str] = []
        seen_code: set[str] = set()
        for code in codes:
            norm = _normalize_code(code)
            if norm in seen_code:
                continue
            seen_code.add(norm)
            deduped.append(code)

        enriched.append(
            EnrichedSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                code="\n\n".join(deduped),
            )
        )

    return enriched


def enriched_to_chunks(
    enriched: list[EnrichedSegment],
    chunk_size: int = 4000,
    overlap: int = 300,
) -> list[dict]:
    """
    Group enriched segments into text chunks for map-reduce note generation.
    Each chunk is {"text": transcript_part, "code": combined_code_for_that_part}.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    full_text = " ".join(seg.text for seg in enriched if seg.text)
    if not full_text.strip():
        return [{"text": "", "code": ""}]

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    text_chunks = splitter.split_text(full_text)

    if not any(seg.code for seg in enriched):
        return [{"text": t, "code": ""} for t in text_chunks]

    seg_spans: list[tuple[int, int, EnrichedSegment]] = []
    pos = 0
    for seg in enriched:
        if not seg.text:
            continue
        start = pos
        pos += len(seg.text) + 1
        seg_spans.append((start, pos, seg))

    result = []
    search_start = 0
    for chunk_text in text_chunks:
        idx = full_text.find(chunk_text[: min(80, len(chunk_text))], search_start)
        if idx < 0:
            idx = search_start
        chunk_end = idx + len(chunk_text)
        search_start = max(0, chunk_end - overlap)

        codes: list[str] = []
        seen: set[str] = set()
        for span_start, span_end, seg in seg_spans:
            if span_end <= idx or span_start >= chunk_end:
                continue
            if seg.code:
                norm = _normalize_code(seg.code)
                if norm not in seen:
                    seen.add(norm)
                    codes.append(seg.code)

        result.append({"text": chunk_text, "code": "\n\n".join(codes)})

    return result
