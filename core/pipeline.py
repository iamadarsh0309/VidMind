"""End-to-end pipeline: video URL/file -> transcript -> title -> summary -> notes."""

from dataclasses import dataclass, field

from core.checkpoint import (
    find_checkpoint_for_chunks,
    is_fully_transcribed,
    save_transcribed,
    session_key_from_chunks,
    transcription_from_checkpoint,
)
from core.code_aligner import align_code_to_segments, enriched_to_chunks
from core.domain import is_software_engineering_content
from core.notes import generate_notes, save_notes
from core.notes_schema import NoteDocument
from core.summarize import generate_title, summarize
from core.transcriber import TranscriptionResult, transcribe_all
from utils.audio_processor import process_input
from utils.video_processor import process_video_for_code


@dataclass
class PipelineResult:
    transcript: str
    title: str
    summary: str
    notes_md: str
    note_paths: dict
    transcription: TranscriptionResult
    notes_doc: NoteDocument | None = None
    domain_ok: bool = True
    domain_reason: str = ""
    modules: list[str] = field(default_factory=list)


def _continue_from_transcription(
    source: str,
    transcription: TranscriptionResult,
    *,
    capture_code: bool = True,
    use_course_planner: bool = True,
    enforce_domain: bool = True,
    output_dir: str = "notes/",
    checkpoint_path: str | None = None,
) -> PipelineResult:
    domain_ok, domain_reason = True, ""
    if enforce_domain:
        domain_ok, domain_reason = is_software_engineering_content(transcription.text)
        if not domain_ok:
            raise ValueError(
                f"Domain check failed: {domain_reason}. "
                "VidMind is configured for software engineering notes only."
            )
        print(f"Domain check passed: {domain_reason}")

    enriched_chunks = None
    if capture_code:
        try:
            code_captures = process_video_for_code(source)
            if code_captures and transcription.segments:
                enriched = align_code_to_segments(transcription.segments, code_captures)
                enriched_chunks = enriched_to_chunks(enriched)
        except Exception as exc:
            print(f"Code capture skipped: {exc}")

    title = generate_title(transcription.text)
    summary = summarize(transcription.text)
    notes_md, notes_doc = generate_notes(
        transcription.text,
        title,
        enriched_chunks=enriched_chunks,
        segments=transcription.segments,
        use_course_planner=use_course_planner,
        checkpoint_path=checkpoint_path,
        resume=True,
    )
    note_paths = save_notes(notes_md, title, output_dir=output_dir, doc=notes_doc)

    return PipelineResult(
        transcript=transcription.text,
        title=title,
        summary=summary,
        notes_md=notes_md,
        note_paths=note_paths,
        transcription=transcription,
        notes_doc=notes_doc,
        domain_ok=domain_ok,
        domain_reason=domain_reason,
        modules=notes_doc.modules if notes_doc else [],
    )


def run_pipeline(
    source: str,
    language: str = "english",
    *,
    capture_code: bool = True,
    use_course_planner: bool = True,
    enforce_domain: bool = True,
    output_dir: str = "notes/",
    skip_transcription: bool = False,
    checkpoint_path: str | None = None,
    reuse_chunks: bool = False,
) -> PipelineResult:
    transcription: TranscriptionResult | None = None
    chunks: list | None = None
    active_checkpoint_path: str | None = checkpoint_path

    if skip_transcription and checkpoint_path:
        from core.checkpoint import load_checkpoint

        data = load_checkpoint(path=checkpoint_path)
        if data:
            transcription = transcription_from_checkpoint(data)
            if transcription and transcription.text:
                print(f"Loaded checkpoint: {checkpoint_path}")
                source = data.get("source", source)
                active_checkpoint_path = checkpoint_path

    if transcription is None and (reuse_chunks or skip_transcription):
        chunks = process_input(source, reuse_chunks=True)
        ckpt = find_checkpoint_for_chunks(chunks)
        if ckpt and is_fully_transcribed(ckpt, len(chunks)):
            transcription = transcription_from_checkpoint(ckpt)
            if transcription:
                print("Skipping Whisper — found completed transcript for these audio chunks.")
                source = ckpt.get("source", source)
                active_checkpoint_path = ckpt.get("_path", active_checkpoint_path)

    if transcription is None:
        if chunks is None:
            chunks = process_input(source, reuse_chunks=reuse_chunks)
        sk = session_key_from_chunks(chunks)
        transcription = transcribe_all(
            chunks,
            language=language,
            source=source,
            session_key=sk,
            resume=True,
        )
        save_transcribed(sk, source, language, transcription, chunks=chunks)

    return _continue_from_transcription(
        source,
        transcription,
        capture_code=capture_code,
        use_course_planner=use_course_planner,
        enforce_domain=enforce_domain,
        output_dir=output_dir,
        checkpoint_path=active_checkpoint_path,
    )
