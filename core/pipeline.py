"""End-to-end pipeline: video URL/file -> transcript -> title -> summary -> notes."""

from dataclasses import dataclass, field

from core.code_aligner import align_code_to_segments, enriched_to_chunks
from core.domain import is_software_engineering_content
from core.notes import create_notes, generate_notes, save_notes
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


def run_pipeline(
    source: str,
    language: str = "english",
    *,
    capture_code: bool = True,
    use_course_planner: bool = True,
    enforce_domain: bool = True,
    output_dir: str = "notes/",
) -> PipelineResult:
    chunks = process_input(source)
    transcription = transcribe_all(chunks, language=language)

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
