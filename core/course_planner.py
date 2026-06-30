"""Course planner: split long transcripts into modules before note generation."""

import json
import os
import re
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm
from core.transcriber import TranscriptSegment

PLANNER_THRESHOLD_CHARS = int(os.getenv("COURSE_PLANNER_THRESHOLD", "12000"))


@dataclass
class CourseModule:
    title: str
    start_seconds: float
    end_seconds: float
    transcript: str
    code: str = ""


PLANNER_SYSTEM = """You are a course structure analyst for software engineering video tutorials.

Given a transcript (with optional timestamps), divide it into logical modules/chapters.

Return ONLY valid JSON array:
[
  {"title": "Module name", "start_seconds": 0, "end_seconds": 600},
  ...
]

Rules:
- 3-12 modules depending on length
- Titles should be specific (e.g. "FastAPI Routing" not "Part 1")
- Modules must cover the full timeline without gaps or overlaps
- end_seconds of module N should equal start_seconds of module N+1
- Last module end_seconds should match the video end
- Focus on software engineering topic boundaries"""


def _format_transcript_with_times(segments: list[TranscriptSegment]) -> str:
    lines = []
    for seg in segments:
        if not seg.text.strip():
            continue
        mins = int(seg.start // 60)
        secs = int(seg.start % 60)
        lines.append(f"[{mins:02d}:{secs:02d}] {seg.text.strip()}")
    return "\n".join(lines)


def _transcript_in_range(
    segments: list[TranscriptSegment],
    start: float,
    end: float,
) -> str:
    parts = []
    for seg in segments:
        if seg.end < start or seg.start > end:
            continue
        parts.append(seg.text.strip())
    return " ".join(parts)


def plan_course(
    transcript: str,
    segments: list[TranscriptSegment] | None = None,
) -> list[CourseModule]:
    """
    Return module list. For short transcripts, returns a single module.
    """
    if len(transcript) < PLANNER_THRESHOLD_CHARS:
        return [
            CourseModule(
                title="Full course",
                start_seconds=0.0,
                end_seconds=segments[-1].end if segments else 0.0,
                transcript=transcript,
            )
        ]

    timed_text = _format_transcript_with_times(segments) if segments else transcript[:20000]
    video_end = segments[-1].end if segments else 0.0

    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM),
        (
            "human",
            "Video end (seconds): {video_end}\n\nTranscript:\n\n{text}",
        ),
    ])
    chain = prompt | get_llm(temperature=0.2) | StrOutputParser()
    raw = chain.invoke({"text": timed_text[:18000], "video_end": video_end})

    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        return [
            CourseModule(
                title="Full course",
                start_seconds=0.0,
                end_seconds=video_end,
                transcript=transcript,
            )
        ]

    try:
        modules_data = json.loads(match.group())
    except json.JSONDecodeError:
        return [
            CourseModule(
                title="Full course",
                start_seconds=0.0,
                end_seconds=video_end,
                transcript=transcript,
            )
        ]

    modules: list[CourseModule] = []
    for item in modules_data:
        start = float(item.get("start_seconds", 0))
        end = float(item.get("end_seconds", video_end))
        title = str(item.get("title", "Module")).strip()
        if segments:
            mod_text = _transcript_in_range(segments, start, end)
        else:
            mod_text = transcript
        modules.append(
            CourseModule(
                title=title,
                start_seconds=start,
                end_seconds=end,
                transcript=mod_text or transcript,
            )
        )

    print(f"Course planner: {len(modules)} module(s)")
    for i, m in enumerate(modules, 1):
        print(f"  {i}. {m.title} ({m.start_seconds:.0f}s - {m.end_seconds:.0f}s)")
    return modules


def attach_code_to_modules(
    modules: list[CourseModule],
    enriched_chunks: list[dict],
) -> list[CourseModule]:
    """Distribute aligned code chunks across modules by proportional text overlap."""
    if not enriched_chunks:
        return modules

    for mod in modules:
        codes = []
        for chunk in enriched_chunks:
            text = chunk.get("text", "")
            code = chunk.get("code", "")
            if code and text and text[:80] in mod.transcript:
                codes.append(code)
        mod.code = "\n\n".join(codes)
    return modules
