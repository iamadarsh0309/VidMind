"""Download video and sample frames for on-screen code OCR."""

import base64
import hashlib
import json
import os
import re
import subprocess
import urllib.request
from dataclasses import dataclass

import yt_dlp

from core.code_validator import validate_raw_code

DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "downloads")
FRAMES_DIR = os.path.join(DOWNLOAD_DIR, "frames")
FRAME_INTERVAL_SECONDS = int(os.getenv("FRAME_INTERVAL_SECONDS", "45"))
FRAME_DIFF_THRESHOLD = float(os.getenv("FRAME_DIFF_THRESHOLD", "0.02"))
VLM_MODEL = os.getenv("VLM_MODEL", "moondream")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
USE_VLM_FALLBACK = os.getenv("USE_VLM_FALLBACK", "true").lower() == "true"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

CODE_CHARS = set("{}()[]=:;@#_")
CODE_KEYWORDS = re.compile(
    r"\b(def|class|import|from|return|async|await|function|const|let|var|public|private|if|else|for|while)\b",
    re.IGNORECASE,
)


@dataclass
class FrameCapture:
    path: str
    time_seconds: float


@dataclass
class CodeCapture:
    time_seconds: float
    code: str
    source_frame: str


def download_youtube_video(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "best[height<=720]/best",
        "outtmpl": output_path,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def extract_frames(
    video_path: str,
    interval_seconds: int = FRAME_INTERVAL_SECONDS,
    output_subdir: str | None = None,
) -> list[FrameCapture]:
    base = os.path.splitext(os.path.basename(video_path))[0]
    safe_base = re.sub(r"[^\w\-]+", "_", base)[:80]
    out_dir = output_subdir or os.path.join(FRAMES_DIR, safe_base)
    os.makedirs(out_dir, exist_ok=True)

    pattern = os.path.join(out_dir, "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps=1/{interval_seconds}",
        "-q:v", "2", pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    frames = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".jpg"):
            continue
        idx = int(name.replace("frame_", "").replace(".jpg", ""))
        frames.append(
            FrameCapture(
                path=os.path.join(out_dir, name),
                time_seconds=(idx - 1) * interval_seconds,
            )
        )
    print(f"Extracted {len(frames)} frame(s) to {out_dir}")
    return frames


def _frame_hash(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _frame_diff_ratio(path_a: str, path_b: str) -> float:
    """Return 0.0 = identical, 1.0 = completely different."""
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return 1.0

    img_a = Image.open(path_a).convert("RGB").resize((160, 90))
    img_b = Image.open(path_b).convert("RGB").resize((160, 90))
    diff = ImageChops.difference(img_a, img_b)
    hist = diff.histogram()
    total = sum(hist)
    if total == 0:
        return 0.0
    # Weighted mean of channel diffs (RGB)
    mean_diff = sum(i * v for i, v in enumerate(hist)) / (total * 255.0)
    return mean_diff


def filter_frames_by_scene_change(
    frames: list[FrameCapture],
    threshold: float = FRAME_DIFF_THRESHOLD,
) -> list[FrameCapture]:
    """
    Keep first frame and frames that differ significantly from the previous kept frame.
    Skips long stretches of identical IDE screens.
    """
    if not frames:
        return frames

    kept = [frames[0]]
    for frame in frames[1:]:
        ratio = _frame_diff_ratio(kept[-1].path, frame.path)
        if ratio >= threshold:
            kept.append(frame)

    removed = len(frames) - len(kept)
    if removed:
        print(f"Scene filter: kept {len(kept)}/{len(frames)} frames ({removed} static skipped)")
    return kept


def dedupe_frames(frames: list[FrameCapture]) -> list[FrameCapture]:
    seen: set[str] = set()
    unique: list[FrameCapture] = []
    for frame in frames:
        h = _frame_hash(frame.path)
        if h in seen:
            continue
        seen.add(h)
        unique.append(frame)
    removed = len(frames) - len(unique)
    if removed:
        print(f"Deduplicated {removed} identical frame(s)")
    return unique


def _ocr_image(image_path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Install OCR deps: pip install pytesseract Pillow, and brew install tesseract"
        ) from exc

    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return pytesseract.image_to_string(img)


def _vlm_extract_code(image_path: str) -> str:
    """Ollama vision model fallback for IDE screenshots."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": VLM_MODEL,
        "prompt": (
            "Extract ONLY the programming code visible on screen. "
            "Return a fenced code block with the correct language tag, or the word EMPTY if no code."
        ),
        "images": [b64],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("response", "").strip()
        if "EMPTY" in text.upper() and "```" not in text:
            return ""
        m = re.search(r"```[\w]*\n(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else text
    except Exception as exc:
        print(f"  VLM fallback failed: {exc}")
        return ""


def _code_score(text: str) -> float:
    if not text or len(text.strip()) < 8:
        return 0.0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0

    code_chars = sum(1 for c in text if c in CODE_CHARS)
    char_ratio = code_chars / max(len(text), 1)
    keyword_hits = len(CODE_KEYWORDS.findall(text))
    avg_len = sum(len(ln) for ln in lines) / len(lines)
    monospace_like = 1.0 if 10 <= avg_len <= 80 else 0.3
    return char_ratio * 2.0 + keyword_hits * 0.5 + monospace_like


def _extract_code_blocks(ocr_text: str, min_score: float = 1.2) -> str:
    blocks: list[str] = []
    current: list[str] = []

    for line in ocr_text.splitlines():
        stripped = line.strip()
        if _code_score(stripped) >= 0.4 or (current and stripped):
            current.append(stripped)
        elif current:
            block = "\n".join(current).strip()
            if _code_score(block) >= min_score:
                blocks.append(block)
            current = []

    if current:
        block = "\n".join(current).strip()
        if _code_score(block) >= min_score:
            blocks.append(block)

    return "\n\n".join(blocks)


def _extract_code_from_frame(frame: FrameCapture) -> str:
    raw = _ocr_image(frame.path)
    code = _extract_code_blocks(raw)

    if _code_score(code) < 1.0 and USE_VLM_FALLBACK:
        print(f"    OCR low confidence @ {frame.time_seconds:.0f}s, trying VLM...")
        vlm_code = _vlm_extract_code(frame.path)
        if _code_score(vlm_code) > _code_score(code):
            code = vlm_code

    if code:
        code = validate_raw_code(code)
    return code


def ocr_code_from_frames(frames: list[FrameCapture]) -> list[CodeCapture]:
    captures: list[CodeCapture] = []
    for i, frame in enumerate(frames):
        print(f"  OCR frame {i + 1}/{len(frames)} @ {frame.time_seconds:.0f}s")
        code = _extract_code_from_frame(frame)
        if code:
            captures.append(
                CodeCapture(
                    time_seconds=frame.time_seconds,
                    code=code,
                    source_frame=frame.path,
                )
            )
    print(f"Detected code in {len(captures)} frame(s)")
    return captures


def process_video_for_code(
    source: str,
    interval_seconds: int = FRAME_INTERVAL_SECONDS,
) -> list[CodeCapture]:
    if source.startswith("http://") or source.startswith("https://"):
        print("Downloading video for frame sampling...")
        video_path = download_youtube_video(source)
    else:
        video_path = source

    print("Sampling frames...")
    frames = extract_frames(video_path, interval_seconds=interval_seconds)
    frames = filter_frames_by_scene_change(frames)
    frames = dedupe_frames(frames)

    print("Running OCR for on-screen code...")
    return ocr_code_from_frames(frames)
