from dotenv import load_dotenv

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

load_dotenv()

from core.pipeline import run_pipeline

source = "https://www.youtube.com/watch?v=55pTFVoclvE"
language = "english"

result = run_pipeline(
    source,
    language=language,
    capture_code=True,
    use_course_planner=True,
    enforce_domain=True,
)

print("\n=== TRANSCRIPT ===\n")
print(result.transcript[:2000], "...")

print("\n=== TITLE ===\n")
print(result.title)

if result.modules:
    print("\n=== MODULES ===\n")
    for m in result.modules:
        print(f" - {m}")

print("\n=== SUMMARY ===\n")
print(result.summary)

print("\n=== NOTES ===\n")
print(f"Saved to:\n  - {result.note_paths['md']}\n  - {result.note_paths['pdf']}\n  - {result.note_paths['json']}")
