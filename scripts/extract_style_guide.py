#!/usr/bin/env python3
"""
Extract style/style_guide.md from the primary reference (default) or note samples.

Usage:
    python scripts/extract_style_guide.py              # from primary reference PDF chunks
    python scripts/extract_style_guide.py --samples   # from handwritten .md samples
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from core.llm import get_llm
from core.primary_reference import load_primary_chunks, primary_reference_status
from core.style import STYLE_GUIDE_PATH, STYLE_SAMPLES_DIR, extract_style_guide_from_samples


def load_primary_samples() -> str:
    chunks = load_primary_chunks()
    parts = []
    for c in chunks[:6]:
        parts.append(f"## {c['basename']}\n\n{c['content'][:3000]}")
    return "\n\n---\n\n".join(parts)


def load_handwritten_samples() -> str:
    paths = sorted(glob.glob(os.path.join(STYLE_SAMPLES_DIR, "*.md")))
    parts = []
    for path in paths:
        if os.path.basename(path).lower() == "readme.md":
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            parts.append(f"## {os.path.basename(path)}\n\n{text}")
    return "\n\n---\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Use handwritten .md samples instead of primary reference",
    )
    args = parser.parse_args()

    if args.samples:
        samples = load_handwritten_samples()
        source = "handwritten samples"
    else:
        status = primary_reference_status()
        if not status["ready"]:
            print(f"Primary reference not extracted. Run:")
            print(f"  python scripts/digitize_pdf.py {status['pdf']}")
            sys.exit(1)
        samples = load_primary_samples()
        source = f"primary reference ({status['name']})"

    if not samples:
        sys.exit("No samples to analyze.")

    os.makedirs(os.path.dirname(STYLE_GUIDE_PATH), exist_ok=True)
    print(f"Extracting style guide from {source} via LLM...")
    guide = extract_style_guide_from_samples(samples, get_llm(temperature=0.2))

    with open(STYLE_GUIDE_PATH, "w", encoding="utf-8") as f:
        f.write(guide)
        f.write("\n")

    print(f"Wrote {STYLE_GUIDE_PATH}")


if __name__ == "__main__":
    main()
