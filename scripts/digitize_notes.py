#!/usr/bin/env python3
"""
OCR handwritten note scans/photos into style_samples/*.md.

Usage:
    python scripts/digitize_notes.py path/to/scan1.jpg path/to/scan2.png
    python scripts/digitize_notes.py style_samples/scans/*.jpg

Requires: tesseract (brew install tesseract)
"""

import argparse
import os
import re
import sys

try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("Install dependencies: pip install pytesseract Pillow")
    sys.exit(1)

STYLE_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "style_samples")
SCANS_DIR = os.path.join(STYLE_SAMPLES_DIR, "scans")


def ocr_image(image_path: str) -> str:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    text = pytesseract.image_to_string(img)
    return text.strip()


def sanitize_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return stem or "note"


def digitize(image_paths: list[str], output_dir: str = STYLE_SAMPLES_DIR) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    written = []

    for image_path in image_paths:
        if not os.path.isfile(image_path):
            print(f"Skip (not found): {image_path}")
            continue

        print(f"OCR: {image_path}")
        text = ocr_image(image_path)
        if not text:
            print(f"  Warning: no text detected in {image_path}")
            continue

        out_name = sanitize_stem(image_path) + ".md"
        out_path = os.path.join(output_dir, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"<!-- digitized from {os.path.basename(image_path)} -->\n\n")
            f.write(text)
            f.write("\n")

        print(f"  -> {out_path}")
        written.append(out_path)

    return written


def main():
    parser = argparse.ArgumentParser(description="OCR handwritten notes into style_samples/")
    parser.add_argument("images", nargs="+", help="Image files (jpg, png, etc.)")
    parser.add_argument(
        "--output-dir",
        default=STYLE_SAMPLES_DIR,
        help="Directory for .md output (default: style_samples/)",
    )
    args = parser.parse_args()

    paths = written = digitize(args.images, args.output_dir)
    print(f"\nDone. Wrote {len(written)} file(s).")
    print("Polish your 3 best samples, then run: python scripts/extract_style_guide.py")


if __name__ == "__main__":
    main()
