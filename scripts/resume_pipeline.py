#!/usr/bin/env python3
"""
Resume VidMind pipeline without re-transcribing.

Usage:
    python scripts/resume_pipeline.py --list
    python scripts/resume_pipeline.py --latest
    python scripts/resume_pipeline.py --latest --no-code
    python scripts/resume_pipeline.py downloads/checkpoints/<file>.json
"""

import argparse
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from core.checkpoint import list_checkpoints
from core.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Resume VidMind from checkpoint")
    parser.add_argument("checkpoint", nargs="?", help="Path to checkpoint .json")
    parser.add_argument("--list", action="store_true", help="List saved checkpoints")
    parser.add_argument("--latest", action="store_true", help="Use most recent checkpoint")
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="Skip on-screen code capture (recommended for talk/slide videos)",
    )
    parser.add_argument("--no-domain", action="store_true", help="Skip software-engineering filter")
    parser.add_argument(
        "--reuse-chunks",
        action="store_true",
        help="Reuse downloads/*_chunk_*.wav (skip re-download/re-chunk if possible)",
    )
    parser.add_argument("--source", help="YouTube URL (required with --reuse-chunks for resume)")
    args = parser.parse_args()

    if args.list:
        cps = list_checkpoints()
        if not cps:
            print("No checkpoints in downloads/checkpoints/")
            return
        for cp in cps:
            print(f"{cp['_path']}")
            print(f"  source: {cp.get('source', '?')[:70]}")
            print(f"  status: {cp.get('status')}  updated: {cp.get('updated_at')}")
            if cp.get("text"):
                print(f"  text: {len(cp['text'])} chars")
            elif cp.get("text_parts"):
                print(f"  chunks done: {cp.get('completed_chunk_index', 0) + 1}")
        return

    path = args.checkpoint
    if args.latest:
        cps = list_checkpoints()
        if not cps:
            print("No checkpoints found. Run the pipeline once first.")
            sys.exit(1)
        path = cps[-1]["_path"]

    if not path and not args.reuse_chunks:
        parser.print_help()
        sys.exit(1)

    source = args.source or ""
    if path:
        print(f"Resuming from: {path}")
        result = run_pipeline(
            source=source,
            skip_transcription=True,
            checkpoint_path=path,
            capture_code=not args.no_code,
            enforce_domain=not args.no_domain,
        )
    else:
        if not source:
            print("--source URL required with --reuse-chunks")
            sys.exit(1)
        print(f"Re-running from existing audio chunks: {source[:70]}...")
        result = run_pipeline(
            source=source,
            capture_code=not args.no_code,
            enforce_domain=not args.no_domain,
            reuse_chunks=True,
            skip_transcription=True,
        )
    print(f"\nTitle: {result.title}")
    print(f"Saved:\n  {result.note_paths['md']}\n  {result.note_paths['pdf']}\n  {result.note_paths['json']}")


if __name__ == "__main__":
    main()
