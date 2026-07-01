#!/usr/bin/env python3
"""Submit user-edited notes to improve future generations."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.feedback import submit_feedback, submit_feedback_from_file


def main():
    parser = argparse.ArgumentParser(description="Index edited notes for feedback learning")
    parser.add_argument("path", help="Path to edited .md file")
    parser.add_argument("--title", default="Approved notes", help="Title for new feedback file")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Index the file in place without copying to notes/approved/",
    )
    args = parser.parse_args()

    if args.in_place:
        out = submit_feedback_from_file(args.path)
    else:
        with open(args.path, encoding="utf-8") as f:
            content = f.read()
        out = submit_feedback(content, args.title, source_path=args.path)

    print(f"Indexed: {out}")


if __name__ == "__main__":
    main()
