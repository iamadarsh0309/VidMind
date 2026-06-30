#!/usr/bin/env python3
"""Install protobuf pin compatible with ChromaDB on Python 3.13."""

import os
import subprocess
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def main():
    cmd = [
        sys.executable, "-m", "pip", "install",
        "protobuf>=3.20.0,<4.0.0",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("\nOK. Verify with:")
    print("  python scripts/digitize_pdf.py style_samples/fastapi_tutorial.pdf --index")


if __name__ == "__main__":
    main()
