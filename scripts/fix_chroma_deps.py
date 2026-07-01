#!/usr/bin/env python3
"""Install deps that commonly break on Python 3.13 (ChromaDB, embeddings, Streamlit)."""

import os
import subprocess
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

PACKAGES = [
    "protobuf>=3.20.0,<4.0.0",
    "torchvision>=0.17.0",
    "langchain-ollama>=0.2.0",
]


def main():
    cmd = [sys.executable, "-m", "pip", "install", *PACKAGES]
    print("Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print("\npip failed — try with uv:")
        print(f"  uv pip install {' '.join(PACKAGES)} --python {sys.executable}")
        sys.exit(1)
    print("\nOK. Restart Streamlit:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
