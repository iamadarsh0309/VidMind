"""Set env vars before heavy ML/protobuf imports. Import this module first."""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
# Reduce transformers log noise when optional vision deps are absent
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
