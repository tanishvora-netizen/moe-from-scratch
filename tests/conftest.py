"""Without a GPU, run Triton kernels in the CPU interpreter. Must be set BEFORE triton is imported."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
except ImportError:
    pass
