"""Where and in what precision code runs: device, dtype, autocast, TF32, and the fallback when Triton can't run.
Training uses all of it. sample.py and analyze_experts.py use autocast_context.
"""
import os
from contextlib import AbstractContextManager, nullcontext

import torch


def pick_device(device: str) -> str:
    """--device: "auto" picks cuda, then mps, then cpu."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(dtype: str, device: str) -> torch.dtype:
    """--dtype: low precision only on CUDA. "auto" picks bf16 where the GPU supports it, else fp16."""
    if device != "cuda":
        return torch.float32
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def pick_impl(impl: str, device: str) -> str:
    """Returns --impl, or "naive" when Triton can't run. Triton needs an NVIDIA GPU or TRITON_INTERPRET=1."""
    if impl == "triton" and device != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        print("[warn] Triton kernels need an NVIDIA GPU -> using --impl naive")
        return "naive"
    return impl


def autocast_context(device: str, dtype: torch.dtype) -> AbstractContextManager:
    """torch.autocast for bf16 or fp16 on CUDA. Anywhere else, a context that does nothing."""
    if device == "cuda" and dtype != torch.float32:
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def allow_tf32(device: str) -> None:
    """On CUDA, lets fp32 matmuls and convolutions use TF32 tensor cores, which are faster at a small precision cost."""
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
