"""A small Mixture-of-Experts language model in the DeepSeekMoE style (Dai et al. 2024).
The package root re-exports the public API.
"""
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig
from .model.transformer import MoETransformer

__all__ = ["ModelConfig", "MoETransformer", "load_checkpoint", "save_checkpoint"]
