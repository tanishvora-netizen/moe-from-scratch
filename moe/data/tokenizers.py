"""Tokenizers: byte-level for small, readable models, and GPT-2 BPE through tiktoken.

Each one's dtype is the numpy type of its token files: uint8 holds 256 ids, uint16 holds GPT-2's 50257.
"""
import numpy as np


class ByteTokenizer:
    """One token per UTF-8 byte, vocab 256, no dependencies.

    Keeps the model at ~7M params and makes the expert-specialization analysis very readable
    (spaces, vowels, punctuation...).
    """

    name, vocab_size, dtype = "byte", 256, np.uint8

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


class GPT2Tokenizer:
    """tiktoken BPE, vocab 50257. The tied embedding alone is ~12.9M params at d_model=256,
    so total params jump well past 10M."""

    name, dtype = "gpt2", np.uint16

    def __init__(self):
        import tiktoken  # optional dependency, only needed for this tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab
        self.eot = self.enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids) -> str:
        return self.enc.decode([int(i) for i in ids])


def get_tokenizer(name: str) -> ByteTokenizer | GPT2Tokenizer:
    """name is "byte" or "gpt2", the --tokenizer flag."""
    return {"byte": ByteTokenizer, "gpt2": GPT2Tokenizer}[name]()
