"""The byte-level tokenizer and GPT-2 BPE through tiktoken. Each one's dtype is the numpy type of its token files,
since uint8 holds 256 ids and uint16 holds GPT-2's 50257.
"""
import numpy as np


class ByteTokenizer:
    """One token per UTF-8 byte, vocab 256, no dependencies.

    It is the default because it keeps the model near 7M parameters, and the specialization analysis can
    group its tokens into readable categories such as spaces, vowels and punctuation.
    """

    name, vocab_size, dtype = "byte", 256, np.uint8

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


class GPT2Tokenizer:
    """GPT-2 BPE through tiktoken, vocab 50257.

    At d_model=256 the tied embedding alone has 12.9M parameters, so the model is well past 10M.
    """

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
