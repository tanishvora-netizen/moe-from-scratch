"""
Datasets -> flat token files (nanoGPT style).

  shakespeare  ~1 MB, downloads in a second. Good for smoke tests.
  tinystories  short children's stories (Eldan & Li, 2023), streamed from Hugging Face.
               The default for real runs: small models learn it well.

Tokenizers:
  byte  vocab 256, no dependencies. Keeps the model at ~7M params and makes the
        expert-specialization analysis very readable (spaces, vowels, punctuation...).
  gpt2  tiktoken BPE, vocab 50257. NOTE: the tied embedding alone is ~12.9M params
        at d_model=256, so total params jump well past 10M.
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

DATA_ROOT = Path(__file__).resolve().parent / "data"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class ByteTokenizer:
    name, vocab_size, dtype = "byte", 256, np.uint8

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


class GPT2Tokenizer:
    name, dtype = "gpt2", np.uint16

    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab
        self.eot = self.enc.eot_token

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids])


def get_tokenizer(name):
    return {"byte": ByteTokenizer, "gpt2": GPT2Tokenizer}[name]()


# ----------------------------------------------------------------------------------------------
def _write_docs(path, docs, tok):
    """Stream documents into a flat binary token file without holding it all in RAM."""
    n_tokens, buf = 0, []
    with open(path, "wb") as f:
        for doc in docs:
            if tok.name == "byte":
                b = (doc + "\n\n").encode("utf-8")
                f.write(b)
                n_tokens += len(b)
            else:
                buf.extend(tok.encode(doc))
                buf.append(tok.eot)
                if len(buf) > 1_000_000:
                    np.array(buf, dtype=tok.dtype).tofile(f)
                    n_tokens += len(buf)
                    buf = []
        if buf:
            np.array(buf, dtype=tok.dtype).tofile(f)
            n_tokens += len(buf)
    return n_tokens


def _tinystories_docs(split, max_bytes):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    total = 0
    for ex in ds:
        text = ex["text"].strip()
        if not text:
            continue
        yield text
        total += len(text.encode("utf-8"))
        if total >= max_bytes:
            break


def prepare_dataset(dataset="tinystories", tokenizer="byte", max_mb=200) -> Path:
    suffix = f"_{max_mb}mb" if dataset == "tinystories" else ""
    out = DATA_ROOT / f"{dataset}_{tokenizer}{suffix}"
    if (out / "meta.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    tok = get_tokenizer(tokenizer)
    print(f"[data] preparing {dataset} with {tokenizer} tokenizer -> {out}")

    if dataset == "shakespeare":
        text = urllib.request.urlopen(SHAKESPEARE_URL).read().decode("utf-8")
        n = int(0.9 * len(text))
        n_train = _write_docs(out / "train.bin", [text[:n]], tok)
        n_val = _write_docs(out / "val.bin", [text[n:]], tok)
    elif dataset == "tinystories":
        max_bytes = max_mb * 1_000_000
        n_train = _write_docs(out / "train.bin", _tinystories_docs("train", max_bytes), tok)
        n_val = _write_docs(out / "val.bin", _tinystories_docs("validation", max(2_000_000, max_bytes // 40)), tok)
    else:
        raise ValueError(dataset)

    meta = {"dataset": dataset, "tokenizer": tokenizer, "vocab_size": tok.vocab_size,
            "dtype": np.dtype(tok.dtype).name, "train_tokens": n_train, "val_tokens": n_val}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[data] train tokens: {n_train:,}  val tokens: {n_val:,}")
    return out


def load_meta(data_dir):
    return json.loads((Path(data_dir) / "meta.json").read_text())


def load_split(data_dir, split):
    meta = load_meta(data_dir)
    return np.memmap(Path(data_dir) / f"{split}.bin", dtype=np.dtype(meta["dtype"]), mode="r")


def sample_offsets(rng, data, batch_size, seq_len):
    return rng.integers(0, len(data) - seq_len - 1, size=batch_size)


def get_batch(data, offsets, seq_len, device):
    x = torch.from_numpy(np.stack([data[i:i + seq_len] for i in offsets]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + seq_len] for i in offsets]).astype(np.int64))
    if str(device).startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
