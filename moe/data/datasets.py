"""Download and tokenize a dataset once into flat token files (nanoGPT style), cached under <repo>/data.

shakespeare: ~1 MB, downloads in a second, good for smoke tests. tinystories: short children's stories
(Eldan & Li, 2023) streamed from Hugging Face, the default for real runs because small models learn it well.
"""
import json
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from .tokenizers import ByteTokenizer, GPT2Tokenizer, get_tokenizer

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"   # <repo>/data
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _write_docs(path: Path, docs: Iterable[str], tok: ByteTokenizer | GPT2Tokenizer) -> int:
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


def _tinystories_docs(split: str, max_bytes: int) -> Iterator[str]:
    from datasets import load_dataset  # optional dependency, only needed for TinyStories
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


def dataset_dir(dataset: str = "tinystories", tokenizer: str = "byte", max_mb: int = 200) -> Path:
    """Where a prepared (dataset, tokenizer) pair is cached: <repo>/data/<dataset>_<tokenizer>[_<N>mb]."""
    suffix = f"_{max_mb}mb" if dataset == "tinystories" else ""
    return DATA_ROOT / f"{dataset}_{tokenizer}{suffix}"


def prepare_dataset(dataset: str = "tinystories", tokenizer: str = "byte", max_mb: int = 200) -> Path:
    """Download and tokenize a dataset once; later calls return the cached directory."""
    out = dataset_dir(dataset, tokenizer, max_mb)
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
