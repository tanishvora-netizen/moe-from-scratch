"""Downloads and tokenizes a dataset once into flat token files, as nanoGPT does, cached under <repo>/data.
shakespeare is about 1 MB, for smoke tests. tinystories (Eldan and Li 2023) streams short children's stories
from Hugging Face and is the default, because small models learn it well.
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
    """Writes docs to path as one flat token file and returns its token count.

    GPT-2 tokens are flushed every million, so a large split never has to fit in RAM.
    """
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
    """Downloads and tokenizes a dataset once. Later calls return the cached directory."""
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
