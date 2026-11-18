"""End to end: scripts/train.py runs a few steps on a tiny local dataset and writes its outputs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

TINY_MODEL = ["--seq_len", "32", "--d_model", "64", "--n_layers", "2", "--n_heads", "4",
              "--n_experts", "8", "--top_k", "2", "--expert_hidden", "16", "--n_shared", "1"]
SHORT_RUN = ["--steps", "4", "--warmup", "2", "--batch_size", "4",
             "--eval_interval", "2", "--eval_iters", "2", "--log_interval", "1"]


@pytest.fixture
def tiny_dataset(tmp_path):
    """A byte-level dataset in the same format prepare_dataset() writes."""
    data = np.frombuffer(b"the quick brown fox jumps over the lazy dog. " * 400, dtype=np.uint8)
    n = int(0.9 * len(data))
    d = tmp_path / "tiny_byte"
    d.mkdir()
    data[:n].tofile(d / "train.bin")
    data[n:].tofile(d / "val.bin")
    meta = {"dataset": "tiny", "tokenizer": "byte", "vocab_size": 256, "dtype": "uint8",
            "train_tokens": n, "val_tokens": len(data) - n}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


@pytest.mark.parametrize("ffn,router", [("dense", "aux_loss"), ("moe", "aux_loss"),
                                        ("moe", "aux_free"), ("moe", "expert_choice")])
def test_train_script_runs(tiny_dataset, tmp_path, ffn, router):
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--data_dir", str(tiny_dataset),
           "--out_dir", str(tmp_path / "runs"), "--name", "t", "--ffn", ffn, "--router", router,
           "--impl", "naive", "--device", "cpu", *TINY_MODEL, *SHORT_RUN]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}   # works with or without `pip install -e .`
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    run = tmp_path / "runs" / "t"
    recs = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs if "val_loss" in r] == [0, 2, 4]
    assert all(np.isfinite(r["train_loss"]) for r in recs if "train_loss" in r)
    if ffn == "moe":
        assert all("maxvio" in r for r in recs if "train_loss" in r)
    if router == "expert_choice" and ffn == "moe":
        assert all("val_loss_token_choice" in r for r in recs if "val_loss" in r)
    assert (run / "ckpt.pt").exists() and (run / "sample.txt").exists()
