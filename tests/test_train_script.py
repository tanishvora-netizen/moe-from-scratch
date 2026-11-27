"""End to end: scripts/train.py runs a few steps on a tiny local dataset, and the analysis scripts
(sample.py, plot_runs.py, analyze_experts.py) work on what it writes."""
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


def run_script(name, *args):
    """Run scripts/<name> the way a user would. Works with or without `pip install -e .`."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def train(data_dir, out_dir, ffn, router, *extra):
    run_script("train.py", "--data_dir", data_dir, "--out_dir", out_dir, "--name", f"{ffn}_{router}",
               "--ffn", ffn, "--router", router, "--impl", "naive", "--device", "cpu",
               *TINY_MODEL, *SHORT_RUN, *extra)
    return out_dir / f"{ffn}_{router}"


def read_metrics(run):
    return [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """A byte-level dataset in the same format prepare_dataset() writes."""
    data = np.frombuffer(b"the quick brown fox jumps over the lazy dog. " * 400, dtype=np.uint8)
    n = int(0.9 * len(data))
    d = tmp_path_factory.mktemp("tiny_byte")
    data[:n].tofile(d / "train.bin")
    data[n:].tofile(d / "val.bin")
    meta = {"dataset": "tiny", "tokenizer": "byte", "vocab_size": 256, "dtype": "uint8",
            "train_tokens": n, "val_tokens": len(data) - n}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


@pytest.fixture(scope="module")
def moe_run(tiny_dataset, tmp_path_factory):
    """One finished aux-free MoE run, shared by the analysis-script tests."""
    return train(tiny_dataset, tmp_path_factory.mktemp("runs"), "moe", "aux_free")


@pytest.mark.parametrize("ffn,router", [("dense", "aux_loss"), ("moe", "aux_loss"),
                                        ("moe", "aux_free"), ("moe", "expert_choice")])
def test_train_script_runs(tiny_dataset, tmp_path, ffn, router):
    run = train(tiny_dataset, tmp_path, ffn, router)
    recs = read_metrics(run)
    assert [r["step"] for r in recs if "val_loss" in r] == [0, 2, 4]
    assert all(np.isfinite(r["train_loss"]) for r in recs if "train_loss" in r)
    if ffn == "moe":
        assert all("maxvio" in r for r in recs if "train_loss" in r)
    if router == "expert_choice" and ffn == "moe":
        assert all("val_loss_token_choice" in r for r in recs if "val_loss" in r)
    assert (run / "ckpt.pt").exists() and (run / "sample.txt").exists()


def test_train_script_with_grad_accum(tiny_dataset, tmp_path):
    run = train(tiny_dataset, tmp_path, "moe", "aux_free", "--grad_accum", "3")
    train_recs = [r for r in read_metrics(run) if "train_loss" in r]
    assert len(train_recs) == 4 and all(np.isfinite(r["train_loss"]) for r in train_recs)


def test_sample_script(moe_run):
    out = run_script("sample.py", "--ckpt", moe_run / "ckpt.pt", "--tokens", 10, "--prompt", "the")
    assert "----- sample 0 -----" in out


def test_analyze_experts_script(moe_run):
    run_script("analyze_experts.py", "--ckpt", moe_run / "ckpt.pt", "--batches", 2, "--batch_size", 4,
               "--min_count", 1)
    summary = json.loads((moe_run / "analysis" / "summary.json").read_text())
    n_layers = int(TINY_MODEL[TINY_MODEL.index("--n_layers") + 1])
    assert len(summary) == n_layers   # every layer is MoE: the run sets no --first_k_dense
    assert all(0.0 <= s["normalized_mi"] <= 1.0 for s in summary)


def test_plot_runs_script(moe_run, tmp_path):
    pytest.importorskip("matplotlib")
    out = run_script("plot_runs.py", moe_run, "--out", tmp_path / "cmp.png")
    assert out.startswith("| run |") and moe_run.name in out
    assert (tmp_path / "cmp.png").exists()
