"""Finished runs for plot_runs.py: read each run's metrics.jsonl and build the Markdown results table."""
import json
from pathlib import Path

Run = tuple[str, list[dict], list[dict]]   # (run name, eval records, train records)


def load_run(run_dir: str | Path) -> Run:
    """Splits <run_dir>/metrics.jsonl into eval records (with "val_loss") and train records (with "train_loss")."""
    lines = (Path(run_dir) / "metrics.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    evals = [record for record in records if "val_loss" in record]
    train = [record for record in records if "train_loss" in record]
    return Path(run_dir).name, evals, train


def median_tok_per_s(train: list[dict]) -> float:
    """Median training throughput (the upper median for an even count)."""
    return sorted(record["tok_per_s"] for record in train)[len(train) // 2]


def _format_or_dash(value: float | None, spec: str) -> str:
    """'-' for metrics a run doesn't have (MaxVio for dense, causal eval for token-choice routers)."""
    return "-" if value is None else format(value, spec)


def results_table(runs: list[Run]) -> str:
    """A Markdown table, one row per run, that can be pasted into the README as is."""
    lines = ["| run | final val loss | causal val loss | MaxVio (last 10%) | tok/s (median) |",
             "|---|---|---|---|---|"]
    for name, evals, train in runs:
        tail = train[-max(1, len(train) // 10):]
        maxvio = sum(record["maxvio"] for record in tail) / len(tail) if "maxvio" in tail[0] else None
        causal = evals[-1].get("val_loss_token_choice")
        lines.append(f"| {name} | {evals[-1]['val_loss']:.4f} | {_format_or_dash(causal, '.4f')} "
                     f"| {_format_or_dash(maxvio, '.3f')} | {median_tok_per_s(train):,.0f} |")
    return "\n".join(lines)
