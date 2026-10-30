"""Shared pieces for Q1: labels, paths, split, metrics."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data" / "frozen"
RESULTS = ROOT / "results" / "q1"
ARTIFACTS = ROOT / "artifacts"
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
SEED = 42


def split_70_15_15(df: pd.DataFrame, seed: int = SEED):
    """Stratified 70/15/15 split, deterministic for a given seed."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(FROZEN / f"q1_{name}.csv")


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error on max-probability confidence."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(LABELS)))))


def classification_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = np.asarray(probs).argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": macro_f1(y, pred),
        "ece": ece(probs, y),
    }


def paired_bootstrap_diff(
    y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n: int = 2000, seed: int = SEED
) -> dict:
    """Paired bootstrap of macro-F1(a) - macro-F1(b) over test items."""
    rng = np.random.default_rng(seed)
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    m = len(y)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        diffs[i] = macro_f1(y[idx], pred_a[idx]) - macro_f1(y[idx], pred_b[idx])
    return {
        "point": macro_f1(y, pred_a) - macro_f1(y, pred_b),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_resamples": n,
    }


def time_per_1k(fn, texts: list[str], repeats: int = 1) -> float:
    """Seconds to run fn over texts, scaled to 1,000 texts (best of repeats)."""
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn(texts)
        best = min(best, time.perf_counter() - t)
    return best * 1000.0 / len(texts)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def update_results(key: str, value) -> None:
    path = RESULTS / "results.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    write_json(path, data)
