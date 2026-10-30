"""Q2: does a scored headline predict the next-6h YES price direction?

Stages (python -m hms.q2 <stage>): baseline, haiku, report. Fine-tuning is in hms.q2_finetune.
Every model writes data/q2_scores/<model>.csv with columns headline_id, split, score.
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from hms import q2_stats
from hms.db import DB_PATH

SCORES = Path("data/q2_scores")
RESULTS = Path("results/q2")
PRIMARY = ["baseline_qh", "haiku_qh", "finetuned_qh"]  # pre-registered primary models (Holm set)
ABLATIONS = {"baseline_h": "baseline_qh", "finetuned_h": "finetuned_qh"}

EVENTS_SQL = """
SELECT e.headline_id, e.market_id, e.t0, e.dp, e.label, e.split, h.title, m.question
FROM events e JOIN headlines h USING (headline_id) JOIN markets m ON m.market_id = e.market_id
WHERE e.split IN ('train', 'val', 'test')
ORDER BY e.t0
"""


def load_events(db: Path = DB_PATH) -> pd.DataFrame:
    df = pd.read_sql_query(EVENTS_SQL, sqlite3.connect(db))
    df["y"] = (df["label"] == "up").astype(int)
    return df


def text(df: pd.DataFrame, with_question: bool) -> pd.Series:
    return (df["question"] + " || " + df["title"]) if with_question else df["title"]


def save(name: str, df: pd.DataFrame, score: np.ndarray) -> None:
    SCORES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"headline_id": df["headline_id"].values, "split": df["split"].values,
                  "score": score}).to_csv(SCORES / f"{name}.csv", index=False)


def run_baseline() -> None:
    df = load_events()
    train = df[(df.split == "train") & (df.label != "flat")]
    for name, wq in (("baseline_qh", True), ("baseline_h", False)):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(vec.fit_transform(text(train, wq)), train["y"])
        t = time.perf_counter()
        score = clf.predict_proba(vec.transform(text(df, wq)))[:, 1]
        per_1k = (time.perf_counter() - t) / len(df) * 1000
        save(name, df, score)
        _write_cost(name, {"latency_s_per_1k": per_1k, "cost_usd_per_1k": 0.0})


def run_haiku() -> None:
    from hms.llm import score_direction
    df = load_events()
    test = df[df.split == "test"]  # flats included: needed for the Spearman metric
    t = time.perf_counter()
    rows = score_direction(list(zip(test["question"], test["title"])))
    wall = time.perf_counter() - t
    save("haiku_qh", test, np.array([r["out"]["p_up"] for r in rows]))
    n = len(rows)
    _write_cost("haiku_qh", {
        "cost_usd_per_1k": sum(r["cost_usd"] for r in rows) / n * 1000,
        "latency_s_per_1k_serial": sum(r["latency_s"] for r in rows) / n * 1000,
        "latency_s_per_1k_wall_8_workers": wall / n * 1000 if wall > 1 else None,
        "n_scored": n})


def _write_cost(name: str, info: dict) -> None:
    path = RESULTS / "cost_latency.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = info
    path.write_text(json.dumps(data, indent=2))


def evaluate(df: pd.DataFrame, name: str) -> dict | None:
    path = SCORES / f"{name}.csv"
    if not path.exists():
        return None
    s = pd.read_csv(path)
    test = df[df.split == "test"].merge(s[["headline_id", "score"]], on="headline_id")
    nf = test[test.label != "flat"]
    y, sc = nf["y"].to_numpy(), nf["score"].to_numpy()
    out = q2_stats.cluster_bootstrap_auc(y, sc, nf["market_id"].to_numpy())
    out.update(hit_rate=q2_stats.hit_rate(y, sc), ece=q2_stats.ece(y, np.clip(sc, 0, 1)),
               spearman_dp_with_flats=q2_stats.spearman(test["score"], test["dp"]),
               n_test_nonflat=int(len(nf)))
    return out


def report() -> None:
    df = load_events()
    test = df[df.split == "test"]
    nf = test[test.label != "flat"]
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    results = {m: evaluate(df, m) for m in PRIMARY + list(ABLATIONS)}
    primary = {m: r for m, r in results.items() if m in PRIMARY and r is not None}
    up_share = float(nf["y"].mean())
    out = {
        "decision": q2_stats.decide(primary, len(nf), nf["market_id"].nunique())
        if len(primary) == len(PRIMARY) else "INCOMPLETE (not all primary models scored)",
        "holm_survivors": q2_stats.holm({m: r["p_one_sided"] for m, r in primary.items()}),
        "n_test_nonflat": int(len(nf)), "n_test_markets": int(nf["market_id"].nunique()),
        "test_up_share": up_share,
        "naive": {"always_majority_hit_rate": max(up_share, 1 - up_share),
                  "random_auc": 0.5, "random_hit_rate": 0.5},
        "counts": counts.to_dict(), "models": results,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("decision", "n_test_nonflat", "n_test_markets")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["baseline", "haiku", "report"])
    {"baseline": run_baseline, "haiku": run_haiku, "report": report}[ap.parse_args().stage]()
