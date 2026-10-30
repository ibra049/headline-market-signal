"""Q1 baseline: TF-IDF (1-2 grams) + logistic regression."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from hms.q1_common import (
    LABEL2ID,
    RESULTS,
    SEED,
    classification_metrics,
    load_split,
    time_per_1k,
    update_results,
)


def main() -> None:
    train, test = load_split("train"), load_split("test")
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10.0, random_state=SEED),
    )
    model.fit(train["text"], train["label"].map(LABEL2ID))
    texts = test["text"].tolist()
    probs = model.predict_proba(texts)
    y = test["label"].map(LABEL2ID).to_numpy()
    out = classification_metrics(probs, y)
    out["latency_s_per_1k_cpu"] = time_per_1k(model.predict_proba, texts, repeats=5)
    out["cost_usd_per_1k"] = 0.0
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS / "probs_baseline.npy", probs)
    update_results("baseline_tfidf_lr", out)
    print(out)


if __name__ == "__main__":
    main()
