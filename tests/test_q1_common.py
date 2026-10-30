import numpy as np
import pandas as pd

from hms.q1_common import ece, paired_bootstrap_diff, split_70_15_15


def _toy(n=200):
    labels = (["negative"] * 30 + ["neutral"] * 110 + ["positive"] * 60)[:n]
    return pd.DataFrame({"text": [f"s{i}" for i in range(n)], "label": labels})


def test_split_is_deterministic_and_disjoint():
    a = split_70_15_15(_toy())
    b = split_70_15_15(_toy())
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)
    sets = [set(p["text"]) for p in a]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(len(s) for s in sets) == 200
    assert len(a[0]) == 140


def test_split_is_stratified():
    train, _, test = split_70_15_15(_toy())
    assert abs(train["label"].eq("neutral").mean() - test["label"].eq("neutral").mean()) < 0.05


def test_ece_zero_when_calibrated():
    probs = np.array([[1.0, 0.0, 0.0]] * 10)
    assert ece(probs, np.zeros(10, dtype=int)) == 0.0


def test_ece_overconfident():
    probs = np.array([[0.9, 0.05, 0.05]] * 10)
    y = np.array([0] * 5 + [1] * 5)
    assert abs(ece(probs, y) - 0.4) < 1e-9


def test_bootstrap_identical_predictions_gives_zero():
    y = np.array([0, 1, 2] * 20)
    ci = paired_bootstrap_diff(y, y, y, n=200)
    assert ci["point"] == 0.0 and ci["ci_low"] == 0.0 and ci["ci_high"] == 0.0


def test_bootstrap_better_model_has_positive_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    good = y.copy()
    bad = rng.integers(0, 3, 300)
    ci = paired_bootstrap_diff(y, good, bad, n=300)
    assert ci["ci_low"] > 0
