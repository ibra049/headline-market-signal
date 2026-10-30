import numpy as np

from hms import q2_stats


def test_holm_step_down():
    got = q2_stats.holm({"a": 0.01, "b": 0.03, "c": 0.04})
    assert got == {"a": True, "b": False, "c": False}  # 0.03 > 0.05 / 2 stops the chain
    assert q2_stats.holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}


def test_cluster_bootstrap_null_and_signal():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(40), 25)
    y = rng.integers(0, 2, size=clusters.size)
    null = q2_stats.cluster_bootstrap_auc(y, rng.random(y.size), clusters, n_boot=300)
    assert null["ci_low"] < 0.5 < null["ci_high"]
    good = q2_stats.cluster_bootstrap_auc(y, y + rng.normal(0, 0.5, y.size), clusters, n_boot=300)
    assert good["ci_low"] > 0.7 and good["p_one_sided"] == 0.0


def test_ece_perfect_and_bad():
    y = np.array([0, 1] * 50)
    assert q2_stats.ece(y, np.full(100, 0.5)) == 0.0
    assert abs(q2_stats.ece(y, np.where(y == 1, 0.0, 1.0)) - 1.0) < 1e-9


def _r(auc, lo, hi, p):
    return {"auc": auc, "ci_low": lo, "ci_high": hi, "p_one_sided": p}


def test_decision_rules():
    null = {m: _r(0.51, 0.48, 0.54, 0.3) for m in "abc"}
    assert q2_stats.decide(null, n_test=1000, n_markets=50) == "REJECT"
    assert q2_stats.decide(null, n_test=200, n_markets=50) == "DATA-BLOCKED"
    wide = {**null, "c": _r(0.52, 0.45, 0.60, 0.3)}
    assert q2_stats.decide(wide, n_test=1000, n_markets=50) == "INCONCLUSIVE"
    strong = {**null, "a": _r(0.58, 0.55, 0.61, 0.001)}
    assert q2_stats.decide(strong, n_test=1000, n_markets=50) == "PROMOTE (forward test only)"
