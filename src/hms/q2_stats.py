"""Q2 statistics: market-cluster bootstrap AUC, Holm correction, pre-registered decision."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

N_BOOT = 2000
SEED = 42


def cluster_bootstrap_auc(y: np.ndarray, s: np.ndarray, clusters: np.ndarray,
                          n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Resample markets with replacement; AUC on the pooled resampled events.
    One-sided p-value for AUC > 0.5 = share of bootstrap AUCs <= 0.5."""
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    index = {c: np.flatnonzero(clusters == c) for c in ids}
    aucs = []
    for _ in range(n_boot):
        rows = np.concatenate([index[c] for c in rng.choice(ids, size=len(ids))])
        if len(np.unique(y[rows])) < 2:
            continue
        aucs.append(roc_auc_score(y[rows], s[rows]))
    aucs = np.array(aucs)
    return {"auc": float(roc_auc_score(y, s)),
            "ci_low": float(np.quantile(aucs, 0.025)), "ci_high": float(np.quantile(aucs, 0.975)),
            "p_one_sided": float(np.mean(aucs <= 0.5)), "n_boot": int(len(aucs))}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down; returns which hypotheses are rejected."""
    order = sorted(pvalues, key=pvalues.get)
    out, still = {}, True
    for i, name in enumerate(order):
        still = still and pvalues[name] <= alpha / (len(order) - i)
        out[name] = still
    return out


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error for a binary probability p = P(y = 1)."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def hit_rate(y: np.ndarray, s: np.ndarray) -> float:
    return float(np.mean((s > 0.5) == (y == 1)))


def spearman(s: np.ndarray, dp: np.ndarray) -> float | None:
    if len(s) < 2 or np.ptp(s) == 0 or np.ptp(dp) == 0:
        return None  # Undefined, represented as JSON null rather than nonstandard NaN.
    return float(spearmanr(s, dp).statistic)


def decide(results: dict[str, dict], n_test: int, n_markets: int) -> str:
    """Pre-registered Q2 decision. results: model -> cluster_bootstrap_auc output."""
    if n_test < 300 or n_markets < 30:
        return "DATA-BLOCKED"
    survives = holm({m: r["p_one_sided"] for m, r in results.items()})
    if any(survives[m] and results[m]["auc"] >= 0.55 for m in results):
        return "PROMOTE (forward test only)"
    if not any(survives.values()) and all(r["ci_high"] < 0.56 for r in results.values()):
        return "REJECT"
    return "INCONCLUSIVE"
