"""Q1 report: paired bootstrap CIs, results table, reliability plot."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hms.q1_common import (  # noqa: E402
    LABEL2ID,
    RESULTS,
    load_split,
    paired_bootstrap_diff,
    update_results,  # noqa: E402
)

MODELS = [
    ("baseline_tfidf_lr", "probs_baseline.npy", "TF-IDF + LR"),
    ("zeroshot_haiku", "probs_haiku.npy", "Haiku 5.5 zero-shot"),
    ("finetuned_distilroberta", "probs_finetuned.npy", "DistilRoBERTa fine-tuned (fp32)"),
]


def verdict(ci: dict) -> str:
    if ci["ci_low"] > 0:
        return "fine-tuned beats zero-shot"
    if ci["ci_high"] < 0:
        return "zero-shot is better"
    return "fine-tuned matches zero-shot (CI includes 0)"


def reliability(ax, probs, y, label, n_bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.sum() >= 5:
            xs.append(conf[mask].mean())
            ys.append(correct[mask].mean())
    ax.plot(xs, ys, marker="o", label=label)


def main() -> None:
    res = json.loads((RESULTS / "results.json").read_text())
    y = load_split("test")["label"].map(LABEL2ID).to_numpy()
    preds = {k: np.load(RESULTS / f).argmax(1) for k, f, _ in MODELS if (RESULTS / f).exists()}
    if (RESULTS / "probs_onnx_int8.npy").exists():
        preds["onnx_int8"] = np.load(RESULTS / "probs_onnx_int8.npy").argmax(1)

    comps = {}
    pairs = [("finetuned_distilroberta", "zeroshot_haiku"), ("onnx_int8", "zeroshot_haiku"),
             ("baseline_tfidf_lr", "zeroshot_haiku"),
             ("finetuned_distilroberta", "baseline_tfidf_lr")]
    for a, b in pairs:
        if a in preds and b in preds:
            ci = paired_bootstrap_diff(y, preds[a], preds[b])
            comps[f"{a}_minus_{b}"] = ci
    if "finetuned_distilroberta_minus_zeroshot_haiku" in comps:
        primary = comps["finetuned_distilroberta_minus_zeroshot_haiku"]
        comps["preregistered_verdict"] = verdict(primary)
    update_results("bootstrap", comps)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    for k, f, name in MODELS + [("optimization", "probs_onnx_int8.npy", "DistilRoBERTa int8 ONNX")]:
        if (RESULTS / f).exists():
            reliability(ax, np.load(RESULTS / f), y, name)
    ax.set_xlabel("confidence (max probability)")
    ax.set_ylabel("accuracy")
    ax.set_title("Q1 reliability, frozen test set")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "calibration.png", dpi=120)

    res = json.loads((RESULTS / "results.json").read_text())
    lines = ["# Q1 results — Financial PhraseBank (66% agree), frozen test set", ""]
    d = res.get("data", {})
    lines.append(f"Train / val / test: {d.get('n_train')} / {d.get('n_val')} / {d.get('n_test')} "
                 f"sentences. Test SHA-256 `{d.get('test_sha256')}`.")
    lines += ["", "| Model | Accuracy | Macro-F1 | ECE | Cost / 1k (USD) | Latency / 1k (s) |",
              "|---|---|---|---|---|---|"]

    def row(name, m, lat):
        cost = m.get("cost_usd_per_1k")
        lines.append(f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['ece']:.3f} | "
                     f"{cost:.4f} | {lat} |")

    if "baseline_tfidf_lr" in res:
        m = res["baseline_tfidf_lr"]
        row("TF-IDF + LR", m, f"{m['latency_s_per_1k_cpu']:.3f} (CPU)")
    if "zeroshot_haiku" in res:
        m = res["zeroshot_haiku"]
        lat = m.get("latency_s_per_1k_concurrency8")
        row("Haiku 5.5 zero-shot", m,
            f"{lat:.1f} (8 parallel requests)" if lat else "n/a")
    if "finetuned_distilroberta" in res:
        m = res["finetuned_distilroberta"]
        row("DistilRoBERTa fine-tuned (fp32, torch)", m,
            f"{m['latency_s_per_1k_mps']:.2f} (MPS) / {m['latency_s_per_1k_cpu_torch']:.2f} (CPU)")
    opt = res.get("optimization", {})
    onnx_rows = [("onnx_fp32", "DistilRoBERTa ONNX fp32"), ("onnx_int8", "DistilRoBERTa ONNX int8")]
    for k, name in onnx_rows:
        if k in opt:
            m = dict(opt[k], cost_usd_per_1k=0.0)
            row(name, m, f"{m['latency_s_per_1k_cpu']:.2f} (CPU)")

    if "sweep" in res:
        lines += ["", "Sweep (selected on validation macro-F1):", "",
                  "| Run | LR | Epochs | Val macro-F1 |", "|---|---|---|---|"]
        for r in res["sweep"]:
            lines.append(f"| {r['run']} | {r['lr']:g} | {r['epochs']} | {r['val_macro_f1']:.4f} |")
        lines.append(f"\nSelected: `{res['finetuned_distilroberta']['selected']['run']}`.")

    if opt:
        lines += ["", f"Quantization: int8 macro-F1 drop vs fp32 ONNX = "
                      f"{opt['int8_macro_f1_drop_vs_fp32']:+.4f} "
                      f"(accepted if ≤ 0.01: {opt['int8_accepted']}). Size "
                      f"{opt['onnx_fp32']['size_mb']} MB → {opt['onnx_int8']['size_mb']} MB."]

    if comps:
        lines += ["", "Paired bootstrap, 2,000 resamples, 95% CI of the macro-F1 difference:", "",
                  "| Comparison | Point | 95% CI |", "|---|---|---|"]
        for k, ci in comps.items():
            if isinstance(ci, dict):
                lines.append(f"| {k} | {ci['point']:+.4f} | [{ci['ci_low']:+.4f}, "
                             f"{ci['ci_high']:+.4f}] |")
        if "preregistered_verdict" in comps:
            lines += ["", f"**Pre-registered verdict:** {comps['preregistered_verdict']}."]
    lines += ["", "## Deviations from the pre-registration", "",
              "- 6 exact duplicate sentences were dropped before splitting (4,217 → 4,211), so "
              "the same sentence cannot be in both train and test.",
              "- Sweep run 1 (`lr2e-05_ep2`) completed with batch 16. An initial attempt at "
              "run 2 exhausted the 8 GB machine's memory, so runs 2–4 used batch 8 with gradient "
              "checkpointing (and eager attention, which MPS needs for it), one process per run. "
              "max_len stayed 128 and the lr × epochs grid did not change. Batch size is not "
              "fixed in the pre-registration.",
              "- Run 1's saved checkpoint reproduced its recorded validation macro-F1 exactly "
              "(0.9053783347607478). Its MLflow status was repaired from FAILED to FINISHED: "
              "training had completed; the failure occurred after test metrics were logged "
              "during model export. No additional sweep run was trained. Results are now saved "
              "before optional registry work, which uses a separate export run.",
              "- Haiku zero-shot settings, not specified in the pre-registration: thinking "
              "disabled, effort low. Its probabilities put the stated confidence on its label "
              "and split the rest evenly over the other two labels; this affects ECE only.",
              "", "![calibration](calibration.png)", ""]
    (RESULTS / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
