"""Render the README from recorded results without favorable rounding or manual verdicts."""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hms.db import DB_PATH  # noqa: E402
from hms.budget import cached_spend  # noqa: E402


def render():
    q1 = json.loads(Path("results/q1/results.json").read_text())
    q2path = Path("results/q2/results.json")
    q2 = json.loads(q2path.read_text()) if q2path.exists() else None
    with sqlite3.connect(DB_PATH) as conn:
        sizes = {name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                 for name in ("markets", "prices", "queries", "headlines_raw",
                              "headlines", "events")}
        selected = conn.execute("SELECT COUNT(*) FROM markets WHERE selected=1").fetchone()[0]
        ok = conn.execute("SELECT COUNT(*) FROM fetch_log WHERE status='ok'").fetchone()[0]
    d = q1["data"]
    lines = ["# Headline market signal", "",
             "Two pre-registered questions: does a small financial sentiment classifier match a "
             "zero-shot LLM at lower cost and latency (Q1), and does a scored headline predict "
             "the next six-hour direction of a Polymarket YES price (Q2)?", "",
             "The frozen protocol and amendment A1 are in "
             "[PREREGISTRATION.md](PREREGISTRATION.md). "
             "All thresholds, metrics and splits remain frozen. See "
             "[docs/STATUS.md](docs/STATUS.md) "
             "for operational limitations.", "", "## Data", "",
             f"Q1: Financial PhraseBank `Sentences_66Agree`, {d['n_raw']} original sentences; "
             f"{d['n_raw'] - d['n_after_exact_dedupe']} exact duplicates removed; "
             f"{d['n_after_exact_dedupe']} unique sentences. "
             "Stratified seed-42 train / validation / "
             f"frozen test: {d['n_train']} / {d['n_val']} / {d['n_test']}. "
             f"Test label counts: `{json.dumps(d['label_counts_test'], sort_keys=True)}`. "
             f"Frozen test SHA-256: `{d['test_sha256']}`.", "",
             f"Q2 DB snapshot: {sizes['markets']} candidate markets, {selected} selected, "
             f"{sizes['prices']} hourly price records, {sizes['queries']} fixed queries, "
             f"{sizes['headlines_raw']} raw headlines, {sizes['headlines']} cleaned headlines, "
             f"{sizes['events']} aligned events. Collection: {ok}/{selected * 3} monthly windows "
             "complete. These are collected counts, not evidence of a market signal.", "",
             "Markets are binary, non-sports/non-esports, excluding short-horizon crypto price "
             "questions, with at least seven days of history, ranked by lifetime volume. "
             "The event window is 2026-08-01 through 2026-10-29 UTC.", "", "## Method", "",
             "Q1 compares TF-IDF word 1–2-grams plus logistic regression, Claude Haiku 5.5 "
             "zero-shot, and DistilRoBERTa-base selected by validation macro-F1 over the four "
             "pre-registered settings. The first completed setting used batch 16; subsequent "
             "settings use batch 8 and gradient checkpointing after a memory crash. Only one "
             "training process runs at once; macOS requires at least 20% memory_pressure free. "
             "Haiku uses effort low. The selected classifier is exported to ONNX fp32 and "
             "dynamic int8; int8 is accepted only if its macro-F1 drop is at most 0.01.", "",
             "Q2 uses English GDELT crawl timestamps, rejects any article outside its requested "
             "half-open collection window, drops titles under five words, and deduplicates "
             "normalized titles within a market for 24 hours. Event time is crawl time plus "
             "15 minutes. The last price at or before event time and six hours later must each "
             "be less than two hours old. A move >=0.005 is up, <=-0.005 down, otherwise flat; "
             "flats are reported and excluded from directional metrics. Train ends September 21, "
             "validation ends October 8, and frozen test covers October 9–29. Label windows "
             "crossing splits are purged.", "",
             "Q2 compares the same three model families. Amendment A1 permits two fine-tuning "
             "settings (2e-5 and 5e-5, two epochs); selection uses validation AUC. Sweep runs "
             "score development data only, save checkpoints, freeze selection on disk, and then "
             "score the test set. The baseline and selected transformer also have headline-only "
             "ablations. Primary AUC intervals use 2,000 market-cluster bootstrap resamples, "
             "with Holm correction over the three primary models. Secondary metrics are "
             "directional hit rate, ECE, and Spearman correlation with price moves "
             "including flats.",
             "", "## Q1 results", "",
             "Values below are copied from [results/q1/results.json](results/q1/results.json) "
             "using the JSON numbers' full recorded precision. Local model dollar cost denotes "
             "API charges only; hardware, electricity and training time are excluded.", "",
             "| Model | Accuracy | Macro-F1 | ECE | USD / 1,000 | Seconds / 1,000 |",
             "|---|---|---|---|---|---|"]
    for key, name, latency in [
        ("baseline_tfidf_lr", "TF-IDF + LR (CPU)", "latency_s_per_1k_cpu"),
        ("zeroshot_haiku", "Haiku 5.5 (8 concurrent requests)", "latency_s_per_1k_concurrency8"),
        ("finetuned_distilroberta", "DistilRoBERTa (torch CPU)", "latency_s_per_1k_cpu_torch"),
    ]:
        m = q1[key]
        lines.append(f"| {name} | {m['accuracy']} | {m['macro_f1']} | {m['ece']} | "
                     f"{m['cost_usd_per_1k']} | {m[latency]} |")
    lines += ["", f"Selected setting: `{q1['finetuned_distilroberta']['selected']['run']}`.", "",
              "| ONNX CPU, batch 32 | Macro-F1 | Seconds / 1,000 | Size MB |",
              "|---|---|---|---|"]
    for key in ("onnx_fp32", "onnx_int8"):
        m = q1["optimization"][key]
        lines.append(f"| {key} | {m['macro_f1']} | {m['latency_s_per_1k_cpu']} | {m['size_mb']} |")
    ci = q1["bootstrap"]["finetuned_distilroberta_minus_zeroshot_haiku"]
    lines += ["", f"Paired bootstrap fine-tuned minus zero-shot macro-F1: {ci['point']}, "
              f"95% CI [{ci['ci_low']}, {ci['ci_high']}], {ci['n_resamples']} resamples.", "",
              f"Q1 report decision, verbatim: `{q1['bootstrap']['preregistered_verdict']}`.", "",
              "Full deviations and calibration: [Q1 report](results/q1/results.md).", "",
              "## Q2 results", ""]
    if q2 is None:
        lines += ["Pending collection; no Q2 decision or model AUCs have been computed."]
    else:
        lines += [f"Pre-registered decision string from `hms.q2 report`, verbatim: "
                  f"`{q2['decision']}`.", "",
                  f"Frozen test: {q2['n_test_nonflat']} non-flat events, "
                  f"{q2['n_test_markets']} markets. Counts by split and label: "
                  f"`{json.dumps(q2['counts'], sort_keys=True)}`.", "",
                  "| Model | AUC | 95% CI lower | 95% CI upper | One-sided p |",
                  "|---|---|---|---|---|"]
        for name, m in q2["models"].items():
            vals = [m.get(k) if m else None for k in ("auc", "ci_low", "ci_high", "p_one_sided")]
            lines.append(f"| {name} | " + " | ".join("unavailable" if v is None else str(v)
                                                     for v in vals) + " |")
        lines += ["", "[Machine results](results/q2/results.json) · "
                  "[Reliability plot](results/q2/calibration.png). "
                  "A plot with no scored events is explicitly marked unavailable; no curve or "
                  "AUC is inferred from missing data."]
    lines += ["", "## What did not work", "",
              "The original Q1 batch-16 sweep exhausted this 8 GB Mac's memory during the next "
              "setting. Amendment A1 records batch 8 and gradient checkpointing. The completed "
              "first checkpoint was revalidated against its exact stored validation macro-F1 "
              "after an MLflow export failure mislabeled the run FAILED; it was not retrained.",
              "",
              "GDELT DOC API frequently returns HTTP 429 from this IP. Collection uses at least "
              "30 seconds between requests, 2/4/8-minute capped backoff, at most six attempts "
              "per fetch, and logs every attempt. A resumable three-hour probe stops collection "
              "if fewer than 20% of the 450 windows are complete. Empty/invalid responses and "
              "saturated ten-day subwindows are not counted as complete. Partial collection "
              "does not authorize Q2 training. Google News RSS was rejected because its "
              "date-only timestamps can leak the price move into the label. No substitute "
              "source has been introduced.", "", "## Run", "",
              "Python 3.12. Install the research environment and keep secrets in the ignored "
              "`.env` or environment variables:", "", "```bash",
              "uv venv --python 3.12 .venv",
              "uv pip install --python .venv -r requirements-train.txt",
              "scripts/q1_run.sh              # Q1, resumes completed training settings",
              "scripts/run_pipeline.sh        # markets → queries → GDELT → alignment",
              "scripts/q2_run.sh              # requires all 450 collection windows complete",
              "```", "", "Individual Q2 stages, in order:", "", "```bash",
              "export PYTHONPATH=src TOKENIZERS_PARALLELISM=false",
              ".venv/bin/python -m hms.build",
              ".venv/bin/python -m hms.q2 baseline",
              ".venv/bin/python -m hms.q2 haiku",
              ".venv/bin/python -m hms.q2_finetune --run lr2e-05_ep2",
              ".venv/bin/python -m hms.q2_finetune --run lr5e-05_ep2",
              ".venv/bin/python -m hms.q2_finetune --select",
              "# Read selected from data/q2_scores/ft_runs/selection.json:",
              ".venv/bin/python -m hms.q2_finetune --run <selected> --headline-only",
              ".venv/bin/python -m hms.q2_finetune --select",
              ".venv/bin/python -m hms.q2 report", "```", "",
              "Run the collector unattended with output preserved:", "", "```bash",
              "PYTHONPATH=src nohup .venv/bin/python -u -m hms.gdelt > data/gdelt.log 2>&1 &",
              "tail -f data/gdelt.log", "```", "",
              "Probe state is in `data/collection_progress.json`. A restart retains the "
              "original probe start and completed windows. Exit 3 means the three-hour probe "
              "failed; owner + Claude must decide on a timestamped alternative before any "
              "affected labels or scores are computed. Do not reset the probe to evade this rule.",
              "", "```mermaid", "flowchart TD",
              "  P[PhraseBank] --> F[Frozen stratified split + SHA-256]",
              "  F --> Q1[Q1 baseline / Haiku / transformer sweep]",
              "  Q1 --> V[Validation selection]",
              "  V --> O[ONNX fp32 / int8 + latency]",
              "  O --> G[CI frozen-test evaluation gate]",
              "  G --> A[Champion FastAPI / Docker / HF Space]",
              "  M[Polymarket markets + hourly prices] --> D[SQLite]",
              "  H[Fixed Haiku queries + GDELT crawl timestamps] --> D",
              "  D --> E[Clean / lag / align / label / purge time splits]",
              "  E --> Q2[Q2 models + validation selection]",
              "  Q2 --> R[Cluster bootstrap / Holm / machine decision]", "```", "",
              "## Serving and CI", "",
              "`serve.app:app` loads `MODEL_DIR` (default `artifacts/champion`) once at startup. "
              "`GET /health` reports model version; `POST /predict` accepts 1–256 texts of at most "
              "1,000 characters. Inference uses tokenizers and ONNX Runtime, max length 128, "
              "dynamic padding and batch 32. The non-root Docker image exposes port 7860.", "",
              "Pull requests and pushes run Ruff and pytest. On main, the eval-gate workflow "
              "downloads HF `candidate` and `main` revisions, verifies the frozen test hash, "
              "and measures candidate macro-F1. The frozen protocol requires an improvement "
              "of at least 0.005 over champion metadata (or no existing champion). The older "
              "MLOps brief described a strict boundary; implementation now follows the "
              "authoritative pre-registration's inclusive boundary. A passing candidate is "
              "committed and version-tagged in HF, then deployed to the Docker Space. "
              "Configure `HF_TOKEN`, `HF_MODEL_REPO`, and `HF_SPACE_REPO`; missing token skips "
              "remote evaluation/deployment. This local task does not push or deploy.", "",
              "```bash", ".venv/bin/ruff check .", ".venv/bin/python -m pytest -q",
              ".venv/bin/python scripts/eval_gate.py --candidate artifacts/candidate \\",
              "  --champion artifacts/champion --test data/frozen/q1_test.csv --min-delta 0.005",
              "```", "", "## Cost and licences", "",
              f"Recorded Anthropic spend: ${cached_spend()} USD, calculated from cached token "
              "usage and the study's recorded rates ($0.10/M input, $0.50/M output, "
              "$0.01/M cache read, $0.125/M cache write). This is an estimate from stored usage, "
              "not an account invoice. Rates were checked against "
              "[Anthropic's Haiku 5.5 pricing](https://www.anthropic.com/claude-haiku-5-5) "
              "on 2026-10-30. Local model API cost is $0/1,000; it excludes hardware "
              "and electricity. Calls reserve budget for concurrent requests and stop before "
              "the recorded total can exceed $5. Cache hits incur no new API charge.", "",
              "Financial PhraseBank: [CC BY-NC-SA 3.0, author dataset card]"
              "(https://huggingface.co/datasets/takala/financial_phrasebank); non-commercial "
              "use with attribution and share-alike conditions. Polymarket data remains "
              "subject to [Polymarket Terms of Use](https://polymarket.com/tos), including "
              "data-access and redistribution restrictions. Data from the GDELT Project "
              "([GDELT terms](https://www.gdeltproject.org/about.html#termsofuse)); attribution "
              "does not transfer publisher copyrights in linked news. No new licence is "
              "granted over third-party data or model weights.", "",
              "**No profit claims.** This is research, not a trading strategy. No fees, spreads, "
              "slippage, execution or capacity are modeled. All negative results are kept. "
              "A PROMOTE status would authorize only a fresh forward test.", ""]
    Path("README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    render()
