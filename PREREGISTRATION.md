# Pre-registration — headline signal for prediction markets

Version 1.0 · frozen 2026-10-30, before any headline was collected or any label
or price move was computed. Changes after this point go in an "Amendments"
section with a date and a reason; thresholds below are never changed.

Only data availability was checked before freezing: the Polymarket CLOB
`prices-history` endpoint returns hourly prices from 2026-07-01 when called with
`startTs`, and the GDELT DOC 2.0 API serves headlines for roughly the last three
months. No price moves, labels or headline counts were looked at.

## Questions

- **Q1 (model quality).** On a public labeled financial-sentiment set, does a
  small fine-tuned transformer match or beat a zero-shot LLM, at lower cost and
  latency?
- **Q2 (market signal).** Does a headline, scored by a model, predict the
  direction of a Polymarket YES price over the next N = 6 hours?

Expected result for Q2: null. Prices on liquid markets usually move before a
headline is crawled. A null result is reported as such.

## Data

### Labeled set (Q1)
- Financial PhraseBank, `sentences_66agree` (≥ 66% annotator agreement), labels
  negative / neutral / positive. Licence CC BY-NC-SA 3.0 (non-commercial use).
- Fallback, only if PhraseBank cannot be loaded at all: `zeroshot/twitter-financial-news-sentiment`
  (MIT). The fallback and the reason are logged in the README.
- No timestamps exist, so the split is stratified random with seed 42:
  70% train / 15% validation / 15% **frozen test**. The test file's SHA-256 is
  recorded in `data/frozen/` when it is created and checked by CI.

### Markets (Q2)
Selected from the Polymarket Gamma API, without looking at prices after
selection:
- binary Yes/No markets, open at some point in 2026-08-01 → 2026-10-29;
- category not sports, esports, or short-horizon crypto price markets
  ("up or down", "above $X on <date>");
- at least 7 days of hourly YES price history inside the window;
- top 150 by lifetime volume after these filters.

### Headlines (Q2)
- Source: GDELT DOC 2.0 API (`mode=artlist`, English), window 2026-08-01 →
  2026-10-29. One keyword query per market, generated once by the LLM from the
  market question and stored in the database. Queries are written before any
  headline is seen and are not edited afterwards.
- Timestamp: GDELT `seendate` (crawl time, UTC). This is at or after
  publication, so it is conservative.
- Cleaning: normalize the title (lowercase, strip punctuation and a trailing
  " - Source" suffix). Within one market, drop a headline whose normalized title
  already appeared in the previous 24 h (the earliest copy is kept). Drop
  titles under 5 words.

### Alignment and label (Q2)
- Event time `t0 = seendate + 15 min` (lag).
- `p0` = last hourly YES price with timestamp ≤ `t0`.
  `p1` = last hourly YES price with timestamp ≤ `t0 + 6 h`.
  If either has no point within the preceding 2 h, the event is dropped.
- Label: `up` if `p1 − p0 ≥ 0.005`, `down` if `p1 − p0 ≤ −0.005`, otherwise
  `flat`. Flat events are counted and reported, then excluded from the
  direction metrics.
- Events with `t0 + 6 h` after 2026-10-29 23:59 UTC are dropped.

### Time split (Q2)
By `t0`:
- train: 2026-08-01 → 2026-09-21
- validation: 2026-09-22 → 2026-10-08
- **frozen test: 2026-10-09 → 2026-10-29**

Purge: an event is dropped from train or validation when its label window
(`t0` → `t0 + 6 h`) crosses into the next split.

## Models

1. **Baseline.** TF-IDF (word 1–2-grams) + logistic regression.
2. **Zero-shot LLM.** Claude Haiku 5.5 (`claude-haiku-5-5`), structured JSON
   output, prompt caching on the fixed instructions. Q1 output: label and
   confidence. Q2 output: P(YES price rises over the next 6 h).
3. **Fine-tuned.** DistilRoBERTa-base in PyTorch + Hugging Face. Sweep of at most
   4 runs over learning rate {2e-5, 5e-5} × epochs {2, 3}, picked on
   validation macro-F1 (Q1) or validation AUC (Q2). Runs tracked in MLflow.
4. **Optional, exploratory only:** a second zero-shot LLM, a Keras head. Not part
   of any decision.

Q2 models are trained on train-period market labels with input
`market question + headline`. The Q2 test set is scored once, after model
selection is finished.

## Metrics and decision rules

### Q1
- Primary: macro-F1 on the frozen test set. Also accuracy, ECE (10 bins),
  cost per 1,000 headlines and latency per 1,000 headlines.
- "Fine-tuned beats zero-shot" is supported only if the paired bootstrap 95% CI
  (2,000 resamples) of macro-F1(fine-tuned) − macro-F1(zero-shot) is above 0.
  "Matches" if the CI includes 0. Otherwise zero-shot is better.

### Q2
- Primary: AUC of each model's P(up) against up/down on the frozen test set,
  with a 95% CI from a **market-cluster bootstrap** (resample markets, 2,000
  times). Headlines about the same market are not independent.
- Secondary: directional hit rate against always-majority-class and random
  baselines; Spearman correlation of score with `p1 − p0` (flats included);
  ECE.
- Multiple comparisons: Holm–Bonferroni over the three primary models
  (baseline, zero-shot, fine-tuned), one-sided test of AUC > 0.5 at α = 0.05,
  p-values from the cluster bootstrap.

Decision (PolySignal statuses):
- **DATA-BLOCKED** if the test set has fewer than 300 non-flat events or fewer
  than 30 markets.
- **PROMOTE (to a forward test only)** if at least one model survives Holm
  and its AUC point estimate is ≥ 0.55. This is not an alpha or profit claim:
  no fees, spreads or execution are modeled, and the next step would be a fresh
  forward test.
- **REJECT** if no model survives Holm and every model's CI upper bound is
  < 0.56.
- **INCONCLUSIVE** otherwise.

### Ablation (pre-specified)
Headline only vs market question + headline, for the baseline and the
fine-tuned model on Q2. Reported, not part of the decision.

## Optimization
The selected Q1 fine-tuned model is exported to ONNX and dynamically quantized
to int8. Reported: CPU latency per 1,000 headlines (batch 32) and test macro-F1,
before and after. Quantization is accepted if macro-F1 drops by ≤ 0.01.

## Deployment gate
The deployed model is the Q1 sentiment classifier. On merge to `main`, CI
evaluates the candidate on the frozen Q1 test set and promotes it only if its
macro-F1 is higher than the current champion's by ≥ 0.005.

## Non-claims
Nothing here is a trading strategy. No fees, spreads, slippage or capacity are
modeled, and no profit is claimed. All negative results are kept.

## Amendments
(none)
