# Status — 2026-10-30

Q1 complete. Financial PhraseBank: 4,217 original / 4,211 unique sentences; train 2,947,
validation 632, frozen test 632. All four sweep checkpoints verified. Run `lr5e-05_ep3`
finished in MLflow. Run `lr2e-05_ep2` had completed training but was marked FAILED during
later selection/export; its saved checkpoint reproduced its recorded validation macro-F1
exactly (0.9053783347607478), so its training status was restored with an evidence tag.
No additional sweep setting was trained. Selected on validation: `lr2e-05_ep2`.
MLflow model registry: `q1-sentiment`, version 1. Optional export/registry errors can no
longer destroy scientific results or relabel completed training runs.

Q1 frozen-test macro-F1: baseline 0.7228421270812104; Haiku 0.8841500106898508;
fine-tuned torch/fp32 0.8681710547803233; ONNX int8 0.8705732764219003.
Machine-generated Q1 decision and paired intervals are in `results/q1/results.json` and
`results/q1/results.md`. ONNX CPU batch-32 latency per 1,000: fp32 104.0395991566342 s →
int8 23.135549248418332 s. Quantization accepted; champion and candidate contain the
selected int8 classifier, tokenizer, and contract metadata. The real frozen-test gate
measured both at 0.8705732764219003, delta 0, promote false.

Q2 collection probe started 2026-10-30 21:05:20 UTC (17:05:20 Toronto). Collection runs
under nohup with log `data/gdelt.log`; checkpoint state `data/collection_progress.json`.
Minimum request spacing 30 seconds, 429 backoff 2 → 4 → 8 minutes (capped), six attempts,
every attempt logged. Articles outside requested [start,end) are discarded. Blank/invalid
responses and saturated ten-day subwindows cannot count as complete. Completed windows
are resumable. The probe stops after three hours if fewer than 90/450 windows are ok and
writes the numbers here. No source substitution is authorized by a failed probe.
Google News RSS remains rejected: date-only timestamps can leak the move into the label.

DB before collection: 179 candidate markets, 150 selected, 329,636 hourly prices,
150 fixed Haiku queries, zero raw/clean headlines or aligned events. No Q2 build or model
run is authorized on partial collection. `scripts/q2_run.sh` checks all 450 windows before
building and running the ordered Q2 stages. Q2 sweep runs now save checkpoints and score
only development data; selection is frozen to disk before test scoring, and repeat
selection reuses already written test scores. Both grid settings are required. The
selected headline-only ablation is scored on the second selection call. A training lock
and macOS memory_pressure >=20% guard cover Q1 and Q2 runs. No Q2 test metrics inspected.

The Q2 report handles insufficient data as the frozen DATA-BLOCKED decision, uses JSON
null for undefined metrics, validates complete probability coverage, and generates a
10-bin reliability curve per scored model. No curves or AUCs will be invented when
collection fails. README is generated from recorded JSON without favorable rounding by
`scripts/write_readme.py`. The required short Claude interpretation review is performed
once after the final collection outcome using `scripts/review_results.py` and cached.

The eval-gate equality boundary now follows PREREGISTRATION.md (improvement >=0.005).
The older MLOps brief had requested >0.005; no frozen threshold or rule was changed.
Regression coverage includes the equality boundary, retries/pacing, timestamp rejection,
resumption, the three-hour stop, spend reservations, empty-data reports, probability
coverage, and freezing selection before test scoring. Ruff and pytest pass; final
check count will be recorded after collection.

Recorded Anthropic usage before final interpretation review: $0.02278268, including
$0.01495088 for Q1 and $0.0078318 for query generation, at the cached token rates
(verified against Anthropic's Haiku 5.5 pricing on 2026-10-30). A concurrent-call budget
guard reserves headroom before requests and stops at the $5 budget. This is a usage-based
estimate, not an account invoice. Secrets remain only in ignored environment/cache paths.

Local commits are blocked by managed filesystem permissions: `.git` is read-only and
Git cannot create `.git/index.lock` (Operation not permitted). The working branch is
`main`; no files have been pushed and no remote repository or deployment was created.
