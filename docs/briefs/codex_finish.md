# Brief for Codex — finish the project (v1, 2026-10-30)

You now own implementation end to end. Claude does one short independent review of the final
results (interpretation); you do not write the final verdict prose beyond what `hms.q2 report`
computes. Read first: `PREREGISTRATION.md` (incl. amendment A1), `docs/STATUS.md`,
`docs/briefs/codex_mlops.md`. Thresholds, metrics, splits and decision rules are frozen: never
change them. Any unavoidable method change = a dated amendment in PREREGISTRATION.md written
BEFORE the affected labels/scores are computed, with the reason.

Rules: no secrets in files (keys are in `.env`, gitignored); no paid services beyond the
Anthropic key already in `.env` (expected total spend < $2; stop and report if Haiku spend
passes $5); commit to `main` locally in small commits ending with a `Co-Authored-By` line for
yourself; do not push or create remote repos (owner does that). Memory: 8 GB Mac — one training
process at a time, batch 8 + gradient checkpointing, check `memory_pressure` free ≥ 20% before
starting a run.

## 1. Finish Q1 (scripts exist: `scripts/q1_run.sh`, `src/hms/q1_*.py`)
State: all 4 sweep checkpoints exist in `artifacts/q1_sweep/` (verify run 4 `lr5e-05_ep3`
finished in MLflow `mlflow.db`, experiment `q1-sentiment`; if not, rerun only that setting with
`python -m hms.q1_finetune --run lr5e-05_ep3`). Then: select on val macro-F1, test metrics,
ONNX fp32 + int8 (`hms.q1_onnx`), write `artifacts/champion/` + `artifacts/candidate/` per the
contract, Haiku results already cached in `data/cache/`, run `hms.q1_report` →
`results/q1/results.md` incl. a "Deviations" section (6 duplicates dropped, batch 8 + grad
checkpointing, Haiku effort low).

## 2. Headlines (DONE by Claude, amendment A2)
GDELT was abandoned (throttled; see A2). Claude switched to the Guardian API
(`src/hms/guardian.py`, key in `.env`). Collection is running: `data/guardian.log`, wait until it
prints `150/150` (pgrep -f hms.guardian empty). Do not touch gdelt.py or rerun collection. Fix the
ruff E501 on the PERIODS line in guardian.py if ruff flags it. Add a small test for
`guardian.parse_ts` and the window filter.

## 3. Build + Q2 models
`python -m hms.build` (prints split×label counts). Then in order: `python -m hms.q2 baseline`;
`python -m hms.q2 haiku`; `python -m hms.q2_finetune --run lr2e-05_ep2`;
`--run lr5e-05_ep2`; `--select`; the headline-only ablation with the selected lr
(`--run <selected> --headline-only`), then `--select` again so `finetuned_h.csv` is written;
`python -m hms.q2 report` → `results/q2/results.json`. Do not look at test results before
`--select` has run. Add a calibration plot for Q2 (reliability curve per model) to `results/q2/`.

## 4. ONNX latency note + README
Q1 latency before/after quantization must be in results. Write `README.md`: question, data
(sizes from the DB/CSVs), method, results tables for Q1 and Q2 copied from the JSON (no
rounding in favour of any hypothesis), the pre-registered decision string verbatim, what didn't
work (Q1 memory crash, GDELT throttling -> Guardian switch, Google News rejection), one-command run
(`scripts/run_pipeline.sh`, `scripts/q1_run.sh`, Q2 commands), a mermaid architecture diagram,
how the CI eval gate works, licences (PhraseBank CC BY-NC-SA 3.0; Polymarket/GDELT terms), and
"no profit claims".

## 5. Checks + done
`ruff check .` and `pytest -q` pass. Update `docs/STATUS.md`. Final reply: dataset sizes,
Q1 macro-F1 ×3, Q2 decision + AUCs with CIs, latency before/after, cost per 1k, number of tests,
total Anthropic spend, and anything you could not do.

Keep it tight: no extra features beyond this brief. Prefer finishing over polishing.
