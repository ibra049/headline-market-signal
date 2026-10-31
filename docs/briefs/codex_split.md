# Codex — split of remaining work (v2, 2026-10-31). Supersedes the order in codex_finish.md.

Claude owns the project again and does: Q2 baseline (done), Q2 Haiku scoring (running, do not
touch `hms.q2 haiku`, `data/q2_scores/haiku_qh.csv`, `data/cache/q2_direction.jsonl`), the Q2
report and the interpretation. DB: `~/.hms/hms.sqlite` (already built; do NOT rerun
`hms.build`, `hms.guardian`, `hms.gdelt`, `run_pipeline.sh`).

You do, in order (rules from codex_finish.md still apply: frozen prereg, no secrets, 8 GB RAM —
one training process at a time, check `memory_pressure` free >= 20% before each run; commit
locally with small commits; `git push` after each commit is OK):
1. Finish Q1 (codex_finish.md section 1): select, test metrics, ONNX fp32+int8, latency before/
   after, champion/candidate dirs, `results/q1/results.md` with a Deviations section.
2. Q2 fine-tune (codex_finish.md section 3, fine-tune part only): `python -m hms.q2_finetune
   --run lr2e-05_ep2`, `--run lr5e-05_ep2`, `--select`, then the headline-only ablation run
   with the selected lr and `--select` again. Do NOT run `python -m hms.q2 report` and do not
   open test-split scores.
3. Add a test for `guardian.parse_ts` + window filter; `ruff check .` and `pytest -q` pass.
4. Draft `README.md` (codex_finish.md section 4) with placeholders `{{Q2_TABLE}}` and
   `{{Q2_DECISION}}` for the Q2 results; Claude fills them. Data source is the Guardian
   (amendment A2), DB location ~/.hms.
Write a 10-line summary to `docs/STATUS.md` when done. Keep it tight; no extras.
