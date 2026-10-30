# Status — 2026-10-30 (usage limit hit mid-session)

Done: pre-registration v1.0 + amendment A1 (before any Q2 data); 150 markets + hourly prices in
data/hms.sqlite; 150 Haiku GDELT queries ($0.008); pipeline/alignment code + tests; Q2 stats/models
code + tests; Codex MLOps layer (serve, Docker, CI, eval gate, HF deploy) reviewed; Q1 baseline
macro-F1 0.723, Q1 Haiku macro-F1 0.884 ($0.024/1k); Q1 fine-tune sweep in progress (fork agent).

Blocker: GDELT DOC API returns 429 from this IP; zero headlines collected. Google News RSS rejected
as a source: date-only timestamps (07:00 GMT) would leak the move into the label.
Controlled slow-rate test was running (task output in session scratch). Next:
1. If slow-rate GDELT works: set PAUSE_S ~20 in src/hms/gdelt.py, rerun scripts/run_pipeline.sh.
2. Else: amendment A2 before labels — alternative timestamped source (e.g. GDELT via BigQuery).
3. Then: python -m hms.q2 baseline / haiku; q2_finetune runs; --select; python -m hms.q2 report.
4. README, push public repo, HF token + deploy.
