# Status — 2026-10-31

- Q1 verified: 4,211 unique sentences; train/val/test 2,947/632/632; all four sweep runs FINISHED; selected lr2e-05_ep2.
- Q1 test macro-F1: baseline 0.7228421270812104; Haiku 0.8841500106898508; fine-tuned 0.8681710547803233; int8 0.8705732764219003.
- Q1 ONNX CPU batch-32 latency/1k: fp32 104.0395991566342 s → int8 23.135549248418332 s; int8 accepted; champion/candidate hashes verified; gate delta 0, no promotion.
- Q2 validation AUC: lr2e-05_ep2 0.4758335322670486; lr5e-05_ep2 0.4900519880109281; frozen selection lr5e-05_ep2.
- Q2 selected headline-only ablation completed; all three MLflow runs FINISHED; finetuned_qh.csv and finetuned_h.csv written; test scores left unopened; sequential CPU runs passed memory guards.
- Guardian collection 150/150: 86,783 raw / 86,457 cleaned / 78,958 eligible events; DB ~/.hms/hms.sqlite; collection and alignment were not rerun.
- Added six Guardian timestamp/window tests; Ruff clean and pytest 94 passed; README drafted with {{Q2_TABLE}} and {{Q2_DECISION}}.
- Claude owns Q2 Haiku/report/interpretation; Haiku files untouched; no new Anthropic calls; local commits blocked by read-only .git (index.lock denied); nothing pushed.
