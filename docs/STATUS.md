# Status — 2026-11-02 (project complete)

- Q1 done: Financial PhraseBank, 4,211 unique sentences, split 2,947 / 632 / 632. Test macro-F1: baseline 0.723, Haiku 0.884, fine-tuned 0.868 (int8 0.871). Verdict: fine-tuned **matches** zero-shot.
- ONNX int8: CPU latency per 1k (batch 32) 104.0 s → 23.1 s; size 329 MB → 82.5 MB; macro-F1 unchanged within 0.01, so quantization accepted.
- Q2 done: 150 markets, 86,783 Guardian headlines; test set 1,101 non-flat events in 36 markets. Decision **INCONCLUSIVE** (no model survives Holm; AUCs 0.49–0.51).
- MLOps: CI (Ruff + 96 tests) green; eval gate promoted champion v1.0.0 to HF `Ibra-049/headline-sentiment`; gate now re-scores the champion on the same runner (CPU-dependent int8 scores caused a false re-promotion on 2026-11-02).
- Deploy: Render free tier via `render.yaml`; HF Docker Spaces dropped (paid). Optional owner step: create the Render Blueprint and set the `RENDER_DEPLOY_HOOK` secret.
- Total Anthropic spend: about $2–2.5.
