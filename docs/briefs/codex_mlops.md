# Brief for Codex — serving, Docker, CI, eval gate

Question: build the MLOps layer for this repo. Version 1. Owner: Codex implements; Claude reviews.
Read `PREREGISTRATION.md` (sections "Deployment gate" and "Optimization") first. Do not edit it.

## Scope (only these files)
- `src/hms/onnx_predict.py` — `class OnnxClassifier(model_dir)` with
  `predict_proba(texts: list[str]) -> np.ndarray` (shape n×3, columns
  negative/neutral/positive) and `predict(texts) -> list[dict]` (`label`, `probs`).
  The model dir contains `model.onnx` (int8 or fp32), Hugging Face tokenizer files
  (`tokenizer.json` etc.), and `metadata.json` = `{"version": str, "macro_f1": float, "labels": ["negative","neutral","positive"]}`.
  Use `onnxruntime` + `tokenizers` only (no torch, no transformers) so the image stays small.
  Max length 128, dynamic padding, batch size 32.
- `serve/app.py` — FastAPI. `GET /health` → `{"status":"ok","model_version":...}`.
  `POST /predict` body `{"texts": [str, ...]}` (1–256 texts, each ≤ 1000 chars) →
  `{"model_version":..., "predictions":[{"label":..., "probs":{...}}]}`. Model dir from env
  `MODEL_DIR` (default `artifacts/champion`). Load once at startup.
- `Dockerfile` — python:3.12-slim, `requirements-serve.txt`, non-root user, port 7860
  (Hugging Face Spaces convention), `HEALTHCHECK` hitting `/health`, copies `serve/`,
  `src/hms/onnx_predict.py` and `artifacts/champion/`.
- `requirements-serve.txt` (pinned), `requirements-dev.txt` (ruff, pytest, httpx).
- `scripts/eval_gate.py` — CLI: `--candidate DIR --champion DIR --test data/frozen/q1_test.csv --min-delta 0.005`.
  Test CSV columns: `text,label` (label is one of the three names). Computes candidate macro-F1
  with `OnnxClassifier`, reads champion `metadata.json` macro_f1, prints a JSON decision
  `{"candidate_f1", "champion_f1", "delta", "promote": bool}` and exits 0. Put the decision in a
  pure function `decide(candidate_f1, champion_f1, min_delta) -> bool` (strictly greater than
  champion + min_delta; no champion → promote). Also verify the test CSV SHA-256 against
  `data/frozen/q1_test.sha256` and fail if it differs.
- `scripts/deploy_hf.sh` — uploads `artifacts/champion/` to the Hugging Face model repo
  `$HF_MODEL_REPO` and pushes the Docker app (Dockerfile, serve/, src/hms/onnx_predict.py,
  requirements-serve.txt, artifacts/champion/) to the Docker Space `$HF_SPACE_REPO`, using
  `huggingface_hub` CLI/Python with `$HF_TOKEN`. Exits 0 with a notice if `HF_TOKEN` is unset.
- `.github/workflows/ci.yml` — on pull_request and push: set up Python 3.12, install
  `requirements-dev.txt` + `requirements-serve.txt` + `scikit-learn numpy pandas`, run
  `ruff check .` and `pytest -q`.
- `.github/workflows/eval-gate.yml` — on push to `main` and workflow_dispatch: download the
  candidate model from HF model repo revision `candidate` and the champion from revision `main`
  (skip with a notice if `HF_TOKEN` secret is missing), run `scripts/eval_gate.py`; if
  `promote` is true, copy candidate → champion on the HF model repo (new commit + tag
  `v<candidate version>`), then run `scripts/deploy_hf.sh`. Write the decision JSON to the job
  summary.
- `tests/test_eval_gate.py`, `tests/test_serve.py`, `tests/test_onnx_predict.py` — tests must
  run without any real model: inject a fake classifier into the app (dependency override or a
  factory), and unit-test `decide`, the SHA check, input validation (empty list, >256 texts,
  overlong text → 422), and `/health`.
- `ruff.toml` (line length 100, target py312).

## Out of scope / rules
- Do not touch `PREREGISTRATION.md`, `data/`, `results/`, or other files in `src/hms/`.
- Do not run training, call any paid API, create accounts, or push anywhere. Do not `git commit`;
  Claude commits.
- No secrets, server addresses or tokens in any file. Secrets come from env / GitHub secrets.
- Use the repo venv: `.venv/bin/python`, `uv pip install --python .venv ...` for new deps.

## Done when
`.venv/bin/ruff check .` and `.venv/bin/python -m pytest -q` both pass. Reply with a short
list of files written and anything you could not do.
