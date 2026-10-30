#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "::notice::HF_TOKEN is unset; skipping Hugging Face deployment."
    exit 0
fi

: "${HF_MODEL_REPO:?HF_MODEL_REPO must be set}"
: "${HF_SPACE_REPO:?HF_SPACE_REPO must be set}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
else
    PYTHON=python
fi

"$PYTHON" - "$ROOT" <<'PY'
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

root = Path(sys.argv[1])
champion = root / "artifacts/champion"
for name in ("model.onnx", "metadata.json", "tokenizer.json"):
    if not (champion / name).is_file():
        raise FileNotFoundError(f"missing champion artifact: {name}")

api = HfApi(token=os.environ["HF_TOKEN"])
ignore = shutil.ignore_patterns(".cache", "__pycache__", "*.pyc", ".git", ".env")
# Stage exactly the app's build context, excluding local caches and unrelated files.
with tempfile.TemporaryDirectory() as temporary:
    stage = Path(temporary)
    for name in ("Dockerfile", "requirements-serve.txt", "src/hms/onnx_predict.py"):
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    shutil.copytree(root / "serve", stage / "serve", ignore=ignore)
    shutil.copytree(champion, stage / "artifacts/champion", ignore=ignore)
    api.upload_folder(
        repo_id=os.environ["HF_MODEL_REPO"],
        repo_type="model",
        revision="main",
        folder_path=stage / "artifacts/champion",
        commit_message="Deploy champion artifacts",
    )
    api.upload_folder(
        repo_id=os.environ["HF_SPACE_REPO"],
        repo_type="space",
        folder_path=stage,
        commit_message="Deploy champion Docker app",
        delete_patterns=["serve/*", "src/hms/onnx_predict.py", "artifacts/champion/*"],
    )
print("Uploaded champion model and Docker Space.")
PY
