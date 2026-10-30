#!/usr/bin/env bash
# Q1 end to end: frozen split, baseline, fine-tune sweep, ONNX int8, Haiku zero-shot, report.
# Needs ANTHROPIC_API_KEY in the environment or in .env for the Haiku step.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false

"$PY" -m hms.q1_data
"$PY" -m hms.q1_baseline
"$PY" -m hms.q1_finetune
"$PY" -m hms.q1_onnx
"$PY" -m hms.q1_haiku
"$PY" -m hms.q1_report
