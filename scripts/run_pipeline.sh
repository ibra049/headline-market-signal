#!/usr/bin/env bash
# End-to-end data pipeline: markets -> prices -> queries -> GDELT headlines -> clean/align -> splits.
# Idempotent: each step resumes from what is already in data/hms.sqlite.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH=src
"$PY" -m hms.markets     # select 150 markets, store hourly YES prices
"$PY" -m hms.queries     # one Haiku-written GDELT query per market (needs ANTHROPIC_API_KEY)
"$PY" -m hms.gdelt       # headlines, 1 request / 5.5 s (about 40-60 min the first time)
"$PY" -m hms.build       # dedupe, align with 15-min lag, label next-6h move, time splits
sqlite3 data/hms.sqlite "SELECT split, COUNT(*) AS events, COUNT(DISTINCT market_id) AS markets FROM events GROUP BY split"
