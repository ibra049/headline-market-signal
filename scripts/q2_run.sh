#!/usr/bin/env bash
# Build and evaluate Q2 only after every selected market/window is collected.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false MLFLOW_DISABLE_AGENT_HINT=1
"$PY" - <<'PY'
from hms.db import connect
from hms.gdelt import chunks
with connect() as conn:
    markets = [r[0] for r in conn.execute('SELECT market_id FROM markets WHERE selected=1')]
    expected = {(mid, s.isoformat(), e.isoformat()) for mid in markets for s, e in chunks()}
    done = set(conn.execute("SELECT market_id,start,end FROM fetch_log WHERE status='ok'"))
    if len(markets) != 150 or not expected <= done:
        raise SystemExit(f'Collection incomplete: {len(expected & done)}/{len(expected)} chunks ok')
PY
"$PY" -m hms.build
"$PY" -m hms.q2 baseline
"$PY" -m hms.q2 haiku
"$PY" -m hms.q2_finetune --run lr2e-05_ep2
"$PY" -m hms.q2_finetune --run lr5e-05_ep2
"$PY" -m hms.q2_finetune --select
selected=$("$PY" - <<'PY'
import json
from hms.q2_finetune import SELECTION
print(json.loads(SELECTION.read_text())['selected'])
PY
)
"$PY" -m hms.q2_finetune --run "$selected" --headline-only
"$PY" -m hms.q2_finetune --select
"$PY" -m hms.q2 report
