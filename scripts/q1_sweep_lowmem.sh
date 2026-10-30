#!/usr/bin/env bash
# Run the remaining Q1 sweep settings one process at a time, with a memory guard (macOS).
# Before each run: wait (60 s polls, up to 20 min) until system free memory >= 20%.
# During a run: stop it if RSS > 5 GB or swap use grows by more than 2 GB.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false MLFLOW_DISABLE_AGENT_HINT=1
RUNS="${*:-lr2e-05_ep2 lr2e-05_ep3 lr5e-05_ep2 lr5e-05_ep3}"

free_pct() { memory_pressure | tail -1 | grep -oE '[0-9]+' | tail -1; }
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }

for run in $RUNS; do
  waited=0
  while [ "$(free_pct)" -lt 20 ]; do
    if [ "$waited" -ge 1200 ]; then
      echo "ABORT $run: free memory stayed below 20% for 20 min"; exit 3
    fi
    echo "wait $run: free $(free_pct)%"; sleep 60; waited=$((waited + 60))
  done
  swap0=$(swap_mb)
  echo "start $run: free $(free_pct)%, swap ${swap0} MB"
  "$PY" -m hms.q1_finetune --run "$run" &
  pid=$!
  peak=0
  while kill -0 "$pid" 2>/dev/null; do
    rss=$(( $(ps -o rss= -p "$pid" 2>/dev/null || echo 0) / 1024 ))
    [ "$rss" -gt "$peak" ] && peak=$rss
    growth=$(( $(swap_mb) - swap0 ))
    if [ "$rss" -gt 5000 ] || [ "$growth" -gt 2000 ]; then
      kill "$pid"; echo "KILLED $run: rss ${rss} MB, swap growth ${growth} MB"; exit 4
    fi
    sleep 5
  done
  wait "$pid"; rc=$?
  echo "done $run: exit $rc, peak rss ${peak} MB, swap growth $(( $(swap_mb) - swap0 )) MB"
  [ "$rc" -eq 0 ] || exit "$rc"
done
echo "SWEEP COMPLETE"
