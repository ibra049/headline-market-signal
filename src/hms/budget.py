"""Conservative Anthropic spend guard using the study's recorded token rates."""

import json
import threading
from contextlib import contextmanager
from pathlib import Path

CACHE_DIR = Path("data/cache")
LIMIT_USD = 5.0
# Covers max_tokens=2048 and these short inputs at the recorded rates, with headroom.
RESERVE_USD = 0.02


class SpendLimitExceeded(RuntimeError):
    pass


def cached_spend(cache_dir=CACHE_DIR):
    total = 0.0
    for path in cache_dir.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "cost_usd" in row:
                total += row["cost_usd"]
            elif "input_tokens" in row:
                total += (row["input_tokens"] * 0.10e-6 + row["output_tokens"] * 0.50e-6
                          + row.get("cache_read", 0) * 0.01e-6
                          + row.get("cache_write", 0) * 0.125e-6)
    return total


class SpendGuard:
    """One stage at a time; reserve for all of its concurrent outstanding calls."""

    def __init__(self):
        self.spent = cached_spend()
        self.reserved = 0.0
        self.lock = threading.Lock()
        self.check()

    def check(self):
        if self.spent > LIMIT_USD:
            raise SpendLimitExceeded(f"STOP: recorded Anthropic spend ${self.spent:.6f} exceeds $5")

    @contextmanager
    def request(self):
        with self.lock:
            self.check()
            if self.spent + self.reserved + RESERVE_USD > LIMIT_USD:
                raise SpendLimitExceeded("STOP: insufficient Anthropic budget for another request")
            self.reserved += RESERVE_USD
        cost = [0.0]
        try:
            yield cost
        finally:
            with self.lock:
                self.reserved -= RESERVE_USD
                self.spent += cost[0]
                self.check()
